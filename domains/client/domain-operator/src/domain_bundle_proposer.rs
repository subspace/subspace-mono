use crate::ExecutionReceiptFor;
use codec::Encode;
use futures::{select, FutureExt};
use sc_client_api::{AuxStore, BlockBackend};
use sc_transaction_pool_api::InPoolTransaction;
use sp_api::{ApiExt, ProvideRuntimeApi};
use sp_block_builder::BlockBuilder;
use sp_blockchain::HeaderBackend;
use sp_domains::core_api::DomainCoreApi;
use sp_domains::{
    BundleHeader, DomainId, DomainsApi, ExecutionReceipt, HeaderHashingFor, OperatorId,
    ProofOfElection,
};
use sp_messenger::MessengerApi;
use sp_runtime::traits::{Block as BlockT, Hash as HashT, Header as HeaderT, NumberFor, One, Zero};
use sp_runtime::Percent;
use sp_transaction_pool::runtime_api::TaggedTransactionQueue;
use sp_weights::Weight;
use std::collections::HashSet;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time;
use subspace_core_primitives::U256;
use subspace_runtime_primitives::Balance;

/// If the bundle utilization is below `BUNDLE_UTILIZATION_THRESHOLD` we will attempt to push
/// at most `MAX_SKIPPED_TRANSACTIONS` number of transactions before quitting for real.
const MAX_SKIPPED_TRANSACTIONS: usize = 8;

const BUNDLE_UTILIZATION_THRESHOLD: Percent = Percent::from_percent(95);

// `PreviousBundledTx` used to keep track of tx that have included in previous bundle and avoid
// to re-including these transactions in the next bundle if the consensus hash did not change.
struct PreviousBundledTx<Block: BlockT, CBlock: BlockT> {
    bundled_at: CBlock::Hash,
    tx_hashes: HashSet<Block::Hash>,
}

impl<Block: BlockT, CBlock: BlockT> PreviousBundledTx<Block, CBlock> {
    fn new() -> Self {
        PreviousBundledTx {
            bundled_at: Default::default(),
            tx_hashes: HashSet::new(),
        }
    }

    fn already_bundled(&self, tx_hash: &Block::Hash) -> bool {
        self.tx_hashes.contains(tx_hash)
    }

    fn maybe_clear(&mut self, consensus_hash: CBlock::Hash) {
        if self.bundled_at != consensus_hash {
            self.bundled_at = consensus_hash;
            self.tx_hashes.clear();
        }
    }

    fn add_bundled(&mut self, tx_hash: Block::Hash) {
        self.tx_hashes.insert(tx_hash);
    }
}

pub struct DomainBundleProposer<Block: BlockT, Client, CBlock: BlockT, CClient, TransactionPool> {
    domain_id: DomainId,
    client: Arc<Client>,
    consensus_client: Arc<CClient>,
    transaction_pool: Arc<TransactionPool>,
    previous_bundled_tx: PreviousBundledTx<Block, CBlock>,
    _phantom_data: PhantomData<(Block, CBlock)>,
}

impl<Block: BlockT, Client, CBlock: BlockT, CClient, TransactionPool> Clone
    for DomainBundleProposer<Block, Client, CBlock, CClient, TransactionPool>
{
    fn clone(&self) -> Self {
        Self {
            domain_id: self.domain_id,
            client: self.client.clone(),
            consensus_client: self.consensus_client.clone(),
            transaction_pool: self.transaction_pool.clone(),
            previous_bundled_tx: PreviousBundledTx::new(),
            _phantom_data: self._phantom_data,
        }
    }
}

pub(super) type ProposeBundleOutput<Block, CBlock> = (
    BundleHeader<NumberFor<CBlock>, <CBlock as BlockT>::Hash, <Block as BlockT>::Header, Balance>,
    Vec<<Block as BlockT>::Extrinsic>,
);

impl<Block, Client, CBlock, CClient, TransactionPool>
    DomainBundleProposer<Block, Client, CBlock, CClient, TransactionPool>
where
    Block: BlockT,
    CBlock: BlockT,
    NumberFor<Block>: Into<NumberFor<CBlock>>,
    Client: HeaderBackend<Block> + BlockBackend<Block> + AuxStore + ProvideRuntimeApi<Block>,
    Client::Api: BlockBuilder<Block>
        + DomainCoreApi<Block>
        + TaggedTransactionQueue<Block>
        + MessengerApi<Block, NumberFor<CBlock>, CBlock::Hash>,
    CClient: HeaderBackend<CBlock> + ProvideRuntimeApi<CBlock>,
    CClient::Api: DomainsApi<CBlock, Block::Header>,
    TransactionPool: sc_transaction_pool_api::TransactionPool<Block = Block, Hash = Block::Hash>,
{
    pub fn new(
        domain_id: DomainId,
        client: Arc<Client>,
        consensus_client: Arc<CClient>,
        transaction_pool: Arc<TransactionPool>,
    ) -> Self {
        Self {
            domain_id,
            client,
            consensus_client,
            transaction_pool,
            previous_bundled_tx: PreviousBundledTx::new(),
            _phantom_data: PhantomData,
        }
    }

    pub(crate) async fn propose_bundle_at(
        &mut self,
        proof_of_election: ProofOfElection,
        tx_range: U256,
        operator_id: OperatorId,
        receipt: ExecutionReceiptFor<Block, CBlock>,
    ) -> sp_blockchain::Result<ProposeBundleOutput<Block, CBlock>> {
        // NOTE: use the domain block that derive the ER to validate the extrinsic to be included
        // in the bundle, so the validity of the extrinsic is committed to the ER that submited together.
        let (parent_number, parent_hash) = (receipt.domain_block_number, receipt.domain_block_hash);
        let consensus_best_hash = self.consensus_client.info().best_hash;

        let mut t1 = self.transaction_pool.ready_at(parent_number).fuse();
        // TODO: proper timeout
        let mut t2 = futures_timer::Delay::new(time::Duration::from_micros(100)).fuse();

        let pending_iterator = select! {
            res = t1 => res,
            _ = t2 => {
                tracing::warn!(
                    "Timeout fired waiting for transaction pool at #{parent_number},{parent_hash}, \
                    proceeding with bundle production."
                );
                self.transaction_pool.ready()
            }
        };

        // Clear the previous bundled tx info whenever the consensus chain tip is changed,
        // this allow the operator to retry for the previous bundled tx in case the previous
        // bundle fail to submit to the consensus chain due to any reason.
        self.previous_bundled_tx.maybe_clear(consensus_best_hash);

        let (domain_bundle_limit, storage_fund_balance, transaction_byte_fee) = {
            let consensus_runtime_api = self.consensus_client.runtime_api();

            let domain_bundle_limit = consensus_runtime_api
                .domain_bundle_limit(consensus_best_hash, self.domain_id)?
                .ok_or_else(|| {
                    sp_blockchain::Error::Application(
                        format!("Domain bundle limit for {:?} not found", self.domain_id).into(),
                    )
                })?;

            let storage_fund_balance = consensus_runtime_api
                .storage_fund_account_balance(consensus_best_hash, operator_id)?;

            let transaction_byte_fee =
                consensus_runtime_api.consensus_chain_byte_fee(consensus_best_hash)?;

            (
                domain_bundle_limit,
                storage_fund_balance,
                transaction_byte_fee,
            )
        };

        let bundle_vrf_hash = U256::from_be_bytes(*proof_of_election.vrf_hash());

        let header_size = receipt.encoded_size()
            + proof_of_election.encoded_size()
            + domain_bundle_limit.max_bundle_weight.encoded_size()
            // Extrinsics root size
            + 32
            // Header signature size
            + 64;

        let mut extrinsics = Vec::new();
        let mut estimated_bundle_weight = Weight::default();
        let mut bundle_size = 0u32;
        let mut skipped = 0;

        // Separate code block to make sure that runtime api instance is dropped after validation is done.
        {
            // We are using one runtime api instance here to maintain storage changes in the instance's internal buffer
            // between runtime calls done in this loop.
            let runtime_api_instance = self.client.runtime_api();
            for pending_tx in pending_iterator {
                let pending_tx_data = pending_tx.data();

                let is_within_tx_range = runtime_api_instance
                    .is_within_tx_range(parent_hash, pending_tx_data, &bundle_vrf_hash, &tx_range)
                    .map_err(|err| {
                        tracing::error!(
                            ?err,
                            ?pending_tx_data,
                            "Error occurred in locating the tx range"
                        );
                    })
                    .unwrap_or(false);
                if !is_within_tx_range {
                    continue;
                }

                // Skip the tx if is is already bundled by a recent bundle
                if self
                    .previous_bundled_tx
                    .already_bundled(&self.transaction_pool.hash_of(pending_tx_data))
                {
                    continue;
                }

                let tx_weight = runtime_api_instance
                    .extrinsic_weight(parent_hash, pending_tx_data)
                    .map_err(|error| {
                        sp_blockchain::Error::Application(Box::from(format!(
                            "Error getting extrinsic weight: {error}"
                        )))
                    })?;
                let next_estimated_bundle_weight =
                    estimated_bundle_weight.saturating_add(tx_weight);
                if next_estimated_bundle_weight.any_gt(domain_bundle_limit.max_bundle_weight) {
                    if skipped < MAX_SKIPPED_TRANSACTIONS
                        && Percent::from_rational(
                            estimated_bundle_weight.ref_time(),
                            domain_bundle_limit.max_bundle_weight.ref_time(),
                        ) < BUNDLE_UTILIZATION_THRESHOLD
                    {
                        skipped += 1;
                        continue;
                    } else {
                        break;
                    }
                }

                let next_bundle_size = bundle_size + pending_tx_data.encoded_size() as u32;
                if next_bundle_size > domain_bundle_limit.max_bundle_size {
                    if skipped < MAX_SKIPPED_TRANSACTIONS
                        && Percent::from_rational(bundle_size, domain_bundle_limit.max_bundle_size)
                            < BUNDLE_UTILIZATION_THRESHOLD
                    {
                        skipped += 1;
                        continue;
                    } else {
                        break;
                    }
                }

                let next_bundle_storage_fee =
                    (header_size as u32 + next_bundle_size) as u128 * transaction_byte_fee;
                if next_bundle_storage_fee > storage_fund_balance {
                    tracing::warn!(
                        ?next_bundle_storage_fee,
                        ?storage_fund_balance,
                        "Insufficient storage fund balance to pay for the bundle storage fee"
                    );
                    break;
                }

                // Double check XDM before adding it to the bundle
                if let Some(false) =
                    runtime_api_instance.is_xdm_mmr_proof_valid(parent_hash, pending_tx_data)?
                {
                    continue;
                }

                // Double check the transaction validity, because the tx pool are re-validate the transaction
                // in pool asynchronously so there is race condition that the operator imported a domain block
                // and start producing bundle immediately before the re-validation based on the latest block
                // is finished, cause the bundle contains illegal tx accidentally and being considered as invalid
                // bundle and slashing on the honest operator.
                //
                // This check is done in similar fashion to block builder's build block.
                // This check needs to be the last check as otherwise if the tx won't be part of bundle due to
                // some other checks, its side effect will still be part of RuntimeApiImpl's changes buffer.
                // let transaction_validity_result =
                //     runtime_api_instance.execute_in_transaction(|api| {
                //         let transaction_validity_result = api.check_extrinsics_and_do_pre_dispatch(
                //             parent_hash,
                //             vec![pending_tx_data.clone()],
                //             parent_number,
                //             parent_hash,
                //         );
                //         // Only commit, if there are no errors (both ApiError and CheckTxValidityError)
                //         if let Ok(Ok(_)) = transaction_validity_result {
                //             sp_api::TransactionOutcome::Commit(transaction_validity_result)
                //         } else {
                //             sp_api::TransactionOutcome::Rollback(transaction_validity_result)
                //         }
                //     })?;
                // if transaction_validity_result.is_err() {
                //     continue;
                // }

                estimated_bundle_weight = next_estimated_bundle_weight;
                bundle_size = next_bundle_size;
                extrinsics.push(pending_tx_data.clone());

                self.previous_bundled_tx
                    .add_bundled(self.transaction_pool.hash_of(pending_tx_data));
            }
        }

        let extrinsics_root = HeaderHashingFor::<Block::Header>::ordered_trie_root(
            extrinsics.iter().map(|xt| xt.encode()).collect(),
            sp_core::storage::StateVersion::V1,
        );

        let header = BundleHeader {
            proof_of_election,
            receipt,
            estimated_bundle_weight,
            bundle_extrinsics_root: extrinsics_root,
        };

        Ok((header, extrinsics))
    }

    /// Returns the receipt in the next domain bundle.
    pub fn load_next_receipt(
        &self,
        domain_best_number_onchain: NumberFor<Block>,
        head_receipt_number: NumberFor<Block>,
    ) -> sp_blockchain::Result<ExecutionReceiptFor<Block, CBlock>> {
        tracing::trace!(
            ?domain_best_number_onchain,
            ?head_receipt_number,
            "Collecting receipt"
        );

        // Both `domain_best_number_onchain` and `head_receipt_number` are zero means the domain just
        // instantiated and nothing have submitted yet so submit the genesis receipt
        if domain_best_number_onchain.is_zero() && head_receipt_number.is_zero() {
            let genesis_hash = self.client.info().genesis_hash;
            let genesis_header = self.client.header(genesis_hash)?.ok_or_else(|| {
                sp_blockchain::Error::Backend(format!(
                    "Domain block header for #{genesis_hash:?} not found",
                ))
            })?;

            return Ok(ExecutionReceipt::genesis(
                *genesis_header.state_root(),
                *genesis_header.extrinsics_root(),
                genesis_hash,
            ));
        }

        // The next receipt must extend the current head receipt
        let receipt_number = head_receipt_number + One::one();

        // Get the domain block hash corresponding to `receipt_number` in the domain canonical chain
        let domain_hash = self.client.hash(receipt_number)?.ok_or_else(|| {
            sp_blockchain::Error::Backend(format!(
                "Domain block hash for #{receipt_number:?} not found"
            ))
        })?;

        crate::load_execution_receipt_by_domain_hash::<Block, CBlock, _>(
            &*self.client,
            domain_hash,
            receipt_number,
        )
    }
}
