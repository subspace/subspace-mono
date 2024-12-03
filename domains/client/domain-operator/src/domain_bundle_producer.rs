use crate::bundle_producer_election_solver::BundleProducerElectionSolver;
use crate::domain_bundle_proposer::DomainBundleProposer;
use crate::utils::OperatorSlotInfo;
use crate::BundleSender;
use codec::Decode;
use sc_client_api::{AuxStore, BlockBackend};
use sp_api::ProvideRuntimeApi;
use sp_block_builder::BlockBuilder;
use sp_blockchain::HeaderBackend;
use sp_consensus_slots::Slot;
use sp_domains::core_api::DomainCoreApi;
use sp_domains::{
    Bundle, BundleHeader, BundleProducerElectionApi, DomainId, DomainsApi, OperatorId,
    OperatorPublicKey, OperatorSignature, ProofOfElection, SealedBundleHeader,
    SealedSingletonReceipt, SingletonReceipt,
};
use sp_keystore::KeystorePtr;
use sp_messenger::MessengerApi;
use sp_runtime::traits::{Block as BlockT, NumberFor, Zero};
use sp_runtime::{RuntimeAppPublic, Saturating};
use sp_transaction_pool::runtime_api::TaggedTransactionQueue;
use std::sync::Arc;
use subspace_runtime_primitives::Balance;
use tracing::info;

/// Type alias for block hash.
pub type BlockHashFor<Block> = <Block as BlockT>::Hash;

/// Type alias for block header.
pub type HeaderFor<Block> = <Block as BlockT>::Header;

/// Type alias for bundle header.
pub type BundleHeaderFor<Block, CBlock> =
    BundleHeader<NumberFor<CBlock>, BlockHashFor<CBlock>, HeaderFor<Block>, Balance>;

/// Type alias for extrinsics.
pub type ExtrinsicFor<Block> = <Block as BlockT>::Extrinsic;

type OpaqueBundle<Block, CBlock> = sp_domains::OpaqueBundle<
    NumberFor<CBlock>,
    <CBlock as BlockT>::Hash,
    <Block as BlockT>::Header,
    Balance,
>;

type SealedSingletonReceiptFor<Block, CBlock> = SealedSingletonReceipt<
    NumberFor<CBlock>,
    <CBlock as BlockT>::Hash,
    <Block as BlockT>::Header,
    Balance,
>;

pub enum DomainProposal<Block: BlockT, CBlock: BlockT> {
    Bundle(OpaqueBundle<Block, CBlock>),
    Receipt(SealedSingletonReceiptFor<Block, CBlock>),
}

impl<Block: BlockT, CBlock: BlockT> DomainProposal<Block, CBlock> {
    pub fn into_opaque_bundle(self) -> Option<OpaqueBundle<Block, CBlock>> {
        match self {
            DomainProposal::Bundle(b) => Some(b),
            DomainProposal::Receipt(_) => None,
        }
    }
}

pub struct DomainBundleProducer<Block, CBlock, Client, CClient, TransactionPool>
where
    Block: BlockT,
    CBlock: BlockT,
{
    domain_id: DomainId,
    consensus_client: Arc<CClient>,
    client: Arc<Client>,
    bundle_sender: Arc<BundleSender<Block, CBlock>>,
    keystore: KeystorePtr,
    bundle_producer_election_solver: BundleProducerElectionSolver<Block, CBlock, CClient>,
    domain_bundle_proposer: DomainBundleProposer<Block, Client, CBlock, CClient, TransactionPool>,
    // TODO: both `skip_empty_bundle_production` and `skip_out_of_order_slot` are only used in the
    // tests, we should introduce a trait for `DomainBundleProducer` and use a wrapper of `DomainBundleProducer`
    // in the test, both `skip_empty_bundle_production` and `skip_out_of_order_slot` should move into the wrapper
    // to keep the production code clean.
    skip_empty_bundle_production: bool,
    skip_out_of_order_slot: bool,
    last_processed_slot: Option<Slot>,
}

impl<Block, CBlock, Client, CClient, TransactionPool> Clone
    for DomainBundleProducer<Block, CBlock, Client, CClient, TransactionPool>
where
    Block: BlockT,
    CBlock: BlockT,
{
    fn clone(&self) -> Self {
        Self {
            domain_id: self.domain_id,
            consensus_client: self.consensus_client.clone(),
            client: self.client.clone(),
            bundle_sender: self.bundle_sender.clone(),
            keystore: self.keystore.clone(),
            bundle_producer_election_solver: self.bundle_producer_election_solver.clone(),
            domain_bundle_proposer: self.domain_bundle_proposer.clone(),
            skip_empty_bundle_production: self.skip_empty_bundle_production,
            skip_out_of_order_slot: self.skip_out_of_order_slot,
            last_processed_slot: None,
        }
    }
}

impl<Block, CBlock, Client, CClient, TransactionPool>
    DomainBundleProducer<Block, CBlock, Client, CClient, TransactionPool>
where
    Block: BlockT,
    CBlock: BlockT,
    NumberFor<Block>: Into<NumberFor<CBlock>>,
    NumberFor<CBlock>: Into<NumberFor<Block>>,
    Client: HeaderBackend<Block> + BlockBackend<Block> + AuxStore + ProvideRuntimeApi<Block>,
    Client::Api: BlockBuilder<Block>
        + DomainCoreApi<Block>
        + TaggedTransactionQueue<Block>
        + MessengerApi<Block, NumberFor<CBlock>, CBlock::Hash>,
    CClient: HeaderBackend<CBlock> + ProvideRuntimeApi<CBlock>,
    CClient::Api: DomainsApi<CBlock, Block::Header> + BundleProducerElectionApi<CBlock, Balance>,
    TransactionPool: sc_transaction_pool_api::TransactionPool<Block = Block, Hash = Block::Hash>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        domain_id: DomainId,
        consensus_client: Arc<CClient>,
        client: Arc<Client>,
        domain_bundle_proposer: DomainBundleProposer<
            Block,
            Client,
            CBlock,
            CClient,
            TransactionPool,
        >,
        bundle_sender: Arc<BundleSender<Block, CBlock>>,
        keystore: KeystorePtr,
        skip_empty_bundle_production: bool,
        skip_out_of_order_slot: bool,
    ) -> Self {
        let bundle_producer_election_solver = BundleProducerElectionSolver::<Block, CBlock, _>::new(
            keystore.clone(),
            consensus_client.clone(),
        );
        Self {
            domain_id,
            consensus_client,
            client,
            bundle_sender,
            keystore,
            bundle_producer_election_solver,
            domain_bundle_proposer,
            skip_empty_bundle_production,
            skip_out_of_order_slot,
            last_processed_slot: None,
        }
    }

    fn sign(
        &self,
        operator_signing_key: &OperatorPublicKey,
        msg: &[u8],
    ) -> sp_blockchain::Result<OperatorSignature> {
        let signature = self
            .keystore
            .sr25519_sign(OperatorPublicKey::ID, operator_signing_key.as_ref(), msg)
            .map_err(|error| {
                sp_blockchain::Error::Application(Box::from(format!(
                    "Error occurred when signing the bundle: {error}"
                )))
            })?
            .ok_or_else(|| {
                sp_blockchain::Error::Application(Box::from(
                    "This should not happen as the existence of key was just checked",
                ))
            })?;

        OperatorSignature::decode(&mut signature.as_ref()).map_err(|err| {
            sp_blockchain::Error::Application(Box::from(format!(
                "Failed to decode the signature of bundle: {err}"
            )))
        })
    }

    #[expect(clippy::type_complexity)]
    pub fn claim_bundle_slot(
        &self,
        operator_id: OperatorId,
        slot_info: &OperatorSlotInfo,
        domain_best_number: NumberFor<Block>,
        consensus_chain_best_hash: BlockHashFor<CBlock>,
    ) -> sp_blockchain::Result<
        Option<(
            NumberFor<Block>,
            NumberFor<Block>,
            ProofOfElection,
            OperatorPublicKey,
        )>,
    > {
        let OperatorSlotInfo {
            slot,
            proof_of_time,
        } = slot_info;

        let domain_best_number_onchain = self
            .consensus_client
            .runtime_api()
            .domain_best_number(consensus_chain_best_hash, self.domain_id)?
            .ok_or_else(|| {
                sp_blockchain::Error::Application(
                    format!(
                        "Failed to get the head domain number for domain {:?} at {:?}",
                        self.domain_id, consensus_chain_best_hash
                    )
                    .into(),
                )
            })?;
        let head_receipt_number = self
            .consensus_client
            .runtime_api()
            .head_receipt_number(consensus_chain_best_hash, self.domain_id)?;

        let should_skip_slot = {
            // Operator is lagging behind the receipt chain on its parent chain as another operator
            // already processed a block higher than the local best and submitted the receipt to
            // the parent chain, we ought to catch up with the consensus block processing before
            // producing new bundle.
            let is_operator_lagging =
                !domain_best_number.is_zero() && domain_best_number <= head_receipt_number;

            let skip_out_of_order_slot = self.skip_out_of_order_slot
                && self
                    .last_processed_slot
                    .map(|last_slot| last_slot >= *slot)
                    .unwrap_or(false);

            is_operator_lagging || skip_out_of_order_slot
        };

        if should_skip_slot {
            tracing::warn!(
                ?domain_best_number,
                "Skipping bundle production on slot {slot}"
            );
            return Ok(None);
        }

        if let Some((proof_of_election, operator_signing_key)) =
            self.bundle_producer_election_solver.solve_challenge(
                *slot,
                consensus_chain_best_hash,
                self.domain_id,
                operator_id,
                *proof_of_time,
            )?
        {
            tracing::info!("📦 Claimed slot {slot}");

            Ok(Some((
                domain_best_number_onchain,
                head_receipt_number,
                proof_of_election,
                operator_signing_key,
            )))
        } else {
            Ok(None)
        }
    }

    pub fn prepare_receipt(
        &self,
        slot_info: &OperatorSlotInfo,
        domain_best_number_onchain: NumberFor<Block>,
        head_receipt_number: NumberFor<Block>,
        proof_of_election: &ProofOfElection,
        operator_signing_key: &OperatorPublicKey,
    ) -> sp_blockchain::Result<Option<DomainProposal<Block, CBlock>>> {
        // When the receipt gap is greater than one, the operator needs to produce a receipt
        // instead of a bundle
        if domain_best_number_onchain.saturating_sub(head_receipt_number) > 1u32.into() {
            info!(
                ?domain_best_number_onchain,
                ?head_receipt_number,
                "🔖 Producing singleton receipt at slot {:?}",
                slot_info.slot
            );

            let receipt = self
                .domain_bundle_proposer
                .load_next_receipt(domain_best_number_onchain, head_receipt_number)?;

            let singleton_receipt = SingletonReceipt {
                proof_of_election: proof_of_election.clone(),
                receipt,
            };

            let signature = {
                let to_sign: BlockHashFor<Block> = singleton_receipt.hash();
                self.sign(operator_signing_key, to_sign.as_ref())?
            };

            let sealed_singleton_receipt: SealedSingletonReceiptFor<Block, CBlock> =
                SealedSingletonReceipt {
                    singleton_receipt,
                    signature,
                };

            Ok(Some(DomainProposal::Receipt(sealed_singleton_receipt)))
        } else {
            Ok(None)
        }
    }

    #[expect(clippy::too_many_arguments)]
    pub async fn prepare_bundle(
        &mut self,
        operator_id: OperatorId,
        slot_info: &OperatorSlotInfo,
        consensus_chain_best_hash: BlockHashFor<CBlock>,
        domain_best_number_onchain: NumberFor<Block>,
        head_receipt_number: NumberFor<Block>,
        proof_of_election: ProofOfElection,
        // TODO: remove when skip_empty_bundle_production is split out
        domain_best_number: NumberFor<Block>,
        // TODO: remove Option when skip_empty_bundle_production is split out
    ) -> sp_blockchain::Result<Option<(BundleHeaderFor<Block, CBlock>, Vec<ExtrinsicFor<Block>>)>>
    {
        let tx_range = self
            .consensus_client
            .runtime_api()
            .domain_tx_range(consensus_chain_best_hash, self.domain_id)
            .map_err(|error| {
                sp_blockchain::Error::Application(Box::from(format!(
                    "Error getting tx range: {error}"
                )))
            })?;

        let receipt = self
            .domain_bundle_proposer
            .load_next_receipt(domain_best_number_onchain, head_receipt_number)?;

        let (bundle_header, extrinsics) = self
            .domain_bundle_proposer
            .propose_bundle_at(proof_of_election.clone(), tx_range, operator_id, receipt)
            .await?;

        // if there are no extrinsics and no receipts to confirm, skip the bundle
        if self.skip_empty_bundle_production
            && extrinsics.is_empty()
            && !self
                .consensus_client
                .runtime_api()
                .non_empty_er_exists(consensus_chain_best_hash, self.domain_id)?
        {
            tracing::warn!(
                ?domain_best_number,
                "Skipping empty bundle production on slot {}",
                slot_info.slot,
            );
            return Ok(None);
        }

        self.last_processed_slot.replace(slot_info.slot);

        Ok(Some((bundle_header, extrinsics)))
    }

    pub fn seal_bundle(
        &self,
        bundle_header: BundleHeaderFor<Block, CBlock>,
        operator_signing_key: &OperatorPublicKey,
        extrinsics: Vec<ExtrinsicFor<Block>>,
    ) -> sp_blockchain::Result<DomainProposal<Block, CBlock>> {
        let signature = {
            let to_sign = bundle_header.hash();
            self.sign(operator_signing_key, to_sign.as_ref())?
        };

        let bundle = Bundle {
            sealed_header: SealedBundleHeader::new(bundle_header, signature),
            extrinsics,
        };

        // TODO: Re-enable the bundle gossip over X-Net when the compact bundle is supported.
        // if let Err(e) = self.bundle_sender.unbounded_send(signed_bundle.clone()) {
        // tracing::error!(error = ?e, "Failed to send transaction bundle");
        // }

        Ok(DomainProposal::Bundle(bundle.into_opaque_bundle()))
    }

    pub async fn produce_bundle(
        &mut self,
        operator_id: OperatorId,
        slot_info: OperatorSlotInfo,
    ) -> sp_blockchain::Result<Option<DomainProposal<Block, CBlock>>> {
        let domain_best_number = self.client.info().best_number;
        let consensus_chain_best_hash = self.consensus_client.info().best_hash;

        let Some((
            domain_best_number_onchain,
            head_receipt_number,
            proof_of_election,
            operator_signing_key,
        )) = self.claim_bundle_slot(
            operator_id,
            &slot_info,
            domain_best_number,
            consensus_chain_best_hash,
        )?
        else {
            return Ok(None);
        };

        if let Some(receipt) = self.prepare_receipt(
            &slot_info,
            domain_best_number_onchain,
            head_receipt_number,
            &proof_of_election,
            &operator_signing_key,
        )? {
            return Ok(Some(receipt));
        }

        let Some((bundle_header, extrinsics)) = self
            .prepare_bundle(
                operator_id,
                &slot_info,
                consensus_chain_best_hash,
                domain_best_number_onchain,
                head_receipt_number,
                proof_of_election,
                domain_best_number,
            )
            .await?
        else {
            return Ok(None);
        };

        info!("🔖 Producing bundle at slot {:?}", slot_info.slot);

        let bundle = self.seal_bundle(bundle_header, &operator_signing_key, extrinsics)?;

        Ok(Some(bundle))
    }
}
