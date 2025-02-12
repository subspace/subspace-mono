//! Provides "fake" runtime API implementation as a workaround for compile-time checks.

use domain_runtime_primitives::opaque::Header as DomainHeader;
use domain_runtime_primitives::{
    BlockNumber as DomainNumber, EthereumAccountId, Hash as DomainHash,
};
use frame_support::weights::Weight;
use sp_consensus_subspace::{ChainConstants, PotParameters, SignedVote, SolutionRanges};
use sp_core::crypto::KeyTypeId;
use sp_core::{OpaqueMetadata, H256};
use sp_domains::bundle_producer_election::BundleProducerElectionParams;
use sp_domains::{
    DomainAllowlistUpdates, DomainId, DomainInstanceData, ExecutionReceiptFor, OperatorId,
    OperatorPublicKey, PermissionedActionAllowedBy,
};
use sp_domains_fraud_proof::fraud_proof::FraudProof;
use sp_domains_fraud_proof::storage_proof::FraudProofStorageKeyRequest;
use sp_messenger::messages::{
    BlockMessagesWithStorageKey, ChainId, ChannelId, CrossDomainMessage, MessageId, MessageKey,
};
use sp_messenger::{ChannelNonce, XdmId};
use sp_runtime::traits::{Block as BlockT, NumberFor};
use sp_runtime::transaction_validity::{TransactionSource, TransactionValidity};
use sp_runtime::{ApplyExtrinsicResult, ExtrinsicInclusionMode};
use sp_subspace_mmr::ConsensusChainMmrLeafProof;
use sp_version::RuntimeVersion;
use std::collections::btree_map::BTreeMap;
use std::collections::btree_set::BTreeSet;
use subspace_core_primitives::objects::BlockObjectMapping;
use subspace_core_primitives::segments::{
    HistorySize, SegmentCommitment, SegmentHeader, SegmentIndex,
};
use subspace_core_primitives::{PublicKey, Randomness, U256};
use subspace_runtime_primitives::opaque::Block;
use subspace_runtime_primitives::{AccountId, Balance, BlockNumber, Moment, Nonce};

mod mmr {
    pub use pallet_mmr::primitives::*;
    use sp_core::H256;
    // Full type for this is actually `<<Runtime as pallet_mmr::Config>::Hashing as sp_runtime::traits::Hash>::Output`
    // but since we don't really want to impl pallet for the fake runtime, we just use the `Keccak256::Output = H256`
    pub type Hash = H256;
}

struct Runtime;

sp_api::impl_runtime_apis! {
    impl sp_api::Core<Block> for Runtime {
        fn version() -> RuntimeVersion {
            unreachable!()
        }

        fn execute_block(_block: Block) {
            unreachable!()
        }

        fn initialize_block(_header: &<Block as BlockT>::Header) -> ExtrinsicInclusionMode {
            unreachable!()
        }
    }

    impl sp_api::Metadata<Block> for Runtime {
        fn metadata() -> OpaqueMetadata {
            unreachable!()
        }

        fn metadata_at_version(_version: u32) -> Option<OpaqueMetadata> {
            unreachable!()
        }

        fn metadata_versions() -> Vec<u32> {
            unreachable!()
        }
    }

    impl sp_block_builder::BlockBuilder<Block> for Runtime {
        fn apply_extrinsic(_extrinsic: <Block as BlockT>::Extrinsic) -> ApplyExtrinsicResult {
            unreachable!()
        }

        fn finalize_block() -> <Block as BlockT>::Header {
            unreachable!()
        }

        fn inherent_extrinsics(_data: sp_inherents::InherentData) -> Vec<<Block as BlockT>::Extrinsic> {
            unreachable!()
        }

        fn check_inherents(
            _block: Block,
            _data: sp_inherents::InherentData,
        ) -> sp_inherents::CheckInherentsResult {
            unreachable!()
        }
    }

    impl sp_transaction_pool::runtime_api::TaggedTransactionQueue<Block> for Runtime {
        fn validate_transaction(
            _source: TransactionSource,
            _tx: <Block as BlockT>::Extrinsic,
            _block_hash: <Block as BlockT>::Hash,
        ) -> TransactionValidity {
            unreachable!()
        }
    }

    impl sp_offchain::OffchainWorkerApi<Block> for Runtime {
        fn offchain_worker(_header: &<Block as BlockT>::Header) {
            unreachable!()
        }
    }

    impl sp_objects::ObjectsApi<Block> for Runtime {
        fn extract_block_object_mapping(_block: Block) -> BlockObjectMapping {
            unreachable!()
        }
    }

    impl sp_consensus_subspace::SubspaceApi<Block, PublicKey> for Runtime {
        fn pot_parameters() -> PotParameters {
            unreachable!()
        }

        fn solution_ranges() -> SolutionRanges {
            unreachable!()
        }

        fn submit_vote_extrinsic(
            _signed_vote: SignedVote<NumberFor<Block>, <Block as BlockT>::Hash, PublicKey>,
        ) {
            unreachable!()
        }

        fn history_size() -> HistorySize {
            unreachable!()
        }

        fn max_pieces_in_sector() -> u16 {
            unreachable!()
        }

        fn segment_commitment(_segment_index: SegmentIndex) -> Option<SegmentCommitment> {
            unreachable!()
        }

        fn extract_segment_headers(_ext: &<Block as BlockT>::Extrinsic) -> Option<Vec<SegmentHeader >> {
            unreachable!()
        }

        fn is_inherent(_ext: &<Block as BlockT>::Extrinsic) -> bool {
            unreachable!()
        }

        fn root_plot_public_key() -> Option<PublicKey> {
            unreachable!()
        }

        fn should_adjust_solution_range() -> bool {
            unreachable!()
        }

        fn chain_constants() -> ChainConstants {
            unreachable!()
        }
    }

    impl sp_domains::DomainsApi<Block, DomainHeader> for Runtime {
        fn submit_bundle_unsigned(
            _opaque_bundle: sp_domains::OpaqueBundle<NumberFor<Block>, <Block as BlockT>::Hash, DomainHeader, Balance>,
        ) {
            unreachable!()
        }

        fn submit_receipt_unsigned(
            _singleton_receipt: sp_domains::SealedSingletonReceipt<NumberFor<Block>, <Block as BlockT>::Hash, DomainHeader, Balance>,
        ) {
            unreachable!()
        }

        fn extract_successful_bundles(
            _domain_id: DomainId,
            _extrinsics: Vec<<Block as BlockT>::Extrinsic>,
        ) -> sp_domains::OpaqueBundles<Block, DomainHeader, Balance> {
            unreachable!()
        }

        fn extrinsics_shuffling_seed() -> Randomness {
            unreachable!()
        }

        fn domain_runtime_code(_domain_id: DomainId) -> Option<Vec<u8>> {
            unreachable!()
        }

        fn runtime_id(_domain_id: DomainId) -> Option<sp_domains::RuntimeId> {
            unreachable!()
        }

        fn runtime_upgrades() -> Vec<sp_domains::RuntimeId> {
            unreachable!()
        }

        fn domain_instance_data(_domain_id: DomainId) -> Option<(DomainInstanceData, NumberFor<Block>)> {
            unreachable!()
        }

        fn domain_timestamp() -> Moment {
            unreachable!()
        }

        fn timestamp() -> Moment {
            unreachable!()
        }

        fn consensus_transaction_byte_fee() -> Balance {
            unreachable!()
        }

        fn consensus_chain_byte_fee() -> Balance {
            unreachable!()
        }

        fn domain_tx_range(_domain_id: DomainId) -> U256 {
            unreachable!()
        }

        fn genesis_state_root(_domain_id: DomainId) -> Option<H256> {
            unreachable!()
        }

        fn head_receipt_number(_domain_id: DomainId) -> DomainNumber {
            unreachable!()
        }

        fn oldest_unconfirmed_receipt_number(_domain_id: DomainId) -> Option<DomainNumber> {
            unreachable!()
        }

        fn domain_bundle_limit(_domain_id: DomainId) -> Option<sp_domains::DomainBundleLimit> {
            unreachable!()
        }

        fn non_empty_er_exists(_domain_id: DomainId) -> bool {
            unreachable!()
        }

        fn domain_best_number(_domain_id: DomainId) -> Option<DomainNumber> {
            unreachable!()
        }

        fn execution_receipt(_receipt_hash: DomainHash) -> Option<ExecutionReceiptFor<DomainHeader, Block, Balance>> {
            unreachable!()
        }

        fn domain_operators(_domain_id: DomainId) -> Option<(BTreeMap<OperatorId, Balance>, Vec<OperatorId>)> {
            unreachable!()
        }

        fn receipt_hash(_domain_id: DomainId, _domain_number: DomainNumber) -> Option<DomainHash> {
            unreachable!()
        }

        fn latest_confirmed_domain_block(_domain_id: DomainId) -> Option<(DomainNumber, DomainHash)>{
            unreachable!()
        }

        fn is_bad_er_pending_to_prune(_domain_id: DomainId, _receipt_hash: DomainHash) -> bool {
            unreachable!()
        }

        fn storage_fund_account_balance(_operator_id: OperatorId) -> Balance {
            unreachable!()
        }

        fn is_domain_runtime_upgraded_since(_domain_id: DomainId, _at: NumberFor<Block>) -> Option<bool> {
            unreachable!()
        }

        fn domain_sudo_call(_domain_id: DomainId) -> Option<Vec<u8>> {
            unreachable!()
        }

        fn evm_domain_contract_creation_allowed_by_call(_domain_id: DomainId) -> Option<PermissionedActionAllowedBy<EthereumAccountId>>{
            unreachable!()
        }

        fn last_confirmed_domain_block_receipt(_domain_id: DomainId) -> Option<ExecutionReceiptFor<DomainHeader, Block, Balance>> {
            unreachable!()
        }
    }

    impl sp_domains::BundleProducerElectionApi<Block, Balance> for Runtime {
        fn bundle_producer_election_params(_domain_id: DomainId) -> Option<BundleProducerElectionParams<Balance>> {
            unreachable!()
        }

        fn operator(_operator_id: OperatorId) -> Option<(OperatorPublicKey, Balance)> {
            unreachable!()
        }
    }

    impl sp_session::SessionKeys<Block> for Runtime {
        fn generate_session_keys(_seed: Option<Vec<u8>>) -> Vec<u8> {
            unreachable!()
        }

        fn decode_session_keys(
            _encoded: Vec<u8>,
        ) -> Option<Vec<(Vec<u8>, KeyTypeId)>> {
            unreachable!()
        }
    }

    impl frame_system_rpc_runtime_api::AccountNonceApi<Block, AccountId, Nonce> for Runtime {
        fn account_nonce(_account: AccountId) -> Nonce {
            unreachable!()
        }
    }

    impl pallet_transaction_payment_rpc_runtime_api::TransactionPaymentApi<Block, Balance> for Runtime {
        fn query_info(
            _uxt: <Block as BlockT>::Extrinsic,
            _len: u32,
        ) -> pallet_transaction_payment_rpc_runtime_api::RuntimeDispatchInfo<Balance> {
            unreachable!()
        }
        fn query_fee_details(
            _uxt: <Block as BlockT>::Extrinsic,
            _len: u32,
        ) -> pallet_transaction_payment::FeeDetails<Balance> {
            unreachable!()
        }
        fn query_weight_to_fee(_weight: Weight) -> Balance {
            unreachable!()
        }
        fn query_length_to_fee(_length: u32) -> Balance {
            unreachable!()
        }
    }

    impl sp_messenger::MessengerApi<Block, BlockNumber, <Block as BlockT>::Hash> for Runtime {
        fn is_xdm_mmr_proof_valid(
            _ext: &<Block as BlockT>::Extrinsic
        ) -> Option<bool> {
            unreachable!()
        }

        fn extract_xdm_mmr_proof(_ext: &<Block as BlockT>::Extrinsic) -> Option<ConsensusChainMmrLeafProof<BlockNumber, <Block as BlockT>::Hash, sp_core::H256>> {
            unreachable!()
        }

        fn confirmed_domain_block_storage_key(_domain_id: DomainId) -> Vec<u8> {
            unreachable!()
        }

        fn outbox_storage_key(_message_key: MessageKey) -> Vec<u8> {
            unreachable!()
        }

        fn inbox_response_storage_key(_message_key: MessageKey) -> Vec<u8> {
            unreachable!()
        }

        fn domain_chains_allowlist_update(_domain_id: DomainId) -> Option<DomainAllowlistUpdates>{
            unreachable!()
        }

        fn xdm_id(_ext: &<Block as BlockT>::Extrinsic) -> Option<XdmId> {
            unreachable!()
        }

        fn channel_nonce(_chain_id: ChainId, _channel_id: ChannelId) -> Option<ChannelNonce> {
            unreachable!()
        }
    }

    impl sp_messenger::RelayerApi<Block, BlockNumber, BlockNumber, <Block as BlockT>::Hash> for Runtime {
        fn block_messages() -> BlockMessagesWithStorageKey {
            unreachable!()
        }

        fn outbox_message_unsigned(_msg: CrossDomainMessage<NumberFor<Block>, <Block as BlockT>::Hash, <Block as BlockT>::Hash>) -> Option<<Block as BlockT>::Extrinsic> {
            unreachable!()
        }

        fn inbox_response_message_unsigned(_msg: CrossDomainMessage<NumberFor<Block>, <Block as BlockT>::Hash, <Block as BlockT>::Hash>) -> Option<<Block as BlockT>::Extrinsic> {
            unreachable!()
        }

        fn should_relay_outbox_message(_dst_chain_id: ChainId, _msg_id: MessageId) -> bool {
            unreachable!()
        }

        fn should_relay_inbox_message_response(_dst_chain_id: ChainId, _msg_id: MessageId) -> bool {
            unreachable!()
        }

        fn updated_channels() -> BTreeSet<(ChainId, ChannelId)> {
            unreachable!()
        }

        fn channel_storage_key(_chain_id: ChainId, _channel_id: ChannelId) -> Vec<u8> {
            unreachable!()
        }
    }

    impl sp_domains_fraud_proof::FraudProofApi<Block, DomainHeader> for Runtime {
        fn submit_fraud_proof_unsigned(_fraud_proof: FraudProof<NumberFor<Block>, <Block as BlockT>::Hash, DomainHeader, H256>) {
            unreachable!()
        }

        fn fraud_proof_storage_key(_req: FraudProofStorageKeyRequest<NumberFor<Block>>) -> Vec<u8> {
            unreachable!()
        }
    }

    impl mmr::MmrApi<Block, mmr::Hash, BlockNumber> for Runtime {
        fn mmr_root() -> Result<mmr::Hash, mmr::Error> {
            unreachable!()
        }

        fn mmr_leaf_count() -> Result<mmr::LeafIndex, mmr::Error> {
            unreachable!()
        }

        fn generate_proof(
            _block_numbers: Vec<BlockNumber>,
            _best_known_block_number: Option<BlockNumber>,
        ) -> Result<(Vec<mmr::EncodableOpaqueLeaf>, mmr::LeafProof<mmr::Hash>), mmr::Error> {
            unreachable!()
        }

        fn verify_proof(_leaves: Vec<mmr::EncodableOpaqueLeaf>, _proof: mmr::LeafProof<mmr::Hash>)
            -> Result<(), mmr::Error>
        {
            unreachable!()
        }

        fn verify_proof_stateless(
            _root: mmr::Hash,
            _leaves: Vec<mmr::EncodableOpaqueLeaf>,
            _proof: mmr::LeafProof<mmr::Hash>
        ) -> Result<(), mmr::Error> {
            unreachable!()
        }
    }

    impl sp_genesis_builder::GenesisBuilder<Block> for Runtime {
        fn build_state(_config: Vec<u8>) -> sp_genesis_builder::Result {
            unreachable!()
        }

        fn get_preset(_id: &Option<sp_genesis_builder::PresetId>) -> Option<Vec<u8>> {
            unreachable!()
        }

        fn preset_names() -> Vec<sp_genesis_builder::PresetId> {
            unreachable!()
        }
    }
}
