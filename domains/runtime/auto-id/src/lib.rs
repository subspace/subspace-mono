#![feature(variant_count)]
#![cfg_attr(not(feature = "std"), no_std)]
// `construct_runtime!` does a lot of recursion and requires us to increase the limit to 256.
#![recursion_limit = "256"]

// Make the WASM binary available.
#[cfg(feature = "std")]
include!(concat!(env!("OUT_DIR"), "/wasm_binary.rs"));

extern crate alloc;

use alloc::borrow::Cow;
#[cfg(not(feature = "std"))]
use alloc::format;
use codec::{Decode, Encode, MaxEncodedLen};
use core::mem;
use domain_runtime_primitives::opaque::Header;
pub use domain_runtime_primitives::{
    block_weights, maximum_block_length, opaque, Balance, BlockNumber, Hash, Nonce,
    EXISTENTIAL_DEPOSIT,
};
use domain_runtime_primitives::{
    AccountId, Address, CheckExtrinsicsValidityError, DecodeExtrinsicError, HoldIdentifier,
    Signature, ERR_BALANCE_OVERFLOW, SLOT_DURATION,
};
use frame_support::dispatch::{DispatchClass, DispatchInfo, GetDispatchInfo};
use frame_support::genesis_builder_helper::{build_state, get_preset};
use frame_support::inherent::ProvideInherent;
use frame_support::pallet_prelude::TypeInfo;
use frame_support::traits::fungible::Credit;
use frame_support::traits::{
    ConstU16, ConstU32, ConstU64, Everything, Imbalance, OnUnbalanced, VariantCount,
};
use frame_support::weights::constants::ParityDbWeight;
use frame_support::weights::{ConstantMultiplier, Weight};
use frame_support::{construct_runtime, parameter_types};
use frame_system::limits::{BlockLength, BlockWeights};
use pallet_block_fees::fees::OnChargeDomainTransaction;
use pallet_transporter::EndpointHandler;
use sp_api::impl_runtime_apis;
use sp_core::crypto::KeyTypeId;
use sp_core::{Get, OpaqueMetadata};
use sp_domains::{ChannelId, DomainAllowlistUpdates, DomainId, Transfers};
use sp_messenger::endpoint::{Endpoint, EndpointHandler as EndpointHandlerT, EndpointId};
use sp_messenger::messages::{
    BlockMessagesWithStorageKey, ChainId, CrossDomainMessage, FeeModel, MessageId, MessageKey,
};
use sp_messenger::{ChannelNonce, XdmId};
use sp_messenger_host_functions::{get_storage_key, StorageKeyRequest};
use sp_mmr_primitives::EncodableOpaqueLeaf;
use sp_runtime::generic::Era;
use sp_runtime::traits::{
    AccountIdLookup, BlakeTwo256, Block as BlockT, Checkable, Keccak256, NumberFor, One,
    SignedExtension, ValidateUnsigned, Zero,
};
use sp_runtime::transaction_validity::{
    InvalidTransaction, TransactionSource, TransactionValidity, TransactionValidityError,
};
use sp_runtime::{generic, impl_opaque_keys, ApplyExtrinsicResult, Digest, ExtrinsicInclusionMode};
pub use sp_runtime::{MultiAddress, Perbill, Permill};
use sp_std::collections::btree_set::BTreeSet;
use sp_std::marker::PhantomData;
use sp_std::prelude::*;
use sp_subspace_mmr::domain_mmr_runtime_interface::{
    is_consensus_block_finalized, verify_mmr_proof,
};
use sp_subspace_mmr::{ConsensusChainMmrLeafProof, MmrLeaf};
use sp_version::RuntimeVersion;
use subspace_runtime_primitives::{
    BlockNumber as ConsensusBlockNumber, Hash as ConsensusBlockHash, Moment,
    SlowAdjustingFeeUpdate, SHANNON, SSC,
};

/// Block type as expected by this runtime.
pub type Block = generic::Block<Header, UncheckedExtrinsic>;

/// A Block signed with a Justification
pub type SignedBlock = generic::SignedBlock<Block>;

/// BlockId type as expected by this runtime.
pub type BlockId = generic::BlockId<Block>;

/// The SignedExtension to the basic transaction logic.
pub type SignedExtra = (
    frame_system::CheckNonZeroSender<Runtime>,
    frame_system::CheckSpecVersion<Runtime>,
    frame_system::CheckTxVersion<Runtime>,
    frame_system::CheckGenesis<Runtime>,
    frame_system::CheckMortality<Runtime>,
    frame_system::CheckNonce<Runtime>,
    domain_check_weight::CheckWeight<Runtime>,
    pallet_transaction_payment::ChargeTransactionPayment<Runtime>,
);

/// Unchecked extrinsic type as expected by this runtime.
pub type UncheckedExtrinsic =
    generic::UncheckedExtrinsic<Address, RuntimeCall, Signature, SignedExtra>;

/// Extrinsic type that has already been checked.
pub type CheckedExtrinsic = generic::CheckedExtrinsic<AccountId, RuntimeCall, SignedExtra>;

/// Executive: handles dispatch to the various modules.
pub type Executive = domain_pallet_executive::Executive<
    Runtime,
    frame_system::ChainContext<Runtime>,
    Runtime,
    AllPalletsWithSystem,
>;

impl_opaque_keys! {
    pub struct SessionKeys {
        /// Primarily used for adding the operator signing key into the Keystore.
        pub operator: sp_domains::OperatorKey,
    }
}

#[sp_version::runtime_version]
pub const VERSION: RuntimeVersion = RuntimeVersion {
    spec_name: Cow::Borrowed("subspace-auto-id-domain"),
    impl_name: Cow::Borrowed("subspace-auto-id-domain"),
    authoring_version: 0,
    spec_version: 0,
    impl_version: 0,
    apis: RUNTIME_API_VERSIONS,
    transaction_version: 0,
    system_version: 2,
};

parameter_types! {
    pub const Version: RuntimeVersion = VERSION;
    pub const BlockHashCount: BlockNumber = 2400;
    pub RuntimeBlockLength: BlockLength = maximum_block_length();
    pub RuntimeBlockWeights: BlockWeights = block_weights();
}

impl frame_system::Config for Runtime {
    /// The identifier used to distinguish between accounts.
    type AccountId = AccountId;
    /// The aggregated dispatch type that is available for extrinsics.
    type RuntimeCall = RuntimeCall;
    /// The aggregated `RuntimeTask` type.
    type RuntimeTask = RuntimeTask;
    /// The lookup mechanism to get account ID from whatever is passed in dispatchers.
    type Lookup = AccountIdLookup<AccountId, ()>;
    /// The type for storing how many extrinsics an account has signed.
    type Nonce = Nonce;
    /// The type for hashing blocks and tries.
    type Hash = Hash;
    /// The hashing algorithm used.
    type Hashing = BlakeTwo256;
    /// The block type.
    type Block = Block;
    /// The ubiquitous event type.
    type RuntimeEvent = RuntimeEvent;
    /// The ubiquitous origin type.
    type RuntimeOrigin = RuntimeOrigin;
    /// Maximum number of block number to block hash mappings to keep (oldest pruned first).
    type BlockHashCount = BlockHashCount;
    /// Runtime version.
    type Version = Version;
    /// Converts a module to an index of this module in the runtime.
    type PalletInfo = PalletInfo;
    /// The data to be stored in an account.
    type AccountData = pallet_balances::AccountData<Balance>;
    /// What to do if a new account is created.
    type OnNewAccount = ();
    /// What to do if an account is fully reaped from the system.
    type OnKilledAccount = ();
    /// The weight of database operations that the runtime can invoke.
    type DbWeight = ParityDbWeight;
    /// The basic call filter to use in dispatchable.
    type BaseCallFilter = Everything;
    /// Weight information for the extrinsics of this pallet.
    type SystemWeightInfo = ();
    /// Block & extrinsics weights: base values and limits.
    type BlockWeights = RuntimeBlockWeights;
    /// The maximum length of a block (in bytes).
    type BlockLength = RuntimeBlockLength;
    type SS58Prefix = ConstU16<6094>;
    /// The action to take on a Runtime Upgrade
    type OnSetCode = ();
    type SingleBlockMigrations = ();
    type MultiBlockMigrator = ();
    type PreInherents = ();
    type PostInherents = ();
    type PostTransactions = ();
    type MaxConsumers = ConstU32<16>;
}

impl pallet_timestamp::Config for Runtime {
    /// A timestamp: milliseconds since the unix epoch.
    type Moment = Moment;
    type OnTimestampSet = ();
    type MinimumPeriod = ConstU64<{ SLOT_DURATION / 2 }>;
    type WeightInfo = ();
}

parameter_types! {
    pub const ExistentialDeposit: Balance = EXISTENTIAL_DEPOSIT;
    pub const MaxLocks: u32 = 50;
    pub const MaxReserves: u32 = 50;
}

/// `DustRemovalHandler` used to collect all the SSC dust left when the account is reaped.
pub struct DustRemovalHandler;

impl OnUnbalanced<Credit<AccountId, Balances>> for DustRemovalHandler {
    fn on_nonzero_unbalanced(dusted_amount: Credit<AccountId, Balances>) {
        BlockFees::note_burned_balance(dusted_amount.peek());
    }
}

impl pallet_balances::Config for Runtime {
    type RuntimeFreezeReason = RuntimeFreezeReason;
    type MaxLocks = MaxLocks;
    /// The type for recording an account's balance.
    type Balance = Balance;
    /// The ubiquitous event type.
    type RuntimeEvent = RuntimeEvent;
    type DustRemoval = DustRemovalHandler;
    type ExistentialDeposit = ExistentialDeposit;
    type AccountStore = System;
    type WeightInfo = pallet_balances::weights::SubstrateWeight<Runtime>;
    type MaxReserves = MaxReserves;
    type ReserveIdentifier = [u8; 8];
    type FreezeIdentifier = ();
    type MaxFreezes = ();
    type RuntimeHoldReason = HoldIdentifierWrapper;
}

parameter_types! {
    pub const OperationalFeeMultiplier: u8 = 5;
    pub const DomainChainByteFee: Balance = 1;
    pub TransactionWeightFee: Balance = 100_000 * SHANNON;
}

impl pallet_block_fees::Config for Runtime {
    type Balance = Balance;
    type DomainChainByteFee = DomainChainByteFee;
}

pub struct FinalDomainTransactionByteFee;

impl Get<Balance> for FinalDomainTransactionByteFee {
    fn get() -> Balance {
        BlockFees::final_domain_transaction_byte_fee()
    }
}

impl pallet_transaction_payment::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type OnChargeTransaction = OnChargeDomainTransaction<Balances>;
    type WeightToFee = ConstantMultiplier<Balance, TransactionWeightFee>;
    type LengthToFee = ConstantMultiplier<Balance, FinalDomainTransactionByteFee>;
    type FeeMultiplierUpdate = SlowAdjustingFeeUpdate<Runtime>;
    type OperationalFeeMultiplier = OperationalFeeMultiplier;
}

impl pallet_auto_id::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type Time = Timestamp;
    type Weights = pallet_auto_id::weights::SubstrateWeight<Self>;
}

pub struct ExtrinsicStorageFees;

impl domain_pallet_executive::ExtrinsicStorageFees<Runtime> for ExtrinsicStorageFees {
    fn extract_signer(xt: UncheckedExtrinsic) -> (Option<AccountId>, DispatchInfo) {
        let dispatch_info = xt.get_dispatch_info();
        let lookup = frame_system::ChainContext::<Runtime>::default();
        let maybe_signer = extract_signer_inner(&xt, &lookup).and_then(|res| res.ok());
        (maybe_signer, dispatch_info)
    }

    fn on_storage_fees_charged(
        charged_fees: Balance,
        tx_size: u32,
    ) -> Result<(), TransactionValidityError> {
        let consensus_storage_fee = BlockFees::consensus_chain_byte_fee()
            .checked_mul(Balance::from(tx_size))
            .ok_or(InvalidTransaction::Custom(ERR_BALANCE_OVERFLOW))?;

        let (paid_consensus_storage_fee, paid_domain_fee) = if charged_fees <= consensus_storage_fee
        {
            (charged_fees, Zero::zero())
        } else {
            (consensus_storage_fee, charged_fees - consensus_storage_fee)
        };

        BlockFees::note_consensus_storage_fee(paid_consensus_storage_fee);
        BlockFees::note_domain_execution_fee(paid_domain_fee);
        Ok(())
    }
}

impl domain_pallet_executive::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type WeightInfo = domain_pallet_executive::weights::SubstrateWeight<Runtime>;
    type Currency = Balances;
    type LengthToFee = <Runtime as pallet_transaction_payment::Config>::LengthToFee;
    type ExtrinsicStorageFees = ExtrinsicStorageFees;
}

parameter_types! {
    pub SelfChainId: ChainId = SelfDomainId::self_domain_id().into();
}

pub struct OnXDMRewards;

impl sp_messenger::OnXDMRewards<Balance> for OnXDMRewards {
    fn on_xdm_rewards(rewards: Balance) {
        BlockFees::note_domain_execution_fee(rewards)
    }
    fn on_chain_protocol_fees(chain_id: ChainId, fees: Balance) {
        // note the burned balance from this chain
        BlockFees::note_burned_balance(fees);
        // note the chain rewards
        BlockFees::note_chain_rewards(chain_id, fees);
    }
}

type MmrHash = <Keccak256 as sp_runtime::traits::Hash>::Output;

pub struct MmrProofVerifier;

impl sp_subspace_mmr::MmrProofVerifier<MmrHash, NumberFor<Block>, Hash> for MmrProofVerifier {
    fn verify_proof_and_extract_leaf(
        mmr_leaf_proof: ConsensusChainMmrLeafProof<NumberFor<Block>, Hash, MmrHash>,
    ) -> Option<MmrLeaf<ConsensusBlockNumber, ConsensusBlockHash>> {
        let ConsensusChainMmrLeafProof {
            consensus_block_number,
            opaque_mmr_leaf: opaque_leaf,
            proof,
            ..
        } = mmr_leaf_proof;

        if !is_consensus_block_finalized(consensus_block_number) {
            return None;
        }

        let leaf: MmrLeaf<ConsensusBlockNumber, ConsensusBlockHash> =
            opaque_leaf.into_opaque_leaf().try_decode()?;

        verify_mmr_proof(vec![EncodableOpaqueLeaf::from_leaf(&leaf)], proof.encode())
            .then_some(leaf)
    }
}

pub struct StorageKeys;

impl sp_messenger::StorageKeys for StorageKeys {
    fn confirmed_domain_block_storage_key(domain_id: DomainId) -> Option<Vec<u8>> {
        get_storage_key(StorageKeyRequest::ConfirmedDomainBlockStorageKey(domain_id))
    }

    fn outbox_storage_key(chain_id: ChainId, message_key: MessageKey) -> Option<Vec<u8>> {
        get_storage_key(StorageKeyRequest::OutboxStorageKey {
            chain_id,
            message_key,
        })
    }

    fn inbox_responses_storage_key(chain_id: ChainId, message_key: MessageKey) -> Option<Vec<u8>> {
        get_storage_key(StorageKeyRequest::InboxResponseStorageKey {
            chain_id,
            message_key,
        })
    }
}

/// Hold identifier for balances for this runtime.
#[derive(
    PartialEq, Eq, Clone, Encode, Decode, TypeInfo, MaxEncodedLen, Ord, PartialOrd, Copy, Debug,
)]
pub struct HoldIdentifierWrapper(HoldIdentifier);

impl VariantCount for HoldIdentifierWrapper {
    const VARIANT_COUNT: u32 = mem::variant_count::<HoldIdentifier>() as u32;
}

impl pallet_messenger::HoldIdentifier<Runtime> for HoldIdentifierWrapper {
    fn messenger_channel() -> Self {
        Self(HoldIdentifier::MessengerChannel)
    }
}

parameter_types! {
    pub const ChannelReserveFee: Balance = 100 * SSC;
    pub const ChannelInitReservePortion: Perbill = Perbill::from_percent(20);
    // TODO update the fee model
    pub const ChannelFeeModel: FeeModel<Balance> = FeeModel{relay_fee: SSC};
}

impl pallet_messenger::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type SelfChainId = SelfChainId;

    fn get_endpoint_handler(endpoint: &Endpoint) -> Option<Box<dyn EndpointHandlerT<MessageId>>> {
        if endpoint == &Endpoint::Id(TransporterEndpointId::get()) {
            Some(Box::new(EndpointHandler(PhantomData::<Runtime>)))
        } else {
            None
        }
    }

    type Currency = Balances;
    type WeightInfo = pallet_messenger::weights::SubstrateWeight<Runtime>;
    type WeightToFee = ConstantMultiplier<Balance, TransactionWeightFee>;
    type OnXDMRewards = OnXDMRewards;
    type MmrHash = MmrHash;
    type MmrProofVerifier = MmrProofVerifier;
    type StorageKeys = StorageKeys;
    type DomainOwner = ();
    type HoldIdentifier = HoldIdentifierWrapper;
    type ChannelReserveFee = ChannelReserveFee;
    type ChannelInitReservePortion = ChannelInitReservePortion;
    type DomainRegistration = ();
    type ChannelFeeModel = ChannelFeeModel;
}

impl<C> frame_system::offchain::SendTransactionTypes<C> for Runtime
where
    RuntimeCall: From<C>,
{
    type Extrinsic = UncheckedExtrinsic;
    type OverarchingCall = RuntimeCall;
}

parameter_types! {
    pub const TransporterEndpointId: EndpointId = 1;
}

impl pallet_transporter::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type SelfChainId = SelfChainId;
    type SelfEndpointId = TransporterEndpointId;
    type Currency = Balances;
    type Sender = Messenger;
    type AccountIdConverter = domain_runtime_primitives::AccountIdConverter;
    type WeightInfo = pallet_transporter::weights::SubstrateWeight<Runtime>;
}

impl pallet_domain_id::Config for Runtime {}

pub struct IntoRuntimeCall;

impl sp_domain_sudo::IntoRuntimeCall<RuntimeCall> for IntoRuntimeCall {
    fn runtime_call(call: Vec<u8>) -> RuntimeCall {
        UncheckedExtrinsic::decode(&mut call.as_slice())
            .expect("must always be a valid extrinsic as checked by consensus chain; qed")
            .function
    }
}

impl pallet_domain_sudo::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type RuntimeCall = RuntimeCall;
    type IntoRuntimeCall = IntoRuntimeCall;
}

impl pallet_utility::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type RuntimeCall = RuntimeCall;
    type PalletsOrigin = OriginCaller;
    type WeightInfo = pallet_utility::weights::SubstrateWeight<Runtime>;
}

// Create the runtime by composing the FRAME pallets that were previously configured.
//
// NOTE: Currently domain runtime does not naturally support the pallets with inherent extrinsics.
construct_runtime!(
    pub struct Runtime {
        // System support stuff.
        System: frame_system = 0,
        // Note: Ensure index of the timestamp matches with the index of timestamp on Consensus
        //  so that consensus can construct encoded extrinsic that matches with Domain encoded
        //  extrinsic.
        Timestamp: pallet_timestamp = 1,
        ExecutivePallet: domain_pallet_executive = 2,
        Utility: pallet_utility = 8,

        // monetary stuff
        Balances: pallet_balances = 20,
        TransactionPayment: pallet_transaction_payment = 21,

        // AutoId
        AutoId: pallet_auto_id = 40,

        // messenger stuff
        // Note: Indexes should match with indexes on other chains and domains
        Messenger: pallet_messenger = 60,
        Transporter: pallet_transporter = 61,

        // domain instance stuff
        SelfDomainId: pallet_domain_id = 90,
        BlockFees: pallet_block_fees = 91,

        // Sudo account
        Sudo: pallet_domain_sudo = 100,
    }
);

fn is_xdm_mmr_proof_valid(ext: &<Block as BlockT>::Extrinsic) -> Option<bool> {
    match &ext.function {
        RuntimeCall::Messenger(pallet_messenger::Call::relay_message { msg })
        | RuntimeCall::Messenger(pallet_messenger::Call::relay_message_response { msg }) => {
            let ConsensusChainMmrLeafProof {
                consensus_block_number,
                opaque_mmr_leaf,
                proof,
                ..
            } = msg.proof.consensus_mmr_proof();

            if !is_consensus_block_finalized(consensus_block_number) {
                return Some(false);
            }

            Some(verify_mmr_proof(vec![opaque_mmr_leaf], proof.encode()))
        }
        _ => None,
    }
}

/// Returns a valid Sudo call.
/// Should extend this function to limit specific calls Sudo can make when needed.
fn is_valid_sudo_call(encoded_ext: Vec<u8>) -> bool {
    UncheckedExtrinsic::decode(&mut encoded_ext.as_slice()).is_ok()
}

fn construct_sudo_call_extrinsic(encoded_ext: Vec<u8>) -> <Block as BlockT>::Extrinsic {
    let ext = UncheckedExtrinsic::decode(&mut encoded_ext.as_slice())
        .expect("must always be an valid extrinsic due to the check above; qed");
    UncheckedExtrinsic::new_unsigned(
        pallet_domain_sudo::Call::sudo {
            call: Box::new(ext.function),
        }
        .into(),
    )
}

fn extract_signer_inner<Lookup>(
    ext: &UncheckedExtrinsic,
    lookup: &Lookup,
) -> Option<Result<AccountId, TransactionValidityError>>
where
    Lookup: sp_runtime::traits::Lookup<Source = Address, Target = AccountId>,
{
    ext.signature
        .as_ref()
        .map(|(signed, _, _)| lookup.lookup(signed.clone()).map_err(|e| e.into()))
}

pub fn extract_signer(
    extrinsics: Vec<UncheckedExtrinsic>,
) -> Vec<(Option<opaque::AccountId>, UncheckedExtrinsic)> {
    let lookup = frame_system::ChainContext::<Runtime>::default();

    extrinsics
        .into_iter()
        .map(|extrinsic| {
            let maybe_signer =
                extract_signer_inner(&extrinsic, &lookup).and_then(|account_result| {
                    account_result.ok().map(|account_id| account_id.encode())
                });
            (maybe_signer, extrinsic)
        })
        .collect()
}

fn extrinsic_era(extrinsic: &<Block as BlockT>::Extrinsic) -> Option<Era> {
    extrinsic.signature.as_ref().map(|(_, _, extra)| extra.4 .0)
}

#[cfg(feature = "runtime-benchmarks")]
mod benches {
    frame_benchmarking::define_benchmarks!(
        [frame_benchmarking, BaselineBench::<Runtime>]
        [frame_system, SystemBench::<Runtime>]
        [domain_pallet_executive, ExecutivePallet]
        [pallet_messenger, Messenger]
        [pallet_auto_id, AutoId]
    );
}

fn check_transaction_and_do_pre_dispatch_inner(
    uxt: &<Block as BlockT>::Extrinsic,
) -> Result<(), TransactionValidityError> {
    let lookup = frame_system::ChainContext::<Runtime>::default();

    let xt = uxt.clone().check(&lookup)?;

    let dispatch_info = xt.get_dispatch_info();

    if dispatch_info.class == DispatchClass::Mandatory {
        return Err(InvalidTransaction::MandatoryValidation.into());
    }

    let encoded_len = uxt.encoded_size();

    // We invoke `pre_dispatch` in addition to `validate_transaction`(even though the validation is almost same)
    // as that will add the side effect of SignedExtension in the storage buffer
    // which would help to maintain context across multiple transaction validity check against same
    // runtime instance.
    match xt.signed {
        // signed transaction
        Some((account_id, extra)) => extra
            .pre_dispatch(&account_id, &xt.function, &dispatch_info, encoded_len)
            .map(|_| ()),
        // unsigned transaction
        None => {
            if let RuntimeCall::Messenger(call) = &xt.function {
                Messenger::pre_dispatch_with_trusted_mmr_proof(call)?;
            } else {
                Runtime::pre_dispatch(&xt.function).map(|_| ())?;
            }
            SignedExtra::pre_dispatch_unsigned(&xt.function, &dispatch_info, encoded_len)
                .map(|_| ())
        }
    }
}

#[cfg(feature = "runtime-benchmarks")]
impl frame_system_benchmarking::Config for Runtime {}

#[cfg(feature = "runtime-benchmarks")]
impl frame_benchmarking::baseline::Config for Runtime {}

impl_runtime_apis! {
    impl sp_api::Core<Block> for Runtime {
        fn version() -> RuntimeVersion {
            VERSION
        }

        fn execute_block(block: Block) {
            Executive::execute_block(block)
        }

        fn initialize_block(header: &<Block as BlockT>::Header) -> ExtrinsicInclusionMode {
            Executive::initialize_block(header)
        }
    }

    impl sp_api::Metadata<Block> for Runtime {
        fn metadata() -> OpaqueMetadata {
            OpaqueMetadata::new(Runtime::metadata().into())
        }

        fn metadata_at_version(version: u32) -> Option<OpaqueMetadata> {
            Runtime::metadata_at_version(version)
        }

        fn metadata_versions() -> Vec<u32> {
            Runtime::metadata_versions()
        }
    }

    impl sp_block_builder::BlockBuilder<Block> for Runtime {
        fn apply_extrinsic(extrinsic: <Block as BlockT>::Extrinsic) -> ApplyExtrinsicResult {
            Executive::apply_extrinsic(extrinsic)
        }

        fn finalize_block() -> <Block as BlockT>::Header {
            Executive::finalize_block()
        }

        fn inherent_extrinsics(data: sp_inherents::InherentData) -> Vec<<Block as BlockT>::Extrinsic> {
            data.create_extrinsics()
        }

        fn check_inherents(
            block: Block,
            data: sp_inherents::InherentData,
        ) -> sp_inherents::CheckInherentsResult {
            data.check_extrinsics(&block)
        }
    }

    impl sp_transaction_pool::runtime_api::TaggedTransactionQueue<Block> for Runtime {
        fn validate_transaction(
            source: TransactionSource,
            tx: <Block as BlockT>::Extrinsic,
            block_hash: <Block as BlockT>::Hash,
        ) -> TransactionValidity {
            Executive::validate_transaction(source, tx, block_hash)
        }
    }

    impl sp_offchain::OffchainWorkerApi<Block> for Runtime {
        fn offchain_worker(header: &<Block as BlockT>::Header) {
            Executive::offchain_worker(header)
        }
    }

    impl sp_session::SessionKeys<Block> for Runtime {
        fn generate_session_keys(seed: Option<Vec<u8>>) -> Vec<u8> {
            SessionKeys::generate(seed)
        }

        fn decode_session_keys(
            encoded: Vec<u8>,
        ) -> Option<Vec<(Vec<u8>, KeyTypeId)>> {
            SessionKeys::decode_into_raw_public_keys(&encoded)
        }
    }

    impl frame_system_rpc_runtime_api::AccountNonceApi<Block, AccountId, Nonce> for Runtime {
        fn account_nonce(account: AccountId) -> Nonce {
            System::account_nonce(account)
        }
    }

    impl pallet_transaction_payment_rpc_runtime_api::TransactionPaymentApi<Block, Balance> for Runtime {
        fn query_info(
            uxt: <Block as BlockT>::Extrinsic,
            len: u32,
        ) -> pallet_transaction_payment_rpc_runtime_api::RuntimeDispatchInfo<Balance> {
            TransactionPayment::query_info(uxt, len)
        }
        fn query_fee_details(
            uxt: <Block as BlockT>::Extrinsic,
            len: u32,
        ) -> pallet_transaction_payment::FeeDetails<Balance> {
            TransactionPayment::query_fee_details(uxt, len)
        }
        fn query_weight_to_fee(weight: Weight) -> Balance {
            TransactionPayment::weight_to_fee(weight)
        }
        fn query_length_to_fee(length: u32) -> Balance {
            TransactionPayment::length_to_fee(length)
        }
    }

    impl sp_domains::core_api::DomainCoreApi<Block> for Runtime {
        fn extract_signer(
            extrinsics: Vec<<Block as BlockT>::Extrinsic>,
        ) -> Vec<(Option<opaque::AccountId>, <Block as BlockT>::Extrinsic)> {
            extract_signer(extrinsics)
        }

        fn is_within_tx_range(
            extrinsic: &<Block as BlockT>::Extrinsic,
            bundle_vrf_hash: &subspace_core_primitives::U256,
            tx_range: &subspace_core_primitives::U256
        ) -> bool {
            use subspace_core_primitives::U256;
            use subspace_core_primitives::hashes::blake3_hash;

            let lookup = frame_system::ChainContext::<Runtime>::default();
            if let Some(signer) = extract_signer_inner(extrinsic, &lookup).and_then(|account_result| {
                    account_result.ok().map(|account_id| account_id.encode())
                }) {
                // Check if the signer Id hash is within the tx range
                let signer_id_hash = U256::from_be_bytes(*blake3_hash(&signer.encode()));
                sp_domains::signer_in_tx_range(bundle_vrf_hash, &signer_id_hash, tx_range)
            } else {
                // Unsigned transactions are always in the range.
                true
            }
        }

        fn initialize_block_with_post_state_root(header: &<Block as BlockT>::Header) -> Vec<u8> {
            Executive::initialize_block(header);
            Executive::storage_root()
        }

        fn apply_extrinsic_with_post_state_root(extrinsic: <Block as BlockT>::Extrinsic) -> Vec<u8> {
            let _ = Executive::apply_extrinsic(extrinsic);
            Executive::storage_root()
        }

        fn construct_set_code_extrinsic(code: Vec<u8>) -> Vec<u8> {
            UncheckedExtrinsic::new_unsigned(
                domain_pallet_executive::Call::set_code {
                    code
                }.into()
            ).encode()
        }

        fn construct_timestamp_extrinsic(moment: Moment) -> <Block as BlockT>::Extrinsic {
            UncheckedExtrinsic::new_unsigned(
                pallet_timestamp::Call::set{ now: moment }.into()
            )
        }

        fn is_inherent_extrinsic(extrinsic: &<Block as BlockT>::Extrinsic) -> bool {
            match &extrinsic.function {
                RuntimeCall::Timestamp(call) => Timestamp::is_inherent(call),
                RuntimeCall::ExecutivePallet(call) => ExecutivePallet::is_inherent(call),
                RuntimeCall::Messenger(call) => Messenger::is_inherent(call),
                RuntimeCall::Sudo(call) => Sudo::is_inherent(call),
                _ => false,
            }
        }

        fn check_extrinsics_and_do_pre_dispatch(uxts: Vec<<Block as BlockT>::Extrinsic>, block_number: BlockNumber,
            block_hash: <Block as BlockT>::Hash) -> Result<(), CheckExtrinsicsValidityError> {
            // Initializing block related storage required for validation
            System::initialize(
                &(block_number + BlockNumber::one()),
                &block_hash,
                &Default::default(),
            );

            for (extrinsic_index, uxt) in uxts.iter().enumerate() {
                check_transaction_and_do_pre_dispatch_inner(uxt).map_err(|e| {
                    CheckExtrinsicsValidityError {
                        extrinsic_index: extrinsic_index as u32,
                        transaction_validity_error: e
                    }
                })?;
            }

            Ok(())
        }

        fn decode_extrinsic(
            opaque_extrinsic: sp_runtime::OpaqueExtrinsic,
        ) -> Result<<Block as BlockT>::Extrinsic, DecodeExtrinsicError> {
            let encoded = opaque_extrinsic.encode();
            UncheckedExtrinsic::decode(&mut encoded.as_slice())
                .map_err(|err| DecodeExtrinsicError(format!("{}", err)))
        }

        fn extrinsic_era(
          extrinsic: &<Block as BlockT>::Extrinsic
        ) -> Option<Era> {
            extrinsic_era(extrinsic)
        }

        fn extrinsic_weight(ext: &<Block as BlockT>::Extrinsic) -> Weight {
            let len = ext.encoded_size() as u64;
            let info = ext.get_dispatch_info();
            info.weight
                .saturating_add(<Runtime as frame_system::Config>::BlockWeights::get().get(info.class).base_extrinsic)
                .saturating_add(Weight::from_parts(0, len))
        }

        fn block_fees() -> sp_domains::BlockFees<Balance> {
            BlockFees::collected_block_fees()
        }

        fn block_digest() -> Digest {
            System::digest()
        }

        fn block_weight() -> Weight {
            System::block_weight().total()
        }

        fn construct_consensus_chain_byte_fee_extrinsic(transaction_byte_fee: Balance) -> <Block as BlockT>::Extrinsic {
            UncheckedExtrinsic::new_unsigned(
                pallet_block_fees::Call::set_next_consensus_chain_byte_fee{ transaction_byte_fee }.into()
            )
        }

        fn construct_domain_update_chain_allowlist_extrinsic(updates: DomainAllowlistUpdates) -> <Block as BlockT>::Extrinsic {
             UncheckedExtrinsic::new_unsigned(
                pallet_messenger::Call::update_domain_allowlist{ updates }.into()
            )
        }

        fn transfers() -> Transfers<Balance> {
            Transporter::chain_transfers()
        }

        fn transfers_storage_key() -> Vec<u8> {
            Transporter::transfers_storage_key()
        }

        fn block_fees_storage_key() -> Vec<u8> {
            BlockFees::block_fees_storage_key()
        }
    }

    impl sp_messenger::MessengerApi<Block, ConsensusBlockNumber, ConsensusBlockHash> for Runtime {
        fn is_xdm_mmr_proof_valid(
            extrinsic: &<Block as BlockT>::Extrinsic,
        ) -> Option<bool> {
            is_xdm_mmr_proof_valid(extrinsic)
        }

        fn extract_xdm_mmr_proof(ext: &<Block as BlockT>::Extrinsic) -> Option<ConsensusChainMmrLeafProof<ConsensusBlockNumber, ConsensusBlockHash, sp_core::H256>> {
            match &ext.function {
                RuntimeCall::Messenger(pallet_messenger::Call::relay_message { msg })
                | RuntimeCall::Messenger(pallet_messenger::Call::relay_message_response { msg }) => {
                    Some(msg.proof.consensus_mmr_proof())
                }
                _ => None,
            }
        }

        fn confirmed_domain_block_storage_key(_domain_id: DomainId) -> Vec<u8> {
            // invalid call from Domain runtime
            vec![]
        }

        fn outbox_storage_key(message_key: MessageKey) -> Vec<u8> {
            Messenger::outbox_storage_key(message_key)
        }

        fn inbox_response_storage_key(message_key: MessageKey) -> Vec<u8> {
            Messenger::inbox_response_storage_key(message_key)
        }

        fn domain_chains_allowlist_update(_domain_id: DomainId) -> Option<DomainAllowlistUpdates>{
            // not valid call on domains
            None
        }

        fn xdm_id(ext: &<Block as BlockT>::Extrinsic) -> Option<XdmId> {
            match &ext.function {
                RuntimeCall::Messenger(pallet_messenger::Call::relay_message { msg })=> {
                    Some(XdmId::RelayMessage((msg.src_chain_id, msg.channel_id, msg.nonce)))
                }
                RuntimeCall::Messenger(pallet_messenger::Call::relay_message_response { msg }) => {
                    Some(XdmId::RelayResponseMessage((msg.src_chain_id, msg.channel_id, msg.nonce)))
                }
                _ => None,
            }
        }

        fn channel_nonce(chain_id: ChainId, channel_id: ChannelId) -> Option<ChannelNonce> {
            Messenger::channel_nonce(chain_id, channel_id)
        }
    }

    impl sp_messenger::RelayerApi<Block, BlockNumber, ConsensusBlockNumber, ConsensusBlockHash> for Runtime {
        fn block_messages() -> BlockMessagesWithStorageKey {
            Messenger::get_block_messages()
        }

        fn outbox_message_unsigned(msg: CrossDomainMessage<NumberFor<Block>, <Block as BlockT>::Hash, <Block as BlockT>::Hash>) -> Option<<Block as BlockT>::Extrinsic> {
            Messenger::outbox_message_unsigned(msg)
        }

        fn inbox_response_message_unsigned(msg: CrossDomainMessage<NumberFor<Block>, <Block as BlockT>::Hash, <Block as BlockT>::Hash>) -> Option<<Block as BlockT>::Extrinsic> {
            Messenger::inbox_response_message_unsigned(msg)
        }

        fn should_relay_outbox_message(dst_chain_id: ChainId, msg_id: MessageId) -> bool {
            Messenger::should_relay_outbox_message(dst_chain_id, msg_id)
        }

        fn should_relay_inbox_message_response(dst_chain_id: ChainId, msg_id: MessageId) -> bool {
            Messenger::should_relay_inbox_message_response(dst_chain_id, msg_id)
        }

        fn updated_channels() -> BTreeSet<(ChainId, ChannelId)> {
            Messenger::updated_channels()
        }

        fn channel_storage_key(chain_id: ChainId, channel_id: ChannelId) -> Vec<u8> {
            Messenger::channel_storage_key(chain_id, channel_id)
        }
    }

    impl sp_domain_sudo::DomainSudoApi<Block> for Runtime {
        fn is_valid_sudo_call(extrinsic: Vec<u8>) -> bool {
            is_valid_sudo_call(extrinsic)
        }

        fn construct_domain_sudo_extrinsic(inner: Vec<u8>) -> <Block as BlockT>::Extrinsic {
            construct_sudo_call_extrinsic(inner)
        }
    }

    impl sp_genesis_builder::GenesisBuilder<Block> for Runtime {
        fn build_state(config: Vec<u8>) -> sp_genesis_builder::Result {
            build_state::<RuntimeGenesisConfig>(config)
        }

        fn get_preset(id: &Option<sp_genesis_builder::PresetId>) -> Option<Vec<u8>> {
            get_preset::<RuntimeGenesisConfig>(id, |_| None)
        }

        fn preset_names() -> Vec<sp_genesis_builder::PresetId> {
            vec![]
        }
    }

    #[cfg(feature = "runtime-benchmarks")]
    impl frame_benchmarking::Benchmark<Block> for Runtime {
        fn benchmark_metadata(extra: bool) -> (
            Vec<frame_benchmarking::BenchmarkList>,
            Vec<frame_support::traits::StorageInfo>,
        ) {
            use frame_benchmarking::{baseline, Benchmarking, BenchmarkList};
            use frame_support::traits::StorageInfoTrait;
            use frame_system_benchmarking::Pallet as SystemBench;
            use baseline::Pallet as BaselineBench;

            let mut list = Vec::<BenchmarkList>::new();

            list_benchmarks!(list, extra);

            let storage_info = AllPalletsWithSystem::storage_info();

            (list, storage_info)
        }

        fn dispatch_benchmark(
            config: frame_benchmarking::BenchmarkConfig
        ) -> Result<Vec<frame_benchmarking::BenchmarkBatch>, alloc::string::String> {
            use frame_benchmarking::{baseline, Benchmarking, BenchmarkBatch};
            use sp_storage::TrackedStorageKey;
            use frame_system_benchmarking::Pallet as SystemBench;
            use frame_support::traits::WhitelistedStorageKeys;
            use baseline::Pallet as BaselineBench;

            let whitelist: Vec<TrackedStorageKey> = AllPalletsWithSystem::whitelisted_storage_keys();

            let mut batches = Vec::<BenchmarkBatch>::new();
            let params = (&config, &whitelist);

            add_benchmarks!(params, batches);

            if batches.is_empty() { return Err("Benchmark not found for this pallet.".into()) }
            Ok(batches)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Runtime, RuntimeBlockWeights as BlockWeights};
    use subspace_runtime_primitives::tests_utils::FeeMultiplierUtils;

    #[test]
    fn multiplier_can_grow_from_zero() {
        FeeMultiplierUtils::<Runtime, BlockWeights>::multiplier_can_grow_from_zero()
    }
}
