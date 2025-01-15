#![feature(variant_count)]
#![cfg_attr(not(feature = "std"), no_std)]
// `construct_runtime!` does a lot of recursion and requires us to increase the limit to 256.
#![recursion_limit = "256"]

mod precompiles;

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
    block_weights, maximum_block_length, maximum_domain_block_weight, opaque, Balance, BlockNumber,
    Hash, Nonce, EXISTENTIAL_DEPOSIT,
};
use domain_runtime_primitives::{
    CheckExtrinsicsValidityError, DecodeExtrinsicError, HoldIdentifier, ERR_BALANCE_OVERFLOW,
    ERR_NONCE_OVERFLOW, SLOT_DURATION,
};
use fp_account::EthereumSignature;
use fp_self_contained::{CheckedSignature, SelfContainedCall};
use frame_support::dispatch::{DispatchClass, DispatchInfo, GetDispatchInfo};
use frame_support::genesis_builder_helper::{build_state, get_preset};
use frame_support::inherent::ProvideInherent;
use frame_support::pallet_prelude::TypeInfo;
use frame_support::traits::fungible::Credit;
use frame_support::traits::{
    ConstU16, ConstU32, ConstU64, Currency, Everything, FindAuthor, Imbalance, OnFinalize,
    OnUnbalanced, VariantCount,
};
use frame_support::weights::constants::{ParityDbWeight, WEIGHT_REF_TIME_PER_SECOND};
use frame_support::weights::{ConstantMultiplier, Weight};
use frame_support::{construct_runtime, parameter_types};
use frame_system::limits::{BlockLength, BlockWeights};
use pallet_block_fees::fees::OnChargeDomainTransaction;
use pallet_ethereum::{
    PostLogContent, Transaction as EthereumTransaction, TransactionAction, TransactionData,
    TransactionStatus,
};
use pallet_evm::{
    Account as EVMAccount, EnsureAddressNever, EnsureAddressRoot, FeeCalculator,
    IdentityAddressMapping, Runner,
};
use pallet_transporter::EndpointHandler;
use sp_api::impl_runtime_apis;
use sp_core::crypto::KeyTypeId;
use sp_core::{Get, OpaqueMetadata, H160, H256, U256};
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
    BlakeTwo256, Block as BlockT, Checkable, DispatchInfoOf, Dispatchable, IdentifyAccount,
    IdentityLookup, Keccak256, NumberFor, One, PostDispatchInfoOf, SignedExtension,
    UniqueSaturatedInto, ValidateUnsigned, Verify, Zero,
};
use sp_runtime::transaction_validity::{
    InvalidTransaction, TransactionSource, TransactionValidity, TransactionValidityError,
};
use sp_runtime::{
    generic, impl_opaque_keys, ApplyExtrinsicResult, ConsensusEngineId, Digest,
    ExtrinsicInclusionMode,
};
pub use sp_runtime::{MultiAddress, Perbill, Permill};
use sp_std::cmp::{max, Ordering};
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

/// Alias to 512-bit hash when used in the context of a transaction signature on the chain.
pub type Signature = EthereumSignature;

/// Some way of identifying an account on the chain. We intentionally make it equivalent
/// to the public key of our transaction signing scheme.
pub type AccountId = <<Signature as Verify>::Signer as IdentifyAccount>::AccountId;

/// The address format for describing accounts.
pub type Address = AccountId;

/// Block type as expected by this runtime.
pub type Block = generic::Block<Header, UncheckedExtrinsic>;

/// A Block signed with a Justification
pub type SignedBlock = generic::SignedBlock<Block>;

/// BlockId type as expected by this runtime.
pub type BlockId = generic::BlockId<Block>;

/// Precompiles we use for EVM
pub type Precompiles = crate::precompiles::Precompiles<Runtime>;

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

/// Custom signed extra for check_and_pre_dispatch.
/// Only Nonce check is updated and rest remains same
type CustomSignedExtra = (
    frame_system::CheckNonZeroSender<Runtime>,
    frame_system::CheckSpecVersion<Runtime>,
    frame_system::CheckTxVersion<Runtime>,
    frame_system::CheckGenesis<Runtime>,
    frame_system::CheckMortality<Runtime>,
    pallet_evm_nonce_tracker::CheckNonce<Runtime>,
    domain_check_weight::CheckWeight<Runtime>,
    pallet_transaction_payment::ChargeTransactionPayment<Runtime>,
);

/// Unchecked extrinsic type as expected by this runtime.
pub type UncheckedExtrinsic =
    fp_self_contained::UncheckedExtrinsic<Address, RuntimeCall, Signature, SignedExtra>;

/// Extrinsic type that has already been checked.
pub type CheckedExtrinsic =
    fp_self_contained::CheckedExtrinsic<AccountId, RuntimeCall, SignedExtra, H160>;

/// Executive: handles dispatch to the various modules.
pub type Executive = domain_pallet_executive::Executive<
    Runtime,
    frame_system::ChainContext<Runtime>,
    Runtime,
    AllPalletsWithSystem,
>;

impl fp_self_contained::SelfContainedCall for RuntimeCall {
    type SignedInfo = H160;

    fn is_self_contained(&self) -> bool {
        match self {
            RuntimeCall::Ethereum(call) => call.is_self_contained(),
            _ => false,
        }
    }

    fn check_self_contained(&self) -> Option<Result<Self::SignedInfo, TransactionValidityError>> {
        match self {
            RuntimeCall::Ethereum(call) => call.check_self_contained(),
            _ => None,
        }
    }

    fn validate_self_contained(
        &self,
        info: &Self::SignedInfo,
        dispatch_info: &DispatchInfoOf<RuntimeCall>,
        len: usize,
    ) -> Option<TransactionValidity> {
        match self {
            RuntimeCall::Ethereum(call) => {
                // Ensure the caller can pay for the consensus chain storage fee
                let consensus_storage_fee =
                    BlockFees::consensus_chain_byte_fee().checked_mul(Balance::from(len as u32))?;
                let withdraw_res = <InnerEVMCurrencyAdapter as pallet_evm::OnChargeEVMTransaction<
                    Runtime,
                >>::withdraw_fee(info, consensus_storage_fee.into());
                if withdraw_res.is_err() {
                    return Some(Err(InvalidTransaction::Payment.into()));
                }

                call.validate_self_contained(info, dispatch_info, len)
            }
            _ => None,
        }
    }

    fn pre_dispatch_self_contained(
        &self,
        info: &Self::SignedInfo,
        dispatch_info: &DispatchInfoOf<RuntimeCall>,
        len: usize,
    ) -> Option<Result<(), TransactionValidityError>> {
        match self {
            RuntimeCall::Ethereum(call) => {
                // Withdraw the consensus chain storage fee from the caller and record
                // it in the `BlockFees`
                let consensus_storage_fee =
                    BlockFees::consensus_chain_byte_fee().checked_mul(Balance::from(len as u32))?;
                match <InnerEVMCurrencyAdapter as pallet_evm::OnChargeEVMTransaction<Runtime>>::withdraw_fee(
                    info,
                    consensus_storage_fee.into(),
                ) {
                    Ok(None) => {}
                    Ok(Some(paid_consensus_storage_fee)) => {
                        BlockFees::note_consensus_storage_fee(paid_consensus_storage_fee.peek())
                    }
                    Err(_) => return Some(Err(InvalidTransaction::Payment.into())),
                }

                // Copied from [`pallet_ethereum::Call::pre_dispatch_self_contained`] with `frame_system::CheckWeight`
                // replaced with `domain_check_weight::CheckWeight`
                if let pallet_ethereum::Call::transact { transaction } = call {
                    if let Err(e) = domain_check_weight::CheckWeight::<Runtime>::do_pre_dispatch(
                        dispatch_info,
                        len,
                    ) {
                        return Some(Err(e));
                    }

                    Some(Ethereum::validate_transaction_in_block(*info, transaction))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn apply_self_contained(
        self,
        info: Self::SignedInfo,
    ) -> Option<sp_runtime::DispatchResultWithInfo<PostDispatchInfoOf<Self>>> {
        match self {
            call @ RuntimeCall::Ethereum(pallet_ethereum::Call::transact { .. }) => {
                Some(call.dispatch(RuntimeOrigin::from(
                    pallet_ethereum::RawOrigin::EthereumTransaction(info),
                )))
            }
            _ => None,
        }
    }
}

impl_opaque_keys! {
    pub struct SessionKeys {
        /// Primarily used for adding the operator signing key into the Keystore.
        pub operator: sp_domains::OperatorKey,
    }
}

#[sp_version::runtime_version]
pub const VERSION: RuntimeVersion = RuntimeVersion {
    spec_name: Cow::Borrowed("subspace-evm-domain"),
    impl_name: Cow::Borrowed("subspace-evm-domain"),
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

    // This part is copied from Substrate's `bin/node/runtime/src/lib.rs`.
    //  The `RuntimeBlockLength` and `RuntimeBlockWeights` exist here because the
    // `DeletionWeightLimit` and `DeletionQueueDepth` depend on those to parameterize
    // the lazy contract deletion.
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
    type Lookup = IdentityLookup<AccountId>;
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

type NegativeImbalance = <Balances as Currency<AccountId>>::NegativeImbalance;

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
    type AccountIdConverter = domain_runtime_primitives::AccountId20Converter;
    type WeightInfo = pallet_transporter::weights::SubstrateWeight<Runtime>;
}

impl pallet_evm_chain_id::Config for Runtime {}

pub struct FindAuthorTruncated;

impl FindAuthor<H160> for FindAuthorTruncated {
    fn find_author<'a, I>(_digests: I) -> Option<H160>
    where
        I: 'a + IntoIterator<Item = (ConsensusEngineId, &'a [u8])>,
    {
        // TODO: returns the executor reward address once we start collecting them
        None
    }
}

/// Current approximation of the gas/s consumption considering
/// EVM execution over compiled WASM (on 4.4Ghz CPU).
pub const GAS_PER_SECOND: u64 = 40_000_000;

/// Approximate ratio of the amount of Weight per Gas.
/// u64 works for approximations because Weight is a very small unit compared to gas.
pub const WEIGHT_PER_GAS: u64 = WEIGHT_REF_TIME_PER_SECOND.saturating_div(GAS_PER_SECOND);

parameter_types! {
    /// EVM block gas limit is set to maximum to allow all the transaction stored on Consensus chain.
    pub BlockGasLimit: U256 = U256::from(
        maximum_domain_block_weight().ref_time() / WEIGHT_PER_GAS
    );
    pub PrecompilesValue: Precompiles = Precompiles::default();
    pub WeightPerGas: Weight = Weight::from_parts(WEIGHT_PER_GAS, 0);
}

type InnerEVMCurrencyAdapter = pallet_evm::EVMCurrencyAdapter<Balances, ()>;

// Implementation of [`pallet_transaction_payment::OnChargeTransaction`] that charges evm transaction
// fees from the transaction sender and collect all the fees (including both the base fee and tip) in
// `pallet_block_fees`
pub struct EVMCurrencyAdapter;

impl pallet_evm::OnChargeEVMTransaction<Runtime> for EVMCurrencyAdapter {
    type LiquidityInfo = Option<NegativeImbalance>;

    fn withdraw_fee(
        who: &H160,
        fee: U256,
    ) -> Result<Self::LiquidityInfo, pallet_evm::Error<Runtime>> {
        InnerEVMCurrencyAdapter::withdraw_fee(who, fee)
    }

    fn correct_and_deposit_fee(
        who: &H160,
        corrected_fee: U256,
        base_fee: U256,
        already_withdrawn: Self::LiquidityInfo,
    ) -> Self::LiquidityInfo {
        if already_withdrawn.is_some() {
            // Record the evm actual transaction fee
            BlockFees::note_domain_execution_fee(corrected_fee.as_u128());
        }

        <InnerEVMCurrencyAdapter as pallet_evm::OnChargeEVMTransaction<
            Runtime,
        >>::correct_and_deposit_fee(who, corrected_fee, base_fee, already_withdrawn)
    }

    fn pay_priority_fee(tip: Self::LiquidityInfo) {
        <InnerEVMCurrencyAdapter as pallet_evm::OnChargeEVMTransaction<Runtime>>::pay_priority_fee(
            tip,
        );
    }
}

parameter_types! {
    pub const GasLimitPovSizeRatio: u64 = 4;
    pub const SuicideQuickClearLimit: u32 = 0;
}

impl pallet_evm::Config for Runtime {
    type FeeCalculator = BaseFee;
    type GasWeightMapping = pallet_evm::FixedGasWeightMapping<Self>;
    type WeightPerGas = WeightPerGas;
    type BlockHashMapping = pallet_ethereum::EthereumBlockHashMapping<Self>;
    type CallOrigin = EnsureAddressRoot<AccountId>;
    type WithdrawOrigin = EnsureAddressNever<AccountId>;
    type AddressMapping = IdentityAddressMapping;
    type Currency = Balances;
    type RuntimeEvent = RuntimeEvent;
    type PrecompilesType = Precompiles;
    type PrecompilesValue = PrecompilesValue;
    type ChainId = EVMChainId;
    type BlockGasLimit = BlockGasLimit;
    type Runner = pallet_evm::runner::stack::Runner<Self>;
    type OnChargeTransaction = EVMCurrencyAdapter;
    type OnCreate = ();
    type FindAuthor = FindAuthorTruncated;
    type GasLimitPovSizeRatio = GasLimitPovSizeRatio;
    type SuicideQuickClearLimit = SuicideQuickClearLimit;
    type Timestamp = Timestamp;
    type WeightInfo = pallet_evm::weights::SubstrateWeight<Self>;
}

impl pallet_evm_nonce_tracker::Config for Runtime {}

parameter_types! {
    pub const PostOnlyBlockHash: PostLogContent = PostLogContent::OnlyBlockHash;
}

impl pallet_ethereum::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type StateRoot = pallet_ethereum::IntermediateStateRoot<Self>;
    type PostLogContent = PostOnlyBlockHash;
    type ExtraDataLength = ConstU32<30>;
}

parameter_types! {
    pub BoundDivision: U256 = U256::from(1024);
}

parameter_types! {
    pub DefaultBaseFeePerGas: U256 = U256::from(1_000_000_000);
    // mark it to 5% increments on beyond target weight.
    pub DefaultElasticity: Permill = Permill::from_parts(50_000);
}

pub struct BaseFeeThreshold;

impl pallet_base_fee::BaseFeeThreshold for BaseFeeThreshold {
    fn lower() -> Permill {
        Permill::zero()
    }
    fn ideal() -> Permill {
        Permill::from_parts(500_000)
    }
    fn upper() -> Permill {
        Permill::from_parts(1_000_000)
    }
}

impl pallet_base_fee::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type Threshold = BaseFeeThreshold;
    type DefaultBaseFeePerGas = DefaultBaseFeePerGas;
    type DefaultElasticity = DefaultElasticity;
}

impl pallet_domain_id::Config for Runtime {}

pub struct IntoRuntimeCall;

impl sp_domain_sudo::IntoRuntimeCall<RuntimeCall> for IntoRuntimeCall {
    fn runtime_call(call: Vec<u8>) -> RuntimeCall {
        UncheckedExtrinsic::decode(&mut call.as_slice())
            .expect("must always be a valid domain extrinsic as checked by consensus chain; qed")
            .0
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
        //  so that consensus can constructed encoded extrinsic that matches with Domain encoded
        //  extrinsic.
        Timestamp: pallet_timestamp = 1,
        ExecutivePallet: domain_pallet_executive = 2,
        Utility: pallet_utility = 8,

        // monetary stuff
        Balances: pallet_balances = 20,
        TransactionPayment: pallet_transaction_payment = 21,

        // messenger stuff
        // Note: Indexes should match with indexes on other chains and domains
        Messenger: pallet_messenger = 60,
        Transporter: pallet_transporter = 61,

        // evm stuff
        Ethereum: pallet_ethereum = 80,
        EVM: pallet_evm = 81,
        EVMChainId: pallet_evm_chain_id = 82,
        BaseFee: pallet_base_fee = 83,
        EVMNoncetracker: pallet_evm_nonce_tracker = 84,

        // domain instance stuff
        SelfDomainId: pallet_domain_id = 90,
        BlockFees: pallet_block_fees = 91,

        // Sudo account
        Sudo: pallet_domain_sudo = 100,
    }
);

#[derive(Clone, Default)]
pub struct TransactionConverter;

impl fp_rpc::ConvertTransaction<UncheckedExtrinsic> for TransactionConverter {
    fn convert_transaction(&self, transaction: pallet_ethereum::Transaction) -> UncheckedExtrinsic {
        UncheckedExtrinsic::new_unsigned(
            pallet_ethereum::Call::<Runtime>::transact { transaction }.into(),
        )
    }
}

impl fp_rpc::ConvertTransaction<opaque::UncheckedExtrinsic> for TransactionConverter {
    fn convert_transaction(
        &self,
        transaction: pallet_ethereum::Transaction,
    ) -> opaque::UncheckedExtrinsic {
        let extrinsic = UncheckedExtrinsic::new_unsigned(
            pallet_ethereum::Call::<Runtime>::transact { transaction }.into(),
        );
        let encoded = extrinsic.encode();
        opaque::UncheckedExtrinsic::decode(&mut &encoded[..])
            .expect("Encoded extrinsic is always valid")
    }
}

fn is_xdm_mmr_proof_valid(ext: &<Block as BlockT>::Extrinsic) -> Option<bool> {
    match &ext.0.function {
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
            call: Box::new(ext.0.function),
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
    if ext.0.function.is_self_contained() {
        ext.0
            .function
            .check_self_contained()
            .map(|signed_info| signed_info.map(|signer| signer.into()))
    } else {
        ext.0
            .signature
            .as_ref()
            .map(|(signed, _, _)| lookup.lookup(*signed).map_err(|e| e.into()))
    }
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
    extrinsic
        .0
        .signature
        .as_ref()
        .map(|(_, _, extra)| extra.4 .0)
}

#[cfg(feature = "runtime-benchmarks")]
mod benches {
    frame_benchmarking::define_benchmarks!(
        [frame_benchmarking, BaselineBench::<Runtime>]
        [frame_system, SystemBench::<Runtime>]
        [domain_pallet_executive, ExecutivePallet]
        [pallet_messenger, Messenger]
    );
}

/// Custom pre_dispatch for extrinsic verification.
/// Most of the logic is same as `pre_dispatch_self_contained` except
/// - we use `validate_self_contained` instead `pre_dispatch_self_contained`
///   since the nonce is not incremented in `pre_dispatch_self_contained`
/// - Manually track the account nonce to check either Stale or Future nonce.
fn pre_dispatch_evm_transaction(
    account_id: H160,
    call: RuntimeCall,
    dispatch_info: &DispatchInfoOf<RuntimeCall>,
    len: usize,
) -> Result<(), TransactionValidityError> {
    match call {
        RuntimeCall::Ethereum(call) => {
            // Withdraw the consensus chain storage fee from the caller and record
            // it in the `BlockFees`
            let consensus_storage_fee = BlockFees::consensus_chain_byte_fee()
                .checked_mul(Balance::from(len as u32))
                .ok_or(InvalidTransaction::Custom(ERR_BALANCE_OVERFLOW))?;
            match <InnerEVMCurrencyAdapter as pallet_evm::OnChargeEVMTransaction<Runtime>>::withdraw_fee(
                &account_id,
                consensus_storage_fee.into(),
            ) {
                Ok(None) => {}
                Ok(Some(paid_consensus_storage_fee)) => {
                    BlockFees::note_consensus_storage_fee(paid_consensus_storage_fee.peek())
                }
                Err(_) => return Err(InvalidTransaction::Payment.into()),
            }

            if let Some(transaction_validity) =
                call.validate_self_contained(&account_id, dispatch_info, len)
            {
                let _ = transaction_validity?;

                let pallet_ethereum::Call::transact { transaction } = call;
                domain_check_weight::CheckWeight::<Runtime>::do_pre_dispatch(dispatch_info, len)?;

                let transaction_data: TransactionData = (&transaction).into();
                let transaction_nonce = transaction_data.nonce;
                // If the current account nonce is greater than the tracked nonce, then
                // pick the highest nonce
                let account_nonce = {
                    let tracked_nonce = EVMNoncetracker::account_nonce(AccountId::from(account_id))
                        .unwrap_or(U256::zero());
                    let account_nonce = EVM::account_basic(&account_id).0.nonce;
                    max(tracked_nonce, account_nonce)
                };

                match transaction_nonce.cmp(&account_nonce) {
                    Ordering::Less => return Err(InvalidTransaction::Stale.into()),
                    Ordering::Greater => return Err(InvalidTransaction::Future.into()),
                    Ordering::Equal => {}
                }

                let next_nonce = account_nonce
                    .checked_add(U256::one())
                    .ok_or(InvalidTransaction::Custom(ERR_NONCE_OVERFLOW))?;

                EVMNoncetracker::set_account_nonce(AccountId::from(account_id), next_nonce);
            }

            Ok(())
        }
        _ => Err(InvalidTransaction::Call.into()),
    }
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
        CheckedSignature::Signed(account_id, extra) => {
            let custom_extra: CustomSignedExtra = (
                extra.0,
                extra.1,
                extra.2,
                extra.3,
                extra.4,
                pallet_evm_nonce_tracker::CheckNonce::from(extra.5 .0),
                extra.6,
                extra.7,
            );

            custom_extra
                .pre_dispatch(&account_id, &xt.function, &dispatch_info, encoded_len)
                .map(|_| ())
        }
        CheckedSignature::Unsigned => {
            if let RuntimeCall::Messenger(call) = &xt.function {
                Messenger::pre_dispatch_with_trusted_mmr_proof(call)?;
            } else {
                Runtime::pre_dispatch(&xt.function).map(|_| ())?;
            }
            SignedExtra::pre_dispatch_unsigned(&xt.function, &dispatch_info, encoded_len)
                .map(|_| ())
        }
        CheckedSignature::SelfContained(account_id) => {
            pre_dispatch_evm_transaction(account_id, xt.function, &dispatch_info, encoded_len)
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

        fn intermediate_roots() -> Vec<[u8; 32]> {
            ExecutivePallet::intermediate_roots()
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
            match &extrinsic.0.function {
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
            extrinsic: &<Block as BlockT>::Extrinsic
        ) -> Option<bool> {
            is_xdm_mmr_proof_valid(extrinsic)
        }

        fn extract_xdm_mmr_proof(ext: &<Block as BlockT>::Extrinsic) -> Option<ConsensusChainMmrLeafProof<ConsensusBlockNumber, ConsensusBlockHash, sp_core::H256>> {
            match &ext.0.function {
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
            match &ext.0.function {
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

    impl fp_rpc::EthereumRuntimeRPCApi<Block> for Runtime {
        fn chain_id() -> u64 {
            <Runtime as pallet_evm::Config>::ChainId::get()
        }

        fn account_basic(address: H160) -> EVMAccount {
            let (account, _) = EVM::account_basic(&address);
            account
        }

        fn gas_price() -> U256 {
            let (gas_price, _) = <Runtime as pallet_evm::Config>::FeeCalculator::min_gas_price();
            gas_price
        }

        fn account_code_at(address: H160) -> Vec<u8> {
            pallet_evm::AccountCodes::<Runtime>::get(address)
        }

        fn author() -> H160 {
            <pallet_evm::Pallet<Runtime>>::find_author()
        }

        fn storage_at(address: H160, index: U256) -> H256 {
            let mut tmp = [0u8; 32];
            index.to_big_endian(&mut tmp);
            pallet_evm::AccountStorages::<Runtime>::get(address, H256::from_slice(&tmp[..]))
        }

        fn call(
            from: H160,
            to: H160,
            data: Vec<u8>,
            value: U256,
            gas_limit: U256,
            max_fee_per_gas: Option<U256>,
            max_priority_fee_per_gas: Option<U256>,
            nonce: Option<U256>,
            estimate: bool,
            access_list: Option<Vec<(H160, Vec<H256>)>>,
        ) -> Result<pallet_evm::CallInfo, sp_runtime::DispatchError> {
            let config = if estimate {
                let mut config = <Runtime as pallet_evm::Config>::config().clone();
                config.estimate = true;
                Some(config)
            } else {
                None
            };

            let is_transactional = false;
            let validate = true;
            let evm_config = config.as_ref().unwrap_or(<Runtime as pallet_evm::Config>::config());

            let gas_limit = gas_limit.min(u64::MAX.into());

            let transaction_data = TransactionData::new(
                pallet_ethereum::TransactionAction::Call(to),
                data.clone(),
                nonce.unwrap_or_default(),
                gas_limit,
                None,
                max_fee_per_gas,
                max_priority_fee_per_gas,
                value,
                Some(<Runtime as pallet_evm::Config>::ChainId::get()),
                access_list.clone().unwrap_or_default(),
            );

            let (weight_limit, proof_size_base_cost) = pallet_ethereum::Pallet::<Runtime>::transaction_weight(&transaction_data);

            <Runtime as pallet_evm::Config>::Runner::call(
                from,
                to,
                data,
                value,
                gas_limit.unique_saturated_into(),
                max_fee_per_gas,
                max_priority_fee_per_gas,
                nonce,
                access_list.unwrap_or_default(),
                is_transactional,
                validate,
                weight_limit,
                proof_size_base_cost,
                evm_config,
            ).map_err(|err| err.error.into())
        }

        fn create(
            from: H160,
            data: Vec<u8>,
            value: U256,
            gas_limit: U256,
            max_fee_per_gas: Option<U256>,
            max_priority_fee_per_gas: Option<U256>,
            nonce: Option<U256>,
            estimate: bool,
            access_list: Option<Vec<(H160, Vec<H256>)>>,
        ) -> Result<pallet_evm::CreateInfo, sp_runtime::DispatchError> {
            let config = if estimate {
                let mut config = <Runtime as pallet_evm::Config>::config().clone();
                config.estimate = true;
                Some(config)
            } else {
                None
            };

            let is_transactional = false;
            let validate = true;
            let weight_limit = None;
            let proof_size_base_cost = None;
            let evm_config = config.as_ref().unwrap_or(<Runtime as pallet_evm::Config>::config());
            <Runtime as pallet_evm::Config>::Runner::create(
                from,
                data,
                value,
                gas_limit.unique_saturated_into(),
                max_fee_per_gas,
                max_priority_fee_per_gas,
                nonce,
                access_list.unwrap_or_default(),
                is_transactional,
                validate,
                weight_limit,
                proof_size_base_cost,
                evm_config,
            ).map_err(|err| err.error.into())
        }

        fn current_transaction_statuses() -> Option<Vec<TransactionStatus>> {
            pallet_ethereum::CurrentTransactionStatuses::<Runtime>::get()
        }

        fn current_block() -> Option<pallet_ethereum::Block> {
            pallet_ethereum::CurrentBlock::<Runtime>::get()
        }

        fn current_receipts() -> Option<Vec<pallet_ethereum::Receipt>> {
            pallet_ethereum::CurrentReceipts::<Runtime>::get()
        }

        fn current_all() -> (
            Option<pallet_ethereum::Block>,
            Option<Vec<pallet_ethereum::Receipt>>,
            Option<Vec<TransactionStatus>>
        ) {
            (
                pallet_ethereum::CurrentBlock::<Runtime>::get(),
                pallet_ethereum::CurrentReceipts::<Runtime>::get(),
                pallet_ethereum::CurrentTransactionStatuses::<Runtime>::get()
            )
        }

        fn extrinsic_filter(
            xts: Vec<<Block as BlockT>::Extrinsic>,
        ) -> Vec<EthereumTransaction> {
            xts.into_iter().filter_map(|xt| match xt.0.function {
                RuntimeCall::Ethereum(pallet_ethereum::Call::transact {  transaction }) => Some(transaction),
                _ => None
            }).collect::<Vec<EthereumTransaction>>()
        }

        fn elasticity() -> Option<Permill> {
            Some(pallet_base_fee::Elasticity::<Runtime>::get())
        }

        fn gas_limit_multiplier_support() {}

        fn pending_block(
            xts: Vec<<Block as BlockT>::Extrinsic>,
        ) -> (Option<pallet_ethereum::Block>, Option<Vec<TransactionStatus>>) {
            for ext in xts.into_iter() {
                let _ = Executive::apply_extrinsic(ext);
            }

            Ethereum::on_finalize(System::block_number() + 1);

            (
                pallet_ethereum::CurrentBlock::<Runtime>::get(),
                pallet_ethereum::CurrentTransactionStatuses::<Runtime>::get()
            )
        }

        fn initialize_pending_block(header: &<Block as BlockT>::Header) {
            Executive::initialize_block(header);
        }
    }

    impl fp_rpc::ConvertTransactionRuntimeApi<Block> for Runtime {
        fn convert_transaction(transaction: EthereumTransaction) -> <Block as BlockT>::Extrinsic {
            UncheckedExtrinsic::new_unsigned(
                pallet_ethereum::Call::transact { transaction }.into(),
            )
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
