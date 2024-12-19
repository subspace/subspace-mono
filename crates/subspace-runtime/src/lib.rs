// Copyright (C) 2021 Subspace Labs, Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

#![cfg_attr(not(feature = "std"), no_std)]
#![feature(const_trait_impl, variant_count)]
// `construct_runtime!` does a lot of recursion and requires us to increase the limit to 256.
#![recursion_limit = "256"]
// TODO: remove when upstream issue is fixed
#![allow(
    non_camel_case_types,
    reason = "https://github.com/rust-lang/rust-analyzer/issues/16514"
)]

mod domains;
mod fees;
mod object_mapping;
mod signed_extensions;

extern crate alloc;

// Make the WASM binary available.
#[cfg(feature = "std")]
include!(concat!(env!("OUT_DIR"), "/wasm_binary.rs"));

use crate::fees::{OnChargeTransaction, TransactionByteFee};
use crate::object_mapping::extract_block_object_mapping;
pub use crate::signed_extensions::{CheckHistorySeeder, DisablePallets};
use alloc::borrow::Cow;
use codec::{Decode, Encode, MaxEncodedLen};
use core::mem;
use core::num::NonZeroU64;
use domain_runtime_primitives::opaque::Header as DomainHeader;
use domain_runtime_primitives::{
    maximum_domain_block_weight, AccountIdConverter, BlockNumber as DomainNumber,
    Hash as DomainHash,
};
use frame_support::genesis_builder_helper::{build_state, get_preset};
use frame_support::inherent::ProvideInherent;
use frame_support::traits::fungible::HoldConsideration;
use frame_support::traits::{
    ConstU16, ConstU32, ConstU64, ConstU8, Currency, EitherOfDiverse, EqualPrivilegeOnly,
    Everything, Get, LinearStoragePrice, OnUnbalanced, Time, VariantCount,
};
use frame_support::weights::constants::ParityDbWeight;
use frame_support::weights::{ConstantMultiplier, Weight};
use frame_support::{construct_runtime, parameter_types, PalletId};
use frame_system::limits::{BlockLength, BlockWeights};
use frame_system::EnsureRoot;
use pallet_collective::{EnsureMember, EnsureProportionAtLeast};
pub use pallet_rewards::RewardPoint;
pub use pallet_subspace::{AllowAuthoringBy, EnableRewardsAt};
use pallet_transporter::EndpointHandler;
use scale_info::TypeInfo;
use sp_api::impl_runtime_apis;
use sp_consensus_slots::{Slot, SlotDuration};
use sp_consensus_subspace::{ChainConstants, PotParameters, SignedVote, SolutionRanges, Vote};
use sp_core::crypto::KeyTypeId;
use sp_core::{ConstBool, OpaqueMetadata, H256};
use sp_domains::bundle_producer_election::BundleProducerElectionParams;
use sp_domains::{
    ChannelId, DomainAllowlistUpdates, DomainId, DomainInstanceData, ExecutionReceiptFor,
    OperatorId, OperatorPublicKey, OperatorRewardSource, DOMAIN_STORAGE_FEE_MULTIPLIER,
    INITIAL_DOMAIN_TX_RANGE,
};
use sp_domains_fraud_proof::fraud_proof::FraudProof;
use sp_domains_fraud_proof::storage_proof::{
    FraudProofStorageKeyProvider, FraudProofStorageKeyRequest,
};
use sp_messenger::endpoint::{Endpoint, EndpointHandler as EndpointHandlerT, EndpointId};
use sp_messenger::messages::{
    BlockMessagesWithStorageKey, ChainId, CrossDomainMessage, FeeModel, MessageId, MessageKey,
};
use sp_messenger::{ChannelNonce, XdmId};
use sp_messenger_host_functions::{get_storage_key, StorageKeyRequest};
use sp_mmr_primitives::EncodableOpaqueLeaf;
use sp_runtime::traits::{
    AccountIdConversion, AccountIdLookup, BlakeTwo256, Block as BlockT, ConstU128, Keccak256,
    NumberFor,
};
use sp_runtime::transaction_validity::{TransactionSource, TransactionValidity};
use sp_runtime::{generic, AccountId32, ApplyExtrinsicResult, ExtrinsicInclusionMode, Perbill};
use sp_std::collections::btree_map::BTreeMap;
use sp_std::collections::btree_set::BTreeSet;
use sp_std::marker::PhantomData;
use sp_std::prelude::*;
use sp_subspace_mmr::subspace_mmr_runtime_interface::consensus_block_hash;
use sp_subspace_mmr::ConsensusChainMmrLeafProof;
use sp_version::RuntimeVersion;
use static_assertions::const_assert;
use subspace_core_primitives::objects::BlockObjectMapping;
use subspace_core_primitives::pieces::Piece;
use subspace_core_primitives::segments::{
    HistorySize, SegmentCommitment, SegmentHeader, SegmentIndex,
};
use subspace_core_primitives::solutions::{
    pieces_to_solution_range, solution_range_to_pieces, SolutionRange,
};
use subspace_core_primitives::{PublicKey, Randomness, SlotNumber, U256};
use subspace_runtime_primitives::{
    maximum_normal_block_length, AccountId, Balance, BlockNumber, FindBlockRewardAddress, Hash,
    HoldIdentifier, Moment, Nonce, Signature, SlowAdjustingFeeUpdate, BLOCK_WEIGHT_FOR_2_SEC,
    MAX_BLOCK_LENGTH, MIN_REPLICATION_FACTOR, NORMAL_DISPATCH_RATIO, SHANNON, SLOT_PROBABILITY,
    SSC,
};

sp_runtime::impl_opaque_keys! {
    pub struct SessionKeys {
    }
}

/// How many pieces one sector is supposed to contain (max)
const MAX_PIECES_IN_SECTOR: u16 = 1000;

// To learn more about runtime versioning and what each of the following value means:
//   https://substrate.dev/docs/en/knowledgebase/runtime/upgrades#runtime-versioning
#[sp_version::runtime_version]
pub const VERSION: RuntimeVersion = RuntimeVersion {
    spec_name: Cow::Borrowed("subspace"),
    impl_name: Cow::Borrowed("subspace"),
    authoring_version: 0,
    spec_version: 1,
    impl_version: 0,
    apis: RUNTIME_API_VERSIONS,
    transaction_version: 0,
    system_version: 2,
};

// TODO: Many of below constants should probably be updatable but currently they are not

// NOTE: Currently it is not possible to change the slot duration after the chain has started.
//       Attempting to do so will brick block production.
const SLOT_DURATION: u64 = 1000;

/// Number of slots between slot arrival and when corresponding block can be produced.
const BLOCK_AUTHORING_DELAY: SlotNumber = 4;

/// Interval, in blocks, between blockchain entropy injection into proof of time chain.
const POT_ENTROPY_INJECTION_INTERVAL: BlockNumber = 50;

/// Interval, in entropy injection intervals, where to take entropy for injection from.
const POT_ENTROPY_INJECTION_LOOKBACK_DEPTH: u8 = 2;

/// Delay after block, in slots, when entropy injection takes effect.
const POT_ENTROPY_INJECTION_DELAY: SlotNumber = 15;

// Entropy injection interval must be bigger than injection delay or else we may end up in a
// situation where we'll need to do more than one injection at the same slot
const_assert!(POT_ENTROPY_INJECTION_INTERVAL as u64 > POT_ENTROPY_INJECTION_DELAY);
// Entropy injection delay must be bigger than block authoring delay or else we may include
// invalid future proofs in parent block, +1 ensures we do not have unnecessary reorgs that will
// inevitably happen otherwise
const_assert!(POT_ENTROPY_INJECTION_DELAY > BLOCK_AUTHORING_DELAY + 1);

/// Era duration in blocks.
const ERA_DURATION_IN_BLOCKS: BlockNumber = 2016;

/// Tx range is adjusted every DOMAIN_TX_RANGE_ADJUSTMENT_INTERVAL blocks.
const TX_RANGE_ADJUSTMENT_INTERVAL_BLOCKS: u64 = 100;

// We assume initial plot size starts with a single sector.
const INITIAL_SOLUTION_RANGE: SolutionRange =
    pieces_to_solution_range(MAX_PIECES_IN_SECTOR as u64, SLOT_PROBABILITY);

/// Number of votes expected per block.
///
/// This impacts solution range for votes in consensus.
const EXPECTED_VOTES_PER_BLOCK: u32 = 9;

/// Number of latest archived segments that are considered "recent history".
const RECENT_SEGMENTS: HistorySize = HistorySize::new(NonZeroU64::new(5).expect("Not zero; qed"));
/// Fraction of pieces from the "recent history" (`recent_segments`) in each sector.
const RECENT_HISTORY_FRACTION: (HistorySize, HistorySize) = (
    HistorySize::new(NonZeroU64::new(1).expect("Not zero; qed")),
    HistorySize::new(NonZeroU64::new(10).expect("Not zero; qed")),
);
/// Minimum lifetime of a plotted sector, measured in archived segment.
const MIN_SECTOR_LIFETIME: HistorySize =
    HistorySize::new(NonZeroU64::new(4).expect("Not zero; qed"));

parameter_types! {
    pub const Version: RuntimeVersion = VERSION;
    pub const BlockHashCount: BlockNumber = 250;
    /// We allow for 2 seconds of compute with a 6 second average block time.
    pub SubspaceBlockWeights: BlockWeights = BlockWeights::with_sensible_defaults(BLOCK_WEIGHT_FOR_2_SEC, NORMAL_DISPATCH_RATIO);
    /// We allow for 3.75 MiB for `Normal` extrinsic with 5 MiB maximum block length.
    pub SubspaceBlockLength: BlockLength = maximum_normal_block_length();
}

pub type SS58Prefix = ConstU16<6094>;

// Configure FRAME pallets to include in runtime.

impl frame_system::Config for Runtime {
    /// The basic call filter to use in dispatchable.
    ///
    /// `Everything` is used here as we use the signed extension
    /// `DisablePallets` as the actual call filter.
    type BaseCallFilter = Everything;
    /// Block & extrinsics weights: base values and limits.
    type BlockWeights = SubspaceBlockWeights;
    /// The maximum length of a block (in bytes).
    type BlockLength = SubspaceBlockLength;
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
    /// The weight of database operations that the runtime can invoke.
    type DbWeight = ParityDbWeight;
    /// Version of the runtime.
    type Version = Version;
    /// Converts a module to the index of the module in `construct_runtime!`.
    ///
    /// This type is being generated by `construct_runtime!`.
    type PalletInfo = PalletInfo;
    /// What to do if a new account is created.
    type OnNewAccount = ();
    /// What to do if an account is fully reaped from the system.
    type OnKilledAccount = ();
    /// The data to be stored in an account.
    type AccountData = pallet_balances::AccountData<Balance>;
    /// Weight information for the extrinsics of this pallet.
    type SystemWeightInfo = ();
    /// This is used as an identifier of the chain.
    type SS58Prefix = SS58Prefix;
    /// The set code logic, just the default since we're not a parachain.
    type OnSetCode = ();
    type SingleBlockMigrations = ();
    type MultiBlockMigrator = ();
    type PreInherents = ();
    type PostInherents = ();
    type PostTransactions = ();
    type MaxConsumers = ConstU32<16>;
}

parameter_types! {
    pub const BlockAuthoringDelay: SlotNumber = BLOCK_AUTHORING_DELAY;
    pub const PotEntropyInjectionInterval: BlockNumber = POT_ENTROPY_INJECTION_INTERVAL;
    pub const PotEntropyInjectionLookbackDepth: u8 = POT_ENTROPY_INJECTION_LOOKBACK_DEPTH;
    pub const PotEntropyInjectionDelay: SlotNumber = POT_ENTROPY_INJECTION_DELAY;
    pub const EraDuration: u32 = ERA_DURATION_IN_BLOCKS;
    pub const SlotProbability: (u64, u64) = SLOT_PROBABILITY;
    pub const ExpectedVotesPerBlock: u32 = EXPECTED_VOTES_PER_BLOCK;
    pub const RecentSegments: HistorySize = RECENT_SEGMENTS;
    pub const RecentHistoryFraction: (HistorySize, HistorySize) = RECENT_HISTORY_FRACTION;
    pub const MinSectorLifetime: HistorySize = MIN_SECTOR_LIFETIME;
    // Disable solution range adjustment at the start of chain.
    // Root origin must enable later
    pub const ShouldAdjustSolutionRange: bool = false;
    pub const BlockSlotCount: u32 = 6;
}

pub struct ConfirmationDepthK;

impl Get<BlockNumber> for ConfirmationDepthK {
    fn get() -> BlockNumber {
        pallet_runtime_configs::ConfirmationDepthK::<Runtime>::get()
    }
}

impl pallet_subspace::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type BlockAuthoringDelay = BlockAuthoringDelay;
    type PotEntropyInjectionInterval = PotEntropyInjectionInterval;
    type PotEntropyInjectionLookbackDepth = PotEntropyInjectionLookbackDepth;
    type PotEntropyInjectionDelay = PotEntropyInjectionDelay;
    type EraDuration = EraDuration;
    type InitialSolutionRange = ConstU64<INITIAL_SOLUTION_RANGE>;
    type SlotProbability = SlotProbability;
    type ConfirmationDepthK = ConfirmationDepthK;
    type RecentSegments = RecentSegments;
    type RecentHistoryFraction = RecentHistoryFraction;
    type MinSectorLifetime = MinSectorLifetime;
    type ExpectedVotesPerBlock = ExpectedVotesPerBlock;
    type MaxPiecesInSector = ConstU16<{ MAX_PIECES_IN_SECTOR }>;
    type ShouldAdjustSolutionRange = ShouldAdjustSolutionRange;
    type EraChangeTrigger = pallet_subspace::NormalEraChange;
    type BlockSlotCount = BlockSlotCount;
    type WeightInfo = pallet_subspace::weights::SubstrateWeight<Runtime>;
}

impl pallet_timestamp::Config for Runtime {
    /// A timestamp: milliseconds since the unix epoch.
    type Moment = Moment;
    type OnTimestampSet = ();
    type MinimumPeriod = ConstU64<{ SLOT_DURATION / 2 }>;
    type WeightInfo = ();
}

parameter_types! {
    // Computed as ED = Account data size * Price per byte, where
    // Price per byte = Min Number of validators * Storage duration (years) * Storage cost per year
    // Account data size (80 bytes)
    // Min Number of redundant validators (100) - For a stable and redundant blockchain we need at least a certain number of full nodes/collators.
    // Storage duration (1 year) - It is theoretically unlimited, accounts will stay around while the chain is alive.
    // Storage cost per year of (12 * 1e-9 * 0.1 ) - SSD storage on cloud hosting costs about 0.1 USD per Gb per month
    pub const ExistentialDeposit: Balance = 10_000_000_000_000 * SHANNON;
}

#[derive(
    PartialEq, Eq, Clone, Encode, Decode, TypeInfo, MaxEncodedLen, Ord, PartialOrd, Copy, Debug,
)]
pub struct HoldIdentifierWrapper(HoldIdentifier);

impl pallet_domains::HoldIdentifier<Runtime> for HoldIdentifierWrapper {
    fn staking_staked() -> Self {
        Self(HoldIdentifier::DomainStaking)
    }

    fn domain_instantiation_id() -> Self {
        Self(HoldIdentifier::DomainInstantiation)
    }

    fn storage_fund_withdrawal() -> Self {
        Self(HoldIdentifier::DomainStorageFund)
    }
}

impl pallet_messenger::HoldIdentifier<Runtime> for HoldIdentifierWrapper {
    fn messenger_channel() -> Self {
        Self(HoldIdentifier::MessengerChannel)
    }
}

impl VariantCount for HoldIdentifierWrapper {
    const VARIANT_COUNT: u32 = mem::variant_count::<HoldIdentifier>() as u32;
}

impl pallet_balances::Config for Runtime {
    type RuntimeFreezeReason = RuntimeFreezeReason;
    type MaxLocks = ConstU32<50>;
    type MaxReserves = ();
    type ReserveIdentifier = [u8; 8];
    /// The type for recording an account's balance.
    type Balance = Balance;
    /// The ubiquitous event type.
    type RuntimeEvent = RuntimeEvent;
    type DustRemoval = ();
    type ExistentialDeposit = ExistentialDeposit;
    type AccountStore = System;
    type WeightInfo = pallet_balances::weights::SubstrateWeight<Runtime>;
    type FreezeIdentifier = ();
    type MaxFreezes = ();
    type RuntimeHoldReason = HoldIdentifierWrapper;
}

parameter_types! {
    pub CreditSupply: Balance = Balances::total_issuance();
    pub TotalSpacePledged: u128 = {
        let pieces = solution_range_to_pieces(Subspace::solution_ranges().current, SLOT_PROBABILITY);
        pieces as u128 * Piece::SIZE as u128
    };
    pub BlockchainHistorySize: u128 = u128::from(Subspace::archived_history_size());
    pub DynamicCostOfStorage: bool = RuntimeConfigs::enable_dynamic_cost_of_storage();
    pub TransactionWeightFee: Balance = 100_000 * SHANNON;
}

impl pallet_transaction_fees::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type MinReplicationFactor = ConstU16<MIN_REPLICATION_FACTOR>;
    type CreditSupply = CreditSupply;
    type TotalSpacePledged = TotalSpacePledged;
    type BlockchainHistorySize = BlockchainHistorySize;
    type Currency = Balances;
    type FindBlockRewardAddress = Subspace;
    type DynamicCostOfStorage = DynamicCostOfStorage;
    type WeightInfo = pallet_transaction_fees::weights::SubstrateWeight<Runtime>;
}

impl pallet_transaction_payment::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type OnChargeTransaction = OnChargeTransaction;
    type OperationalFeeMultiplier = ConstU8<5>;
    type WeightToFee = ConstantMultiplier<Balance, TransactionWeightFee>;
    type LengthToFee = ConstantMultiplier<Balance, TransactionByteFee>;
    type FeeMultiplierUpdate = SlowAdjustingFeeUpdate<Runtime>;
}

impl pallet_utility::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type RuntimeCall = RuntimeCall;
    type PalletsOrigin = OriginCaller;
    type WeightInfo = pallet_utility::weights::SubstrateWeight<Runtime>;
}

impl pallet_sudo::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type RuntimeCall = RuntimeCall;
    type WeightInfo = pallet_sudo::weights::SubstrateWeight<Runtime>;
}

pub type CouncilCollective = pallet_collective::Instance1;

// Macro to implement 'Get' trait for each field of 'CouncilDemocracyConfigParams'
macro_rules! impl_get_council_democracy_field_block_number {
    ($field_type_name:ident, $field:ident) => {
        pub struct $field_type_name;

        impl Get<BlockNumber> for $field_type_name {
            fn get() -> BlockNumber {
                pallet_runtime_configs::CouncilDemocracyConfig::<Runtime>::get().$field
            }
        }
    };
}

impl_get_council_democracy_field_block_number! {CouncilMotionDuration, council_motion_duration}

parameter_types! {
    // maximum dispatch weight of a given council motion
    // currently set to 50% of maximum block weight
    pub MaxProposalWeight: Weight = Perbill::from_percent(50) * SubspaceBlockWeights::get().max_block;
}

pub type EnsureRootOr<O> = EitherOfDiverse<EnsureRoot<AccountId>, O>;
pub type AllCouncil = EnsureProportionAtLeast<AccountId, CouncilCollective, 1, 1>;
pub type TwoThirdsCouncil = EnsureProportionAtLeast<AccountId, CouncilCollective, 2, 3>;
pub type HalfCouncil = EnsureProportionAtLeast<AccountId, CouncilCollective, 1, 2>;

// TODO: update params for mainnnet
impl pallet_collective::Config<CouncilCollective> for Runtime {
    type DefaultVote = pallet_collective::PrimeDefaultVote;
    type MaxMembers = ConstU32<100>;
    type MaxProposalWeight = MaxProposalWeight;
    type MaxProposals = ConstU32<100>;
    /// Duration of voting for a given council motion.
    type MotionDuration = CouncilMotionDuration;
    type Proposal = RuntimeCall;
    type RuntimeEvent = RuntimeEvent;
    type RuntimeOrigin = RuntimeOrigin;
    type SetMembersOrigin = EnsureRootOr<AllCouncil>;
    type WeightInfo = pallet_collective::weights::SubstrateWeight<Runtime>;
}

// TODO: update params for mainnnet
parameter_types! {
    pub PreimageBaseDeposit: Balance = 100 * SSC;
    pub PreimageByteDeposit: Balance = SSC;
    pub const PreImageHoldReason: HoldIdentifierWrapper = HoldIdentifierWrapper(HoldIdentifier::Preimage);
}

impl pallet_preimage::Config for Runtime {
    type Consideration = HoldConsideration<
        AccountId,
        Balances,
        PreImageHoldReason,
        LinearStoragePrice<PreimageBaseDeposit, PreimageByteDeposit, Balance>,
    >;
    type Currency = Balances;
    type ManagerOrigin = EnsureRoot<AccountId>;
    type RuntimeEvent = RuntimeEvent;
    type WeightInfo = pallet_preimage::weights::SubstrateWeight<Runtime>;
}

parameter_types! {
    pub MaximumSchedulerWeight: Weight = Perbill::from_percent(80) * SubspaceBlockWeights::get().max_block;
    // Retry a scheduled item every 10 blocks (2 minutes) until the preimage exists.
    pub const NoPreimagePostponement: Option<u32> = Some(10);
}

impl pallet_scheduler::Config for Runtime {
    type MaxScheduledPerBlock = ConstU32<50>;
    type MaximumWeight = MaximumSchedulerWeight;
    type OriginPrivilegeCmp = EqualPrivilegeOnly;
    type PalletsOrigin = OriginCaller;
    type Preimages = Preimage;
    type RuntimeCall = RuntimeCall;
    type RuntimeEvent = RuntimeEvent;
    type RuntimeOrigin = RuntimeOrigin;
    type ScheduleOrigin = EnsureRoot<AccountId>;
    type WeightInfo = pallet_scheduler::weights::SubstrateWeight<Runtime>;
}

type NegativeImbalance = <Balances as Currency<AccountId>>::NegativeImbalance;

pub struct DemocracySlash;
impl OnUnbalanced<NegativeImbalance> for DemocracySlash {
    fn on_nonzero_unbalanced(slashed: NegativeImbalance) {
        Balances::resolve_creating(&TreasuryAccount::get(), slashed);
    }
}

impl_get_council_democracy_field_block_number! {CooloffPeriod, democracy_cooloff_period}
impl_get_council_democracy_field_block_number! {EnactmentPeriod, democracy_enactment_period}
impl_get_council_democracy_field_block_number! {FastTrackVotingPeriod, democracy_fast_track_voting_period}
impl_get_council_democracy_field_block_number! {LaunchPeriod, democracy_launch_period}
impl_get_council_democracy_field_block_number! {VoteLockingPeriod, democracy_vote_locking_period}
impl_get_council_democracy_field_block_number! {VotingPeriod, democracy_voting_period}

// TODO: update params for mainnnet
impl pallet_democracy::Config for Runtime {
    type BlacklistOrigin = EnsureRoot<AccountId>;
    /// To cancel a proposal before it has been passed and slash its backers, must be root.
    type CancelProposalOrigin = EnsureRoot<AccountId>;
    /// Origin to cancel a proposal.
    type CancellationOrigin = EnsureRootOr<TwoThirdsCouncil>;
    /// Period in blocks where an external proposal may not be re-submitted
    /// after being vetoed.
    type CooloffPeriod = CooloffPeriod;
    type Currency = Balances;
    /// The minimum period of locking and the period between a proposal being
    /// approved and enacted.
    type EnactmentPeriod = EnactmentPeriod;
    /// A unanimous council can have the next scheduled referendum be a straight
    /// default-carries (negative turnout biased) vote.
    /// 100% council vote.
    type ExternalDefaultOrigin = AllCouncil;
    /// A simple majority can have the next scheduled referendum be a straight
    /// majority-carries vote.
    /// 50% of council votes.
    type ExternalMajorityOrigin = HalfCouncil;
    /// A simple majority of the council can decide what their next motion is.
    /// 50% council votes.
    type ExternalOrigin = HalfCouncil;
    /// Half of the council can have an ExternalMajority/ExternalDefault vote
    /// be tabled immediately and with a shorter voting/enactment period.
    type FastTrackOrigin = EnsureRootOr<HalfCouncil>;
    /// Voting period for Fast track voting.
    type FastTrackVotingPeriod = FastTrackVotingPeriod;
    type InstantAllowed = ConstBool<true>;
    type InstantOrigin = EnsureRootOr<AllCouncil>;
    /// How often (in blocks) new public referenda are launched.
    type LaunchPeriod = LaunchPeriod;
    type MaxBlacklisted = ConstU32<100>;
    type MaxDeposits = ConstU32<100>;
    type MaxProposals = ConstU32<100>;
    type MaxVotes = ConstU32<100>;
    /// The minimum amount to be used as a deposit for a public referendum
    /// proposal.
    type MinimumDeposit = ConstU128<{ 1000 * SSC }>;
    type PalletsOrigin = OriginCaller;
    type Preimages = Preimage;
    type RuntimeEvent = RuntimeEvent;
    type Scheduler = Scheduler;
    /// Handler for the unbalanced reduction when slashing a preimage deposit.
    type Slash = DemocracySlash;
    /// Origin used to submit proposals.
    /// Currently set to Council member so that no one can submit new proposals except council through democracy
    type SubmitOrigin = EnsureMember<AccountId, CouncilCollective>;
    /// Any single council member may veto a coming council proposal, however they
    /// can only do it once and it lasts only for the cooloff period.
    type VetoOrigin = EnsureMember<AccountId, CouncilCollective>;
    type VoteLockingPeriod = VoteLockingPeriod;
    /// How often (in blocks) to check for new votes.
    type VotingPeriod = VotingPeriod;
    type WeightInfo = pallet_democracy::weights::SubstrateWeight<Runtime>;
}

impl pallet_history_seeding::Config for Runtime {
    type WeightInfo = pallet_history_seeding::weights::SubstrateWeight<Runtime>;
}

parameter_types! {
    pub const SelfChainId: ChainId = ChainId::Consensus;
}

pub struct OnXDMRewards;

impl sp_messenger::OnXDMRewards<Balance> for OnXDMRewards {
    fn on_xdm_rewards(reward: Balance) {
        if let Some(block_author) = Subspace::find_block_reward_address() {
            let _ = Balances::deposit_creating(&block_author, reward);
        }
    }

    fn on_chain_protocol_fees(chain_id: ChainId, fees: Balance) {
        // on consensus chain, reward the domain operators
        // balance is already on this consensus runtime
        if let ChainId::Domain(domain_id) = chain_id {
            Domains::reward_domain_operators(domain_id, OperatorRewardSource::XDMProtocolFees, fees)
        }
    }
}

pub struct MmrProofVerifier;

impl sp_subspace_mmr::MmrProofVerifier<mmr::Hash, NumberFor<Block>, Hash> for MmrProofVerifier {
    fn verify_proof_and_extract_leaf(
        mmr_leaf_proof: ConsensusChainMmrLeafProof<NumberFor<Block>, Hash, mmr::Hash>,
    ) -> Option<mmr::Leaf> {
        let mmr_root = SubspaceMmr::mmr_root_hash(mmr_leaf_proof.consensus_block_number)?;
        Self::verify_proof_stateless(mmr_root, mmr_leaf_proof)
    }

    fn verify_proof_stateless(
        mmr_root: mmr::Hash,
        mmr_leaf_proof: ConsensusChainMmrLeafProof<NumberFor<Block>, Hash, mmr::Hash>,
    ) -> Option<mmr::Leaf> {
        let ConsensusChainMmrLeafProof {
            opaque_mmr_leaf,
            proof,
            ..
        } = mmr_leaf_proof;

        pallet_mmr::verify_leaves_proof::<mmr::Hashing, _>(
            mmr_root,
            vec![mmr::DataOrHash::Data(
                EncodableOpaqueLeaf(opaque_mmr_leaf.0.clone()).into_opaque_leaf(),
            )],
            proof,
        )
        .ok()?;

        let leaf: mmr::Leaf = opaque_mmr_leaf.into_opaque_leaf().try_decode()?;

        Some(leaf)
    }
}

pub struct StorageKeys;

impl sp_messenger::StorageKeys for StorageKeys {
    fn confirmed_domain_block_storage_key(domain_id: DomainId) -> Option<Vec<u8>> {
        Some(Domains::confirmed_domain_block_storage_key(domain_id))
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

parameter_types! {
    // TODO: update value
    pub const ChannelReserveFee: Balance = 100 * SSC;
    pub const ChannelInitReservePortion: Perbill = Perbill::from_percent(20);
    // TODO update the fee model
    pub const ChannelFeeModel: FeeModel<Balance> = FeeModel{relay_fee: SSC};
}

pub struct DomainRegistration;
impl sp_messenger::DomainRegistration for DomainRegistration {
    fn is_domain_registered(domain_id: DomainId) -> bool {
        Domains::is_domain_registered(domain_id)
    }
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
    type MmrHash = mmr::Hash;
    type MmrProofVerifier = MmrProofVerifier;
    type StorageKeys = StorageKeys;
    type DomainOwner = Domains;
    type HoldIdentifier = HoldIdentifierWrapper;
    type ChannelReserveFee = ChannelReserveFee;
    type ChannelInitReservePortion = ChannelInitReservePortion;
    type DomainRegistration = DomainRegistration;
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
    type AccountIdConverter = AccountIdConverter;
    type WeightInfo = pallet_transporter::weights::SubstrateWeight<Runtime>;
}

parameter_types! {
    pub const MaximumReceiptDrift: BlockNumber = 128;
    pub const InitialDomainTxRange: u64 = INITIAL_DOMAIN_TX_RANGE;
    pub const DomainTxRangeAdjustmentInterval: u64 = TX_RANGE_ADJUSTMENT_INTERVAL_BLOCKS;
    /// Runtime upgrade is delayed for 1 day at 6 sec block time.
    pub const DomainRuntimeUpgradeDelay: BlockNumber = 14_400;
    /// Minimum operator stake to become an operator.
    // TODO: this value should be properly updated before mainnet
    pub const MinOperatorStake: Balance = 100 * SSC;
    /// Minimum nominator stake to nominate and operator.
    // TODO: this value should be properly updated before mainnet
    pub const MinNominatorStake: Balance = SSC;
    /// Use the consensus chain's `Normal` extrinsics block size limit as the domain block size limit
    pub MaxDomainBlockSize: u32 = NORMAL_DISPATCH_RATIO * MAX_BLOCK_LENGTH;
    /// Use the consensus chain's `Normal` extrinsics block weight limit as the domain block weight limit
    pub MaxDomainBlockWeight: Weight = maximum_domain_block_weight();
    pub const DomainInstantiationDeposit: Balance = 100 * SSC;
    pub const MaxDomainNameLength: u32 = 32;
    pub const BlockTreePruningDepth: u32 = 14_400;
    pub const StakeWithdrawalLockingPeriod: DomainNumber = 14_400;
    // TODO: revisit these. For now epoch every 10 mins for a 6 second block and only 100 number of staking
    // operations allowed within each epoch.
    pub const StakeEpochDuration: DomainNumber = 100;
    pub TreasuryAccount: AccountId = PalletId(*b"treasury").into_account_truncating();
    pub const MaxPendingStakingOperation: u32 = 512;
    pub const DomainsPalletId: PalletId = PalletId(*b"domains_");
    pub const MaxInitialDomainAccounts: u32 = 10;
    pub const MinInitialDomainAccountBalance: Balance = SSC;
    pub const BundleLongevity: u32 = 5;
    pub const WithdrawalLimit: u32 = 32;
}

// `BlockSlotCount` must at least keep the slot for the current and the parent block, it also need to
// keep enough block slot for bundle validation
const_assert!(BlockSlotCount::get() >= 2 && BlockSlotCount::get() > BundleLongevity::get());

// `BlockHashCount` must greater than `BlockSlotCount` because we need to use the block number found
// with `BlockSlotCount` to get the block hash.
const_assert!(BlockHashCount::get() > BlockSlotCount::get());

// Minimum operator stake must be >= minimum nominator stake since operator is also a nominator.
const_assert!(MinOperatorStake::get() >= MinNominatorStake::get());

// Stake Withdrawal locking period must be >= Block tree pruning depth
const_assert!(StakeWithdrawalLockingPeriod::get() >= BlockTreePruningDepth::get());

pub struct BlockSlot;

impl pallet_domains::BlockSlot<Runtime> for BlockSlot {
    fn future_slot(block_number: BlockNumber) -> Option<sp_consensus_slots::Slot> {
        let block_slots = Subspace::block_slots();
        block_slots
            .get(&block_number)
            .map(|slot| *slot + Slot::from(BlockAuthoringDelay::get()))
    }

    fn slot_produced_after(to_check: sp_consensus_slots::Slot) -> Option<BlockNumber> {
        let block_slots = Subspace::block_slots();
        for (block_number, slot) in block_slots.into_iter().rev() {
            if to_check > slot {
                return Some(block_number);
            }
        }
        None
    }
}

pub struct OnChainRewards;

impl sp_domains::OnChainRewards<Balance> for OnChainRewards {
    fn on_chain_rewards(chain_id: ChainId, reward: Balance) {
        match chain_id {
            ChainId::Consensus => {
                if let Some(block_author) = Subspace::find_block_reward_address() {
                    let _ = Balances::deposit_creating(&block_author, reward);
                }
            }
            ChainId::Domain(domain_id) => Domains::reward_domain_operators(
                domain_id,
                OperatorRewardSource::XDMProtocolFees,
                reward,
            ),
        }
    }
}

impl pallet_domains::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type DomainHash = DomainHash;
    type DomainHeader = sp_runtime::generic::Header<DomainNumber, BlakeTwo256>;
    type ConfirmationDepthK = ConfirmationDepthK;
    type DomainRuntimeUpgradeDelay = DomainRuntimeUpgradeDelay;
    type Currency = Balances;
    type HoldIdentifier = HoldIdentifierWrapper;
    type WeightInfo = pallet_domains::weights::SubstrateWeight<Runtime>;
    type InitialDomainTxRange = InitialDomainTxRange;
    type DomainTxRangeAdjustmentInterval = DomainTxRangeAdjustmentInterval;
    type MinOperatorStake = MinOperatorStake;
    type MinNominatorStake = MinNominatorStake;
    type MaxDomainBlockSize = MaxDomainBlockSize;
    type MaxDomainBlockWeight = MaxDomainBlockWeight;
    type DomainInstantiationDeposit = DomainInstantiationDeposit;
    type MaxDomainNameLength = MaxDomainNameLength;
    type Share = Balance;
    type BlockTreePruningDepth = BlockTreePruningDepth;
    type ConsensusSlotProbability = SlotProbability;
    type StakeWithdrawalLockingPeriod = StakeWithdrawalLockingPeriod;
    type StakeEpochDuration = StakeEpochDuration;
    type TreasuryAccount = TreasuryAccount;
    type MaxPendingStakingOperation = MaxPendingStakingOperation;
    type Randomness = Subspace;
    type PalletId = DomainsPalletId;
    type StorageFee = TransactionFees;
    type BlockTimestamp = pallet_timestamp::Pallet<Runtime>;
    type BlockSlot = BlockSlot;
    type BundleLongevity = BundleLongevity;
    type DomainsTransfersTracker = Transporter;
    type MaxInitialDomainAccounts = MaxInitialDomainAccounts;
    type MinInitialDomainAccountBalance = MinInitialDomainAccountBalance;
    type DomainBundleSubmitted = Messenger;
    type OnDomainInstantiated = Messenger;
    type Balance = Balance;
    type MmrHash = mmr::Hash;
    type MmrProofVerifier = MmrProofVerifier;
    type FraudProofStorageKeyProvider = StorageKeyProvider;
    type OnChainRewards = OnChainRewards;
    type WithdrawalLimit = WithdrawalLimit;
}

parameter_types! {
    pub const AvgBlockspaceUsageNumBlocks: BlockNumber = 100;
    pub const ProposerTaxOnVotes: (u32, u32) = (1, 10);
}

impl pallet_rewards::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type Currency = Balances;
    type AvgBlockspaceUsageNumBlocks = AvgBlockspaceUsageNumBlocks;
    type TransactionByteFee = TransactionByteFee;
    type MaxRewardPoints = ConstU32<20>;
    type ProposerTaxOnVotes = ProposerTaxOnVotes;
    type RewardsEnabled = Subspace;
    type FindBlockRewardAddress = Subspace;
    type FindVotingRewardAddresses = Subspace;
    type WeightInfo = pallet_rewards::weights::SubstrateWeight<Runtime>;
    type OnReward = ();
}

impl pallet_runtime_configs::Config for Runtime {
    type WeightInfo = pallet_runtime_configs::weights::SubstrateWeight<Runtime>;
}

mod mmr {
    use super::Runtime;
    pub use pallet_mmr::primitives::*;

    pub type Leaf = <<Runtime as pallet_mmr::Config>::LeafData as LeafDataProvider>::LeafData;
    pub type Hashing = <Runtime as pallet_mmr::Config>::Hashing;
    pub type Hash = <Hashing as sp_runtime::traits::Hash>::Output;
}

pub struct BlockHashProvider;

impl pallet_mmr::BlockHashProvider<BlockNumber, Hash> for BlockHashProvider {
    fn block_hash(block_number: BlockNumber) -> Hash {
        consensus_block_hash(block_number).expect("Hash must exist for a given block number.")
    }
}

impl pallet_mmr::Config for Runtime {
    const INDEXING_PREFIX: &'static [u8] = mmr::INDEXING_PREFIX;
    type Hashing = Keccak256;
    type LeafData = SubspaceMmr;
    type OnNewRoot = SubspaceMmr;
    type BlockHashProvider = BlockHashProvider;
    type WeightInfo = ();
    #[cfg(feature = "runtime-benchmarks")]
    type BenchmarkHelper = ();
}

parameter_types! {
    pub const MmrRootHashCount: u32 = 1024;
}

impl pallet_subspace_mmr::Config for Runtime {
    type MmrRootHash = mmr::Hash;
    type MmrRootHashCount = MmrRootHashCount;
}

construct_runtime!(
    pub struct Runtime {
        System: frame_system = 0,
        Timestamp: pallet_timestamp = 1,

        Subspace: pallet_subspace = 2,
        Rewards: pallet_rewards = 4,

        Balances: pallet_balances = 5,
        TransactionFees: pallet_transaction_fees = 6,
        TransactionPayment: pallet_transaction_payment = 7,
        Utility: pallet_utility = 8,

        Domains: pallet_domains = 12,
        RuntimeConfigs: pallet_runtime_configs = 14,

        Mmr: pallet_mmr = 30,
        SubspaceMmr: pallet_subspace_mmr = 31,

        // messenger stuff
        // Note: Indexes should match with indexes on other chains and domains
        Messenger: pallet_messenger exclude_parts { Inherent } = 60,
        Transporter: pallet_transporter = 61,

        // council and democracy
        Scheduler: pallet_scheduler = 81,
        Council: pallet_collective::<Instance1> = 82,
        Democracy: pallet_democracy = 83,
        Preimage: pallet_preimage = 84,

        HistorySeeding: pallet_history_seeding = 91,

        // Reserve some room for other pallets as we'll remove sudo pallet eventually.
        Sudo: pallet_sudo = 100,
    }
);

/// The address format for describing accounts.
pub type Address = sp_runtime::MultiAddress<AccountId, ()>;
/// Block header type as expected by this runtime.
pub type Header = generic::Header<BlockNumber, BlakeTwo256>;
/// Block type as expected by this runtime.
pub type Block = generic::Block<Header, UncheckedExtrinsic>;

/// The SignedExtension to the basic transaction logic.
pub type SignedExtra = (
    frame_system::CheckNonZeroSender<Runtime>,
    frame_system::CheckSpecVersion<Runtime>,
    frame_system::CheckTxVersion<Runtime>,
    frame_system::CheckGenesis<Runtime>,
    frame_system::CheckMortality<Runtime>,
    frame_system::CheckNonce<Runtime>,
    frame_system::CheckWeight<Runtime>,
    pallet_transaction_payment::ChargeTransactionPayment<Runtime>,
    DisablePallets,
    CheckHistorySeeder<Runtime>,
);
/// Unchecked extrinsic type as expected by this runtime.
pub type UncheckedExtrinsic =
    generic::UncheckedExtrinsic<Address, RuntimeCall, Signature, SignedExtra>;

/// Executive: handles dispatch to the various modules.
pub type Executive = frame_executive::Executive<
    Runtime,
    Block,
    frame_system::ChainContext<Runtime>,
    Runtime,
    AllPalletsWithSystem,
>;

fn extract_segment_headers(ext: &UncheckedExtrinsic) -> Option<Vec<SegmentHeader>> {
    match &ext.function {
        RuntimeCall::Subspace(pallet_subspace::Call::store_segment_headers { segment_headers }) => {
            Some(segment_headers.clone())
        }
        _ => None,
    }
}

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

            let mmr_root = SubspaceMmr::mmr_root_hash(consensus_block_number)?;

            Some(
                pallet_mmr::verify_leaves_proof::<mmr::Hashing, _>(
                    mmr_root,
                    vec![mmr::DataOrHash::Data(
                        EncodableOpaqueLeaf(opaque_mmr_leaf.0.clone()).into_opaque_leaf(),
                    )],
                    proof,
                )
                .is_ok(),
            )
        }
        _ => None,
    }
}

struct RewardAddress([u8; 32]);

impl From<PublicKey> for RewardAddress {
    #[inline]
    fn from(public_key: PublicKey) -> Self {
        Self(*public_key)
    }
}

impl From<RewardAddress> for AccountId32 {
    #[inline]
    fn from(reward_address: RewardAddress) -> Self {
        reward_address.0.into()
    }
}

pub struct StorageKeyProvider;
impl FraudProofStorageKeyProvider<NumberFor<Block>> for StorageKeyProvider {
    fn storage_key(req: FraudProofStorageKeyRequest<NumberFor<Block>>) -> Vec<u8> {
        match req {
            FraudProofStorageKeyRequest::InvalidInherentExtrinsicData => {
                pallet_domains::BlockInvalidInherentExtrinsicData::<Runtime>::hashed_key().to_vec()
            }
            FraudProofStorageKeyRequest::SuccessfulBundles(domain_id) => {
                pallet_domains::SuccessfulBundles::<Runtime>::hashed_key_for(domain_id)
            }
            FraudProofStorageKeyRequest::DomainAllowlistUpdates(domain_id) => {
                Messenger::domain_allow_list_update_storage_key(domain_id)
            }
            FraudProofStorageKeyRequest::DomainRuntimeUpgrades => {
                pallet_domains::DomainRuntimeUpgrades::<Runtime>::hashed_key().to_vec()
            }
            FraudProofStorageKeyRequest::RuntimeRegistry(runtime_id) => {
                pallet_domains::RuntimeRegistry::<Runtime>::hashed_key_for(runtime_id)
            }
            FraudProofStorageKeyRequest::DomainSudoCall(domain_id) => {
                pallet_domains::DomainSudoCalls::<Runtime>::hashed_key_for(domain_id)
            }
            FraudProofStorageKeyRequest::MmrRoot(block_number) => {
                pallet_subspace_mmr::MmrRootHashes::<Runtime>::hashed_key_for(block_number)
            }
        }
    }
}

#[cfg(feature = "runtime-benchmarks")]
mod benches {
    frame_benchmarking::define_benchmarks!(
        [frame_benchmarking, BaselineBench::<Runtime>]
        [frame_system, SystemBench::<Runtime>]
        [pallet_balances, Balances]
        [pallet_domains, Domains]
        [pallet_mmr, Mmr]
        [pallet_rewards, Rewards]
        [pallet_runtime_configs, RuntimeConfigs]
        [pallet_subspace, Subspace]
        [pallet_timestamp, Timestamp]
        [pallet_messenger, Messenger]
        [pallet_transporter, Transporter]
        [pallet_history_seeding, HistorySeeding]
    );
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
            Executive::execute_block(block);
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

    impl sp_objects::ObjectsApi<Block> for Runtime {
        fn extract_block_object_mapping(block: Block) -> BlockObjectMapping {
            extract_block_object_mapping(block)
        }
    }

    impl sp_consensus_subspace::SubspaceApi<Block, PublicKey> for Runtime {
        fn pot_parameters() -> PotParameters {
            Subspace::pot_parameters()
        }

        fn solution_ranges() -> SolutionRanges {
            Subspace::solution_ranges()
        }

        fn submit_vote_extrinsic(
            signed_vote: SignedVote<NumberFor<Block>, <Block as BlockT>::Hash, PublicKey>,
        ) {
            let SignedVote { vote, signature } = signed_vote;
            let Vote::V0 {
                height,
                parent_hash,
                slot,
                solution,
                proof_of_time,
                future_proof_of_time,
            } = vote;

            Subspace::submit_vote(SignedVote {
                vote: Vote::V0 {
                    height,
                    parent_hash,
                    slot,
                    solution: solution.into_reward_address_format::<RewardAddress, AccountId32>(),
                    proof_of_time,
                    future_proof_of_time,
                },
                signature,
            })
        }

        fn history_size() -> HistorySize {
            <pallet_subspace::Pallet<Runtime>>::history_size()
        }

        fn max_pieces_in_sector() -> u16 {
            MAX_PIECES_IN_SECTOR
        }

        fn segment_commitment(segment_index: SegmentIndex) -> Option<SegmentCommitment> {
            Subspace::segment_commitment(segment_index)
        }

        fn extract_segment_headers(ext: &<Block as BlockT>::Extrinsic) -> Option<Vec<SegmentHeader >> {
            extract_segment_headers(ext)
        }

        fn is_inherent(ext: &<Block as BlockT>::Extrinsic) -> bool {
            match &ext.function {
                RuntimeCall::Subspace(call) => Subspace::is_inherent(call),
                RuntimeCall::Timestamp(call) => Timestamp::is_inherent(call),
                _ => false,
            }
        }

        fn root_plot_public_key() -> Option<PublicKey> {
            Subspace::root_plot_public_key()
        }

        fn should_adjust_solution_range() -> bool {
            Subspace::should_adjust_solution_range()
        }

        fn chain_constants() -> ChainConstants {
            ChainConstants::V0 {
                confirmation_depth_k: ConfirmationDepthK::get(),
                block_authoring_delay: Slot::from(BlockAuthoringDelay::get()),
                era_duration: EraDuration::get(),
                slot_probability: SlotProbability::get(),
                slot_duration: SlotDuration::from_millis(SLOT_DURATION),
                recent_segments: RecentSegments::get(),
                recent_history_fraction: RecentHistoryFraction::get(),
                min_sector_lifetime: MinSectorLifetime::get(),
            }
        }
    }

    impl sp_domains::DomainsApi<Block, DomainHeader> for Runtime {
        fn submit_bundle_unsigned(
            opaque_bundle: sp_domains::OpaqueBundle<NumberFor<Block>, <Block as BlockT>::Hash, DomainHeader, Balance>,
        ) {
            Domains::submit_bundle_unsigned(opaque_bundle)
        }

        fn submit_receipt_unsigned(
            singleton_receipt: sp_domains::SealedSingletonReceipt<NumberFor<Block>, <Block as BlockT>::Hash, DomainHeader, Balance>,
        ) {
            Domains::submit_receipt_unsigned(singleton_receipt)
        }

        fn extract_successful_bundles(
            domain_id: DomainId,
            extrinsics: Vec<<Block as BlockT>::Extrinsic>,
        ) -> sp_domains::OpaqueBundles<Block, DomainHeader, Balance> {
            crate::domains::extract_successful_bundles(domain_id, extrinsics)
        }

        fn extrinsics_shuffling_seed() -> Randomness {
            Randomness::from(Domains::extrinsics_shuffling_seed().to_fixed_bytes())
        }

        fn domain_runtime_code(domain_id: DomainId) -> Option<Vec<u8>> {
            Domains::domain_runtime_code(domain_id)
        }

        fn runtime_id(domain_id: DomainId) -> Option<sp_domains::RuntimeId> {
            Domains::runtime_id(domain_id)
        }

        fn domain_instance_data(domain_id: DomainId) -> Option<(DomainInstanceData, NumberFor<Block>)> {
            Domains::domain_instance_data(domain_id)
        }

        fn timestamp() -> Moment{
            Timestamp::now()
        }

        fn domain_tx_range(domain_id: DomainId) -> U256 {
            Domains::domain_tx_range(domain_id)
        }

        fn genesis_state_root(domain_id: DomainId) -> Option<H256> {
            Domains::genesis_state_root(domain_id)
        }

        fn head_receipt_number(domain_id: DomainId) -> DomainNumber {
            Domains::head_receipt_number(domain_id)
        }

        fn oldest_unconfirmed_receipt_number(domain_id: DomainId) -> Option<DomainNumber> {
            Domains::oldest_unconfirmed_receipt_number(domain_id)
        }

        fn domain_bundle_limit(domain_id: DomainId) -> Option<sp_domains::DomainBundleLimit> {
            Domains::domain_bundle_limit(domain_id).ok().flatten()
        }

        fn non_empty_er_exists(domain_id: DomainId) -> bool {
            Domains::non_empty_er_exists(domain_id)
        }

        fn domain_best_number(domain_id: DomainId) -> Option<DomainNumber> {
            Domains::domain_best_number(domain_id).ok()
        }

        fn execution_receipt(receipt_hash: DomainHash) -> Option<ExecutionReceiptFor<DomainHeader, Block, Balance>> {
            Domains::execution_receipt(receipt_hash)
        }

        fn domain_operators(domain_id: DomainId) -> Option<(BTreeMap<OperatorId, Balance>, Vec<OperatorId>)> {
            Domains::domain_staking_summary(domain_id).map(|summary| {
                let next_operators = summary.next_operators.into_iter().collect();
                (summary.current_operators, next_operators)
            })
        }

        fn receipt_hash(domain_id: DomainId, domain_number: DomainNumber) -> Option<DomainHash> {
            Domains::receipt_hash(domain_id, domain_number)
        }

        fn consensus_chain_byte_fee() -> Balance {
            DOMAIN_STORAGE_FEE_MULTIPLIER * TransactionFees::transaction_byte_fee()
        }

        fn latest_confirmed_domain_block(domain_id: DomainId) -> Option<(DomainNumber, DomainHash)>{
            Domains::latest_confirmed_domain_block(domain_id)
        }

        fn is_bad_er_pending_to_prune(domain_id: DomainId, receipt_hash: DomainHash) -> bool {
            Domains::execution_receipt(receipt_hash).map(
                |er| Domains::is_bad_er_pending_to_prune(domain_id, er.domain_block_number)
            )
            .unwrap_or(false)
        }

        fn storage_fund_account_balance(operator_id: OperatorId) -> Balance {
            Domains::storage_fund_account_balance(operator_id)
        }

        fn is_domain_runtime_upgraded_since(domain_id: DomainId, at: NumberFor<Block>) -> Option<bool> {
            Domains::is_domain_runtime_upgraded_since(domain_id, at)
        }

        fn domain_sudo_call(domain_id: DomainId) -> Option<Vec<u8>> {
            Domains::domain_sudo_call(domain_id)
        }

        fn last_confirmed_domain_block_receipt(domain_id: DomainId) -> Option<ExecutionReceiptFor<DomainHeader, Block, Balance>>{
            Domains::latest_confirmed_domain_execution_receipt(domain_id)
        }
    }

    impl sp_domains::BundleProducerElectionApi<Block, Balance> for Runtime {
        fn bundle_producer_election_params(domain_id: DomainId) -> Option<BundleProducerElectionParams<Balance>> {
            Domains::bundle_producer_election_params(domain_id)
        }

        fn operator(operator_id: OperatorId) -> Option<(OperatorPublicKey, Balance)> {
            Domains::operator(operator_id)
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

    impl sp_messenger::MessengerApi<Block, BlockNumber, <Block as BlockT>::Hash> for Runtime {
        fn is_xdm_mmr_proof_valid(
            ext: &<Block as BlockT>::Extrinsic
        ) -> Option<bool> {
            is_xdm_mmr_proof_valid(ext)
        }

        fn extract_xdm_mmr_proof(ext: &<Block as BlockT>::Extrinsic) -> Option<ConsensusChainMmrLeafProof<BlockNumber, <Block as BlockT>::Hash, sp_core::H256>> {
            match &ext.function {
                RuntimeCall::Messenger(pallet_messenger::Call::relay_message { msg })
                | RuntimeCall::Messenger(pallet_messenger::Call::relay_message_response { msg }) => {
                    Some(msg.proof.consensus_mmr_proof())
                }
                _ => None,
            }
        }

        fn confirmed_domain_block_storage_key(domain_id: DomainId) -> Vec<u8> {
            Domains::confirmed_domain_block_storage_key(domain_id)
        }

        fn outbox_storage_key(message_key: MessageKey) -> Vec<u8> {
            Messenger::outbox_storage_key(message_key)
        }

        fn inbox_response_storage_key(message_key: MessageKey) -> Vec<u8> {
            Messenger::inbox_response_storage_key(message_key)
        }

        fn domain_chains_allowlist_update(domain_id: DomainId) -> Option<DomainAllowlistUpdates>{
            Messenger::domain_chains_allowlist_update(domain_id)
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

    impl sp_messenger::RelayerApi<Block, BlockNumber, BlockNumber, <Block as BlockT>::Hash> for Runtime {
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

    impl sp_domains_fraud_proof::FraudProofApi<Block, DomainHeader> for Runtime {
        fn submit_fraud_proof_unsigned(fraud_proof: FraudProof<NumberFor<Block>, <Block as BlockT>::Hash, DomainHeader, H256>) {
            Domains::submit_fraud_proof_unsigned(fraud_proof)
        }

        fn fraud_proof_storage_key(req: FraudProofStorageKeyRequest<NumberFor<Block>>) -> Vec<u8> {
            <StorageKeyProvider as FraudProofStorageKeyProvider<NumberFor<Block>>>::storage_key(req)
        }
    }

    impl mmr::MmrApi<Block, mmr::Hash, BlockNumber> for Runtime {
        fn mmr_root() -> Result<mmr::Hash, mmr::Error> {
            Ok(Mmr::mmr_root())
        }

        fn mmr_leaf_count() -> Result<mmr::LeafIndex, mmr::Error> {
            Ok(Mmr::mmr_leaves())
        }

        fn generate_proof(
            block_numbers: Vec<BlockNumber>,
            best_known_block_number: Option<BlockNumber>,
        ) -> Result<(Vec<mmr::EncodableOpaqueLeaf>, mmr::LeafProof<mmr::Hash>), mmr::Error> {
            Mmr::generate_proof(block_numbers, best_known_block_number).map(
                |(leaves, proof)| {
                    (
                        leaves
                            .into_iter()
                            .map(|leaf| mmr::EncodableOpaqueLeaf::from_leaf(&leaf))
                            .collect(),
                        proof,
                    )
                },
            )
        }

        fn verify_proof(leaves: Vec<mmr::EncodableOpaqueLeaf>, proof: mmr::LeafProof<mmr::Hash>)
            -> Result<(), mmr::Error>
        {
            let leaves = leaves.into_iter().map(|leaf|
                leaf.into_opaque_leaf()
                .try_decode()
                .ok_or(mmr::Error::Verify)).collect::<Result<Vec<mmr::Leaf>, mmr::Error>>()?;
            Mmr::verify_leaves(leaves, proof)
        }

        fn verify_proof_stateless(
            root: mmr::Hash,
            leaves: Vec<mmr::EncodableOpaqueLeaf>,
            proof: mmr::LeafProof<mmr::Hash>
        ) -> Result<(), mmr::Error> {
            let nodes = leaves.into_iter().map(|leaf|mmr::DataOrHash::Data(leaf.into_opaque_leaf())).collect();
            pallet_mmr::verify_leaves_proof::<mmr::Hashing, _>(root, nodes, proof)
        }
    }

    impl sp_genesis_builder::GenesisBuilder<Block> for Runtime {
        fn build_state(config: Vec<u8>) -> sp_genesis_builder::Result {
            build_state::<RuntimeGenesisConfig>(config)
        }

        fn get_preset(_id: &Option<sp_genesis_builder::PresetId>) -> Option<Vec<u8>> {
            // By passing `None` the upstream `get_preset` will return the default value of `RuntimeGenesisConfig`
            get_preset::<RuntimeGenesisConfig>(&None, |_| None)
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
            use sp_core::storage::TrackedStorageKey;

            use frame_system_benchmarking::Pallet as SystemBench;
            use baseline::Pallet as BaselineBench;

            use frame_support::traits::WhitelistedStorageKeys;
            let whitelist: Vec<TrackedStorageKey> = AllPalletsWithSystem::whitelisted_storage_keys();

            let mut batches = Vec::<BenchmarkBatch>::new();
            let params = (&config, &whitelist);
            add_benchmarks!(params, batches);

            Ok(batches)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Runtime, SubspaceBlockWeights as BlockWeights};
    use subspace_runtime_primitives::tests_utils::FeeMultiplierUtils;

    #[test]
    fn multiplier_can_grow_from_zero() {
        FeeMultiplierUtils::<Runtime, BlockWeights>::multiplier_can_grow_from_zero()
    }
}
