//! Runtime primitives for Subspace Network.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod utility;

#[cfg(not(feature = "std"))]
extern crate alloc;

use crate::time::{BLOCKS_IN_AN_MINUTE, BLOCKS_IN_A_DAY};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use codec::{Codec, Decode, Encode, MaxEncodedLen};
use frame_support::pallet_prelude::Weight;
use frame_support::traits::tokens;
use frame_support::weights::constants::WEIGHT_REF_TIME_PER_SECOND;
use frame_support::{Deserialize, Serialize};
use frame_system::limits::BlockLength;
use pallet_transaction_payment::{Multiplier, TargetedFeeAdjustment};
use scale_info::TypeInfo;
use sp_core::parameter_types;
use sp_runtime::traits::{Bounded, IdentifyAccount, Verify};
use sp_runtime::{FixedPointNumber, MultiSignature, Perbill, Perquintill};
pub use subspace_core_primitives::BlockNumber;

/// Minimum desired number of replicas of the blockchain to be stored by the network,
/// impacts storage fees.
pub const MIN_REPLICATION_FACTOR: u16 = 25;

/// The smallest unit of the token is called Shannon.
pub const SHANNON: Balance = 1;
/// Subspace Credits have 18 decimal places.
pub const DECIMAL_PLACES: u8 = 18;
/// One Subspace Credit.
pub const SSC: Balance = (10 * SHANNON).pow(DECIMAL_PLACES as u32);
/// A ratio of `Normal` dispatch class within block, for `BlockWeight` and `BlockLength`.
pub const NORMAL_DISPATCH_RATIO: Perbill = Perbill::from_percent(75);
/// 1 in 6 slots (on average, not counting collisions) will have a block.
/// Must match ratio between block and slot duration in constants above.
pub const SLOT_PROBABILITY: (u64, u64) = (1, 6);
/// The block weight for 2 seconds of compute
pub const BLOCK_WEIGHT_FOR_2_SEC: Weight =
    Weight::from_parts(WEIGHT_REF_TIME_PER_SECOND.saturating_mul(2), u64::MAX);

/// Maximum block length for non-`Normal` extrinsic is 5 MiB.
pub const MAX_BLOCK_LENGTH: u32 = 5 * 1024 * 1024;

/// We allow for 3.75 MiB for `Normal` extrinsic with 5 MiB maximum block length.
pub fn maximum_normal_block_length() -> BlockLength {
    BlockLength::max_with_normal_ratio(MAX_BLOCK_LENGTH, NORMAL_DISPATCH_RATIO)
}

/// Alias to 512-bit hash when used in the context of a transaction signature on the chain.
pub type Signature = MultiSignature;

/// Some way of identifying an account on the chain. We intentionally make it equivalent
/// to the public key of our transaction signing scheme.
pub type AccountId = <<Signature as Verify>::Signer as IdentifyAccount>::AccountId;

/// Balance of an account.
pub type Balance = u128;

/// Index of a transaction in the chain.
pub type Nonce = u32;

/// A hash of some data used by the chain.
pub type Hash = sp_core::H256;

/// Type used for expressing timestamp.
pub type Moment = u64;

/// Opaque types.
///
/// These are used by the CLI to instantiate machinery that don't need to know the specifics of the
/// runtime. They can then be made to be agnostic over specific formats of data like extrinsics,
/// allowing for them to continue syncing the network through upgrades to even the core data
/// structures.
pub mod opaque {
    use super::BlockNumber;
    use sp_runtime::generic;
    use sp_runtime::traits::BlakeTwo256;
    pub use sp_runtime::OpaqueExtrinsic as UncheckedExtrinsic;

    /// Opaque block header type.
    pub type Header = generic::Header<BlockNumber, BlakeTwo256>;
    /// Opaque block type.
    pub type Block = generic::Block<Header, UncheckedExtrinsic>;
}

pub mod time {
    /// Expected block time in milliseconds.
    ///
    /// Since Subspace is probabilistic this is the average expected block time that
    /// we are targeting. Blocks will be produced at a minimum duration defined
    /// by `SLOT_DURATION`, but some slots will not be allocated to any
    /// farmer and hence no block will be produced. We expect to have this
    /// block time on average following the defined slot duration and the value
    /// of `c` configured for Subspace (where `1 - c` represents the probability of
    /// a slot being empty).
    /// This value is only used indirectly to define the unit constants below
    /// that are expressed in blocks. The rest of the code should use
    /// `SLOT_DURATION` instead (like the Timestamp pallet for calculating the
    /// minimum period).
    ///
    /// Based on:
    /// <https://research.web3.foundation/en/latest/polkadot/block-production/Babe.html#-6.-practical-results>
    pub const MILLISECS_PER_BLOCK: u64 = 6000;
    /// Approximate number of block in a minute.
    pub const BLOCKS_IN_AN_MINUTE: u32 = (60 * 1000) / MILLISECS_PER_BLOCK as u32;
    /// Approximate number of blocks in an hour.
    pub const BLOCKS_IN_AN_HOUR: u32 = 60 * BLOCKS_IN_AN_MINUTE;
    /// Approximate number of blocks in a day.
    pub const BLOCKS_IN_A_DAY: u32 = 24 * BLOCKS_IN_AN_HOUR;
}

#[derive(Copy, Clone, Encode, Decode, TypeInfo, Serialize, Deserialize, MaxEncodedLen, Debug)]
pub struct CouncilDemocracyConfigParams<BlockNumber> {
    /// Council motion duration.
    pub council_motion_duration: BlockNumber,
    /// Democracy cooloff period.
    pub democracy_cooloff_period: BlockNumber,
    /// Democracy enactment period.
    pub democracy_enactment_period: BlockNumber,
    /// Fast track voting period.
    pub democracy_fast_track_voting_period: BlockNumber,
    /// Launch period.
    pub democracy_launch_period: BlockNumber,
    /// Vote locking period.
    pub democracy_vote_locking_period: BlockNumber,
    /// Voting period.
    pub democracy_voting_period: BlockNumber,
}

impl<BlockNumber: From<u32>> Default for CouncilDemocracyConfigParams<BlockNumber> {
    fn default() -> Self {
        Self {
            council_motion_duration: BLOCKS_IN_A_DAY.into(),
            democracy_cooloff_period: BLOCKS_IN_A_DAY.into(),
            democracy_enactment_period: BLOCKS_IN_A_DAY.into(),
            democracy_fast_track_voting_period: (2 * BLOCKS_IN_A_DAY).into(),
            democracy_launch_period: (2 * BLOCKS_IN_A_DAY).into(),
            democracy_vote_locking_period: BLOCKS_IN_A_DAY.into(),
            democracy_voting_period: BLOCKS_IN_A_DAY.into(),
        }
    }
}

impl<BlockNumber: From<u32>> CouncilDemocracyConfigParams<BlockNumber> {
    /// Production params for Council democracy config.
    pub fn production_params() -> Self {
        Self::default()
    }

    /// Fast period params for Council democracy config.
    pub fn fast_params() -> Self {
        Self {
            council_motion_duration: (15 * BLOCKS_IN_AN_MINUTE).into(),
            democracy_cooloff_period: (5 * BLOCKS_IN_AN_MINUTE).into(),
            democracy_enactment_period: (15 * BLOCKS_IN_AN_MINUTE).into(),
            democracy_fast_track_voting_period: (5 * BLOCKS_IN_AN_MINUTE).into(),
            democracy_launch_period: (15 * BLOCKS_IN_AN_MINUTE).into(),
            democracy_vote_locking_period: BLOCKS_IN_AN_MINUTE.into(),
            democracy_voting_period: (15 * BLOCKS_IN_AN_MINUTE).into(),
        }
    }
}

/// A trait for determining whether rewards are enabled or not
pub trait RewardsEnabled {
    /// Determine whether rewards are enabled or not
    fn rewards_enabled() -> bool;
}

/// A trait for finding the address for a block reward based on the `PreRuntime` digests contained within it.
pub trait FindBlockRewardAddress<RewardAddress> {
    /// Find the address for a block rewards based on the pre-runtime digests.
    fn find_block_reward_address() -> Option<RewardAddress>;
}

/// A trait for finding the addresses for voting reward based on transactions found in the block.
pub trait FindVotingRewardAddresses<RewardAddress> {
    /// Find the addresses for voting rewards based on transactions found in the block.
    fn find_voting_reward_addresses() -> Vec<RewardAddress>;
}

pub trait StorageFee<Balance> {
    /// Return the consensus transaction byte fee.
    fn transaction_byte_fee() -> Balance;

    /// Note the charged storage fee.
    fn note_storage_fees(fee: Balance);
}

parameter_types! {
    /// The portion of the `NORMAL_DISPATCH_RATIO` that we adjust the fees with. Blocks filled less
    /// than this will decrease the weight and more will increase.
    pub const TargetBlockFullness: Perquintill = Perquintill::from_percent(50);
    /// The adjustment variable of the runtime. Higher values will cause `TargetBlockFullness` to
    /// change the fees more rapidly.
    pub AdjustmentVariable: Multiplier = Multiplier::saturating_from_rational(75, 1_000_000);
    /// Minimum amount of the multiplier. This value cannot be too low. A test case should ensure
    /// that combined with `AdjustmentVariable`, we can recover from the minimum.
    /// See `multiplier_can_grow_from_zero`.
    pub MinimumMultiplier: Multiplier = Multiplier::saturating_from_rational(1, 10u128);
    /// The maximum amount of the multiplier.
    pub MaximumMultiplier: Multiplier = Bounded::max_value();
}

/// Parameterized slow adjusting fee updated based on
/// <https://research.web3.foundation/Polkadot/overview/token-economics#2-slow-adjusting-mechanism>
pub type SlowAdjustingFeeUpdate<R> = TargetedFeeAdjustment<
    R,
    TargetBlockFullness,
    AdjustmentVariable,
    MinimumMultiplier,
    MaximumMultiplier,
>;

#[derive(Encode, Decode, TypeInfo)]
pub struct BlockTransactionByteFee<Balance: Codec> {
    // The value of `transaction_byte_fee` for the current block
    pub current: Balance,
    // The value of `transaction_byte_fee` for the next block
    pub next: Balance,
}

impl<Balance: Codec + tokens::Balance> Default for BlockTransactionByteFee<Balance> {
    fn default() -> Self {
        BlockTransactionByteFee {
            current: Balance::max_value(),
            next: Balance::max_value(),
        }
    }
}

#[derive(
    PartialEq, Eq, Clone, Encode, Decode, TypeInfo, MaxEncodedLen, Ord, PartialOrd, Copy, Debug,
)]
pub enum HoldIdentifier {
    DomainStaking,
    DomainInstantiation,
    DomainStorageFund,
    MessengerChannel,
    Preimage,
}

#[cfg(feature = "testing")]
pub mod tests_utils {
    use frame_support::dispatch::DispatchClass;
    use frame_support::weights::Weight;
    use frame_system::limits::BlockWeights;
    use pallet_transaction_payment::{Multiplier, MultiplierUpdate};
    use sp_runtime::traits::{Convert, Get};
    use sp_runtime::BuildStorage;
    use std::marker::PhantomData;

    pub struct FeeMultiplierUtils<Runtime, BlockWeightsGetter>(
        PhantomData<(Runtime, BlockWeightsGetter)>,
    );

    impl<Runtime, BlockWeightsGetter> FeeMultiplierUtils<Runtime, BlockWeightsGetter>
    where
        Runtime: frame_system::Config + pallet_transaction_payment::Config,
        BlockWeightsGetter: Get<BlockWeights>,
    {
        fn max_normal() -> Weight {
            let block_weights = BlockWeightsGetter::get();
            block_weights
                .get(DispatchClass::Normal)
                .max_total
                .unwrap_or(block_weights.max_block)
        }

        fn min_multiplier() -> Multiplier {
            <Runtime as pallet_transaction_payment::Config>::FeeMultiplierUpdate::min()
        }

        fn target() -> Weight {
            <Runtime as pallet_transaction_payment::Config>::FeeMultiplierUpdate::target()
                * Self::max_normal()
        }

        // update based on runtime impl.
        fn runtime_multiplier_update(fm: Multiplier) -> Multiplier {
            <Runtime as pallet_transaction_payment::Config>::FeeMultiplierUpdate::convert(fm)
        }

        fn run_with_system_weight<F>(w: Weight, assertions: F)
        where
            F: Fn(),
        {
            let mut t: sp_io::TestExternalities = frame_system::GenesisConfig::<Runtime>::default()
                .build_storage()
                .unwrap()
                .into();
            t.execute_with(|| {
                frame_system::Pallet::<Runtime>::set_block_consumed_resources(w, 0);
                assertions()
            });
        }

        // The following function is taken from test with same name from
        // https://github.com/paritytech/polkadot-sdk/blob/91851951856b8effe627fb1d151fe336a51eef2d/substrate/bin/node/runtime/src/impls.rs#L234
        // with some small surface changes.
        pub fn multiplier_can_grow_from_zero()
        where
            Runtime: pallet_transaction_payment::Config,
            BlockWeightsGetter: Get<BlockWeights>,
        {
            // if the min is too small, then this will not change, and we are doomed forever.
            // the block ref time is 1/100th bigger than target.
            Self::run_with_system_weight(
                Self::target().set_ref_time((Self::target().ref_time() / 100) * 101),
                || {
                    let next = Self::runtime_multiplier_update(Self::min_multiplier());
                    assert!(
                        next > Self::min_multiplier(),
                        "{:?} !> {:?}",
                        next,
                        Self::min_multiplier()
                    );
                },
            );

            // the block proof size is 1/100th bigger than target.
            Self::run_with_system_weight(
                Self::target().set_proof_size((Self::target().proof_size() / 100) * 101),
                || {
                    let next = Self::runtime_multiplier_update(Self::min_multiplier());
                    assert!(
                        next > Self::min_multiplier(),
                        "{:?} !> {:?}",
                        next,
                        Self::min_multiplier()
                    );
                },
            )
        }
    }
}
