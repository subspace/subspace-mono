// Copyright (C) 2021 Subspace Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Pallet for charging and re-distributing transaction fees.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(rust_2018_idioms, missing_debug_implementations)]

pub mod weights;

use codec::{Codec, Decode, Encode};
use frame_support::sp_runtime::traits::Zero;
use frame_support::sp_runtime::SaturatedConversion;
use frame_support::traits::{Currency, Get};
use frame_support::weights::Weight;
use frame_system::pallet_prelude::*;
pub use pallet::*;
use scale_info::TypeInfo;
use subspace_runtime_primitives::FindBlockRewardAddress;

type BalanceOf<T> =
    <<T as Config>::Currency as Currency<<T as frame_system::Config>::AccountId>>::Balance;

pub trait WeightInfo {
    fn on_initialize() -> Weight;
}

#[derive(Encode, Decode, TypeInfo)]
struct CollectedFees<Balance: Codec> {
    storage: Balance,
    compute: Balance,
    tips: Balance,
}

#[frame_support::pallet]
mod pallet {
    use super::{BalanceOf, CollectedFees, WeightInfo};
    use frame_support::pallet_prelude::*;
    use frame_support::traits::Currency;
    use frame_system::pallet_prelude::*;
    use subspace_runtime_primitives::{BlockTransactionByteFee, FindBlockRewardAddress};

    #[pallet::config]
    pub trait Config: frame_system::Config {
        /// `pallet-transaction-fees` events
        type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;

        /// Minimum desired number of replicas of the blockchain to be stored by the network,
        /// impacts storage fees.
        #[pallet::constant]
        type MinReplicationFactor: Get<u16>;

        /// How many credits there is in circulation.
        #[pallet::constant]
        type CreditSupply: Get<BalanceOf<Self>>;

        /// How much space there is on the network.
        #[pallet::constant]
        type TotalSpacePledged: Get<u128>;

        /// How big is the history of the blockchain in archived state (thus includes erasure
        /// coding, but not replication).
        #[pallet::constant]
        type BlockchainHistorySize: Get<u128>;

        type Currency: Currency<Self::AccountId>;

        type FindBlockRewardAddress: FindBlockRewardAddress<Self::AccountId>;

        /// Whether dynamic cost of storage should be used
        type DynamicCostOfStorage: Get<bool>;

        type WeightInfo: WeightInfo;
    }

    /// The value of `transaction_byte_fee` for both the current and the next block.
    ///
    /// The `next` value of `transaction_byte_fee` is updated at block finalization and used to
    /// validate extrinsic to be included in the next block, the value is move to `current` at
    /// block initialization and used to execute extrinsic in the current block. Together it
    /// ensure we use the same value for both validating and executing the extrinsic.
    ///
    /// NOTE: both the `current` and `next` value is set to the default `Balance::max_value` in
    /// the genesis block which means there will be no signed extrinsic included in block #1.
    #[pallet::storage]
    pub(super) type TransactionByteFee<T> =
        StorageValue<_, BlockTransactionByteFee<BalanceOf<T>>, ValueQuery>;

    /// Temporary value (cleared at block finalization) used to determine if the `transaction_byte_fee`
    /// is used to validate extrinsic or execute extrinsic.
    #[pallet::storage]
    pub(super) type IsDuringBlockExecution<T: Config> = StorageValue<_, bool, ValueQuery>;

    /// Temporary value (cleared at block finalization) which contains current block author, so we
    /// can issue fees during block finalization.
    #[pallet::storage]
    pub(super) type BlockAuthor<T: Config> = StorageValue<_, T::AccountId>;

    /// Temporary value (cleared at block finalization) which contains current block fees, so we can
    /// issue fees during block finalization.
    #[pallet::storage]
    pub(super) type CollectedBlockFees<T: Config> = StorageValue<_, CollectedFees<BalanceOf<T>>>;

    /// Pallet transaction fees for issuing fees to block authors.
    #[pallet::pallet]
    #[pallet::without_storage_info]
    pub struct Pallet<T>(_);

    /// `pallet-transaction-fees` events
    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        /// Storage fees.
        #[codec(index = 0)]
        BlockFees {
            /// Block author that received the fees.
            who: T::AccountId,
            /// Amount of collected storage fees.
            storage: BalanceOf<T>,
            /// Amount of collected compute fees.
            compute: BalanceOf<T>,
            /// Amount of collected tips.
            tips: BalanceOf<T>,
        },
        /// Fees burned due to equivocated block author or rewards not enabled.
        #[codec(index = 1)]
        BurnedBlockFees {
            /// Amount of burned storage fees.
            storage: BalanceOf<T>,
            /// Amount of burned compute fees.
            compute: BalanceOf<T>,
            /// Amount of burned tips.
            tips: BalanceOf<T>,
        },
    }

    #[pallet::hooks]
    impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T>
    where
        BalanceOf<T>: From<u8> + From<u64>,
    {
        fn on_initialize(now: BlockNumberFor<T>) -> Weight {
            Self::do_initialize(now);
            T::WeightInfo::on_initialize()
        }

        fn on_finalize(now: BlockNumberFor<T>) {
            Self::do_finalize(now);
        }
    }
}

impl<T: Config> Pallet<T>
where
    BalanceOf<T>: From<u64>,
{
    fn do_initialize(_n: BlockNumberFor<T>) {
        // Block author may equivocate, in which case they'll not be present here
        if let Some(block_author) = T::FindBlockRewardAddress::find_block_reward_address() {
            BlockAuthor::<T>::put(block_author);
        }

        CollectedBlockFees::<T>::put(CollectedFees {
            storage: BalanceOf::<T>::zero(),
            compute: BalanceOf::<T>::zero(),
            tips: BalanceOf::<T>::zero(),
        });

        // Update the `current` value to the `next`
        TransactionByteFee::<T>::mutate(|transaction_byte_fee| {
            transaction_byte_fee.current = transaction_byte_fee.next
        });
        IsDuringBlockExecution::<T>::set(true);
    }

    fn do_finalize(_n: BlockNumberFor<T>) {
        // Update the value for the next `transaction_byte_fee`
        TransactionByteFee::<T>::mutate(|transaction_byte_fee| {
            transaction_byte_fee.next = Self::calculate_transaction_byte_fee()
        });
        IsDuringBlockExecution::<T>::take();

        let collected_fees = CollectedBlockFees::<T>::take()
            .expect("`CollectedBlockFees` was set in `on_initialize`; qed");

        let total = collected_fees.storage + collected_fees.compute + collected_fees.tips;

        if !total.is_zero() {
            // Block author may equivocate, in which case they'll not be present here
            if let Some(block_author) = BlockAuthor::<T>::take() {
                let _imbalance = T::Currency::deposit_creating(&block_author, total);
                Self::deposit_event(Event::<T>::BlockFees {
                    who: block_author.clone(),
                    storage: collected_fees.storage,
                    compute: collected_fees.compute,
                    tips: collected_fees.tips,
                });
            } else {
                // If farmer equivocated, fees are burned
                let amount = collected_fees.storage + collected_fees.compute + collected_fees.tips;
                if !amount.is_zero() {
                    Self::deposit_event(Event::<T>::BurnedBlockFees {
                        storage: collected_fees.storage,
                        compute: collected_fees.compute,
                        tips: collected_fees.tips,
                    });
                }
            }
        }
    }

    /// Return the current `transaction_byte_fee` value for executing extrinsic and
    /// return the next `transaction_byte_fee` value for validating extrinsic to be
    /// included in the next block
    pub fn transaction_byte_fee() -> BalanceOf<T> {
        if !T::DynamicCostOfStorage::get() {
            return BalanceOf::<T>::from(1);
        }

        if IsDuringBlockExecution::<T>::get() {
            TransactionByteFee::<T>::get().current
        } else {
            TransactionByteFee::<T>::get().next
        }
    }

    pub fn calculate_transaction_byte_fee() -> BalanceOf<T> {
        let credit_supply = T::CreditSupply::get();

        match (T::TotalSpacePledged::get() / u128::from(T::MinReplicationFactor::get()))
            .checked_sub(T::BlockchainHistorySize::get())
        {
            Some(free_space) if free_space > 0 => {
                credit_supply / BalanceOf::<T>::saturated_from(free_space)
            }
            _ => credit_supply,
        }
    }

    pub fn note_transaction_fees(
        storage_fee: BalanceOf<T>,
        compute_fee: BalanceOf<T>,
        tip: BalanceOf<T>,
    ) {
        CollectedBlockFees::<T>::mutate(|collected_block_fees| {
            // `CollectedBlockFees` was set in `on_initialize` if it is `None` means this
            // function is called offchain (i.e. transaction validation) thus safe to skip
            if let Some(collected_block_fees) = collected_block_fees.as_mut() {
                collected_block_fees.storage += storage_fee;
                collected_block_fees.compute += compute_fee;
                collected_block_fees.tips += tip;
            }
        });
    }
}

impl<T: Config> subspace_runtime_primitives::StorageFee<BalanceOf<T>> for Pallet<T>
where
    BalanceOf<T>: From<u64>,
{
    fn transaction_byte_fee() -> BalanceOf<T> {
        Self::transaction_byte_fee()
    }

    fn note_storage_fees(storage_fee: BalanceOf<T>) {
        Self::note_transaction_fees(storage_fee, Zero::zero(), Zero::zero())
    }
}
