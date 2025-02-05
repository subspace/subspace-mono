// Copyright (C) 2023 Subspace Labs, Inc.
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

//! Pallet EVM Account nonce tracker

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

mod check_nonce;
pub mod create_contract;
pub mod traits;

pub use check_nonce::CheckNonce;
pub use pallet::*;
use sp_core::U256;
use sp_domains::PermissionedActionAllowedBy;

#[frame_support::pallet]
mod pallet {
    use codec::Codec;
    use frame_support::pallet_prelude::*;
    use frame_system::pallet_prelude::*;
    use sp_core::U256;
    use sp_domains::PermissionedActionAllowedBy;

    #[pallet::config]
    pub trait Config: frame_system::Config {}

    /// Storage to hold evm account nonces.
    /// This is only used for pre_dispatch since EVM pre_dispatch does not
    /// increment account nonce.
    #[pallet::storage]
    pub(super) type AccountNonce<T: Config> =
        StorageMap<_, Identity, T::AccountId, U256, OptionQuery>;

    /// Storage to hold EVM contract creation allow list accounts.
    /// Unlike domain instantiation, no storage value means "anyone can create contracts".
    ///
    /// At genesis, this is set via DomainConfigParams, DomainRuntimeInfo::Evm, and
    /// RawGenesis::set_evm_contract_creation_allowed_by().
    // When this type name is changed, evm_contract_creation_allowed_by_storage_key() also needs to
    // be updated.
    #[pallet::storage]
    pub(super) type ContractCreationAllowedBy<T: Config> =
        StorageValue<_, PermissionedActionAllowedBy<T::AccountId>, ValueQuery, DefaultToAnyone>;

    /// Default value for ContractCreationAllowedBy if it is not set.
    pub struct DefaultToAnyone;

    impl<AccountId: Codec + Clone> Get<PermissionedActionAllowedBy<AccountId>> for DefaultToAnyone {
        fn get() -> PermissionedActionAllowedBy<AccountId> {
            PermissionedActionAllowedBy::Anyone
        }
    }

    /// Pallet EVM account nonce tracker.
    #[pallet::pallet]
    #[pallet::without_storage_info]
    pub struct Pallet<T>(_);

    #[pallet::call]
    impl<T: Config> Pallet<T> {
        /// Replace ContractCreationAllowedBy setting in storage, as a domain sudo call.
        #[pallet::call_index(0)]
        #[pallet::weight(<T as frame_system::Config>::DbWeight::get().reads_writes(0, 1))]
        pub fn set_contract_creation_allowed_by(
            origin: OriginFor<T>,
            contract_creation_allowed_by: PermissionedActionAllowedBy<T::AccountId>,
        ) -> DispatchResult {
            ensure_root(origin)?;
            ContractCreationAllowedBy::<T>::put(contract_creation_allowed_by);
            Ok(())
        }
    }

    #[pallet::hooks]
    impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
        fn on_finalize(_now: BlockNumberFor<T>) {
            // clear the nonce storage, since we would start with updated nonce
            // during the pre_dispatch in next block
            let _ = AccountNonce::<T>::clear(u32::MAX, None);
        }
    }
}

impl<T: Config> Pallet<T> {
    /// Returns current nonce for the given account.
    pub fn account_nonce(account: T::AccountId) -> Option<U256> {
        AccountNonce::<T>::get(account)
    }

    /// Set nonce to the account.
    pub fn set_account_nonce(account: T::AccountId, nonce: U256) {
        AccountNonce::<T>::set(account, Some(nonce))
    }

    /// Returns true if the supplied account is allowed to create contracts.
    pub fn is_allowed_to_create_contracts(signer: &T::AccountId) -> bool {
        ContractCreationAllowedBy::<T>::get().is_allowed(signer)
    }

    /// Returns true if any account is allowed to create contracts.
    pub fn is_allowed_to_create_unsigned_contracts() -> bool {
        ContractCreationAllowedBy::<T>::get().is_anyone_allowed()
    }

    /// Returns the current contract creation allow list.
    /// Mainly used in tests.
    pub fn contract_creation_allowed_by() -> PermissionedActionAllowedBy<T::AccountId> {
        ContractCreationAllowedBy::<T>::get()
    }
}
