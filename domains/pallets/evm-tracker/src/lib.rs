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
use domain_runtime_primitives::EthereumAccountId;
pub use pallet::*;
use sp_core::U256;
use sp_domains::PermissionedActionAllowedBy;

#[frame_support::pallet]
mod pallet {
    use domain_runtime_primitives::EthereumAccountId;
    use frame_support::pallet_prelude::*;
    use frame_system::pallet_prelude::*;
    use sp_core::U256;
    use sp_domains::PermissionedActionAllowedBy;
    use sp_evm_tracker::{InherentError, InherentType, INHERENT_IDENTIFIER};

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
    pub(super) type ContractCreationAllowedBy<T: Config> = StorageValue<
        _,
        PermissionedActionAllowedBy<EthereumAccountId>,
        ValueQuery,
        DefaultToAnyone,
    >;

    /// Default value for ContractCreationAllowedBy if it is not set.
    pub struct DefaultToAnyone;

    impl Get<PermissionedActionAllowedBy<EthereumAccountId>> for DefaultToAnyone {
        fn get() -> PermissionedActionAllowedBy<EthereumAccountId> {
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
            contract_creation_allowed_by: PermissionedActionAllowedBy<EthereumAccountId>,
        ) -> DispatchResult {
            ensure_root(origin)?;
            ContractCreationAllowedBy::<T>::put(contract_creation_allowed_by);
            Ok(())
        }

        /// An inherent call to set ContractCreationAllowedBy.
        #[pallet::call_index(1)]
        #[pallet::weight((T::DbWeight::get().reads_writes(1, 1), DispatchClass::Mandatory))]
        pub fn inherent_set_contract_creation_allowed_by(
            origin: OriginFor<T>,
            contract_creation_allowed_by: PermissionedActionAllowedBy<EthereumAccountId>,
        ) -> DispatchResult {
            ensure_none(origin)?;
            ContractCreationAllowedBy::<T>::put(contract_creation_allowed_by);
            Ok(())
        }
    }

    #[pallet::inherent]
    impl<T: Config> ProvideInherent for Pallet<T> {
        type Call = Call<T>;
        type Error = InherentError;
        const INHERENT_IDENTIFIER: InherentIdentifier = INHERENT_IDENTIFIER;

        fn create_inherent(data: &InherentData) -> Option<Self::Call> {
            let inherent_data = data
                .get_data::<InherentType>(&INHERENT_IDENTIFIER)
                .expect("EVM tracker inherent data not correctly encoded")
                .expect("EVM tracker inherent data must be provided");

            inherent_data
                .maybe_call
                .map(|contract_creation_allowed_by| {
                    Call::inherent_set_contract_creation_allowed_by {
                        contract_creation_allowed_by,
                    }
                })
        }

        fn is_inherent_required(data: &InherentData) -> Result<Option<Self::Error>, Self::Error> {
            let inherent_data = data
                .get_data::<InherentType>(&INHERENT_IDENTIFIER)
                .expect("EVM tracker inherent data not correctly encoded")
                .expect("EVM tracker inherent data must be provided");

            Ok(inherent_data
                .maybe_call
                .map(|_encoded_call| InherentError::MissingRuntimeCall))
        }

        fn check_inherent(call: &Self::Call, data: &InherentData) -> Result<(), Self::Error> {
            let maybe_provided_call = Self::create_inherent(data);

            if let Some(provided_call) = maybe_provided_call {
                if Self::is_inherent(call) && call != &provided_call {
                    return Err(InherentError::IncorrectRuntimeCall);
                }
            } else {
                return Err(InherentError::MissingRuntimeCall);
            }

            Ok(())
        }

        fn is_inherent(call: &Self::Call) -> bool {
            matches!(call, Call::inherent_set_contract_creation_allowed_by { .. })
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
    pub fn is_allowed_to_create_contracts(signer: &EthereumAccountId) -> bool {
        ContractCreationAllowedBy::<T>::get().is_allowed(signer)
    }

    /// Returns true if any account is allowed to create contracts.
    pub fn is_allowed_to_create_unsigned_contracts() -> bool {
        ContractCreationAllowedBy::<T>::get().is_anyone_allowed()
    }

    /// Returns the current contract creation allow list.
    /// Mainly used in tests.
    pub fn contract_creation_allowed_by() -> PermissionedActionAllowedBy<EthereumAccountId> {
        ContractCreationAllowedBy::<T>::get()
    }
}
