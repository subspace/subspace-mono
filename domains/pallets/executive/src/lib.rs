// This file is part of Substrate.

// Copyright (C) 2017-2022 Parity Technologies (UK) Ltd.
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

//! # Domain Executive Module
//!
//! This module is derived from frame_executive with some custom modifications for
//! including the failed extrinsic during `pre/post_dispatch` in the block and
//! collecting the intermediate storage roots in the block execution required for
//! the fraud proof of decoupled execution in Subspace.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "runtime-benchmarks")]
mod benchmarking;
#[cfg(test)]
mod mock;
#[cfg(test)]
mod tests;
pub mod weights;

#[cfg(not(feature = "std"))]
extern crate alloc;

use codec::{Codec, Encode};
use frame_support::defensive_assert;
use frame_support::dispatch::{
    DispatchClass, DispatchErrorWithPostInfo, DispatchInfo, GetDispatchInfo, Pays, PostDispatchInfo,
};
use frame_support::storage::with_storage_layer;
use frame_support::traits::fungible::{Inspect, Mutate};
use frame_support::traits::tokens::{Fortitude, Precision, Preservation};
use frame_support::traits::{
    BeforeAllRuntimeMigrations, EnsureInherentsAreFirst, ExecuteBlock, Get, OffchainWorker,
    OnFinalize, OnIdle, OnInitialize, OnPoll, OnRuntimeUpgrade, PostTransactions,
};
use frame_support::weights::{Weight, WeightToFee};
use frame_system::pallet_prelude::*;
pub use pallet::*;
use sp_runtime::traits::{
    Applyable, Block as BlockT, CheckEqual, Checkable, Dispatchable, Header, NumberFor, One,
    ValidateUnsigned, Zero,
};
use sp_runtime::transaction_validity::{
    InvalidTransaction, TransactionSource, TransactionValidity, TransactionValidityError,
};
use sp_runtime::{ApplyExtrinsicResult, DispatchError, ExtrinsicInclusionMode};
use sp_std::marker::PhantomData;
use sp_std::prelude::*;

pub type CheckedOf<E, C> = <E as Checkable<C>>::Checked;
pub type CallOf<E, C> = <CheckedOf<E, C> as Applyable>::Call;
pub type OriginOf<E, C> = <CallOf<E, C> as Dispatchable>::RuntimeOrigin;

/// Trait trait used to charge the extrinsic storage.
pub trait ExtrinsicStorageFees<T: Config> {
    /// Extracts signer from given extrinsic and its dispatch info.
    fn extract_signer(xt: ExtrinsicOf<T>) -> (Option<AccountIdOf<T>>, DispatchInfo);
    /// Hook to note operator rewards for charged storage fees.
    fn on_storage_fees_charged(
        charged_fees: BalanceOf<T>,
        tx_size: u32,
    ) -> Result<(), TransactionValidityError>;
}

type AccountIdOf<T> = <T as frame_system::Config>::AccountId;
type BalanceOf<T> = <<T as Config>::Currency as Inspect<AccountIdOf<T>>>::Balance;
type ExtrinsicOf<T> = <BlockOf<T> as BlockT>::Extrinsic;
type BlockOf<T> = <T as frame_system::Config>::Block;
type BlockHashOf<T> = <BlockOf<T> as BlockT>::Hash;

// TODO: not store the intermediate storage root in the state but
// calculate the storage root outside the runtime after executing the extrinsic directly.
#[frame_support::pallet]
mod pallet {
    use crate::weights::WeightInfo;
    use crate::{BalanceOf, ExtrinsicStorageFees};
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    use frame_support::pallet_prelude::*;
    use frame_support::traits::fungible::Mutate;
    use frame_support::weights::WeightToFee;
    use frame_system::pallet_prelude::*;
    use frame_system::SetCode;
    use sp_executive::{InherentError, InherentType, INHERENT_IDENTIFIER};

    #[pallet::config]
    pub trait Config: frame_system::Config {
        type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;
        type WeightInfo: WeightInfo;
        type Currency: Mutate<Self::AccountId>;
        type LengthToFee: WeightToFee<Balance = BalanceOf<Self>>;
        type ExtrinsicStorageFees: ExtrinsicStorageFees<Self>;
    }

    #[pallet::pallet]
    #[pallet::without_storage_info]
    pub struct Pallet<T>(_);

    #[pallet::call]
    impl<T: Config> Pallet<T> {
        /// Sets new runtime code after doing necessary checks.
        /// Same as frame_system::Call::set_code but without root origin.
        #[pallet::call_index(0)]
        #[pallet::weight((T::WeightInfo::set_code(), DispatchClass::Mandatory))]
        pub fn set_code(origin: OriginFor<T>, code: Vec<u8>) -> DispatchResult {
            ensure_none(origin)?;
            <frame_system::pallet::Pallet<T>>::can_set_code(&code)?;
            <T as frame_system::Config>::OnSetCode::set_code(code)?;
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
                .expect("Executive inherent data not correctly encoded")
                .expect("Executive inherent data must be provided");

            inherent_data.maybe_code.map(|code| Call::set_code { code })
        }

        fn is_inherent_required(data: &InherentData) -> Result<Option<Self::Error>, Self::Error> {
            let inherent_data = data
                .get_data::<InherentType>(&INHERENT_IDENTIFIER)
                .expect("Executive inherent data not correctly encoded")
                .expect("Executive inherent data must be provided");

            Ok(if inherent_data.maybe_code.is_none() {
                None
            } else {
                Some(InherentError::MissingRuntimeCode)
            })
        }

        fn check_inherent(call: &Self::Call, data: &InherentData) -> Result<(), Self::Error> {
            let inherent_data = data
                .get_data::<InherentType>(&INHERENT_IDENTIFIER)
                .expect("Executive inherent data not correctly encoded")
                .expect("Executive inherent data must be provided");

            if let Some(provided_code) = inherent_data.maybe_code {
                if let Call::set_code { code } = call {
                    if code != &provided_code {
                        return Err(InherentError::IncorrectRuntimeCode);
                    }
                }
            } else {
                return Err(InherentError::MissingRuntimeCode);
            }

            Ok(())
        }

        fn is_inherent(call: &Self::Call) -> bool {
            matches!(call, Call::set_code { .. })
        }
    }

    #[pallet::event]
    pub enum Event<T: Config> {}

    #[pallet::hooks]
    impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
        fn on_initialize(_block_number: BlockNumberFor<T>) -> Weight {
            // Reset the intermediate storage roots from last block.
            IntermediateRoots::<T>::kill();
            // TODO: Probably needs a different value
            Weight::from_parts(1, 0)
        }
    }

    /// Intermediate storage roots collected during the block execution.
    #[pallet::storage]
    #[pallet::getter(fn intermediate_roots)]
    pub(super) type IntermediateRoots<T: Config> = StorageValue<_, Vec<[u8; 32]>, ValueQuery>;
}

impl<T: Config> Pallet<T> {
    pub(crate) fn push_root(root: Vec<u8>) {
        IntermediateRoots::<T>::append(
            TryInto::<[u8; 32]>::try_into(root)
                .expect("root is a SCALE encoded hash which uses H256; qed"),
        );
    }
}

/// Same semantics with `frame_executive::Executive`.
///
/// One extra generic parameter:
/// - `ExecutiveConfig`: Something that implements `domain_pallet_executive::Config`.
pub struct Executive<
    ExecutiveConfig,
    Context,
    UnsignedValidator,
    AllPalletsWithSystem,
    OnRuntimeUpgrade = (),
>(
    PhantomData<(
        ExecutiveConfig,
        Context,
        UnsignedValidator,
        AllPalletsWithSystem,
        OnRuntimeUpgrade,
    )>,
);

impl<
        ExecutiveConfig: Config + frame_system::Config + EnsureInherentsAreFirst<BlockOf<ExecutiveConfig>>,
        Context: Default,
        UnsignedValidator,
        AllPalletsWithSystem: OnRuntimeUpgrade
            + BeforeAllRuntimeMigrations
            + OnInitialize<BlockNumberFor<ExecutiveConfig>>
            + OnIdle<BlockNumberFor<ExecutiveConfig>>
            + OnFinalize<BlockNumberFor<ExecutiveConfig>>
            + OffchainWorker<BlockNumberFor<ExecutiveConfig>>
            + OnPoll<BlockNumberFor<ExecutiveConfig>>,
        COnRuntimeUpgrade: OnRuntimeUpgrade,
    > ExecuteBlock<BlockOf<ExecutiveConfig>>
    for Executive<
        ExecutiveConfig,
        Context,
        UnsignedValidator,
        AllPalletsWithSystem,
        COnRuntimeUpgrade,
    >
where
    ExtrinsicOf<ExecutiveConfig>: Checkable<Context> + Codec,
    CheckedOf<ExtrinsicOf<ExecutiveConfig>, Context>: Applyable + GetDispatchInfo,
    CallOf<ExtrinsicOf<ExecutiveConfig>, Context>:
        Dispatchable<Info = DispatchInfo, PostInfo = PostDispatchInfo>,
    OriginOf<ExtrinsicOf<ExecutiveConfig>, Context>: From<Option<AccountIdOf<ExecutiveConfig>>>,
    UnsignedValidator: ValidateUnsigned<Call = CallOf<ExtrinsicOf<ExecutiveConfig>, Context>>,
{
    fn execute_block(block: BlockOf<ExecutiveConfig>) {
        Executive::<
            ExecutiveConfig,
            Context,
            UnsignedValidator,
            AllPalletsWithSystem,
            COnRuntimeUpgrade,
        >::execute_block(block);
    }
}

impl<
        ExecutiveConfig: Config + frame_system::Config + EnsureInherentsAreFirst<BlockOf<ExecutiveConfig>>,
        Context: Default,
        UnsignedValidator,
        AllPalletsWithSystem: OnRuntimeUpgrade
            + BeforeAllRuntimeMigrations
            + OnInitialize<BlockNumberFor<ExecutiveConfig>>
            + OnIdle<BlockNumberFor<ExecutiveConfig>>
            + OnFinalize<BlockNumberFor<ExecutiveConfig>>
            + OffchainWorker<BlockNumberFor<ExecutiveConfig>>
            + OnPoll<BlockNumberFor<ExecutiveConfig>>,
        COnRuntimeUpgrade: OnRuntimeUpgrade,
    >
    Executive<ExecutiveConfig, Context, UnsignedValidator, AllPalletsWithSystem, COnRuntimeUpgrade>
where
    ExtrinsicOf<ExecutiveConfig>: Checkable<Context> + Codec,
    CheckedOf<ExtrinsicOf<ExecutiveConfig>, Context>: Applyable + GetDispatchInfo,
    CallOf<ExtrinsicOf<ExecutiveConfig>, Context>:
        Dispatchable<Info = DispatchInfo, PostInfo = PostDispatchInfo>,
    OriginOf<ExtrinsicOf<ExecutiveConfig>, Context>: From<Option<AccountIdOf<ExecutiveConfig>>>,
    UnsignedValidator: ValidateUnsigned<Call = CallOf<ExtrinsicOf<ExecutiveConfig>, Context>>,
{
    /// Returns the latest storage root.
    pub fn storage_root() -> Vec<u8> {
        let version = <ExecutiveConfig as frame_system::Config>::Version::get().state_version();
        sp_io::storage::root(version)
        // [1u8; 32].into()
    }

    /// Wrapped `frame_executive::Executive::execute_on_runtime_upgrade`.
    pub fn execute_on_runtime_upgrade() -> Weight {
        frame_executive::Executive::<
            ExecutiveConfig,
            BlockOf<ExecutiveConfig>,
            Context,
            UnsignedValidator,
            AllPalletsWithSystem,
            COnRuntimeUpgrade,
        >::execute_on_runtime_upgrade()
    }

    /// Wrapped `frame_executive::Executive::initialize_block`.
    ///
    /// Note the storage root in the end.
    pub fn initialize_block(header: &HeaderFor<ExecutiveConfig>) -> ExtrinsicInclusionMode {
        frame_executive::Executive::<
            ExecutiveConfig,
            BlockOf<ExecutiveConfig>,
            Context,
            UnsignedValidator,
            AllPalletsWithSystem,
            COnRuntimeUpgrade,
        >::initialize_block(header)
    }

    // TODO: https://github.com/paritytech/substrate/issues/10711
    fn initial_checks(block: &BlockOf<ExecutiveConfig>) -> u32 {
        sp_tracing::enter_span!(sp_tracing::Level::TRACE, "initial_checks");
        let header = block.header();

        // Check that `parent_hash` is correct.
        let n = *header.number();
        assert!(
            n > BlockNumberFor::<ExecutiveConfig>::zero()
                && <frame_system::Pallet<ExecutiveConfig>>::block_hash(
                    n - BlockNumberFor::<ExecutiveConfig>::one()
                ) == *header.parent_hash(),
            "Parent hash should be valid.",
        );

        match ExecutiveConfig::ensure_inherents_are_first(block) {
            Ok(num) => num,
            Err(i) => panic!("Invalid inherent position for extrinsic at index {}", i),
        }
    }

    /// Wrapped `frame_executive::Executive::execute_block`.
    ///
    /// The purpose is to use our custom [`Executive::apply_extrinsic`] and
    /// the [`Executive::finalize_block`] logic.
    pub fn execute_block(block: BlockOf<ExecutiveConfig>) {
        sp_io::init_tracing();
        sp_tracing::within_span! {
            sp_tracing::info_span!("execute_block", ?block);
            // Execute `on_runtime_upgrade` and `on_initialize`.
            let mode = Self::initialize_block(block.header());
            let num_inherents = Self::initial_checks(&block) as usize;
            let (header, extrinsics) = block.deconstruct();
            let num_extrinsics = extrinsics.len();

            if mode == ExtrinsicInclusionMode::OnlyInherents && num_extrinsics > num_inherents {
                // Invalid block
                panic!("Only inherents are allowed in this block")
            }

            Self::apply_extrinsics(extrinsics.into_iter());

            // In this case there were no transactions to trigger this state transition:
            if !<frame_system::Pallet<ExecutiveConfig>>::inherents_applied() {
                defensive_assert!(num_inherents == num_extrinsics);
                frame_executive::Executive::<
                    ExecutiveConfig,
                    BlockOf<ExecutiveConfig>,
                    Context,
                    UnsignedValidator,
                    AllPalletsWithSystem,
                    COnRuntimeUpgrade,
                >::inherents_applied();
            }

            <frame_system::Pallet<ExecutiveConfig>>::note_finished_extrinsics();
            <ExecutiveConfig as frame_system::Config>::PostTransactions::post_transactions();

            Self::idle_and_finalize_hook(*header.number());
            Self::final_checks(&header);
        }
    }

    /// Wrapped `frame_executive::Executive::apply_extrinsics`.
    fn apply_extrinsics(extrinsics: impl Iterator<Item = ExtrinsicOf<ExecutiveConfig>>) {
        extrinsics.into_iter().for_each(|e| {
            if let Err(e) = Self::apply_extrinsic(e) {
                let err: &'static str = e.into();
                panic!("{}", err)
            }
        });

        // Note the storage root before finalizing the block so that the block imported during the
        // syncing process produces the same storage root with the one processed based on
        // the consensus block.
        Pallet::<ExecutiveConfig>::push_root(Self::storage_root());
    }

    /// Wrapped `frame_executive::Executive::finalize_block`.
    pub fn finalize_block() -> HeaderFor<ExecutiveConfig> {
        Pallet::<ExecutiveConfig>::push_root(Self::storage_root());
        frame_executive::Executive::<
            ExecutiveConfig,
            BlockOf<ExecutiveConfig>,
            Context,
            UnsignedValidator,
            AllPalletsWithSystem,
            COnRuntimeUpgrade,
        >::finalize_block()
    }

    /// Wrapped `frame_executive::Executive::on_idle_hook` and `frame_executive::Executive::on_finalize_hook`
    fn idle_and_finalize_hook(block_number: NumberFor<BlockOf<ExecutiveConfig>>) {
        let weight = <frame_system::Pallet<ExecutiveConfig>>::block_weight();
        let max_weight = <<ExecutiveConfig as frame_system::Config>::BlockWeights as frame_support::traits::Get<_>>::get().max_block;
        let remaining_weight = max_weight.saturating_sub(weight.total());

        if remaining_weight.all_gt(Weight::zero()) {
            let used_weight = AllPalletsWithSystem::on_idle(block_number, remaining_weight);
            <frame_system::Pallet<ExecutiveConfig>>::register_extra_weight_unchecked(
                used_weight,
                DispatchClass::Mandatory,
            );
        }

        AllPalletsWithSystem::on_finalize(block_number);
    }

    /// Wrapped `frame_executive::Executive::apply_extrinsic`.
    ///
    /// Note the storage root in the beginning.
    pub fn apply_extrinsic(uxt: ExtrinsicOf<ExecutiveConfig>) -> ApplyExtrinsicResult {
        Pallet::<ExecutiveConfig>::push_root(Self::storage_root());

        // apply the extrinsic within another transaction so that changes can be reverted.
        let res = with_storage_layer(|| {
            frame_executive::Executive::<
                ExecutiveConfig,
                BlockOf<ExecutiveConfig>,
                Context,
                UnsignedValidator,
                AllPalletsWithSystem,
                COnRuntimeUpgrade,
            >::apply_extrinsic(uxt.clone())
            .map_err(|err| DispatchError::Other(err.into()))
        });

        // apply extrinsic failed with transaction validity error
        // this could happen for following scenarios
        // - Bad extrinsic Signature
        //      This extrinsic will be ignored by the operators during the bundle check
        //      and marks such bundle as Invalid.
        // - Extrinsic execution failed
        //      There are multiple scenarios why this can happen
        //      - Inherent extrinsic failed. If this happens, we should fail to apply extrinsic
        //      - Pre and Post dispatch fails. Check the test `test_domain_block_builder_include_ext_with_failed_predispatch`
        //        why this could happen. If it fail due to this, then we revert the inner storage changes
        //        but still include extrinsic so that we can clear inconsistency between block body and trace roots.
        let res = match res {
            Ok(dispatch_outcome) => Ok(dispatch_outcome),
            Err(err) => {
                let encoded = uxt.encode();
                let (maybe_signer, dispatch_info) =
                    ExecutiveConfig::ExtrinsicStorageFees::extract_signer(uxt);
                // if this is mandatory extrinsic, then transaction should not execute
                // we should fail here.
                if dispatch_info.class == DispatchClass::Mandatory {
                    return Err(TransactionValidityError::Invalid(
                        InvalidTransaction::MandatoryValidation,
                    ));
                }

                // charge signer for extrinsic storage
                if let Some(signer) = maybe_signer {
                    let storage_fees = ExecutiveConfig::LengthToFee::weight_to_fee(
                        &Weight::from_parts(encoded.len() as u64, 0),
                    );

                    // best effort to charge the fees to signer.
                    // if signer does not have enough balance, we continue
                    let maybe_charged_fees = ExecutiveConfig::Currency::burn_from(
                        &signer,
                        storage_fees,
                        Preservation::Expendable,
                        Precision::BestEffort,
                        Fortitude::Force,
                    );

                    if let Ok(charged_fees) = maybe_charged_fees {
                        ExecutiveConfig::ExtrinsicStorageFees::on_storage_fees_charged(
                            charged_fees,
                            encoded.len() as u32,
                        )?;
                    }
                }

                // note the extrinsic into system storage
                <frame_system::Pallet<ExecutiveConfig>>::note_extrinsic(encoded);

                // note extrinsic applied. it pays no fees since there was no execution.
                // we also set the weight to zero since there was no execution done.
                let r = Err(DispatchErrorWithPostInfo {
                    post_info: PostDispatchInfo {
                        actual_weight: None,
                        pays_fee: Pays::No,
                    },
                    error: err,
                });

                <frame_system::Pallet<ExecutiveConfig>>::note_applied_extrinsic(&r, dispatch_info);
                Ok(Err(err))
            }
        };

        // TODO: Critical!!! https://github.com/paritytech/substrate/pull/10922#issuecomment-1068997467
        log::debug!(
            target: "domain::runtime::executive",
            "[apply_extrinsic] after: {:?}",
            {
                use codec::Decode;
                BlockHashOf::<ExecutiveConfig>::decode(&mut Self::storage_root().as_slice()).unwrap()
            }
        );
        res
    }

    // TODO: https://github.com/paritytech/substrate/issues/10711
    fn final_checks(header: &HeaderFor<ExecutiveConfig>) {
        sp_tracing::enter_span!(sp_tracing::Level::TRACE, "final_checks");
        // remove temporaries
        let new_header = <frame_system::Pallet<ExecutiveConfig>>::finalize();

        // check digest
        assert_eq!(
            header.digest().logs().len(),
            new_header.digest().logs().len(),
            "Number of digest items must match that calculated."
        );
        let items_zip = header
            .digest()
            .logs()
            .iter()
            .zip(new_header.digest().logs().iter());
        for (header_item, computed_item) in items_zip {
            header_item.check_equal(computed_item);
            assert_eq!(
                header_item, computed_item,
                "Digest item must match that calculated."
            );
        }

        // check storage root.
        let storage_root = new_header.state_root();
        header.state_root().check_equal(storage_root);
        assert!(
            header.state_root() == storage_root,
            "Storage root must match that calculated."
        );

        assert!(
            header.extrinsics_root() == new_header.extrinsics_root(),
            "Transaction trie root must be valid.",
        );
    }

    /// Wrapped `frame_executive::Executive::validate_transaction`.
    pub fn validate_transaction(
        source: TransactionSource,
        uxt: ExtrinsicOf<ExecutiveConfig>,
        block_hash: BlockHashOf<ExecutiveConfig>,
    ) -> TransactionValidity {
        frame_executive::Executive::<
            ExecutiveConfig,
            BlockOf<ExecutiveConfig>,
            Context,
            UnsignedValidator,
            AllPalletsWithSystem,
            COnRuntimeUpgrade,
        >::validate_transaction(source, uxt, block_hash)
    }

    /// Wrapped `frame_executive::Executive::offchain_worker`.
    pub fn offchain_worker(header: &HeaderFor<ExecutiveConfig>) {
        frame_executive::Executive::<
            ExecutiveConfig,
            BlockOf<ExecutiveConfig>,
            Context,
            UnsignedValidator,
            AllPalletsWithSystem,
            COnRuntimeUpgrade,
        >::offchain_worker(header)
    }
}
