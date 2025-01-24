use crate::{Runtime, RuntimeCall, RuntimeConfigs};
use codec::{Decode, Encode};
use scale_info::TypeInfo;
use sp_runtime::traits::{DispatchInfoOf, SignedExtension};
use sp_runtime::transaction_validity::{
    InvalidTransaction, TransactionValidity, TransactionValidityError, ValidTransaction,
};
use sp_std::prelude::*;
use subspace_runtime_primitives::utility::nested_utility_call_iter;

/// Disable balance transfers, if configured in the runtime.
#[derive(Debug, Encode, Decode, Clone, Eq, PartialEq, Default, TypeInfo)]
pub struct DisablePallets;

impl SignedExtension for DisablePallets {
    const IDENTIFIER: &'static str = "DisablePallets";
    type AccountId = <Runtime as frame_system::Config>::AccountId;
    type Call = <Runtime as frame_system::Config>::RuntimeCall;
    type AdditionalSigned = ();
    type Pre = ();

    fn additional_signed(&self) -> Result<Self::AdditionalSigned, TransactionValidityError> {
        Ok(())
    }

    fn validate(
        &self,
        _who: &Self::AccountId,
        call: &Self::Call,
        _info: &DispatchInfoOf<Self::Call>,
        _len: usize,
    ) -> TransactionValidity {
        // Disable normal balance transfers.
        if !RuntimeConfigs::enable_balance_transfers() && contains_balance_transfer(call) {
            InvalidTransaction::Call.into()
        } else {
            Ok(ValidTransaction::default())
        }
    }

    fn pre_dispatch(
        self,
        who: &Self::AccountId,
        call: &Self::Call,
        info: &DispatchInfoOf<Self::Call>,
        len: usize,
    ) -> Result<Self::Pre, TransactionValidityError> {
        self.validate(who, call, info, len)?;
        Ok(())
    }

    fn validate_unsigned(
        call: &Self::Call,
        _info: &DispatchInfoOf<Self::Call>,
        _len: usize,
    ) -> TransactionValidity {
        if matches!(call, RuntimeCall::Domains(_)) && !RuntimeConfigs::enable_domains() {
            InvalidTransaction::Call.into()
        } else {
            Ok(ValidTransaction::default())
        }
    }
}

fn contains_balance_transfer(call: &RuntimeCall) -> bool {
    for call in nested_utility_call_iter::<Runtime>(call) {
        // Other calls are inconclusive, they might contain nested calls
        if let RuntimeCall::Balances(
            pallet_balances::Call::transfer_allow_death { .. }
            | pallet_balances::Call::transfer_keep_alive { .. }
            | pallet_balances::Call::transfer_all { .. },
        ) = call
        {
            return true;
        }
    }

    false
}
