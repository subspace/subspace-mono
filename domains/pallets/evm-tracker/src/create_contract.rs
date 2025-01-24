//! Contract creation allow list implementations

use crate::traits::{AccountIdFor, MaybeIntoEthCall, MaybeIntoEvmCall};
use codec::{Decode, Encode};
use domain_runtime_primitives::{EthereumAccountId, ERR_CONTRACT_CREATION_NOT_ALLOWED};
use frame_support::pallet_prelude::{PhantomData, TypeInfo};
use frame_system::pallet_prelude::{OriginFor, RuntimeCallFor};
use pallet_ethereum::{Transaction as EthereumTransaction, TransactionAction};
use scale_info::prelude::fmt;
use sp_runtime::traits::{DispatchInfoOf, SignedExtension};
use sp_runtime::transaction_validity::{
    InvalidTransaction, TransactionValidity, TransactionValidityError, ValidTransaction,
};
use subspace_runtime_primitives::utility::{nested_utility_call_iter, MaybeIntoUtilityCall};

/// Rejects contracts that can't be created under the current allow list.
/// Returns false if the call is a contract call, and the account is *not* allowed to call it.
/// Otherwise, returns true.
pub fn is_create_contract_allowed<Runtime>(
    call: &RuntimeCallFor<Runtime>,
    signer: &EthereumAccountId,
) -> bool
where
    Runtime: frame_system::Config<AccountId = EthereumAccountId>
        + pallet_ethereum::Config
        + pallet_evm::Config
        + pallet_utility::Config
        + crate::Config,
    RuntimeCallFor<Runtime>:
        MaybeIntoEthCall<Runtime> + MaybeIntoEvmCall<Runtime> + MaybeIntoUtilityCall<Runtime>,
    for<'block> &'block RuntimeCallFor<Runtime>:
        From<&'block <Runtime as pallet_utility::Config>::RuntimeCall>,
    Result<pallet_ethereum::RawOrigin, OriginFor<Runtime>>: From<OriginFor<Runtime>>,
{
    // Only enter allocating code if this account can't create contracts
    if !crate::Pallet::<Runtime>::is_allowed_to_create_contracts(signer)
        && is_create_contract::<Runtime>(call)
    {
        return false;
    }

    // If it's not a contract call, or the account is allowed to create contracts, return true.
    true
}

/// If anyone is allowed to create contracts, allows contracts. Otherwise, rejects contracts.
/// Returns false if the call is a contract call, and there is a specific (possibly empty) allow
/// list. Otherwise, returns true.
pub fn is_create_unsigned_contract_allowed<Runtime>(call: &RuntimeCallFor<Runtime>) -> bool
where
    Runtime: frame_system::Config
        + pallet_ethereum::Config
        + pallet_evm::Config
        + pallet_utility::Config
        + crate::Config,
    RuntimeCallFor<Runtime>:
        MaybeIntoEthCall<Runtime> + MaybeIntoEvmCall<Runtime> + MaybeIntoUtilityCall<Runtime>,
    for<'block> &'block RuntimeCallFor<Runtime>:
        From<&'block <Runtime as pallet_utility::Config>::RuntimeCall>,
    Result<pallet_ethereum::RawOrigin, OriginFor<Runtime>>: From<OriginFor<Runtime>>,
{
    // Only enter allocating code if unsigned contracts can't be created
    if !crate::Pallet::<Runtime>::is_allowed_to_create_unsigned_contracts()
        && is_create_contract::<Runtime>(call)
    {
        return false;
    }

    // If it's not a contract call, or anyone is allowed to create contracts, return true.
    true
}

/// Returns true if the call is a contract creation call.
pub fn is_create_contract<Runtime>(call: &RuntimeCallFor<Runtime>) -> bool
where
    Runtime: frame_system::Config
        + pallet_ethereum::Config
        + pallet_evm::Config
        + pallet_utility::Config,
    RuntimeCallFor<Runtime>:
        MaybeIntoEthCall<Runtime> + MaybeIntoEvmCall<Runtime> + MaybeIntoUtilityCall<Runtime>,
    for<'block> &'block RuntimeCallFor<Runtime>:
        From<&'block <Runtime as pallet_utility::Config>::RuntimeCall>,
    Result<pallet_ethereum::RawOrigin, OriginFor<Runtime>>: From<OriginFor<Runtime>>,
{
    for call in nested_utility_call_iter::<Runtime>(call) {
        if let Some(call) = call.maybe_into_eth_call() {
            match call {
                pallet_ethereum::Call::transact {
                    transaction: EthereumTransaction::Legacy(transaction),
                    ..
                } => {
                    if transaction.action == TransactionAction::Create {
                        return true;
                    }
                }
                pallet_ethereum::Call::transact {
                    transaction: EthereumTransaction::EIP2930(transaction),
                    ..
                } => {
                    if transaction.action == TransactionAction::Create {
                        return true;
                    }
                }
                pallet_ethereum::Call::transact {
                    transaction: EthereumTransaction::EIP1559(transaction),
                    ..
                } => {
                    if transaction.action == TransactionAction::Create {
                        return true;
                    }
                }
                // Inconclusive, other calls might create contracts.
                _ => {}
            }
        }

        if let Some(pallet_evm::Call::create { .. } | pallet_evm::Call::create2 { .. }) =
            call.maybe_into_evm_call()
        {
            return true;
        }
    }

    false
}

/// Reject contract creation, unless the account is in the current evm contract allow list.
#[derive(Debug, Encode, Decode, Clone, Eq, PartialEq, TypeInfo)]
pub struct CheckContractCreation<Runtime>(PhantomData<Runtime>);

impl<Runtime> CheckContractCreation<Runtime> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<Runtime> Default for CheckContractCreation<Runtime> {
    fn default() -> Self {
        Self::new()
    }
}

// Unsigned calls can't create contracts. Only pallet-evm and pallet-ethereum can create contracts.
// For pallet-evm all contracts are signed extrinsics, for pallet-ethereum there is only one
// extrinsic that is self-contained.
impl<Runtime> SignedExtension for CheckContractCreation<Runtime>
where
    Runtime: frame_system::Config<AccountId = EthereumAccountId>
        + pallet_ethereum::Config
        + pallet_evm::Config
        + pallet_utility::Config
        + crate::Config
        + scale_info::TypeInfo
        + fmt::Debug
        + Send
        + Sync,
    RuntimeCallFor<Runtime>:
        MaybeIntoEthCall<Runtime> + MaybeIntoEvmCall<Runtime> + MaybeIntoUtilityCall<Runtime>,
    for<'block> &'block RuntimeCallFor<Runtime>:
        From<&'block <Runtime as pallet_utility::Config>::RuntimeCall>,
    Result<pallet_ethereum::RawOrigin, OriginFor<Runtime>>: From<OriginFor<Runtime>>,
{
    const IDENTIFIER: &'static str = "CheckContractCreation";
    type AccountId = AccountIdFor<Runtime>;
    type Call = RuntimeCallFor<Runtime>;
    type AdditionalSigned = ();
    type Pre = ();

    fn additional_signed(&self) -> Result<Self::AdditionalSigned, TransactionValidityError> {
        Ok(())
    }

    fn validate(
        &self,
        who: &Self::AccountId,
        call: &Self::Call,
        _info: &DispatchInfoOf<Self::Call>,
        _len: usize,
    ) -> TransactionValidity {
        // Reject contract creation unless the account is in the allow list.
        if !is_create_contract_allowed::<Runtime>(call, who) {
            InvalidTransaction::Custom(ERR_CONTRACT_CREATION_NOT_ALLOWED).into()
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
        // Reject unsigned contract creation unless anyone is allowed to create them.
        if !is_create_unsigned_contract_allowed::<Runtime>(call) {
            InvalidTransaction::Custom(ERR_CONTRACT_CREATION_NOT_ALLOWED).into()
        } else {
            Ok(ValidTransaction::default())
        }
    }

    fn pre_dispatch_unsigned(
        call: &Self::Call,
        info: &DispatchInfoOf<Self::Call>,
        len: usize,
    ) -> Result<(), TransactionValidityError> {
        Self::validate_unsigned(call, info, len)?;
        Ok(())
    }
}
