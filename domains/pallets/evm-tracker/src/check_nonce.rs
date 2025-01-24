use crate::Config;
#[cfg(not(feature = "std"))]
use alloc::vec;
use codec::{Decode, Encode};
use core::cmp::max;
use core::fmt;
use core::result::Result;
use frame_support::dispatch::DispatchInfo;
use frame_support::pallet_prelude::{
    InvalidTransaction, TransactionLongevity, TransactionValidity, TransactionValidityError,
    TypeInfo, ValidTransaction,
};
use frame_support::sp_runtime::traits::{DispatchInfoOf, One, SignedExtension};
use sp_runtime::traits::{Dispatchable, Zero};
#[cfg(feature = "std")]
use std::vec;

#[derive(Encode, Decode, Clone, Eq, PartialEq, TypeInfo)]
#[scale_info(skip_type_params(T))]
pub struct CheckNonce<T: Config>(#[codec(compact)] pub T::Nonce);

impl<T: Config> CheckNonce<T> {
    /// utility constructor. Used only in client/factory code.
    pub fn from(nonce: T::Nonce) -> Self {
        Self(nonce)
    }
}

impl<T: Config> fmt::Debug for CheckNonce<T> {
    #[cfg(feature = "std")]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CheckNonce({})", self.0)
    }

    #[cfg(not(feature = "std"))]
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

impl<T: Config> SignedExtension for CheckNonce<T>
where
    T::RuntimeCall: Dispatchable<Info = DispatchInfo>,
{
    const IDENTIFIER: &'static str = "CheckNonce";
    type AccountId = T::AccountId;
    type Call = T::RuntimeCall;
    type AdditionalSigned = ();
    type Pre = ();

    fn additional_signed(&self) -> Result<(), TransactionValidityError> {
        Ok(())
    }

    fn validate(
        &self,
        who: &Self::AccountId,
        _call: &Self::Call,
        _info: &DispatchInfoOf<Self::Call>,
        _len: usize,
    ) -> TransactionValidity {
        let account = frame_system::Account::<T>::get(who);
        if account.providers.is_zero() && account.sufficients.is_zero() {
            // Nonce storage not paid for
            return InvalidTransaction::Payment.into();
        }
        if self.0 < account.nonce {
            return InvalidTransaction::Stale.into();
        }

        let provides = vec![Encode::encode(&(who, self.0))];
        let requires = if account.nonce < self.0 {
            vec![Encode::encode(&(who, self.0 - One::one()))]
        } else {
            vec![]
        };

        Ok(ValidTransaction {
            priority: 0,
            requires,
            provides,
            longevity: TransactionLongevity::MAX,
            propagate: true,
        })
    }

    fn pre_dispatch(
        self,
        who: &Self::AccountId,
        _call: &Self::Call,
        _info: &DispatchInfoOf<Self::Call>,
        _len: usize,
    ) -> Result<(), TransactionValidityError> {
        let mut account = frame_system::Account::<T>::get(who);
        if account.providers.is_zero() && account.sufficients.is_zero() {
            // Nonce storage not paid for
            return Err(InvalidTransaction::Payment.into());
        }
        // if a sender sends an evm transaction first and substrate transaction
        // after with same nonce, then reject the second transaction
        // if sender reverse the transaction types, substrate first and evm second,
        // evm transaction will be rejected, since substrate updates nonce in pre_dispatch.
        let account_nonce = if let Some(tracked_nonce) = crate::AccountNonce::<T>::get(who.clone())
        {
            max(tracked_nonce.as_u32().into(), account.nonce)
        } else {
            account.nonce
        };

        if self.0 != account_nonce {
            return Err(if self.0 < account.nonce {
                InvalidTransaction::Stale
            } else {
                InvalidTransaction::Future
            }
            .into());
        }
        account.nonce += T::Nonce::one();
        frame_system::Account::<T>::insert(who, account);
        Ok(())
    }
}
