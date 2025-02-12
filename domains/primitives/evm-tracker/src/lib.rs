//! Inherents for EVM tracker
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use codec::{Decode, Encode};
#[cfg(feature = "std")]
use sp_inherents::{Error, InherentData};
use sp_inherents::{InherentIdentifier, IsFatalError};

/// Executive inherent identifier.
pub const INHERENT_IDENTIFIER: InherentIdentifier = *b"dmnevmtr";

#[derive(Debug, Encode)]
#[cfg_attr(feature = "std", derive(Decode))]
pub enum InherentError {
    MissingRuntimeCall,
    InvalidRuntimeCall,
    IncorrectRuntimeCall,
}

impl IsFatalError for InherentError {
    fn is_fatal_error(&self) -> bool {
        true
    }
}

/// The type of the Subspace inherent data.
#[derive(Debug, Encode, Decode)]
pub struct InherentType {
    /// EVM tracker "set contract creation allowed by" call
    pub maybe_call: Option<Vec<u8>>,
}

/// Provides the set code inherent data.
#[cfg(feature = "std")]
pub struct InherentDataProvider {
    data: InherentType,
}

#[cfg(feature = "std")]
impl InherentDataProvider {
    /// Create new inherent data provider from the given `data`.
    pub fn new(maybe_call: Option<Vec<u8>>) -> Self {
        Self {
            data: InherentType { maybe_call },
        }
    }

    /// Returns the `data` of this inherent data provider.
    pub fn data(&self) -> &InherentType {
        &self.data
    }
}

#[cfg(feature = "std")]
#[async_trait::async_trait]
impl sp_inherents::InherentDataProvider for InherentDataProvider {
    async fn provide_inherent_data(&self, inherent_data: &mut InherentData) -> Result<(), Error> {
        inherent_data.put_data(INHERENT_IDENTIFIER, &self.data)
    }

    async fn try_handle_error(
        &self,
        identifier: &InherentIdentifier,
        error: &[u8],
    ) -> Option<Result<(), Error>> {
        if *identifier != INHERENT_IDENTIFIER {
            return None;
        }

        let error = InherentError::decode(&mut &*error).ok()?;

        Some(Err(Error::Application(Box::from(format!("{error:?}")))))
    }
}

/// Trait to convert Unchecked extrinsic into a Pallet-specific call
pub trait IntoEvmTrackerCall<Call> {
    fn into_evm_tracker_call(call: Vec<u8>) -> Call;
}

sp_api::decl_runtime_apis! {
    /// Api to check and verify the evm-tracker extrinsic calls
    pub trait EvmTrackerApi {
        /// Returns true if evm-tracker exists in the runtime, and extrinsic is valid.
        fn is_valid_evm_contract_creation_allowed_by_call(extrinsic: Vec<u8>) -> bool;

        /// Returns an encoded extrinsic for domain "set contract creation allowed by" call.
        fn construct_evm_contract_creation_allowed_by_extrinsic(extrinsic: Vec<u8>) -> Block::Extrinsic;
    }
}
