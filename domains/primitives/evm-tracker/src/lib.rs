//! Inherents for EVM tracker
#![cfg_attr(not(feature = "std"), no_std)]

use codec::{Decode, Encode};
use domain_runtime_primitives::EthereumAccountId;
use sp_domains::PermissionedActionAllowedBy;
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
    pub maybe_call: Option<PermissionedActionAllowedBy<EthereumAccountId>>,
}

/// Provides the set code inherent data.
#[cfg(feature = "std")]
pub struct InherentDataProvider {
    data: InherentType,
}

#[cfg(feature = "std")]
impl InherentDataProvider {
    /// Create new inherent data provider from the given `data`.
    pub fn new(maybe_call: Option<PermissionedActionAllowedBy<EthereumAccountId>>) -> Self {
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

sp_api::decl_runtime_apis! {
    /// Api to check and verify the evm-tracker extrinsic calls
    pub trait EvmTrackerApi {
        /// Returns true if evm-tracker exists in the runtime, and the decoded argument is valid.
        fn is_valid_evm_contract_creation_allowed_by_call(decoded_argument: PermissionedActionAllowedBy<EthereumAccountId>) -> bool;

        /// Returns an encoded extrinsic for domain "set contract creation allowed by" call.
        fn construct_evm_contract_creation_allowed_by_extrinsic(decoded_argument: PermissionedActionAllowedBy<EthereumAccountId>) -> Block::Extrinsic;
    }
}
