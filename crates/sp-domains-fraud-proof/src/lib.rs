// Copyright (C) 2022 Subspace Labs, Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Subspace fraud proof primitives for consensus chain.
#![cfg_attr(not(feature = "std"), no_std)]
// `generic_const_exprs` is an incomplete feature
#![allow(incomplete_features)]
// TODO: This feature is not actually used in this crate, but is added as a workaround for
//  https://github.com/rust-lang/rust/issues/133199
#![feature(generic_const_exprs)]
#![feature(associated_type_defaults)]

#[cfg(feature = "std")]
pub mod execution_prover;
pub mod fraud_proof;
#[cfg(feature = "std")]
mod host_functions;
mod runtime_interface;
pub mod storage_proof;
#[cfg(test)]
mod tests;
pub mod verification;

#[cfg(not(feature = "std"))]
extern crate alloc;

use crate::fraud_proof::FraudProof;
use crate::storage_proof::FraudProofStorageKeyRequest;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use codec::{Decode, Encode};
#[cfg(feature = "std")]
pub use host_functions::{
    FraudProofExtension, FraudProofHostFunctions, FraudProofHostFunctionsImpl,
};
pub use runtime_interface::fraud_proof_runtime_interface;
#[cfg(feature = "std")]
pub use runtime_interface::fraud_proof_runtime_interface::HostFunctions;
use scale_info::TypeInfo;
use sp_core::H256;
use sp_domains::DomainAllowlistUpdates;
use sp_runtime::traits::{Header as HeaderT, NumberFor};
use sp_runtime::transaction_validity::{InvalidTransaction, TransactionValidity};
use sp_runtime::OpaqueExtrinsic;
use sp_runtime_interface::pass_by;
use sp_runtime_interface::pass_by::PassBy;
use subspace_core_primitives::U256;
use subspace_runtime_primitives::{Balance, Moment};

/// Custom invalid validity code for the extrinsics in pallet-domains.
// When updating these error codes, check for clashes between:
// <https://github.com/autonomys/subspace/blob/main/domains/primitives/runtime/src/lib.rs#L85-L88>
// <https://github.com/autonomys/subspace/blob/main/domains/pallets/messenger/src/lib.rs#L49-L53>
#[repr(u8)]
pub enum InvalidTransactionCode {
    TransactionProof = 101,
    ExecutionReceipt = 102,
    Bundle = 103,
    FraudProof = 104,
    BundleStorageFeePayment = 105,
}

impl From<InvalidTransactionCode> for InvalidTransaction {
    #[inline]
    fn from(invalid_code: InvalidTransactionCode) -> Self {
        InvalidTransaction::Custom(invalid_code as u8)
    }
}

impl From<InvalidTransactionCode> for TransactionValidity {
    #[inline]
    fn from(invalid_code: InvalidTransactionCode) -> Self {
        InvalidTransaction::Custom(invalid_code as u8).into()
    }
}

/// Type that specifies the request of storage keys
#[derive(Debug, Decode, Encode, TypeInfo, PartialEq, Eq, Clone)]
pub enum StorageKeyRequest {
    /// Domain's transfers storage key
    Transfers,
}

/// Type that maybe holds an encoded set_code extrinsic with upgraded runtime
#[derive(Debug, Decode, Encode, TypeInfo, PartialEq, Eq, Clone)]
pub enum SetCodeExtrinsic {
    /// No runtime upgrade.
    None,
    /// Holds an encoded set_code extrinsic with an upgraded runtime.
    EncodedExtrinsic(Vec<u8>),
}

/// Type that maybe holds an encoded update domain chain allowlist extrinsic
#[derive(Debug, Decode, Encode, TypeInfo, PartialEq, Eq, Clone)]
pub enum DomainChainAllowlistUpdateExtrinsic {
    /// No updates
    None,
    /// Holds an encoded extrinsic with updates.
    EncodedExtrinsic(Vec<u8>),
}

#[derive(Debug, Decode, Encode, TypeInfo, PartialEq, Eq, Clone)]
pub struct DomainInherentExtrinsicData {
    pub timestamp: Moment,
    pub maybe_domain_runtime_upgrade: Option<Vec<u8>>,
    pub consensus_transaction_byte_fee: Balance,
    pub domain_chain_allowlist: DomainAllowlistUpdates,
    pub maybe_sudo_runtime_call: Option<Vec<u8>>,
}

impl PassBy for DomainInherentExtrinsicData {
    type PassBy = pass_by::Codec<Self>;
}

#[derive(Debug, Decode, Encode, TypeInfo, PartialEq, Eq, Clone)]
pub struct DomainInherentExtrinsic {
    domain_timestamp_extrinsic: Vec<u8>,
    maybe_domain_chain_allowlist_extrinsic: Option<Vec<u8>>,
    consensus_chain_byte_fee_extrinsic: Vec<u8>,
    maybe_domain_set_code_extrinsic: Option<Vec<u8>>,
    maybe_domain_sudo_call_extrinsic: Option<Vec<u8>>,
}

#[derive(Debug, Decode, Encode, TypeInfo, PartialEq, Eq, Clone)]
pub enum DomainStorageKeyRequest {
    BlockFees,
    Transfers,
}

impl PassBy for DomainStorageKeyRequest {
    type PassBy = pass_by::Codec<Self>;
}

#[derive(Debug, Decode, Encode, TypeInfo, PartialEq, Eq, Clone)]
pub enum StatelessDomainRuntimeCall {
    IsTxInRange {
        opaque_extrinsic: OpaqueExtrinsic,
        domain_tx_range: U256,
        bundle_vrf_hash: U256,
    },
    IsInherentExtrinsic(OpaqueExtrinsic),
    IsDecodableExtrinsic(OpaqueExtrinsic),
    IsValidDomainSudoCall(Vec<u8>),
}

impl PassBy for StatelessDomainRuntimeCall {
    type PassBy = pass_by::Codec<Self>;
}

sp_api::decl_runtime_apis! {
    /// API necessary for fraud proof.
    pub trait FraudProofApi<DomainHeader: HeaderT> {
        /// Submit the fraud proof via an unsigned extrinsic.
        fn submit_fraud_proof_unsigned(fraud_proof: FraudProof<NumberFor<Block>, Block::Hash, DomainHeader, H256>);

        /// Return the storage key used in fraud proof
        fn fraud_proof_storage_key(req: FraudProofStorageKeyRequest<NumberFor<Block>>) -> Vec<u8>;
    }
}
