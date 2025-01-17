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

//! Common primitives for subspace domain runtime.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
pub use fp_account::{AccountId20, EthereumSignature as EVMSignature};
use frame_support::dispatch::DispatchClass;
use frame_support::weights::constants::{BlockExecutionWeight, ExtrinsicBaseWeight};
use frame_system::limits::{BlockLength, BlockWeights};
use parity_scale_codec::{Decode, Encode};
use scale_info::TypeInfo;
use serde::{Deserialize, Serialize};
use sp_runtime::generic::UncheckedExtrinsic;
use sp_runtime::traits::{Convert, IdentifyAccount, Verify};
use sp_runtime::transaction_validity::TransactionValidityError;
use sp_runtime::{MultiAddress, MultiSignature, Perbill};
use sp_weights::constants::WEIGHT_REF_TIME_PER_SECOND;
use sp_weights::Weight;
pub use subspace_runtime_primitives::HoldIdentifier;
use subspace_runtime_primitives::{MAX_BLOCK_LENGTH, SHANNON};

/// Alias to 512-bit hash when used in the context of a transaction signature on the chain.
pub type Signature = MultiSignature;

/// Some way of identifying an account on the chain. We intentionally make it equivalent
/// to the public key of our transaction signing scheme.
pub type AccountId = <<Signature as Verify>::Signer as IdentifyAccount>::AccountId;

/// Balance of an account.
pub type Balance = u128;

/// Index of a transaction in the chain.
pub type Nonce = u32;

/// A hash of some data used by the chain.
pub type Hash = sp_core::H256;

/// An index to a block.
pub type BlockNumber = u32;

/// The address format for describing accounts.
pub type Address = MultiAddress<AccountId, ()>;

/// Slot duration that is same as consensus chain runtime.
pub const SLOT_DURATION: u64 = 1000;

/// The EVM chain Id type
pub type EVMChainId = u64;

/// Alias to 512-bit hash when used in the context of a transaction signature on the EVM chain.
pub type EthereumSignature = EVMSignature;

/// Some way of identifying an account on the EVM chain. We intentionally make it equivalent
/// to the public key of the EVM transaction signing scheme.
pub type EthereumAccountId = <<EthereumSignature as Verify>::Signer as IdentifyAccount>::AccountId;

/// Dispatch ratio for domains
pub const NORMAL_DISPATCH_RATIO: Perbill = Perbill::from_percent(65);

/// The maximum domain block weight with 3.25 MiB as proof size
/// Consensus allows 3.75 MiB but Fraud proof can carry extra size along with proof size
/// So we set the proof size to 3.25 MiB
pub fn maximum_domain_block_weight() -> Weight {
    let consensus_maximum_normal_block_length =
        *maximum_block_length().max.get(DispatchClass::Normal) as u64;
    let weight =
        NORMAL_DISPATCH_RATIO * Weight::from_parts(WEIGHT_REF_TIME_PER_SECOND.saturating_mul(2), 0);
    weight.set_proof_size(consensus_maximum_normal_block_length)
}

/// Custom error when nonce overflow occurs.
pub const ERR_NONCE_OVERFLOW: u8 = 100;
/// Custom error when balance overflow occurs.
pub const ERR_BALANCE_OVERFLOW: u8 = 200;

/// Maximum block length for all dispatches.
/// This is set to 3.75 MiB since consensus chain supports on 3.75 MiB for normal
pub fn maximum_block_length() -> BlockLength {
    BlockLength::max_with_normal_ratio(MAX_BLOCK_LENGTH, NORMAL_DISPATCH_RATIO)
}

/// Computed as ED = Account data size * Price per byte, where
/// Price per byte = Min Number of validators * Storage duration (years) * Storage cost per year
/// Account data size (80 bytes)
/// Min Number of redundant validators (10) - For a stable and redundant blockchain we need at least a certain number of full nodes/collators.
/// Storage duration (1 year) - It is theoretically unlimited, accounts will stay around while the chain is alive.
/// Storage cost per year of (12 * 1e-9 * 0.1 ) - SSD storage on cloud hosting costs about 0.1 USD per Gb per month
pub const EXISTENTIAL_DEPOSIT: Balance = 1_000_000_000_000 * SHANNON;

/// We assume that ~5% of the block weight is consumed by `on_initialize` handlers. This is
/// used to limit the maximal weight of a single extrinsic.
const AVERAGE_ON_INITIALIZE_RATIO: Perbill = Perbill::from_percent(5);

pub fn block_weights() -> BlockWeights {
    // NOTE: this value is used in many places as the `max_block` weight (e.g. `used_weight / max_block`
    // is used in tx fee adjustment), but it does not serve as a limit of the total weight of a domain block,
    // which is `max_bundle_weight * number_of_bundle` and bundle production is probabilistic.
    // See [`domain-check-weight::CheckWeight`] for more detail about the domain block weight check.
    //
    // TODO: instead of using a constant weight value for all the domain runtimes, perhaps derive and use the
    // target domain block weight bases on `max_bundle_weight` and `bundle_slot_probability`.
    let maximum_block_weight = maximum_domain_block_weight();

    // If the bundle slot probability is the same as the consensus slot probability then
    // there is one bundle per block, such a bundle is allowed to consume the whole domain
    // block weight
    let max_extrinsic_weight = maximum_block_weight - ExtrinsicBaseWeight::get();

    BlockWeights::builder()
        .base_block(BlockExecutionWeight::get())
        .for_class(DispatchClass::all(), |weights| {
            weights.base_extrinsic = ExtrinsicBaseWeight::get();
            // maximum weight of each transaction would be the maximum weight of
            // single bundle
            weights.max_extrinsic = Some(max_extrinsic_weight);
            // explicitly set max_total weight for normal dispatches to maximum
            weights.max_total = Some(maximum_block_weight);
        })
        .avg_block_initialization(AVERAGE_ON_INITIALIZE_RATIO)
        .build_or_panic()
}

/// Extracts the signer from an unchecked extrinsic.
///
/// Used by executor to extract the optional signer when shuffling the extrinsics.
pub trait Signer<AccountId, Lookup> {
    /// Returns the AccountId of signer.
    fn signer(&self, lookup: &Lookup) -> Option<AccountId>;
}

impl<Address, AccountId, Call, Signature, Extra, Lookup> Signer<AccountId, Lookup>
    for UncheckedExtrinsic<Address, Call, Signature, Extra>
where
    Address: Clone,
    Extra: sp_runtime::traits::SignedExtension<AccountId = AccountId>,
    Lookup: sp_runtime::traits::Lookup<Source = Address, Target = AccountId>,
{
    fn signer(&self, lookup: &Lookup) -> Option<AccountId> {
        self.signature
            .as_ref()
            .and_then(|(signed, _, _)| lookup.lookup(signed.clone()).ok())
    }
}

/// MultiAccountId used by all the domains to describe their account type.
#[derive(
    Debug, Encode, Decode, Clone, Eq, PartialEq, TypeInfo, Serialize, Deserialize, Ord, PartialOrd,
)]
pub enum MultiAccountId {
    /// 32 byte account Id.
    AccountId32([u8; 32]),
    /// 20 byte account Id. Ex: Ethereum
    AccountId20([u8; 20]),
    /// Some raw bytes
    Raw(Vec<u8>),
}

/// Extensible conversion trait. Generic over both source and destination types.
pub trait TryConvertBack<A, B>: Convert<A, B> {
    /// Make conversion back.
    fn try_convert_back(b: B) -> Option<A>;
}

/// An AccountId32 to MultiAccount converter.
pub struct AccountIdConverter;

impl Convert<AccountId, MultiAccountId> for AccountIdConverter {
    fn convert(account_id: AccountId) -> MultiAccountId {
        MultiAccountId::AccountId32(account_id.into())
    }
}

impl TryConvertBack<AccountId, MultiAccountId> for AccountIdConverter {
    fn try_convert_back(multi_account_id: MultiAccountId) -> Option<AccountId> {
        match multi_account_id {
            MultiAccountId::AccountId32(acc) => Some(AccountId::new(acc)),
            _ => None,
        }
    }
}

/// An AccountId20 to MultiAccount converter.
pub struct AccountId20Converter;

impl Convert<AccountId20, MultiAccountId> for AccountId20Converter {
    fn convert(account_id: AccountId20) -> MultiAccountId {
        MultiAccountId::AccountId20(account_id.into())
    }
}

impl TryConvertBack<AccountId20, MultiAccountId> for AccountId20Converter {
    fn try_convert_back(multi_account_id: MultiAccountId) -> Option<AccountId20> {
        match multi_account_id {
            MultiAccountId::AccountId20(acc) => Some(AccountId20::from(acc)),
            _ => None,
        }
    }
}

#[derive(Debug, Decode, Encode, TypeInfo, PartialEq, Eq, Clone)]
pub struct CheckExtrinsicsValidityError {
    pub extrinsic_index: u32,
    pub transaction_validity_error: TransactionValidityError,
}

#[derive(Debug, Decode, Encode, TypeInfo, PartialEq, Eq, Clone)]
pub struct DecodeExtrinsicError(pub String);

/// fullu qualified method name of check_extrinsics_and_do_pre_dispatch runtime api.
/// Used to call state machine.
/// Change it when the runtime api's name is changed in the interface.
pub const CHECK_EXTRINSICS_AND_DO_PRE_DISPATCH_METHOD_NAME: &str =
    "DomainCoreApi_check_extrinsics_and_do_pre_dispatch";

/// Opaque types.
///
/// These are used by the CLI to instantiate machinery that don't need to know the specifics of the
/// runtime. They can then be made to be agnostic over specific formats of data like extrinsics,
/// allowing for them to continue syncing the network through upgrades to even the core data
/// structures.
pub mod opaque {
    use crate::BlockNumber;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    use sp_runtime::generic;
    use sp_runtime::traits::BlakeTwo256;
    pub use sp_runtime::OpaqueExtrinsic as UncheckedExtrinsic;

    /// Opaque block header type.
    pub type Header = generic::Header<BlockNumber, BlakeTwo256>;
    /// Opaque block type.
    pub type Block = generic::Block<Header, UncheckedExtrinsic>;
    /// Opaque block identifier type.
    pub type BlockId = generic::BlockId<Block>;
    /// Opaque account identifier type.
    pub type AccountId = Vec<u8>;
}

#[cfg(test)]
mod test {
    use super::block_weights;

    #[test]
    fn test_block_weights() {
        // validate and build block weights
        let _block_weights = block_weights();
    }
}
