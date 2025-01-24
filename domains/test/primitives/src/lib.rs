//! Test primitives crates that expose extensions for testing.

#![cfg_attr(not(feature = "std"), no_std)]

use codec::{Decode, Encode};
use domain_runtime_primitives::EthereumAccountId;
use sp_domains::PermissionedActionAllowedBy;
use sp_messenger::messages::{ChainId, ChannelId};
use subspace_runtime_primitives::Moment;

sp_api::decl_runtime_apis! {
    /// Api that returns the domain timestamp
    pub trait TimestampApi {
        /// Returns the current domain timestamp
        fn domain_timestamp() -> Moment;
    }
}

sp_api::decl_runtime_apis! {
    /// Api for querying onchain state in tests
    pub trait OnchainStateApi<AccountId, Balance>
    where
        AccountId: Encode + Decode,
        Balance: Encode + Decode
    {
        /// Api to get the free balance of the given account
        fn free_balance(account_id: AccountId) -> Balance;

        /// Returns the last open channel for a given domain.
        fn get_open_channel_for_chain(dst_chain_id: ChainId) -> Option<ChannelId>;

        /// Api to get the current domain transaction byte fee
        fn consensus_transaction_byte_fee() -> Balance;
    }
}

sp_api::decl_runtime_apis! {
    /// Api for querying onchain EVM state in tests
    pub trait EvmOnchainStateApi
    {
        /// Returns the current EVM contract creation allow list.
        /// Returns `None` if this is not an EVM domain, or if the allow list isn't set (allow all).
        fn evm_contract_creation_allowed_by() -> Option<PermissionedActionAllowedBy<EthereumAccountId>>;
    }
}
