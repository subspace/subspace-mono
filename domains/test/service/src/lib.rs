// Copyright 2019-2021 Parity Technologies (UK) Ltd.
// This file is part of Cumulus.

// Cumulus is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// Cumulus is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with Cumulus.  If not, see <http://www.gnu.org/licenses/>.

//! Crate used for testing with Domain.

#![feature(trait_upcasting)]

pub mod chain_spec;
pub mod domain;
pub mod keyring;

pub use domain::*;
use domain_runtime_primitives::opaque::Block;
pub use evm_domain_test_runtime;
use frame_support::dispatch::{DispatchInfo, PostDispatchInfo};
use frame_system::pallet_prelude::{BlockNumberFor, RuntimeCallFor};
pub use keyring::Keyring as EcdsaKeyring;
use sc_network::config::{NonReservedPeerMode, TransportConfig};
use sc_network::multiaddr;
use sc_service::config::{
    DatabaseSource, ExecutorConfiguration, KeystoreConfig, MultiaddrWithPeerId,
    NetworkConfiguration, OffchainWorkerConfig, PruningMode, RpcBatchRequestConfig,
    RpcConfiguration, WasmExecutionMethod, WasmtimeInstantiationStrategy,
};
use sc_service::{
    BasePath, BlocksPruning, ChainSpec, Configuration as ServiceConfiguration,
    Error as ServiceError, Role,
};
use serde::de::DeserializeOwned;
use sp_arithmetic::traits::SaturatedConversion;
use sp_blockchain::HeaderBackend;
use sp_core::{Get, H256};
use sp_domains::DomainId;
pub use sp_keyring::Sr25519Keyring;
use sp_runtime::codec::{Decode, Encode};
use sp_runtime::generic;
use sp_runtime::generic::SignedPayload;
use sp_runtime::traits::Dispatchable;
use std::fmt::{Debug, Display};
use std::str::FromStr;

/// The domain id of the evm domain
pub const EVM_DOMAIN_ID: DomainId = DomainId::new(0u32);

/// The domain id of the auto-id domain
pub const AUTO_ID_DOMAIN_ID: DomainId = DomainId::new(1u32);

/// Create a domain node `Configuration`.
///
/// By default an in-memory socket will be used, therefore you need to provide nodes if you want the
/// node to be connected to other nodes. If `nodes_exclusive` is `true`, the node will only connect
/// to the given `nodes` and not to any other node.
#[allow(clippy::too_many_arguments)]
pub fn node_config(
    domain_id: DomainId,
    tokio_handle: tokio::runtime::Handle,
    key_seed: String,
    nodes: Vec<MultiaddrWithPeerId>,
    nodes_exclusive: bool,
    role: Role,
    base_path: BasePath,
    chain_spec: Box<dyn ChainSpec>,
) -> Result<ServiceConfiguration, ServiceError> {
    let root = base_path.path().to_path_buf();

    let domain_name = format!("{domain_id:?}");

    let mut network_config = NetworkConfiguration::new(
        format!("{key_seed} ({domain_name})"),
        "network/test/0.1",
        Default::default(),
        None,
    );

    if nodes_exclusive {
        network_config.default_peers_set.reserved_nodes = nodes;
        network_config.default_peers_set.non_reserved_mode = NonReservedPeerMode::Deny;
    } else {
        network_config.boot_nodes = nodes;
    }

    network_config.allow_non_globals_in_dht = true;

    network_config
        .listen_addresses
        .push(multiaddr::Protocol::Memory(rand::random()).into());

    // NOTE: Block sync is disabled for the domain subnet thus the major sync state may not be accurate,
    // which will cause transaction not propagate through network properly, setting the `force_synced`
    // flag can workaround this issue.
    network_config.force_synced = true;

    network_config.transport = TransportConfig::MemoryOnly;

    Ok(ServiceConfiguration {
        impl_name: "domain-test-node".to_string(),
        impl_version: "0.1".to_string(),
        role,
        tokio_handle,
        transaction_pool: Default::default(),
        network: network_config,
        keystore: KeystoreConfig::InMemory,
        database: DatabaseSource::ParityDb {
            path: root.join("paritydb"),
        },
        trie_cache_maximum_size: Some(16 * 1024 * 1024),
        state_pruning: Some(PruningMode::ArchiveAll),
        blocks_pruning: BlocksPruning::KeepAll,
        chain_spec,
        executor: ExecutorConfiguration {
            wasm_method: WasmExecutionMethod::Compiled {
                instantiation_strategy: WasmtimeInstantiationStrategy::PoolingCopyOnWrite,
            },
            max_runtime_instances: 8,
            default_heap_pages: None,
            runtime_cache_size: 2,
        },
        rpc: RpcConfiguration {
            addr: None,
            max_request_size: 0,
            max_response_size: 0,
            id_provider: None,
            max_subs_per_conn: 0,
            port: 0,
            message_buffer_capacity: 0,
            batch_config: RpcBatchRequestConfig::Disabled,
            max_connections: 0,
            cors: None,
            methods: Default::default(),
            rate_limit: None,
            rate_limit_whitelisted_ips: vec![],
            rate_limit_trust_proxy_headers: false,
        },
        prometheus_config: None,
        telemetry_endpoints: None,
        offchain_worker: OffchainWorkerConfig {
            enabled: true,
            indexing_enabled: false,
        },
        force_authoring: false,
        disable_grandpa: false,
        dev_key_seed: Some(key_seed),
        tracing_targets: None,
        tracing_receiver: Default::default(),
        announce_block: true,
        data_path: base_path.path().into(),
        base_path,
        wasm_runtime_overrides: None,
    })
}

type SignedExtraFor<Runtime> = (
    frame_system::CheckNonZeroSender<Runtime>,
    frame_system::CheckSpecVersion<Runtime>,
    frame_system::CheckTxVersion<Runtime>,
    frame_system::CheckGenesis<Runtime>,
    frame_system::CheckMortality<Runtime>,
    frame_system::CheckNonce<Runtime>,
    domain_check_weight::CheckWeight<Runtime>,
    pallet_transaction_payment::ChargeTransactionPayment<Runtime>,
);

type UncheckedExtrinsicFor<Runtime> = generic::UncheckedExtrinsic<
    <Runtime as DomainRuntime>::Address,
    <Runtime as frame_system::Config>::RuntimeCall,
    <Runtime as DomainRuntime>::Signature,
    SignedExtraFor<Runtime>,
>;

type BalanceOf<T> = <<T as pallet_transaction_payment::Config>::OnChargeTransaction as pallet_transaction_payment::OnChargeTransaction<T>>::Balance;

pub fn construct_extrinsic_raw_payload<Runtime, Client>(
    client: impl AsRef<Client>,
    function: RuntimeCallFor<Runtime>,
    immortal: bool,
    nonce: u32,
    tip: BalanceOf<Runtime>,
) -> (
    SignedPayload<RuntimeCallFor<Runtime>, SignedExtraFor<Runtime>>,
    SignedExtraFor<Runtime>,
)
where
    Runtime: frame_system::Config<Hash = H256> + pallet_transaction_payment::Config + Send + Sync,
    RuntimeCallFor<Runtime>:
        Dispatchable<Info = DispatchInfo, PostInfo = PostDispatchInfo> + Send + Sync,
    BalanceOf<Runtime>: Send + Sync + From<u64> + sp_runtime::FixedPointOperand,
    u64: From<BlockNumberFor<Runtime>>,
    Client: HeaderBackend<Block>,
{
    let current_block_hash = client.as_ref().info().best_hash;
    let current_block = client.as_ref().info().best_number.saturated_into();
    let genesis_block = client.as_ref().hash(0).unwrap().unwrap();
    let period = u64::from(<Runtime as frame_system::Config>::BlockHashCount::get())
        .checked_next_power_of_two()
        .map(|c| c / 2)
        .unwrap_or(2);
    let extra: SignedExtraFor<Runtime> = (
        frame_system::CheckNonZeroSender::<Runtime>::new(),
        frame_system::CheckSpecVersion::<Runtime>::new(),
        frame_system::CheckTxVersion::<Runtime>::new(),
        frame_system::CheckGenesis::<Runtime>::new(),
        frame_system::CheckMortality::<Runtime>::from(if immortal {
            generic::Era::Immortal
        } else {
            generic::Era::mortal(period, current_block)
        }),
        frame_system::CheckNonce::<Runtime>::from(nonce.into()),
        domain_check_weight::CheckWeight::<Runtime>::new(),
        pallet_transaction_payment::ChargeTransactionPayment::<Runtime>::from(tip),
    );
    (
        generic::SignedPayload::<RuntimeCallFor<Runtime>, SignedExtraFor<Runtime>>::from_raw(
            function,
            extra.clone(),
            ((), 1, 0, genesis_block, current_block_hash, (), (), ()),
        ),
        extra,
    )
}

pub trait DomainRuntime {
    type Keyring: Copy;
    type AccountId: DeserializeOwned
        + Encode
        + Decode
        + Clone
        + Debug
        + Display
        + FromStr
        + Sync
        + Send
        + 'static;
    type Address: Encode + Decode;
    type Signature: Encode + Decode;
    fn sign(key: Self::Keyring, payload: &[u8]) -> Self::Signature;
    fn account_id(key: Self::Keyring) -> Self::AccountId;
    fn address(key: Self::Keyring) -> Self::Address;
    fn to_seed(key: Self::Keyring) -> String;
}

impl DomainRuntime for evm_domain_test_runtime::Runtime {
    type Keyring = EcdsaKeyring;
    type AccountId = evm_domain_test_runtime::AccountId;
    type Address = evm_domain_test_runtime::Address;
    type Signature = evm_domain_test_runtime::Signature;

    fn sign(key: Self::Keyring, payload: &[u8]) -> Self::Signature {
        evm_domain_test_runtime::Signature::new(key.sign(payload))
    }

    fn account_id(key: Self::Keyring) -> Self::AccountId {
        key.to_account_id()
    }

    fn address(key: Self::Keyring) -> Self::Address {
        key.to_account_id()
    }

    fn to_seed(key: Self::Keyring) -> String {
        key.to_seed()
    }
}

impl DomainRuntime for auto_id_domain_test_runtime::Runtime {
    type Keyring = Sr25519Keyring;
    type AccountId = auto_id_domain_test_runtime::AccountId;
    type Address = auto_id_domain_test_runtime::Address;
    type Signature = auto_id_domain_test_runtime::Signature;

    fn sign(key: Self::Keyring, payload: &[u8]) -> Self::Signature {
        key.sign(payload).into()
    }

    fn account_id(key: Self::Keyring) -> Self::AccountId {
        key.to_account_id()
    }

    fn address(key: Self::Keyring) -> Self::Address {
        sp_runtime::MultiAddress::Id(key.to_account_id())
    }

    fn to_seed(key: Self::Keyring) -> String {
        key.to_seed()
    }
}

/// Construct an extrinsic that can be applied to the test runtime.
pub fn construct_extrinsic_generic<Runtime, Client>(
    client: impl AsRef<Client>,
    function: impl Into<<Runtime as frame_system::Config>::RuntimeCall>,
    caller: Runtime::Keyring,
    immortal: bool,
    nonce: u32,
    tip: BalanceOf<Runtime>,
) -> UncheckedExtrinsicFor<Runtime>
where
    Runtime: frame_system::Config<Hash = H256>
        + pallet_transaction_payment::Config
        + DomainRuntime
        + Send
        + Sync,
    Runtime::RuntimeCall:
        Dispatchable<Info = DispatchInfo, PostInfo = PostDispatchInfo> + Send + Sync,
    BalanceOf<Runtime>: Send + Sync + From<u64> + sp_runtime::FixedPointOperand,
    u64: From<BlockNumberFor<Runtime>>,
    Client: HeaderBackend<Block>,
{
    let function = function.into();
    let (raw_payload, extra) =
        construct_extrinsic_raw_payload(client, function.clone(), immortal, nonce, tip);
    let signature = raw_payload.using_encoded(|e| Runtime::sign(caller, e));
    let address = Runtime::address(caller);
    UncheckedExtrinsicFor::<Runtime>::new_signed(function, address, signature, extra)
}

/// Construct an unsigned extrinsic that can be applied to the test runtime.
pub fn construct_unsigned_extrinsic<Runtime>(
    function: impl Into<<Runtime as frame_system::Config>::RuntimeCall>,
) -> UncheckedExtrinsicFor<Runtime>
where
    Runtime: frame_system::Config<Hash = H256>
        + pallet_transaction_payment::Config
        + DomainRuntime
        + Send
        + Sync,
    Runtime::RuntimeCall:
        Dispatchable<Info = DispatchInfo, PostInfo = PostDispatchInfo> + Send + Sync,
    BalanceOf<Runtime>: Send + Sync + From<u64> + sp_runtime::FixedPointOperand,
{
    let function = function.into();
    UncheckedExtrinsicFor::<Runtime>::new_unsigned(function)
}
