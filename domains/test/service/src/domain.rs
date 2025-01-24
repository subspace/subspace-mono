//! Utilities used for testing with the domain.
#![warn(missing_docs)]

use crate::chain_spec::create_domain_spec;
use crate::{
    construct_extrinsic_generic, node_config, BalanceOf, DomainRuntime, EcdsaKeyring,
    Sr25519Keyring, UncheckedExtrinsicFor, AUTO_ID_DOMAIN_ID, EVM_DOMAIN_ID,
};
use cross_domain_message_gossip::ChainMsg;
use domain_client_operator::snap_sync::ConsensusChainSyncParams;
use domain_client_operator::{fetch_domain_bootstrap_info, BootstrapResult, OperatorStreams};
use domain_runtime_primitives::opaque::Block;
use domain_runtime_primitives::{Balance, EthereumAccountId};
use domain_service::providers::DefaultProvider;
use domain_service::FullClient;
use domain_test_primitives::{EvmOnchainStateApi, OnchainStateApi};
use frame_support::dispatch::{DispatchInfo, PostDispatchInfo};
use frame_system::pallet_prelude::{BlockNumberFor, RuntimeCallFor};
use pallet_transaction_payment_rpc::TransactionPaymentRuntimeApi;
use sc_client_api::HeaderBackend;
use sc_domains::RuntimeExecutor;
use sc_network::service::traits::NetworkService;
use sc_network::{NetworkRequest, NetworkStateInfo};
use sc_network_sync::SyncingService;
use sc_service::config::MultiaddrWithPeerId;
use sc_service::{BasePath, Role, RpcHandlers, TFullBackend, TaskManager};
use sc_transaction_pool_api::OffchainTransactionPoolFactory;
use sc_utils::mpsc::{tracing_unbounded, TracingUnboundedSender};
use sp_api::{ApiExt, ConstructRuntimeApi, Metadata, ProvideRuntimeApi};
use sp_block_builder::BlockBuilder;
use sp_consensus_subspace::SubspaceApi;
use sp_core::{Encode, H256};
use sp_domains::core_api::DomainCoreApi;
use sp_domains::{DomainId, OperatorId, PermissionedActionAllowedBy};
use sp_messenger::messages::{ChainId, ChannelId};
use sp_messenger::{MessengerApi, RelayerApi};
use sp_offchain::OffchainWorkerApi;
use sp_runtime::traits::{Block as BlockT, Dispatchable, NumberFor};
use sp_runtime::OpaqueExtrinsic;
use sp_session::SessionKeys;
use sp_transaction_pool::runtime_api::TaggedTransactionQueue;
use std::future::Future;
use std::sync::Arc;
use subspace_runtime_primitives::opaque::Block as CBlock;
use subspace_runtime_primitives::Nonce;
use subspace_test_service::MockConsensusNode;
use substrate_frame_rpc_system::AccountNonceApi;
use substrate_test_client::{
    BlockchainEventsExt, RpcHandlersExt, RpcTransactionError, RpcTransactionOutput,
};

/// The backend type used by the test service.
pub type Backend = TFullBackend<Block>;

type Client<RuntimeApi> = FullClient<Block, RuntimeApi>;

/// Domain executor for the test service.
pub type DomainOperator<RuntimeApi> =
    domain_service::DomainOperator<Block, CBlock, subspace_test_client::Client, RuntimeApi>;

/// A generic domain node instance used for testing.
pub struct DomainNode<Runtime, RuntimeApi>
where
    Runtime: DomainRuntime,
    RuntimeApi: ConstructRuntimeApi<Block, Client<RuntimeApi>> + Send + Sync + 'static,
    RuntimeApi::RuntimeApi: ApiExt<Block>
        + Metadata<Block>
        + BlockBuilder<Block>
        + OffchainWorkerApi<Block>
        + SessionKeys<Block>
        + DomainCoreApi<Block>
        + MessengerApi<Block, NumberFor<CBlock>, <CBlock as BlockT>::Hash>
        + TaggedTransactionQueue<Block>
        + AccountNonceApi<Block, Runtime::AccountId, Nonce>
        + TransactionPaymentRuntimeApi<Block, Balance>
        + RelayerApi<Block, NumberFor<Block>, NumberFor<CBlock>, <CBlock as BlockT>::Hash>,
{
    /// The domain id
    pub domain_id: DomainId,
    /// The node's account key
    pub key: Runtime::Keyring,
    /// TaskManager's instance.
    pub task_manager: TaskManager,
    /// Client's instance.
    pub client: Arc<Client<RuntimeApi>>,
    /// Client backend.
    pub backend: Arc<Backend>,
    /// Code executor.
    pub code_executor: Arc<RuntimeExecutor>,
    /// Network service.
    pub network_service: Arc<dyn NetworkService>,
    /// Sync service.
    pub sync_service: Arc<SyncingService<Block>>,
    /// The `MultiaddrWithPeerId` to this node. This is useful if you want to pass it as "boot node"
    /// to other nodes.
    pub addr: MultiaddrWithPeerId,
    /// RPCHandlers to make RPC queries.
    pub rpc_handlers: RpcHandlers,
    /// Domain oeprator.
    pub operator: DomainOperator<RuntimeApi>,
    /// Sink to the node's tx pool
    pub tx_pool_sink: TracingUnboundedSender<ChainMsg>,
}

impl<Runtime, RuntimeApi> DomainNode<Runtime, RuntimeApi>
where
    Runtime: frame_system::Config<Hash = H256>
        + pallet_transaction_payment::Config
        + DomainRuntime
        + Send
        + Sync,
    Runtime::RuntimeCall:
        Dispatchable<Info = DispatchInfo, PostInfo = PostDispatchInfo> + Send + Sync,
    crate::BalanceOf<Runtime>: Send + Sync + From<u64> + sp_runtime::FixedPointOperand,
    u64: From<BlockNumberFor<Runtime>>,
    RuntimeApi: ConstructRuntimeApi<Block, Client<RuntimeApi>> + Send + Sync + 'static,
    RuntimeApi::RuntimeApi: ApiExt<Block>
        + Metadata<Block>
        + BlockBuilder<Block>
        + OffchainWorkerApi<Block>
        + SessionKeys<Block>
        + DomainCoreApi<Block>
        + TaggedTransactionQueue<Block>
        + AccountNonceApi<Block, <Runtime as DomainRuntime>::AccountId, Nonce>
        + TransactionPaymentRuntimeApi<Block, Balance>
        + MessengerApi<Block, NumberFor<CBlock>, <CBlock as BlockT>::Hash>
        + RelayerApi<Block, NumberFor<Block>, NumberFor<CBlock>, <CBlock as BlockT>::Hash>
        + OnchainStateApi<Block, <Runtime as DomainRuntime>::AccountId, Balance>,
{
    #[allow(clippy::too_many_arguments)]
    async fn build(
        domain_id: DomainId,
        tokio_handle: tokio::runtime::Handle,
        key: Runtime::Keyring,
        base_path: BasePath,
        domain_nodes: Vec<MultiaddrWithPeerId>,
        domain_nodes_exclusive: bool,
        skip_empty_bundle_production: bool,
        maybe_operator_id: Option<OperatorId>,
        role: Role,
        mock_consensus_node: &mut MockConsensusNode,
    ) -> Self {
        let BootstrapResult {
            domain_instance_data,
            domain_created_at,
            imported_block_notification_stream,
        } = fetch_domain_bootstrap_info::<Block, _, _>(&*mock_consensus_node.client, domain_id)
            .await
            .expect("Failed to get domain instance data");
        let chain_spec = create_domain_spec(domain_instance_data.raw_genesis);
        let key_seed = Runtime::to_seed(key);
        let domain_config = node_config(
            domain_id,
            tokio_handle.clone(),
            key_seed,
            domain_nodes,
            domain_nodes_exclusive,
            role,
            BasePath::new(base_path.path().join(format!("domain-{domain_id:?}"))),
            Box::new(chain_spec) as Box<_>,
        )
        .expect("could not generate domain node Configuration");

        let span = sc_tracing::tracing::info_span!(
            sc_tracing::logging::PREFIX_LOG_SPAN,
            name = domain_config.network.node_name.as_str()
        );
        let _enter = span.enter();

        let multiaddr = domain_config.network.listen_addresses[0].clone();

        let operator_streams = OperatorStreams {
            // Set `consensus_block_import_throttling_buffer_size` to 0 to ensure the primary chain will not be
            // ahead of the execution chain by more than one block, thus slot will not be skipped in test.
            consensus_block_import_throttling_buffer_size: 0,
            block_importing_notification_stream: mock_consensus_node
                .block_importing_notification_stream(),
            imported_block_notification_stream,
            new_slot_notification_stream: mock_consensus_node.new_slot_notification_stream(),
            acknowledgement_sender_stream: mock_consensus_node.new_acknowledgement_sender_stream(),
            _phantom: Default::default(),
        };

        let (domain_message_sink, domain_message_receiver) =
            tracing_unbounded("domain_message_channel", 100);
        let gossip_msg_sink = mock_consensus_node
            .xdm_gossip_worker_builder()
            .gossip_msg_sink();

        let maybe_operator_id = role
            .is_authority()
            .then_some(maybe_operator_id.unwrap_or(if domain_id == EVM_DOMAIN_ID { 0 } else { 1 }));

        let consensus_best_hash = mock_consensus_node.client.info().best_hash;
        let chain_constants = mock_consensus_node
            .client
            .runtime_api()
            .chain_constants(consensus_best_hash)
            .unwrap();

        let domain_params = domain_service::DomainParams {
            domain_id,
            domain_config,
            domain_created_at,
            consensus_client: mock_consensus_node.client.clone(),
            consensus_offchain_tx_pool_factory: OffchainTransactionPoolFactory::new(
                mock_consensus_node.transaction_pool.clone(),
            ),
            domain_sync_oracle: mock_consensus_node.sync_service.clone(),
            consensus_network: mock_consensus_node.network_service.clone(),
            operator_streams,
            gossip_message_sink: gossip_msg_sink,
            domain_message_receiver,
            provider: DefaultProvider,
            skip_empty_bundle_production,
            skip_out_of_order_slot: true,
            maybe_operator_id,
            confirmation_depth_k: chain_constants.confirmation_depth_k(),
            consensus_chain_sync_params: None::<
                ConsensusChainSyncParams<_, Arc<dyn NetworkRequest + Sync + Send>>,
            >,
        };

        let domain_node = domain_service::new_full::<
            _,
            _,
            _,
            _,
            _,
            _,
            RuntimeApi,
            <Runtime as DomainRuntime>::AccountId,
            _,
            _,
        >(domain_params)
        .await
        .expect("failed to build domain node");

        let domain_service::NewFull {
            task_manager,
            client,
            backend,
            code_executor,
            network_service,
            sync_service,
            network_starter,
            rpc_handlers,
            operator,
            ..
        } = domain_node;

        if role.is_authority() {
            mock_consensus_node
                .xdm_gossip_worker_builder()
                .push_chain_sink(ChainId::Domain(domain_id), domain_message_sink.clone());
        }

        let addr = MultiaddrWithPeerId {
            multiaddr,
            peer_id: network_service.local_peer_id(),
        };

        network_starter.start_network();

        DomainNode {
            domain_id,
            key,
            task_manager,
            client,
            backend,
            code_executor,
            network_service,
            sync_service,
            addr,
            rpc_handlers,
            operator,
            tx_pool_sink: domain_message_sink,
        }
    }

    /// Wait for `count` blocks to be imported in the node and then exit. This function will not
    /// return if no blocks are ever created, thus you should restrict the maximum amount of time of
    /// the test execution.
    pub fn wait_for_blocks(&self, count: usize) -> impl Future<Output = ()> {
        self.client.wait_for_blocks(count)
    }

    /// Get the nonce of the node account
    pub fn account_nonce(&self) -> u32 {
        self.client
            .runtime_api()
            .account_nonce(
                self.client.info().best_hash,
                <Runtime as DomainRuntime>::account_id(self.key),
            )
            .expect("Fail to get account nonce")
    }

    /// Sends an system.remark extrinsic to the pool.
    pub async fn send_system_remark(&mut self) {
        let nonce = self.account_nonce();
        let _ = self
            .construct_and_send_extrinsic(frame_system::Call::remark {
                remark: nonce.encode(),
            })
            .await
            .map(|_| ());
    }

    /// Construct an extrinsic with the current nonce of the node account and send it to this node.
    pub async fn construct_and_send_extrinsic(
        &mut self,
        function: impl Into<<Runtime as frame_system::Config>::RuntimeCall>,
    ) -> Result<RpcTransactionOutput, RpcTransactionError> {
        self.construct_and_send_extrinsic_with(self.account_nonce(), 0.into(), function)
            .await
    }

    /// Construct an extrinsic with the given nonce and tip for the node account and send it to this node.
    pub async fn construct_and_send_extrinsic_with(
        &self,
        nonce: u32,
        tip: BalanceOf<Runtime>,
        function: impl Into<<Runtime as frame_system::Config>::RuntimeCall>,
    ) -> Result<RpcTransactionOutput, RpcTransactionError> {
        let extrinsic = construct_extrinsic_generic::<Runtime, _>(
            &self.client,
            function,
            self.key,
            false,
            nonce,
            tip,
        );
        self.rpc_handlers.send_transaction(extrinsic.into()).await
    }

    /// Construct an extrinsic.
    pub fn construct_extrinsic(
        &mut self,
        nonce: u32,
        function: impl Into<<Runtime as frame_system::Config>::RuntimeCall>,
    ) -> UncheckedExtrinsicFor<Runtime> {
        construct_extrinsic_generic::<Runtime, _>(
            &self.client,
            function,
            self.key,
            false,
            nonce,
            0.into(),
        )
    }

    /// Construct an extrinsic with the given transaction tip.
    pub fn construct_extrinsic_with_tip(
        &mut self,
        nonce: u32,
        tip: BalanceOf<Runtime>,
        function: impl Into<<Runtime as frame_system::Config>::RuntimeCall>,
    ) -> UncheckedExtrinsicFor<Runtime> {
        construct_extrinsic_generic::<Runtime, _>(
            &self.client,
            function,
            self.key,
            false,
            nonce,
            tip,
        )
    }

    /// Send an extrinsic to this node.
    pub async fn send_extrinsic(
        &self,
        extrinsic: impl Into<OpaqueExtrinsic>,
    ) -> Result<RpcTransactionOutput, RpcTransactionError> {
        self.rpc_handlers.send_transaction(extrinsic.into()).await
    }

    /// Get the free balance of the given account
    pub fn free_balance(&self, account_id: <Runtime as DomainRuntime>::AccountId) -> Balance {
        self.client
            .runtime_api()
            .free_balance(self.client.info().best_hash, account_id)
            .expect("Fail to get account free balance")
    }

    /// Returns the open XDM channel for given chain
    pub fn get_open_channel_for_chain(&self, chain_id: ChainId) -> Option<ChannelId> {
        self.client
            .runtime_api()
            .get_open_channel_for_chain(self.client.info().best_hash, chain_id)
            .expect("Fail to get open channel for Chain")
    }

    /// Construct an unsigned extrinsic that can be applied to the test runtime.
    pub fn construct_unsigned_extrinsic(
        &self,
        function: impl Into<<Runtime as frame_system::Config>::RuntimeCall>,
    ) -> UncheckedExtrinsicFor<Runtime>
    where
        Runtime:
            frame_system::Config<Hash = H256> + pallet_transaction_payment::Config + Send + Sync,
        RuntimeCallFor<Runtime>:
            Dispatchable<Info = DispatchInfo, PostInfo = PostDispatchInfo> + Send + Sync,
        BalanceOf<Runtime>: Send + Sync + From<u64> + sp_runtime::FixedPointOperand,
    {
        let function = function.into();
        UncheckedExtrinsicFor::<Runtime>::new_unsigned(function)
    }
}

impl<Runtime, RuntimeApi> DomainNode<Runtime, RuntimeApi>
where
    Runtime: frame_system::Config<Hash = H256>
        + pallet_transaction_payment::Config
        + DomainRuntime
        + Send
        + Sync,
    RuntimeApi: ConstructRuntimeApi<Block, Client<RuntimeApi>> + Send + Sync + 'static,
    RuntimeApi::RuntimeApi: Metadata<Block>
        + BlockBuilder<Block>
        + OffchainWorkerApi<Block>
        + SessionKeys<Block>
        + DomainCoreApi<Block>
        + TaggedTransactionQueue<Block>
        + AccountNonceApi<Block, <Runtime as DomainRuntime>::AccountId, Nonce>
        + TransactionPaymentRuntimeApi<Block, Balance>
        + MessengerApi<Block, NumberFor<CBlock>, <CBlock as BlockT>::Hash>
        + RelayerApi<Block, NumberFor<Block>, NumberFor<CBlock>, <CBlock as BlockT>::Hash>
        + EvmOnchainStateApi<Block>,
{
    /// Returns the current EVM contract creation allow list.
    /// Returns `None` if this is not an EVM domain, or if the allow list isn't set (allow all).
    pub fn evm_contract_creation_allowed_by(
        &self,
    ) -> Option<PermissionedActionAllowedBy<EthereumAccountId>> {
        self.client
            .runtime_api()
            .evm_contract_creation_allowed_by(self.client.info().best_hash)
            .expect("Failed to get EVM contact creation allow list")
    }
}

/// A builder to create a [`DomainNode`].
pub struct DomainNodeBuilder {
    tokio_handle: tokio::runtime::Handle,
    domain_nodes: Vec<MultiaddrWithPeerId>,
    domain_nodes_exclusive: bool,
    skip_empty_bundle_production: bool,
    base_path: BasePath,
    maybe_operator_id: Option<OperatorId>,
}

impl DomainNodeBuilder {
    /// Create a new instance of `Self`.
    ///
    /// `tokio_handle` - The tokio handler to use.
    /// `base_path` - Where databases will be stored.
    pub fn new(tokio_handle: tokio::runtime::Handle, base_path: BasePath) -> Self {
        DomainNodeBuilder {
            tokio_handle,
            domain_nodes: Vec::new(),
            domain_nodes_exclusive: false,
            skip_empty_bundle_production: false,
            base_path,
            maybe_operator_id: None,
        }
    }

    /// Instruct the node to exclusively connect to registered parachain nodes.
    ///
    /// Domain nodes can be registered using [`Self::connect_to_domain_node`].
    pub fn exclusively_connect_to_registered_parachain_nodes(mut self) -> Self {
        self.domain_nodes_exclusive = true;
        self
    }

    /// Make the node connect to the given domain node.
    ///
    /// By default the node will not be connected to any node or will be able to discover any other
    /// node.
    pub fn connect_to_domain_node(mut self, addr: MultiaddrWithPeerId) -> Self {
        self.domain_nodes.push(addr);
        self
    }

    /// Skip empty bundle production when there is no non-empty domain block need to confirm
    pub fn skip_empty_bundle(mut self) -> Self {
        self.skip_empty_bundle_production = true;
        self
    }

    /// Set the operator id
    pub fn operator_id(mut self, operator_id: OperatorId) -> Self {
        self.maybe_operator_id = Some(operator_id);
        self
    }

    /// Build an EVM domain node
    pub async fn build_evm_node(
        self,
        role: Role,
        key: EcdsaKeyring,
        mock_consensus_node: &mut MockConsensusNode,
    ) -> EvmDomainNode {
        DomainNode::build(
            EVM_DOMAIN_ID,
            self.tokio_handle,
            key,
            self.base_path,
            self.domain_nodes,
            self.domain_nodes_exclusive,
            self.skip_empty_bundle_production,
            self.maybe_operator_id,
            role,
            mock_consensus_node,
        )
        .await
    }

    /// Build an Auto ID domain node
    pub async fn build_auto_id_node(
        self,
        role: Role,
        key: Sr25519Keyring,
        mock_consensus_node: &mut MockConsensusNode,
    ) -> AutoIdDomainNode {
        DomainNode::build(
            AUTO_ID_DOMAIN_ID,
            self.tokio_handle,
            key,
            self.base_path,
            self.domain_nodes,
            self.domain_nodes_exclusive,
            self.skip_empty_bundle_production,
            self.maybe_operator_id,
            role,
            mock_consensus_node,
        )
        .await
    }
}

/// The evm domain node
pub type EvmDomainNode =
    DomainNode<evm_domain_test_runtime::Runtime, evm_domain_test_runtime::RuntimeApi>;

/// The evm domain client
pub type EvmDomainClient = Client<evm_domain_test_runtime::RuntimeApi>;

/// The auto-id domain node
pub type AutoIdDomainNode =
    DomainNode<auto_id_domain_test_runtime::Runtime, auto_id_domain_test_runtime::RuntimeApi>;

/// The auto-id domain client
pub type AutoIdDomainClient = Client<auto_id_domain_test_runtime::RuntimeApi>;
