use crate::commands::{CreateDomainKeyOptions, InsertDomainKeyOptions};
use crate::domain::{auto_id_chain_spec, evm_chain_spec};
use clap::{Parser, ValueEnum};
use domain_runtime_primitives::opaque::Block as DomainBlock;
use domain_runtime_primitives::MultiAccountId;
use parity_scale_codec::Encode;
use sc_cli::{
    BlockNumberOrHash, ChainSpec, CliConfiguration, DefaultConfigurationValues, ImportParams,
    KeystoreParams, NetworkParams, Role, RunCmd as SubstrateRunCmd, RuntimeVersion, SharedParams,
    SubstrateCli,
};
use sc_client_api::backend::AuxStore;
use sc_network::config::NodeKeyConfig;
use sc_service::config::{KeystoreConfig, PrometheusConfig};
use sc_service::{BasePath, Configuration, DatabaseSource};
use sp_blockchain::HeaderBackend;
use sp_domain_digests::AsPredigest;
use sp_domains::storage::RawGenesis;
use sp_domains::{
    DomainId, DomainRuntimeConfig, OperatorAllowList, OperatorId, OperatorPublicKey, RuntimeType,
};
use sp_runtime::generic::BlockId;
use sp_runtime::traits::Header;
use sp_runtime::DigestItem;
use std::io::Write;
use std::path::Path;
use subspace_runtime::Block;
use subspace_runtime_primitives::{AccountId, Balance};

/// Sub-commands supported by the operator.
#[derive(Debug, clap::Subcommand)]
#[allow(clippy::large_enum_variant)]
pub enum Subcommand {
    /// Domain key management
    #[clap(subcommand)]
    Key(DomainKey),

    /// Export the state of a given block into a chain spec.
    ExportState(sc_cli::ExportStateCmd),

    /// Revert the chain to a previous state.
    Revert(sc_cli::RevertCmd),

    /// Sub-commands concerned with benchmarking.
    #[clap(subcommand)]
    Benchmark(frame_benchmarking_cli::BenchmarkCmd),

    /// Build the genesis storage of the evm domain chain in json format
    BuildGenesisStorage(BuildGenesisStorageCmd),

    /// The `export-execution-receipt` command used to get the ER from the auxiliary storage of the operator client
    ExportExecutionReceipt(ExportExecutionReceiptCmd),
}

#[derive(Debug, clap::Subcommand)]
pub enum DomainKey {
    /// Create key and import into domain's keystore
    Create(CreateDomainKeyOptions),
    /// Insert key into domain's keystore
    Insert(InsertDomainKeyOptions),
}

#[derive(Debug, Parser)]
pub struct DomainCli {
    /// Run a domain node.
    #[clap(flatten)]
    pub run: SubstrateRunCmd,

    #[clap(long)]
    pub domain_id: DomainId,

    /// Use provided operator id to submit bundles.
    #[arg(long)]
    pub operator_id: Option<OperatorId>,
}

#[derive(Debug, Copy, Clone)]
pub enum SpecId {
    Dev,
    Taurus,
    DevNet,
}

impl DomainCli {
    /// Constructs a new instance of [`DomainCli`].
    pub fn new(domain_args: impl Iterator<Item = String>) -> Self {
        DomainCli::parse_from([Self::executable_name()].into_iter().chain(domain_args))
    }

    /// Creates domain configuration from domain cli.
    pub fn create_domain_configuration(
        &self,
        cmd: &impl CliConfiguration,
        base_path: &Path,
        tokio_handle: tokio::runtime::Handle,
    ) -> sc_cli::Result<Configuration> {
        let mut domain_config = SubstrateCli::create_configuration(self, cmd, tokio_handle)?;

        // Change default paths to Subspace structure
        let domain_base_path = base_path.join(self.domain_id.to_string());
        {
            domain_config.database = DatabaseSource::ParityDb {
                path: domain_base_path.join("db"),
            };
            domain_config.keystore = KeystoreConfig::Path {
                path: domain_base_path.join("keystore"),
                password: match domain_config.keystore {
                    KeystoreConfig::Path { password, .. } => password,
                    KeystoreConfig::InMemory => None,
                },
            };
            // Network directory is shared with consensus chain
            if let Some(net_config_path) = &mut domain_config.network.net_config_path {
                *net_config_path = base_path.join("network");

                if let NodeKeyConfig::Ed25519(sc_network::config::Secret::File(node_key_file)) =
                    &mut domain_config.network.node_key
                {
                    *node_key_file = net_config_path.join("secret_ed25519");
                }
            }

            domain_config.base_path = BasePath::new(domain_base_path.clone());
            domain_config.data_path.clone_from(&domain_base_path);
        }
        Ok(domain_config)
    }
}

impl SubstrateCli for DomainCli {
    fn impl_name() -> String {
        "Subspace Domain".into()
    }

    fn impl_version() -> String {
        env!("SUBSTRATE_CLI_IMPL_VERSION").into()
    }

    fn executable_name() -> String {
        // Customize to make sure directory used for data by default is the same regardless of the
        // name of the executable file.
        "subspace-node".to_string()
    }

    fn description() -> String {
        "Subspace Domain".into()
    }

    fn author() -> String {
        env!("CARGO_PKG_AUTHORS").into()
    }

    fn support_url() -> String {
        "https://github.com/autonomys/subspace/issues/new".into()
    }

    fn copyright_start_year() -> i32 {
        2022
    }

    fn load_spec(&self, id: &str) -> Result<Box<dyn ChainSpec>, String> {
        // TODO: properly fetch the runtime name of `self.domain_id`
        let runtime_name = if self.domain_id == 0.into() {
            "auto-id"
        } else {
            "evm"
        };

        match runtime_name {
            "evm" => evm_chain_spec::load_chain_spec(id),
            "auto-id" => auto_id_chain_spec::load_chain_spec(id),
            unknown_name => Err(format!("Unknown runtime: {unknown_name}")),
        }
    }
}

impl DefaultConfigurationValues for DomainCli {
    fn p2p_listen_port() -> u16 {
        30334
    }

    fn rpc_listen_port() -> u16 {
        9945
    }

    fn prometheus_listen_port() -> u16 {
        9616
    }
}

impl CliConfiguration<Self> for DomainCli {
    fn shared_params(&self) -> &SharedParams {
        self.run.shared_params()
    }

    fn import_params(&self) -> Option<&ImportParams> {
        self.run.import_params()
    }

    fn network_params(&self) -> Option<&NetworkParams> {
        self.run.network_params()
    }

    fn keystore_params(&self) -> Option<&KeystoreParams> {
        self.run.keystore_params()
    }

    fn base_path(&self) -> sc_cli::Result<Option<BasePath>> {
        self.shared_params().base_path()
    }

    fn rpc_addr(
        &self,
        default_listen_port: u16,
    ) -> sc_cli::Result<Option<Vec<sc_cli::RpcEndpoint>>> {
        self.run.rpc_addr(default_listen_port)
    }

    fn prometheus_config(
        &self,
        default_listen_port: u16,
        chain_spec: &Box<dyn ChainSpec>,
    ) -> sc_cli::Result<Option<PrometheusConfig>> {
        self.run.prometheus_config(default_listen_port, chain_spec)
    }

    fn chain_id(&self, is_dev: bool) -> sc_cli::Result<String> {
        self.run.chain_id(is_dev)
    }

    fn role(&self, _is_dev: bool) -> sc_cli::Result<sc_service::Role> {
        if self.run.validator {
            return Err(sc_cli::Error::Input(
                "use `--operator-id` argument to run as operator".to_string(),
            ));
        }

        // is authority when operator_id is passed.
        let is_authority = self.operator_id.is_some();

        Ok(if is_authority {
            Role::Authority
        } else {
            Role::Full
        })
    }

    fn transaction_pool(
        &self,
        is_dev: bool,
    ) -> sc_cli::Result<sc_service::config::TransactionPoolOptions> {
        self.run.transaction_pool(is_dev)
    }

    fn trie_cache_maximum_size(&self) -> sc_cli::Result<Option<usize>> {
        self.run.trie_cache_maximum_size()
    }

    fn rpc_methods(&self) -> sc_cli::Result<sc_service::config::RpcMethods> {
        self.run.rpc_methods()
    }

    fn rpc_max_connections(&self) -> sc_cli::Result<u32> {
        self.run.rpc_max_connections()
    }

    fn rpc_cors(&self, is_dev: bool) -> sc_cli::Result<Option<Vec<String>>> {
        self.run.rpc_cors(is_dev)
    }

    fn default_heap_pages(&self) -> sc_cli::Result<Option<u64>> {
        self.run.default_heap_pages()
    }

    fn force_authoring(&self) -> sc_cli::Result<bool> {
        self.run.force_authoring()
    }

    fn disable_grandpa(&self) -> sc_cli::Result<bool> {
        self.run.disable_grandpa()
    }

    fn max_runtime_instances(&self) -> sc_cli::Result<Option<usize>> {
        self.run.max_runtime_instances()
    }

    fn announce_block(&self) -> sc_cli::Result<bool> {
        self.run.announce_block()
    }

    fn dev_key_seed(&self, is_dev: bool) -> sc_cli::Result<Option<String>> {
        self.run.dev_key_seed(is_dev)
    }

    fn telemetry_endpoints(
        &self,
        chain_spec: &Box<dyn ChainSpec>,
    ) -> sc_cli::Result<Option<sc_telemetry::TelemetryEndpoints>> {
        self.run.telemetry_endpoints(chain_spec)
    }
}

#[derive(Debug, Clone, ValueEnum)]
pub enum DomainRuntimeType {
    Evm,
    AutoId,
}

/// The `build-genesis-storage` command used to build the genesis storage of the evm domain chain.
#[derive(Debug, Clone, Parser)]
pub struct BuildGenesisStorageCmd {
    // The domain runtime type
    #[arg(long)]
    pub runtime_type: DomainRuntimeType,

    /// The base struct of the build-genesis-storage command.
    #[clap(flatten)]
    pub shared_params: SharedParams,
}

impl BuildGenesisStorageCmd {
    /// Run the build-genesis-storage command
    pub fn run(&self) -> sc_cli::Result<()> {
        let is_dev = self.shared_params.is_dev();
        let chain_id = self.shared_params.chain_id(is_dev);
        let domain_chain_spec = match chain_id.as_str() {
            "taurus" | "devnet" | "dev" => match self.runtime_type {
                DomainRuntimeType::Evm => evm_chain_spec::load_chain_spec(&chain_id)?,
                DomainRuntimeType::AutoId => auto_id_chain_spec::load_chain_spec(&chain_id)?,
            },
            unknown_id => {
                eprintln!("unknown chain {unknown_id:?}, expected taurus, devnet, dev, or local",);
                return Ok(());
            }
        };

        let raw_genesis_storage = {
            let storage = domain_chain_spec
                .build_storage()
                .expect("Failed to build genesis storage from genesis runtime config");
            let raw_genesis = RawGenesis::from_storage(storage);
            raw_genesis.encode()
        };

        if std::io::stdout()
            .write_all(raw_genesis_storage.as_ref())
            .is_err()
        {
            let _ = std::io::stderr().write_all(b"Error writing to stdout\n");
        }
        Ok(())
    }
}

/// The `export-execution-receipt` command used to get the ER from the auxiliary storage of the operator client
#[derive(Debug, Clone, Parser)]
pub struct ExportExecutionReceiptCmd {
    /// Get the `ExecutionReceipt` by domain block number or hash
    #[arg(long, conflicts_with_all = &["consensus_block_hash"])]
    pub domain_block: Option<BlockNumberOrHash>,

    /// Get the `ExecutionReceipt` by consensus block hash
    #[arg(long, conflicts_with_all = &["domain_block"])]
    pub consensus_block_hash: Option<BlockNumberOrHash>,

    /// The base struct of the export-execution-receipt command.
    #[clap(flatten)]
    pub shared_params: SharedParams,

    /// Domain arguments
    ///
    /// The command-line arguments provided first will be passed to the embedded consensus node,
    /// while the arguments provided after `--` will be passed to the domain node.
    ///
    /// subspace-node export-execution-receipt [consensus-chain-args] -- [domain-args]
    #[arg(raw = true)]
    pub domain_args: Vec<String>,
}

impl CliConfiguration for ExportExecutionReceiptCmd {
    fn shared_params(&self) -> &SharedParams {
        &self.shared_params
    }
}

impl ExportExecutionReceiptCmd {
    /// Run the export-execution-receipt command
    pub fn run<Backend, Client>(
        &self,
        domain_client: &Client,
        domain_backend: &Backend,
    ) -> sc_cli::Result<()>
    where
        Backend: AuxStore,
        Client: HeaderBackend<DomainBlock>,
    {
        let consensus_block_hash = match (&self.consensus_block_hash, &self.domain_block) {
            // Get ER by consensus block hash
            (Some(raw_consensus_block_hash), None) => {
                match raw_consensus_block_hash.parse::<Block>()? {
                    BlockId::Hash(h) => h,
                    BlockId::Number(_) => {
                        eprintln!(
                            "unexpected input {raw_consensus_block_hash:?}, expected consensus block hash",
                        );
                        return Ok(());
                    }
                }
            }
            // Get ER by domain block hash or number
            (None, Some(raw_domain_block)) => {
                let domain_block_hash = match raw_domain_block.parse::<DomainBlock>()? {
                    BlockId::Hash(h) => h,
                    BlockId::Number(number) => domain_client.hash(number)?.ok_or_else(|| {
                        sp_blockchain::Error::Backend(format!(
                            "Domain block hash for #{number:?} not found",
                        ))
                    })?,
                };
                let domain_header = domain_client.header(domain_block_hash)?.ok_or_else(|| {
                    sp_blockchain::Error::Backend(format!(
                        "Header for domain block {domain_block_hash:?} not found"
                    ))
                })?;

                domain_header
                    .digest()
                    .convert_first(DigestItem::as_consensus_block_info)
                    .ok_or_else(|| {
                        sp_blockchain::Error::Application(Box::from(
                            "Domain block header for {domain_hash:?} must have consensus block info predigest"
                        ))
                    })?
            }
            _ => {
                eprintln!("Expect the domain-block or consensus-block-hash argument",);
                return Ok(());
            }
        };

        match domain_client_operator::load_execution_receipt::<Backend, DomainBlock, Block>(
            domain_backend,
            consensus_block_hash,
        )? {
            Some(er) => {
                println!("ExecutionReceipt of consensus block {consensus_block_hash:?}:\n{er:?}",);
            }
            None => {
                println!("ExecutionReceipt of consensus block {consensus_block_hash:?} not found",);
            }
        }
        Ok(())
    }
}

/// Genesis domain
pub struct GenesisDomain {
    /// encoded raw genesis
    pub raw_genesis: Vec<u8>,
    pub runtime_name: String,
    pub runtime_type: RuntimeType,
    pub runtime_version: RuntimeVersion,
    pub domain_name: String,
    pub initial_balances: Vec<(MultiAccountId, Balance)>,
    pub operator_allow_list: OperatorAllowList<AccountId>,
    pub operator_signing_key: OperatorPublicKey,
    pub domain_runtime_config: DomainRuntimeConfig,
}

/// Genesis Operator list params
pub(crate) struct GenesisOperatorParams {
    pub operator_allow_list: OperatorAllowList<subspace_runtime_primitives::AccountId>,
    pub operator_signing_key: OperatorPublicKey,
}
