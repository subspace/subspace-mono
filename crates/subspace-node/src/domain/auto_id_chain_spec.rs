//! AutoId domain configurations.

use crate::chain_spec_utils::{
    chain_spec_properties, get_account_id_from_seed, get_public_key_from_seed,
};
use crate::domain::cli::{GenesisDomain, GenesisOperatorParams, SpecId};
use auto_id_domain_runtime::{BalancesConfig, RuntimeGenesisConfig, SystemConfig, WASM_BINARY};
use domain_runtime_primitives::{AccountIdConverter, MultiAccountId};
use hex_literal::hex;
use parity_scale_codec::Encode;
use sc_chain_spec::GenericChainSpec;
use sc_service::ChainType;
use sp_core::crypto::{AccountId32, UncheckedFrom};
use sp_domains::storage::RawGenesis;
use sp_domains::{DomainRuntimeConfig, OperatorAllowList, OperatorPublicKey, RuntimeType};
use sp_runtime::traits::Convert;
use sp_runtime::BuildStorage;
use std::collections::BTreeSet;
use subspace_runtime_primitives::{Balance, SSC};

/// Development keys that will be injected automatically on polkadotjs apps
fn get_dev_accounts() -> Vec<AccountId32> {
    vec![
        get_account_id_from_seed("Alice"),
        get_account_id_from_seed("Bob"),
        get_account_id_from_seed("Alice//stash"),
        get_account_id_from_seed("Bob//stash"),
    ]
}

pub fn development_config(
    runtime_genesis_config: RuntimeGenesisConfig,
) -> Result<GenericChainSpec, String> {
    Ok(GenericChainSpec::builder(
        WASM_BINARY.ok_or_else(|| "WASM binary was not build, please build it!".to_string())?,
        None,
    )
    .with_name("Development")
    .with_id("auto_id_domain_dev")
    .with_chain_type(ChainType::Development)
    .with_genesis_config(
        serde_json::to_value(runtime_genesis_config)
            .map_err(|error| format!("Failed to serialize genesis config: {error}"))?,
    )
    .with_properties(chain_spec_properties())
    .build())
}

pub fn taurus_config(
    runtime_genesis_config: RuntimeGenesisConfig,
) -> Result<GenericChainSpec, String> {
    Ok(GenericChainSpec::builder(
        WASM_BINARY.ok_or_else(|| "WASM binary was not build, please build it!".to_string())?,
        None,
    )
    .with_name("Autonomys Taurus AutoId Domain")
    .with_id("autonomys_taurus_auto_id_domain")
    .with_chain_type(ChainType::Live)
    .with_genesis_config(
        serde_json::to_value(runtime_genesis_config)
            .map_err(|error| format!("Failed to serialize genesis config: {error}"))?,
    )
    .with_protocol_id("autonomys-taurus-auto-id-domain")
    .with_properties(chain_spec_properties())
    .build())
}

pub fn devnet_config(
    runtime_genesis_config: RuntimeGenesisConfig,
) -> Result<GenericChainSpec, String> {
    Ok(GenericChainSpec::builder(
        WASM_BINARY.ok_or_else(|| "WASM binary was not build, please build it!".to_string())?,
        None,
    )
    .with_name("Subspace Devnet AutoId Domain")
    .with_id("subspace_devnet_auto_id_domain")
    .with_chain_type(ChainType::Custom("Testnet".to_string()))
    .with_genesis_config(
        serde_json::to_value(runtime_genesis_config)
            .map_err(|error| format!("Failed to serialize genesis config: {error}"))?,
    )
    .with_protocol_id("subspace-devnet-auto-id-domain")
    .with_properties(chain_spec_properties())
    .build())
}

pub fn load_chain_spec(spec_id: &str) -> Result<Box<dyn sc_cli::ChainSpec>, String> {
    let chain_spec = match spec_id {
        "taurus" => taurus_config(get_testnet_genesis_by_spec_id(SpecId::Taurus))?,
        "devnet" => devnet_config(get_testnet_genesis_by_spec_id(SpecId::DevNet))?,
        "dev" => development_config(get_testnet_genesis_by_spec_id(SpecId::Dev))?,
        path => GenericChainSpec::from_json_file(std::path::PathBuf::from(path))?,
    };
    Ok(Box::new(chain_spec))
}

pub fn get_testnet_genesis_by_spec_id(spec_id: SpecId) -> RuntimeGenesisConfig {
    match spec_id {
        SpecId::Dev => testnet_genesis(),
        SpecId::Taurus => testnet_genesis(),
        SpecId::DevNet => testnet_genesis(),
    }
}

pub fn get_testnet_endowed_accounts_by_spec_id(spec_id: SpecId) -> Vec<(MultiAccountId, Balance)> {
    match spec_id {
        SpecId::Dev => get_dev_accounts()
            .into_iter()
            .map(|acc| (AccountIdConverter::convert(acc), 1_000_000 * SSC))
            .collect(),
        SpecId::DevNet => {
            let accounts = get_dev_accounts();
            let alice_account = accounts[0].clone();
            vec![(AccountIdConverter::convert(alice_account), 1_000_000 * SSC)]
        }
        SpecId::Taurus => vec![],
    }
}

fn testnet_genesis() -> RuntimeGenesisConfig {
    RuntimeGenesisConfig {
        system: SystemConfig::default(),
        balances: BalancesConfig::default(),
        ..Default::default()
    }
}

fn get_operator_params(
    spec_id: SpecId,
    sudo_account: subspace_runtime_primitives::AccountId,
) -> GenesisOperatorParams {
    match spec_id {
        SpecId::Dev => GenesisOperatorParams {
            operator_allow_list: OperatorAllowList::Anyone,
            operator_signing_key: get_public_key_from_seed::<OperatorPublicKey>("Alice"),
        },
        SpecId::Taurus => GenesisOperatorParams {
            operator_allow_list: OperatorAllowList::Operators(BTreeSet::from_iter(vec![
                sudo_account.clone(),
            ])),
            operator_signing_key: OperatorPublicKey::unchecked_from(hex!(
                "3458e79cd1f106a4a7eaed78b46fe97dffcf0f619d1278a4dd4e4c9e862fc348"
            )),
        },
        SpecId::DevNet => GenesisOperatorParams {
            operator_allow_list: OperatorAllowList::Anyone,
            operator_signing_key: OperatorPublicKey::unchecked_from(hex!(
                "18df97b9335e11f239f8f3f8041819d42f27b60845cf209416fdba8de15f4b7c"
            )),
        },
    }
}

pub fn get_genesis_domain(
    spec_id: SpecId,
    sudo_account: subspace_runtime_primitives::AccountId,
) -> Result<GenesisDomain, String> {
    let chain_spec = match spec_id {
        SpecId::Dev => development_config(get_testnet_genesis_by_spec_id(spec_id))?,
        SpecId::Taurus => taurus_config(get_testnet_genesis_by_spec_id(spec_id))?,
        SpecId::DevNet => devnet_config(get_testnet_genesis_by_spec_id(spec_id))?,
    };

    let GenesisOperatorParams {
        operator_allow_list,
        operator_signing_key,
    } = get_operator_params(spec_id, sudo_account);

    let storage = chain_spec
        .build_storage()
        .expect("Failed to build genesis storage from genesis runtime config");
    let raw_genesis = RawGenesis::from_storage(storage);
    Ok(GenesisDomain {
        raw_genesis: raw_genesis.encode(),
        runtime_name: "auto-id".to_string(),
        runtime_type: RuntimeType::AutoId,
        runtime_version: auto_id_domain_runtime::VERSION,
        domain_name: "auto-id".to_string(),
        initial_balances: get_testnet_endowed_accounts_by_spec_id(spec_id),
        operator_allow_list,
        operator_signing_key,
        domain_runtime_config: DomainRuntimeConfig::default_auto_id(),
    })
}
