//! Chain specification for the evm domain.

use crate::chain_spec::get_from_seed;
use codec::Encode;
use domain_runtime_primitives::AccountId20Converter;
use evm_domain_test_runtime::{
    AccountId as AccountId20, Precompiles, RuntimeGenesisConfig, Signature,
};
use sc_chain_spec::{ChainType, GenericChainSpec, NoExtension};
use sp_core::{ecdsa, Pair, Public};
use sp_domains::storage::RawGenesis;
use sp_domains::{
    DomainId, DomainRuntimeConfig, GenesisDomain, OperatorAllowList, OperatorPublicKey, RuntimeType,
};
use sp_runtime::traits::{Convert, IdentifyAccount, Verify};
use sp_runtime::{BuildStorage, Percent};
use subspace_runtime_primitives::{AccountId, Balance, SSC};

type AccountPublic = <Signature as Verify>::Signer;

/// Helper function to generate an account ID from seed.
pub fn get_account_id_from_seed<TPublic: Public>(seed: &str) -> AccountId20
where
    AccountPublic: From<<TPublic::Pair as Pair>::Public>,
{
    AccountPublic::from(
        TPublic::Pair::from_string(&format!("//{seed}"), None)
            .expect("static values are valid; qed")
            .public(),
    )
    .into_account()
}

pub(crate) fn endowed_accounts() -> Vec<AccountId20> {
    vec![
        get_account_id_from_seed::<ecdsa::Public>("Alice"),
        get_account_id_from_seed::<ecdsa::Public>("Bob"),
        get_account_id_from_seed::<ecdsa::Public>("Charlie"),
        get_account_id_from_seed::<ecdsa::Public>("Dave"),
        get_account_id_from_seed::<ecdsa::Public>("Eve"),
        get_account_id_from_seed::<ecdsa::Public>("Ferdie"),
        get_account_id_from_seed::<ecdsa::Public>("Alice//stash"),
        get_account_id_from_seed::<ecdsa::Public>("Bob//stash"),
        get_account_id_from_seed::<ecdsa::Public>("Charlie//stash"),
        get_account_id_from_seed::<ecdsa::Public>("Dave//stash"),
        get_account_id_from_seed::<ecdsa::Public>("Eve//stash"),
        get_account_id_from_seed::<ecdsa::Public>("Ferdie//stash"),
    ]
}

/// Get the genesis config of the evm domain
pub fn testnet_evm_genesis() -> RuntimeGenesisConfig {
    // This is the simplest bytecode to revert without returning any data.
    // We will pre-deploy it under all of our precompiles to ensure they can be called from
    // within contracts.
    // (PUSH1 0x00 PUSH1 0x00 REVERT)
    let revert_bytecode = vec![0x60, 0x00, 0x60, 0x00, 0xFD];

    RuntimeGenesisConfig {
        system: evm_domain_test_runtime::SystemConfig::default(),
        balances: evm_domain_test_runtime::BalancesConfig::default(),
        evm_chain_id: evm_domain_test_runtime::EVMChainIdConfig {
            chain_id: 100,
            ..Default::default()
        },
        evm: evm_domain_test_runtime::EVMConfig {
            // We need _some_ code inserted at the precompile address so that
            // the evm will actually call the address.
            accounts: Precompiles::used_addresses()
                .into_iter()
                .map(|addr| {
                    (
                        addr,
                        fp_evm::GenesisAccount {
                            nonce: Default::default(),
                            balance: Default::default(),
                            storage: Default::default(),
                            code: revert_bytecode.clone(),
                        },
                    )
                })
                .collect(),
            ..Default::default()
        },
        self_domain_id: evm_domain_test_runtime::SelfDomainIdConfig {
            // Set the domain id of the genesis domain to an arbitrary value
            // it should be overwritten with the correct value
            domain_id: Some(DomainId::new(123)),
            ..Default::default()
        },
        ..Default::default()
    }
}

pub fn get_genesis_domain(
    sudo_account: subspace_runtime_primitives::AccountId,
) -> Result<GenesisDomain<AccountId, Balance>, String> {
    let raw_genesis_storage = {
        let domain_chain_spec = GenericChainSpec::<NoExtension, ()>::builder(
            evm_domain_test_runtime::WASM_BINARY
                .ok_or_else(|| "Development wasm not available".to_string())?,
            None,
        )
        .with_chain_type(ChainType::Development)
        .with_genesis_config(
            serde_json::to_value(testnet_evm_genesis())
                .map_err(|error| format!("Failed to serialize genesis config: {error}"))?,
        )
        .build();
        let storage = domain_chain_spec
            .build_storage()
            .expect("Failed to build genesis storage from genesis runtime config");
        let raw_genesis = RawGenesis::from_storage(storage);
        raw_genesis.encode()
    };

    Ok(GenesisDomain {
        runtime_name: "evm".to_owned(),
        runtime_type: RuntimeType::Evm,
        runtime_version: evm_domain_test_runtime::VERSION,
        raw_genesis_storage,

        // Domain config, mainly for placeholder the concrete value TBD
        owner_account_id: sudo_account,
        domain_name: "evm-domain".to_owned(),
        bundle_slot_probability: (1, 1),
        operator_allow_list: OperatorAllowList::Anyone,

        signing_key: get_from_seed::<OperatorPublicKey>("Alice"),
        minimum_nominator_stake: 100 * SSC,
        nomination_tax: Percent::from_percent(5),
        initial_balances: endowed_accounts()
            .iter()
            .cloned()
            .map(|k| (AccountId20Converter::convert(k), 2_000_000 * SSC))
            .collect(),

        domain_runtime_config: DomainRuntimeConfig::default_evm(),
    })
}
