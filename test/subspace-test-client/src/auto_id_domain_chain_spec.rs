//! Chain specification for the auto-id domain.

use crate::chain_spec::get_from_seed;
use auto_id_domain_test_runtime::{BalancesConfig, RuntimeGenesisConfig, SystemConfig};
use codec::Encode;
use domain_runtime_primitives::AccountIdConverter;
use sc_chain_spec::{ChainType, GenericChainSpec, NoExtension};
use sp_core::crypto::AccountId32;
use sp_core::{sr25519, Pair, Public};
use sp_domains::storage::RawGenesis;
use sp_domains::{
    DomainRuntimeConfig, GenesisDomain, OperatorAllowList, OperatorPublicKey, RuntimeType,
};
use sp_runtime::traits::{Convert, IdentifyAccount};
use sp_runtime::{BuildStorage, MultiSigner, Percent};
use subspace_runtime_primitives::{AccountId, Balance, SSC};

/// Get public key from keypair seed.
pub(crate) fn get_public_key_from_seed<TPublic: Public>(
    seed: &'static str,
) -> <TPublic::Pair as Pair>::Public {
    TPublic::Pair::from_string(&format!("//{seed}"), None)
        .expect("Static values are valid; qed")
        .public()
}

/// Generate an account ID from seed.
pub(crate) fn get_account_id_from_seed(seed: &'static str) -> AccountId32 {
    MultiSigner::from(get_public_key_from_seed::<sr25519::Public>(seed)).into_account()
}

pub(crate) fn endowed_accounts() -> Vec<AccountId32> {
    vec![
        get_account_id_from_seed("Alice"),
        get_account_id_from_seed("Bob"),
        get_account_id_from_seed("Charlie"),
        get_account_id_from_seed("Dave"),
        get_account_id_from_seed("Eve"),
        get_account_id_from_seed("Ferdie"),
        get_account_id_from_seed("Alice//stash"),
        get_account_id_from_seed("Bob//stash"),
        get_account_id_from_seed("Charlie//stash"),
        get_account_id_from_seed("Dave//stash"),
        get_account_id_from_seed("Eve//stash"),
        get_account_id_from_seed("Ferdie//stash"),
    ]
}

fn testnet_auto_id_genesis() -> RuntimeGenesisConfig {
    RuntimeGenesisConfig {
        system: SystemConfig::default(),
        balances: BalancesConfig::default(),
        ..Default::default()
    }
}

pub fn get_genesis_domain(
    sudo_account: subspace_runtime_primitives::AccountId,
) -> Result<GenesisDomain<AccountId, Balance>, String> {
    let raw_genesis_storage = {
        let domain_chain_spec = GenericChainSpec::<NoExtension, ()>::builder(
            auto_id_domain_test_runtime::WASM_BINARY
                .ok_or_else(|| "Development wasm not available".to_string())?,
            None,
        )
        .with_chain_type(ChainType::Development)
        .with_genesis_config(
            serde_json::to_value(testnet_auto_id_genesis())
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
        runtime_name: "auto-id".to_owned(),
        runtime_type: RuntimeType::AutoId,
        runtime_version: auto_id_domain_test_runtime::VERSION,
        raw_genesis_storage,

        // Domain config, mainly for placeholder the concrete value TBD
        owner_account_id: sudo_account,
        domain_name: "auto-id-domain".to_owned(),
        bundle_slot_probability: (1, 1),
        operator_allow_list: OperatorAllowList::Anyone,

        signing_key: get_from_seed::<OperatorPublicKey>("Bob"),
        minimum_nominator_stake: 100 * SSC,
        nomination_tax: Percent::from_percent(5),
        initial_balances: endowed_accounts()
            .iter()
            .cloned()
            .map(|k| (AccountIdConverter::convert(k), 2_000_000 * SSC))
            .collect(),

        domain_runtime_config: DomainRuntimeConfig::default_auto_id(),
    })
}
