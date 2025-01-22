//! Test setup shared by sp-domains-fraud-proof and domain-operator tests

use crate::test_ethereum_tx::{
    EIP1559UnsignedTransaction, EIP2930UnsignedTransaction, LegacyUnsignedTransaction,
};
use ethereum::TransactionV2 as Transaction;
use frame_support::pallet_prelude::DispatchClass;
use pallet_evm::GasWeightMapping;
use sp_core::{keccak_256, Get, H160, H256, U256};

#[derive(Clone)]
pub struct AccountInfo {
    pub address: H160,
    pub private_key: H256,
}

pub fn address_build(seed_number: u128) -> AccountInfo {
    let mut seed = [0u8; 32];
    seed[0..16].copy_from_slice(&seed_number.to_be_bytes());
    let private_key = H256::from_slice(&seed);
    let secret_key = libsecp256k1::SecretKey::parse_slice(&private_key[..]).unwrap();
    let public_key = &libsecp256k1::PublicKey::from_secret_key(&secret_key).serialize()[1..65];
    let address = H160::from(H256::from(keccak_256(public_key)));

    let mut data = [0u8; 32];
    data[0..20].copy_from_slice(&address[..]);

    AccountInfo {
        private_key,
        address,
    }
}

pub fn generate_legacy_tx<TestRuntime: frame_system::Config + pallet_evm::Config>(
    account_info: AccountInfo,
    nonce: U256,
    action: ethereum::TransactionAction,
    input: Vec<u8>,
    gas_price: U256,
) -> Transaction {
    let limits: frame_system::limits::BlockWeights =
        <TestRuntime as frame_system::Config>::BlockWeights::get();
    // `limits.get(DispatchClass::Normal).max_extrinsic` is too large to use as `gas_limit`
    // thus use `base_extrinsic`
    let max_extrinsic = limits.get(DispatchClass::Normal).base_extrinsic * 1000;
    let max_extrinsic_gas =
        <TestRuntime as pallet_evm::Config>::GasWeightMapping::weight_to_gas(max_extrinsic);

    LegacyUnsignedTransaction {
        nonce,
        gas_price,
        gas_limit: U256::from(max_extrinsic_gas),
        action,
        value: U256::zero(),
        input,
    }
    .sign(&account_info.private_key)
}

pub fn generate_eip2930_tx<TestRuntime: frame_system::Config + pallet_evm::Config>(
    account_info: AccountInfo,
    nonce: U256,
    action: ethereum::TransactionAction,
    input: Vec<u8>,
    gas_price: U256,
) -> Transaction {
    let limits: frame_system::limits::BlockWeights =
        <TestRuntime as frame_system::Config>::BlockWeights::get();
    // `limits.get(DispatchClass::Normal).max_extrinsic` is too large to use as `gas_limit`
    // thus use `base_extrinsic`
    let max_extrinsic = limits.get(DispatchClass::Normal).base_extrinsic * 100;
    let max_extrinsic_gas =
        <TestRuntime as pallet_evm::Config>::GasWeightMapping::weight_to_gas(max_extrinsic);

    EIP2930UnsignedTransaction {
        nonce,
        gas_price,
        gas_limit: U256::from(max_extrinsic_gas),
        action,
        value: U256::one(),
        input,
    }
    .sign(&account_info.private_key, None)
}

pub fn generate_eip1559_tx<TestRuntime: frame_system::Config + pallet_evm::Config>(
    account_info: AccountInfo,
    nonce: U256,
    action: ethereum::TransactionAction,
    input: Vec<u8>,
    gas_price: U256,
) -> Transaction {
    let limits: frame_system::limits::BlockWeights =
        <TestRuntime as frame_system::Config>::BlockWeights::get();
    // `limits.get(DispatchClass::Normal).max_extrinsic` is too large to use as `gas_limit`
    // thus use `base_extrinsic`
    let max_extrinsic = limits.get(DispatchClass::Normal).base_extrinsic * 1000;
    let max_extrinsic_gas =
        <TestRuntime as pallet_evm::Config>::GasWeightMapping::weight_to_gas(max_extrinsic);

    EIP1559UnsignedTransaction {
        nonce,
        max_priority_fee_per_gas: U256::from(1),
        max_fee_per_gas: gas_price,
        gas_limit: U256::from(max_extrinsic_gas),
        action,
        value: U256::zero(),
        input,
    }
    .sign(&account_info.private_key, None)
}
