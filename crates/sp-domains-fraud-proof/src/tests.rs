use codec::Encode;
use domain_runtime_primitives::{Balance, CheckExtrinsicsValidityError};
use domain_test_service::evm_domain_test_runtime::{
    Runtime as TestRuntime, RuntimeCall, Signature, UncheckedExtrinsic as RuntimeUncheckedExtrinsic,
};
use domain_test_service::EcdsaKeyring::{Alice, Charlie};
use domain_test_service::Sr25519Keyring::Ferdie;
use domain_test_service::{construct_extrinsic_raw_payload, EvmDomainNode};
use ethereum::TransactionV2 as Transaction;
use fp_rpc::EthereumRuntimeRPCApi;
use pallet_ethereum::Call;
use rand::distributions::{Distribution, Uniform};
use sc_client_api::{HeaderBackend, StorageProof};
use sc_service::{BasePath, Role};
use sp_api::{ApiExt, ProvideRuntimeApi, TransactionOutcome};
use sp_core::ecdsa::Pair;
use sp_core::{keccak_256, Pair as _, U256};
use sp_domains::core_api::DomainCoreApi;
use sp_domains::test_ethereum::{
    address_build, generate_eip1559_tx, generate_eip2930_tx, generate_legacy_tx, AccountInfo,
};
use sp_runtime::traits::{Extrinsic, Zero};
use sp_runtime::transaction_validity::{InvalidTransaction, TransactionValidityError};
use sp_runtime::OpaqueExtrinsic;
use subspace_test_service::{produce_block_with, produce_blocks, MockConsensusNode};
use tempfile::TempDir;

/// Generate a self-contained EVM domain extrinsic.
// This function depends on the macro-constructed `TestRuntime::RuntimeCall` enum, so it can't be
// shared via `sp_domains::test_ethereum`.
pub fn generate_evm_domain_extrinsic(tx: Transaction) -> RuntimeUncheckedExtrinsic {
    let call = Call::<TestRuntime>::transact { transaction: tx };
    fp_self_contained::UncheckedExtrinsic::new(RuntimeCall::Ethereum(call), None).unwrap()
}

async fn benchmark_bundle_with_evm_tx(
    tx_to_create: u32,
    mut alice: EvmDomainNode,
    mut ferdie: MockConsensusNode,
) -> (Vec<Vec<u8>>, StorageProof) {
    let account_infos = (1..tx_to_create)
        .map(|i| address_build(i as u128))
        .collect::<Vec<AccountInfo>>();

    let alice_balance = alice.free_balance(Alice.to_account_id());

    let other_accounts_balance = alice_balance / (tx_to_create as u128 + 2);

    let mut alice_nonce = alice.account_nonce();
    for account_info in &account_infos {
        let alice_balance_transfer_extrinsic = alice.construct_extrinsic(
            alice_nonce,
            pallet_balances::Call::transfer_allow_death {
                dest: account_info.address.0.into(),
                value: other_accounts_balance,
            },
        );
        alice
            .send_extrinsic(alice_balance_transfer_extrinsic)
            .await
            .expect("Failed to send extrinsic");
        alice_nonce += 1;
    }

    const TX_TYPES: u32 = 4;
    let mut thread_rng = rand::thread_rng();
    let between = Uniform::from(0..TX_TYPES);
    let (slot, _) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();
    produce_blocks!(ferdie, alice, 1).await.unwrap();

    let gas_price = alice
        .client
        .runtime_api()
        .gas_price(alice.client.info().best_hash)
        .unwrap();

    let mut runtime_instance = alice.client.runtime_api();
    runtime_instance.record_proof();

    // Each account can then either do: substrate balance transfer or evm transaction
    // call `validate_transaction_and_do_pre_dispatch` on every tx
    for account_info in account_infos.iter() {
        let tx_type_to_use = between.sample(&mut thread_rng);
        assert!(
            tx_type_to_use <= TX_TYPES,
            "random tx_type is out of bounds"
        );
        let extrinsic = match tx_type_to_use {
            0 => {
                let evm_tx = generate_eip1559_tx::<TestRuntime>(
                    account_info.clone(),
                    U256::zero(),
                    ethereum::TransactionAction::Create,
                    vec![1u8; 100],
                    gas_price,
                );
                generate_evm_domain_extrinsic(evm_tx)
            }
            1 => {
                let evm_tx = generate_eip2930_tx::<TestRuntime>(
                    account_info.clone(),
                    U256::zero(),
                    ethereum::TransactionAction::Create,
                    vec![1u8; 100],
                    gas_price,
                );
                generate_evm_domain_extrinsic(evm_tx)
            }
            2 => {
                let evm_tx = generate_legacy_tx::<TestRuntime>(
                    account_info.clone(),
                    U256::zero(),
                    ethereum::TransactionAction::Create,
                    vec![1u8; 100],
                    gas_price,
                );
                generate_evm_domain_extrinsic(evm_tx)
            }
            3 => {
                let ecdsa_key = Pair::from_seed_slice(&account_info.private_key.0).unwrap();
                let function: RuntimeCall = pallet_balances::Call::transfer_allow_death {
                    dest: account_info.address.0.into(),
                    value: other_accounts_balance,
                }
                .into();
                let (raw_payload, extra) =
                    construct_extrinsic_raw_payload(&alice.client, function.clone(), false, 0, 1);
                let signature = raw_payload.using_encoded(|e| {
                    let msg = keccak_256(e);
                    ecdsa_key.sign_prehashed(&msg)
                });
                fp_self_contained::UncheckedExtrinsic(
                    sp_runtime::generic::UncheckedExtrinsic::new_signed(
                        function.clone(),
                        ecdsa_key.public().into(),
                        Signature::new(signature),
                        extra,
                    ),
                )
            }
            _ => {
                panic!("not implemented");
            }
        };

        assert_eq!(
            runtime_instance
                .check_extrinsics_and_do_pre_dispatch(
                    alice.client.as_ref().info().best_hash,
                    vec![extrinsic.clone().into()],
                    alice.client.as_ref().info().best_number,
                    alice.client.as_ref().info().best_hash,
                )
                .unwrap(),
            Ok(())
        );
    }

    // get the storage proof afterwards and measure its size
    let proof_recorder = runtime_instance.proof_recorder().unwrap();
    let storage_proof = proof_recorder.to_storage_proof();
    let recorded_keys = proof_recorder
        .recorded_keys()
        .drain()
        .next()
        .unwrap()
        .1
        .drain()
        .map(|(k, _v)| k.to_vec())
        .collect::<Vec<Vec<u8>>>();
    (recorded_keys, storage_proof)
}

#[tokio::test(flavor = "multi_thread")]
async fn domain_bundle_storage_proof_benchmark() {
    let directory = TempDir::new().expect("Must be able to create temporary directory");

    let mut builder = sc_cli::LoggerBuilder::new("");
    builder.with_colors(false);
    let _ = builder.init();

    let tokio_handle = tokio::runtime::Handle::current();

    // Start Ferdie
    let mut ferdie = MockConsensusNode::run(
        tokio_handle.clone(),
        Ferdie,
        BasePath::new(directory.path().join("ferdie")),
    );

    // Run Alice (a evm domain authority node)
    let alice = domain_test_service::DomainNodeBuilder::new(
        tokio_handle.clone(),
        BasePath::new(directory.path().join("alice")),
    )
    .build_evm_node(Role::Authority, Alice, &mut ferdie)
    .await;

    let (recorded_keys, storage_proof) = benchmark_bundle_with_evm_tx(400, alice, ferdie).await;
    println!(
        "Result: Storage proof nodes: {:?}, keys accessed: {:?}, Storage proof total size (in mem): {:?}, Storage proof total size (in wire): {:?}",
        storage_proof.iter_nodes().count(),
        recorded_keys.len(),
        storage_proof.clone()
            .into_iter_nodes()
            .collect::<Vec<Vec<u8>>>()
            .iter()
            .fold(0, |size, element| size + element.len()),
        storage_proof.encoded_size()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn storage_change_of_the_same_runtime_instance_should_perserved_cross_runtime_calls() {
    let directory = TempDir::new().expect("Must be able to create temporary directory");

    let mut builder = sc_cli::LoggerBuilder::new("");
    builder.with_colors(false);
    let _ = builder.init();

    let tokio_handle = tokio::runtime::Handle::current();

    // Start Ferdie
    let mut ferdie = MockConsensusNode::run(
        tokio_handle.clone(),
        Ferdie,
        BasePath::new(directory.path().join("ferdie")),
    );

    // Run Alice (a evm domain authority node)
    let mut alice = domain_test_service::DomainNodeBuilder::new(
        tokio_handle.clone(),
        BasePath::new(directory.path().join("alice")),
    )
    .build_evm_node(Role::Authority, Alice, &mut ferdie)
    .await;

    let best_hash = alice.client.as_ref().info().best_hash;
    let best_number = alice.client.as_ref().info().best_number;

    // transfer most of the alice's balance
    let alice_balance = alice.free_balance(Alice.to_account_id());

    let mut alice_nonce = alice.account_nonce();

    let transfer_to_charlie_with_big_tip_1 = alice.construct_extrinsic_with_tip(
        alice_nonce,
        (alice_balance / 3) * 2,
        pallet_balances::Call::transfer_allow_death {
            dest: Charlie.to_account_id(),
            value: 1,
        },
    );
    alice_nonce += 1;

    let transfer_to_charlie_with_big_tip_2 = alice.construct_extrinsic_with_tip(
        alice_nonce,
        (alice_balance / 3) * 2,
        pallet_balances::Call::transfer_allow_death {
            dest: Charlie.to_account_id(),
            value: 1,
        },
    );

    // A runtime instance preserve changes in between
    let runtime_instance = alice.client.runtime_api();

    // First transaction should be okay
    // Second transaction  should error out with exact error encountered during check of bundle validity
    // to prove that state is preserved between two calls to same runtime instance
    assert_eq!(
        runtime_instance
            .check_extrinsics_and_do_pre_dispatch(
                best_hash,
                vec![
                    transfer_to_charlie_with_big_tip_1.clone().into(),
                    transfer_to_charlie_with_big_tip_2.clone().into(),
                ],
                best_number,
                best_hash,
            )
            .unwrap(),
        Err(CheckExtrinsicsValidityError {
            extrinsic_index: 1,
            transaction_validity_error: TransactionValidityError::Invalid(
                InvalidTransaction::Payment
            ),
        })
    );

    // If second tx is executed in dedicated runtime instance it would error with `InvalidTransaction::Future`
    // as the nonce at the block for this account is less than the tx's nonce. And there is no overlay
    // preserved between runtime instance.
    assert_eq!(
        alice
            .client
            .runtime_api()
            .check_extrinsics_and_do_pre_dispatch(
                best_hash,
                vec![transfer_to_charlie_with_big_tip_2.clone().into()],
                best_number,
                best_hash,
            )
            .unwrap(),
        Err(CheckExtrinsicsValidityError {
            extrinsic_index: 0,
            transaction_validity_error: TransactionValidityError::Invalid(
                InvalidTransaction::Future
            ),
        })
    );

    let test_commit_mode = vec![true, false];

    for commit_mode in test_commit_mode {
        alice_nonce = alice.account_nonce();

        let transfer_with_big_tip_1 = alice.construct_extrinsic_with_tip(
            alice_nonce,
            alice_balance / 3,
            pallet_balances::Call::transfer_allow_death {
                dest: Charlie.to_account_id(),
                value: 1,
            },
        );
        alice_nonce += 1;

        let transfer_with_big_tip_2 = alice.construct_extrinsic_with_tip(
            alice_nonce,
            alice_balance / 3,
            pallet_balances::Call::transfer_allow_death {
                dest: Charlie.to_account_id(),
                value: 1,
            },
        );

        if commit_mode {
            alice_nonce += 1;
        }

        let transfer_with_big_tip_3 = alice.construct_extrinsic_with_tip(
            alice_nonce,
            alice_balance / 3,
            pallet_balances::Call::transfer_allow_death {
                dest: Charlie.to_account_id(),
                value: 1,
            },
        );

        // A runtime instance can rollback changes safely.
        let runtime_instance = alice.client.runtime_api();

        assert!(runtime_instance
            .check_extrinsics_and_do_pre_dispatch(
                best_hash,
                vec![transfer_with_big_tip_1.clone().into()],
                best_number,
                best_hash,
            )
            .unwrap()
            .is_ok());

        assert!(runtime_instance
            .execute_in_transaction(|api| {
                if commit_mode {
                    TransactionOutcome::Commit(api.check_extrinsics_and_do_pre_dispatch(
                        best_hash,
                        vec![transfer_with_big_tip_2.clone().into()],
                        best_number,
                        best_hash,
                    ))
                } else {
                    TransactionOutcome::Rollback(api.check_extrinsics_and_do_pre_dispatch(
                        best_hash,
                        vec![transfer_with_big_tip_2.clone().into()],
                        best_number,
                        best_hash,
                    ))
                }
            })
            .is_ok());

        assert_eq!(
            runtime_instance
                .check_extrinsics_and_do_pre_dispatch(
                    best_hash,
                    vec![transfer_with_big_tip_3.clone().into()],
                    best_number,
                    best_hash,
                )
                .unwrap(),
            if commit_mode {
                Err(CheckExtrinsicsValidityError {
                    extrinsic_index: 0,
                    transaction_validity_error: TransactionValidityError::Invalid(
                        InvalidTransaction::Payment,
                    ),
                })
            } else {
                Ok(())
            }
        )
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn check_bundle_validity_runtime_api_should_work() {
    let directory = TempDir::new().expect("Must be able to create temporary directory");

    let mut builder = sc_cli::LoggerBuilder::new("");
    builder.with_colors(false);
    let _ = builder.init();

    let tokio_handle = tokio::runtime::Handle::current();

    // Start Ferdie
    let mut ferdie = MockConsensusNode::run(
        tokio_handle.clone(),
        Ferdie,
        BasePath::new(directory.path().join("ferdie")),
    );

    // Run Alice (a evm domain authority node)
    let mut alice = domain_test_service::DomainNodeBuilder::new(
        tokio_handle.clone(),
        BasePath::new(directory.path().join("alice")),
    )
    .build_evm_node(Role::Authority, Alice, &mut ferdie)
    .await;

    let mut alice_nonce = alice.account_nonce();
    let best_hash = alice.client.as_ref().info().best_hash;
    let best_number = alice.client.as_ref().info().best_number;
    let dummy_runtime_call = pallet_balances::Call::transfer_allow_death {
        dest: Charlie.to_account_id(),
        value: 1,
    };

    let transfer_to_charlie =
        alice.construct_extrinsic_with_tip(alice_nonce, 0, dummy_runtime_call.clone());
    alice_nonce += 1;

    let transfer_to_charlie_2 =
        alice.construct_extrinsic_with_tip(alice_nonce, 0, dummy_runtime_call.clone());

    let sample_valid_bundle_extrinsics = vec![
        OpaqueExtrinsic::from_bytes(&transfer_to_charlie.encode()).unwrap(),
        OpaqueExtrinsic::from_bytes(&transfer_to_charlie_2.encode()).unwrap(),
    ];

    {
        let runtime_instance = alice.client.runtime_api();
        assert_eq!(
            runtime_instance
                .check_extrinsics_and_do_pre_dispatch(
                    best_hash,
                    sample_valid_bundle_extrinsics,
                    best_number,
                    best_hash,
                )
                .unwrap(),
            Ok(())
        );
    }

    let transfer_to_charlie_3_duplicate_nonce_extrinsic =
        alice.construct_extrinsic_with_tip(alice_nonce, 0, dummy_runtime_call.clone());

    // Here tx index: 2 is invalid
    let sample_invalid_bundle_with_same_nonce_extrinsic = vec![
        OpaqueExtrinsic::from_bytes(&transfer_to_charlie.encode()).unwrap(),
        OpaqueExtrinsic::from_bytes(&transfer_to_charlie_2.encode()).unwrap(),
        OpaqueExtrinsic::from_bytes(&transfer_to_charlie_3_duplicate_nonce_extrinsic.encode())
            .unwrap(),
    ];

    {
        let runtime_instance = alice.client.runtime_api();
        assert_eq!(
            runtime_instance
                .check_extrinsics_and_do_pre_dispatch(
                    best_hash,
                    sample_invalid_bundle_with_same_nonce_extrinsic,
                    best_number,
                    best_hash,
                )
                .unwrap(),
            Err(CheckExtrinsicsValidityError {
                extrinsic_index: 2,
                transaction_validity_error: TransactionValidityError::Invalid(
                    InvalidTransaction::Stale
                ),
            })
        );
    }

    // Resetting the nonce
    alice_nonce = alice.account_nonce();

    // transfer most of the alice's balance
    let alice_balance = alice.free_balance(Alice.to_account_id());

    let transfer_to_charlie_with_big_tip_1 = alice.construct_extrinsic_with_tip(
        alice_nonce,
        (alice_balance / 3) * 2,
        dummy_runtime_call.clone(),
    );

    alice_nonce += 1;

    let transfer_to_charlie_with_big_tip_2 = alice.construct_extrinsic_with_tip(
        alice_nonce,
        (alice_balance / 3) * 2,
        dummy_runtime_call.clone(),
    );

    // In single tx context both are valid (both tx are identical except nonce), so no need to test both of them independently.
    // only testing one should suffice
    assert_eq!(
        alice
            .client
            .runtime_api()
            .check_extrinsics_and_do_pre_dispatch(
                best_hash,
                vec![
                    OpaqueExtrinsic::from_bytes(&transfer_to_charlie_with_big_tip_1.encode())
                        .unwrap()
                ],
                best_number,
                best_hash,
            )
            .unwrap(),
        Ok(())
    );

    // Tx index: 1 is invalid since low balance to pay for the tip
    let transfer_to_charlie_with_big_tip = vec![
        transfer_to_charlie_with_big_tip_1.clone().into(),
        transfer_to_charlie_with_big_tip_2.clone().into(),
    ];

    {
        let runtime_instance = alice.client.runtime_api();
        assert_eq!(
            runtime_instance
                .check_extrinsics_and_do_pre_dispatch(
                    best_hash,
                    transfer_to_charlie_with_big_tip,
                    best_number,
                    best_hash,
                )
                .unwrap(),
            Err(CheckExtrinsicsValidityError {
                extrinsic_index: 1,
                transaction_validity_error: TransactionValidityError::Invalid(
                    InvalidTransaction::Payment
                ),
            })
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_evm_domain_block_fee() {
    let directory = TempDir::new().expect("Must be able to create temporary directory");

    let mut builder = sc_cli::LoggerBuilder::new("");
    builder.with_colors(false);
    let _ = builder.init();

    let tokio_handle = tokio::runtime::Handle::current();

    // Start Ferdie
    let mut ferdie = MockConsensusNode::run(
        tokio_handle.clone(),
        Ferdie,
        BasePath::new(directory.path().join("ferdie")),
    );

    // Run Alice (a evm domain authority node)
    let mut alice = domain_test_service::DomainNodeBuilder::new(
        tokio_handle.clone(),
        BasePath::new(directory.path().join("alice")),
    )
    .build_evm_node(Role::Authority, Alice, &mut ferdie)
    .await;

    produce_blocks!(ferdie, alice, 3).await.unwrap();

    // Create more accounts and fund these accounts
    let tx_to_create = 3;
    let account_infos = (0..tx_to_create)
        .map(|i| address_build(i as u128 + 1))
        .collect::<Vec<AccountInfo>>();
    let alice_balance = alice.free_balance(Alice.to_account_id());
    for (i, account_info) in account_infos.iter().enumerate() {
        let alice_balance_transfer_extrinsic = alice.construct_extrinsic(
            alice.account_nonce() + i as u32,
            pallet_balances::Call::transfer_allow_death {
                dest: account_info.address.0.into(),
                value: alice_balance / (tx_to_create as u128 + 2),
            },
        );
        alice
            .send_extrinsic(alice_balance_transfer_extrinsic)
            .await
            .expect("Failed to send extrinsic");
    }
    produce_blocks!(ferdie, alice, 3).await.unwrap();

    let pre_free_balance: Balance = account_infos
        .iter()
        .map(|acc| alice.free_balance(acc.address.0.into()))
        .sum();
    let gas_price = alice
        .client
        .runtime_api()
        .gas_price(alice.client.info().best_hash)
        .unwrap();

    // Construct and send evm transaction
    #[allow(clippy::type_complexity)]
    let tx_generators: Vec<
        Box<dyn Fn(AccountInfo, U256, ethereum::TransactionAction, Vec<u8>, U256) -> Transaction>,
    > = vec![
        Box::new(generate_eip2930_tx::<TestRuntime>),
        Box::new(generate_eip1559_tx::<TestRuntime>),
        Box::new(generate_legacy_tx::<TestRuntime>),
    ];
    for (i, (acc, tx_generator)) in account_infos
        .iter()
        .zip(tx_generators.into_iter())
        .enumerate()
    {
        let tx = generate_evm_domain_extrinsic(tx_generator(
            acc.clone(),
            U256::zero(),
            ethereum::TransactionAction::Create,
            vec![(i + 1) as u8; 100],
            gas_price,
        ));
        alice
            .send_extrinsic(tx)
            .await
            .expect("Failed to send extrinsic");
    }

    // Produce a bundle that contains the just sent extrinsic
    let (slot, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(bundle.extrinsics.len(), 3);
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();
    let consensus_block_hash = ferdie.client.info().best_hash;

    // Produce one more bundle, this bundle should contains the ER of the previous bundle
    let (_, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let receipt = bundle.into_receipt();
    assert_eq!(receipt.consensus_block_hash, consensus_block_hash);

    // All the transaction fee is collected as operator reward
    let domain_block_fees = alice
        .client
        .runtime_api()
        .block_fees(receipt.domain_block_hash)
        .unwrap();
    let free_balance_changes = pre_free_balance
        - account_infos
            .iter()
            .map(|acc| alice.free_balance(acc.address.0.into()))
            .sum::<Balance>();

    assert_eq!(
        free_balance_changes,
        domain_block_fees.domain_execution_fee + domain_block_fees.consensus_storage_fee
    );
    assert!(!domain_block_fees.consensus_storage_fee.is_zero());
    assert_eq!(domain_block_fees, receipt.block_fees);
}

// #[tokio::test(flavor = "multi_thread")]
// #[ignore]
// async fn execution_proof_creation_and_verification_should_work() {
//     let directory = TempDir::new().expect("Must be able to create temporary directory");
//
//     let mut builder = sc_cli::LoggerBuilder::new("");
//     builder.with_colors(false);
//     let _ = builder.init();
//
//     let tokio_handle = tokio::runtime::Handle::current();
//
//     // Start Ferdie
//     let mut ferdie = MockConsensusNode::run(
//         tokio_handle.clone(),
//         Ferdie,
//         BasePath::new(directory.path().join("ferdie")),
//     );
//
//     // Run Alice (a evm domain authority node)
//     let mut alice = domain_test_service::DomainNodeBuilder::new(
//         tokio_handle.clone(),
//
//         BasePath::new(directory.path().join("alice")),
//     )
//     .build_evm_node(Role::Authority, Alice, &mut ferdie)
//     .await;
//
//     // Run Bob (a evm domain full node)
//     let bob = domain_test_service::DomainNodeBuilder::new(
//         tokio_handle,
//         Bob,
//         BasePath::new(directory.path().join("bob")),
//     )
//     .build_evm_node(Role::Full, Alice, &mut ferdie)
//     .await;
//
//     // Bob is able to sync blocks.
//     produce_blocks!(ferdie, alice, 1, bob).await.unwrap();
//
//     let alice_nonce = alice.account_nonce();
//     let transfer_to_charlie = alice.construct_extrinsic(
//         alice_nonce,
//         pallet_balances::Call::transfer_allow_death {
//             dest: Charlie.to_account_id(),
//             value: 8,
//         },
//     );
//     let transfer_to_dave = alice.construct_extrinsic(
//         alice_nonce + 1,
//         pallet_balances::Call::transfer_allow_death {
//             dest: Dave.to_account_id(),
//             value: 8,
//         },
//     );
//     let transfer_to_charlie_again = alice.construct_extrinsic(
//         alice_nonce + 2,
//         pallet_balances::Call::transfer_allow_death {
//             dest: Charlie.to_account_id(),
//             value: 88,
//         },
//     );
//
//     let test_txs = vec![
//         transfer_to_charlie.clone(),
//         transfer_to_dave.clone(),
//         transfer_to_charlie_again.clone(),
//     ];
//
//     for tx in test_txs.iter() {
//         alice
//             .send_extrinsic(tx.clone())
//             .await
//             .expect("Failed to send extrinsic");
//     }
//
//     // Produce a domain bundle to include the above test txs and wait for `alice`
//     // to apply these txs
//     let (slot, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
//     assert!(bundle.is_some());
//     produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
//         .await
//         .unwrap();
//
//     let best_hash = alice.client.info().best_hash;
//     let header = alice.client.header(best_hash).unwrap().unwrap();
//     let parent_header = alice.client.header(*header.parent_hash()).unwrap().unwrap();
//
//     let create_block_builder = || {
//         let primary_hash = ferdie.client.hash(*header.number()).unwrap().unwrap();
//         let digest = Digest {
//             logs: vec![DigestItem::consensus_block_info((
//                 *header.number(),
//                 primary_hash,
//             ))],
//         };
//         BlockBuilder::new(
//             &*alice.client,
//             parent_header.hash(),
//             *parent_header.number(),
//             RecordProof::No,
//             digest,
//             &*alice.backend,
//             test_txs.clone().into_iter().map(Into::into).collect(),
//             Default::default(),
//         )
//         .unwrap()
//     };
//
//     let intermediate_roots = alice
//         .client
//         .runtime_api()
//         .intermediate_roots(best_hash)
//         .expect("Get intermediate roots");
//
//     if intermediate_roots.len() != test_txs.len() + 1 {
//         panic!(
//             "🐛 ERROR: runtime API `intermediate_roots()` obviously returned a wrong result, intermediate_roots: {:?}",
//             intermediate_roots.into_iter().map(Hash::from).collect::<Vec<_>>(),
//         );
//     }
//
//     let primary_hash = ferdie.client.hash(*header.number()).unwrap().unwrap();
//     let new_header = Header::new(
//         *header.number(),
//         Default::default(),
//         Default::default(),
//         parent_header.hash(),
//         Digest {
//             logs: vec![DigestItem::consensus_block_info((
//                 *header.number(),
//                 primary_hash,
//             ))],
//         },
//     );
//     let execution_phase = ExecutionPhase::InitializeBlock;
//     let initialize_block_call_data = new_header.encode();
//
//     let prover = ExecutionProver::new(alice.backend.clone(), alice.code_executor.clone());
//
//     // Test `initialize_block`.
//     let storage_proof = prover
//         .prove_execution::<sp_trie::PrefixedMemoryDB<BlakeTwo256>>(
//             parent_header.hash(),
//             &execution_phase,
//             &initialize_block_call_data,
//             None,
//         )
//         .expect("Create `initialize_block` proof");
//
//     // Test `initialize_block` verification.
//     let execution_result = prover
//         .check_execution_proof(
//             parent_header.hash(),
//             &execution_phase,
//             &initialize_block_call_data,
//             *parent_header.state_root(),
//             storage_proof.clone(),
//         )
//         .expect("Check `initialize_block` proof");
//     let post_execution_root = execution_phase
//         .decode_execution_result::<Header>(execution_result)
//         .unwrap();
//     assert_eq!(post_execution_root, intermediate_roots[0].into());
//
//     let invalid_transaction_proof_verifier = InvalidTransactionProofVerifier::new(
//         ferdie.client.clone(),
//         Arc::new(ferdie.executor.clone()),
//         TestVerifierClient::new(ferdie.client.clone(), alice.client.clone()),
//     );
//
//     let proof_verifier =
//         ProofVerifier::<Block, _, _>::new(Arc::new(invalid_transaction_proof_verifier));
//
//     let invalid_state_transition_proof = InvalidStateTransitionProof {
//         domain_id: TEST_DOMAIN_ID,
//         bad_receipt_hash: Hash::random(),
//         proof: storage_proof,
//         execution_phase,
//     };
//     let fraud_proof = FraudProof::InvalidStateTransition(invalid_state_transition_proof);
//     assert!(proof_verifier.verify(&fraud_proof).is_ok());
//
//     // Test extrinsic execution.
//     let encoded_test_txs: Vec<_> = test_txs.iter().map(Encode::encode).collect();
//     for (target_extrinsic_index, encoded_tx) in encoded_test_txs.clone().into_iter().enumerate() {
//         let storage_changes = create_block_builder()
//             .prepare_storage_changes_before(target_extrinsic_index)
//             .unwrap_or_else(|_| {
//                 panic!("Get StorageChanges before extrinsic #{target_extrinsic_index}")
//             });
//
//         let delta = storage_changes.transaction;
//         let post_delta_root = storage_changes.transaction_storage_root;
//
//         let execution_phase = {
//             let proof_of_inclusion = StorageProofProvider::<
//                 LayoutV1<BlakeTwo256>,
//             >::generate_enumerated_proof_of_inclusion(
//                 encoded_test_txs.as_slice(), target_extrinsic_index as u32
//             ).unwrap();
//             ExecutionPhase::ApplyExtrinsic {
//                 proof_of_inclusion,
//                 mismatch_index: target_extrinsic_index as u32,
//                 extrinsic: encoded_tx.clone(),
//             }
//         };
//
//         let storage_proof = prover
//             .prove_execution(
//                 parent_header.hash(),
//                 &execution_phase,
//                 &encoded_tx,
//                 Some((delta, post_delta_root)),
//             )
//             .expect("Create extrinsic execution proof");
//
//         let target_trace_root: Hash = intermediate_roots[target_extrinsic_index].into();
//         assert_eq!(target_trace_root, post_delta_root);
//
//         // Test `apply_extrinsic` verification.
//         let execution_result = prover
//             .check_execution_proof(
//                 parent_header.hash(),
//                 &execution_phase,
//                 &encoded_tx,
//                 post_delta_root,
//                 storage_proof.clone(),
//             )
//             .expect("Check extrinsic execution proof");
//         let post_execution_root = execution_phase
//             .decode_execution_result::<Header>(execution_result)
//             .unwrap();
//         assert_eq!(
//             post_execution_root,
//             intermediate_roots[target_extrinsic_index + 1].into()
//         );
//
//         let invalid_state_transition_proof = InvalidStateTransitionProof {
//             domain_id: TEST_DOMAIN_ID,
//             bad_receipt_hash: Hash::random(),
//             proof: storage_proof,
//             execution_phase,
//         };
//         let fraud_proof = FraudProof::InvalidStateTransition(invalid_state_transition_proof);
//         assert!(proof_verifier.verify(&fraud_proof).is_ok());
//     }
//
//     // Test `finalize_block`
//     let storage_changes = create_block_builder()
//         .prepare_storage_changes_before_finalize_block()
//         .expect("Get StorageChanges before `finalize_block`");
//
//     let delta = storage_changes.transaction;
//     let post_delta_root = storage_changes.transaction_storage_root;
//
//     assert_eq!(post_delta_root, intermediate_roots.last().unwrap().into());
//
//     let execution_phase = ExecutionPhase::FinalizeBlock;
//     let finalize_block_call_data = Vec::new();
//
//     let storage_proof = prover
//         .prove_execution(
//             parent_header.hash(),
//             &execution_phase,
//             &finalize_block_call_data,
//             Some((delta, post_delta_root)),
//         )
//         .expect("Create `finalize_block` proof");
//
//     // Test `finalize_block` verification.
//     let execution_result = prover
//         .check_execution_proof(
//             parent_header.hash(),
//             &execution_phase,
//             &finalize_block_call_data,
//             post_delta_root,
//             storage_proof.clone(),
//         )
//         .expect("Check `finalize_block` proof");
//     let post_execution_root = execution_phase
//         .decode_execution_result::<Header>(execution_result)
//         .unwrap();
//     assert_eq!(post_execution_root, *header.state_root());
//
//     let invalid_state_transition_proof = InvalidStateTransitionProof {
//         domain_id: TEST_DOMAIN_ID,
//         bad_receipt_hash: Hash::random(),
//         proof: storage_proof,
//         execution_phase,
//     };
//     let fraud_proof = FraudProof::InvalidStateTransition(invalid_state_transition_proof);
//     assert!(proof_verifier.verify(&fraud_proof).is_ok());
// }

// #[tokio::test(flavor = "multi_thread")]
// #[ignore]
// async fn invalid_execution_proof_should_not_work() {
//     let directory = TempDir::new().expect("Must be able to create temporary directory");
//
//     let mut builder = sc_cli::LoggerBuilder::new("");
//     builder.with_colors(false);
//     let _ = builder.init();
//
//     let tokio_handle = tokio::runtime::Handle::current();
//
//     // Start Ferdie
//     let mut ferdie = MockConsensusNode::run(
//         tokio_handle.clone(),
//         Ferdie,
//         BasePath::new(directory.path().join("ferdie")),
//     );
//
//     // Run Alice (a evm domain authority node)
//     let mut alice = domain_test_service::DomainNodeBuilder::new(
//         tokio_handle.clone(),
//
//         BasePath::new(directory.path().join("alice")),
//     )
//     .build_evm_node(Role::Authority, Alice, &mut ferdie)
//     .await;
//
//     // Run Bob (a evm domain full node)
//     let bob = domain_test_service::DomainNodeBuilder::new(
//         tokio_handle,
//         Bob,
//         BasePath::new(directory.path().join("bob")),
//     )
//     .build_evm_node(Role::Full, Alice, &mut ferdie)
//     .await;
//
//     // Bob is able to sync blocks.
//     produce_blocks!(ferdie, alice, 1, bob).await.unwrap();
//
//     let alice_nonce = alice.account_nonce();
//     let transfer_to_charlie = alice.construct_extrinsic(
//         alice_nonce,
//         pallet_balances::Call::transfer_allow_death {
//             dest: Charlie.to_account_id(),
//             value: 8,
//         },
//     );
//     let transfer_to_charlie_again = alice.construct_extrinsic(
//         alice_nonce + 1,
//         pallet_balances::Call::transfer_allow_death {
//             dest: Charlie.to_account_id(),
//             value: 8,
//         },
//     );
//
//     let test_txs = vec![
//         transfer_to_charlie.clone(),
//         transfer_to_charlie_again.clone(),
//     ];
//     let encoded_test_txs: Vec<_> = test_txs.iter().map(Encode::encode).collect();
//
//     for tx in test_txs.iter() {
//         alice
//             .send_extrinsic(tx.clone())
//             .await
//             .expect("Failed to send extrinsic");
//     }
//
//     // Produce a domain bundle to include the above test tx
//     let (slot, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
//     assert!(bundle.is_some());
//
//     // Wait for `alice` to apply these txs
//     produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
//         .await
//         .unwrap();
//
//     let best_hash = alice.client.info().best_hash;
//     let header = alice.client.header(best_hash).unwrap().unwrap();
//     let parent_header = alice.client.header(*header.parent_hash()).unwrap().unwrap();
//
//     let create_block_builder = || {
//         BlockBuilder::new(
//             &*alice.client,
//             parent_header.hash(),
//             *parent_header.number(),
//             RecordProof::No,
//             Digest {
//                 logs: vec![DigestItem::consensus_block_info((
//                     *header.number(),
//                     header.hash(),
//                 ))],
//             },
//             &*alice.backend,
//             test_txs.clone().into_iter().map(Into::into).collect(),
//             Default::default(),
//         )
//         .unwrap()
//     };
//
//     let prover = ExecutionProver::new(alice.backend.clone(), alice.code_executor.clone());
//
//     let create_extrinsic_proof = |extrinsic_index: usize| {
//         let storage_changes = create_block_builder()
//             .prepare_storage_changes_before(extrinsic_index)
//             .unwrap_or_else(|_| panic!("Get StorageChanges before extrinsic #{extrinsic_index}"));
//
//         let delta = storage_changes.transaction;
//         let post_delta_root = storage_changes.transaction_storage_root;
//
//         let execution_phase = {
//             let proof_of_inclusion = StorageProofProvider::<
//                 LayoutV1<BlakeTwo256>,
//             >::generate_enumerated_proof_of_inclusion(
//                 encoded_test_txs.as_slice(), extrinsic_index as u32
//             ).unwrap();
//             ExecutionPhase::ApplyExtrinsic {
//                 proof_of_inclusion,
//                 mismatch_index: extrinsic_index as u32,
//                 extrinsic: encoded_test_txs[extrinsic_index].clone(),
//             }
//         };
//         let apply_extrinsic_call_data = encoded_test_txs[extrinsic_index].clone();
//
//         let proof = prover
//             .prove_execution(
//                 parent_header.hash(),
//                 &execution_phase,
//                 &apply_extrinsic_call_data,
//                 Some((delta, post_delta_root)),
//             )
//             .expect("Create extrinsic execution proof");
//
//         (proof, post_delta_root, execution_phase)
//     };
//
//     let (proof0, post_delta_root0, execution_phase0) = create_extrinsic_proof(0);
//     let (proof1, post_delta_root1, execution_phase1) = create_extrinsic_proof(1);
//
//     let check_proof_executor = |post_delta_root: Hash, proof: StorageProof| {
//         let execution_phase = create_extrinsic_proof(1).2;
//         let apply_extrinsic_call_data = transfer_to_charlie_again.encode();
//         prover.check_execution_proof(
//             parent_header.hash(),
//             &execution_phase,
//             &apply_extrinsic_call_data,
//             post_delta_root,
//             proof,
//         )
//     };
//
//     assert!(check_proof_executor(post_delta_root1, proof0.clone()).is_err());
//     assert!(check_proof_executor(post_delta_root0, proof1.clone()).is_err());
//     assert!(check_proof_executor(post_delta_root0, proof0.clone()).is_ok());
//     assert!(check_proof_executor(post_delta_root1, proof1.clone()).is_ok());
//
//     let invalid_transaction_proof_verifier = InvalidTransactionProofVerifier::new(
//         ferdie.client.clone(),
//         Arc::new(ferdie.executor.clone()),
//         TestVerifierClient::new(ferdie.client.clone(), alice.client.clone()),
//     );
//
//     let proof_verifier =
//         ProofVerifier::<Block, _, _>::new(Arc::new(invalid_transaction_proof_verifier));
//
//     let invalid_state_transition_proof = InvalidStateTransitionProof {
//         domain_id: TEST_DOMAIN_ID,
//         bad_receipt_hash: Hash::random(),
//         proof: proof1,
//         execution_phase: execution_phase0.clone(),
//     };
//     let fraud_proof = FraudProof::InvalidStateTransition(invalid_state_transition_proof);
//     assert!(proof_verifier.verify(&fraud_proof).is_err());
//
//     let invalid_state_transition_proof = InvalidStateTransitionProof {
//         domain_id: TEST_DOMAIN_ID,
//         bad_receipt_hash: Hash::random(),
//         proof: proof0.clone(),
//         execution_phase: execution_phase1,
//     };
//     let fraud_proof = FraudProof::InvalidStateTransition(invalid_state_transition_proof);
//     assert!(proof_verifier.verify(&fraud_proof).is_err());
//
//     let invalid_state_transition_proof = InvalidStateTransitionProof {
//         domain_id: TEST_DOMAIN_ID,
//         bad_receipt_hash: Hash::random(),
//         proof: proof0,
//         execution_phase: execution_phase0,
//     };
//     let fraud_proof = FraudProof::InvalidStateTransition(invalid_state_transition_proof);
//     assert!(proof_verifier.verify(&fraud_proof).is_ok());
// }

// TODO: Unlock test when gossip message validator are supported in DecEx v2.
// #[tokio::test(flavor = "multi_thread")]
// async fn test_invalid_transaction_proof_creation_and_verification() {
//     let directory = TempDir::new().expect("Must be able to create temporary directory");

//     let mut builder = sc_cli::LoggerBuilder::new("runtime=debug");
//     builder.with_colors(false);
//     let _ = builder.init();

//     let tokio_handle = tokio::runtime::Handle::current();

//     // Start Ferdie
//     let mut ferdie = MockConsensusNode::run(
//         tokio_handle.clone(),
//         Ferdie,
//         BasePath::new(directory.path().join("ferdie")),
//     );

//     // Run Alice (a system domain authority node)
//     let mut alice = domain_test_service::DomainNodeBuilder::new(
//         tokio_handle.clone(),
//
//         BasePath::new(directory.path().join("alice")),
//     )
//     .build_evm_node(Role::Authority, &mut ferdie)
//     .await;

//     produce_blocks!(ferdie, alice, 3).await.unwrap();

//     alice
//         .construct_and_send_extrinsic(pallet_balances::Call::transfer_allow_death {
//             dest: domain_test_service::evm_domain_test_runtime::Address::Id(One.public().into()),
//             value: 500 + 1,
//         })
//         .await
//         .expect("Send an extrinsic to transfer some balance from Alice to One");

//     ferdie.produce_slot_and_wait_for_bundle_submission().await;

//     produce_blocks!(ferdie, alice, 1).await.unwrap();

//     let (_slot, maybe_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;

//     produce_blocks!(ferdie, alice, 3).await.unwrap();

//     // This is an invalid transaction.
//     let transfer_from_one_to_bob = alice.construct_extrinsic_with_caller(
//         One,
//         pallet_balances::Call::transfer_allow_death {
//             dest: domain_test_service::evm_domain_test_runtime::Address::Id(Bob.public().into()),
//             value: 1000,
//         },
//     );

//     let mut bundle_with_bad_extrinsics = maybe_bundle.unwrap();
//     bundle_with_bad_extrinsics.extrinsics =
//         vec![OpaqueExtrinsic::from_bytes(&transfer_from_one_to_bob.encode()).unwrap()];
//     bundle_with_bad_extrinsics.sealed_header.signature = alice
//         .key
//         .pair()
//         .sign(bundle_with_bad_extrinsics.sealed_header.pre_hash().as_ref())
//         .into();

//     alice
//         .gossip_message_validator
//         .validate_gossiped_bundle(&bundle_with_bad_extrinsics)
//         .expect("Create an invalid transaction proof and submit to tx pool");

//     let extract_fraud_proof_from_tx_pool = || {
//         let ready_txs = ferdie
//             .transaction_pool
//             .pool()
//             .validated_pool()
//             .ready()
//             .collect::<Vec<_>>();

//         ready_txs
//             .into_iter()
//             .find_map(|ready_tx| {
//                 let uxt = subspace_test_runtime::UncheckedExtrinsic::decode(
//                     &mut ready_tx.data.encode().as_slice(),
//                 )
//                 .unwrap();
//                 match uxt.function {
//                     subspace_test_runtime::RuntimeCall::Domains(
//                         pallet_domains::Call::submit_fraud_proof { fraud_proof },
//                     ) => Some(fraud_proof),
//                     _ => None,
//                 }
//             })
//             .expect("Can not find submit_fraud_proof extrinsic")
//     };

//     let good_invalid_transaction_proof = extract_fraud_proof_from_tx_pool();

//     let invalid_state_transition_proof_verifier = InvalidStateTransitionProofVerifier::new(
//         ferdie.client.clone(),
//         ferdie.executor.clone(),
//         TestVerifierClient::new(ferdie.client.clone(), alice.client.clone()),
//     );

//     let invalid_transaction_proof_verifier = InvalidTransactionProofVerifier::new(
//         ferdie.client.clone(),
//         Arc::new(ferdie.executor.clone()),
//         TestVerifierClient::new(ferdie.client.clone(), alice.client.clone()),
//     );

//     let proof_verifier = ProofVerifier::<Block, _, _>::new(
//         Arc::new(invalid_transaction_proof_verifier),
//         Arc::new(invalid_state_transition_proof_verifier),
//     );

//     assert!(
//         proof_verifier
//             .verify(&good_invalid_transaction_proof)
//             .is_ok(),
//         "Valid proof must be accepeted"
//     );

//     ferdie
//         .produce_blocks(1)
//         .await
//         .expect("FraudProof verification in the block import pipeline is fine too");
// }
