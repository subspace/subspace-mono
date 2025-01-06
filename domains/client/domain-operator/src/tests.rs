use crate::domain_block_processor::{DomainBlockProcessor, PendingConsensusBlocks};
use crate::domain_bundle_producer::{BundleProducer, TestBundleProducer};
use crate::domain_bundle_proposer::DomainBundleProposer;
use crate::fraud_proof::{FraudProofGenerator, TraceDiffType};
use crate::tests::TxPoolError::InvalidTransaction as TxPoolInvalidTransaction;
use crate::OperatorSlotInfo;
use codec::{Decode, Encode};
use cross_domain_message_gossip::get_channel_state;
use domain_runtime_primitives::{AccountId20Converter, AccountIdConverter, Hash};
use domain_test_primitives::{OnchainStateApi, TimestampApi};
use domain_test_service::evm_domain_test_runtime::{Header, UncheckedExtrinsic};
use domain_test_service::EcdsaKeyring::{Alice, Bob, Charlie, Eve};
use domain_test_service::Sr25519Keyring::{self, Alice as Sr25519Alice, Ferdie};
use domain_test_service::{construct_extrinsic_generic, AUTO_ID_DOMAIN_ID, EVM_DOMAIN_ID};
use futures::StreamExt;
use pallet_domains::OperatorConfig;
use pallet_messenger::ChainAllowlistUpdate;
use sc_client_api::{Backend, BlockBackend, BlockchainEvents, HeaderBackend};
use sc_domains::generate_mmr_proof;
use sc_service::{BasePath, Role};
use sc_transaction_pool::error::Error as PoolError;
use sc_transaction_pool_api::error::Error as TxPoolError;
use sc_transaction_pool_api::TransactionPool;
use sc_utils::mpsc::tracing_unbounded;
use sp_api::{ProvideRuntimeApi, StorageProof};
use sp_consensus::SyncOracle;
use sp_core::storage::StateVersion;
use sp_core::traits::{FetchRuntimeCode, SpawnEssentialNamed};
use sp_core::{Pair, H256};
use sp_domain_digests::AsPredigest;
use sp_domains::core_api::DomainCoreApi;
use sp_domains::merkle_tree::MerkleTree;
use sp_domains::{
    Bundle, BundleValidity, ChainId, ChannelId, DomainsApi, HeaderHashingFor, InboxedBundle,
    InvalidBundleType, Transfers,
};
use sp_domains_fraud_proof::fraud_proof::{
    ApplyExtrinsicMismatch, ExecutionPhase, FinalizeBlockMismatch, FraudProofVariant,
    InvalidBlockFeesProof, InvalidBundlesProofData, InvalidDomainBlockHashProof,
    InvalidExtrinsicsRootProof, InvalidTransfersProof,
};
use sp_domains_fraud_proof::InvalidTransactionCode;
use sp_messenger::messages::{CrossDomainMessage, Proof};
use sp_messenger::MessengerApi;
use sp_mmr_primitives::{EncodableOpaqueLeaf, LeafProof as MmrProof};
use sp_runtime::generic::{BlockId, DigestItem};
use sp_runtime::traits::{
    BlakeTwo256, Block as BlockT, Convert, Hash as HashT, Header as HeaderT, Zero,
};
use sp_runtime::transaction_validity::{
    InvalidTransaction, TransactionSource, TransactionValidityError,
};
use sp_runtime::OpaqueExtrinsic;
use sp_state_machine::backend::AsTrieBackend;
use sp_subspace_mmr::ConsensusChainMmrLeafProof;
use sp_transaction_pool::runtime_api::TaggedTransactionQueue;
use sp_weights::Weight;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use subspace_core_primitives::pot::PotOutput;
use subspace_runtime_primitives::opaque::Block as CBlock;
use subspace_runtime_primitives::{Balance, SSC};
use subspace_test_service::{
    produce_block_with, produce_blocks, produce_blocks_until, MockConsensusNode,
};
use tempfile::TempDir;
use tracing::error;

const TIMEOUT: Duration = Duration::from_mins(2);

fn number_of(consensus_node: &MockConsensusNode, block_hash: Hash) -> u32 {
    consensus_node
        .client
        .number(block_hash)
        .unwrap_or_else(|err| panic!("Failed to fetch number for {block_hash}: {err}"))
        .unwrap_or_else(|| panic!("header {block_hash} not in the chain"))
}

#[tokio::test(flavor = "multi_thread")]
async fn test_domain_instance_bootstrapper() {
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

    let expected_genesis_state_root = ferdie
        .client
        .runtime_api()
        .genesis_state_root(ferdie.client.info().best_hash, EVM_DOMAIN_ID)
        .unwrap()
        .unwrap();

    // Run Alice (a evm domain authority node)
    let alice = domain_test_service::DomainNodeBuilder::new(
        tokio_handle.clone(),
        BasePath::new(directory.path().join("alice")),
    )
    .build_evm_node(Role::Authority, Alice, &mut ferdie)
    .await;

    let genesis_state_root = *alice
        .client
        .header(alice.client.info().genesis_hash)
        .unwrap()
        .unwrap()
        .state_root();

    assert_eq!(expected_genesis_state_root, genesis_state_root);

    produce_blocks!(ferdie, alice, 3)
        .await
        .expect("3 consensus blocks produced successfully");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_domain_chain_fork_choice() {
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

    produce_blocks!(ferdie, alice, 3).await.unwrap();
    assert_eq!(alice.client.info().best_number, 3);

    let domain_hash_3 = alice.client.info().best_hash;
    let common_consensus_hash = ferdie.client.info().best_hash;

    // Fork A produce a consenus block that contains no bundle
    let slot = ferdie.produce_slot();
    let fork_a_block_hash = ferdie
        .produce_block_with_slot_at(slot, common_consensus_hash, Some(vec![]))
        .await
        .unwrap();

    let mut alice_import_notification_stream = alice.client.every_import_notification_stream();

    // Fork B produce a consenus block that contains bundles
    let (slot, _) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let fork_b_block_hash = ferdie
        .produce_block_with_slot_at(slot, common_consensus_hash, None)
        .await
        .unwrap();

    // Fork B is still in the non-canonical fork thus the consensus block and bundle
    // won't be processed
    assert!(alice_import_notification_stream.try_recv().is_err());

    // The consensus fork A is still the best fork, because fork A don't derive any new
    // domain block thus the best domain block is still #3
    assert_eq!(ferdie.client.info().best_hash, fork_a_block_hash);
    assert_eq!(alice.client.info().best_number, 3);
    assert_eq!(alice.client.info().best_hash, domain_hash_3);

    // Produce one more consensus block on fork B to make it the best fork
    let (slot, _) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let fork_b_block_hash = ferdie
        .produce_block_with_slot_at(slot, fork_b_block_hash, Some(vec![]))
        .await
        .unwrap();
    assert_eq!(ferdie.client.info().best_hash, fork_b_block_hash);

    // It should also trigger the operator to processe consensus blocks at fork B
    // thus produce more domain blocks
    let domain_block_4 = alice_import_notification_stream.next().await.unwrap();
    assert_eq!(*domain_block_4.header.number(), 4);
    assert_eq!(alice.client.info().best_number, 4);
    assert_eq!(alice.client.info().best_hash, domain_block_4.header.hash());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_domain_block_production() {
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

    for i in 0..50 {
        let (tx, slot) = if i % 2 == 0 {
            // Produce bundle and include it in the primary block hence produce a domain block
            let (slot, _) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
            // `None` means collect tx from the tx pool
            (None, slot)
        } else {
            // No bundle hence not produce domain block
            (Some(vec![]), ferdie.produce_slot())
        };
        ferdie
            .produce_block_with_slot_at(slot, ferdie.client.info().best_hash, tx)
            .await
            .unwrap();
    }
    // Domain block only produced when there is bundle contains in the primary block
    assert_eq!(ferdie.client.info().best_number, 50);
    assert_eq!(alice.client.info().best_number, 25);

    let consensus_block_hash = ferdie.client.info().best_hash;
    let domain_block_number = alice.client.info().best_number;
    let domain_block_hash = alice.client.info().best_hash;

    // Fork A
    // Produce 5 more primary blocks and producing domain block for every primary block
    produce_blocks!(ferdie, alice, 3).await.unwrap();

    // Record the block hashes for later use
    let mut fork_b_parent_hash = ferdie.client.info().best_hash;
    let fork_a_block_hash_3 = alice.client.info().best_hash;

    produce_blocks!(ferdie, alice, 2).await.unwrap();
    assert_eq!(alice.client.info().best_number, domain_block_number + 5);

    // Fork B
    // Produce 6 more primary blocks but only producing 3 domain blocks because there are
    // more primary block on fork B, it will become the best fork of the domain chain
    for _ in 0..3 {
        let slot = ferdie.produce_slot();
        fork_b_parent_hash = ferdie
            .produce_block_with_slot_at(slot, fork_b_parent_hash, Some(vec![]))
            .await
            .unwrap();
    }
    assert_eq!(alice.client.info().best_number, domain_block_number + 3);
    assert_eq!(alice.client.info().best_hash, fork_a_block_hash_3);

    // Fork C
    // Produce 10 more primary blocks and do not produce any domain block but because there are
    // more primary block on fork C, it will become the best fork of the domain chain
    let mut fork_c_parent_hash = consensus_block_hash;
    for _ in 0..10 {
        let slot = ferdie.produce_slot();
        fork_c_parent_hash = ferdie
            .produce_block_with_slot_at(slot, fork_c_parent_hash, Some(vec![]))
            .await
            .unwrap();
    }
    // The best block fall back to an ancestor block
    assert_eq!(alice.client.info().best_number, domain_block_number);
    assert_eq!(alice.client.info().best_hash, domain_block_hash);

    // Simply producing more block on fork C
    for _ in 0..10 {
        let (slot, opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
        let tx = subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into();
        // Produce consensus block that only contains the `submit_bundle` extrinsic instead of
        // any extrinsic in the tx pool, because the background worker `txpool-notifications` may
        // submit extrinsic from the retracted blocks of other fork to the tx pool and these tx
        // is invalid in the fork C
        ferdie
            .produce_block_with_slot_at(slot, ferdie.client.info().best_hash, Some(vec![tx]))
            .await
            .unwrap();
    }
    assert_eq!(alice.client.info().best_number, domain_block_number + 10);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_processing_empty_consensus_block() {
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

    let domain_block_processor = DomainBlockProcessor {
        domain_id: alice.domain_id,
        domain_created_at: 1,
        client: alice.client.clone(),
        consensus_client: ferdie.client.clone(),
        backend: alice.backend.clone(),
        domain_confirmation_depth: 256u32,
        block_import: Arc::new(Box::new(alice.client.clone())),
        import_notification_sinks: Default::default(),
        domain_sync_oracle: ferdie.sync_service.clone(),
    };

    let domain_genesis_hash = alice.client.info().best_hash;
    for _ in 0..10 {
        // Produce consensus block with no tx thus no bundle
        let slot = ferdie.produce_slot();
        ferdie
            .produce_block_with_slot_at(slot, ferdie.client.info().best_hash, Some(vec![]))
            .await
            .unwrap();

        let consensus_best_hash = ferdie.client.info().best_hash;
        let consensus_best_number = ferdie.client.info().best_number;
        let res = domain_block_processor
            .pending_imported_consensus_blocks(consensus_best_hash, consensus_best_number);
        match res {
            // The consensus block is processed in the above `produce_block_with_extrinsics` call,
            // thus calling `pending_imported_consensus_blocks` again will result in an `Ok(None)`
            Ok(None) => {
                // The `latest_consensus_block` that mapped to the genesis domain block must be updated
                let latest_consensus_block: Hash =
                    crate::aux_schema::latest_consensus_block_hash_for(
                        &*alice.backend,
                        &domain_genesis_hash,
                    )
                    .unwrap()
                    .unwrap();
                assert_eq!(latest_consensus_block, consensus_best_hash);
            }
            // For consensus block at `domain_created_at + 1` (namely block #2), `pending_imported_consensus_blocks`
            // will return `PendingConsensusBlocks` directly
            Ok(Some(PendingConsensusBlocks {
                initial_parent,
                consensus_imports,
            })) if consensus_best_number == 2 => {
                assert_eq!(initial_parent.0, domain_genesis_hash);
                assert_eq!(consensus_imports.len(), 1);
                assert_eq!(consensus_imports[0].hash, consensus_best_hash,);
            }
            _ => panic!("unexpected result: {res:?}"),
        }
    }
    // The domain chain is not progressed as all the consensus blocks are empty
    assert_eq!(ferdie.client.info().best_number, 10);
    assert_eq!(alice.client.info().best_number, 0);

    // To process non-empty consensus blocks and produce domain blocks
    produce_blocks!(ferdie, alice, 3).await.unwrap();
    assert_eq!(alice.client.info().best_number, 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_domain_block_deriving_from_multiple_bundles() {
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

    produce_blocks!(ferdie, alice, 3).await.unwrap();

    let pre_eve_free_balance = alice.free_balance(Eve.to_account_id());
    for caller in [Alice, Bob, Charlie] {
        let tx = construct_extrinsic_generic::<evm_domain_test_runtime::Runtime, _>(
            &alice.client,
            pallet_balances::Call::transfer_allow_death {
                dest: Eve.to_account_id(),
                value: 1,
            },
            caller,
            false,
            0,
            0u128,
        );
        alice
            .send_extrinsic(tx)
            .await
            .expect("Failed to send extrinsic");

        // Produce a bundle and submit to the tx pool of the consensus node
        let (_, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
        assert_eq!(bundle.extrinsics.len(), 1);
    }

    let slot = ferdie.produce_slot();
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();
    assert_eq!(
        alice.free_balance(Eve.to_account_id()),
        pre_eve_free_balance + 3
    );
    let domain_block_number = alice.client.info().best_number;

    // Produce one more bundle to submit the receipt of the above 3 bundles
    produce_blocks!(ferdie, alice, 1).await.unwrap();

    // The receipt should be submitted successfully thus head receipt number
    // is updated
    let head_receipt_number = ferdie
        .client
        .runtime_api()
        .head_receipt_number(ferdie.client.info().best_hash, 0u32.into())
        .unwrap();
    assert_eq!(domain_block_number, head_receipt_number);
}

#[tokio::test(flavor = "multi_thread")]
async fn collected_receipts_should_be_on_the_same_branch_with_current_best_block() {
    let directory = TempDir::new().expect("Must be able to create temporary directory");

    let mut builder = sc_cli::LoggerBuilder::new("");
    builder.with_colors(false);
    let _ = builder.init();

    let tokio_handle = tokio::runtime::Handle::current();

    // Start Ferdie
    let mut consensus_node = MockConsensusNode::run(
        tokio_handle.clone(),
        Ferdie,
        BasePath::new(directory.path().join("ferdie")),
    );

    // Run Alice (a evm domain authority node)
    let alice = domain_test_service::DomainNodeBuilder::new(
        tokio_handle.clone(),
        BasePath::new(directory.path().join("alice")),
    )
    .build_evm_node(Role::Authority, Alice, &mut consensus_node)
    .await;

    produce_blocks!(consensus_node, alice, 3)
        .await
        .expect("3 consensus blocks produced successfully");

    let best_consensus_hash = consensus_node.client.info().best_hash;
    let best_consensus_number = consensus_node.client.info().best_number;
    assert_eq!(best_consensus_number, 3);

    assert_eq!(alice.client.info().best_number, 3);
    let domain_block_2_hash = *alice
        .client
        .header(alice.client.info().best_hash)
        .unwrap()
        .unwrap()
        .parent_hash();

    let best_header = consensus_node
        .client
        .header(best_consensus_hash)
        .unwrap()
        .unwrap();

    let parent_hash = *best_header.parent_hash();

    // Consensus chain forks:
    //      3
    //   /
    // 2 -- 3a
    //   \
    //      3b -- 4
    let slot = consensus_node.produce_slot();
    let fork_block_hash_3a = consensus_node
        .produce_block_with_slot_at(slot, parent_hash, Some(vec![]))
        .await
        .expect("Produced first consensus fork block 3a at height #3");
    // A fork block 3a at #3 produced.
    assert_eq!(number_of(&consensus_node, fork_block_hash_3a), 3);
    assert_ne!(fork_block_hash_3a, best_consensus_hash);
    // Best hash unchanged due to the longest chain fork choice.
    assert_eq!(consensus_node.client.info().best_hash, best_consensus_hash);
    // Hash of block number #3 unchanged.
    assert_eq!(
        consensus_node.client.hash(3).unwrap().unwrap(),
        best_consensus_hash
    );

    let consensus_block_info =
        |best_header: Header| -> (u32, Hash) { (*best_header.number(), best_header.hash()) };
    let receipts_consensus_info =
        |bundle: Bundle<OpaqueExtrinsic, u32, sp_core::H256, Header, Balance>| {
            (
                bundle.receipt().consensus_block_number,
                bundle.receipt().consensus_block_hash,
            )
        };

    // Produce a bundle after the fork block #3a has been produced.
    let signed_bundle = consensus_node
        .notify_new_slot_and_wait_for_bundle(slot)
        .await;

    let expected_receipts_consensus_info = consensus_block_info(best_header.clone());

    // TODO: make MaximumReceiptDrift configurable in order to submit all the pending receipts at
    // once, now the max drift is 2, the receipts is limitted to [2, 3]. Once configurable, we
    // can make the assertation straightforward:
    // assert_eq!(receipts_consensus_info, expected_receipts_consensus_info).
    //
    // Receipts are always collected against the current best block.
    assert_eq!(
        receipts_consensus_info(signed_bundle.unwrap()),
        expected_receipts_consensus_info
    );

    let slot = consensus_node.produce_slot();
    let fork_block_hash_3b = consensus_node
        .produce_block_with_slot_at(slot, parent_hash, Some(vec![]))
        .await
        .expect("Produced second consensus fork block 3b at height #3");
    // Another fork block 3b at #3 produced,
    assert_eq!(number_of(&consensus_node, fork_block_hash_3b), 3);
    assert_ne!(fork_block_hash_3b, best_consensus_hash);
    // Best hash unchanged due to the longest chain fork choice.
    assert_eq!(consensus_node.client.info().best_hash, best_consensus_hash);
    // Hash of block number #3 unchanged.
    assert_eq!(
        consensus_node.client.hash(3).unwrap().unwrap(),
        best_consensus_hash
    );

    // Produce a bundle after the fork block #3b has been produced.
    let signed_bundle = consensus_node
        .notify_new_slot_and_wait_for_bundle(slot)
        .await;
    // Receipts are always collected against the current best block.
    assert_eq!(
        receipts_consensus_info(signed_bundle.unwrap()),
        expected_receipts_consensus_info
    );

    // Produce a new tip at #4.
    let slot = consensus_node.produce_slot();
    produce_block_with!(
        consensus_node.produce_block_with_slot_at(slot, fork_block_hash_3b, Some(vec![])),
        alice
    )
    .await
    .expect("Produce a new block on top of the second fork block at height #3");
    let new_best_hash = consensus_node.client.info().best_hash;
    let new_best_number = consensus_node.client.info().best_number;
    assert_eq!(new_best_number, 4);

    // The domain best block should be reverted to #2 because the primary block #3b and #4 do
    // not contains any bundles
    assert_eq!(alice.client.info().best_number, 2);
    assert_eq!(alice.client.info().best_hash, domain_block_2_hash);

    // Hash of block number #3 is updated to the second fork block 3b.
    assert_eq!(
        consensus_node.client.hash(3).unwrap().unwrap(),
        fork_block_hash_3b
    );

    let new_best_header = consensus_node
        .client
        .header(new_best_hash)
        .unwrap()
        .unwrap();

    assert_eq!(*new_best_header.parent_hash(), fork_block_hash_3b);

    // Produce a bundle after the new block #4 has been produced.
    let (_slot, signed_bundle) = consensus_node
        .produce_slot_and_wait_for_bundle_submission()
        .await;

    // In the new best fork, the receipt header number is 1 thus it produce the receipt
    // of next block namely block 2
    let hash_2 = consensus_node.client.hash(2).unwrap().unwrap();
    let header_2 = consensus_node.client.header(hash_2).unwrap().unwrap();
    assert_eq!(
        receipts_consensus_info(signed_bundle),
        consensus_block_info(header_2)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_domain_tx_propagate() {
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

    // Run Bob (a evm domain full node)
    let mut bob = domain_test_service::DomainNodeBuilder::new(
        tokio_handle,
        BasePath::new(directory.path().join("bob")),
    )
    .connect_to_domain_node(alice.addr.clone())
    .build_evm_node(Role::Full, Bob, &mut ferdie)
    .await;

    produce_blocks!(ferdie, alice, 5, bob).await.unwrap();
    while alice.sync_service.is_major_syncing() || bob.sync_service.is_major_syncing() {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    let pre_bob_free_balance = alice.free_balance(bob.key.to_account_id());
    // Construct and send an extrinsic to bob, as bob is not a authority node, the extrinsic has
    // to propagate to alice to get executed
    bob.construct_and_send_extrinsic(pallet_balances::Call::transfer_allow_death {
        dest: Alice.to_account_id(),
        value: 123,
    })
    .await
    .expect("Failed to send extrinsic");

    produce_blocks_until!(
        ferdie,
        alice,
        {
            // ensure bob has reduced balance since alice might submit other transactions which cost
            // and so exact balance check is not feasible
            alice.free_balance(bob.key.to_account_id()) <= pre_bob_free_balance - 123
        },
        bob
    )
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_executor_full_node_catching_up() {
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
    // Produce 5 consensus block to initialize genesis domain
    //
    // This also make the first consensus block being processed by the alice's domain
    // block processor be block #5, to ensure it can correctly handle the consensus
    // block even it is out of order.
    for _ in 0..5 {
        let slot = ferdie.produce_slot();
        ferdie.produce_block_with_slot(slot).await.unwrap();
    }

    // Run Alice (a evm domain authority node)
    let alice = domain_test_service::DomainNodeBuilder::new(
        tokio_handle.clone(),
        BasePath::new(directory.path().join("alice")),
    )
    .build_evm_node(Role::Authority, Alice, &mut ferdie)
    .await;

    // Run Bob (a evm domain full node)
    let bob = domain_test_service::DomainNodeBuilder::new(
        tokio_handle,
        BasePath::new(directory.path().join("bob")),
    )
    .build_evm_node(Role::Full, Bob, &mut ferdie)
    .await;

    // Bob is able to sync blocks.
    produce_blocks!(ferdie, alice, 3, bob).await.unwrap();

    let alice_block_hash = alice
        .client
        .expect_block_hash_from_id(&BlockId::Number(2))
        .unwrap();
    let bob_block_hash = bob
        .client
        .expect_block_hash_from_id(&BlockId::Number(2))
        .unwrap();
    assert_eq!(
        alice_block_hash, bob_block_hash,
        "Domain authority node and full node must have the same state"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_executor_inherent_timestamp_is_set() {
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

    // Run Bob who runs the authority node for core domain
    let bob = domain_test_service::DomainNodeBuilder::new(
        tokio_handle.clone(),
        BasePath::new(directory.path().join("bob")),
    )
    .build_evm_node(Role::Authority, Bob, &mut ferdie)
    .await;

    produce_blocks!(ferdie, alice, 1, bob).await.unwrap();

    let consensus_timestamp = ferdie
        .client
        .runtime_api()
        .timestamp(ferdie.client.info().best_hash)
        .unwrap();

    let domain_timestamp = bob
        .client
        .runtime_api()
        .timestamp(bob.client.info().best_hash)
        .unwrap();

    assert_eq!(
        consensus_timestamp, domain_timestamp,
        "Timestamp should be preset on domain and must match Consensus runtime timestamp"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_bad_invalid_state_transition_proof_is_rejected() {
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

    let fraud_proof_generator = FraudProofGenerator::new(
        alice.client.clone(),
        ferdie.client.clone(),
        alice.backend.clone(),
        alice.code_executor.clone(),
    );

    produce_blocks!(ferdie, alice, 5).await.unwrap();

    alice
        .construct_and_send_extrinsic(pallet_balances::Call::transfer_allow_death {
            dest: Bob.to_account_id(),
            value: 1,
        })
        .await
        .expect("Failed to send extrinsic");

    // Produce a bundle that contains the previously sent extrinsic and record that bundle for later use
    let (slot, target_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(target_bundle.extrinsics.len(), 1);
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();

    // We get the receipt of target bundle
    let (slot, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let valid_receipt = bundle.into_receipt();
    assert_eq!(valid_receipt.execution_trace.len(), 5);
    let valid_receipt_hash = valid_receipt.hash::<BlakeTwo256>();

    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();

    // trace index 0 is the index of the state root after applying DomainCoreApi::initialize_block_with_post_state_root
    // trace index 1 is the index of the state root after applying inherent timestamp extrinsic
    // trace index 2 is the index of the state root after applying above balance transfer extrinsic
    // trace index 3 is the index of the state root after applying BlockBuilder::finalize_block
    for trace_diff_type in [
        TraceDiffType::Mismatch,
        TraceDiffType::Shorter,
        TraceDiffType::Longer,
    ] {
        for mismatch_index in 0..valid_receipt.execution_trace.len() {
            let dummy_execution_trace = match trace_diff_type {
                TraceDiffType::Shorter => valid_receipt
                    .execution_trace
                    .clone()
                    .drain(..)
                    .take(mismatch_index + 1)
                    .collect(),
                TraceDiffType::Longer => {
                    let mut long_trace = valid_receipt.execution_trace.clone();
                    long_trace.push(H256::default());
                    long_trace
                }
                TraceDiffType::Mismatch => {
                    let mut modified_trace = valid_receipt.execution_trace.clone();
                    modified_trace[mismatch_index] = H256::default();
                    modified_trace
                }
            };

            // For some combination of (TraceDiffType, mismatch_trace_index) there is no difference in trace
            // In this case the fraud proof cannot be generated.
            if valid_receipt.execution_trace == dummy_execution_trace {
                continue;
            }

            // Abusing this method to generate every possible variant of ExecutionPhase
            let result_execution_phase = fraud_proof_generator.find_mismatched_execution_phase(
                valid_receipt.domain_block_hash,
                &valid_receipt.execution_trace,
                &dummy_execution_trace,
            );

            assert!(result_execution_phase.is_ok());
            assert!(result_execution_phase
                .as_ref()
                .is_ok_and(|maybe_execution_phase| maybe_execution_phase.is_some()));

            let execution_phase = result_execution_phase
                .expect("already checked for error above; qed")
                .expect("we already checked for  None above; qed");

            let mut fraud_proof = fraud_proof_generator
                .generate_invalid_state_transition_proof(
                    EVM_DOMAIN_ID,
                    execution_phase,
                    &valid_receipt,
                    dummy_execution_trace.len(),
                    valid_receipt_hash,
                )
                .expect(
                    "Fraud proof generation should succeed for every valid execution phase; qed",
                );

            let submit_fraud_proof_extrinsic =
                subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
                    pallet_domains::Call::submit_fraud_proof {
                        fraud_proof: Box::new(fraud_proof.clone()),
                    }
                    .into(),
                )
                .into();

            let response = ferdie
                .submit_transaction(submit_fraud_proof_extrinsic)
                .await;

            assert!(matches!(
                response,
                Err(PoolError::Pool(TxPoolInvalidTransaction(
                    InvalidTransaction::Custom(tx_code)
                ))) if tx_code == InvalidTransactionCode::FraudProof as u8
            ));

            // To prevent receiving TemporarilyBanned error from the pool
            ferdie.clear_tx_pool().await.unwrap();

            // Modify fraud proof's mismatch index to a higher value and try to submit it.
            match &fraud_proof.proof {
                FraudProofVariant::InvalidStateTransition(invalid_state_transition_fraud_proof) => {
                    match &invalid_state_transition_fraud_proof.execution_phase {
                        ExecutionPhase::ApplyExtrinsic {
                            extrinsic_proof,
                            mismatch: ApplyExtrinsicMismatch::StateRoot(_),
                        } => {
                            let mut modified_invalid_state_transition_fraud_proof =
                                invalid_state_transition_fraud_proof.clone();
                            modified_invalid_state_transition_fraud_proof.execution_phase =
                                ExecutionPhase::ApplyExtrinsic {
                                    extrinsic_proof: extrinsic_proof.clone(),
                                    mismatch: ApplyExtrinsicMismatch::StateRoot(u32::MAX),
                                };
                            fraud_proof.proof = FraudProofVariant::InvalidStateTransition(
                                modified_invalid_state_transition_fraud_proof,
                            );
                        }
                        ExecutionPhase::FinalizeBlock {
                            mismatch: FinalizeBlockMismatch::Longer(_),
                        } => {
                            let mut modified_invalid_state_transition_fraud_proof =
                                invalid_state_transition_fraud_proof.clone();
                            modified_invalid_state_transition_fraud_proof.execution_phase =
                                ExecutionPhase::FinalizeBlock {
                                    mismatch: FinalizeBlockMismatch::Longer(u32::MAX),
                                };
                            fraud_proof.proof = FraudProofVariant::InvalidStateTransition(
                                modified_invalid_state_transition_fraud_proof,
                            );
                        }
                        _ => {}
                    }
                }
                _ => unreachable!(),
            }

            let submit_fraud_proof_with_invalid_mismatch_index_extrinsic =
                subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
                    pallet_domains::Call::submit_fraud_proof {
                        fraud_proof: Box::new(fraud_proof),
                    }
                    .into(),
                )
                .into();

            let response = ferdie
                .submit_transaction(submit_fraud_proof_with_invalid_mismatch_index_extrinsic)
                .await;

            assert!(matches!(
                response,
                Err(PoolError::Pool(TxPoolInvalidTransaction(
                    InvalidTransaction::Custom(tx_code)
                ))) if tx_code == InvalidTransactionCode::FraudProof as u8
            ));

            // To prevent receiving TemporarilyBanned error from the pool
            ferdie.clear_tx_pool().await.unwrap();
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_initialize_block_proof_creation_and_verification_should_work() {
    test_invalid_state_transition_proof_creation_and_verification(TraceDiffType::Mismatch, 0).await
}

#[tokio::test(flavor = "multi_thread")]
async fn test_apply_extrinsic_proof_inherent_ext_creation_and_verification_should_work() {
    test_invalid_state_transition_proof_creation_and_verification(TraceDiffType::Mismatch, 1).await
}

#[tokio::test(flavor = "multi_thread")]
async fn test_apply_extrinsic_proof_normal_ext_creation_and_verification_should_work() {
    test_invalid_state_transition_proof_creation_and_verification(TraceDiffType::Mismatch, 3).await
}

#[tokio::test(flavor = "multi_thread")]
async fn test_finalize_block_proof_creation_and_verification_should_work() {
    test_invalid_state_transition_proof_creation_and_verification(TraceDiffType::Mismatch, 4).await
}

#[tokio::test(flavor = "multi_thread")]
async fn test_short_trace_for_inherent_apply_extrinsic_proof_creation_and_verification_should_work()
{
    test_invalid_state_transition_proof_creation_and_verification(TraceDiffType::Shorter, 1).await
}

#[tokio::test(flavor = "multi_thread")]
async fn test_short_trace_for_normal_ext_apply_extrinsic_proof_creation_and_verification_should_work(
) {
    test_invalid_state_transition_proof_creation_and_verification(TraceDiffType::Shorter, 3).await
}

#[tokio::test(flavor = "multi_thread")]
async fn test_long_trace_for_finalize_block_proof_creation_and_verification_should_work() {
    // In case of TraceDiffType::Longer the bad ER always has two more elements. So, no effect of
    // `mismatch_trace_index`
    test_invalid_state_transition_proof_creation_and_verification(TraceDiffType::Longer, 0).await
}

/// This test will create and verify a invalid state transition proof that targets the receipt of
/// a bundle that contains 2 extrinsic (including 1 inherent timestamp extrinsic), thus there are
/// 4 trace roots in total, passing the `mismatch_trace_index` as:
/// - 0 to test the `initialize_block` state transition
/// - 1 to test the `apply_extrinsic` state transition with the inherent timestamp extrinsic
/// - 2 to test the `apply_extrinsic` state transition with the inherent `set_consensus_chain_byte_fee` extrinsic
/// - 3 to test the `apply_extrinsic` state transition with regular domain extrinsic
/// - 4 to test the `finalize_block` state transition
///
/// TraceDiffType can be passed as `TraceDiffType::Shorter`, `TraceDiffType::Longer`
/// and `TraceDiffType::Mismatch`
async fn test_invalid_state_transition_proof_creation_and_verification(
    trace_diff_type: TraceDiffType,
    mismatch_trace_index: u32,
) {
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

    let bundle_to_tx = |opaque_bundle| {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into()
    };

    produce_blocks!(ferdie, alice, 5).await.unwrap();

    alice
        .construct_and_send_extrinsic(pallet_balances::Call::transfer_allow_death {
            dest: Bob.to_account_id(),
            value: 1,
        })
        .await
        .expect("Failed to send extrinsic");

    // Produce a bundle that contains the previously sent extrinsic and record that bundle for later use
    let (slot, target_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(target_bundle.extrinsics.len(), 1);
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();

    // Get a bundle from the txn pool and modify the receipt of the target bundle to an invalid one
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let original_length = opaque_bundle
        .sealed_header
        .header
        .receipt
        .execution_trace
        .len();
    let (bad_receipt_hash, bad_submit_bundle_tx) = {
        let receipt = &mut opaque_bundle.sealed_header.header.receipt;
        assert_eq!(receipt.execution_trace.len(), 5);

        match trace_diff_type {
            TraceDiffType::Longer => {
                receipt.execution_trace.push(Default::default());
                receipt.execution_trace.push(Default::default());
            }
            TraceDiffType::Shorter => {
                receipt.execution_trace = receipt
                    .execution_trace
                    .clone()
                    .drain(..)
                    .take((mismatch_trace_index + 1) as usize)
                    .collect();
            }
            TraceDiffType::Mismatch => {
                receipt.execution_trace[mismatch_trace_index as usize] = Default::default();
            }
        }

        receipt.execution_trace_root = {
            let trace: Vec<_> = receipt
                .execution_trace
                .iter()
                .map(|t| t.encode().try_into().unwrap())
                .collect();
            MerkleTree::from_leaves(trace.as_slice())
                .root()
                .unwrap()
                .into()
        };
        receipt.final_state_root = *receipt.execution_trace.last().unwrap();
        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        (
            opaque_bundle.receipt().hash::<BlakeTwo256>(),
            bundle_to_tx(opaque_bundle),
        )
    };

    // Wait for the fraud proof that targets the bad ER
    let wait_for_fraud_proof_fut = ferdie.wait_for_fraud_proof(move |fp| {
        if let FraudProofVariant::InvalidStateTransition(proof) = &fp.proof {
            match (trace_diff_type, mismatch_trace_index) {
                (TraceDiffType::Mismatch, mismatch_trace_index) => match mismatch_trace_index {
                    0 => assert!(matches!(
                        proof.execution_phase,
                        ExecutionPhase::InitializeBlock
                    )),
                    // 1 for the inherent timestamp extrinsic, 1 for the inherent `set_consensus_chain_byte_fee`
                    // extrinsic, 3 for the above `transfer_allow_death` extrinsic
                    1..=3 => assert!(matches!(
                        proof.execution_phase,
                        ExecutionPhase::ApplyExtrinsic {
                            mismatch: ApplyExtrinsicMismatch::StateRoot(trace_index),
                            ..
                        } if trace_index == mismatch_trace_index
                    )),
                    4 => assert!(matches!(
                        proof.execution_phase,
                        ExecutionPhase::FinalizeBlock {
                            mismatch: FinalizeBlockMismatch::StateRoot
                        }
                    )),
                    _ => unreachable!(),
                },
                (TraceDiffType::Shorter, _) => {
                    assert!(matches!(
                        proof.execution_phase,
                        ExecutionPhase::ApplyExtrinsic {
                            mismatch: ApplyExtrinsicMismatch::Shorter,
                            ..
                        }
                    ))
                }
                (TraceDiffType::Longer, _) => {
                    assert!(matches!(
                        proof.execution_phase,
                        ExecutionPhase::FinalizeBlock {
                            mismatch: FinalizeBlockMismatch::Longer(trace_index)
                        } if trace_index as usize == original_length - 1
                    ))
                }
            }
            true
        } else {
            false
        }
    });

    // Produce a consensus block that contains `bad_submit_bundle_tx` and the bad receipt should
    // be added to the consensus chain block tree
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();
    assert!(ferdie.does_receipt_exist(bad_receipt_hash).unwrap());

    // When the system domain node process the primary block that contains `bad_submit_bundle_tx`,
    // it will generate and submit a fraud proof
    let timed_out = tokio::time::timeout(TIMEOUT, wait_for_fraud_proof_fut)
        .await
        .inspect_err(|_| error!("fraud proof was not created before the timeout"))
        .is_err();

    // Produce a consensus block that contains the fraud proof, the fraud proof wil be verified
    // and executed, and prune the bad receipt from the block tree
    ferdie.produce_blocks(1).await.unwrap();
    assert!(!ferdie.does_receipt_exist(bad_receipt_hash).unwrap());
    // We check for timeouts last, because they are the least useful test failure message.
    assert!(!timed_out);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_true_invalid_bundles_inherent_extrinsic_proof_creation_and_verification() {
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

    let bundle_to_tx = |opaque_bundle| {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into()
    };

    let inherent_extrinsic = || {
        let now = 1234;
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_timestamp::Call::set { now }.into(),
        )
        .into()
    };

    produce_blocks!(ferdie, alice, 5).await.unwrap();

    alice
        .construct_and_send_extrinsic(pallet_balances::Call::transfer_allow_death {
            dest: Bob.to_account_id(),
            value: 1,
        })
        .await
        .expect("Failed to send extrinsic");

    // Produce a bundle that contains the previously sent extrinsic and record that bundle for later use
    let (slot, target_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(target_bundle.extrinsics.len(), 1);
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();

    // Get a bundle from the txn pool and modify the receipt of the target bundle to an invalid one
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let extrinsics: Vec<Vec<u8>>;
    let bundle_extrinsic_root;
    let bad_submit_bundle_tx = {
        opaque_bundle.extrinsics.push(inherent_extrinsic());
        extrinsics = opaque_bundle
            .extrinsics
            .clone()
            .into_iter()
            .map(|ext| ext.encode())
            .collect();
        bundle_extrinsic_root =
            BlakeTwo256::ordered_trie_root(extrinsics.clone(), StateVersion::V1);
        opaque_bundle.sealed_header.header.bundle_extrinsics_root = bundle_extrinsic_root;
        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        bundle_to_tx(opaque_bundle)
    };

    // Produce a block that contains `bad_submit_bundle_tx`
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();

    // produce another bundle that marks the previous extrinsic as invalid.
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;

    let (bad_receipt_hash, bad_submit_bundle_tx) = {
        let bad_receipt = &mut opaque_bundle.sealed_header.header.receipt;
        // bad receipt marks this particular bundle as valid even though bundle contains inherent extrinsic
        bad_receipt.inboxed_bundles =
            vec![InboxedBundle::valid(H256::random(), bundle_extrinsic_root)];

        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        (
            opaque_bundle.receipt().hash::<BlakeTwo256>(),
            bundle_to_tx(opaque_bundle),
        )
    };

    // Wait for the fraud proof that targets the bad ER
    let wait_for_fraud_proof_fut = ferdie.wait_for_fraud_proof(move |fp| {
        if let FraudProofVariant::InvalidBundles(proof) = &fp.proof {
            if let InvalidBundleType::InherentExtrinsic(_) = proof.invalid_bundle_type {
                assert!(proof.is_true_invalid_fraud_proof);
                return true;
            }
        }
        false
    });

    // Produce a consensus block that contains `bad_submit_bundle_tx` and the bad receipt should
    // be added to the consensus chain block tree
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();
    assert!(ferdie.does_receipt_exist(bad_receipt_hash).unwrap());

    let timed_out = tokio::time::timeout(TIMEOUT, wait_for_fraud_proof_fut)
        .await
        .inspect_err(|_| error!("fraud proof was not created before the timeout"))
        .is_err();

    // Produce a consensus block that contains the fraud proof, the fraud proof wil be verified
    // and executed, and prune the bad receipt from the block tree
    ferdie.produce_blocks(1).await.unwrap();
    assert!(!ferdie.does_receipt_exist(bad_receipt_hash).unwrap());
    // We check for timeouts last, because they are the least useful test failure message.
    assert!(!timed_out);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_false_invalid_bundles_inherent_extrinsic_proof_creation_and_verification() {
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

    let bundle_to_tx = |opaque_bundle| {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into()
    };

    produce_blocks!(ferdie, alice, 5).await.unwrap();

    alice
        .construct_and_send_extrinsic(pallet_balances::Call::transfer_allow_death {
            dest: Bob.to_account_id(),
            value: 1,
        })
        .await
        .expect("Failed to send extrinsic");

    // Produce a bundle that contains the previously sent extrinsic and record that bundle for later use
    let (slot, target_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(target_bundle.extrinsics.len(), 1);
    let extrinsics: Vec<Vec<u8>> = target_bundle
        .extrinsics
        .clone()
        .into_iter()
        .map(|ext| ext.encode())
        .collect();
    let bundle_extrinsic_root =
        BlakeTwo256::ordered_trie_root(extrinsics.clone(), StateVersion::V1);
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();

    // produce another bundle that marks the previous valid extrinsic as invalid.
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;

    let (bad_receipt_hash, bad_submit_bundle_tx) = {
        let bad_receipt = &mut opaque_bundle.sealed_header.header.receipt;
        // bad receipt marks this particular bundle as invalid even though bundle does not contain
        // inherent extrinsic
        bad_receipt.inboxed_bundles = vec![InboxedBundle::invalid(
            InvalidBundleType::InherentExtrinsic(0),
            bundle_extrinsic_root,
        )];

        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        (
            opaque_bundle.receipt().hash::<BlakeTwo256>(),
            bundle_to_tx(opaque_bundle),
        )
    };

    // Wait for the fraud proof that targets the bad ER
    let wait_for_fraud_proof_fut = ferdie.wait_for_fraud_proof(move |fp| {
        if let FraudProofVariant::InvalidBundles(proof) = &fp.proof {
            if let InvalidBundleType::InherentExtrinsic(_) = proof.invalid_bundle_type {
                assert!(!proof.is_true_invalid_fraud_proof);
                return true;
            }
        }
        false
    });

    // Produce a consensus block that contains `bad_submit_bundle_tx` and the bad receipt should
    // be added to the consensus chain block tree
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();
    assert!(ferdie.does_receipt_exist(bad_receipt_hash).unwrap());

    let timed_out = tokio::time::timeout(TIMEOUT, wait_for_fraud_proof_fut)
        .await
        .inspect_err(|_| error!("fraud proof was not created before the timeout"))
        .is_err();

    // Produce a consensus block that contains the fraud proof, the fraud proof wil be verified
    // and executed, and prune the bad receipt from the block tree
    ferdie.produce_blocks(1).await.unwrap();
    assert!(!ferdie.does_receipt_exist(bad_receipt_hash).unwrap());
    // We check for timeouts last, because they are the least useful test failure message.
    assert!(!timed_out);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_true_invalid_bundles_undecodeable_tx_proof_creation_and_verification() {
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

    let bundle_to_tx = |opaque_bundle| {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into()
    };

    let undecodable_tx = || {
        let undecodable_extrinsic = rand::random::<[u8; 5]>().to_vec();
        OpaqueExtrinsic::from_bytes(&undecodable_extrinsic.encode())
            .expect("raw byte encoding and decoding never fails; qed")
    };

    produce_blocks!(ferdie, alice, 5).await.unwrap();

    alice
        .construct_and_send_extrinsic(pallet_balances::Call::transfer_allow_death {
            dest: Bob.to_account_id(),
            value: 1,
        })
        .await
        .expect("Failed to send extrinsic");

    // Produce a bundle that contains the previously sent extrinsic and record that bundle for later use
    let (slot, target_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(target_bundle.extrinsics.len(), 1);
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();

    // Get a bundle from the txn pool and modify the receipt of the target bundle to an invalid one
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let extrinsics: Vec<Vec<u8>>;
    let bundle_extrinsic_root;
    let bad_submit_bundle_tx = {
        opaque_bundle.extrinsics.push(undecodable_tx());
        extrinsics = opaque_bundle
            .extrinsics
            .clone()
            .into_iter()
            .map(|ext| ext.encode())
            .collect();
        bundle_extrinsic_root =
            BlakeTwo256::ordered_trie_root(extrinsics.clone(), StateVersion::V1);
        opaque_bundle.sealed_header.header.bundle_extrinsics_root = bundle_extrinsic_root;
        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        bundle_to_tx(opaque_bundle)
    };

    // Produce a block that contains `bad_submit_bundle_tx`
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();

    // produce another bundle that marks the previous extrinsic as invalid.
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;

    let (bad_receipt_hash, bad_submit_bundle_tx) = {
        let bad_receipt = &mut opaque_bundle.sealed_header.header.receipt;
        // bad receipt marks this particular bundle as valid even though bundle contains inherent extrinsic
        bad_receipt.inboxed_bundles =
            vec![InboxedBundle::valid(H256::random(), bundle_extrinsic_root)];

        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        (
            opaque_bundle.receipt().hash::<BlakeTwo256>(),
            bundle_to_tx(opaque_bundle),
        )
    };

    // Wait for the fraud proof that targets the bad ER
    let wait_for_fraud_proof_fut = ferdie.wait_for_fraud_proof(move |fp| {
        if let FraudProofVariant::InvalidBundles(proof) = &fp.proof {
            if let InvalidBundleType::UndecodableTx(_) = proof.invalid_bundle_type {
                assert!(proof.is_true_invalid_fraud_proof);
                return true;
            }
        }
        false
    });

    // Produce a consensus block that contains `bad_submit_bundle_tx` and the bad receipt should
    // be added to the consensus chain block tree
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();
    assert!(ferdie.does_receipt_exist(bad_receipt_hash).unwrap());

    let timed_out = tokio::time::timeout(TIMEOUT, wait_for_fraud_proof_fut)
        .await
        .inspect_err(|_| error!("fraud proof was not created before the timeout"))
        .is_err();

    // Produce a consensus block that contains the fraud proof, the fraud proof wil be verified
    // and executed, and prune the bad receipt from the block tree
    ferdie.produce_blocks(1).await.unwrap();
    assert!(!ferdie.does_receipt_exist(bad_receipt_hash).unwrap());
    // We check for timeouts last, because they are the least useful test failure message.
    assert!(!timed_out);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_false_invalid_bundles_undecodeable_tx_proof_creation_and_verification() {
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

    let bundle_to_tx = |opaque_bundle| {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into()
    };

    produce_blocks!(ferdie, alice, 5).await.unwrap();

    alice
        .construct_and_send_extrinsic(pallet_balances::Call::transfer_allow_death {
            dest: Bob.to_account_id(),
            value: 1,
        })
        .await
        .expect("Failed to send extrinsic");

    // Produce a bundle that contains the previously sent extrinsic and record that bundle for later use
    let (slot, target_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(target_bundle.extrinsics.len(), 1);
    let extrinsics: Vec<Vec<u8>> = target_bundle
        .extrinsics
        .clone()
        .into_iter()
        .map(|ext| ext.encode())
        .collect();
    let bundle_extrinsic_root =
        BlakeTwo256::ordered_trie_root(extrinsics.clone(), StateVersion::V1);
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();

    // produce another bundle that marks the previous valid extrinsic as invalid.
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;

    let (bad_receipt_hash, bad_submit_bundle_tx) = {
        let bad_receipt = &mut opaque_bundle.sealed_header.header.receipt;
        // bad receipt marks this particular bundle as invalid even though bundle does not contain
        // inherent extrinsic
        bad_receipt.inboxed_bundles = vec![InboxedBundle::invalid(
            InvalidBundleType::UndecodableTx(0),
            bundle_extrinsic_root,
        )];

        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        (
            opaque_bundle.receipt().hash::<BlakeTwo256>(),
            bundle_to_tx(opaque_bundle),
        )
    };

    // Wait for the fraud proof that targets the bad ER
    let wait_for_fraud_proof_fut = ferdie.wait_for_fraud_proof(move |fp| {
        if let FraudProofVariant::InvalidBundles(proof) = &fp.proof {
            if let InvalidBundleType::UndecodableTx(_) = proof.invalid_bundle_type {
                assert!(!proof.is_true_invalid_fraud_proof);
                return true;
            }
        }
        false
    });

    // Produce a consensus block that contains `bad_submit_bundle_tx` and the bad receipt should
    // be added to the consensus chain block tree
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();
    assert!(ferdie.does_receipt_exist(bad_receipt_hash).unwrap());

    let timed_out = tokio::time::timeout(TIMEOUT, wait_for_fraud_proof_fut)
        .await
        .inspect_err(|_| error!("fraud proof was not created before the timeout"))
        .is_err();

    // Produce a consensus block that contains the fraud proof, the fraud proof wil be verified
    // and executed, and prune the bad receipt from the block tree
    ferdie.produce_blocks(1).await.unwrap();
    assert!(!ferdie.does_receipt_exist(bad_receipt_hash).unwrap());
    // We check for timeouts last, because they are the least useful test failure message.
    assert!(!timed_out);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_true_invalid_bundles_illegal_xdm_proof_creation_and_verification() {
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

    let bundle_to_tx = |opaque_bundle| {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into()
    };

    // Produce a bundle that contains the previously sent extrinsic and record that bundle for later use
    let (slot, target_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(target_bundle.extrinsics.len(), 0);
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();

    // Get a bundle from the txn pool and modify the receipt of the target bundle to an invalid one
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;

    let bundle_extrinsic_root;
    let bad_submit_bundle_tx = {
        let invalid_xdm = evm_domain_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_messenger::Call::relay_message {
                msg: CrossDomainMessage {
                    src_chain_id: ChainId::Consensus,
                    dst_chain_id: ChainId::Domain(EVM_DOMAIN_ID),
                    channel_id: Default::default(),
                    nonce: Default::default(),
                    proof: Proof::Domain {
                        consensus_chain_mmr_proof: ConsensusChainMmrLeafProof {
                            consensus_block_number: 1,
                            consensus_block_hash: Default::default(),
                            opaque_mmr_leaf: EncodableOpaqueLeaf(vec![0, 1, 2]),
                            proof: MmrProof {
                                leaf_indices: vec![],
                                leaf_count: 0,
                                items: vec![],
                            },
                        },
                        domain_proof: StorageProof::empty(),
                        message_proof: StorageProof::empty(),
                    },
                    weight_tag: Default::default(),
                },
            }
            .into(),
        )
        .into();
        opaque_bundle.extrinsics = vec![invalid_xdm];
        let extrinsics: Vec<Vec<u8>> = opaque_bundle
            .extrinsics
            .clone()
            .into_iter()
            .map(|ext| ext.encode())
            .collect();
        bundle_extrinsic_root =
            BlakeTwo256::ordered_trie_root(extrinsics.clone(), StateVersion::V1);
        opaque_bundle.sealed_header.header.bundle_extrinsics_root = bundle_extrinsic_root;
        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        bundle_to_tx(opaque_bundle)
    };

    // Produce a block that contains `bad_submit_bundle_tx`
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();

    // produce another bundle that marks the previous extrinsic as invalid.
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;

    let (bad_receipt_hash, bad_submit_bundle_tx) = {
        let bad_receipt = &mut opaque_bundle.sealed_header.header.receipt;
        // bad receipt marks this particular bundle as valid even though bundle contains illegal extrinsic
        bad_receipt.inboxed_bundles =
            vec![InboxedBundle::valid(H256::random(), bundle_extrinsic_root)];

        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        (
            opaque_bundle.receipt().hash::<BlakeTwo256>(),
            bundle_to_tx(opaque_bundle),
        )
    };

    // Wait for the fraud proof that targets the bad ER
    let wait_for_fraud_proof_fut = ferdie.wait_for_fraud_proof(move |fp| {
        if let FraudProofVariant::InvalidBundles(proof) = &fp.proof {
            if let InvalidBundleType::InvalidXDM(extrinsic_index) = proof.invalid_bundle_type {
                assert!(proof.is_true_invalid_fraud_proof);
                assert_eq!(extrinsic_index, 0);
                return true;
            }
        }
        false
    });

    // Produce a consensus block that contains `bad_submit_bundle_tx` and the bad receipt should
    // be added to the consensus chain block tree
    // Produce a block that contains `bad_submit_bundle_tx`
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();
    assert!(ferdie.does_receipt_exist(bad_receipt_hash).unwrap());

    let timed_out = tokio::time::timeout(TIMEOUT, wait_for_fraud_proof_fut)
        .await
        .inspect_err(|_| error!("fraud proof was not created before the timeout"))
        .is_err();

    // Produce a consensus block that contains the fraud proof, the fraud proof wil be verified
    // and executed, and prune the bad receipt from the block tree
    ferdie.produce_blocks(1).await.unwrap();
    assert!(!ferdie.does_receipt_exist(bad_receipt_hash).unwrap());
    // We check for timeouts last, because they are the least useful test failure message.
    assert!(!timed_out);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_true_invalid_bundles_illegal_extrinsic_proof_creation_and_verification() {
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

    let bundle_to_tx = |opaque_bundle| {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into()
    };

    // Produce a bundle that contains the previously sent extrinsic and record that bundle for later use
    let (slot, target_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(target_bundle.extrinsics.len(), 0);
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();

    // Get a bundle from the txn pool and modify the receipt of the target bundle to an invalid one
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;

    let alice_balance = alice.free_balance(Alice.to_account_id());
    let mut alice_nonce = alice.account_nonce();
    let transfer_to_charlie_with_big_tip_1 = alice.construct_extrinsic_with_tip(
        alice_nonce,
        alice_balance / 3,
        pallet_balances::Call::transfer_allow_death {
            dest: Charlie.to_account_id(),
            value: 1,
        },
    );
    alice_nonce += 1;

    let transfer_to_charlie_with_big_tip_2 = alice.construct_extrinsic_with_tip(
        alice_nonce,
        alice_balance / 3,
        pallet_balances::Call::transfer_allow_death {
            dest: Charlie.to_account_id(),
            value: 1,
        },
    );
    alice_nonce += 1;

    // This would be illegal
    let transfer_to_charlie_with_big_tip_3 = alice.construct_extrinsic_with_tip(
        alice_nonce,
        alice_balance / 3,
        pallet_balances::Call::transfer_allow_death {
            dest: Charlie.to_account_id(),
            value: 1,
        },
    );

    let extrinsics: Vec<Vec<u8>>;
    let bundle_extrinsic_root;
    let bad_submit_bundle_tx = {
        opaque_bundle.extrinsics = vec![
            transfer_to_charlie_with_big_tip_1.into(),
            transfer_to_charlie_with_big_tip_2.into(),
            transfer_to_charlie_with_big_tip_3.into(),
        ];
        extrinsics = opaque_bundle
            .extrinsics
            .clone()
            .into_iter()
            .map(|ext| ext.encode())
            .collect();
        bundle_extrinsic_root =
            BlakeTwo256::ordered_trie_root(extrinsics.clone(), StateVersion::V1);
        opaque_bundle.sealed_header.header.bundle_extrinsics_root = bundle_extrinsic_root;
        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        bundle_to_tx(opaque_bundle)
    };

    // Produce a block that contains `bad_submit_bundle_tx`
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();

    // produce another bundle that marks the previous extrinsic as invalid.
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;

    let (bad_receipt_hash, bad_submit_bundle_tx) = {
        let bad_receipt = &mut opaque_bundle.sealed_header.header.receipt;
        // bad receipt marks this particular bundle as valid even though bundle contains illegal extrinsic
        bad_receipt.inboxed_bundles =
            vec![InboxedBundle::valid(H256::random(), bundle_extrinsic_root)];

        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        (
            opaque_bundle.receipt().hash::<BlakeTwo256>(),
            bundle_to_tx(opaque_bundle),
        )
    };

    // Wait for the fraud proof that targets the bad ER
    let wait_for_fraud_proof_fut = ferdie.wait_for_fraud_proof(move |fp| {
        if let FraudProofVariant::InvalidBundles(proof) = &fp.proof {
            if let InvalidBundleType::IllegalTx(extrinsic_index) = proof.invalid_bundle_type {
                assert!(proof.is_true_invalid_fraud_proof);
                assert_eq!(extrinsic_index, 2);
                return true;
            }
        }
        false
    });

    // Produce a consensus block that contains `bad_submit_bundle_tx` and the bad receipt should
    // be added to the consensus chain block tree
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();
    assert!(ferdie.does_receipt_exist(bad_receipt_hash).unwrap());

    let timed_out = tokio::time::timeout(TIMEOUT, wait_for_fraud_proof_fut)
        .await
        .inspect_err(|_| error!("fraud proof was not created before the timeout"))
        .is_err();

    // Produce a consensus block that contains the fraud proof, the fraud proof wil be verified
    // and executed, and prune the bad receipt from the block tree
    ferdie.produce_blocks(1).await.unwrap();
    assert!(!ferdie.does_receipt_exist(bad_receipt_hash).unwrap());
    // We check for timeouts last, because they are the least useful test failure message.
    assert!(!timed_out);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_false_invalid_bundles_illegal_extrinsic_proof_creation_and_verification() {
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

    let bundle_to_tx = |opaque_bundle| {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into()
    };

    // transfer most of the alice's balance
    let alice_balance = alice.free_balance(Alice.to_account_id());

    let mut alice_nonce = alice.account_nonce();

    let transfer_to_charlie_with_big_tip_1 = alice.construct_extrinsic_with_tip(
        alice_nonce,
        alice_balance / 3,
        pallet_balances::Call::transfer_allow_death {
            dest: Charlie.to_account_id(),
            value: 1,
        },
    );
    alice_nonce += 1;

    let transfer_to_charlie_with_big_tip_2 = alice.construct_extrinsic_with_tip(
        alice_nonce,
        alice_balance / 3,
        pallet_balances::Call::transfer_allow_death {
            dest: Charlie.to_account_id(),
            value: 1,
        },
    );

    produce_blocks!(ferdie, alice, 5).await.unwrap();

    alice
        .send_extrinsic(transfer_to_charlie_with_big_tip_1.clone())
        .await
        .expect("Failed to send extrinsic");

    alice
        .send_extrinsic(transfer_to_charlie_with_big_tip_2.clone())
        .await
        .expect("Failed to send extrinsic");

    // Produce a bundle that contains the previously sent extrinsic and record that bundle for later use
    let (slot, target_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(target_bundle.extrinsics.len(), 2);
    let bundle_extrinsic_root = target_bundle.extrinsics_root();
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();

    // produce another bundle that marks the previous valid extrinsic as invalid.
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;

    let (bad_receipt_hash, bad_submit_bundle_tx) = {
        let bad_receipt = &mut opaque_bundle.sealed_header.header.receipt;
        // bad receipt marks this particular bundle as invalid even though bundle does not contain
        // illegal tx
        bad_receipt.inboxed_bundles = vec![InboxedBundle::invalid(
            InvalidBundleType::IllegalTx(1),
            bundle_extrinsic_root,
        )];

        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        (
            opaque_bundle.receipt().hash::<BlakeTwo256>(),
            bundle_to_tx(opaque_bundle),
        )
    };

    // Wait for the fraud proof that targets the bad ER
    let wait_for_fraud_proof_fut = ferdie.wait_for_fraud_proof(move |fp| {
        if let FraudProofVariant::InvalidBundles(proof) = &fp.proof {
            if let InvalidBundleType::IllegalTx(extrinsic_index) = proof.invalid_bundle_type {
                assert!(!proof.is_true_invalid_fraud_proof);
                assert_eq!(extrinsic_index, 1);
                return true;
            }
        }
        false
    });

    // Produce a consensus block that contains `bad_submit_bundle_tx` and the bad receipt should
    // be added to the consensus chain block tree
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();
    assert!(ferdie.does_receipt_exist(bad_receipt_hash).unwrap());

    let timed_out = tokio::time::timeout(TIMEOUT, wait_for_fraud_proof_fut)
        .await
        .inspect_err(|_| error!("fraud proof was not created before the timeout"))
        .is_err();

    // Produce a consensus block that contains the fraud proof, the fraud proof wil be verified
    // and executed, and prune the bad receipt from the block tree
    ferdie.produce_blocks(1).await.unwrap();
    assert!(!ferdie.does_receipt_exist(bad_receipt_hash).unwrap());
    // We check for timeouts last, because they are the least useful test failure message.
    assert!(!timed_out);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_true_invalid_bundle_weight_proof_creation_and_verification() {
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

    let bundle_to_tx = |opaque_bundle| {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into()
    };

    produce_blocks!(ferdie, alice, 5).await.unwrap();

    alice
        .construct_and_send_extrinsic(pallet_balances::Call::transfer_allow_death {
            dest: Bob.to_account_id(),
            value: 1,
        })
        .await
        .expect("Failed to send extrinsic");

    // Get a bundle from the txn pool and modify the receipt of the target bundle to an invalid one
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(opaque_bundle.extrinsics.len(), 1);
    let bundle_extrinsic_root = opaque_bundle.sealed_header.header.bundle_extrinsics_root;
    let bad_submit_bundle_tx = {
        opaque_bundle.sealed_header.header.estimated_bundle_weight = Weight::from_all(123456);
        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        bundle_to_tx(opaque_bundle)
    };

    // Produce a block that contains `bad_submit_bundle_tx`
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();

    // produce another bundle that marks the bundle as valid.
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;

    let (bad_receipt_hash, bad_submit_bundle_tx) = {
        let bad_receipt = &mut opaque_bundle.sealed_header.header.receipt;
        // bad receipt marks this particular bundle as valid even though bundle contains invalid `estimated_bundle_weight`
        bad_receipt.inboxed_bundles =
            vec![InboxedBundle::valid(H256::random(), bundle_extrinsic_root)];

        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        (
            opaque_bundle.receipt().hash::<BlakeTwo256>(),
            bundle_to_tx(opaque_bundle),
        )
    };

    // Wait for the fraud proof that targets the bad ER
    let wait_for_fraud_proof_fut = ferdie.wait_for_fraud_proof(move |fp| {
        if let FraudProofVariant::InvalidBundles(proof) = &fp.proof {
            if InvalidBundleType::InvalidBundleWeight == proof.invalid_bundle_type {
                assert!(proof.is_true_invalid_fraud_proof);
                return true;
            }
        }
        false
    });

    // Produce a consensus block that contains `bad_submit_bundle_tx` and the bad receipt should
    // be added to the consensus chain block tree
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();
    assert!(ferdie.does_receipt_exist(bad_receipt_hash).unwrap());

    let timed_out = tokio::time::timeout(TIMEOUT, wait_for_fraud_proof_fut)
        .await
        .inspect_err(|_| error!("fraud proof was not created before the timeout"))
        .is_err();

    // Produce a consensus block that contains the fraud proof, the fraud proof wil be verified
    // and executed, and prune the bad receipt from the block tree
    ferdie.produce_blocks(1).await.unwrap();
    assert!(!ferdie.does_receipt_exist(bad_receipt_hash).unwrap());
    // We check for timeouts last, because they are the least useful test failure message.
    assert!(!timed_out);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_false_invalid_bundle_weight_proof_creation_and_verification() {
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

    let bundle_to_tx = |opaque_bundle| {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into()
    };

    produce_blocks!(ferdie, alice, 5).await.unwrap();

    alice
        .construct_and_send_extrinsic(pallet_balances::Call::transfer_allow_death {
            dest: Bob.to_account_id(),
            value: 1,
        })
        .await
        .expect("Failed to send extrinsic");

    // Produce a bundle that contains the previously sent extrinsic and record that bundle for later use
    let (slot, target_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(target_bundle.extrinsics.len(), 1);
    let extrinsics: Vec<Vec<u8>> = target_bundle
        .extrinsics
        .clone()
        .into_iter()
        .map(|ext| ext.encode())
        .collect();
    let bundle_extrinsic_root = BlakeTwo256::ordered_trie_root(extrinsics, StateVersion::V1);
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();

    // produce another bundle that marks the previous valid bundle as invalid.
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;

    let (bad_receipt_hash, bad_submit_bundle_tx) = {
        let bad_receipt = &mut opaque_bundle.sealed_header.header.receipt;
        // bad receipt marks this particular bundle as `InvalidBundleWeight`
        bad_receipt.inboxed_bundles = vec![InboxedBundle::invalid(
            InvalidBundleType::InvalidBundleWeight,
            bundle_extrinsic_root,
        )];

        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        (
            opaque_bundle.receipt().hash::<BlakeTwo256>(),
            bundle_to_tx(opaque_bundle),
        )
    };

    // Wait for the fraud proof that targets the bad ER
    let wait_for_fraud_proof_fut = ferdie.wait_for_fraud_proof(move |fp| {
        if let FraudProofVariant::InvalidBundles(proof) = &fp.proof {
            if InvalidBundleType::InvalidBundleWeight == proof.invalid_bundle_type {
                assert!(!proof.is_true_invalid_fraud_proof);
                return true;
            }
        }
        false
    });

    // Produce a consensus block that contains `bad_submit_bundle_tx` and the bad receipt should
    // be added to the consensus chain block tree
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();
    assert!(ferdie.does_receipt_exist(bad_receipt_hash).unwrap());

    let timed_out = tokio::time::timeout(TIMEOUT, wait_for_fraud_proof_fut)
        .await
        .inspect_err(|_| error!("fraud proof was not created before the timeout"))
        .is_err();

    // Produce a consensus block that contains the fraud proof, the fraud proof wil be verified
    // and executed, and prune the bad receipt from the block tree
    ferdie.produce_blocks(1).await.unwrap();
    assert!(!ferdie.does_receipt_exist(bad_receipt_hash).unwrap());
    // We check for timeouts last, because they are the least useful test failure message.
    assert!(!timed_out);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_false_invalid_bundles_non_exist_extrinsic_proof_creation_and_verification() {
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

    let bundle_to_tx = |opaque_bundle| {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into()
    };

    produce_blocks!(ferdie, alice, 5).await.unwrap();

    alice
        .construct_and_send_extrinsic(pallet_balances::Call::transfer_allow_death {
            dest: Bob.to_account_id(),
            value: 1,
        })
        .await
        .expect("Failed to send extrinsic");

    // Produce a bundle that contains the previously sent extrinsic and record that bundle for later use
    let (slot, target_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(target_bundle.extrinsics.len(), 1);
    let extrinsics: Vec<Vec<u8>> = target_bundle
        .extrinsics
        .clone()
        .into_iter()
        .map(|ext| ext.encode())
        .collect();
    let bundle_extrinsic_root =
        BlakeTwo256::ordered_trie_root(extrinsics.clone(), StateVersion::V1);
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();

    // produce another bundle that marks the previous valid extrinsic as invalid.
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;

    let (bad_receipt_hash, bad_submit_bundle_tx) = {
        let bad_receipt = &mut opaque_bundle.sealed_header.header.receipt;
        // bad receipt marks a non-exist extrinsic as invalid
        bad_receipt.inboxed_bundles = vec![InboxedBundle::invalid(
            InvalidBundleType::InherentExtrinsic(u32::MAX),
            bundle_extrinsic_root,
        )];

        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        (
            opaque_bundle.receipt().hash::<BlakeTwo256>(),
            bundle_to_tx(opaque_bundle),
        )
    };

    // Wait for the fraud proof that targets the bad ER
    let wait_for_fraud_proof_fut = ferdie.wait_for_fraud_proof(move |fp| {
        if let FraudProofVariant::InvalidBundles(proof) = &fp.proof {
            if let InvalidBundlesProofData::Bundle(_) = proof.proof_data {
                assert_eq!(fp.targeted_bad_receipt_hash(), bad_receipt_hash);
                return true;
            }
        }
        false
    });

    // Produce a consensus block that contains `bad_submit_bundle_tx` and the bad receipt should
    // be added to the consensus chain block tree
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();
    assert!(ferdie.does_receipt_exist(bad_receipt_hash).unwrap());

    let timed_out = tokio::time::timeout(TIMEOUT, wait_for_fraud_proof_fut)
        .await
        .inspect_err(|_| error!("fraud proof was not created before the timeout"))
        .is_err();

    // Produce a consensus block that contains the fraud proof, the fraud proof wil be verified
    // and executed, and prune the bad receipt from the block tree
    ferdie.produce_blocks(1).await.unwrap();
    assert!(!ferdie.does_receipt_exist(bad_receipt_hash).unwrap());
    // We check for timeouts last, because they are the least useful test failure message.
    assert!(!timed_out);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_invalid_block_fees_proof_creation() {
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

    let bundle_to_tx = |opaque_bundle| {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into()
    };

    produce_blocks!(ferdie, alice, 5).await.unwrap();

    // Get a bundle from the txn pool and modify the receipt of the target bundle to an invalid one
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let (bad_receipt_hash, bad_submit_bundle_tx) = {
        let receipt = &mut opaque_bundle.sealed_header.header.receipt;
        receipt.block_fees.consensus_storage_fee = 12345;
        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        (
            opaque_bundle.receipt().hash::<BlakeTwo256>(),
            bundle_to_tx(opaque_bundle),
        )
    };

    // Wait for the fraud proof that targets the bad ER
    let wait_for_fraud_proof_fut = ferdie.wait_for_fraud_proof(move |fp| {
        matches!(
            fp.proof,
            FraudProofVariant::InvalidBlockFees(InvalidBlockFeesProof { .. })
        )
    });

    // Produce a consensus block that contains `bad_submit_bundle_tx` and the bad receipt should
    // be added to the consensus chain block tree
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();
    assert!(ferdie.does_receipt_exist(bad_receipt_hash).unwrap());

    // When the domain node operator processes the primary block that contains `bad_submit_bundle_tx`,
    // it will generate and submit a fraud proof
    let timed_out = tokio::time::timeout(TIMEOUT, wait_for_fraud_proof_fut)
        .await
        .inspect_err(|_| error!("fraud proof was not created before the timeout"))
        .is_err();

    // Produce a consensus block that contains the fraud proof, the fraud proof wil be verified
    // and executed, and prune the bad receipt from the block tree
    ferdie.produce_blocks(1).await.unwrap();
    assert!(!ferdie.does_receipt_exist(bad_receipt_hash).unwrap());
    // We check for timeouts last, because they are the least useful test failure message.
    assert!(!timed_out);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_invalid_transfers_fraud_proof() {
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

    let bundle_to_tx = |opaque_bundle| {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into()
    };

    produce_blocks!(ferdie, alice, 5).await.unwrap();

    alice
        .construct_and_send_extrinsic(pallet_balances::Call::transfer_allow_death {
            dest: Bob.to_account_id(),
            value: 1,
        })
        .await
        .expect("Failed to send extrinsic");

    // Produce a bundle that contains the previously sent extrinsic and record that bundle for later use
    let (slot, target_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(target_bundle.extrinsics.len(), 1);
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();

    // Get a bundle from the txn pool and modify the receipt of the target bundle to an invalid one
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let (bad_receipt_hash, bad_submit_bundle_tx) = {
        let receipt = &mut opaque_bundle.sealed_header.header.receipt;
        receipt.transfers = Transfers {
            transfers_in: BTreeMap::from([(ChainId::Consensus, 10 * SSC)]),
            transfers_out: BTreeMap::from([(ChainId::Consensus, 10 * SSC)]),
            rejected_transfers_claimed: Default::default(),
            transfers_rejected: Default::default(),
        };
        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        (
            opaque_bundle.receipt().hash::<BlakeTwo256>(),
            bundle_to_tx(opaque_bundle),
        )
    };

    // Wait for the fraud proof that targets the bad ER
    let wait_for_fraud_proof_fut = ferdie.wait_for_fraud_proof(move |fp| {
        matches!(
            fp.proof,
            FraudProofVariant::InvalidTransfers(InvalidTransfersProof { .. })
        )
    });

    // Produce a consensus block that contains `bad_submit_bundle_tx` and the bad receipt should
    // be added to the consensus chain block tree
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();
    assert!(ferdie.does_receipt_exist(bad_receipt_hash).unwrap());

    // When the domain node operator processes the primary block that contains `bad_submit_bundle_tx`,
    // it will generate and submit a fraud proof
    let timed_out = tokio::time::timeout(TIMEOUT, wait_for_fraud_proof_fut)
        .await
        .inspect_err(|_| error!("fraud proof was not created before the timeout"))
        .is_err();

    // Produce a consensus block that contains the fraud proof, the fraud proof wil be verified
    // and executed, and prune the bad receipt from the block tree
    ferdie.produce_blocks(1).await.unwrap();
    assert!(!ferdie.does_receipt_exist(bad_receipt_hash).unwrap());
    // We check for timeouts last, because they are the least useful test failure message.
    assert!(!timed_out);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_invalid_domain_block_hash_proof_creation() {
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

    let bundle_to_tx = |opaque_bundle| {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into()
    };

    produce_blocks!(ferdie, alice, 5).await.unwrap();

    alice
        .construct_and_send_extrinsic(pallet_balances::Call::transfer_allow_death {
            dest: Bob.to_account_id(),
            value: 1,
        })
        .await
        .expect("Failed to send extrinsic");

    // Produce a bundle that contains the previously sent extrinsic and record that bundle for later use
    let (slot, target_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(target_bundle.extrinsics.len(), 1);
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();

    // Get a bundle from the txn pool and modify the receipt of the target bundle to an invalid one
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let (bad_receipt_hash, bad_submit_bundle_tx) = {
        let receipt = &mut opaque_bundle.sealed_header.header.receipt;
        receipt.domain_block_hash = Default::default();
        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        (
            opaque_bundle.receipt().hash::<BlakeTwo256>(),
            bundle_to_tx(opaque_bundle),
        )
    };

    // Wait for the fraud proof that targets the bad ER
    let wait_for_fraud_proof_fut = ferdie.wait_for_fraud_proof(move |fp| {
        matches!(
            fp.proof,
            FraudProofVariant::InvalidDomainBlockHash(InvalidDomainBlockHashProof { .. })
        )
    });

    // Produce a consensus block that contains `bad_submit_bundle_tx` and the bad receipt should
    // be added to the consensus chain block tree
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();
    assert!(ferdie.does_receipt_exist(bad_receipt_hash).unwrap());

    // When the domain node operator processes the primary block that contains `bad_submit_bundle_tx`,
    // it will generate and submit a fraud proof
    let timed_out = tokio::time::timeout(TIMEOUT, wait_for_fraud_proof_fut)
        .await
        .inspect_err(|_| error!("fraud proof was not created before the timeout"))
        .is_err();

    // Produce a consensus block that contains the fraud proof, the fraud proof wil be verified
    // and executed, and prune the bad receipt from the block tree
    ferdie.produce_blocks(1).await.unwrap();
    assert!(!ferdie.does_receipt_exist(bad_receipt_hash).unwrap());
    // We check for timeouts last, because they are the least useful test failure message.
    assert!(!timed_out);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_invalid_domain_extrinsics_root_proof_creation() {
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

    let bundle_to_tx = |opaque_bundle| {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into()
    };

    produce_blocks!(ferdie, alice, 5).await.unwrap();

    alice
        .construct_and_send_extrinsic(pallet_balances::Call::transfer_allow_death {
            dest: Bob.to_account_id(),
            value: 1,
        })
        .await
        .expect("Failed to send extrinsic");

    // Produce a bundle that contains the previously sent extrinsic and record that bundle for later use
    let (slot, target_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(target_bundle.extrinsics.len(), 1);
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();

    // Get a bundle from the txn pool and modify the receipt of the target bundle to an invalid one
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let (bad_receipt_hash, bad_submit_bundle_tx) = {
        let receipt = &mut opaque_bundle.sealed_header.header.receipt;
        receipt.domain_block_extrinsic_root = Default::default();
        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        (
            opaque_bundle.receipt().hash::<BlakeTwo256>(),
            bundle_to_tx(opaque_bundle),
        )
    };

    // Wait for the fraud proof that targets the bad ER
    let wait_for_fraud_proof_fut = ferdie.wait_for_fraud_proof(move |fp| {
        matches!(
            fp.proof,
            FraudProofVariant::InvalidExtrinsicsRoot(InvalidExtrinsicsRootProof { .. })
        )
    });

    // Produce a consensus block that contains `bad_submit_bundle_tx` and the bad receipt should
    // be added to the consensus chain block tree
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();
    assert!(ferdie.does_receipt_exist(bad_receipt_hash).unwrap());

    // When the domain node operator processes the primary block that contains `bad_submit_bundle_tx`,
    // it will generate and submit a fraud proof
    let timed_out = tokio::time::timeout(TIMEOUT, wait_for_fraud_proof_fut)
        .await
        .inspect_err(|_| error!("fraud proof was not created before the timeout"))
        .is_err();

    // Produce a consensus block that contains the fraud proof, the fraud proof wil be verified
    // and executed, and prune the bad receipt from the block tree
    ferdie.produce_blocks(1).await.unwrap();
    assert!(!ferdie.does_receipt_exist(bad_receipt_hash).unwrap());
    // We check for timeouts last, because they are the least useful test failure message.
    assert!(!timed_out);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_domain_block_builder_include_ext_with_failed_execution() {
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
    produce_blocks!(ferdie, alice, 1).await.unwrap();

    // produce a transfer from alice to bob
    // we send two transactions, first one must succeed, while second one will fail
    // second extrinsic fails due to not enough balance but should still be included in the domain
    // block body
    let pre_alice_free_balance = alice.free_balance(Alice.to_account_id());
    let alice_account_nonce = alice.account_nonce();

    // first ext that must succeed
    let tx = alice.construct_extrinsic(
        alice_account_nonce,
        pallet_balances::Call::transfer_allow_death {
            dest: Bob.to_account_id(),
            // we send half of alice balance to bob
            value: pre_alice_free_balance / 2,
        },
    );
    alice
        .send_extrinsic(tx)
        .await
        .expect("Failed to send extrinsic");

    // second ext that will fail due to balance restrictions.
    let tx = alice.construct_extrinsic(
        alice_account_nonce + 1,
        pallet_balances::Call::transfer_allow_death {
            dest: Bob.to_account_id(),
            // try to send value that is more than what alice has
            value: pre_alice_free_balance,
        },
    );
    alice
        .send_extrinsic(tx)
        .await
        .expect("Failed to send extrinsic");

    // Produce a bundle and submit to the tx pool of the consensus node
    let (_slot, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(bundle.extrinsics.len(), 2);

    // produce block and import domain block
    produce_blocks!(ferdie, alice, 1).await.unwrap();

    // domain block body should have 3 extrinsics
    // timestamp inherent, `consensus_chain_byte_fee` inherent, successful transfer, failed transfer
    let best_hash = alice.client.info().best_hash;
    let domain_block_extrinsics = alice.client.block_body(best_hash).unwrap();
    assert_eq!(domain_block_extrinsics.unwrap().len(), 4);

    // next bundle should have er which should a total of 5 trace roots
    // pre_timestamp_root + pre_consensus_chain_byte_fee_root + pre_success_ext_root + pre_failed_ext_root
    // + pre_finalize_block_root + post_finalize_block_root
    let (_slot, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let er = bundle.receipt();
    assert_eq!(er.execution_trace.len(), 6);
    assert_eq!(er.execution_trace[5], er.final_state_root);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_domain_block_builder_include_ext_with_failed_predispatch() {
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
    produce_blocks!(ferdie, alice, 1).await.unwrap();

    // produce a transfer from alice to bob in one bundle, which should succeed
    let pre_alice_free_balance = alice.free_balance(Alice.to_account_id());
    let alice_account_nonce = alice.account_nonce();

    let tx = alice.construct_extrinsic(
        alice_account_nonce,
        pallet_balances::Call::transfer_allow_death {
            dest: Bob.to_account_id(),
            // we send half of alice balance to bob
            value: pre_alice_free_balance / 2,
        },
    );
    alice
        .send_extrinsic(tx)
        .await
        .expect("Failed to send extrinsic");

    // Produce a bundle and submit to the tx pool of the consensus node
    let (_slot, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(bundle.extrinsics.len(), 1);

    // we produce another bundle with similar transaction
    // second one will fail at pre dispatch since we use same nonce
    // we change the value to something else so that ext is different
    // send the tip so that previous ext is replaced with current in Domain tx pool
    let tip = 1000000;
    let tx = alice.construct_extrinsic_with_tip(
        // use the same nonce
        alice_account_nonce,
        tip,
        pallet_balances::Call::transfer_allow_death {
            dest: Bob.to_account_id(),
            value: pre_alice_free_balance / 3,
        },
    );
    alice
        .send_extrinsic(tx)
        .await
        .expect("Failed to send extrinsic");

    // Produce a bundle and submit to the tx pool of the consensus node
    let (slot, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(bundle.extrinsics.len(), 1);

    // produce block and import domain block
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();

    // domain block body should have 3 extrinsics
    // timestamp inherent, successful transfer, failed transfer
    let best_hash = alice.client.info().best_hash;
    let domain_block_extrinsics = alice.client.block_body(best_hash).unwrap();
    assert_eq!(domain_block_extrinsics.clone().unwrap().len(), 4);

    // next bundle should have er which should a total of 5 trace roots
    // pre_timestamp_root + pre_consensus_chain_byte_fee_root + pre_success_ext_root + pre_failed_ext_root
    // + pre_finalize_block_root + post_finalize_block_root
    let (_slot, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let er = bundle.sealed_header.header.receipt;

    assert_eq!(er.execution_trace.len(), 6);
    assert_eq!(er.execution_trace[5], er.final_state_root);

    let header = alice.client.header(best_hash).unwrap().unwrap();
    assert_eq!(
        *header.extrinsics_root(),
        BlakeTwo256::ordered_trie_root(
            domain_block_extrinsics
                .unwrap()
                .iter()
                .map(Encode::encode)
                .collect(),
            sp_core::storage::StateVersion::V1,
        )
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_valid_bundle_proof_generation_and_verification() {
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

    for i in 0..3 {
        let tx = alice.construct_extrinsic(
            alice.account_nonce() + i,
            pallet_balances::Call::transfer_allow_death {
                dest: Bob.to_account_id(),
                value: 1,
            },
        );
        alice
            .send_extrinsic(tx)
            .await
            .expect("Failed to send extrinsic");

        // Produce a bundle and submit to the tx pool of the consensus node
        let (slot, _) = ferdie.produce_slot_and_wait_for_bundle_submission().await;

        // In the last iteration, produce a consensus block which will included all the previous bundles
        if i == 2 {
            produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
                .await
                .unwrap();
        }
    }
    let bundle_to_tx = |opaque_bundle| {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into()
    };
    let proof_to_tx = |fraud_proof| {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_fraud_proof { fraud_proof }.into(),
        )
        .into()
    };

    // Produce a bundle that will include the reciept of the last 3 bundles and modified the receipt's
    // `inboxed_bundles` field to make it invalid
    let (slot, mut bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let bundle_index = 1;
    let (bad_receipt, bad_submit_bundle_tx) = {
        assert_eq!(bundle.receipt().inboxed_bundles.len(), 3);

        bundle.sealed_header.header.receipt.inboxed_bundles[bundle_index].bundle =
            BundleValidity::Valid(H256::random());
        bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(bundle.sealed_header.pre_hash().as_ref())
            .into();

        (bundle.receipt().clone(), bundle_to_tx(bundle))
    };

    // Produce a consensus block that contains `bad_submit_bundle_tx` and the bad receipt should
    // be added to the consensus chain block tree
    let mut import_tx_stream = ferdie.transaction_pool.import_notification_stream();
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();
    assert!(ferdie
        .does_receipt_exist(bad_receipt.hash::<BlakeTwo256>())
        .unwrap());

    // When the domain node operator processes the primary block that contains `bad_submit_bundle_tx`,
    // it will generate and submit a fraud proof
    while let Some(ready_tx_hash) = import_tx_stream.next().await {
        let ready_tx = ferdie
            .transaction_pool
            .ready_transaction(&ready_tx_hash)
            .unwrap();
        let ext = subspace_test_runtime::UncheckedExtrinsic::decode(
            &mut ready_tx.data.encode().as_slice(),
        )
        .unwrap();
        if let subspace_test_runtime::RuntimeCall::Domains(
            pallet_domains::Call::submit_fraud_proof { fraud_proof },
        ) = ext.function
        {
            if let FraudProofVariant::ValidBundle(ref proof) = fraud_proof.proof {
                // The fraud proof is targetting the `bad_receipt`
                assert_eq!(
                    fraud_proof.bad_receipt_hash,
                    bad_receipt.hash::<HeaderHashingFor<Header>>()
                );

                // If the fraud proof target a non-exist receipt then it is invalid
                let mut bad_fraud_proof = fraud_proof.clone();
                bad_fraud_proof.bad_receipt_hash = H256::random();
                assert!(ferdie
                    .submit_transaction(proof_to_tx(bad_fraud_proof))
                    .await
                    .is_err());

                // If the fraud proof point to non-exist bundle then it is invalid
                let (mut bad_fraud_proof, mut bad_proof) = (fraud_proof.clone(), proof.clone());
                bad_proof.bundle_with_proof.bundle_index = u32::MAX;
                bad_fraud_proof.proof = FraudProofVariant::ValidBundle(bad_proof);
                assert!(ferdie
                    .submit_transaction(proof_to_tx(bad_fraud_proof))
                    .await
                    .is_err());

                break;
            }
        }
    }

    // Produce a consensus block that contains the fraud proof, the fraud proof wil be verified
    // and executed, and prune the bad receipt from the block tree
    ferdie.produce_blocks(1).await.unwrap();
    assert!(!ferdie
        .does_receipt_exist(bad_receipt.hash::<BlakeTwo256>())
        .unwrap());
}

// TODO: Add a new test which simulates a situation that an executor produces a fraud proof
// when an invalid receipt is received.

// TODO: construct a minimal consensus runtime code and use the `set_code` extrinsic to actually
// cover the case that the new domain runtime are updated accordingly upon the new consensus runtime.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn set_new_code_should_work() {
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

    produce_blocks!(ferdie, alice, 1).await.unwrap();

    let new_runtime_wasm_blob = b"new_runtime_wasm_blob".to_vec();

    let best_number = alice.client.info().best_number;
    let consensus_block_number = best_number + 1;
    // Although we're processing the bundle manually, the original bundle processor still works in
    // the meanwhile, it's possible the executor alice already processed this consensus block, expecting next
    // consensus block, in which case we use a dummy consensus hash instead.
    //
    // Nice to disable the built-in bundle processor and have a full control of the executor block
    // production manually.
    let consensus_block_hash = ferdie
        .client
        .hash(consensus_block_number)
        .unwrap()
        .unwrap_or_else(Hash::random);
    alice
        .operator
        .clone()
        .process_bundles((consensus_block_hash, consensus_block_number, true))
        .await;

    let best_hash = alice.client.info().best_hash;
    let state = alice.backend.state_at(best_hash).expect("Get state");
    let trie_backend = state.as_trie_backend();
    let state_runtime_code = sp_state_machine::backend::BackendRuntimeCode::new(trie_backend);
    let runtime_code = state_runtime_code.fetch_runtime_code().unwrap();
    let logs = alice.client.header(best_hash).unwrap().unwrap().digest.logs;
    if logs != vec![DigestItem::RuntimeEnvironmentUpdated] {
        let extrinsics = alice
            .client
            .block_body(best_hash)
            .unwrap()
            .unwrap()
            .into_iter()
            .map(|encoded_extrinsic| {
                UncheckedExtrinsic::decode(&mut encoded_extrinsic.encode().as_slice()).unwrap()
            })
            .collect::<Vec<_>>();
        panic!("`set_code` not executed, extrinsics in the block: {extrinsics:?}")
    }
    assert_eq!(runtime_code, new_runtime_wasm_blob);
}

#[tokio::test(flavor = "multi_thread")]
async fn pallet_domains_unsigned_extrinsics_should_work() {
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

    // Run Bob (a evm domain full node)
    let bob = domain_test_service::DomainNodeBuilder::new(
        tokio_handle,
        BasePath::new(directory.path().join("bob")),
    )
    .build_evm_node(Role::Full, Bob, &mut ferdie)
    .await;

    produce_blocks!(ferdie, alice, 1).await.unwrap();

    // Get a bundle from alice's tx pool and used as bundle template.
    let (slot, _) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let _alice_key = alice.key;
    // Drop alice in order to control the execution chain by submitting the receipts manually later.
    drop(alice);

    produce_block_with!(ferdie.produce_block_with_slot(slot), bob)
        .await
        .unwrap();

    // let ferdie_client = ferdie.client.clone();
    // let create_submit_bundle = |consensus_block_number: BlockNumber| {
    // let consensus_block_hash = ferdie_client.hash(consensus_block_number).unwrap().unwrap();
    // let execution_receipt =
    // crate::aux_schema::load_execution_receipt(&*bob.backend, consensus_block_hash)
    // .expect("Failed to load execution receipt from the local aux_db")
    // .unwrap_or_else(|| {
    // panic!(
    // "The requested execution receipt for block {consensus_block_number} does not exist"
    // )
    // });

    // let mut opaque_bundle = bundle_template.clone();
    // opaque_bundle.sealed_header.header.consensus_block_number = consensus_block_number;
    // opaque_bundle.sealed_header.header.consensus_block_hash = consensus_block_hash;
    // opaque_bundle.sealed_header.signature = alice_key
    // .pair()
    // .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
    // .into();
    // opaque_bundle.receipt = execution_receipt;

    //     subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
    //         pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
    //     )
    //     .into()
    // };

    // TODO: Unlock once `head_receipt_number` API is usable.
    // let ferdie_client = ferdie.client.clone();
    // let head_receipt_number = || {
    //     let best_hash = ferdie_client.info().best_hash;
    //     ferdie_client
    //         .runtime_api()
    //         .head_receipt_number(best_hash, DomainId::SYSTEM)
    //         .expect("Failed to get head receipt number")
    // };

    // ferdie
    //     .submit_transaction(create_submit_bundle(1))
    //     .await
    //     .unwrap();
    // produce_blocks!(ferdie, bob, 1).await.unwrap();
    // ferdie
    //     .submit_transaction(create_submit_bundle(2))
    //     .await
    //     .unwrap();
    // produce_blocks!(ferdie, bob, 1).await.unwrap();
    // assert_eq!(head_receipt_number(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn stale_and_in_future_bundle_should_be_rejected() {
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

    produce_blocks!(ferdie, alice, 10).await.unwrap();
    let bundle_to_tx = |opaque_bundle| {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into()
    };
    let slot_info = |slot, proof_of_time| OperatorSlotInfo {
        slot,
        proof_of_time,
    };
    let operator_id = 0;
    let mut bundle_producer = {
        let domain_bundle_proposer = DomainBundleProposer::new(
            EVM_DOMAIN_ID,
            alice.client.clone(),
            ferdie.client.clone(),
            alice.operator.transaction_pool.clone(),
        );
        let (bundle_sender, _bundle_receiver) =
            sc_utils::mpsc::tracing_unbounded("domain_bundle_stream", 100);
        TestBundleProducer::new(
            EVM_DOMAIN_ID,
            ferdie.client.clone(),
            alice.client.clone(),
            domain_bundle_proposer,
            Arc::new(bundle_sender),
            alice.operator.keystore.clone(),
            false,
            false,
        )
    };

    let (_, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    ferdie.clear_tx_pool().await.unwrap();

    // Bundle is valid and can submit to tx pool for 5 consensus blocks
    for _ in 0..5 {
        ferdie
            .submit_transaction(bundle_to_tx(bundle.clone()))
            .await
            .unwrap();

        ferdie.produce_block_with_extrinsics(vec![]).await.unwrap();
        ferdie.clear_tx_pool().await.unwrap();
    }

    // Bundle will be rejected because its PoT is stale now
    match ferdie
        .submit_transaction(bundle_to_tx(bundle))
        .await
        .unwrap_err()
    {
        sc_transaction_pool::error::Error::Pool(TxPoolError::InvalidTransaction(invalid_tx)) => {
            assert_eq!(invalid_tx, InvalidTransactionCode::Bundle.into())
        }
        e => panic!("Unexpected error: {e}"),
    }

    // Produce a bundle with slot newer than the consensus block slot but less its the future slot
    let slot = ferdie.produce_slot();
    let (_, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bundle_to_tx(bundle)])
        ),
        alice
    )
    .await
    .unwrap();

    // Bundle with unknow PoT and in future slot will be rejected before entring the tx pool
    let (valid_slot, valid_pot) = ferdie.produce_slot();
    let slot_in_future = u64::MAX.into();
    let unknow_pot = PotOutput::from(
        <&[u8] as TryInto<[u8; 16]>>::try_into(
            &subspace_runtime_primitives::Hash::random().to_fixed_bytes()[..16],
        )
        .expect("slice with length of 16 must able convert into [u8; 16]; qed"),
    );

    let valid_bundle = bundle_producer
        .produce_bundle(operator_id, slot_info(valid_slot, valid_pot))
        .await
        .unwrap()
        .and_then(|res| res.into_opaque_bundle())
        .unwrap();
    let bundle_with_unknow_pot = bundle_producer
        .produce_bundle(operator_id, slot_info(valid_slot, unknow_pot))
        .await
        .unwrap()
        .and_then(|res| res.into_opaque_bundle())
        .unwrap();
    let bundle_with_slot_in_future = bundle_producer
        .produce_bundle(operator_id, slot_info(slot_in_future, valid_pot))
        .await
        .unwrap()
        .and_then(|res| res.into_opaque_bundle())
        .unwrap();
    for bundle in [
        bundle_with_unknow_pot.clone(),
        bundle_with_slot_in_future.clone(),
    ] {
        match ferdie
            .submit_transaction(bundle_to_tx(bundle))
            .await
            .unwrap_err()
        {
            sc_transaction_pool::error::Error::Pool(TxPoolError::InvalidTransaction(
                invalid_tx,
            )) => {
                assert_eq!(invalid_tx, InvalidTransactionCode::Bundle.into())
            }
            e => panic!("Unexpected error: {e}"),
        }
    }
    ferdie
        .submit_transaction(bundle_to_tx(valid_bundle))
        .await
        .unwrap();
    produce_blocks!(ferdie, alice, 1).await.unwrap();

    // Even try building consensus block with these invalid bundles, they will fail to pass
    // the `pre_dispatch` check, and won't included in the consensus block
    let pre_ferdie_best_number = ferdie.client.info().best_number;
    let pre_alice_best_number = alice.client.info().best_number;

    ferdie
        .produce_block_with_extrinsics(vec![
            bundle_to_tx(bundle_with_unknow_pot),
            bundle_to_tx(bundle_with_slot_in_future),
        ])
        .await
        .unwrap();
    assert_eq!(ferdie.client.info().best_number, pre_ferdie_best_number + 1);
    assert_eq!(alice.client.info().best_number, pre_alice_best_number);
}

#[tokio::test(flavor = "multi_thread")]
async fn existing_bundle_can_be_resubmitted_to_new_fork() {
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

    produce_blocks!(ferdie, alice, 3).await.unwrap();

    let pre_alice_best_number = alice.client.info().best_number;
    let mut parent_hash = ferdie.client.info().best_hash;

    let (slot, opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let submit_bundle_tx = subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
        pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
    )
    .into();

    // Wait one block to ensure the bundle is stored on this fork
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();

    // Create fork that also contains the bundle and build one more blocks on it to make
    // it the new best fork
    parent_hash = ferdie
        .produce_block_with_slot_at(slot, parent_hash, Some(vec![submit_bundle_tx]))
        .await
        .unwrap();
    let slot = ferdie.produce_slot();
    ferdie
        .produce_block_with_slot_at(slot, parent_hash, Some(vec![]))
        .await
        .unwrap();

    assert_eq!(alice.client.info().best_number, pre_alice_best_number + 1);

    produce_blocks!(ferdie, alice, 1).await.unwrap();
    assert_eq!(alice.client.info().best_number, pre_alice_best_number + 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_domain_sudo_calls() {
    let directory = TempDir::new().expect("Must be able to create temporary directory");

    let mut builder = sc_cli::LoggerBuilder::new("");
    builder.with_colors(false);
    let _ = builder.init();

    let tokio_handle = tokio::runtime::Handle::current();

    // Start Ferdie with Alice Key since that is the sudo key
    let mut ferdie = MockConsensusNode::run(
        tokio_handle.clone(),
        Sr25519Alice,
        BasePath::new(directory.path().join("ferdie")),
    );

    // Run Alice (an evm domain)
    let mut alice = domain_test_service::DomainNodeBuilder::new(
        tokio_handle.clone(),
        BasePath::new(directory.path().join("alice")),
    )
    .build_evm_node(Role::Authority, Alice, &mut ferdie)
    .await;

    // Run the cross domain gossip message worker
    ferdie.start_cross_domain_gossip_message_worker();

    produce_blocks!(ferdie, alice, 3).await.unwrap();

    // add domain to consensus chain allowlist
    ferdie
        .construct_and_send_extrinsic_with(pallet_sudo::Call::sudo {
            call: Box::new(subspace_test_runtime::RuntimeCall::Messenger(
                pallet_messenger::Call::update_consensus_chain_allowlist {
                    update: ChainAllowlistUpdate::Add(ChainId::Domain(EVM_DOMAIN_ID)),
                },
            )),
        })
        .await
        .expect("Failed to construct and send consensus chain allowlist update");

    // produce another block so allowlist on consensus is updated
    produce_blocks!(ferdie, alice, 1).await.unwrap();

    // add consensus chain to domain chain allow list
    ferdie
        .construct_and_send_extrinsic_with(subspace_test_runtime::RuntimeCall::Messenger(
            pallet_messenger::Call::initiate_domain_update_chain_allowlist {
                domain_id: EVM_DOMAIN_ID,
                update: ChainAllowlistUpdate::Add(ChainId::Consensus),
            },
        ))
        .await
        .expect("Failed to construct and send domain chain allowlist update");

    // produce another block so allowlist on domain are updated
    produce_blocks!(ferdie, alice, 1).await.unwrap();

    // Open channel between the Consensus chain and EVM domains
    alice
        .construct_and_send_extrinsic(evm_domain_test_runtime::RuntimeCall::Messenger(
            pallet_messenger::Call::initiate_channel {
                dst_chain_id: ChainId::Consensus,
                params: pallet_messenger::InitiateChannelParams {
                    max_outgoing_messages: 100,
                },
            },
        ))
        .await
        .expect("Failed to construct and send extrinsic");

    // Wait until channel open
    produce_blocks_until!(ferdie, alice, {
        alice
            .get_open_channel_for_chain(ChainId::Consensus)
            .is_some()
    })
    .await
    .unwrap();

    produce_blocks!(ferdie, alice, 20).await.unwrap();

    // close channel on Domain using domain sudo.
    // Sudo on consensus chain will send a sudo call to domain
    // once the call is executed in the domain,
    // closing channel flow should happen and channel on both domain
    // consensus chain should be closed.
    let channel_id = alice
        .get_open_channel_for_chain(ChainId::Consensus)
        .unwrap();
    let sudo_unsigned_extrinsic = alice
        .construct_unsigned_extrinsic(evm_domain_test_runtime::RuntimeCall::Messenger(
            pallet_messenger::Call::close_channel {
                chain_id: ChainId::Consensus,
                channel_id,
            },
        ))
        .encode();
    ferdie
        .construct_and_send_extrinsic_with(pallet_sudo::Call::sudo {
            call: Box::new(subspace_test_runtime::RuntimeCall::Domains(
                pallet_domains::Call::send_domain_sudo_call {
                    domain_id: EVM_DOMAIN_ID,
                    call: sudo_unsigned_extrinsic,
                },
            )),
        })
        .await
        .expect("Failed to construct and send consensus chain to close channel");

    // Wait until channel close
    produce_blocks_until!(ferdie, alice, {
        alice
            .get_open_channel_for_chain(ChainId::Consensus)
            .is_none()
    })
    .await
    .unwrap();

    // produce 150 more blocks to ensure nothing went wrong
    produce_blocks!(ferdie, alice, 150).await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_xdm_between_consensus_and_domain_should_work() {
    let directory = TempDir::new().expect("Must be able to create temporary directory");

    let mut builder = sc_cli::LoggerBuilder::new("");
    builder.with_colors(false);
    let _ = builder.init();

    let tokio_handle = tokio::runtime::Handle::current();

    // Start Ferdie with Alice Key since that is the sudo key
    let mut ferdie = MockConsensusNode::run(
        tokio_handle.clone(),
        Sr25519Alice,
        BasePath::new(directory.path().join("ferdie")),
    );

    // Run Alice (an evm domain)
    let mut alice = domain_test_service::DomainNodeBuilder::new(
        tokio_handle.clone(),
        BasePath::new(directory.path().join("alice")),
    )
    .build_evm_node(Role::Authority, Alice, &mut ferdie)
    .await;

    // Run the cross domain gossip message worker
    ferdie.start_cross_domain_gossip_message_worker();

    produce_blocks!(ferdie, alice, 3).await.unwrap();

    // add domain to consensus chain allowlist
    ferdie
        .construct_and_send_extrinsic_with(pallet_sudo::Call::sudo {
            call: Box::new(subspace_test_runtime::RuntimeCall::Messenger(
                pallet_messenger::Call::update_consensus_chain_allowlist {
                    update: ChainAllowlistUpdate::Add(ChainId::Domain(EVM_DOMAIN_ID)),
                },
            )),
        })
        .await
        .expect("Failed to construct and send consensus chain allowlist update");

    // produce another block so allowlist on consensus is updated
    produce_blocks!(ferdie, alice, 1).await.unwrap();

    // add consensus chain to domain chain allow list
    ferdie
        .construct_and_send_extrinsic_with(subspace_test_runtime::RuntimeCall::Messenger(
            pallet_messenger::Call::initiate_domain_update_chain_allowlist {
                domain_id: EVM_DOMAIN_ID,
                update: ChainAllowlistUpdate::Add(ChainId::Consensus),
            },
        ))
        .await
        .expect("Failed to construct and send domain chain allowlist update");

    // produce another block so allowlist on  domain are updated
    produce_blocks!(ferdie, alice, 1).await.unwrap();

    // there should be zero channel updates on both consensus and domain chain
    assert!(get_channel_state(
        &*ferdie.client,
        ChainId::Consensus,
        EVM_DOMAIN_ID.into(),
        ChannelId::zero()
    )
    .unwrap()
    .is_none());

    assert!(get_channel_state(
        &*ferdie.client,
        EVM_DOMAIN_ID.into(),
        ChainId::Consensus,
        ChannelId::zero(),
    )
    .unwrap()
    .is_none());

    // Open channel between the Consensus chain and EVM domains
    alice
        .construct_and_send_extrinsic(evm_domain_test_runtime::RuntimeCall::Messenger(
            pallet_messenger::Call::initiate_channel {
                dst_chain_id: ChainId::Consensus,
                params: pallet_messenger::InitiateChannelParams {
                    max_outgoing_messages: 100,
                },
            },
        ))
        .await
        .expect("Failed to construct and send extrinsic");
    // Wait until channel open
    produce_blocks_until!(ferdie, alice, {
        alice
            .get_open_channel_for_chain(ChainId::Consensus)
            .is_some()
    })
    .await
    .unwrap();

    produce_blocks!(ferdie, alice, 20).await.unwrap();

    // there should be channel updates on both consensus and domain chain

    // consensus channel update
    let channel_update = get_channel_state(
        &*ferdie.client,
        ChainId::Consensus,
        EVM_DOMAIN_ID.into(),
        ChannelId::zero(),
    )
    .unwrap()
    .unwrap();

    // next channel inbox nonce on consensus from domain should be 1
    assert_eq!(channel_update.next_inbox_nonce, 1.into());

    // domain channel update
    let channel_update = get_channel_state(
        &*ferdie.client,
        EVM_DOMAIN_ID.into(),
        ChainId::Consensus,
        ChannelId::zero(),
    )
    .unwrap()
    .unwrap();

    // next channel outbox nonce on domain to consensus should be 1
    assert_eq!(channel_update.next_outbox_nonce, 1.into());
    // received outbox response nonce on domain from consensus chain should be 0
    assert_eq!(
        channel_update.latest_response_received_message_nonce,
        Some(0.into())
    );

    // Transfer balance from
    let pre_alice_free_balance = alice.free_balance(alice.key.to_account_id());
    let pre_ferdie_free_balance = ferdie.free_balance(ferdie.key.to_account_id());
    let transfer_amount = 10;
    alice
        .construct_and_send_extrinsic(pallet_transporter::Call::transfer {
            dst_location: pallet_transporter::Location {
                chain_id: ChainId::Consensus,
                account_id: AccountIdConverter::convert(Sr25519Alice.into()),
            },
            amount: transfer_amount,
        })
        .await
        .expect("Failed to construct and send extrinsic");
    // Wait until transfer succeed
    produce_blocks_until!(ferdie, alice, {
        let post_alice_free_balance = alice.free_balance(alice.key.to_account_id());
        let post_ferdie_free_balance = ferdie.free_balance(ferdie.key.to_account_id());

        post_alice_free_balance < pre_alice_free_balance - transfer_amount
            && post_ferdie_free_balance == pre_ferdie_free_balance + transfer_amount
    })
    .await
    .unwrap();

    produce_blocks!(ferdie, alice, 20).await.unwrap();

    // there should be channel updates on both consensus and domain chain

    // consensus channel update
    let channel_update = get_channel_state(
        &*ferdie.client,
        ChainId::Consensus,
        EVM_DOMAIN_ID.into(),
        ChannelId::zero(),
    )
    .unwrap()
    .unwrap();

    // next channel inbox nonce on consensus from domain should be 2
    assert_eq!(channel_update.next_inbox_nonce, 2.into());

    // close channel on consensus chain using sudo since
    // channel is opened on domain
    let channel_id = alice
        .get_open_channel_for_chain(ChainId::Consensus)
        .unwrap();
    ferdie
        .construct_and_send_extrinsic_with(pallet_sudo::Call::sudo {
            call: Box::new(subspace_test_runtime::RuntimeCall::Messenger(
                pallet_messenger::Call::close_channel {
                    chain_id: ChainId::Domain(EVM_DOMAIN_ID),
                    channel_id,
                },
            )),
        })
        .await
        .expect("Failed to construct and send consensus chain to close channel");
    // Wait until channel close
    produce_blocks_until!(ferdie, alice, {
        alice
            .get_open_channel_for_chain(ChainId::Consensus)
            .is_none()
    })
    .await
    .unwrap();

    produce_blocks!(ferdie, alice, 20).await.unwrap();

    // there should be channel updates on both consensus and domain chain

    // domain channel update
    let channel_update = get_channel_state(
        &*ferdie.client,
        EVM_DOMAIN_ID.into(),
        ChainId::Consensus,
        ChannelId::zero(),
    )
    .unwrap()
    .unwrap();

    // next channel outbox nonce on domain to consensus should be 2
    assert_eq!(channel_update.next_outbox_nonce, 2.into());
    // next channel inbox nonce on domain should be 1
    assert_eq!(channel_update.next_inbox_nonce, 1.into());
    // received outbox response nonce on domain from consensus chain should be 1
    assert_eq!(
        channel_update.latest_response_received_message_nonce,
        Some(1.into())
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_xdm_between_domains_should_work() {
    let directory = TempDir::new().expect("Must be able to create temporary directory");

    let mut builder = sc_cli::LoggerBuilder::new("");
    builder.with_colors(false);
    let _ = builder.init();

    let tokio_handle = tokio::runtime::Handle::current();

    // Start Ferdie
    let mut ferdie = MockConsensusNode::run(
        tokio_handle.clone(),
        Sr25519Alice,
        BasePath::new(directory.path().join("ferdie")),
    );

    // Run Alice (a system domain authority node)
    let mut alice = domain_test_service::DomainNodeBuilder::new(
        tokio_handle.clone(),
        BasePath::new(directory.path().join("alice")),
    )
    .build_evm_node(Role::Authority, Alice, &mut ferdie)
    .await;

    // Run Bob (a auto-id domain authority node)
    let mut bob = domain_test_service::DomainNodeBuilder::new(
        tokio_handle.clone(),
        BasePath::new(directory.path().join("bob")),
    )
    .build_auto_id_node(Role::Authority, Sr25519Keyring::Bob, &mut ferdie)
    .await;

    // Run the cross domain gossip message worker
    ferdie.start_cross_domain_gossip_message_worker();

    produce_blocks!(ferdie, alice, 3, bob).await.unwrap();

    // add consensus chain to domain chain allow list
    ferdie
        .construct_and_send_extrinsic_with(subspace_test_runtime::RuntimeCall::Messenger(
            pallet_messenger::Call::initiate_domain_update_chain_allowlist {
                domain_id: EVM_DOMAIN_ID,
                update: ChainAllowlistUpdate::Add(ChainId::Domain(AUTO_ID_DOMAIN_ID)),
            },
        ))
        .await
        .expect("Failed to construct and send domain chain allowlist update");

    // produce another block so allowlist on  domain are updated
    produce_blocks!(ferdie, alice, 1, bob).await.unwrap();

    // add consensus chain to domain chain allow list
    ferdie
        .construct_and_send_extrinsic_with(subspace_test_runtime::RuntimeCall::Messenger(
            pallet_messenger::Call::initiate_domain_update_chain_allowlist {
                domain_id: AUTO_ID_DOMAIN_ID,
                update: ChainAllowlistUpdate::Add(ChainId::Domain(EVM_DOMAIN_ID)),
            },
        ))
        .await
        .expect("Failed to construct and send domain chain allowlist update");

    // produce another block so allowlist on  domain are updated
    produce_blocks!(ferdie, alice, 1, bob).await.unwrap();

    // Open channel between the evm domain and the auto-id domain
    bob.construct_and_send_extrinsic(auto_id_domain_test_runtime::RuntimeCall::Messenger(
        pallet_messenger::Call::initiate_channel {
            dst_chain_id: ChainId::Domain(EVM_DOMAIN_ID),
            params: pallet_messenger::InitiateChannelParams {
                max_outgoing_messages: 100,
            },
        },
    ))
    .await
    .expect("Failed to construct and send extrinsic");

    // Wait until channel open
    produce_blocks_until!(
        ferdie,
        alice,
        {
            alice
                .get_open_channel_for_chain(ChainId::Domain(AUTO_ID_DOMAIN_ID))
                .is_some()
                && bob
                    .get_open_channel_for_chain(ChainId::Domain(EVM_DOMAIN_ID))
                    .is_some()
        },
        bob
    )
    .await
    .unwrap();

    // Transfer balance cross the system domain and the core payments domain
    let pre_alice_free_balance = alice.free_balance(alice.key.to_account_id());
    let pre_bob_free_balance = bob.free_balance(bob.key.to_account_id());
    let transfer_amount = 10;
    alice
        .construct_and_send_extrinsic(pallet_transporter::Call::transfer {
            dst_location: pallet_transporter::Location {
                chain_id: ChainId::Domain(AUTO_ID_DOMAIN_ID),
                account_id: AccountIdConverter::convert(Sr25519Keyring::Bob.into()),
            },
            amount: transfer_amount,
        })
        .await
        .expect("Failed to construct and send extrinsic");
    // Wait until transfer succeed
    produce_blocks_until!(
        ferdie,
        alice,
        {
            let post_alice_free_balance = alice.free_balance(alice.key.to_account_id());
            let post_bob_free_balance = bob.free_balance(bob.key.to_account_id());

            post_alice_free_balance <= pre_alice_free_balance - transfer_amount
                && post_bob_free_balance == pre_bob_free_balance + transfer_amount
        },
        bob
    )
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_unordered_cross_domains_message_should_work() {
    let directory = TempDir::new().expect("Must be able to create temporary directory");

    let mut builder = sc_cli::LoggerBuilder::new("");
    builder.with_colors(false);
    let _ = builder.init();

    let tokio_handle = tokio::runtime::Handle::current();

    // Start Ferdie
    let mut ferdie = MockConsensusNode::run(
        tokio_handle.clone(),
        Sr25519Alice,
        BasePath::new(directory.path().join("ferdie")),
    );

    // Run Alice (a system domain authority node)
    let mut alice = domain_test_service::DomainNodeBuilder::new(
        tokio_handle.clone(),
        BasePath::new(directory.path().join("alice")),
    )
    .build_evm_node(Role::Authority, Alice, &mut ferdie)
    .await;

    let evm_domain_tx_pool_sink = ferdie
        .xdm_gossip_worker_builder()
        .remove_chain_sink(&ChainId::Domain(EVM_DOMAIN_ID))
        .unwrap();
    let (reorder_xdm_sink, mut reorder_xdm_receiver) = tracing_unbounded("reorder_xdm", 100);
    ferdie
        .xdm_gossip_worker_builder()
        .push_chain_sink(ChainId::Domain(EVM_DOMAIN_ID), reorder_xdm_sink);

    alice
        .task_manager
        .spawn_essential_handle()
        .spawn_essential_blocking(
            "reordering-xdm",
            None,
            Box::pin(async move {
                let mut i = 0;
                let mut msg_buffer = VecDeque::new();
                while let Some(xdm) = reorder_xdm_receiver.next().await {
                    if i % 3 == 0 {
                        msg_buffer.push_back(xdm);
                        if let Some(xdm) = msg_buffer.pop_front() {
                            if i % 2 == 0 {
                                evm_domain_tx_pool_sink.unbounded_send(xdm).unwrap();
                            }
                        }
                    } else {
                        evm_domain_tx_pool_sink.unbounded_send(xdm).unwrap();
                    }
                    i += 1;
                }
            }),
        );

    // Run the cross domain gossip message worker
    ferdie.start_cross_domain_gossip_message_worker();

    produce_blocks!(ferdie, alice, 3).await.unwrap();

    // add domain to consensus chain allowlist
    ferdie
        .construct_and_send_extrinsic_with(pallet_sudo::Call::sudo {
            call: Box::new(subspace_test_runtime::RuntimeCall::Messenger(
                pallet_messenger::Call::update_consensus_chain_allowlist {
                    update: ChainAllowlistUpdate::Add(ChainId::Domain(EVM_DOMAIN_ID)),
                },
            )),
        })
        .await
        .expect("Failed to construct and send consensus chain allowlist update");

    // produce another block so allowlist on consensus is updated
    produce_blocks!(ferdie, alice, 1).await.unwrap();

    // add consensus chain to domain chain allow list
    ferdie
        .construct_and_send_extrinsic_with(subspace_test_runtime::RuntimeCall::Messenger(
            pallet_messenger::Call::initiate_domain_update_chain_allowlist {
                domain_id: EVM_DOMAIN_ID,
                update: ChainAllowlistUpdate::Add(ChainId::Consensus),
            },
        ))
        .await
        .expect("Failed to construct and send domain chain allowlist update");

    // produce another block so allowlist on  domain are updated
    produce_blocks!(ferdie, alice, 1).await.unwrap();

    // Open channel between the evm domain and the auto-id domain
    alice
        .construct_and_send_extrinsic(evm_domain_test_runtime::RuntimeCall::Messenger(
            pallet_messenger::Call::initiate_channel {
                dst_chain_id: ChainId::Consensus,
                params: pallet_messenger::InitiateChannelParams {
                    max_outgoing_messages: 100,
                },
            },
        ))
        .await
        .expect("Failed to construct and send extrinsic");

    // Wait until channel open
    produce_blocks_until!(ferdie, alice, {
        alice
            .get_open_channel_for_chain(ChainId::Consensus)
            .is_some()
    })
    .await
    .unwrap();

    // Transfer balance cross the system domain and the core payments domain
    let pre_alice_free_balance = alice.free_balance(alice.key.to_account_id());
    let pre_ferdie_free_balance = ferdie.free_balance(ferdie.key.to_account_id());
    let transfer_amount = 10;
    for _ in 0..20 {
        ferdie
            .construct_and_send_extrinsic_with(pallet_transporter::Call::transfer {
                dst_location: pallet_transporter::Location {
                    chain_id: ChainId::Domain(EVM_DOMAIN_ID),
                    account_id: AccountId20Converter::convert(Alice.to_account_id()),
                },
                amount: transfer_amount,
            })
            .await
            .expect("Failed to construct and send extrinsic");
        produce_blocks!(ferdie, alice, 1).await.unwrap();
    }
    // Wait until transfer succeed
    produce_blocks_until!(ferdie, alice, {
        let post_alice_free_balance = alice.free_balance(alice.key.to_account_id());
        let post_ferdie_free_balance = ferdie.free_balance(ferdie.key.to_account_id());

        post_alice_free_balance == pre_alice_free_balance + transfer_amount * 20
            && post_ferdie_free_balance <= pre_ferdie_free_balance - transfer_amount * 20
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
// TODO: https://github.com/autonomys/subspace/pull/1954 broke this on Windows, we suspect the test
//  is racy, but didn't find why and why it only fails on Windows. This needs to be fixed and test
//  un-ignored on Windows.
#[cfg_attr(windows, ignore)]
async fn test_restart_domain_operator() {
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

    produce_blocks!(ferdie, alice, 5).await.unwrap();
    let next_slot = ferdie.next_slot();

    // Stop Ferdie and Alice and delete their database lock files
    drop(ferdie);
    drop(alice);
    std::fs::remove_file(directory.path().join("ferdie/paritydb/lock")).unwrap();
    std::fs::remove_file(
        directory
            .path()
            .join(format!("alice/domain-{EVM_DOMAIN_ID:?}"))
            .as_path()
            .join("paritydb/lock"),
    )
    .unwrap();

    // Restart Ferdie
    let mut ferdie = MockConsensusNode::run(
        tokio_handle.clone(),
        Ferdie,
        BasePath::new(directory.path().join("ferdie")),
    );
    ferdie.set_next_slot(next_slot);

    // Restart Alice
    let alice = domain_test_service::DomainNodeBuilder::new(
        tokio_handle.clone(),
        BasePath::new(directory.path().join("alice")),
    )
    .build_evm_node(Role::Authority, Alice, &mut ferdie)
    .await;

    produce_blocks!(ferdie, alice, 5).await.unwrap();

    // Chain should progress base on previous run
    assert_eq!(ferdie.client.info().best_number, 10);
    assert_eq!(alice.client.info().best_number, 10);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_domain_transaction_fee_and_operator_reward() {
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

    // Construct and submit an extrinsic with tip
    let pre_alice_free_balance = alice.free_balance(Alice.to_account_id());
    let tip = 123456;
    let tx = alice.construct_extrinsic_with_tip(
        alice.account_nonce(),
        tip,
        frame_system::Call::remark {
            remark: vec![1; 10 * 1024],
        },
    );
    alice
        .send_extrinsic(tx)
        .await
        .expect("Failed to send extrinsic");

    // Produce a bundle that contains the just sent extrinsic
    let (slot, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(bundle.extrinsics.len(), 1);
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();
    let consensus_block_hash = ferdie.client.info().best_hash;

    // Produce one more bundle, this bundle should contains the ER of the previous bundle
    let (_, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let receipt = bundle.into_receipt();
    assert_eq!(receipt.consensus_block_hash, consensus_block_hash);

    // Transaction fee (including the tip) is deducted from alice's account
    let alice_free_balance_changes =
        pre_alice_free_balance - alice.free_balance(Alice.to_account_id());
    assert!(alice_free_balance_changes >= tip);

    let domain_block_fees = alice
        .client
        .runtime_api()
        .block_fees(receipt.domain_block_hash)
        .unwrap();

    // All the transaction fee is collected as operator reward
    assert_eq!(
        alice_free_balance_changes,
        domain_block_fees.domain_execution_fee + domain_block_fees.consensus_storage_fee
    );
    assert_eq!(domain_block_fees, receipt.block_fees);
    assert!(!domain_block_fees.consensus_storage_fee.is_zero());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_multiple_consensus_blocks_derive_similar_domain_block() {
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
    let common_block_hash = ferdie.client.info().best_hash;
    let bundle_to_tx = |opaque_bundle| {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into()
    };

    // submit a remark from alice
    let alice_account_nonce = alice.account_nonce();
    let tx = alice.construct_extrinsic(
        alice_account_nonce,
        frame_system::Call::remark {
            remark: vec![0u8; 8],
        },
    );
    alice
        .send_extrinsic(tx)
        .await
        .expect("Failed to send extrinsic");

    // Fork A with bundle that contains above transaction
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let consensus_block_hash_fork_a = ferdie
        .produce_block_with_slot_at(
            slot,
            common_block_hash,
            Some(vec![bundle_to_tx(opaque_bundle.clone())]),
        )
        .await
        .unwrap();

    // Fork B
    let bundle = {
        opaque_bundle.extrinsics = vec![];
        opaque_bundle.sealed_header.header.bundle_extrinsics_root =
            sp_domains::EMPTY_EXTRINSIC_ROOT;
        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        opaque_bundle
    };

    let consensus_block_hash_fork_b = ferdie
        .produce_block_with_slot_at(slot, common_block_hash, Some(vec![bundle_to_tx(bundle)]))
        .await
        .unwrap();
    // Produce one more consensus block to make fork B the best fork and trigger processing
    // on fork B
    let slot = ferdie.produce_slot();
    ferdie
        .produce_block_with_slot_at(slot, consensus_block_hash_fork_b, Some(vec![]))
        .await
        .unwrap();

    assert_ne!(consensus_block_hash_fork_a, consensus_block_hash_fork_b);

    // Both consensus block from fork A and B should derive a corresponding domain block
    let domain_block_hash_fork_a =
        crate::aux_schema::best_domain_hash_for(&*alice.client, &consensus_block_hash_fork_a)
            .unwrap()
            .unwrap();
    let domain_block_hash_fork_b =
        crate::aux_schema::best_domain_hash_for(&*alice.client, &consensus_block_hash_fork_b)
            .unwrap()
            .unwrap();
    assert_ne!(domain_block_hash_fork_a, domain_block_hash_fork_b);

    // The domain block header should contains digest that point to the consensus block, which
    // devrive the domain block
    let get_header = |hash| alice.client.header(hash).unwrap().unwrap();
    let get_digest_consensus_block_hash = |header: &Header| -> <CBlock as BlockT>::Hash {
        header
            .digest()
            .convert_first(DigestItem::as_consensus_block_info)
            .unwrap()
    };
    let domain_block_header_fork_a = get_header(domain_block_hash_fork_a);
    let domain_block_header_fork_b = get_header(domain_block_hash_fork_b);
    let digest_consensus_block_hash_fork_a =
        get_digest_consensus_block_hash(&domain_block_header_fork_a);
    assert_eq!(
        digest_consensus_block_hash_fork_a,
        consensus_block_hash_fork_a
    );
    let digest_consensus_block_hash_fork_b =
        get_digest_consensus_block_hash(&domain_block_header_fork_b);
    assert_eq!(
        digest_consensus_block_hash_fork_b,
        consensus_block_hash_fork_b
    );

    // The domain block headers should have the same `parent_hash`
    // but different digest and `state_root` (because digest is stored on chain)
    assert_eq!(
        domain_block_header_fork_a.parent_hash(),
        domain_block_header_fork_b.parent_hash()
    );
    assert_ne!(
        domain_block_header_fork_a.digest(),
        domain_block_header_fork_b.digest()
    );
    assert_ne!(
        domain_block_header_fork_a.state_root(),
        domain_block_header_fork_b.state_root()
    );

    // Simply produce more block
    produce_blocks!(ferdie, alice, 3).await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_skip_empty_bundle_production() {
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

    // Run Alice (a evm domain authority node) with `skip_empty_bundle_production` set to `true`
    let mut alice = domain_test_service::DomainNodeBuilder::new(
        tokio_handle.clone(),
        BasePath::new(directory.path().join("alice")),
    )
    .skip_empty_bundle()
    .build_evm_node(Role::Authority, Alice, &mut ferdie)
    .await;

    // Wait for `BlockTreePruningDepth + 1` blocks which is 10 + 1 in test
    // to enure the genesis ER is confirmed
    produce_blocks!(ferdie, alice, 11).await.unwrap();
    let consensus_block_number = ferdie.client.info().best_number;
    let domain_block_number = alice.client.info().best_number;

    let slot = ferdie.produce_slot();
    let bundle = ferdie.notify_new_slot_and_wait_for_bundle(slot).await;
    ferdie.produce_block_with_slot(slot).await.unwrap();

    // Alice will skip producing bundle since there is no domain extrinsic
    assert!(bundle.is_none());
    assert_eq!(ferdie.client.info().best_number, consensus_block_number + 1);
    assert_eq!(alice.client.info().best_number, domain_block_number);

    // Send a domain extrinsic, Alice will start producing bundle
    alice.send_system_remark().await;
    let (slot, _) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();
    assert_eq!(ferdie.client.info().best_number, consensus_block_number + 2);
    assert_eq!(alice.client.info().best_number, domain_block_number + 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_bad_receipt_chain() {
    let directory = TempDir::new().expect("Must be able to create temporary directory");

    let mut builder = sc_cli::LoggerBuilder::new("");
    builder.with_colors(false);
    let _ = builder.init();

    let tokio_handle = tokio::runtime::Handle::current();

    // Start Ferdie
    let mut ferdie = MockConsensusNode::run(
        tokio_handle.clone(),
        Sr25519Alice,
        BasePath::new(directory.path().join("ferdie")),
    );

    // Run Alice (a evm domain authority node)
    let alice = domain_test_service::DomainNodeBuilder::new(
        tokio_handle.clone(),
        BasePath::new(directory.path().join("alice")),
    )
    .build_evm_node(Role::Authority, Alice, &mut ferdie)
    .await;

    let bundle_to_tx = |opaque_bundle| {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into()
    };

    let mut bundle_producer = {
        let domain_bundle_proposer = DomainBundleProposer::new(
            EVM_DOMAIN_ID,
            alice.client.clone(),
            ferdie.client.clone(),
            alice.operator.transaction_pool.clone(),
        );
        let (bundle_sender, _bundle_receiver) =
            sc_utils::mpsc::tracing_unbounded("domain_bundle_stream", 100);
        TestBundleProducer::new(
            EVM_DOMAIN_ID,
            ferdie.client.clone(),
            alice.client.clone(),
            domain_bundle_proposer,
            Arc::new(bundle_sender),
            alice.operator.keystore.clone(),
            false,
            false,
        )
    };

    produce_blocks!(ferdie, alice, 15).await.unwrap();

    // Get a bundle from the txn pool and modify the receipt of the target bundle to an invalid one
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let stale_bundle = opaque_bundle.clone();
    let (bad_receipt_hash, bad_submit_bundle_tx) = {
        let receipt = &mut opaque_bundle.sealed_header.header.receipt;
        receipt.domain_block_hash = Default::default();
        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        (
            opaque_bundle.receipt().hash::<BlakeTwo256>(),
            bundle_to_tx(opaque_bundle),
        )
    };

    // Produce a consensus block that contains `bad_submit_bundle_tx` and the bad receipt should
    // be added to the consensus chain block tree
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();
    assert!(ferdie.does_receipt_exist(bad_receipt_hash).unwrap());

    // Remove the fraud proof from tx pool
    ferdie.clear_tx_pool().await.unwrap();

    // Wait for a fraud proof that target the first bad ER
    let wait_for_fraud_proof_fut = ferdie.wait_for_fraud_proof(move |fp| {
        matches!(
            fp.proof,
            FraudProofVariant::InvalidDomainBlockHash(InvalidDomainBlockHashProof { .. })
        ) && fp.targeted_bad_receipt_hash() == bad_receipt_hash
    });

    // Produce more bundle with bad ER that use previous bad ER as parent
    let mut parent_bad_receipt_hash = bad_receipt_hash;
    let mut bad_receipt_descendants = vec![];
    for _ in 0..7 {
        let slot = ferdie.produce_slot();
        let bundle = bundle_producer
            .produce_bundle(
                0,
                OperatorSlotInfo {
                    slot: slot.0,
                    proof_of_time: slot.1,
                },
            )
            .await
            .expect("produce bundle must success")
            .and_then(|res| res.into_opaque_bundle())
            .expect("must win the challenge");
        let (receipt_hash, bad_submit_bundle_tx) = {
            let mut opaque_bundle = bundle;
            let receipt = &mut opaque_bundle.sealed_header.header.receipt;
            receipt.parent_domain_block_receipt_hash = parent_bad_receipt_hash;
            receipt.domain_block_hash = Default::default();
            opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
                .pair()
                .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
                .into();
            (
                opaque_bundle.receipt().hash::<BlakeTwo256>(),
                bundle_to_tx(opaque_bundle),
            )
        };
        parent_bad_receipt_hash = receipt_hash;
        bad_receipt_descendants.push(receipt_hash);
        ferdie
            .produce_block_with_slot_at(
                slot,
                ferdie.client.info().best_hash,
                Some(vec![bad_submit_bundle_tx]),
            )
            .await
            .unwrap();
    }

    // All bad ERs should be added to the consensus chain block tree
    for receipt_hash in bad_receipt_descendants.iter().chain(&[bad_receipt_hash]) {
        assert!(ferdie.does_receipt_exist(*receipt_hash).unwrap());
    }

    // The fraud proof should be submitted
    let timed_out = tokio::time::timeout(TIMEOUT, wait_for_fraud_proof_fut)
        .await
        .inspect_err(|_| error!("fraud proof was not created before the timeout"))
        .is_err();

    // The first bad ER should be pruned and its descendants are marked as pending to prune
    ferdie.produce_blocks(1).await.unwrap();
    assert!(!ferdie.does_receipt_exist(bad_receipt_hash).unwrap());

    let ferdie_best_hash = ferdie.client.info().best_hash;
    let runtime_api = ferdie.client.runtime_api();
    for receipt_hash in &bad_receipt_descendants {
        assert!(ferdie.does_receipt_exist(*receipt_hash).unwrap());
        assert!(runtime_api
            .is_bad_er_pending_to_prune(ferdie_best_hash, EVM_DOMAIN_ID, *receipt_hash)
            .unwrap());
    }

    // There should be a receipt gap
    let head_domain_number = ferdie
        .client
        .runtime_api()
        .domain_best_number(ferdie_best_hash, EVM_DOMAIN_ID)
        .unwrap()
        .unwrap();
    let head_receipt_number = ferdie
        .client
        .runtime_api()
        .head_receipt_number(ferdie_best_hash, EVM_DOMAIN_ID)
        .unwrap();
    assert_eq!(head_domain_number - head_receipt_number, 9);
    // The previou bundle will be rejected as there is a receipt gap
    match ferdie
        .submit_transaction(bundle_to_tx(stale_bundle))
        .await
        .unwrap_err()
    {
        sc_transaction_pool::error::Error::Pool(TxPoolError::InvalidTransaction(invalid_tx)) => {
            assert_eq!(invalid_tx, InvalidTransactionCode::Bundle.into())
        }
        e => panic!("Unexpected error: {e}"),
    }

    // Register another operator as Alice is slashed
    ferdie
        .construct_and_send_extrinsic_with(pallet_domains::Call::register_operator {
            domain_id: EVM_DOMAIN_ID,
            amount: 1000 * SSC,
            config: OperatorConfig {
                signing_key: Sr25519Keyring::Charlie.public().into(),
                minimum_nominator_stake: Balance::MAX,
                nomination_tax: Default::default(),
            },
        })
        .await
        .unwrap();
    ferdie.produce_blocks(1).await.unwrap();

    ferdie
        .construct_and_send_extrinsic_with(pallet_sudo::Call::sudo {
            call: Box::new(subspace_test_runtime::RuntimeCall::Domains(
                pallet_domains::Call::force_staking_epoch_transition {
                    domain_id: EVM_DOMAIN_ID,
                },
            )),
        })
        .await
        .unwrap();
    ferdie.produce_blocks(1).await.unwrap();

    let alice_best_number = alice.client.info().best_number;
    drop(alice);

    // Start another operator node
    let bob = domain_test_service::DomainNodeBuilder::new(
        tokio_handle.clone(),
        BasePath::new(directory.path().join("bob")),
    )
    .operator_id(2)
    .build_evm_node(Role::Authority, Charlie, &mut ferdie)
    .await;
    ferdie.produce_blocks(1).await.unwrap();

    let bob_best_number = bob.client.info().best_number;
    assert_eq!(alice_best_number, bob_best_number);

    // Bad receipt should be pruned as singletone receipt submitting
    for receipt_hash in vec![bad_receipt_hash]
        .into_iter()
        .chain(bad_receipt_descendants)
    {
        let slot = ferdie.produce_slot();
        ferdie.notify_new_slot_and_wait_for_bundle(slot).await;
        ferdie.produce_block_with_slot(slot).await.unwrap();
        assert!(!ferdie.does_receipt_exist(receipt_hash).unwrap());
    }

    // The receipt gap should be fill up
    let ferdie_best_hash = ferdie.client.info().best_hash;
    let head_domain_number = ferdie
        .client
        .runtime_api()
        .domain_best_number(ferdie_best_hash, EVM_DOMAIN_ID)
        .unwrap()
        .unwrap();
    let head_receipt_number = ferdie
        .client
        .runtime_api()
        .head_receipt_number(ferdie_best_hash, EVM_DOMAIN_ID)
        .unwrap();
    assert_eq!(head_domain_number - head_receipt_number, 1);
    assert_eq!(bob_best_number, bob.client.info().best_number);

    produce_blocks!(ferdie, bob, 15).await.unwrap();
    // We check for timeouts last, because they are the least useful test failure message.
    assert!(!timed_out);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_domain_chain_storage_price_should_be_aligned_with_the_consensus_chain() {
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

    // The domain transaction byte is non-zero on the consensus chain genesis but
    // it is zero in the domain chain genesis
    let consensus_chain_byte_fee = ferdie
        .client
        .runtime_api()
        .consensus_chain_byte_fee(ferdie.client.info().best_hash)
        .unwrap();
    let operator_consensus_chain_byte_fee = alice
        .client
        .runtime_api()
        .consensus_chain_byte_fee(alice.client.info().best_hash)
        .unwrap();
    assert!(operator_consensus_chain_byte_fee.is_zero());
    assert!(!consensus_chain_byte_fee.is_zero());

    produce_blocks!(ferdie, alice, 1).await.unwrap();

    // The domain transaction byte of the domain chain should be updated to the consensus chain's
    // through the inherent extrinsic of domain block #1
    let consensus_chain_byte_fee = ferdie
        .client
        .runtime_api()
        .consensus_chain_byte_fee(ferdie.client.info().best_hash)
        .unwrap();
    let operator_consensus_chain_byte_fee = alice
        .client
        .runtime_api()
        .consensus_chain_byte_fee(alice.client.info().best_hash)
        .unwrap();
    assert_eq!(consensus_chain_byte_fee, operator_consensus_chain_byte_fee);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_skip_duplicated_tx_in_previous_bundle() {
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

    let bob_pre_balance = alice.free_balance(Bob.to_account_id());
    let call = pallet_balances::Call::transfer_allow_death {
        dest: Bob.to_account_id(),
        value: 1,
    };

    // Send a tx and produce a bundle, it will include the tx
    alice
        .construct_and_send_extrinsic_with(alice.account_nonce(), 0u32.into(), call.clone())
        .await
        .expect("Failed to send extrinsic");
    let (slot, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(bundle.extrinsics.len(), 1);

    // Produce a few more bundles, all of them will be empty since the only tx in the tx pool is already pick
    // up by the previous bundle
    for _ in 0..3 {
        let (_, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
        assert!(bundle.extrinsics.is_empty());
    }

    // Produce a domain that include all the bundles
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();
    assert_eq!(alice.free_balance(Bob.to_account_id()), bob_pre_balance + 1);

    // Produce a bundle with a tx but not include it in the next consensus block
    alice
        .construct_and_send_extrinsic_with(alice.account_nonce(), 0u32.into(), call.clone())
        .await
        .expect("Failed to send extrinsic");
    let (slot, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(bundle.extrinsics.len(), 1);
    ferdie
        .produce_block_with_slot_at(slot, ferdie.client.info().best_hash, Some(vec![]))
        .await
        .unwrap();

    // Even the tx is inclued in a previous bundle, after the consensus chain's tip changed, the operator
    // will resubmit the tx in the next bundle as retry
    let (slot, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(bundle.extrinsics.len(), 1);

    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();
    assert_eq!(alice.free_balance(Bob.to_account_id()), bob_pre_balance + 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_handle_duplicated_tx_with_diff_nonce_in_previous_bundle() {
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

    let nonce = alice.account_nonce();
    let bob_pre_balance = alice.free_balance(Bob.to_account_id());
    let call = pallet_balances::Call::transfer_allow_death {
        dest: Bob.to_account_id(),
        value: 1,
    };

    // Send a tx and produce a bundle, it will include the tx
    alice
        .construct_and_send_extrinsic_with(nonce, 0u32.into(), call.clone())
        .await
        .expect("Failed to send extrinsic");
    let (_, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(bundle.extrinsics.len(), 1);

    // Send a new tx with the same `nonce` and a tip then produce a bundle, this tx will replace
    // the previous tx in the tx pool and included in the bundle
    alice
        .construct_and_send_extrinsic_with(nonce, 1u32.into(), call.clone())
        .await
        .expect("Failed to send extrinsic");
    let (_, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(bundle.extrinsics.len(), 1);

    // Send a tx with `nonce + 1` and produce a bundle, it won't include this tx because the tx
    // with `nonce` is included in previous bundle and is not submitted to the consensus chain yet
    alice
        .construct_and_send_extrinsic_with(nonce + 1, 0u32.into(), call.clone())
        .await
        .expect("Failed to send extrinsic");
    let (slot, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert!(bundle.extrinsics.is_empty());

    // Produce a domain that include all the bundles
    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();
    assert_eq!(alice.free_balance(Bob.to_account_id()), bob_pre_balance + 1);

    // Send a tx with `nonce + 2` and produce a bundle, it will include both the previous `nonce + 1`
    // tx and the `nonce + 2` tx
    alice
        .construct_and_send_extrinsic_with(nonce + 2, 0u32.into(), call.clone())
        .await
        .expect("Failed to send extrinsic");
    let (slot, bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    assert_eq!(bundle.extrinsics.len(), 2);

    produce_block_with!(ferdie.produce_block_with_slot(slot), alice)
        .await
        .unwrap();
    assert_eq!(alice.free_balance(Bob.to_account_id()), bob_pre_balance + 3);
    assert_eq!(alice.account_nonce(), nonce + 3);
}

// TODO: add test to ensure MMR proof from diff fork wil be rejected
#[tokio::test(flavor = "multi_thread")]
async fn test_verify_mmr_proof_stateless() {
    use subspace_test_primitives::OnchainStateApi as _;
    let directory = TempDir::new().expect("Must be able to create temporary directory");

    let mut builder = sc_cli::LoggerBuilder::new("");
    builder.with_colors(false);
    let _ = builder.init();

    let tokio_handle = tokio::runtime::Handle::current();

    // Start Ferdie with Alice Key since that is the sudo key
    let mut ferdie = MockConsensusNode::run_with_finalization_depth(
        tokio_handle.clone(),
        Sr25519Alice,
        BasePath::new(directory.path().join("ferdie")),
        // finalization depth
        Some(10),
    );

    // Run Alice (an evm domain)
    let alice = domain_test_service::DomainNodeBuilder::new(
        tokio_handle.clone(),
        BasePath::new(directory.path().join("alice")),
    )
    .build_evm_node(Role::Authority, Alice, &mut ferdie)
    .await;

    produce_blocks!(ferdie, alice, 3).await.unwrap();

    let to_prove = ferdie.client.info().best_number;
    let expected_state_root = *ferdie
        .client
        .header(ferdie.client.info().best_hash)
        .unwrap()
        .unwrap()
        .state_root();

    // Can't generate MMR proof for the current best block
    assert!(generate_mmr_proof(&ferdie.client, to_prove).is_err());

    produce_blocks!(ferdie, alice, 1).await.unwrap();

    let proof = generate_mmr_proof(&ferdie.client, to_prove).unwrap();
    for i in 0..20 {
        let res = ferdie
            .client
            .runtime_api()
            .verify_proof_and_extract_leaf(ferdie.client.info().best_hash, proof.clone())
            .unwrap()
            .map(|leaf| leaf.state_root());

        produce_blocks!(ferdie, alice, 1).await.unwrap();

        // The MMR is proof is valid for the first `MmrRootHashCount` (i.e. 15) blocks but then expired
        if i < 15 {
            assert_eq!(res, Some(expected_state_root));
        } else {
            assert_eq!(res, None);
        }
    }

    // Re-generate MMR proof for the same block, it will be valid for the next `MmrRootHashCount` blocks
    let proof = generate_mmr_proof(&ferdie.client, to_prove).unwrap();
    for _ in 0..15 {
        let res = ferdie
            .client
            .runtime_api()
            .verify_proof_and_extract_leaf(ferdie.client.info().best_hash, proof.clone())
            .unwrap()
            .map(|leaf| leaf.state_root());

        assert_eq!(res, Some(expected_state_root));
        produce_blocks!(ferdie, alice, 1).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_equivocated_bundle_check() {
    let directory = TempDir::new().expect("Must be able to create temporary directory");

    let mut builder = sc_cli::LoggerBuilder::new("");
    builder.with_colors(false);
    let _ = builder.init();

    let tokio_handle = tokio::runtime::Handle::current();

    // Start Ferdie with Alice Key since that is the sudo key
    let mut ferdie = MockConsensusNode::run_with_finalization_depth(
        tokio_handle.clone(),
        Ferdie,
        BasePath::new(directory.path().join("ferdie")),
        // finalization depth
        Some(10),
    );

    // Run Alice (an evm domain)
    let alice = domain_test_service::DomainNodeBuilder::new(
        tokio_handle.clone(),
        BasePath::new(directory.path().join("alice")),
    )
    .build_evm_node(Role::Authority, Alice, &mut ferdie)
    .await;

    produce_blocks!(ferdie, alice, 3).await.unwrap();

    let bundle_to_tx = |opaque_bundle| -> OpaqueExtrinsic {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into()
    };

    // Get a bundle from the txn pool
    let (_, opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let proof_of_election = opaque_bundle.sealed_header.header.proof_of_election.clone();

    // Construct an equivocated bundle that with the same slot but different content
    let submit_equivocated_bundle_tx = {
        let mut equivocated_bundle = opaque_bundle.clone();
        equivocated_bundle
            .sealed_header
            .header
            .estimated_bundle_weight = Weight::from_all(123);
        equivocated_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(equivocated_bundle.sealed_header.pre_hash().as_ref())
            .into();
        bundle_to_tx(equivocated_bundle)
    };

    // Submit equivocated bundle to tx pool will failed because the equivocated bundle has
    // the same `provides` tag as the original bundle
    match ferdie
        .submit_transaction(submit_equivocated_bundle_tx.clone())
        .await
        .unwrap_err()
    {
        sc_transaction_pool::error::Error::Pool(TxPoolError::TooLowPriority { .. }) => {}
        e => panic!("Unexpected error: {e}"),
    }

    // Produce consensus block with equivocated bundle which will fail to include in block,
    // in production the whole block will be discard by the honest farmer
    ferdie
        .produce_block_with_extrinsics(vec![
            bundle_to_tx(opaque_bundle.clone()),
            submit_equivocated_bundle_tx,
        ])
        .await
        .unwrap();
    let block_hash = ferdie.client.info().best_hash;
    let block_body = ferdie.client.block_body(block_hash).unwrap().unwrap();
    let bundles = ferdie
        .client
        .runtime_api()
        .extract_successful_bundles(block_hash, EVM_DOMAIN_ID, block_body)
        .unwrap();
    assert_eq!(bundles, vec![opaque_bundle]);

    // Produce consensus block with duplicated bundle which will fail to include in block,
    // in production the whole block will be discard by the honest farmer
    let (_, opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    ferdie
        .produce_block_with_extrinsics(vec![
            bundle_to_tx(opaque_bundle.clone()),
            bundle_to_tx(opaque_bundle.clone()),
        ])
        .await
        .unwrap();
    let block_hash = ferdie.client.info().best_hash;
    let block_body = ferdie.client.block_body(block_hash).unwrap().unwrap();
    let bundles = ferdie
        .client
        .runtime_api()
        .extract_successful_bundles(block_hash, EVM_DOMAIN_ID, block_body)
        .unwrap();
    assert_eq!(bundles, vec![opaque_bundle]);

    // Construct an equivocated bundle that reuse an old `proof_of_election`
    let (_, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let submit_equivocated_bundle_tx = {
        opaque_bundle.sealed_header.header.proof_of_election = proof_of_election;
        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        bundle_to_tx(opaque_bundle)
    };
    // It will fail to submit to the tx pool
    match ferdie
        .submit_transaction(submit_equivocated_bundle_tx.clone())
        .await
        .unwrap_err()
    {
        sc_transaction_pool::error::Error::Pool(TxPoolError::InvalidTransaction(invalid_tx)) => {
            assert_eq!(invalid_tx, InvalidTransactionCode::Bundle.into())
        }
        e => panic!("Unexpected error: {e}"),
    }
    // Also fail to include in block
    let pre_ferdie_best_number = ferdie.client.info().best_number;
    let pre_alice_best_number = alice.client.info().best_number;
    ferdie
        .produce_block_with_extrinsics(vec![submit_equivocated_bundle_tx])
        .await
        .unwrap();
    assert_eq!(ferdie.client.info().best_number, pre_ferdie_best_number + 1);
    assert_eq!(alice.client.info().best_number, pre_alice_best_number);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_xdm_false_invalid_fraud_proof() {
    let directory = TempDir::new().expect("Must be able to create temporary directory");

    let mut builder = sc_cli::LoggerBuilder::new("");
    builder.with_colors(false);
    let _ = builder.init();

    let tokio_handle = tokio::runtime::Handle::current();

    // Start Ferdie with Alice Key since that is the sudo key
    let mut ferdie = MockConsensusNode::run(
        tokio_handle.clone(),
        Sr25519Alice,
        BasePath::new(directory.path().join("ferdie")),
    );

    // Run Alice (an evm domain)
    let mut alice = domain_test_service::DomainNodeBuilder::new(
        tokio_handle.clone(),
        BasePath::new(directory.path().join("alice")),
    )
    .build_evm_node(Role::Authority, Alice, &mut ferdie)
    .await;

    let bundle_to_tx = |opaque_bundle| {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into()
    };

    // Run the cross domain gossip message worker
    ferdie.start_cross_domain_gossip_message_worker();

    produce_blocks!(ferdie, alice, 3).await.unwrap();

    // add domain to consensus chain allowlist
    ferdie
        .construct_and_send_extrinsic_with(pallet_sudo::Call::sudo {
            call: Box::new(subspace_test_runtime::RuntimeCall::Messenger(
                pallet_messenger::Call::update_consensus_chain_allowlist {
                    update: ChainAllowlistUpdate::Add(ChainId::Domain(EVM_DOMAIN_ID)),
                },
            )),
        })
        .await
        .expect("Failed to construct and send consensus chain allowlist update");

    // produce another block so allowlist on consensus is updated
    produce_blocks!(ferdie, alice, 1).await.unwrap();

    // add consensus chain to domain chain allow list
    ferdie
        .construct_and_send_extrinsic_with(subspace_test_runtime::RuntimeCall::Messenger(
            pallet_messenger::Call::initiate_domain_update_chain_allowlist {
                domain_id: EVM_DOMAIN_ID,
                update: ChainAllowlistUpdate::Add(ChainId::Consensus),
            },
        ))
        .await
        .expect("Failed to construct and send domain chain allowlist update");

    // produce another block so allowlist on  domain are updated
    produce_blocks!(ferdie, alice, 1).await.unwrap();

    // Open channel between the Consensus chain and EVM domains
    alice
        .construct_and_send_extrinsic(evm_domain_test_runtime::RuntimeCall::Messenger(
            pallet_messenger::Call::initiate_channel {
                dst_chain_id: ChainId::Consensus,
                params: pallet_messenger::InitiateChannelParams {
                    max_outgoing_messages: 100,
                },
            },
        ))
        .await
        .expect("Failed to construct and send extrinsic");
    // Wait until channel open
    produce_blocks_until!(ferdie, alice, {
        alice
            .get_open_channel_for_chain(ChainId::Consensus)
            .is_some()
    })
    .await
    .unwrap();

    // Transfer balance through XDM
    ferdie
        .construct_and_send_extrinsic_with(pallet_transporter::Call::transfer {
            dst_location: pallet_transporter::Location {
                chain_id: ChainId::Domain(EVM_DOMAIN_ID),
                account_id: AccountId20Converter::convert(Alice.to_account_id()),
            },
            amount: 10,
        })
        .await
        .expect("Failed to construct and send extrinsic");

    // Wait until a bundle that cantains the XDM
    let mut maybe_opaque_bundle = None;
    produce_blocks_until!(ferdie, alice, {
        let alice_best_hash = alice.client.info().best_hash;
        let (_, opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
        for tx in opaque_bundle.extrinsics.iter() {
            if alice
                .client
                .runtime_api()
                .extract_xdm_mmr_proof(alice_best_hash, tx)
                .unwrap()
                .is_some()
            {
                maybe_opaque_bundle.replace(opaque_bundle);
                break;
            }
        }
        maybe_opaque_bundle.is_some()
    })
    .await
    .unwrap();

    // Produce a block that contains the XDM
    let opaque_bundle = maybe_opaque_bundle.unwrap();
    let bundle_extrinsic_root = opaque_bundle.extrinsics_root();
    let slot = ferdie.produce_slot();
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bundle_to_tx(opaque_bundle)])
        ),
        alice
    )
    .await
    .unwrap();

    // produce another bundle that marks the XDM as invalid.
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let (bad_receipt_hash, bad_submit_bundle_tx) = {
        let bad_receipt = &mut opaque_bundle.sealed_header.header.receipt;
        bad_receipt.inboxed_bundles = vec![InboxedBundle::invalid(
            InvalidBundleType::InvalidXDM(0),
            bundle_extrinsic_root,
        )];

        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        (
            opaque_bundle.receipt().hash::<BlakeTwo256>(),
            bundle_to_tx(opaque_bundle),
        )
    };

    // Wait for the fraud proof that targets the bad ER
    let wait_for_fraud_proof_fut = ferdie.wait_for_fraud_proof(move |fp| {
        if let FraudProofVariant::InvalidBundles(proof) = &fp.proof {
            if let InvalidBundleType::InvalidXDM(extrinsic_index) = proof.invalid_bundle_type {
                assert!(!proof.is_true_invalid_fraud_proof);
                assert_eq!(extrinsic_index, 0);
                return true;
            }
        }
        false
    });

    // Produce a consensus block that contains `bad_submit_bundle_tx` and the bad receipt should
    // be added to the consensus chain block tree
    // Produce a block that contains `bad_submit_bundle_tx`
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();
    assert!(ferdie.does_receipt_exist(bad_receipt_hash).unwrap());

    let timed_out = tokio::time::timeout(TIMEOUT, wait_for_fraud_proof_fut)
        .await
        .inspect_err(|_| error!("fraud proof was not created before the timeout"))
        .is_err();

    // Produce a consensus block that contains the fraud proof, the fraud proof wil be verified
    // and executed, and prune the bad receipt from the block tree
    ferdie.produce_blocks(1).await.unwrap();
    assert!(!ferdie.does_receipt_exist(bad_receipt_hash).unwrap());
    // We check for timeouts last, because they are the least useful test failure message.
    assert!(!timed_out);
}

// TODO: this test is flaky and sometime hang forever in CI thus disable it temporary,
// do investigate and fix it
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_stale_fork_xdm_true_invalid_fraud_proof() {
    let directory = TempDir::new().expect("Must be able to create temporary directory");

    let mut builder = sc_cli::LoggerBuilder::new("");
    builder.with_colors(false);
    let _ = builder.init();

    let tokio_handle = tokio::runtime::Handle::current();

    // Start Ferdie with Alice Key since that is the sudo key
    let mut ferdie = MockConsensusNode::run(
        tokio_handle.clone(),
        Sr25519Alice,
        BasePath::new(directory.path().join("ferdie")),
    );

    // Run Alice (an evm domain)
    let mut alice = domain_test_service::DomainNodeBuilder::new(
        tokio_handle.clone(),
        BasePath::new(directory.path().join("alice")),
    )
    .build_evm_node(Role::Authority, Alice, &mut ferdie)
    .await;

    let bundle_to_tx = |opaque_bundle| {
        subspace_test_runtime::UncheckedExtrinsic::new_unsigned(
            pallet_domains::Call::submit_bundle { opaque_bundle }.into(),
        )
        .into()
    };

    // Run the cross domain gossip message worker
    ferdie.start_cross_domain_gossip_message_worker();

    produce_blocks!(ferdie, alice, 3).await.unwrap();

    // add domain to consensus chain allowlist
    ferdie
        .construct_and_send_extrinsic_with(pallet_sudo::Call::sudo {
            call: Box::new(subspace_test_runtime::RuntimeCall::Messenger(
                pallet_messenger::Call::update_consensus_chain_allowlist {
                    update: ChainAllowlistUpdate::Add(ChainId::Domain(EVM_DOMAIN_ID)),
                },
            )),
        })
        .await
        .expect("Failed to construct and send consensus chain allowlist update");

    // produce another block so allowlist on consensus is updated
    produce_blocks!(ferdie, alice, 1).await.unwrap();

    // add consensus chain to domain chain allow list
    ferdie
        .construct_and_send_extrinsic_with(subspace_test_runtime::RuntimeCall::Messenger(
            pallet_messenger::Call::initiate_domain_update_chain_allowlist {
                domain_id: EVM_DOMAIN_ID,
                update: ChainAllowlistUpdate::Add(ChainId::Consensus),
            },
        ))
        .await
        .expect("Failed to construct and send domain chain allowlist update");

    // produce another block so allowlist on  domain are updated
    produce_blocks!(ferdie, alice, 1).await.unwrap();

    // Open channel between the Consensus chain and EVM domains
    alice
        .construct_and_send_extrinsic(evm_domain_test_runtime::RuntimeCall::Messenger(
            pallet_messenger::Call::initiate_channel {
                dst_chain_id: ChainId::Consensus,
                params: pallet_messenger::InitiateChannelParams {
                    max_outgoing_messages: 100,
                },
            },
        ))
        .await
        .expect("Failed to construct and send extrinsic");
    // Wait until channel open
    produce_blocks_until!(ferdie, alice, {
        alice
            .get_open_channel_for_chain(ChainId::Consensus)
            .is_some()
    })
    .await
    .unwrap();

    // Transfer balance through XDM
    ferdie
        .construct_and_send_extrinsic_with(pallet_transporter::Call::transfer {
            dst_location: pallet_transporter::Location {
                chain_id: ChainId::Domain(EVM_DOMAIN_ID),
                account_id: AccountId20Converter::convert(Alice.to_account_id()),
            },
            amount: 10,
        })
        .await
        .expect("Failed to construct and send extrinsic");

    // Wait until a bundle that cantains the XDM
    let mut maybe_opaque_bundle = None;
    produce_blocks_until!(ferdie, alice, {
        let alice_best_hash = alice.client.info().best_hash;
        let (_, opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
        for tx in opaque_bundle.extrinsics.iter() {
            if alice
                .client
                .runtime_api()
                .extract_xdm_mmr_proof(alice_best_hash, tx)
                .unwrap()
                .is_some()
            {
                maybe_opaque_bundle.replace(opaque_bundle);
                break;
            }
        }
        maybe_opaque_bundle.is_some()
    })
    .await
    .unwrap();

    let xdm = maybe_opaque_bundle.unwrap().extrinsics[0].clone();
    let mmr_proof_constructed_at_number = alice
        .client
        .runtime_api()
        .extract_xdm_mmr_proof(alice.client.info().best_hash, &xdm)
        .unwrap()
        .unwrap()
        .consensus_block_number;

    // Construct a fork and make it the best fork so the XDM mmr proof is in the stale fork
    let best_number = ferdie.client.info().best_number;
    let mut fork_block_hash = ferdie
        .client
        .hash(mmr_proof_constructed_at_number - 2)
        .unwrap()
        .unwrap();
    for _ in 0..(best_number - mmr_proof_constructed_at_number) + 5 {
        let slot = ferdie.produce_slot();
        fork_block_hash = ferdie
            .produce_block_with_slot_at(slot, fork_block_hash, Some(vec![]))
            .await
            .unwrap();
    }
    // The XDM should be rejected as its mmr proof is in the stale fork
    let alice_best_hash = alice.client.info().best_hash;
    let res = alice
        .client
        .runtime_api()
        .validate_transaction(
            alice_best_hash,
            TransactionSource::External,
            xdm.clone(),
            alice_best_hash,
        )
        .unwrap();
    assert_eq!(
        res,
        Err(TransactionValidityError::Invalid(
            InvalidTransaction::BadProof
        ))
    );

    // Construct an bundle that contain the invalid xdm and submit the bundle
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let (bundle_extrinsic_root, bad_submit_bundle_tx) = {
        opaque_bundle.extrinsics.push(xdm);
        let extrinsics = opaque_bundle
            .extrinsics
            .iter()
            .map(|ext| ext.encode())
            .collect();
        opaque_bundle.sealed_header.header.bundle_extrinsics_root =
            BlakeTwo256::ordered_trie_root(extrinsics, StateVersion::V1);
        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        (opaque_bundle.extrinsics_root(), bundle_to_tx(opaque_bundle))
    };
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();

    // produce another bundle that marks the invalid xdm as valid.
    let (slot, mut opaque_bundle) = ferdie.produce_slot_and_wait_for_bundle_submission().await;
    let (bad_receipt_hash, bad_submit_bundle_tx) = {
        let bad_receipt = &mut opaque_bundle.sealed_header.header.receipt;
        assert!(bad_receipt.inboxed_bundles[0].is_invalid());
        bad_receipt.inboxed_bundles =
            vec![InboxedBundle::valid(H256::random(), bundle_extrinsic_root)];
        opaque_bundle.sealed_header.signature = Sr25519Keyring::Alice
            .pair()
            .sign(opaque_bundle.sealed_header.pre_hash().as_ref())
            .into();
        (
            opaque_bundle.receipt().hash::<BlakeTwo256>(),
            bundle_to_tx(opaque_bundle),
        )
    };

    // Wait for the fraud proof that targets the bad ER
    let wait_for_fraud_proof_fut = ferdie.wait_for_fraud_proof(move |fp| {
        if let FraudProofVariant::InvalidBundles(proof) = &fp.proof {
            if let InvalidBundleType::InvalidXDM(extrinsic_index) = proof.invalid_bundle_type {
                assert!(proof.is_true_invalid_fraud_proof);
                assert_eq!(extrinsic_index, 0);
                return true;
            }
        }
        false
    });

    // Produce a consensus block that contains `bad_submit_bundle_tx` and the bad receipt should
    // be added to the consensus chain block tree
    // Produce a block that contains `bad_submit_bundle_tx`
    produce_block_with!(
        ferdie.produce_block_with_slot_at(
            slot,
            ferdie.client.info().best_hash,
            Some(vec![bad_submit_bundle_tx])
        ),
        alice
    )
    .await
    .unwrap();
    assert!(ferdie.does_receipt_exist(bad_receipt_hash).unwrap());

    let timed_out = tokio::time::timeout(TIMEOUT, wait_for_fraud_proof_fut)
        .await
        .inspect_err(|_| error!("fraud proof was not created before the timeout"))
        .is_err();

    // Produce a consensus block that contains the fraud proof, the fraud proof wil be verified
    // and executed, and prune the bad receipt from the block tree
    ferdie.produce_blocks(1).await.unwrap();
    assert!(!ferdie.does_receipt_exist(bad_receipt_hash).unwrap());
    // We check for timeouts last, because they are the least useful test failure message.
    assert!(!timed_out);
}
