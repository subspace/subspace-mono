// This file is part of Substrate.

// Copyright (C) 2017-2021 Parity Technologies (UK) Ltd.
// SPDX-License-Identifier: GPL-3.0-or-later WITH Classpath-exception-2.0

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Substrate block builder
//!
//! This crate provides the [`BlockBuilder`] utility and the corresponding runtime api
//! [`BlockBuilder`](sp_block_builder::BlockBuilder).
//!
//! The block builder utility is used in the node as an abstraction over the runtime api to
//! initialize a block, to push extrinsics and to finalize a block.

#![warn(missing_docs)]
#![feature(let_chains)]

mod custom_api;

use crate::custom_api::{TrieBackendApi, TrieDeltaBackendFor};
use codec::Encode;
pub use custom_api::{create_delta_backend, CollectedStorageChanges, DeltaBackend};
use sc_client_api::{backend, ExecutorProvider};
use sp_api::{ProvideRuntimeApi, TransactionOutcome};
pub use sp_block_builder::BlockBuilder as BlockBuilderApi;
use sp_blockchain::{ApplyExtrinsicFailed, Error};
use sp_core::traits::CodeExecutor;
use sp_runtime::traits::{Block as BlockT, Hash, HashingFor, Header as HeaderT, NumberFor, One};
use sp_runtime::Digest;
use sp_state_machine::OverlayedChanges;
use std::collections::VecDeque;
use std::sync::Arc;

/// A block that was build by [`BlockBuilder`] plus some additional data.
///
/// This additional data includes the `storage_changes`, these changes can be applied to the
/// backend to get the state of the block.
pub struct BuiltBlock<Block: BlockT> {
    /// The actual block that was build.
    pub block: Block,
    /// The changes that need to be applied to the backend to get the state of the build block.
    pub storage_changes: CollectedStorageChanges<HashingFor<Block>>,
}

impl<Block: BlockT> BuiltBlock<Block> {
    /// Convert into the inner values.
    pub fn into_inner(self) -> (Block, CollectedStorageChanges<HashingFor<Block>>) {
        (self.block, self.storage_changes)
    }
}

/// Utility for building new (valid) blocks from a stream of extrinsics.
pub struct BlockBuilder<Client, Block: BlockT, Backend: backend::Backend<Block>, Exec> {
    extrinsics: VecDeque<Block::Extrinsic>,
    api: TrieBackendApi<Client, Block, Backend, Exec>,
}

impl<Client, Block, Backend, Exec> BlockBuilder<Client, Block, Backend, Exec>
where
    Block: BlockT,
    Client: ProvideRuntimeApi<Block> + ExecutorProvider<Block>,
    Client::Api: BlockBuilderApi<Block>,
    Backend: backend::Backend<Block>,
    Exec: CodeExecutor,
{
    /// Create a new instance of builder based on the given `parent_hash` and `parent_number`.
    ///
    /// While proof recording is enabled, all accessed trie nodes are saved.
    /// These recorded trie nodes can be used by a third party to prove the
    /// output of this block builder without having access to the full storage.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: Arc<Client>,
        parent_hash: Block::Hash,
        parent_number: NumberFor<Block>,
        inherent_digests: Digest,
        backend: Arc<Backend>,
        exec: Arc<Exec>,
        mut extrinsics: VecDeque<Block::Extrinsic>,
        maybe_inherent_data: Option<sp_inherents::InherentData>,
    ) -> Result<Self, Error> {
        let mut api = TrieBackendApi::new(parent_hash, parent_number, client, backend, exec)?;
        let header = <Block::Header as HeaderT>::new(
            parent_number + One::one(),
            Default::default(),
            Default::default(),
            parent_hash,
            inherent_digests,
        );

        api.execute_in_transaction(
            |api: &TrieBackendApi<Client, Block, Backend, Exec>,
             backend: &TrieDeltaBackendFor<Backend::State, Block>,
             overlayed_changes: &mut OverlayedChanges<HashingFor<Block>>| {
                match api.initialize_block(header, backend, overlayed_changes) {
                    Ok(_) => TransactionOutcome::Commit(Ok(())),
                    Err(e) => TransactionOutcome::Rollback(Err(e)),
                }
            },
        )?;

        if let Some(inherent_data) = maybe_inherent_data {
            let inherent_extrinsics = Self::create_inherents(&mut api, inherent_data)?;

            // reverse and push the inherents so that order is maintained
            for inherent_extrinsic in inherent_extrinsics.into_iter().rev() {
                extrinsics.push_front(inherent_extrinsic)
            }
        }

        Ok(Self { extrinsics, api })
    }

    /// Execute the block's list of extrinsics.
    fn execute_extrinsics(&mut self) -> Result<(), Error> {
        for (index, xt) in self.extrinsics.iter().enumerate() {
            let res = self.api.execute_in_transaction(
                |api: &TrieBackendApi<Client, Block, Backend, Exec>,
                 backend: &TrieDeltaBackendFor<Backend::State, Block>,
                 overlayed_changes: &mut OverlayedChanges<HashingFor<Block>>| {
                    match api.apply_extrinsic(xt.clone(), backend, overlayed_changes) {
                        Ok(Ok(_)) => TransactionOutcome::Commit(Ok(())),
                        Ok(Err(tx_validity)) => TransactionOutcome::Rollback(Err(
                            ApplyExtrinsicFailed::Validity(tx_validity).into(),
                        )),
                        Err(e) => TransactionOutcome::Rollback(Err(e)),
                    }
                },
            );

            if let Err(e) = res {
                tracing::debug!("Apply extrinsic at index {index} failed: {e}");
            }
        }

        Ok(())
    }

    fn collect_storage_changes(&mut self) -> Option<CollectedStorageChanges<HashingFor<Block>>> {
        self.api.collect_storage_changes()
    }

    /// Returns the state before executing the extrinsic at given extrinsic index.
    pub fn prepare_storage_changes_before(
        &mut self,
        extrinsic_index: usize,
    ) -> Result<CollectedStorageChanges<HashingFor<Block>>, Error> {
        for (index, xt) in self.extrinsics.iter().enumerate() {
            if index == extrinsic_index {
                return self
                    .collect_storage_changes()
                    .ok_or(Error::Execution(Box::new("No execution is done")));
            }

            // TODO: rethink what to do if an error occurs when executing the transaction.
            self.api.execute_in_transaction(
                |api: &TrieBackendApi<Client, Block, Backend, Exec>,
                 backend: &TrieDeltaBackendFor<Backend::State, Block>,
                 overlayed_changes: &mut OverlayedChanges<HashingFor<Block>>| {
                    match api.apply_extrinsic(xt.clone(), backend, overlayed_changes) {
                        Ok(Ok(_)) => TransactionOutcome::Commit(Ok(())),
                        Ok(Err(tx_validity)) => TransactionOutcome::Rollback(Err(
                            ApplyExtrinsicFailed::Validity(tx_validity).into(),
                        )),
                        Err(e) => TransactionOutcome::Rollback(Err(e)),
                    }
                },
            )?;
        }

        Err(Error::Execution(Box::new(format!(
            "Invalid extrinsic index, got: {}, max: {}",
            extrinsic_index,
            self.extrinsics.len()
        ))))
    }

    /// Returns the state before finalizing the block.
    pub fn prepare_storage_changes_before_finalize_block(
        &mut self,
    ) -> Result<CollectedStorageChanges<HashingFor<Block>>, Error> {
        self.execute_extrinsics()?;
        self.collect_storage_changes()
            .ok_or(Error::Execution(Box::new("No execution is done")))
    }

    /// Consume the builder to build a valid `Block` containing all pushed extrinsics.
    ///
    /// Returns the build `Block`, the changes to the storage and an optional `StorageProof`
    /// supplied by `self.api`, combined as [`BuiltBlock`].
    /// The storage proof will be `Some(_)` when proof recording was enabled.
    pub fn build(mut self) -> Result<BuiltBlock<Block>, Error> {
        self.execute_extrinsics()?;
        let header = self.api.execute_in_transaction(
            |api: &TrieBackendApi<Client, Block, Backend, Exec>,
             backend: &TrieDeltaBackendFor<Backend::State, Block>,
             overlayed_changes: &mut OverlayedChanges<HashingFor<Block>>| {
                match api.finalize_block(backend, overlayed_changes) {
                    Ok(header) => TransactionOutcome::Commit(Ok(header)),
                    Err(e) => TransactionOutcome::Rollback(Err(e)),
                }
            },
        )?;

        debug_assert_eq!(
            header.extrinsics_root().clone(),
            HashingFor::<Block>::ordered_trie_root(
                self.extrinsics.iter().map(Encode::encode).collect(),
                sp_core::storage::StateVersion::V1
            ),
        );

        let storage_changes = self
            .collect_storage_changes()
            .expect("must always have the storage changes due to execution above");

        Ok(BuiltBlock {
            block: Block::new(header, self.extrinsics.into()),
            storage_changes,
        })
    }

    /// Create the inherents for the block.
    ///
    /// Returns the inherents created by the runtime or an error if something failed.
    pub(crate) fn create_inherents(
        api: &mut TrieBackendApi<Client, Block, Backend, Exec>,
        inherent_data: sp_inherents::InherentData,
    ) -> Result<VecDeque<Block::Extrinsic>, Error> {
        let exts = api.execute_in_transaction(
            |api: &TrieBackendApi<Client, Block, Backend, Exec>,
             backend: &TrieDeltaBackendFor<Backend::State, Block>,
             overlayed_changes: &mut OverlayedChanges<HashingFor<Block>>| {
                // `create_inherents` should not change any state, to ensure this we always rollback
                // the transaction.
                TransactionOutcome::Rollback(api.inherent_extrinsics(
                    inherent_data,
                    backend,
                    overlayed_changes,
                ))
            },
        )?;
        Ok(VecDeque::from(exts))
    }
}

// TODO: Unlock this test, it got broken in https://github.com/autonomys/subspace/pull/1548 and
//  doesn't run on Windows at all, also needs to not use substrate_test_runtime_client
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use sp_blockchain::HeaderBackend;
//     use sp_core::Blake2Hasher;
//     use sp_state_machine::Backend;
//     // TODO: Remove `substrate_test_runtime_client` dependency for faster build time
//     use std::collections::VecDeque;
//     use substrate_test_runtime_client::{DefaultTestClientBuilderExt, TestClientBuilderExt};
//
//     #[test]
//     #[ignore]
//     fn block_building_storage_proof_does_not_include_runtime_by_default() {
//         let (client, backend) =
//             substrate_test_runtime_client::TestClientBuilder::new().build_with_backend();
//
//         let block = BlockBuilder::new(
//             &client,
//             client.info().best_hash,
//             client.info().best_number,
//             RecordProof::Yes,
//             Default::default(),
//             &*backend,
//             VecDeque::new(),
//             Default::default(),
//         )
//         .unwrap()
//         .build()
//         .unwrap();
//
//         let proof = block.proof.expect("Proof is build on request");
//
//         let backend = sp_state_machine::create_proof_check_backend::<Blake2Hasher>(
//             block.storage_changes.transaction_storage_root,
//             proof,
//         )
//         .unwrap();
//
//         assert!(backend
//             .storage(sp_core::storage::well_known_keys::CODE)
//             .unwrap_err()
//             .contains("Database missing expected key"));
//     }
// }
