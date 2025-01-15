//! Custom API that is efficient to collect storage roots.

use codec::{Codec, Decode, Encode};
use hash_db::{HashDB, Hasher, Prefix};
use sc_client_api::{backend, ExecutorProvider, StateBackend};
use sp_core::offchain::OffchainOverlayedChange;
use sp_core::traits::{CallContext, CodeExecutor, FetchRuntimeCode};
use sp_inherents::InherentData;
use sp_runtime::traits::{Block as BlockT, HashingFor, NumberFor};
use sp_runtime::{ApplyExtrinsicResult, ExtrinsicInclusionMode, TransactionOutcome};
use sp_state_machine::backend::AsTrieBackend;
use sp_state_machine::{
    BackendTransaction, DBValue, IndexOperation, OverlayedChanges, StateMachine, StorageChanges,
    StorageKey, StorageValue, TrieBackend, TrieBackendBuilder, TrieBackendStorage,
};
use std::borrow::Cow;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

type TrieBackendStorageFor<State, Block> =
    <State as StateBackend<HashingFor<Block>>>::TrieBackendStorage;

pub(crate) type TrieDeltaBackendFor<'a, State, Block> = TrieBackend<
    DeltaBackend<'a, TrieBackendStorageFor<State, Block>, HashingFor<Block>>,
    HashingFor<Block>,
>;

struct MappedStorageChanges<H: Hasher> {
    /// All changes to the main storage.
    ///
    /// A value of `None` means that it was deleted.
    pub main_storage_changes: HashMap<StorageKey, Option<StorageValue>>,
    /// All changes to the child storages.
    pub child_storage_changes: HashMap<StorageKey, HashMap<StorageKey, Option<StorageValue>>>,
    /// Offchain state changes to write to the offchain database.
    pub offchain_storage_changes: HashMap<(Vec<u8>, Vec<u8>), OffchainOverlayedChange>,
    /// A transaction for the backend that contains all changes from
    /// [`main_storage_changes`](StorageChanges::main_storage_changes) and from
    /// [`child_storage_changes`](StorageChanges::child_storage_changes) but excluding
    /// [`offchain_storage_changes`](StorageChanges::offchain_storage_changes).
    pub transaction: BackendTransaction<H>,
    /// The storage root after applying the transaction.
    pub transaction_storage_root: H::Out,
    /// Changes to the transaction index,
    pub transaction_index_changes: Vec<IndexOperation>,
}

#[derive(Clone)]
struct RuntimeCode {
    code: Vec<u8>,
    heap_pages: Option<u64>,
    hash: Vec<u8>,
}

impl<H: Hasher> From<MappedStorageChanges<H>> for StorageChanges<H> {
    fn from(value: MappedStorageChanges<H>) -> Self {
        let MappedStorageChanges {
            main_storage_changes,
            child_storage_changes,
            offchain_storage_changes,
            transaction,
            transaction_storage_root,
            transaction_index_changes,
        } = value;
        StorageChanges {
            main_storage_changes: main_storage_changes.into_iter().collect(),
            child_storage_changes: child_storage_changes
                .into_iter()
                .map(|(k, v)| (k, v.into_iter().collect()))
                .collect(),
            offchain_storage_changes: offchain_storage_changes.into_iter().collect(),
            transaction,
            transaction_storage_root,
            transaction_index_changes,
        }
    }
}

impl<H: Hasher> From<StorageChanges<H>> for MappedStorageChanges<H> {
    fn from(value: StorageChanges<H>) -> Self {
        let StorageChanges {
            main_storage_changes,
            child_storage_changes,
            offchain_storage_changes,
            transaction,
            transaction_storage_root,
            transaction_index_changes,
        } = value;
        MappedStorageChanges {
            main_storage_changes: main_storage_changes.into_iter().collect(),
            child_storage_changes: child_storage_changes
                .into_iter()
                .map(|(k, v)| (k, v.into_iter().collect()))
                .collect(),
            offchain_storage_changes: offchain_storage_changes.into_iter().collect(),
            transaction,
            transaction_storage_root,
            transaction_index_changes,
        }
    }
}

impl<H: Hasher> MappedStorageChanges<H> {
    fn consolidate_storage_changes(&mut self, changes: StorageChanges<H>) -> H::Out {
        let StorageChanges {
            main_storage_changes,
            child_storage_changes,
            offchain_storage_changes,
            transaction,
            transaction_storage_root,
            mut transaction_index_changes,
        } = changes;
        self.main_storage_changes.extend(main_storage_changes);
        child_storage_changes
            .into_iter()
            .for_each(|(k, v)| self.child_storage_changes.entry(k).or_default().extend(v));
        self.offchain_storage_changes
            .extend(offchain_storage_changes);
        self.transaction.consolidate(transaction);
        self.transaction_index_changes
            .append(&mut transaction_index_changes);
        let previous_storage_root = self.transaction_storage_root;
        self.transaction_storage_root = transaction_storage_root;
        previous_storage_root
    }
}

/// Storage changes are the collected throughout the execution.
pub struct CollectedStorageChanges<H: Hasher> {
    /// Storage changes that are captured during the execution.
    /// Can be used for block import.
    pub storage_changes: StorageChanges<H>,
    /// Intermediate storage roots in between the execution.
    pub intermediate_roots: Vec<H::Out>,
}

/// Create a new trie backend with memory DB delta changes.
///
/// This can be used to verify any extrinsic-specific execution on the combined state of `backend`
/// and `delta`.
pub fn create_delta_backend<'a, S, H>(
    backend: &'a TrieBackend<S, H>,
    delta: &'a BackendTransaction<H>,
    post_delta_root: H::Out,
) -> TrieBackend<DeltaBackend<'a, S, H>, H>
where
    S: 'a + TrieBackendStorage<H>,
    H: 'a + Hasher,
    H::Out: Codec,
{
    create_delta_backend_with_maybe_delta(backend, Some(delta), post_delta_root)
}

fn create_delta_backend_with_maybe_delta<'a, S, H>(
    backend: &'a TrieBackend<S, H>,
    maybe_delta: Option<&'a BackendTransaction<H>>,
    post_delta_root: H::Out,
) -> TrieBackend<DeltaBackend<'a, S, H>, H>
where
    S: 'a + TrieBackendStorage<H>,
    H: 'a + Hasher,
    H::Out: Codec,
{
    let essence = backend.essence();
    let delta_backend = DeltaBackend {
        backend: essence.backend_storage(),
        delta: maybe_delta,
        _phantom: PhantomData::<H>,
    };
    TrieBackendBuilder::new(delta_backend, post_delta_root).build()
}

/// DeltaBackend provides the TrieBackend using main backend and some delta changes
/// that are not part of the main backend.
pub struct DeltaBackend<'a, S, H>
where
    S: 'a + TrieBackendStorage<H>,
    H: 'a + Hasher,
{
    backend: &'a S,
    delta: Option<&'a BackendTransaction<H>>,
    _phantom: PhantomData<H>,
}

impl<'a, S, H> TrieBackendStorage<H> for DeltaBackend<'a, S, H>
where
    S: 'a + TrieBackendStorage<H>,
    H: 'a + Hasher,
{
    fn get(&self, key: &H::Out, prefix: Prefix) -> Result<Option<DBValue>, String> {
        if let Some(db) = &self.delta
            && let Some(v) = HashDB::get(*db, key, prefix)
        {
            Ok(Some(v))
        } else {
            Ok(self.backend.get(key, prefix)?)
        }
    }
}

pub(crate) struct TrieBackendApi<Client, Block: BlockT, Backend: backend::Backend<Block>, Exec> {
    parent_hash: Block::Hash,
    parent_number: NumberFor<Block>,
    client: Arc<Client>,
    state: Backend::State,
    executor: Arc<Exec>,
    maybe_storage_changes: Option<MappedStorageChanges<HashingFor<Block>>>,
    intermediate_roots: Vec<Block::Hash>,
    runtime_code: RuntimeCode,
}

impl<Client, Block, Backend, Exec> FetchRuntimeCode for TrieBackendApi<Client, Block, Backend, Exec>
where
    Block: BlockT,
    Backend: backend::Backend<Block>,
{
    fn fetch_runtime_code(&self) -> Option<Cow<[u8]>> {
        Some(Cow::from(&self.runtime_code.code))
    }
}

impl<Client, Block, Backend, Exec> TrieBackendApi<Client, Block, Backend, Exec>
where
    Block: BlockT,
    Backend: backend::Backend<Block>,
    Client: ExecutorProvider<Block>,
    Exec: CodeExecutor,
{
    pub(crate) fn new(
        parent_hash: Block::Hash,
        parent_number: NumberFor<Block>,
        client: Arc<Client>,
        backend: Arc<Backend>,
        executor: Arc<Exec>,
    ) -> Result<Self, sp_blockchain::Error> {
        let state = backend.state_at(parent_hash)?;
        let trie_backend = state.as_trie_backend();
        let state_runtime_code = sp_state_machine::backend::BackendRuntimeCode::new(trie_backend);
        let sp_core::traits::RuntimeCode {
            code_fetcher,
            heap_pages,
            hash,
        } = state_runtime_code
            .runtime_code()
            .map_err(sp_blockchain::Error::RuntimeCode)?;
        let runtime_code = RuntimeCode {
            code: code_fetcher
                .fetch_runtime_code()
                .map(|c| c.to_vec())
                .ok_or(sp_blockchain::Error::RuntimeCode("missing runtime code"))?,
            heap_pages,
            hash,
        };
        Ok(Self {
            parent_hash,
            parent_number,
            client,
            state,
            executor,
            maybe_storage_changes: None,
            intermediate_roots: vec![],
            runtime_code,
        })
    }

    fn runtime_code(&self) -> sp_core::traits::RuntimeCode {
        sp_core::traits::RuntimeCode {
            code_fetcher: self,
            heap_pages: self.runtime_code.heap_pages,
            hash: self.runtime_code.hash.clone(),
        }
    }

    fn call_function<R: Decode>(
        &self,
        method: &str,
        call_data: Vec<u8>,
        trie_backend: &TrieDeltaBackendFor<Backend::State, Block>,
        overlayed_changes: &mut OverlayedChanges<HashingFor<Block>>,
    ) -> Result<R, sp_blockchain::Error> {
        let mut extensions = self
            .client
            .execution_extensions()
            .extensions(self.parent_hash, self.parent_number);

        let runtime_code = self.runtime_code();

        let result = StateMachine::<_, _, _>::new(
            trie_backend,
            overlayed_changes,
            &*self.executor,
            method,
            call_data.as_slice(),
            &mut extensions,
            &runtime_code,
            CallContext::Onchain,
        )
        .set_parent_hash(self.parent_hash)
        .execute()?;

        R::decode(&mut result.as_slice())
            .map_err(|err| sp_blockchain::Error::CallResultDecode("failed to decode Result", err))
    }

    fn consolidate_storage_changes(&mut self, changes: StorageChanges<HashingFor<Block>>) {
        let changes = if let Some(mut mapped_changes) = self.maybe_storage_changes.take() {
            let previous_root = mapped_changes.consolidate_storage_changes(changes);
            self.intermediate_roots.push(previous_root);
            mapped_changes
        } else {
            changes.into()
        };
        self.maybe_storage_changes = Some(changes)
    }

    pub(crate) fn initialize_block(
        &self,
        header: Block::Header,
        backend: &TrieDeltaBackendFor<Backend::State, Block>,
        overlayed_changes: &mut OverlayedChanges<HashingFor<Block>>,
    ) -> Result<ExtrinsicInclusionMode, sp_blockchain::Error> {
        let call_data = header.encode();
        self.call_function(
            "Core_initialize_block",
            call_data,
            backend,
            overlayed_changes,
        )
    }

    pub(crate) fn apply_extrinsic(
        &self,
        extrinsic: <Block as BlockT>::Extrinsic,
        backend: &TrieDeltaBackendFor<Backend::State, Block>,
        overlayed_changes: &mut OverlayedChanges<HashingFor<Block>>,
    ) -> Result<ApplyExtrinsicResult, sp_blockchain::Error> {
        let call_data = extrinsic.encode();
        self.call_function(
            "BlockBuilder_apply_extrinsic",
            call_data,
            backend,
            overlayed_changes,
        )
    }

    pub(crate) fn finalize_block(
        &self,
        backend: &TrieDeltaBackendFor<Backend::State, Block>,
        overlayed_changes: &mut OverlayedChanges<HashingFor<Block>>,
    ) -> Result<Block::Header, sp_blockchain::Error> {
        self.call_function(
            "BlockBuilder_finalize_block",
            vec![],
            backend,
            overlayed_changes,
        )
    }

    pub(crate) fn inherent_extrinsics(
        &self,
        inherent: InherentData,
        backend: &TrieDeltaBackendFor<Backend::State, Block>,
        overlayed_changes: &mut OverlayedChanges<HashingFor<Block>>,
    ) -> Result<Vec<<Block as BlockT>::Extrinsic>, sp_blockchain::Error> {
        let call_data = inherent.encode();
        self.call_function(
            "BlockBuilder_inherent_extrinsics",
            call_data,
            backend,
            overlayed_changes,
        )
    }

    /// Collect storage changes returns the storage changes and intermediate roots collected so far.
    /// The changes are reset after this call.
    /// Could return None if there were no execution done.
    pub(crate) fn collect_storage_changes(
        &mut self,
    ) -> Option<CollectedStorageChanges<HashingFor<Block>>> {
        let mut intermediate_roots = self.intermediate_roots.drain(..).collect::<Vec<_>>();
        // we did not add the last transaction storage root.
        // include that and then return
        let maybe_storage_changes = self.maybe_storage_changes.take();
        if let Some(storage_changes) = maybe_storage_changes {
            intermediate_roots.push(storage_changes.transaction_storage_root);
            Some(CollectedStorageChanges {
                storage_changes: storage_changes.into(),
                intermediate_roots,
            })
        } else {
            None
        }
    }

    pub(crate) fn execute_in_transaction<F, R>(
        &mut self,
        call: F,
    ) -> Result<R, sp_blockchain::Error>
    where
        F: FnOnce(
            &Self,
            &TrieDeltaBackendFor<Backend::State, Block>,
            &mut OverlayedChanges<HashingFor<Block>>,
        ) -> TransactionOutcome<Result<R, sp_blockchain::Error>>,
        R: Decode,
    {
        let trie_backend = self.state.as_trie_backend();
        let mut overlayed_changes = OverlayedChanges::default();
        let (state_root, delta) = if let Some(changes) = &self.maybe_storage_changes {
            (changes.transaction_storage_root, Some(&changes.transaction))
        } else {
            (*trie_backend.root(), None)
        };
        let trie_delta_backend =
            create_delta_backend_with_maybe_delta(trie_backend, delta, state_root);

        let outcome = call(self, &trie_delta_backend, &mut overlayed_changes);
        match outcome {
            TransactionOutcome::Commit(result) => {
                let state_version = sp_core::storage::StateVersion::V1;
                let storage_changes = overlayed_changes
                    .drain_storage_changes(&trie_delta_backend, state_version)
                    .map_err(sp_blockchain::Error::StorageChanges)?;
                self.consolidate_storage_changes(storage_changes);
                result
            }
            TransactionOutcome::Rollback(result) => result,
        }
    }
}
