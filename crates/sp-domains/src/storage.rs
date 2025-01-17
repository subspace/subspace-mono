#[cfg(not(feature = "std"))]
extern crate alloc;

use crate::{
    evm_chain_id_storage_key, evm_contract_creation_allowed_by_storage_key,
    self_domain_id_storage_key, DomainId, PermissionedActionAllowedBy,
};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use domain_runtime_primitives::{EVMChainId, EthereumAccountId};
use hash_db::Hasher;
use parity_scale_codec::{Codec, Decode, Encode};
use scale_info::TypeInfo;
use sp_core::storage::{well_known_keys, ChildInfo};
#[cfg(feature = "std")]
use sp_core::storage::{Storage, StorageChild};
use sp_runtime::StateVersion;
use sp_state_machine::{Backend, TrieBackend, TrieBackendBuilder};
use sp_std::collections::btree_map::BTreeMap;
use sp_trie::{empty_trie_root, LayoutV0, MemoryDB};

/// Create a new empty instance of in-memory backend.
///
/// NOTE: this function is port from `sp_state_machine::in_memory_backend::new_in_mem` which is
/// only export for `std` but we need to use it in `no_std`
fn new_in_mem<H>() -> TrieBackend<MemoryDB<H>, H>
where
    H: Hasher,
    H::Out: Codec + Ord,
{
    let db = MemoryDB::default();
    // V1 is same as V0 for an empty trie.
    TrieBackendBuilder::new(db, empty_trie_root::<LayoutV0<H>>()).build()
}

// NOTE: this is port from `sp_core::storage::StorageKey` with `TypeInfo` supported
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Encode, Decode, TypeInfo)]
pub struct StorageKey(pub Vec<u8>);

// NOTE: this is port from `sp_core::storage::StorageData` with `TypeInfo` supported
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Encode, Decode, TypeInfo)]
pub struct StorageData(pub Vec<u8>);

type GenesisStorage = BTreeMap<StorageKey, StorageData>;

/// Raw storage content for genesis block
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Encode, Decode, TypeInfo)]
pub struct RawGenesis {
    top: GenesisStorage,
    children_default: BTreeMap<StorageKey, GenesisStorage>,
}

impl RawGenesis {
    pub fn set_domain_id(&mut self, domain_id: DomainId) {
        let _ = self.top.insert(
            self_domain_id_storage_key(),
            StorageData(domain_id.encode()),
        );
    }

    pub fn set_evm_chain_id(&mut self, chain_id: EVMChainId) {
        let _ = self
            .top
            .insert(evm_chain_id_storage_key(), StorageData(chain_id.encode()));
    }

    pub fn set_evm_contract_creation_allowed_by(
        &mut self,
        contract_creation_allowed_by: &PermissionedActionAllowedBy<EthereumAccountId>,
    ) {
        let _ = self.top.insert(
            evm_contract_creation_allowed_by_storage_key(),
            StorageData(contract_creation_allowed_by.encode()),
        );
    }

    pub fn set_top_storages(&mut self, storages: Vec<(StorageKey, StorageData)>) {
        for (k, v) in storages {
            let _ = self.top.insert(k, v);
        }
    }

    fn set_runtime_code(&mut self, code: Vec<u8>) {
        let _ = self.top.insert(
            StorageKey(well_known_keys::CODE.to_vec()),
            StorageData(code),
        );
    }

    pub fn get_runtime_code(&self) -> Option<&[u8]> {
        self.top
            .get(&StorageKey(well_known_keys::CODE.to_vec()))
            .map(|sd| sd.0.as_ref())
    }

    pub fn take_runtime_code(&mut self) -> Option<Vec<u8>> {
        self.top
            .remove(&StorageKey(well_known_keys::CODE.to_vec()))
            .map(|sd| sd.0)
    }

    pub fn state_root<H>(&self, state_version: StateVersion) -> H::Out
    where
        H: Hasher,
        H::Out: Codec + Ord,
    {
        let backend = new_in_mem::<H>();

        // NOTE: the `(k, v)` of `children_default` are iterated separately because the
        // `full_storage_root` required `&ChildInfo` as input but if we simply map `k`
        // to `&ChildInfo` it will fail due to temporary value can't live long enough.
        let child_infos: Vec<_> = self
            .children_default
            .keys()
            .map(|k| ChildInfo::new_default(k.0.as_slice()))
            .collect();
        let child_delta = child_infos.iter().zip(
            self.children_default
                .values()
                .map(|v| v.iter().map(|(k, v)| (&k.0[..], Some(&v.0[..])))),
        );

        let (root, _) = backend.full_storage_root(
            self.top.iter().map(|(k, v)| (&k.0[..], Some(&v.0[..]))),
            child_delta,
            state_version,
        );

        root
    }

    pub fn dummy(code: Vec<u8>) -> Self {
        let mut raw_genesis = Self::default();
        raw_genesis.set_runtime_code(code);
        raw_genesis
    }
}

#[cfg(feature = "std")]
impl RawGenesis {
    /// Construct `RawGenesis` from a given storage
    //
    /// NOTE: This function is part from `sc-chain-spec::GenesisSource::resolve`
    pub fn from_storage(storage: Storage) -> Self {
        let top = storage
            .top
            .into_iter()
            .map(|(k, v)| (StorageKey(k), StorageData(v)))
            .collect();

        let children_default = storage
            .children_default
            .into_iter()
            .map(|(k, child)| {
                (
                    StorageKey(k),
                    child
                        .data
                        .into_iter()
                        .map(|(k, v)| (StorageKey(k), StorageData(v)))
                        .collect(),
                )
            })
            .collect();

        RawGenesis {
            top,
            children_default,
        }
    }

    /// Convert `RawGenesis` to storage, the opposite of `from_storage`
    //
    /// NOTE: This function is part from `<sc-chain-spec::ChainSpec as BuildStorage>::assimilate_storage`
    pub fn into_storage(self) -> Storage {
        let RawGenesis {
            top: map,
            children_default: children_map,
        } = self;
        let mut storage = Storage::default();

        storage.top.extend(map.into_iter().map(|(k, v)| (k.0, v.0)));

        children_map.into_iter().for_each(|(k, v)| {
            let child_info = ChildInfo::new_default(k.0.as_slice());
            storage
                .children_default
                .entry(k.0)
                .or_insert_with(|| StorageChild {
                    data: Default::default(),
                    child_info,
                })
                .data
                .extend(v.into_iter().map(|(k, v)| (k.0, v.0)));
        });

        storage
    }
}
