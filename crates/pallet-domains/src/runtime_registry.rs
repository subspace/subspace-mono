//! Runtime registry for domains

#[cfg(not(feature = "std"))]
extern crate alloc;

use crate::pallet::{
    DomainRuntimeUpgrades, NextRuntimeId, RuntimeRegistry, ScheduledRuntimeUpgrades,
};
use crate::{BalanceOf, Config, Event};
#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use codec::{Decode, Encode};
use domain_runtime_primitives::{AccountId20, EVMChainId, MultiAccountId, TryConvertBack};
use frame_support::PalletError;
use frame_system::pallet_prelude::*;
use frame_system::AccountInfo;
use scale_info::TypeInfo;
use sp_core::crypto::AccountId32;
use sp_core::Hasher;
use sp_domains::storage::{RawGenesis, StorageData, StorageKey};
use sp_domains::{
    AutoIdDomainRuntimeConfig, DomainId, DomainsDigestItem, EvmDomainRuntimeConfig, RuntimeId,
    RuntimeObject, RuntimeType,
};
use sp_runtime::traits::{CheckedAdd, Get, Zero};
use sp_runtime::DigestItem;
use sp_std::vec;
use sp_version::RuntimeVersion;

/// Runtime specific errors
#[derive(TypeInfo, Encode, Decode, PalletError, Debug, PartialEq)]
pub enum Error {
    FailedToExtractRuntimeVersion,
    InvalidSpecName,
    SpecVersionNeedsToIncrease,
    MaxRuntimeId,
    MissingRuntimeObject,
    RuntimeUpgradeAlreadyScheduled,
    MaxScheduledBlockNumber,
    FailedToDecodeRawGenesis,
    RuntimeCodeNotFoundInRawGenesis,
    InvalidAccountIdType,
}

/// Domain runtime specific information to create domain raw genesis.
#[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq)]
pub enum DomainRuntimeInfo {
    Evm {
        /// The dynamic EVM chain id for this domain.
        chain_id: EVMChainId,
        /// The EVM-specific domain runtime config.
        domain_runtime_config: EvmDomainRuntimeConfig,
    },
    AutoId {
        /// The AutoId-specific domain runtime config.
        domain_runtime_config: AutoIdDomainRuntimeConfig,
    },
}

impl Default for DomainRuntimeInfo {
    fn default() -> Self {
        Self::Evm {
            chain_id: 0,
            domain_runtime_config: EvmDomainRuntimeConfig::default(),
        }
    }
}

#[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq)]
pub struct DomainRuntimeUpgradeEntry<Hash> {
    // The consensus block hash at which the upgrade happened
    pub at_hash: Hash,
    // The expected number of ER (from differnt domains) that derive from the consensus
    // block `at_hash`, the `reference_count` will decrease by one as one such ER is
    // confirmed and the whole entry will remove from the state after it become zero.
    pub reference_count: u32,
}

fn derive_initial_balances_storages<T: Config, AccountId: Encode>(
    total_issuance: BalanceOf<T>,
    balances: Vec<(AccountId, BalanceOf<T>)>,
) -> Vec<(StorageKey, StorageData)> {
    let total_issuance_key = sp_domains::domain_total_issuance_storage_key();
    let mut initial_storages = vec![(total_issuance_key, StorageData(total_issuance.encode()))];
    for (account_id, balance) in balances {
        let account_storage_key = sp_domains::domain_account_storage_key(account_id);
        let account_info = AccountInfo {
            nonce: domain_runtime_primitives::Nonce::zero(),
            consumers: 0,
            // providers are set to 1 for new accounts
            providers: 1,
            sufficients: 0,
            data: pallet_balances::AccountData {
                free: balance,
                ..Default::default()
            },
        };
        initial_storages.push((account_storage_key, StorageData(account_info.encode())))
    }

    initial_storages
}

// Return a complete raw genesis with runtime code and domain id set properly
pub fn into_complete_raw_genesis<T: Config>(
    runtime_obj: RuntimeObject<BlockNumberFor<T>, T::Hash>,
    domain_id: DomainId,
    domain_runtime_info: &DomainRuntimeInfo,
    total_issuance: BalanceOf<T>,
    initial_balances: Vec<(MultiAccountId, BalanceOf<T>)>,
) -> Result<RawGenesis, Error> {
    let RuntimeObject {
        mut raw_genesis, ..
    } = runtime_obj;
    raw_genesis.set_domain_id(domain_id);
    match domain_runtime_info {
        DomainRuntimeInfo::Evm {
            chain_id,
            domain_runtime_config,
        } => {
            raw_genesis.set_evm_chain_id(*chain_id);
            raw_genesis.set_evm_contract_creation_allowed_by(
                &domain_runtime_config.initial_contract_creation_allow_list,
            );

            let initial_balances = initial_balances.into_iter().try_fold(
                Vec::<(AccountId20, BalanceOf<T>)>::new(),
                |mut balances, (account_id, balance)| {
                    let account_id =
                        domain_runtime_primitives::AccountId20Converter::try_convert_back(
                            account_id,
                        )
                        .ok_or(Error::InvalidAccountIdType)?;

                    balances.push((account_id, balance));
                    Ok(balances)
                },
            )?;
            raw_genesis.set_top_storages(derive_initial_balances_storages::<T, _>(
                total_issuance,
                initial_balances,
            ));
        }
        DomainRuntimeInfo::AutoId {
            domain_runtime_config: AutoIdDomainRuntimeConfig {},
        } => {
            let initial_balances = initial_balances.into_iter().try_fold(
                Vec::<(AccountId32, BalanceOf<T>)>::new(),
                |mut balances, (account_id, balance)| {
                    let account_id =
                        domain_runtime_primitives::AccountIdConverter::try_convert_back(account_id)
                            .ok_or(Error::InvalidAccountIdType)?;

                    balances.push((account_id, balance));
                    Ok(balances)
                },
            )?;
            raw_genesis.set_top_storages(derive_initial_balances_storages::<T, _>(
                total_issuance,
                initial_balances,
            ));
        }
    }

    Ok(raw_genesis)
}

#[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq)]
pub struct ScheduledRuntimeUpgrade<Hash> {
    pub raw_genesis: RawGenesis,
    pub version: RuntimeVersion,
    pub hash: Hash,
}

/// Extracts the runtime version of the provided code.
pub(crate) fn runtime_version(code: &[u8]) -> Result<RuntimeVersion, Error> {
    sp_io::misc::runtime_version(code)
        .and_then(|v| RuntimeVersion::decode(&mut &v[..]).ok())
        .ok_or(Error::FailedToExtractRuntimeVersion)
}

/// Upgrades current runtime with new runtime.
// TODO: we can use upstream's `can_set_code` after some adjustments
pub(crate) fn can_upgrade_code(
    current_version: &RuntimeVersion,
    update_code: &[u8],
) -> Result<RuntimeVersion, Error> {
    let new_version = runtime_version(update_code)?;

    if new_version.spec_name != current_version.spec_name {
        return Err(Error::InvalidSpecName);
    }

    if new_version.spec_version <= current_version.spec_version {
        return Err(Error::SpecVersionNeedsToIncrease);
    }

    Ok(new_version)
}

/// Registers a new domain runtime..
pub(crate) fn do_register_runtime<T: Config>(
    runtime_name: String,
    runtime_type: RuntimeType,
    raw_genesis_storage: Vec<u8>,
    at: BlockNumberFor<T>,
) -> Result<RuntimeId, Error> {
    let raw_genesis: RawGenesis = Decode::decode(&mut raw_genesis_storage.as_slice())
        .map_err(|_| Error::FailedToDecodeRawGenesis)?;

    let code = raw_genesis
        .get_runtime_code()
        .ok_or(Error::RuntimeCodeNotFoundInRawGenesis)?;

    let version = runtime_version(code)?;
    let runtime_hash = T::Hashing::hash(code);
    let runtime_id = NextRuntimeId::<T>::get();

    RuntimeRegistry::<T>::insert(
        runtime_id,
        RuntimeObject {
            runtime_name,
            runtime_type,
            hash: runtime_hash,
            raw_genesis,
            version,
            created_at: at,
            updated_at: at,
            runtime_upgrades: 0u32,
            instance_count: 0,
        },
    );

    let next_runtime_id = runtime_id.checked_add(1).ok_or(Error::MaxRuntimeId)?;
    NextRuntimeId::<T>::set(next_runtime_id);

    Ok(runtime_id)
}

// TODO: Remove once `do_register_runtime` works at genesis.
/// Registers a new domain runtime at genesis.
pub(crate) fn register_runtime_at_genesis<T: Config>(
    runtime_name: String,
    runtime_type: RuntimeType,
    runtime_version: RuntimeVersion,
    raw_genesis_storage: Vec<u8>,
    at: BlockNumberFor<T>,
) -> Result<RuntimeId, Error> {
    let raw_genesis: RawGenesis = Decode::decode(&mut raw_genesis_storage.as_slice())
        .map_err(|_| Error::FailedToDecodeRawGenesis)?;

    let code = raw_genesis
        .get_runtime_code()
        .ok_or(Error::RuntimeCodeNotFoundInRawGenesis)?;

    let runtime_hash = T::Hashing::hash(code);
    let runtime_id = NextRuntimeId::<T>::get();

    RuntimeRegistry::<T>::insert(
        runtime_id,
        RuntimeObject {
            runtime_name,
            runtime_type,
            hash: runtime_hash,
            raw_genesis,
            version: runtime_version,
            created_at: at,
            updated_at: at,
            runtime_upgrades: 0u32,
            instance_count: 0,
        },
    );

    let next_runtime_id = runtime_id.checked_add(1).ok_or(Error::MaxRuntimeId)?;
    NextRuntimeId::<T>::set(next_runtime_id);

    Ok(runtime_id)
}

/// Schedules a runtime upgrade after `DomainRuntimeUpgradeDelay` from current block number.
pub(crate) fn do_schedule_runtime_upgrade<T: Config>(
    runtime_id: RuntimeId,
    raw_genesis_storage: Vec<u8>,
    current_block_number: BlockNumberFor<T>,
) -> Result<BlockNumberFor<T>, Error> {
    let runtime_obj = RuntimeRegistry::<T>::get(runtime_id).ok_or(Error::MissingRuntimeObject)?;

    let new_raw_genesis: RawGenesis = Decode::decode(&mut raw_genesis_storage.as_slice())
        .map_err(|_| Error::FailedToDecodeRawGenesis)?;

    let new_code = new_raw_genesis
        .get_runtime_code()
        .ok_or(Error::RuntimeCodeNotFoundInRawGenesis)?;

    let new_runtime_version = can_upgrade_code(&runtime_obj.version, new_code)?;
    let new_runtime_hash = T::Hashing::hash(new_code);
    let scheduled_upgrade = ScheduledRuntimeUpgrade {
        raw_genesis: new_raw_genesis,
        version: new_runtime_version,
        hash: new_runtime_hash,
    };
    let scheduled_at = current_block_number
        .checked_add(&T::DomainRuntimeUpgradeDelay::get())
        .ok_or(Error::MaxScheduledBlockNumber)?;

    ScheduledRuntimeUpgrades::<T>::insert(scheduled_at, runtime_id, scheduled_upgrade);

    Ok(scheduled_at)
}

pub(crate) fn do_upgrade_runtimes<T: Config>(at: BlockNumberFor<T>) {
    for (runtime_id, scheduled_update) in ScheduledRuntimeUpgrades::<T>::drain_prefix(at) {
        RuntimeRegistry::<T>::mutate(runtime_id, |maybe_runtime_object| {
            let runtime_obj = maybe_runtime_object
                .as_mut()
                .expect("Runtime object exists since an upgrade is scheduled after verification");

            runtime_obj.raw_genesis = scheduled_update.raw_genesis;
            runtime_obj.version = scheduled_update.version;
            runtime_obj.hash = scheduled_update.hash;
            runtime_obj.runtime_upgrades = runtime_obj.runtime_upgrades.saturating_add(1);
            runtime_obj.updated_at = at;
        });

        // Record the runtime upgrade
        DomainRuntimeUpgrades::<T>::mutate(|upgrades| upgrades.push(runtime_id));

        // deposit digest log for light clients
        frame_system::Pallet::<T>::deposit_log(DigestItem::domain_runtime_upgrade(runtime_id));

        // deposit event to signal runtime upgrade is complete
        frame_system::Pallet::<T>::deposit_event(<T as Config>::RuntimeEvent::from(
            Event::DomainRuntimeUpgraded { runtime_id },
        ));
    }
}

#[cfg(test)]
mod tests {
    use crate::pallet::{NextRuntimeId, RuntimeRegistry, ScheduledRuntimeUpgrades};
    use crate::runtime_registry::Error as RuntimeRegistryError;
    use crate::tests::{
        new_test_ext, DomainRuntimeUpgradeDelay, Domains, ReadRuntimeVersion, System, Test,
    };
    use crate::Error;
    use codec::Encode;
    use frame_support::assert_ok;
    use frame_support::dispatch::RawOrigin;
    use frame_support::traits::OnInitialize;
    use sp_domains::storage::RawGenesis;
    use sp_domains::{DomainsDigestItem, RuntimeId, RuntimeObject, RuntimeType};
    use sp_runtime::traits::BlockNumberProvider;
    use sp_runtime::{Digest, DispatchError};
    use sp_version::RuntimeVersion;

    #[test]
    fn create_domain_runtime() {
        let version = RuntimeVersion {
            spec_name: "test".into(),
            impl_name: Default::default(),
            authoring_version: 0,
            spec_version: 1,
            impl_version: 1,
            apis: Default::default(),
            transaction_version: 1,
            system_version: 0,
        };
        let read_runtime_version = ReadRuntimeVersion(version.encode());

        let mut ext = new_test_ext();
        ext.register_extension(sp_core::traits::ReadRuntimeVersionExt::new(
            read_runtime_version,
        ));
        ext.execute_with(|| {
            let raw_genesis_storage = RawGenesis::dummy(vec![1, 2, 3, 4]).encode();
            let res = crate::Pallet::<Test>::register_domain_runtime(
                RawOrigin::Root.into(),
                "evm".to_owned(),
                RuntimeType::Evm,
                raw_genesis_storage,
            );

            assert_ok!(res);
            let runtime_obj = RuntimeRegistry::<Test>::get(0).unwrap();
            assert_eq!(runtime_obj.version, version);
            assert_eq!(NextRuntimeId::<Test>::get(), 1)
        })
    }

    #[test]
    fn schedule_domain_runtime_upgrade() {
        let mut ext = new_test_ext();
        ext.execute_with(|| {
            RuntimeRegistry::<Test>::insert(
                0,
                RuntimeObject {
                    runtime_name: "evm".to_owned(),
                    runtime_type: Default::default(),
                    runtime_upgrades: 0,
                    hash: Default::default(),
                    raw_genesis: RawGenesis::dummy(vec![1, 2, 3, 4]),
                    version: RuntimeVersion {
                        spec_name: "test".into(),
                        spec_version: 1,
                        impl_version: 1,
                        transaction_version: 1,
                        ..Default::default()
                    },
                    created_at: Default::default(),
                    updated_at: Default::default(),
                    instance_count: 0,
                },
            );

            NextRuntimeId::<Test>::set(1);
        });

        let test_data = vec![
            (
                "test1",
                1,
                Err(Error::<Test>::RuntimeRegistry(
                    RuntimeRegistryError::InvalidSpecName,
                )),
            ),
            (
                "test",
                1,
                Err(Error::<Test>::RuntimeRegistry(
                    RuntimeRegistryError::SpecVersionNeedsToIncrease,
                )),
            ),
            ("test", 2, Ok(())),
        ];

        for (spec_name, spec_version, expected) in test_data.into_iter() {
            let version = RuntimeVersion {
                spec_name: spec_name.into(),
                spec_version,
                impl_version: 1,
                transaction_version: 1,
                ..Default::default()
            };
            let read_runtime_version = ReadRuntimeVersion(version.encode());
            ext.register_extension(sp_core::traits::ReadRuntimeVersionExt::new(
                read_runtime_version,
            ));

            ext.execute_with(|| {
                frame_system::Pallet::<Test>::set_block_number(100u64);
                let res = crate::Pallet::<Test>::upgrade_domain_runtime(
                    RawOrigin::Root.into(),
                    0,
                    RawGenesis::dummy(vec![6, 7, 8, 9]).encode(),
                );

                assert_eq!(res, expected.map_err(DispatchError::from))
            })
        }

        // verify upgrade
        ext.execute_with(|| {
            let runtime_obj = RuntimeRegistry::<Test>::get(0).unwrap();
            assert_eq!(
                runtime_obj.version,
                RuntimeVersion {
                    spec_name: "test".into(),
                    spec_version: 1,
                    impl_version: 1,
                    transaction_version: 1,
                    ..Default::default()
                }
            );
            assert_eq!(runtime_obj.runtime_upgrades, 0);
            assert_eq!(runtime_obj.raw_genesis, RawGenesis::dummy(vec![1, 2, 3, 4]),);

            let block_number = frame_system::Pallet::<Test>::current_block_number();
            let scheduled_block_number = block_number
                .checked_add(DomainRuntimeUpgradeDelay::get())
                .unwrap();
            let scheduled_upgrade =
                ScheduledRuntimeUpgrades::<Test>::get(scheduled_block_number, 0).unwrap();
            assert_eq!(
                scheduled_upgrade.version,
                RuntimeVersion {
                    spec_name: "test".into(),
                    spec_version: 2,
                    impl_version: 1,
                    transaction_version: 1,
                    ..Default::default()
                }
            )
        })
    }

    fn go_to_block(block: u64) {
        for i in System::block_number() + 1..=block {
            let parent_hash = if System::block_number() > 1 {
                let header = System::finalize();
                header.hash()
            } else {
                System::parent_hash()
            };

            System::reset_events();
            let digest = sp_runtime::testing::Digest { logs: vec![] };
            System::initialize(&i, &parent_hash, &digest);
            Domains::on_initialize(i);
        }
    }

    fn fetch_upgraded_runtime_from_digest(digest: Digest) -> Option<RuntimeId> {
        for log in digest.logs {
            match log.as_domain_runtime_upgrade() {
                None => continue,
                Some(runtime_id) => return Some(runtime_id),
            }
        }

        None
    }

    #[test]
    fn upgrade_scheduled_domain_runtime() {
        let mut ext = new_test_ext();
        let mut version = RuntimeVersion {
            spec_name: "test".into(),
            impl_name: Default::default(),
            authoring_version: 0,
            spec_version: 1,
            impl_version: 1,
            apis: Default::default(),
            transaction_version: 1,
            system_version: 0,
        };

        ext.execute_with(|| {
            RuntimeRegistry::<Test>::insert(
                0,
                RuntimeObject {
                    runtime_name: "evm".to_owned(),
                    runtime_type: Default::default(),
                    runtime_upgrades: 0,
                    hash: Default::default(),
                    raw_genesis: RawGenesis::dummy(vec![1, 2, 3, 4]),
                    version: version.clone(),
                    created_at: Default::default(),
                    updated_at: Default::default(),
                    instance_count: 0,
                },
            );

            NextRuntimeId::<Test>::set(1);
        });

        version.spec_version = 2;
        let read_runtime_version = ReadRuntimeVersion(version.encode());
        ext.register_extension(sp_core::traits::ReadRuntimeVersionExt::new(
            read_runtime_version,
        ));

        ext.execute_with(|| {
            let res = crate::Pallet::<Test>::upgrade_domain_runtime(
                RawOrigin::Root.into(),
                0,
                RawGenesis::dummy(vec![6, 7, 8, 9]).encode(),
            );
            assert_ok!(res);

            let current_block = frame_system::Pallet::<Test>::current_block_number();
            let scheduled_block_number = current_block
                .checked_add(DomainRuntimeUpgradeDelay::get())
                .unwrap();

            go_to_block(scheduled_block_number);
            assert_eq!(
                ScheduledRuntimeUpgrades::<Test>::get(scheduled_block_number, 0),
                None
            );

            let runtime_obj = RuntimeRegistry::<Test>::get(0).unwrap();
            assert_eq!(runtime_obj.version, version);

            let digest = System::digest();
            assert_eq!(Some(0), fetch_upgraded_runtime_from_digest(digest))
        });
    }
}
