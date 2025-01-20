//! Migration module for pallet-domains storage version v2 to v3.
//!
//! TODO: remove this module after it has been deployed to Taurus.

use crate::{Config, Pallet};
use frame_support::migrations::VersionedMigration;
use frame_support::traits::UncheckedOnRuntimeUpgrade;
use frame_support::weights::Weight;

pub type VersionCheckedMigrateDomainsV2ToV3<T> = VersionedMigration<
    2,
    3,
    VersionUncheckedMigrateV2ToV3<T>,
    Pallet<T>,
    <T as frame_system::Config>::DbWeight,
>;

pub struct VersionUncheckedMigrateV2ToV3<T>(sp_std::marker::PhantomData<T>);
impl<T: Config> UncheckedOnRuntimeUpgrade for VersionUncheckedMigrateV2ToV3<T> {
    fn on_runtime_upgrade() -> Weight {
        domain_registry_structure_migration::migrate_domain_registry_structure::<T>()
    }
}

mod domain_registry_structure_migration {
    use crate::domain_registry::{DomainConfig as DomainConfigV3, DomainObject as DomainObjectV3};
    use crate::pallet::DomainRegistry as DomainRegistryV3;
    use crate::runtime_registry::DomainRuntimeInfo as DomainRuntimeInfoV3;
    use crate::{BalanceOf, BlockNumberFor, Config, Pallet, ReceiptHashFor};
    use codec::{Decode, Encode};
    use domain_runtime_primitives::{EVMChainId, MultiAccountId};
    use frame_support::pallet_prelude::{OptionQuery, TypeInfo, Weight};
    use frame_support::{storage_alias, Identity};
    use scale_info::prelude::string::String;
    use sp_core::Get;
    use sp_domains::{
        AutoIdDomainRuntimeConfig,
        DomainId,
        // changed in V3, but we use Into to convert to it
        // DomainRuntimeConfig as DomainRuntimeConfigV3,
        EvmDomainRuntimeConfig,
        OperatorAllowList,
        RuntimeId,
    };
    use sp_runtime::Vec;

    #[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq)]
    pub struct DomainConfigV2<AccountId: Ord, Balance> {
        /// A user defined name for this domain, should be a human-readable UTF-8 encoded string.
        pub domain_name: String,
        /// A pointer to the `RuntimeRegistry` entry for this domain.
        pub runtime_id: RuntimeId,
        /// The max bundle size for this domain, may not exceed the system-wide `MaxDomainBlockSize` limit.
        pub max_bundle_size: u32,
        /// The max bundle weight for this domain, may not exceed the system-wide `MaxDomainBlockWeight` limit.
        pub max_bundle_weight: Weight,
        /// The probability of successful bundle in a slot (active slots coefficient). This defines the
        /// expected bundle production rate, must be `> 0` and `≤ 1`.
        pub bundle_slot_probability: (u64, u64),
        /// Allowed operators to operate for this domain.
        pub operator_allow_list: OperatorAllowList<AccountId>,
        // Initial balances for Domain.
        pub initial_balances: Vec<(MultiAccountId, Balance)>,
    }

    /// Domain runtime specific information to create domain raw genesis.
    #[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq, Copy)]
    #[allow(clippy::upper_case_acronyms)]
    pub enum DomainRuntimeInfoV2 {
        EVM { chain_id: EVMChainId },
        AutoId,
    }

    #[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq)]
    pub struct DomainObjectV2<Number, ReceiptHash, AccountId: Ord, Balance> {
        /// The address of the domain creator, used to validate updating the domain config.
        pub owner_account_id: AccountId,
        /// The consensus chain block number when the domain first instantiated.
        pub created_at: Number,
        /// The hash of the genesis execution receipt for this domain.
        pub genesis_receipt_hash: ReceiptHash,
        /// The domain config.
        pub domain_config: DomainConfigV2<AccountId, Balance>,
        /// Domain runtime specific information.
        pub domain_runtime_info: DomainRuntimeInfoV2,
        /// The amount of balance hold on the domain owner account
        pub domain_instantiation_deposit: Balance,
    }

    #[storage_alias]
    pub(super) type DomainRegistry<T: Config> = StorageMap<
        Pallet<T>,
        Identity,
        DomainId,
        DomainObjectV2<
            BlockNumberFor<T>,
            ReceiptHashFor<T>,
            <T as frame_system::Config>::AccountId,
            BalanceOf<T>,
        >,
        OptionQuery,
    >;

    pub(super) fn migrate_domain_registry_structure<T: Config>() -> Weight {
        let mut domain_count = 0;

        DomainRegistryV3::<T>::translate_values::<
            DomainObjectV2<BlockNumberFor<T>, ReceiptHashFor<T>, T::AccountId, BalanceOf<T>>,
            _,
        >(|domain_object_v2| {
            domain_count += 1;

            let domain_runtime_info = match domain_object_v2.domain_runtime_info {
                DomainRuntimeInfoV2::EVM { chain_id } => {
                    DomainRuntimeInfoV3::Evm {
                        chain_id,
                        // Added in V3
                        domain_runtime_config: EvmDomainRuntimeConfig::default(),
                    }
                }
                DomainRuntimeInfoV2::AutoId => {
                    DomainRuntimeInfoV3::AutoId {
                        // Added in V3
                        domain_runtime_config: AutoIdDomainRuntimeConfig::default(),
                    }
                }
            };

            Some(DomainObjectV3 {
                owner_account_id: domain_object_v2.owner_account_id,
                created_at: domain_object_v2.created_at,
                genesis_receipt_hash: domain_object_v2.genesis_receipt_hash,
                domain_config: DomainConfigV3 {
                    domain_name: domain_object_v2.domain_config.domain_name,
                    runtime_id: domain_object_v2.domain_config.runtime_id,
                    max_bundle_size: domain_object_v2.domain_config.max_bundle_size,
                    max_bundle_weight: domain_object_v2.domain_config.max_bundle_weight,
                    bundle_slot_probability: domain_object_v2.domain_config.bundle_slot_probability,
                    operator_allow_list: domain_object_v2.domain_config.operator_allow_list,
                    initial_balances: domain_object_v2.domain_config.initial_balances,
                },
                // Modified in V3
                domain_runtime_info,
                domain_instantiation_deposit: domain_object_v2.domain_instantiation_deposit,
            })
        });

        // 1 read and 1 write per domain
        T::DbWeight::get().reads_writes(domain_count, domain_count)
    }
}

#[cfg(test)]
mod tests {
    use super::domain_registry_structure_migration::{
        migrate_domain_registry_structure, DomainConfigV2, DomainObjectV2, DomainRegistry,
        DomainRuntimeInfoV2,
    };
    use crate::domain_registry::{DomainConfig as DomainConfigV3, DomainObject as DomainObjectV3};
    use crate::pallet::DomainRegistry as DomainRegistryV3;
    use crate::runtime_registry::DomainRuntimeInfo as DomainRuntimeInfoV3;
    use crate::tests::{new_test_ext, Test};
    use sp_domains::{
        AutoIdDomainRuntimeConfig, DomainId, EvmDomainRuntimeConfig, OperatorAllowList,
        PermissionedActionAllowedBy,
    };

    #[test]
    fn test_domain_registry_structure_migration_evm() {
        let mut ext = new_test_ext();
        let domain_id: DomainId = 0.into();
        let chain_id = 8u32.into();
        let domain = DomainObjectV2 {
            owner_account_id: 1u32.into(),
            created_at: 2u32.into(),
            genesis_receipt_hash: Default::default(),
            domain_config: DomainConfigV2 {
                domain_name: "test-evm-migrate".to_string(),
                runtime_id: 3u32,
                max_bundle_size: 4,
                max_bundle_weight: 5.into(),
                bundle_slot_probability: (6, 7),
                operator_allow_list: OperatorAllowList::Anyone,
                initial_balances: vec![],
            },
            domain_runtime_info: DomainRuntimeInfoV2::EVM { chain_id },
            domain_instantiation_deposit: 9u32.into(),
        };

        ext.execute_with(|| DomainRegistry::<Test>::set(domain_id, Some(domain.clone())));

        ext.commit_all().unwrap();

        ext.execute_with(|| {
            let weights = migrate_domain_registry_structure::<Test>();
            assert_eq!(
                weights,
                <Test as frame_system::Config>::DbWeight::get().reads_writes(1, 1),
            );
            assert_eq!(
                DomainRegistryV3::<Test>::get(domain_id),
                Some(DomainObjectV3 {
                    owner_account_id: domain.owner_account_id,
                    created_at: domain.created_at,
                    genesis_receipt_hash: domain.genesis_receipt_hash,
                    domain_config: DomainConfigV3 {
                        domain_name: domain.domain_config.domain_name,
                        runtime_id: domain.domain_config.runtime_id,
                        max_bundle_size: domain.domain_config.max_bundle_size,
                        max_bundle_weight: domain.domain_config.max_bundle_weight,
                        bundle_slot_probability: domain.domain_config.bundle_slot_probability,
                        operator_allow_list: domain.domain_config.operator_allow_list,
                        initial_balances: domain.domain_config.initial_balances,
                    },
                    domain_runtime_info: DomainRuntimeInfoV3::Evm {
                        chain_id,
                        domain_runtime_config: EvmDomainRuntimeConfig {
                            initial_contract_creation_allow_list:
                                PermissionedActionAllowedBy::Anyone
                        },
                    },
                    domain_instantiation_deposit: domain.domain_instantiation_deposit,
                })
            );
        });
    }

    #[test]
    fn test_domain_registry_structure_migration_auto_id() {
        let mut ext = new_test_ext();
        let domain_id: DomainId = 10.into();
        let domain = DomainObjectV2 {
            owner_account_id: 11u32.into(),
            created_at: 12u32.into(),
            genesis_receipt_hash: Default::default(),
            domain_config: DomainConfigV2 {
                domain_name: "test-auto-id-migrate".to_string(),
                runtime_id: 13u32,
                max_bundle_size: 14,
                max_bundle_weight: 15.into(),
                bundle_slot_probability: (16, 17),
                operator_allow_list: OperatorAllowList::Anyone,
                initial_balances: vec![],
            },
            domain_runtime_info: DomainRuntimeInfoV2::AutoId,
            domain_instantiation_deposit: 19u32.into(),
        };

        ext.execute_with(|| DomainRegistry::<Test>::set(domain_id, Some(domain.clone())));

        ext.commit_all().unwrap();

        ext.execute_with(|| {
            let weights = migrate_domain_registry_structure::<Test>();
            assert_eq!(
                weights,
                <Test as frame_system::Config>::DbWeight::get().reads_writes(1, 1),
            );
            assert_eq!(
                DomainRegistryV3::<Test>::get(domain_id),
                Some(DomainObjectV3 {
                    owner_account_id: domain.owner_account_id,
                    created_at: domain.created_at,
                    genesis_receipt_hash: domain.genesis_receipt_hash,
                    domain_config: DomainConfigV3 {
                        domain_name: domain.domain_config.domain_name,
                        runtime_id: domain.domain_config.runtime_id,
                        max_bundle_size: domain.domain_config.max_bundle_size,
                        max_bundle_weight: domain.domain_config.max_bundle_weight,
                        bundle_slot_probability: domain.domain_config.bundle_slot_probability,
                        operator_allow_list: domain.domain_config.operator_allow_list,
                        initial_balances: domain.domain_config.initial_balances,
                    },
                    domain_runtime_info: DomainRuntimeInfoV3::AutoId {
                        domain_runtime_config: AutoIdDomainRuntimeConfig {}
                    },
                    domain_instantiation_deposit: domain.domain_instantiation_deposit,
                })
            );
        });
    }
}
