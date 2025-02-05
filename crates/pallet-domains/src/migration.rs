//! Migration module for pallet-domains
use crate::{Config, Pallet};
use frame_support::migrations::VersionedMigration;
use frame_support::traits::UncheckedOnRuntimeUpgrade;
use frame_support::weights::Weight;

pub type VersionCheckedMigrateDomainsV1ToV2<T> = VersionedMigration<
    1,
    2,
    VersionUncheckedMigrateV1ToV2<T>,
    Pallet<T>,
    <T as frame_system::Config>::DbWeight,
>;

pub struct VersionUncheckedMigrateV1ToV2<T>(sp_std::marker::PhantomData<T>);
impl<T: Config> UncheckedOnRuntimeUpgrade for VersionUncheckedMigrateV1ToV2<T> {
    fn on_runtime_upgrade() -> Weight {
        operator_structure_migration::migrate_operator_structure::<T>()
    }
}

mod operator_structure_migration {
    use crate::pallet::Operators as OperatorsV2;
    use crate::staking::{Operator as OperatorV2, OperatorStatus};
    use crate::{BalanceOf, Config, DomainBlockNumberFor, Pallet};
    use codec::{Decode, Encode};
    use frame_support::pallet_prelude::{OptionQuery, TypeInfo, Weight};
    use frame_support::{storage_alias, Identity};
    use sp_core::Get;
    use sp_domains::{DomainId, OperatorId, OperatorPublicKey};
    use sp_runtime::Percent;

    #[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq)]
    pub struct Operator<Balance, Share, DomainBlockNumber> {
        pub signing_key: OperatorPublicKey,
        pub current_domain_id: DomainId,
        pub next_domain_id: DomainId,
        pub minimum_nominator_stake: Balance,
        pub nomination_tax: Percent,
        /// Total active stake of combined nominators under this operator.
        pub current_total_stake: Balance,
        /// Total rewards this operator received this current epoch.
        pub current_epoch_rewards: Balance,
        /// Total shares of all the nominators under this operator.
        pub current_total_shares: Share,
        /// The status of the operator, it may be stale due to the `OperatorStatus::PendingSlash` is
        /// not assigned to this field directly, thus MUST use the `status()` method to query the status
        /// instead.
        pub(super) partial_status: OperatorStatus<DomainBlockNumber>,
        /// Total deposits during the previous epoch
        pub deposits_in_epoch: Balance,
        /// Total withdrew shares during the previous epoch
        pub withdrawals_in_epoch: Share,
        /// Total balance deposited to the bundle storage fund
        pub total_storage_fee_deposit: Balance,
    }

    #[storage_alias]
    pub type Operators<T: Config> = StorageMap<
        Pallet<T>,
        Identity,
        OperatorId,
        Operator<BalanceOf<T>, <T as Config>::Share, DomainBlockNumberFor<T>>,
        OptionQuery,
    >;

    pub(super) fn migrate_operator_structure<T: Config>() -> Weight {
        // On Taurus, the operator 0-8 are registered before the runtime upgrade that brings the new
        // structure, for operator (if any) registered after that runtime upgrade it should be in new
        // structure already, thus the migration should only handle operator 0-8
        let affected_operator = 8;
        let mut operator_count = 0;
        for operator_id in 0..=affected_operator {
            if let Some(operator) = Operators::<T>::take(operator_id) {
                OperatorsV2::<T>::set(
                    operator_id,
                    Some(OperatorV2 {
                        signing_key: operator.signing_key,
                        current_domain_id: operator.current_domain_id,
                        next_domain_id: operator.next_domain_id,
                        minimum_nominator_stake: operator.minimum_nominator_stake,
                        nomination_tax: operator.nomination_tax,
                        current_total_stake: operator.current_total_stake,
                        current_total_shares: operator.current_total_shares,
                        partial_status: operator.partial_status,
                        deposits_in_epoch: operator.deposits_in_epoch,
                        withdrawals_in_epoch: operator.withdrawals_in_epoch,
                        total_storage_fee_deposit: operator.total_storage_fee_deposit,
                    }),
                );
                operator_count += 1;
            }
        }

        // 1 read and 1 write per old operator
        // 1 write per new operator
        T::DbWeight::get().reads_writes(operator_count, operator_count * 2)
    }
}

#[cfg(test)]
mod tests {
    use super::operator_structure_migration::{migrate_operator_structure, Operator, Operators};
    use crate::pallet::Operators as OperatorsV2;
    use crate::staking::{Operator as OperatorV2, OperatorStatus};
    use crate::tests::{new_test_ext, Test};
    use crate::Config;
    use sp_core::crypto::Ss58Codec;
    use sp_domains::OperatorPublicKey;

    #[test]
    fn test_operator_structure_migration() {
        let mut ext = new_test_ext();
        let operator_id = 0;
        let operator = Operator {
            signing_key: OperatorPublicKey::from_ss58check(
                "5Gv1Uopoqo1k7125oDtFSCmxH4DzuCiBU7HBKu2bF1GZFsEb",
            )
            .unwrap(),
            current_domain_id: 0u32.into(),
            next_domain_id: 0u32.into(),
            minimum_nominator_stake: <Test as Config>::MinNominatorStake::get(),
            nomination_tax: Default::default(),
            current_total_stake: 1u32.into(),
            current_epoch_rewards: 2u32.into(),
            current_total_shares: 3u32.into(),
            partial_status: OperatorStatus::Registered,
            deposits_in_epoch: 4u32.into(),
            withdrawals_in_epoch: 5u32.into(),
            total_storage_fee_deposit: 6u32.into(),
        };

        ext.execute_with(|| Operators::<Test>::set(operator_id, Some(operator.clone())));

        ext.commit_all().unwrap();

        ext.execute_with(|| {
            let weights = migrate_operator_structure::<Test>();
            assert_eq!(
                weights,
                <Test as frame_system::Config>::DbWeight::get().reads_writes(1, 2),
            );
            assert_eq!(
                OperatorsV2::<Test>::get(operator_id),
                Some(OperatorV2 {
                    signing_key: operator.signing_key,
                    current_domain_id: operator.current_domain_id,
                    next_domain_id: operator.next_domain_id,
                    minimum_nominator_stake: operator.minimum_nominator_stake,
                    nomination_tax: operator.nomination_tax,
                    current_total_stake: operator.current_total_stake,
                    current_total_shares: operator.current_total_shares,
                    partial_status: operator.partial_status,
                    deposits_in_epoch: operator.deposits_in_epoch,
                    withdrawals_in_epoch: operator.withdrawals_in_epoch,
                    total_storage_fee_deposit: operator.total_storage_fee_deposit,
                })
            );
        });
    }
}
