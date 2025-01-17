//! Staking for domains

#[cfg(not(feature = "std"))]
extern crate alloc;

use crate::bundle_storage_fund::{self, deposit_reserve_for_storage_fund};
use crate::pallet::{
    Deposits, DomainRegistry, DomainStakingSummary, HeadDomainNumber, NextOperatorId,
    NominatorCount, OperatorIdOwner, Operators, PendingSlashes, PendingStakingOperationCount,
    Withdrawals,
};
use crate::staking_epoch::{mint_funds, mint_into_treasury};
use crate::{
    BalanceOf, Config, DepositOnHold, DomainBlockNumberFor, Event, HoldIdentifier, NominatorId,
    OperatorEpochSharePrice, Pallet, ReceiptHashFor, SlashedReason,
};
use codec::{Decode, Encode};
use frame_support::traits::fungible::{Inspect, MutateHold};
use frame_support::traits::tokens::{Fortitude, Precision, Preservation};
use frame_support::{ensure, PalletError};
use frame_system::pallet_prelude::BlockNumberFor;
use scale_info::TypeInfo;
use sp_core::{sr25519, Get};
use sp_domains::{DomainId, EpochIndex, OperatorId, OperatorPublicKey, OperatorRewardSource};
use sp_runtime::traits::{CheckedAdd, CheckedSub, Zero};
use sp_runtime::{Perbill, Percent, Perquintill, Saturating};
use sp_std::collections::btree_map::BTreeMap;
use sp_std::collections::btree_set::BTreeSet;
use sp_std::collections::vec_deque::VecDeque;
use sp_std::vec::IntoIter;

/// A nominators deposit.
#[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq, Default)]
pub(crate) struct Deposit<Share: Copy, Balance: Copy> {
    pub(crate) known: KnownDeposit<Share, Balance>,
    pub(crate) pending: Option<PendingDeposit<Balance>>,
}

/// A share price is parts per billion of shares/ssc.
/// Note: Shares must always be equal to or lower than ssc.
#[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq, Default)]
pub struct SharePrice(Perbill);

impl SharePrice {
    /// Creates a new instance of share price from shares and stake.
    pub(crate) fn new<T: Config>(shares: T::Share, stake: BalanceOf<T>) -> Self {
        SharePrice(if shares.is_zero() || stake.is_zero() {
            Perbill::one()
        } else {
            Perbill::from_rational(shares, stake.into())
        })
    }

    /// Converts stake to shares based on the share price
    pub(crate) fn stake_to_shares<T: Config>(&self, stake: BalanceOf<T>) -> T::Share {
        if self.0.is_one() {
            stake.into()
        } else {
            self.0.mul_floor(stake).into()
        }
    }

    /// Converts shares to stake based on the share price
    pub(crate) fn shares_to_stake<T: Config>(&self, shares: T::Share) -> BalanceOf<T> {
        if self.0.is_one() {
            shares.into()
        } else {
            self.0.saturating_reciprocal_mul_floor(shares.into())
        }
    }
}

/// Unique epoch identifier across all domains. A combination of Domain and its epoch.
#[derive(TypeInfo, Debug, Encode, Decode, Copy, Clone, PartialEq, Eq)]
pub struct DomainEpoch(DomainId, EpochIndex);

impl DomainEpoch {
    pub(crate) fn deconstruct(self) -> (DomainId, EpochIndex) {
        (self.0, self.1)
    }
}

impl From<(DomainId, EpochIndex)> for DomainEpoch {
    fn from((domain_id, epoch_idx): (DomainId, EpochIndex)) -> Self {
        Self(domain_id, epoch_idx)
    }
}

pub struct NewDeposit<Balance> {
    pub(crate) staking: Balance,
    pub(crate) storage_fee_deposit: Balance,
}

/// A nominator's shares against their deposits to given operator pool.
#[derive(TypeInfo, Debug, Encode, Decode, Copy, Clone, PartialEq, Eq, Default)]
pub(crate) struct KnownDeposit<Share: Copy, Balance: Copy> {
    pub(crate) shares: Share,
    pub(crate) storage_fee_deposit: Balance,
}

/// A nominators pending deposit in SSC that needs to be converted to shares once domain epoch is complete.
#[derive(TypeInfo, Debug, Encode, Decode, Copy, Clone, PartialEq, Eq)]
pub(crate) struct PendingDeposit<Balance: Copy> {
    pub(crate) effective_domain_epoch: DomainEpoch,
    pub(crate) amount: Balance,
    pub(crate) storage_fee_deposit: Balance,
}

impl<Balance: Copy + CheckedAdd> PendingDeposit<Balance> {
    fn total(&self) -> Result<Balance, Error> {
        self.amount
            .checked_add(&self.storage_fee_deposit)
            .ok_or(Error::BalanceOverflow)
    }
}

/// A nominator's withdrawal from a given operator pool.
#[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq, Default)]
pub(crate) struct Withdrawal<Balance, Share, DomainBlockNumber> {
    /// Total withdrawal amount requested by the nominator that are in unlocking state excluding withdrawal
    /// in shares and the storage fee
    pub(crate) total_withdrawal_amount: Balance,
    /// Total amount of storage fee on withdraw (including withdrawal in shares)
    pub(crate) total_storage_fee_withdrawal: Balance,
    /// Individual withdrawal amounts with their unlocking block for a given domain
    pub(crate) withdrawals: VecDeque<WithdrawalInBalance<DomainBlockNumber, Balance>>,
    /// Withdrawal that was initiated by nominator and not converted to balance due to
    /// unfinished domain epoch.
    pub(crate) withdrawal_in_shares: Option<WithdrawalInShares<DomainBlockNumber, Share, Balance>>,
}

#[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq)]
pub(crate) struct WithdrawalInBalance<DomainBlockNumber, Balance> {
    pub(crate) unlock_at_confirmed_domain_block_number: DomainBlockNumber,
    pub(crate) amount_to_unlock: Balance,
    pub(crate) storage_fee_refund: Balance,
}

#[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq)]
pub(crate) struct WithdrawalInShares<DomainBlockNumber, Share, Balance> {
    pub(crate) domain_epoch: DomainEpoch,
    pub(crate) unlock_at_confirmed_domain_block_number: DomainBlockNumber,
    pub(crate) shares: Share,
    pub(crate) storage_fee_refund: Balance,
}

#[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq)]
pub struct OperatorDeregisteredInfo<DomainBlockNumber> {
    pub domain_epoch: DomainEpoch,
    pub unlock_at_confirmed_domain_block_number: DomainBlockNumber,
}

impl<DomainBlockNumber> From<(DomainId, EpochIndex, DomainBlockNumber)>
    for OperatorDeregisteredInfo<DomainBlockNumber>
{
    fn from(value: (DomainId, EpochIndex, DomainBlockNumber)) -> Self {
        OperatorDeregisteredInfo {
            domain_epoch: (value.0, value.1).into(),
            unlock_at_confirmed_domain_block_number: value.2,
        }
    }
}

/// Type that represents an operator status.
#[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq)]
pub enum OperatorStatus<DomainBlockNumber> {
    Registered,
    /// De-registered at given domain epoch.
    Deregistered(OperatorDeregisteredInfo<DomainBlockNumber>),
    Slashed,
    PendingSlash,
}

/// Type that represents an operator details.
#[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq)]
pub struct Operator<Balance, Share, DomainBlockNumber> {
    pub signing_key: OperatorPublicKey,
    pub current_domain_id: DomainId,
    pub next_domain_id: DomainId,
    pub minimum_nominator_stake: Balance,
    pub nomination_tax: Percent,
    /// Total active stake of combined nominators under this operator.
    pub current_total_stake: Balance,
    /// Total shares of all the nominators under this operator.
    pub current_total_shares: Share,
    /// The status of the operator, it may be stale due to the `OperatorStatus::PendingSlash` is
    /// not assigned to this field directly, thus MUST use the `status()` method to query the status
    /// instead.
    partial_status: OperatorStatus<DomainBlockNumber>,
    /// Total deposits during the previous epoch
    pub deposits_in_epoch: Balance,
    /// Total withdrew shares during the previous epoch
    pub withdrawals_in_epoch: Share,
    /// Total balance deposited to the bundle storage fund
    pub total_storage_fee_deposit: Balance,
}

impl<Balance, Share, DomainBlockNumber> Operator<Balance, Share, DomainBlockNumber> {
    pub fn status<T: Config>(&self, operator_id: OperatorId) -> &OperatorStatus<DomainBlockNumber> {
        if matches!(self.partial_status, OperatorStatus::Slashed) {
            &OperatorStatus::Slashed
        } else if Pallet::<T>::is_operator_pending_to_slash(self.current_domain_id, operator_id) {
            &OperatorStatus::PendingSlash
        } else {
            &self.partial_status
        }
    }

    pub fn update_status(&mut self, new_status: OperatorStatus<DomainBlockNumber>) {
        self.partial_status = new_status;
    }
}

#[cfg(test)]
impl<Balance: Zero, Share: Zero, DomainBlockNumber> Operator<Balance, Share, DomainBlockNumber> {
    pub(crate) fn dummy(
        domain_id: DomainId,
        signing_key: OperatorPublicKey,
        minimum_nominator_stake: Balance,
    ) -> Self {
        Operator {
            signing_key,
            current_domain_id: domain_id,
            next_domain_id: domain_id,
            minimum_nominator_stake,
            nomination_tax: Default::default(),
            current_total_stake: Zero::zero(),
            current_total_shares: Zero::zero(),
            partial_status: OperatorStatus::Registered,
            deposits_in_epoch: Zero::zero(),
            withdrawals_in_epoch: Zero::zero(),
            total_storage_fee_deposit: Zero::zero(),
        }
    }
}

#[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq)]
pub struct StakingSummary<OperatorId, Balance> {
    /// Current epoch index for the domain.
    pub current_epoch_index: EpochIndex,
    /// Total active stake for the current epoch.
    pub current_total_stake: Balance,
    /// Current operators for this epoch
    pub current_operators: BTreeMap<OperatorId, Balance>,
    /// Operators for the next epoch.
    pub next_operators: BTreeSet<OperatorId>,
    /// Operator's current Epoch rewards
    pub current_epoch_rewards: BTreeMap<OperatorId, Balance>,
}

#[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq)]
pub struct OperatorConfig<Balance> {
    pub signing_key: OperatorPublicKey,
    pub minimum_nominator_stake: Balance,
    pub nomination_tax: Percent,
}

#[derive(TypeInfo, Encode, Decode, PalletError, Debug, PartialEq)]
pub enum Error {
    MaximumOperatorId,
    DomainNotInitialized,
    PendingOperatorSwitch,
    InsufficientBalance,
    InsufficientShares,
    ZeroWithdraw,
    BalanceFreeze,
    MinimumOperatorStake,
    UnknownOperator,
    MinimumNominatorStake,
    BalanceOverflow,
    BalanceUnderflow,
    NotOperatorOwner,
    OperatorNotRegistered,
    UnknownNominator,
    MissingOperatorOwner,
    MintBalance,
    BlockNumberOverflow,
    RemoveLock,
    EpochOverflow,
    ShareUnderflow,
    ShareOverflow,
    TooManyPendingStakingOperation,
    OperatorNotAllowed,
    InvalidOperatorSigningKey,
    MissingOperatorEpochSharePrice,
    MissingWithdrawal,
    EpochNotComplete,
    UnlockPeriodNotComplete,
    OperatorNotDeregistered,
    BundleStorageFund(bundle_storage_fund::Error),
    UnconfirmedER,
    TooManyWithdrawals,
}

// Increase `PendingStakingOperationCount` by one and check if the `MaxPendingStakingOperation`
// limit is exceeded
fn note_pending_staking_operation<T: Config>(domain_id: DomainId) -> Result<(), Error> {
    let pending_op_count = PendingStakingOperationCount::<T>::get(domain_id);

    ensure!(
        pending_op_count < T::MaxPendingStakingOperation::get(),
        Error::TooManyPendingStakingOperation
    );

    PendingStakingOperationCount::<T>::set(domain_id, pending_op_count.saturating_add(1));

    Ok(())
}

pub fn do_register_operator<T: Config>(
    operator_owner: T::AccountId,
    domain_id: DomainId,
    amount: BalanceOf<T>,
    config: OperatorConfig<BalanceOf<T>>,
) -> Result<(OperatorId, EpochIndex), Error> {
    note_pending_staking_operation::<T>(domain_id)?;

    DomainStakingSummary::<T>::try_mutate(domain_id, |maybe_domain_stake_summary| {
        ensure!(
            config.signing_key != OperatorPublicKey::from(sr25519::Public::default()),
            Error::InvalidOperatorSigningKey
        );

        ensure!(
            config.minimum_nominator_stake >= T::MinNominatorStake::get(),
            Error::MinimumNominatorStake
        );

        let domain_obj = DomainRegistry::<T>::get(domain_id).ok_or(Error::DomainNotInitialized)?;
        ensure!(
            domain_obj
                .domain_config
                .operator_allow_list
                .is_operator_allowed(&operator_owner),
            Error::OperatorNotAllowed
        );

        let operator_id = NextOperatorId::<T>::get();
        let next_operator_id = operator_id.checked_add(1).ok_or(Error::MaximumOperatorId)?;
        NextOperatorId::<T>::set(next_operator_id);

        OperatorIdOwner::<T>::insert(operator_id, operator_owner.clone());

        // reserve stake balance
        ensure!(
            amount >= T::MinOperatorStake::get(),
            Error::MinimumOperatorStake
        );

        let new_deposit =
            deposit_reserve_for_storage_fund::<T>(operator_id, &operator_owner, amount)
                .map_err(Error::BundleStorageFund)?;

        hold_deposit::<T>(&operator_owner, operator_id, new_deposit.staking)?;

        let domain_stake_summary = maybe_domain_stake_summary
            .as_mut()
            .ok_or(Error::DomainNotInitialized)?;

        let OperatorConfig {
            signing_key,
            minimum_nominator_stake,
            nomination_tax,
        } = config;

        let operator = Operator {
            signing_key: signing_key.clone(),
            current_domain_id: domain_id,
            next_domain_id: domain_id,
            minimum_nominator_stake,
            nomination_tax,
            current_total_stake: Zero::zero(),
            current_total_shares: Zero::zero(),
            partial_status: OperatorStatus::Registered,
            // sum total deposits added during this epoch.
            deposits_in_epoch: new_deposit.staking,
            withdrawals_in_epoch: Zero::zero(),
            total_storage_fee_deposit: new_deposit.storage_fee_deposit,
        };
        Operators::<T>::insert(operator_id, operator);
        // update stake summary to include new operator for next epoch
        domain_stake_summary.next_operators.insert(operator_id);
        // update pending transfers
        let current_domain_epoch = (domain_id, domain_stake_summary.current_epoch_index).into();
        do_calculate_previous_epoch_deposit_shares_and_add_new_deposit::<T>(
            operator_id,
            operator_owner,
            current_domain_epoch,
            new_deposit,
        )?;

        Ok((operator_id, domain_stake_summary.current_epoch_index))
    })
}

pub(crate) struct DepositInfo<Balance> {
    /// If this nominator is currently nominating the operator.
    /// If there are multiple deposits in same epoch, still returns true
    nominating: bool,
    /// Final current deposit in this epoch.
    total_deposit: Balance,
    /// If this is the first deposit in this epoch.
    first_deposit_in_epoch: bool,
}

/// Calculates shares for any pending deposit for previous epoch using the the epoch share price and
/// then create a new pending deposit in the current epoch.
/// If there is a pending deposit for the current epoch, then simply increment the amount.
/// Returns updated deposit info
pub(crate) fn do_calculate_previous_epoch_deposit_shares_and_add_new_deposit<T: Config>(
    operator_id: OperatorId,
    nominator_id: NominatorId<T>,
    current_domain_epoch: DomainEpoch,
    new_deposit: NewDeposit<BalanceOf<T>>,
) -> Result<DepositInfo<BalanceOf<T>>, Error> {
    Deposits::<T>::try_mutate(operator_id, nominator_id, |maybe_deposit| {
        let mut deposit = maybe_deposit.take().unwrap_or_default();
        do_convert_previous_epoch_deposits::<T>(operator_id, &mut deposit)?;

        // add or create new pending deposit
        let (pending_deposit, deposit_info) = match deposit.pending {
            None => {
                let pending_deposit = PendingDeposit {
                    effective_domain_epoch: current_domain_epoch,
                    amount: new_deposit.staking,
                    storage_fee_deposit: new_deposit.storage_fee_deposit,
                };

                let deposit_info = DepositInfo {
                    nominating: !deposit.known.shares.is_zero(),
                    total_deposit: pending_deposit.total()?,
                    first_deposit_in_epoch: true,
                };

                (pending_deposit, deposit_info)
            }
            Some(pending_deposit) => {
                let pending_deposit = PendingDeposit {
                    effective_domain_epoch: current_domain_epoch,
                    amount: pending_deposit
                        .amount
                        .checked_add(&new_deposit.staking)
                        .ok_or(Error::BalanceOverflow)?,
                    storage_fee_deposit: pending_deposit
                        .storage_fee_deposit
                        .checked_add(&new_deposit.storage_fee_deposit)
                        .ok_or(Error::BalanceOverflow)?,
                };

                let deposit_info = DepositInfo {
                    nominating: !deposit.known.shares.is_zero(),
                    total_deposit: pending_deposit.total()?,
                    first_deposit_in_epoch: false,
                };

                (pending_deposit, deposit_info)
            }
        };

        deposit.pending = Some(pending_deposit);
        *maybe_deposit = Some(deposit);
        Ok(deposit_info)
    })
}

pub(crate) fn do_convert_previous_epoch_deposits<T: Config>(
    operator_id: OperatorId,
    deposit: &mut Deposit<T::Share, BalanceOf<T>>,
) -> Result<(), Error> {
    // if it is one of the previous domain epoch, then calculate shares for the epoch and update known deposit
    let epoch_share_price = match deposit.pending.and_then(|pending_deposit| {
        OperatorEpochSharePrice::<T>::get(operator_id, pending_deposit.effective_domain_epoch)
    }) {
        Some(p) => p,
        None => return Ok(()),
    };

    if let Some(PendingDeposit {
        amount,
        storage_fee_deposit,
        ..
    }) = deposit.pending.take()
    {
        let new_shares = epoch_share_price.stake_to_shares::<T>(amount);
        deposit.known.shares = deposit
            .known
            .shares
            .checked_add(&new_shares)
            .ok_or(Error::ShareOverflow)?;
        deposit.known.storage_fee_deposit = deposit
            .known
            .storage_fee_deposit
            .checked_add(&storage_fee_deposit)
            .ok_or(Error::BalanceOverflow)?;
    }

    Ok(())
}

/// Converts any epoch withdrawals into balance using the operator epoch share price.
/// If there is no share price available, this will be no-op
pub(crate) fn do_convert_previous_epoch_withdrawal<T: Config>(
    operator_id: OperatorId,
    withdrawal: &mut Withdrawal<BalanceOf<T>, T::Share, DomainBlockNumberFor<T>>,
) -> Result<(), Error> {
    let epoch_share_price = match withdrawal
        .withdrawal_in_shares
        .as_ref()
        .and_then(|withdraw| OperatorEpochSharePrice::<T>::get(operator_id, withdraw.domain_epoch))
    {
        Some(p) => p,
        None => return Ok(()),
    };

    if let Some(WithdrawalInShares {
        unlock_at_confirmed_domain_block_number,
        shares,
        storage_fee_refund,
        domain_epoch: _,
    }) = withdrawal.withdrawal_in_shares.take()
    {
        let withdrawal_amount = epoch_share_price.shares_to_stake::<T>(shares);
        withdrawal.total_withdrawal_amount = withdrawal
            .total_withdrawal_amount
            .checked_add(&withdrawal_amount)
            .ok_or(Error::BalanceOverflow)?;

        let withdraw_in_balance = WithdrawalInBalance {
            unlock_at_confirmed_domain_block_number,
            amount_to_unlock: withdrawal_amount,
            storage_fee_refund,
        };
        withdrawal.withdrawals.push_back(withdraw_in_balance);
    }

    Ok(())
}

pub(crate) fn do_nominate_operator<T: Config>(
    operator_id: OperatorId,
    nominator_id: T::AccountId,
    amount: BalanceOf<T>,
) -> Result<(), Error> {
    Operators::<T>::try_mutate(operator_id, |maybe_operator| {
        let operator = maybe_operator.as_mut().ok_or(Error::UnknownOperator)?;

        ensure!(
            *operator.status::<T>(operator_id) == OperatorStatus::Registered,
            Error::OperatorNotRegistered
        );

        // If the this is the first staking request of this operator `note_pending_staking_operation` for it
        if operator.deposits_in_epoch.is_zero() && operator.withdrawals_in_epoch.is_zero() {
            note_pending_staking_operation::<T>(operator.current_domain_id)?;
        }

        let domain_stake_summary = DomainStakingSummary::<T>::get(operator.current_domain_id)
            .ok_or(Error::DomainNotInitialized)?;

        // Reserve for the bundle storage fund
        let new_deposit = deposit_reserve_for_storage_fund::<T>(operator_id, &nominator_id, amount)
            .map_err(Error::BundleStorageFund)?;

        hold_deposit::<T>(&nominator_id, operator_id, new_deposit.staking)?;
        Pallet::<T>::deposit_event(Event::OperatorNominated {
            operator_id,
            nominator_id: nominator_id.clone(),
            amount: new_deposit.staking,
        });

        // increment total deposit for operator pool within this epoch
        operator.deposits_in_epoch = operator
            .deposits_in_epoch
            .checked_add(&new_deposit.staking)
            .ok_or(Error::BalanceOverflow)?;

        // Increase total storage fee deposit as there is new deposit to the storage fund
        operator.total_storage_fee_deposit = operator
            .total_storage_fee_deposit
            .checked_add(&new_deposit.storage_fee_deposit)
            .ok_or(Error::BalanceOverflow)?;

        let current_domain_epoch = (
            operator.current_domain_id,
            domain_stake_summary.current_epoch_index,
        )
            .into();

        let DepositInfo {
            nominating,
            total_deposit,
            first_deposit_in_epoch,
        } = do_calculate_previous_epoch_deposit_shares_and_add_new_deposit::<T>(
            operator_id,
            nominator_id,
            current_domain_epoch,
            new_deposit,
        )?;

        // if not a nominator, then ensure
        // - amount >= operator's minimum nominator stake amount.
        // - nominator count does not exceed max nominators.
        // - if first nomination, then increment the nominator count.
        if !nominating {
            ensure!(
                total_deposit >= operator.minimum_nominator_stake,
                Error::MinimumNominatorStake
            );

            if first_deposit_in_epoch {
                NominatorCount::<T>::try_mutate(operator_id, |count| {
                    *count += 1;
                    Ok(())
                })?;
            }
        }

        Ok(())
    })
}

pub(crate) fn hold_deposit<T: Config>(
    who: &T::AccountId,
    operator_id: OperatorId,
    amount: BalanceOf<T>,
) -> Result<(), Error> {
    // ensure there is enough free balance to lock
    ensure!(
        T::Currency::reducible_balance(who, Preservation::Preserve, Fortitude::Polite) >= amount,
        Error::InsufficientBalance
    );

    DepositOnHold::<T>::try_mutate((operator_id, who), |deposit_on_hold| {
        *deposit_on_hold = deposit_on_hold
            .checked_add(&amount)
            .ok_or(Error::BalanceOverflow)?;
        Ok(())
    })?;

    let pending_deposit_hold_id = T::HoldIdentifier::staking_staked();
    T::Currency::hold(&pending_deposit_hold_id, who, amount).map_err(|_| Error::BalanceFreeze)?;

    Ok(())
}

pub(crate) fn do_deregister_operator<T: Config>(
    operator_owner: T::AccountId,
    operator_id: OperatorId,
) -> Result<(), Error> {
    ensure!(
        OperatorIdOwner::<T>::get(operator_id) == Some(operator_owner),
        Error::NotOperatorOwner
    );

    Operators::<T>::try_mutate(operator_id, |maybe_operator| {
        let operator = maybe_operator.as_mut().ok_or(Error::UnknownOperator)?;

        ensure!(
            *operator.status::<T>(operator_id) == OperatorStatus::Registered,
            Error::OperatorNotRegistered
        );

        DomainStakingSummary::<T>::try_mutate(
            operator.current_domain_id,
            |maybe_domain_stake_summary| {
                let stake_summary = maybe_domain_stake_summary
                    .as_mut()
                    .ok_or(Error::DomainNotInitialized)?;

                let head_domain_number = HeadDomainNumber::<T>::get(operator.current_domain_id);
                let unlock_operator_at_domain_block_number = head_domain_number
                    .checked_add(&T::StakeWithdrawalLockingPeriod::get())
                    .ok_or(Error::BlockNumberOverflow)?;
                let operator_deregister_info = (
                    operator.current_domain_id,
                    stake_summary.current_epoch_index,
                    unlock_operator_at_domain_block_number,
                )
                    .into();

                operator.update_status(OperatorStatus::Deregistered(operator_deregister_info));

                stake_summary.next_operators.remove(&operator_id);
                Ok(())
            },
        )
    })
}

/// Different type of withdrawal
///
/// NOTE: if the deposit was made in the current epoch, the user may not be able to withdraw it
/// until the current epoch ends
#[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq)]
pub enum WithdrawStake<Balance, Share> {
    /// Withdraw all stake
    All,
    /// Withdraw a given percentage of the stake
    Percent(Percent),
    /// Withdraw a given amount of stake, calculated by the share price at
    /// this instant, it may not be accurate and may withdraw a bit more
    /// stake if a reward happens later in this epoch
    Stake(Balance),
    /// Withdraw a given amount of share
    Share(Share),
}

impl<Balance: Zero, Share: Zero> WithdrawStake<Balance, Share> {
    pub fn is_zero(&self) -> bool {
        match self {
            Self::All => false,
            Self::Percent(p) => p.is_zero(),
            Self::Stake(s) => s.is_zero(),
            Self::Share(s) => s.is_zero(),
        }
    }
}

// A helper function used to calculate the share price at this instant
fn current_share_price<T: Config>(
    operator_id: OperatorId,
    operator: &Operator<BalanceOf<T>, T::Share, DomainBlockNumberFor<T>>,
    domain_stake_summary: &StakingSummary<OperatorId, BalanceOf<T>>,
) -> SharePrice {
    // Total stake including any reward within this epoch.
    let total_stake = domain_stake_summary
        .current_epoch_rewards
        .get(&operator_id)
        .and_then(|rewards| {
            let operator_tax = operator.nomination_tax.mul_floor(*rewards);
            operator
                .current_total_stake
                .checked_add(rewards)?
                // deduct operator tax
                .checked_sub(&operator_tax)
        })
        .unwrap_or(operator.current_total_stake);

    SharePrice::new::<T>(operator.current_total_shares, total_stake)
}

pub(crate) fn do_withdraw_stake<T: Config>(
    operator_id: OperatorId,
    nominator_id: NominatorId<T>,
    to_withdraw: WithdrawStake<BalanceOf<T>, T::Share>,
) -> Result<(), Error> {
    Operators::<T>::try_mutate(operator_id, |maybe_operator| {
        let operator = maybe_operator.as_mut().ok_or(Error::UnknownOperator)?;
        ensure!(
            *operator.status::<T>(operator_id) == OperatorStatus::Registered,
            Error::OperatorNotRegistered
        );

        ensure!(!to_withdraw.is_zero(), Error::ZeroWithdraw);

        // If the this is the first staking request of this operator `note_pending_staking_operation` for it
        if operator.deposits_in_epoch.is_zero() && operator.withdrawals_in_epoch.is_zero() {
            note_pending_staking_operation::<T>(operator.current_domain_id)?;
        }

        // calculate shares for any previous epoch
        let domain_stake_summary = DomainStakingSummary::<T>::get(operator.current_domain_id)
            .ok_or(Error::DomainNotInitialized)?;
        let domain_current_epoch = (
            operator.current_domain_id,
            domain_stake_summary.current_epoch_index,
        )
            .into();

        let known_share =
            Deposits::<T>::try_mutate(operator_id, nominator_id.clone(), |maybe_deposit| {
                let deposit = maybe_deposit.as_mut().ok_or(Error::UnknownNominator)?;
                do_convert_previous_epoch_deposits::<T>(operator_id, deposit)?;
                Ok(deposit.known.shares)
            })?;

        Withdrawals::<T>::try_mutate(operator_id, nominator_id.clone(), |maybe_withdrawal| {
            if let Some(withdrawal) = maybe_withdrawal {
                do_convert_previous_epoch_withdrawal::<T>(operator_id, withdrawal)?;
                if withdrawal.withdrawals.len() as u32 >= T::WithdrawalLimit::get() {
                    return Err(Error::TooManyWithdrawals);
                }
            }
            Ok(())
        })?;

        let operator_owner =
            OperatorIdOwner::<T>::get(operator_id).ok_or(Error::UnknownOperator)?;

        let is_operator_owner = operator_owner == nominator_id;

        let shares_withdrew = match to_withdraw {
            WithdrawStake::All => known_share,
            WithdrawStake::Percent(p) => p.mul_floor(known_share),
            WithdrawStake::Stake(s) => {
                let share_price =
                    current_share_price::<T>(operator_id, operator, &domain_stake_summary);
                share_price.stake_to_shares::<T>(s)
            }
            WithdrawStake::Share(s) => s,
        };

        Deposits::<T>::try_mutate(operator_id, nominator_id.clone(), |maybe_deposit| {
            let deposit = maybe_deposit.as_mut().ok_or(Error::UnknownNominator)?;
            let known_shares = deposit.known.shares;

            let (remaining_shares, shares_withdrew) = {
                let remaining_shares = known_shares
                    .checked_sub(&shares_withdrew)
                    .ok_or(Error::InsufficientShares)?;

                // short circuit to check if remaining shares can be zero
                if remaining_shares.is_zero() {
                    if is_operator_owner {
                        return Err(Error::MinimumOperatorStake);
                    }

                    (remaining_shares, shares_withdrew)
                } else {
                    let share_price =
                        current_share_price::<T>(operator_id, operator, &domain_stake_summary);

                    let remaining_storage_fee =
                        Perbill::from_rational(remaining_shares, known_shares)
                            .mul_floor(deposit.known.storage_fee_deposit);

                    let remaining_stake = share_price
                        .shares_to_stake::<T>(remaining_shares)
                        .checked_add(&remaining_storage_fee)
                        .ok_or(Error::BalanceOverflow)?;

                    // ensure the remaining share value is atleast the defined minimum
                    // MinOperatorStake if a nominator is operator pool owner
                    if is_operator_owner && remaining_stake.lt(&T::MinOperatorStake::get()) {
                        return Err(Error::MinimumOperatorStake);
                    }

                    // if not an owner, if remaining balance < MinNominatorStake, then withdraw all shares.
                    if !is_operator_owner && remaining_stake.lt(&operator.minimum_nominator_stake) {
                        (T::Share::zero(), known_shares)
                    } else {
                        (remaining_shares, shares_withdrew)
                    }
                }
            };

            // Withdraw storage fund, the `withdraw_storage_fee` amount of fund will be transfered
            // and hold on the nominator account
            let storage_fee_to_withdraw = Perbill::from_rational(shares_withdrew, known_shares)
                .mul_floor(deposit.known.storage_fee_deposit);

            let withdraw_storage_fee = {
                let storage_fund_redeem_price = bundle_storage_fund::storage_fund_redeem_price::<T>(
                    operator_id,
                    operator.total_storage_fee_deposit,
                );
                bundle_storage_fund::withdraw_and_hold::<T>(
                    operator_id,
                    &nominator_id,
                    storage_fund_redeem_price.redeem(storage_fee_to_withdraw),
                )
                .map_err(Error::BundleStorageFund)?
            };

            deposit.known.storage_fee_deposit = deposit
                .known
                .storage_fee_deposit
                .checked_sub(&storage_fee_to_withdraw)
                .ok_or(Error::BalanceOverflow)?;

            operator.total_storage_fee_deposit = operator
                .total_storage_fee_deposit
                .checked_sub(&storage_fee_to_withdraw)
                .ok_or(Error::BalanceOverflow)?;

            // update operator pool to note withdrew shares in the epoch
            operator.withdrawals_in_epoch = operator
                .withdrawals_in_epoch
                .checked_add(&shares_withdrew)
                .ok_or(Error::ShareOverflow)?;

            deposit.known.shares = remaining_shares;
            if remaining_shares.is_zero() {
                if let Some(pending_deposit) = deposit.pending {
                    // if there is a pending deposit, then ensure
                    // the new deposit is atleast minimum nominator stake
                    ensure!(
                        pending_deposit.total()? >= operator.minimum_nominator_stake,
                        Error::MinimumNominatorStake
                    );
                } else {
                    // reduce nominator count if withdraw all and there are no pending deposits
                    NominatorCount::<T>::mutate(operator_id, |count| {
                        *count -= 1;
                    });
                }
            }

            let head_domain_number = HeadDomainNumber::<T>::get(operator.current_domain_id);
            let unlock_at_confirmed_domain_block_number = head_domain_number
                .checked_add(&T::StakeWithdrawalLockingPeriod::get())
                .ok_or(Error::BlockNumberOverflow)?;

            Withdrawals::<T>::try_mutate(operator_id, nominator_id, |maybe_withdrawal| {
                let mut withdrawal = maybe_withdrawal.take().unwrap_or_default();
                // if this is some, then the withdrawal was initiated in this current epoch due to conversion
                // of previous epoch withdrawals from shares to balances above. So just update it instead
                let new_withdrawal_in_shares = match withdrawal.withdrawal_in_shares.take() {
                    Some(WithdrawalInShares {
                        domain_epoch,
                        shares,
                        storage_fee_refund,
                        ..
                    }) => WithdrawalInShares {
                        domain_epoch,
                        shares: shares
                            .checked_add(&shares_withdrew)
                            .ok_or(Error::ShareOverflow)?,
                        unlock_at_confirmed_domain_block_number,
                        storage_fee_refund: storage_fee_refund
                            .checked_add(&withdraw_storage_fee)
                            .ok_or(Error::BalanceOverflow)?,
                    },
                    None => WithdrawalInShares {
                        domain_epoch: domain_current_epoch,
                        unlock_at_confirmed_domain_block_number,
                        shares: shares_withdrew,
                        storage_fee_refund: withdraw_storage_fee,
                    },
                };
                withdrawal.withdrawal_in_shares = Some(new_withdrawal_in_shares);
                withdrawal.total_storage_fee_withdrawal = withdrawal
                    .total_storage_fee_withdrawal
                    .checked_add(&withdraw_storage_fee)
                    .ok_or(Error::BalanceOverflow)?;

                *maybe_withdrawal = Some(withdrawal);
                Ok(())
            })
        })
    })
}

/// Unlocks any withdraws that are ready to be unlocked.
pub(crate) fn do_unlock_funds<T: Config>(
    operator_id: OperatorId,
    nominator_id: NominatorId<T>,
) -> Result<(), Error> {
    let operator = Operators::<T>::get(operator_id).ok_or(Error::UnknownOperator)?;
    ensure!(
        *operator.status::<T>(operator_id) == OperatorStatus::Registered,
        Error::OperatorNotRegistered
    );

    Withdrawals::<T>::try_mutate_exists(operator_id, nominator_id.clone(), |maybe_withdrawal| {
        let withdrawal = maybe_withdrawal.as_mut().ok_or(Error::MissingWithdrawal)?;
        do_convert_previous_epoch_withdrawal::<T>(operator_id, withdrawal)?;

        ensure!(!withdrawal.withdrawals.is_empty(), Error::MissingWithdrawal);

        let head_domain_number = HeadDomainNumber::<T>::get(operator.current_domain_id);

        let mut total_unlocked_amount = BalanceOf::<T>::zero();
        let mut total_storage_fee_refund = BalanceOf::<T>::zero();
        loop {
            if withdrawal
                .withdrawals
                .front()
                .map(|w| w.unlock_at_confirmed_domain_block_number > head_domain_number)
                .unwrap_or(true)
            {
                break;
            }

            let WithdrawalInBalance {
                amount_to_unlock,
                storage_fee_refund,
                ..
            } = withdrawal
                .withdrawals
                .pop_front()
                .expect("Must not empty as checked above; qed");

            total_unlocked_amount = total_unlocked_amount
                .checked_add(&amount_to_unlock)
                .ok_or(Error::BalanceOverflow)?;

            total_storage_fee_refund = total_storage_fee_refund
                .checked_add(&storage_fee_refund)
                .ok_or(Error::BalanceOverflow)?;
        }

        // There is withdrawal but none being processed meaning the first withdrawal's unlock period has
        // not completed yet
        ensure!(
            !total_unlocked_amount.is_zero() || !total_storage_fee_refund.is_zero(),
            Error::UnlockPeriodNotComplete
        );

        // deduct the amount unlocked from total
        withdrawal.total_withdrawal_amount = withdrawal
            .total_withdrawal_amount
            .checked_sub(&total_unlocked_amount)
            .ok_or(Error::BalanceUnderflow)?;

        withdrawal.total_storage_fee_withdrawal = withdrawal
            .total_storage_fee_withdrawal
            .checked_sub(&total_storage_fee_refund)
            .ok_or(Error::BalanceUnderflow)?;

        // If the amount to release is more than currently locked,
        // mint the diff and release the rest
        let (amount_to_mint, amount_to_release) = DepositOnHold::<T>::try_mutate(
            (operator_id, nominator_id.clone()),
            |deposit_on_hold| {
                let amount_to_release = total_unlocked_amount.min(*deposit_on_hold);
                let amount_to_mint = total_unlocked_amount.saturating_sub(*deposit_on_hold);

                *deposit_on_hold = deposit_on_hold.saturating_sub(amount_to_release);

                Ok((amount_to_mint, amount_to_release))
            },
        )?;

        // Mint any gains
        if !amount_to_mint.is_zero() {
            mint_funds::<T>(&nominator_id, amount_to_mint)?;
        }
        // Release staking fund
        if !amount_to_release.is_zero() {
            let staked_hold_id = T::HoldIdentifier::staking_staked();
            T::Currency::release(
                &staked_hold_id,
                &nominator_id,
                amount_to_release,
                Precision::Exact,
            )
            .map_err(|_| Error::RemoveLock)?;
        }

        Pallet::<T>::deposit_event(Event::NominatedStakedUnlocked {
            operator_id,
            nominator_id: nominator_id.clone(),
            unlocked_amount: total_unlocked_amount,
        });

        // Release storage fund
        let storage_fund_hold_id = T::HoldIdentifier::storage_fund_withdrawal();
        T::Currency::release(
            &storage_fund_hold_id,
            &nominator_id,
            total_storage_fee_refund,
            Precision::Exact,
        )
        .map_err(|_| Error::RemoveLock)?;

        Pallet::<T>::deposit_event(Event::StorageFeeUnlocked {
            operator_id,
            nominator_id: nominator_id.clone(),
            storage_fee: total_storage_fee_refund,
        });

        // if there are no withdrawals, then delete the storage as well
        if withdrawal.withdrawals.is_empty() && withdrawal.withdrawal_in_shares.is_none() {
            *maybe_withdrawal = None;
            // if there is no deposit or pending deposits, then clean up the deposit state as well
            Deposits::<T>::mutate_exists(operator_id, nominator_id, |maybe_deposit| {
                if let Some(deposit) = maybe_deposit
                    && deposit.known.shares.is_zero()
                    && deposit.pending.is_none()
                {
                    *maybe_deposit = None
                }
            });
        }

        Ok(())
    })
}

/// Unlocks an already de-registered operator's nominator given unlock wait period is complete.
pub(crate) fn do_unlock_nominator<T: Config>(
    operator_id: OperatorId,
    nominator_id: NominatorId<T>,
) -> Result<(), Error> {
    Operators::<T>::try_mutate_exists(operator_id, |maybe_operator| {
        // take the operator so this operator info is removed once we unlock the operator.
        let mut operator = maybe_operator.take().ok_or(Error::UnknownOperator)?;
        let OperatorDeregisteredInfo {
            domain_epoch,
            unlock_at_confirmed_domain_block_number,
        } = match operator.status::<T>(operator_id) {
            OperatorStatus::Deregistered(operator_deregistered_info) => operator_deregistered_info,
            _ => return Err(Error::OperatorNotDeregistered),
        };

        let (domain_id, _) = domain_epoch.deconstruct();
        let head_domain_number = HeadDomainNumber::<T>::get(domain_id);
        ensure!(
            *unlock_at_confirmed_domain_block_number <= head_domain_number,
            Error::UnlockPeriodNotComplete
        );

        let mut total_shares = operator.current_total_shares;
        let mut total_stake = operator.current_total_stake;
        let share_price = SharePrice::new::<T>(total_shares, total_stake);

        let mut total_storage_fee_deposit = operator.total_storage_fee_deposit;
        let storage_fund_redeem_price = bundle_storage_fund::storage_fund_redeem_price::<T>(
            operator_id,
            total_storage_fee_deposit,
        );
        let mut deposit = Deposits::<T>::take(operator_id, nominator_id.clone())
            .ok_or(Error::UnknownNominator)?;

        // convert any deposits from the previous epoch to shares
        do_convert_previous_epoch_deposits::<T>(operator_id, &mut deposit)?;

        // if there are any withdrawals from this operator, account for them
        // if the withdrawals has share price noted, then convert them to SSC
        // if no share price, then it must be intitated in the epoch before operator de-registered,
        // so get the shares as is and include them in the total staked shares.
        let (
            amount_ready_to_withdraw,
            total_storage_fee_withdrawal,
            shares_withdrew_in_current_epoch,
        ) = Withdrawals::<T>::take(operator_id, nominator_id.clone())
            .map(|mut withdrawal| {
                do_convert_previous_epoch_withdrawal::<T>(operator_id, &mut withdrawal)?;
                Ok((
                    withdrawal.total_withdrawal_amount,
                    withdrawal.total_storage_fee_withdrawal,
                    withdrawal
                        .withdrawal_in_shares
                        .map(|WithdrawalInShares { shares, .. }| shares)
                        .unwrap_or_default(),
                ))
            })
            .unwrap_or(Ok((Zero::zero(), Zero::zero(), Zero::zero())))?;

        // include all the known shares and shares that were withdrawn in the current epoch
        let nominator_shares = deposit
            .known
            .shares
            .checked_add(&shares_withdrew_in_current_epoch)
            .ok_or(Error::ShareOverflow)?;

        // current staked amount
        let nominator_staked_amount = share_price.shares_to_stake::<T>(nominator_shares);

        // amount deposited by this nominator before operator de-registered.
        let amount_deposited_in_epoch = deposit
            .pending
            .map(|pending_deposit| pending_deposit.amount)
            .unwrap_or_default();

        let total_amount_to_unlock = nominator_staked_amount
            .checked_add(&amount_ready_to_withdraw)
            .and_then(|amount| amount.checked_add(&amount_deposited_in_epoch))
            .ok_or(Error::BalanceOverflow)?;

        // Remove the lock and mint any gains
        let current_locked_amount = DepositOnHold::<T>::take((operator_id, nominator_id.clone()));
        if let Some(amount_to_mint) = total_amount_to_unlock.checked_sub(&current_locked_amount) {
            mint_funds::<T>(&nominator_id, amount_to_mint)?;
        }
        if !current_locked_amount.is_zero() {
            let staked_hold_id = T::HoldIdentifier::staking_staked();
            T::Currency::release(
                &staked_hold_id,
                &nominator_id,
                current_locked_amount,
                Precision::Exact,
            )
            .map_err(|_| Error::RemoveLock)?;
        }

        Pallet::<T>::deposit_event(Event::NominatedStakedUnlocked {
            operator_id,
            nominator_id: nominator_id.clone(),
            unlocked_amount: total_amount_to_unlock,
        });

        total_stake = total_stake.saturating_sub(nominator_staked_amount);
        total_shares = total_shares.saturating_sub(nominator_shares);

        // Withdraw all storage fee for the nominator
        let nominator_total_storage_fee_deposit = deposit
            .pending
            .map(|pending_deposit| pending_deposit.storage_fee_deposit)
            .unwrap_or(Zero::zero())
            .checked_add(&deposit.known.storage_fee_deposit)
            .ok_or(Error::BalanceOverflow)?;

        bundle_storage_fund::withdraw_to::<T>(
            operator_id,
            &nominator_id,
            storage_fund_redeem_price.redeem(nominator_total_storage_fee_deposit),
        )
        .map_err(Error::BundleStorageFund)?;

        // Release all storage fee on withdraw of the nominator
        let storage_fund_hold_id = T::HoldIdentifier::storage_fund_withdrawal();
        T::Currency::release(
            &storage_fund_hold_id,
            &nominator_id,
            total_storage_fee_withdrawal,
            Precision::Exact,
        )
        .map_err(|_| Error::RemoveLock)?;

        Pallet::<T>::deposit_event(Event::StorageFeeUnlocked {
            operator_id,
            nominator_id: nominator_id.clone(),
            storage_fee: total_storage_fee_withdrawal,
        });

        // reduce total storage fee deposit with nominator total fee deposit
        total_storage_fee_deposit =
            total_storage_fee_deposit.saturating_sub(nominator_total_storage_fee_deposit);

        let current_nominator_count = NominatorCount::<T>::get(operator_id);
        let operator_owner =
            OperatorIdOwner::<T>::get(operator_id).ok_or(Error::UnknownOperator)?;

        // reduce the nominator count for operator if the nominator is not operator owner
        // since operator own nominator is not counted into nominator count for operator.
        let current_nominator_count =
            if operator_owner != nominator_id && current_nominator_count > 0 {
                let new_nominator_count = current_nominator_count - 1;
                NominatorCount::<T>::set(operator_id, new_nominator_count);
                new_nominator_count
            } else {
                current_nominator_count
            };

        // operator state can be cleaned if all the nominators have unlocked their stake and operator
        // themself unlocked their stake.
        let cleanup_operator = current_nominator_count == 0
            && !Deposits::<T>::contains_key(operator_id, operator_owner);

        if cleanup_operator {
            do_cleanup_operator::<T>(operator_id, total_stake)?
        } else {
            // set update total shares, total stake and total storage fee deposit for operator
            operator.current_total_shares = total_shares;
            operator.current_total_stake = total_stake;
            operator.total_storage_fee_deposit = total_storage_fee_deposit;

            *maybe_operator = Some(operator);
        }

        Ok(())
    })
}

/// Removes all operator storages and mints the total stake back to treasury.
pub(crate) fn do_cleanup_operator<T: Config>(
    operator_id: OperatorId,
    total_stake: BalanceOf<T>,
) -> Result<(), Error> {
    // transfer any remaining storage fund to treasury
    bundle_storage_fund::transfer_all_to_treasury::<T>(operator_id)
        .map_err(Error::BundleStorageFund)?;

    // transfer any remaining amount to treasury
    mint_into_treasury::<T>(total_stake)?;

    // remove OperatorOwner Details
    OperatorIdOwner::<T>::remove(operator_id);

    // remove operator epoch share prices
    let _ = OperatorEpochSharePrice::<T>::clear_prefix(operator_id, u32::MAX, None);

    // remove nominator count for this operator.
    NominatorCount::<T>::remove(operator_id);

    Ok(())
}

/// Distribute the reward to the operators equally and drop any dust to treasury.
pub(crate) fn do_reward_operators<T: Config>(
    domain_id: DomainId,
    source: OperatorRewardSource<BlockNumberFor<T>>,
    operators: IntoIter<OperatorId>,
    rewards: BalanceOf<T>,
) -> Result<(), Error> {
    if rewards.is_zero() {
        return Ok(());
    }
    DomainStakingSummary::<T>::mutate(domain_id, |maybe_stake_summary| {
        let stake_summary = maybe_stake_summary
            .as_mut()
            .ok_or(Error::DomainNotInitialized)?;

        let total_count = operators.len() as u64;
        // calculate the operator weights based on the number of times they are repeated in the original list.
        let operator_weights = operators.into_iter().fold(
            BTreeMap::<OperatorId, u64>::new(),
            |mut acc, operator_id| {
                acc.entry(operator_id)
                    .and_modify(|weight| *weight += 1)
                    .or_insert(1);
                acc
            },
        );

        let mut allocated_rewards = BalanceOf::<T>::zero();
        for (operator_id, weight) in operator_weights {
            let operator_reward = {
                let distribution = Perquintill::from_rational(weight, total_count);
                distribution.mul_floor(rewards)
            };

            stake_summary
                .current_epoch_rewards
                .entry(operator_id)
                .and_modify(|rewards| *rewards = rewards.saturating_add(operator_reward))
                .or_insert(operator_reward);

            Pallet::<T>::deposit_event(Event::OperatorRewarded {
                source: source.clone(),
                operator_id,
                reward: operator_reward,
            });

            allocated_rewards = allocated_rewards
                .checked_add(&operator_reward)
                .ok_or(Error::BalanceOverflow)?;
        }

        // mint remaining funds to treasury
        mint_into_treasury::<T>(
            rewards
                .checked_sub(&allocated_rewards)
                .ok_or(Error::BalanceUnderflow)?,
        )
    })
}

/// Freezes the slashed operators and moves the operator to be removed once the domain they are
/// operating finishes the epoch.
pub(crate) fn do_mark_operators_as_slashed<T: Config>(
    operator_ids: impl AsRef<[OperatorId]>,
    slash_reason: SlashedReason<DomainBlockNumberFor<T>, ReceiptHashFor<T>>,
) -> Result<(), Error> {
    for operator_id in operator_ids.as_ref() {
        Operators::<T>::try_mutate(operator_id, |maybe_operator| {
            let operator = match maybe_operator.as_mut() {
                // If the operator is already slashed and removed due to fraud proof, when the operator
                // is slash again due to invalid bundle, which happen after the ER is confirmed, we can
                // not find the operator here thus just return.
                None => return Ok(()),
                Some(operator) => operator,
            };
            let mut pending_slashes =
                PendingSlashes::<T>::get(operator.current_domain_id).unwrap_or_default();

            if pending_slashes.contains(operator_id) {
                return Ok(());
            }

            DomainStakingSummary::<T>::try_mutate(
                operator.current_domain_id,
                |maybe_domain_stake_summary| {
                    let stake_summary = maybe_domain_stake_summary
                        .as_mut()
                        .ok_or(Error::DomainNotInitialized)?;

                    // slash and remove operator from next and current epoch set
                    operator.update_status(OperatorStatus::Slashed);
                    stake_summary.current_operators.remove(operator_id);
                    stake_summary.next_operators.remove(operator_id);
                    stake_summary.current_total_stake = stake_summary
                        .current_total_stake
                        .checked_sub(&operator.current_total_stake)
                        .ok_or(Error::BalanceUnderflow)?;

                    pending_slashes.insert(*operator_id);
                    PendingSlashes::<T>::insert(operator.current_domain_id, pending_slashes);
                    Pallet::<T>::deposit_event(Event::OperatorSlashed {
                        operator_id: *operator_id,
                        reason: slash_reason.clone(),
                    });
                    Ok(())
                },
            )
        })?
    }

    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::domain_registry::{DomainConfig, DomainObject};
    use crate::pallet::{
        Config, Deposits, DomainRegistry, DomainStakingSummary, HeadDomainNumber, NextOperatorId,
        NominatorCount, OperatorIdOwner, Operators, PendingSlashes, Withdrawals,
    };
    use crate::staking::{
        do_convert_previous_epoch_withdrawal, do_mark_operators_as_slashed, do_nominate_operator,
        do_reward_operators, do_unlock_funds, do_withdraw_stake, Error as StakingError, Operator,
        OperatorConfig, OperatorStatus, StakingSummary, WithdrawStake,
    };
    use crate::staking_epoch::{do_finalize_domain_current_epoch, do_slash_operator};
    use crate::tests::{new_test_ext, ExistentialDeposit, RuntimeOrigin, Test};
    use crate::{
        bundle_storage_fund, BalanceOf, Error, NominatorId, SlashedReason, MAX_NOMINATORS_TO_SLASH,
    };
    use frame_support::traits::fungible::Mutate;
    use frame_support::traits::Currency;
    use frame_support::weights::Weight;
    use frame_support::{assert_err, assert_ok};
    use sp_core::{sr25519, Pair, U256};
    use sp_domains::{
        DomainId, OperatorAllowList, OperatorId, OperatorPair, OperatorPublicKey,
        OperatorRewardSource,
    };
    use sp_runtime::traits::Zero;
    use sp_runtime::{PerThing, Perbill};
    use std::collections::{BTreeMap, BTreeSet};
    use std::vec;
    use subspace_runtime_primitives::SSC;

    type Balances = pallet_balances::Pallet<Test>;
    type Domains = crate::Pallet<Test>;

    const STORAGE_FEE_RESERVE: Perbill = Perbill::from_percent(20);

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn register_operator(
        domain_id: DomainId,
        operator_account: <Test as frame_system::Config>::AccountId,
        operator_free_balance: BalanceOf<Test>,
        operator_stake: BalanceOf<Test>,
        minimum_nominator_stake: BalanceOf<Test>,
        signing_key: OperatorPublicKey,
        mut nominators: BTreeMap<NominatorId<Test>, (BalanceOf<Test>, BalanceOf<Test>)>,
    ) -> (OperatorId, OperatorConfig<BalanceOf<Test>>) {
        nominators.insert(operator_account, (operator_free_balance, operator_stake));
        for nominator in &nominators {
            Balances::set_balance(nominator.0, nominator.1 .0);
            assert_eq!(Balances::usable_balance(nominator.0), nominator.1 .0);
        }
        nominators.remove(&operator_account);

        if !DomainRegistry::<Test>::contains_key(domain_id) {
            let domain_config = DomainConfig {
                domain_name: String::from_utf8(vec![0; 1024]).unwrap(),
                runtime_id: 0,
                max_bundle_size: u32::MAX,
                max_bundle_weight: Weight::MAX,
                bundle_slot_probability: (0, 0),
                operator_allow_list: OperatorAllowList::Anyone,
                initial_balances: Default::default(),
                domain_runtime_config: Default::default(),
            };

            let domain_obj = DomainObject {
                owner_account_id: 0,
                created_at: 0,
                genesis_receipt_hash: Default::default(),
                domain_config,
                domain_runtime_info: Default::default(),
                domain_instantiation_deposit: Default::default(),
            };

            DomainRegistry::<Test>::insert(domain_id, domain_obj);
        }

        if !DomainStakingSummary::<Test>::contains_key(domain_id) {
            DomainStakingSummary::<Test>::insert(
                domain_id,
                StakingSummary {
                    current_epoch_index: 0,
                    current_total_stake: 0,
                    current_operators: BTreeMap::new(),
                    next_operators: BTreeSet::new(),
                    current_epoch_rewards: BTreeMap::new(),
                },
            );
        }

        let operator_config = OperatorConfig {
            signing_key,
            minimum_nominator_stake,
            nomination_tax: Default::default(),
        };

        let res = Domains::register_operator(
            RuntimeOrigin::signed(operator_account),
            domain_id,
            operator_stake,
            operator_config.clone(),
        );
        assert_ok!(res);

        let operator_id = NextOperatorId::<Test>::get() - 1;
        let mut expected_nominator_count = 0;
        for nominator in nominators {
            if nominator.1 .1.is_zero() {
                continue;
            }

            expected_nominator_count += 1;
            let res = Domains::nominate_operator(
                RuntimeOrigin::signed(nominator.0),
                operator_id,
                nominator.1 .1,
            );
            assert_ok!(res);
        }

        let nominator_count = NominatorCount::<Test>::get(operator_id) as usize;
        assert_eq!(nominator_count, expected_nominator_count);

        (operator_id, operator_config)
    }

    #[test]
    fn test_register_operator_invalid_signing_key() {
        let domain_id = DomainId::new(0);
        let operator_account = 1;

        let mut ext = new_test_ext();
        ext.execute_with(|| {
            let operator_config = OperatorConfig {
                signing_key: OperatorPublicKey::from(sr25519::Public::default()),
                minimum_nominator_stake: Default::default(),
                nomination_tax: Default::default(),
            };

            let res = Domains::register_operator(
                RuntimeOrigin::signed(operator_account),
                domain_id,
                Default::default(),
                operator_config,
            );
            assert_err!(
                res,
                Error::<Test>::Staking(StakingError::InvalidOperatorSigningKey)
            );
        });
    }

    #[test]
    fn test_register_operator_minimum_nominator_stake() {
        let domain_id = DomainId::new(0);
        let operator_account = 1;
        let pair = OperatorPair::from_seed(&U256::from(0u32).into());

        let mut ext = new_test_ext();
        ext.execute_with(|| {
            let operator_config = OperatorConfig {
                signing_key: pair.public(),
                minimum_nominator_stake: Default::default(),
                nomination_tax: Default::default(),
            };

            let res = Domains::register_operator(
                RuntimeOrigin::signed(operator_account),
                domain_id,
                Default::default(),
                operator_config,
            );
            assert_err!(
                res,
                Error::<Test>::Staking(StakingError::MinimumNominatorStake)
            );
        });
    }

    #[test]
    fn test_register_operator() {
        let domain_id = DomainId::new(0);
        let operator_account = 1;
        let operator_free_balance = 2500 * SSC;
        let operator_total_stake = 1000 * SSC;
        let operator_stake = 800 * SSC;
        let operator_storage_fee_deposit = 200 * SSC;
        let pair = OperatorPair::from_seed(&U256::from(0u32).into());

        let mut ext = new_test_ext();
        ext.execute_with(|| {
            let (operator_id, mut operator_config) = register_operator(
                domain_id,
                operator_account,
                operator_free_balance,
                operator_total_stake,
                SSC,
                pair.public(),
                BTreeMap::new(),
            );

            assert_eq!(NextOperatorId::<Test>::get(), 1);
            // operator_id should be 0 and be registered
            assert_eq!(
                OperatorIdOwner::<Test>::get(operator_id).unwrap(),
                operator_account
            );
            assert_eq!(
                Operators::<Test>::get(operator_id).unwrap(),
                Operator {
                    signing_key: pair.public(),
                    current_domain_id: domain_id,
                    next_domain_id: domain_id,
                    minimum_nominator_stake: SSC,
                    nomination_tax: Default::default(),
                    current_total_stake: operator_stake,
                    current_total_shares: operator_stake,
                    partial_status: OperatorStatus::Registered,
                    deposits_in_epoch: 0,
                    withdrawals_in_epoch: 0,
                    total_storage_fee_deposit: operator_storage_fee_deposit,
                }
            );

            let stake_summary = DomainStakingSummary::<Test>::get(domain_id).unwrap();
            assert!(stake_summary.next_operators.contains(&operator_id));
            assert_eq!(stake_summary.current_total_stake, operator_stake);

            assert_eq!(
                Balances::usable_balance(operator_account),
                operator_free_balance - operator_total_stake - ExistentialDeposit::get()
            );

            // registering with same operator key is allowed
            let res = Domains::register_operator(
                RuntimeOrigin::signed(operator_account),
                domain_id,
                operator_stake,
                operator_config.clone(),
            );
            assert_ok!(res);

            // cannot use the locked funds to register a new operator
            let new_pair = OperatorPair::from_seed(&U256::from(1u32).into());
            operator_config.signing_key = new_pair.public();
            let res = Domains::register_operator(
                RuntimeOrigin::signed(operator_account),
                domain_id,
                operator_stake,
                operator_config,
            );
            assert_err!(
                res,
                Error::<Test>::Staking(crate::staking::Error::InsufficientBalance)
            );

            let nominator_count = NominatorCount::<Test>::get(operator_id);
            assert_eq!(nominator_count, 0);
        });
    }

    #[test]
    fn nominate_operator() {
        let domain_id = DomainId::new(0);
        let operator_account = 1;
        let operator_free_balance = 1500 * SSC;
        let operator_total_stake = 1000 * SSC;
        let operator_stake = 800 * SSC;
        let operator_storage_fee_deposit = 200 * SSC;
        let pair = OperatorPair::from_seed(&U256::from(0u32).into());

        let nominator_account = 2;
        let nominator_free_balance = 150 * SSC;
        let nominator_total_stake = 100 * SSC;
        let nominator_stake = 80 * SSC;
        let nominator_storage_fee_deposit = 20 * SSC;

        let mut ext = new_test_ext();
        ext.execute_with(|| {
            let (operator_id, _) = register_operator(
                domain_id,
                operator_account,
                operator_free_balance,
                operator_total_stake,
                10 * SSC,
                pair.public(),
                BTreeMap::from_iter(vec![(
                    nominator_account,
                    (nominator_free_balance, nominator_total_stake),
                )]),
            );

            let domain_staking_summary = DomainStakingSummary::<Test>::get(domain_id).unwrap();
            assert_eq!(domain_staking_summary.current_total_stake, operator_stake);

            let operator = Operators::<Test>::get(operator_id).unwrap();
            assert_eq!(operator.current_total_stake, operator_stake);
            assert_eq!(operator.current_total_shares, operator_stake);
            assert_eq!(
                operator.total_storage_fee_deposit,
                operator_storage_fee_deposit + nominator_storage_fee_deposit
            );
            assert_eq!(operator.deposits_in_epoch, nominator_stake);

            let pending_deposit = Deposits::<Test>::get(0, nominator_account)
                .unwrap()
                .pending
                .unwrap();
            assert_eq!(pending_deposit.amount, nominator_stake);
            assert_eq!(
                pending_deposit.storage_fee_deposit,
                nominator_storage_fee_deposit
            );
            assert_eq!(pending_deposit.total().unwrap(), nominator_total_stake);

            assert_eq!(
                Balances::usable_balance(nominator_account),
                nominator_free_balance - nominator_total_stake - ExistentialDeposit::get()
            );

            // another transfer with an existing transfer in place should lead to single
            let addtional_nomination_total_stake = 40 * SSC;
            let addtional_nomination_stake = 32 * SSC;
            let addtional_nomination_storage_fee_deposit = 8 * SSC;
            let res = Domains::nominate_operator(
                RuntimeOrigin::signed(nominator_account),
                operator_id,
                addtional_nomination_total_stake,
            );
            assert_ok!(res);
            let pending_deposit = Deposits::<Test>::get(0, nominator_account)
                .unwrap()
                .pending
                .unwrap();
            assert_eq!(
                pending_deposit.amount,
                nominator_stake + addtional_nomination_stake
            );
            assert_eq!(
                pending_deposit.storage_fee_deposit,
                nominator_storage_fee_deposit + addtional_nomination_storage_fee_deposit
            );

            let operator = Operators::<Test>::get(operator_id).unwrap();
            assert_eq!(operator.current_total_stake, operator_stake);
            assert_eq!(
                operator.deposits_in_epoch,
                nominator_stake + addtional_nomination_stake
            );
            assert_eq!(
                operator.total_storage_fee_deposit,
                operator_storage_fee_deposit
                    + nominator_storage_fee_deposit
                    + addtional_nomination_storage_fee_deposit
            );

            let nominator_count = NominatorCount::<Test>::get(operator_id);
            assert_eq!(nominator_count, 1);

            // do epoch transition
            do_finalize_domain_current_epoch::<Test>(domain_id).unwrap();

            let operator = Operators::<Test>::get(operator_id).unwrap();
            assert_eq!(
                operator.current_total_stake,
                operator_stake + nominator_stake + addtional_nomination_stake
            );

            let domain_staking_summary = DomainStakingSummary::<Test>::get(domain_id).unwrap();
            assert_eq!(
                domain_staking_summary.current_total_stake,
                operator_stake + nominator_stake + addtional_nomination_stake
            );
        });
    }

    #[test]
    fn operator_deregistration() {
        let domain_id = DomainId::new(0);
        let operator_account = 1;
        let operator_stake = 200 * SSC;
        let operator_free_balance = 250 * SSC;
        let pair = OperatorPair::from_seed(&U256::from(0u32).into());
        let mut ext = new_test_ext();
        ext.execute_with(|| {
            let (operator_id, _) = register_operator(
                domain_id,
                operator_account,
                operator_free_balance,
                operator_stake,
                SSC,
                pair.public(),
                BTreeMap::new(),
            );

            let res =
                Domains::deregister_operator(RuntimeOrigin::signed(operator_account), operator_id);
            assert_ok!(res);

            let domain_stake_summary = DomainStakingSummary::<Test>::get(domain_id).unwrap();
            assert!(!domain_stake_summary.next_operators.contains(&operator_id));

            let operator = Operators::<Test>::get(operator_id).unwrap();
            assert_eq!(
                *operator.status::<Test>(operator_id),
                OperatorStatus::Deregistered(
                    (
                        domain_id,
                        domain_stake_summary.current_epoch_index,
                        // since the Withdrawals locking period is 5 and confirmed domain block is 0
                        5
                    )
                        .into()
                )
            );

            // operator nomination will not work since the operator is already de-registered
            let new_domain_id = DomainId::new(1);
            let domain_config = DomainConfig {
                domain_name: String::from_utf8(vec![0; 1024]).unwrap(),
                runtime_id: 0,
                max_bundle_size: u32::MAX,
                max_bundle_weight: Weight::MAX,
                bundle_slot_probability: (0, 0),
                operator_allow_list: OperatorAllowList::Anyone,
                initial_balances: Default::default(),
                domain_runtime_config: Default::default(),
            };

            let domain_obj = DomainObject {
                owner_account_id: 0,
                created_at: 0,
                genesis_receipt_hash: Default::default(),
                domain_config,
                domain_runtime_info: Default::default(),
                domain_instantiation_deposit: Default::default(),
            };

            DomainRegistry::<Test>::insert(new_domain_id, domain_obj);
            DomainStakingSummary::<Test>::insert(
                new_domain_id,
                StakingSummary {
                    current_epoch_index: 0,
                    current_total_stake: 0,
                    current_operators: BTreeMap::new(),
                    next_operators: BTreeSet::new(),
                    current_epoch_rewards: BTreeMap::new(),
                },
            );

            // nominations will not work since the is frozen
            let nominator_account = 100;
            let nominator_stake = 100 * SSC;
            let res = Domains::nominate_operator(
                RuntimeOrigin::signed(nominator_account),
                operator_id,
                nominator_stake,
            );
            assert_err!(
                res,
                Error::<Test>::Staking(crate::staking::Error::OperatorNotRegistered)
            );
        });
    }

    type WithdrawWithResult = Vec<(Share, Result<(), StakingError>)>;

    /// Expected withdrawal amount.
    /// Bool indicates to include exisitential deposit while asserting the final balance
    /// since ED is not holded back from usable balance when there are no holds on the account.
    type ExpectedWithdrawAmount = Option<(BalanceOf<Test>, bool)>;

    /// The storage fund change in SSC, `true` means increase of the storage fund, `false` means decrease.
    type StorageFundChange = (bool, u32);

    pub(crate) type Share = <Test as Config>::Share;

    struct WithdrawParams {
        minimum_nominator_stake: BalanceOf<Test>,
        nominators: Vec<(NominatorId<Test>, BalanceOf<Test>)>,
        operator_reward: BalanceOf<Test>,
        nominator_id: NominatorId<Test>,
        withdraws: WithdrawWithResult,
        maybe_deposit: Option<BalanceOf<Test>>,
        expected_withdraw: ExpectedWithdrawAmount,
        expected_nominator_count_reduced_by: u32,
        storage_fund_change: StorageFundChange,
    }

    fn withdraw_stake(params: WithdrawParams) {
        let WithdrawParams {
            minimum_nominator_stake,
            nominators,
            operator_reward,
            nominator_id,
            withdraws,
            maybe_deposit,
            expected_withdraw,
            expected_nominator_count_reduced_by,
            storage_fund_change,
        } = params;
        let domain_id = DomainId::new(0);
        let operator_account = 0;
        let pair = OperatorPair::from_seed(&U256::from(0u32).into());
        let mut total_balance = nominators.iter().map(|n| n.1).sum::<BalanceOf<Test>>()
            + operator_reward
            + maybe_deposit.unwrap_or(0);

        let mut nominators = BTreeMap::from_iter(
            nominators
                .into_iter()
                .map(|(id, bal)| (id, (bal + ExistentialDeposit::get(), bal)))
                .collect::<Vec<(NominatorId<Test>, (BalanceOf<Test>, BalanceOf<Test>))>>(),
        );

        let mut ext = new_test_ext();
        ext.execute_with(|| {
            let (operator_free_balance, operator_stake) =
                nominators.remove(&operator_account).unwrap();
            let (operator_id, _) = register_operator(
                domain_id,
                operator_account,
                operator_free_balance,
                operator_stake,
                minimum_nominator_stake,
                pair.public(),
                nominators,
            );

            do_finalize_domain_current_epoch::<Test>(domain_id).unwrap();

            if !operator_reward.is_zero() {
                do_reward_operators::<Test>(
                    domain_id,
                    OperatorRewardSource::Dummy,
                    vec![operator_id].into_iter(),
                    operator_reward,
                )
                .unwrap();
            }

            let head_domain_number = HeadDomainNumber::<Test>::get(domain_id);
            let nominator_count = NominatorCount::<Test>::get(operator_id);

            if let Some(deposit_amount) = maybe_deposit {
                Balances::mint_into(&nominator_id, deposit_amount).unwrap();
                let res = Domains::nominate_operator(
                    RuntimeOrigin::signed(nominator_id),
                    operator_id,
                    deposit_amount,
                );
                assert_ok!(res);
            }

            let operator = Operators::<Test>::get(operator_id).unwrap();
            let (is_storage_fund_increased, storage_fund_change_amount) = storage_fund_change;
            if is_storage_fund_increased {
                bundle_storage_fund::refund_storage_fee::<Test>(
                    storage_fund_change_amount as u128 * SSC,
                    BTreeMap::from_iter([(operator_id, 1)]),
                )
                .unwrap();
                assert_eq!(
                    operator.total_storage_fee_deposit + storage_fund_change_amount as u128 * SSC,
                    bundle_storage_fund::total_balance::<Test>(operator_id)
                );
                total_balance += storage_fund_change_amount as u128 * SSC;
            } else {
                bundle_storage_fund::charge_bundle_storage_fee::<Test>(
                    operator_id,
                    storage_fund_change_amount,
                )
                .unwrap();
                assert_eq!(
                    operator.total_storage_fee_deposit - storage_fund_change_amount as u128 * SSC,
                    bundle_storage_fund::total_balance::<Test>(operator_id)
                );
                total_balance -= storage_fund_change_amount as u128 * SSC;
            }

            for (withdraw, expected_result) in withdraws {
                let withdraw_share_amount = STORAGE_FEE_RESERVE.left_from_one().mul_ceil(withdraw);
                let res = Domains::withdraw_stake(
                    RuntimeOrigin::signed(nominator_id),
                    operator_id,
                    WithdrawStake::Share(withdraw_share_amount),
                );
                assert_eq!(
                    res,
                    expected_result.map_err(|err| Error::<Test>::Staking(err).into())
                );
            }

            do_finalize_domain_current_epoch::<Test>(domain_id).unwrap();

            if let Some((withdraw, include_ed)) = expected_withdraw {
                let previous_usable_balance = Balances::usable_balance(nominator_id);

                // Update `HeadDomainNumber` to ensure unlock success
                HeadDomainNumber::<Test>::set(
                    domain_id,
                    head_domain_number
                        + <Test as crate::Config>::StakeWithdrawalLockingPeriod::get(),
                );
                assert_ok!(do_unlock_funds::<Test>(operator_id, nominator_id));

                let expected_balance = if include_ed {
                    total_balance += crate::tests::ExistentialDeposit::get();
                    previous_usable_balance + withdraw + crate::tests::ExistentialDeposit::get()
                } else {
                    previous_usable_balance + withdraw
                };

                assert_eq!(Balances::usable_balance(nominator_id), expected_balance);

                // ensure there are no withdrawals left
                assert!(Withdrawals::<Test>::get(operator_id, nominator_id).is_none());
            }

            let new_nominator_count = NominatorCount::<Test>::get(operator_id);
            assert_eq!(
                nominator_count - expected_nominator_count_reduced_by,
                new_nominator_count
            );

            // if the nominator count reduced, then there should be no storage for deposits as well
            if new_nominator_count < nominator_count {
                assert!(Deposits::<Test>::get(operator_id, nominator_id).is_none())
            }

            // The total balance is distributed in different places but never changed
            let operator = Operators::<Test>::get(operator_id).unwrap();
            assert_eq!(
                total_balance,
                Balances::usable_balance(nominator_id)
                    + operator.current_total_stake
                    + bundle_storage_fund::total_balance::<Test>(operator_id)
            );
        });
    }

    #[test]
    fn withdraw_stake_operator_all() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: 20 * SSC,
            nominator_id: 0,
            withdraws: vec![(150 * SSC, Err(StakingError::MinimumOperatorStake))],
            maybe_deposit: None,
            expected_withdraw: None,
            expected_nominator_count_reduced_by: 0,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_operator_below_minimum() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: 20 * SSC,
            nominator_id: 0,
            withdraws: vec![(65 * SSC, Err(StakingError::MinimumOperatorStake))],
            maybe_deposit: None,
            expected_withdraw: None,
            expected_nominator_count_reduced_by: 0,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_operator_below_minimum_no_rewards() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: Zero::zero(),
            nominator_id: 0,
            withdraws: vec![(51 * SSC, Err(StakingError::MinimumOperatorStake))],
            maybe_deposit: None,
            expected_withdraw: None,
            expected_nominator_count_reduced_by: 0,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_operator_above_minimum() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: 20 * SSC,
            nominator_id: 0,
            withdraws: vec![(58 * SSC, Ok(()))],
            // given the reward, operator will get 164.28 SSC
            // taking 58 shares will give this following approximate amount.
            maybe_deposit: None,
            expected_withdraw: Some((63523809519881179143, false)),
            expected_nominator_count_reduced_by: 0,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_operator_above_minimum_multiple_withdraws_error() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: 20 * SSC,
            nominator_id: 0,
            withdraws: vec![
                (58 * SSC, Ok(())),
                (5 * SSC, Err(StakingError::MinimumOperatorStake)),
            ],
            maybe_deposit: None,
            expected_withdraw: Some((63523809519881179143, false)),
            expected_nominator_count_reduced_by: 0,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_operator_above_minimum_multiple_withdraws() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: 20 * SSC,
            nominator_id: 0,
            withdraws: vec![(53 * SSC, Ok(())), (5 * SSC, Ok(()))],
            maybe_deposit: None,
            expected_withdraw: Some((63523809515796643053, false)),
            expected_nominator_count_reduced_by: 0,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_operator_above_minimum_no_rewards() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: Zero::zero(),
            nominator_id: 0,
            withdraws: vec![(49 * SSC, Ok(()))],
            maybe_deposit: None,
            expected_withdraw: Some((48999999980000000000, false)),
            expected_nominator_count_reduced_by: 0,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_operator_above_minimum_multiple_withdraws_no_rewards() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: Zero::zero(),
            nominator_id: 0,
            withdraws: vec![(29 * SSC, Ok(())), (20 * SSC, Ok(()))],
            maybe_deposit: None,
            expected_withdraw: Some((48999999986852892560, false)),
            expected_nominator_count_reduced_by: 0,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_operator_above_minimum_multiple_withdraws_no_rewards_with_errors() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: Zero::zero(),
            nominator_id: 0,
            withdraws: vec![
                (29 * SSC, Ok(())),
                (20 * SSC, Ok(())),
                (20 * SSC, Err(StakingError::MinimumOperatorStake)),
            ],
            maybe_deposit: None,
            expected_withdraw: Some((48999999986852892560, false)),
            expected_nominator_count_reduced_by: 0,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_nominator_below_minimum_with_rewards() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: 20 * SSC,
            nominator_id: 1,
            withdraws: vec![(45 * SSC, Ok(()))],
            // given nominator remaining stake goes below minimum
            // we withdraw everything, so for their 50 shares with reward,
            // price would be following
            maybe_deposit: None,
            expected_withdraw: Some((54761904775759637192, true)),
            expected_nominator_count_reduced_by: 1,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_nominator_below_minimum_with_rewards_multiple_withdraws() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: 20 * SSC,
            nominator_id: 1,
            withdraws: vec![(25 * SSC, Ok(())), (20 * SSC, Ok(()))],
            // given nominator remaining stake goes below minimum
            // we withdraw everything, so for their 50 shares with reward,
            // price would be following
            maybe_deposit: None,
            expected_withdraw: Some((54761904775759637192, true)),
            expected_nominator_count_reduced_by: 1,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_nominator_below_minimum_with_rewards_multiple_withdraws_with_errors() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: 20 * SSC,
            nominator_id: 1,
            withdraws: vec![
                (25 * SSC, Ok(())),
                (20 * SSC, Ok(())),
                (20 * SSC, Err(StakingError::InsufficientShares)),
            ],
            // given nominator remaining stake goes below minimum
            // we withdraw everything, so for their 50 shares with reward,
            // price would be following
            maybe_deposit: None,
            expected_withdraw: Some((54761904775759637192, true)),
            expected_nominator_count_reduced_by: 1,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_nominator_below_minimum_no_reward() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: Zero::zero(),
            nominator_id: 1,
            withdraws: vec![(45 * SSC, Ok(()))],
            maybe_deposit: None,
            expected_withdraw: Some((50 * SSC, true)),
            expected_nominator_count_reduced_by: 1,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_nominator_below_minimum_no_reward_multiple_rewards() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: Zero::zero(),
            nominator_id: 1,
            withdraws: vec![(25 * SSC, Ok(())), (20 * SSC, Ok(()))],
            maybe_deposit: None,
            expected_withdraw: Some((50 * SSC, true)),
            expected_nominator_count_reduced_by: 1,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_nominator_below_minimum_no_reward_multiple_rewards_with_errors() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: Zero::zero(),
            nominator_id: 1,
            withdraws: vec![
                (25 * SSC, Ok(())),
                (20 * SSC, Ok(())),
                (20 * SSC, Err(StakingError::InsufficientShares)),
            ],
            maybe_deposit: None,
            expected_withdraw: Some((50 * SSC, true)),
            expected_nominator_count_reduced_by: 1,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_nominator_above_minimum() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: 20 * SSC,
            nominator_id: 1,
            withdraws: vec![(40 * SSC, Ok(()))],
            maybe_deposit: None,
            expected_withdraw: Some((43809523820607709753, false)),
            expected_nominator_count_reduced_by: 0,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_nominator_above_minimum_multiple_withdraws() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: 20 * SSC,
            nominator_id: 1,
            withdraws: vec![(35 * SSC, Ok(())), (5 * SSC, Ok(()))],
            maybe_deposit: None,
            expected_withdraw: Some((43809523819607709753, false)),
            expected_nominator_count_reduced_by: 0,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_nominator_above_minimum_withdraw_all_multiple_withdraws_error() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: 20 * SSC,
            nominator_id: 1,
            withdraws: vec![
                (35 * SSC, Ok(())),
                (5 * SSC, Ok(())),
                (15 * SSC, Err(StakingError::InsufficientShares)),
            ],
            maybe_deposit: None,
            expected_withdraw: Some((43809523819607709753, false)),
            expected_nominator_count_reduced_by: 0,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_nominator_above_minimum_no_rewards() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: Zero::zero(),
            nominator_id: 1,
            withdraws: vec![(39 * SSC, Ok(()))],
            maybe_deposit: None,
            expected_withdraw: Some((39 * SSC, false)),
            expected_nominator_count_reduced_by: 0,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_nominator_above_minimum_no_rewards_multiple_withdraws() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: Zero::zero(),
            nominator_id: 1,
            withdraws: vec![(35 * SSC, Ok(())), (5 * SSC - 100000000000, Ok(()))],
            maybe_deposit: None,
            expected_withdraw: Some((39999999898000000000, false)),
            expected_nominator_count_reduced_by: 0,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_nominator_above_minimum_no_rewards_multiple_withdraws_with_errors() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: Zero::zero(),
            nominator_id: 1,
            withdraws: vec![
                (35 * SSC, Ok(())),
                (5 * SSC - 100000000000, Ok(())),
                (15 * SSC, Err(StakingError::InsufficientShares)),
            ],
            maybe_deposit: None,
            expected_withdraw: Some((39999999898000000000, false)),
            expected_nominator_count_reduced_by: 0,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_nominator_no_rewards_multiple_withdraws_with_error_min_nominator_stake() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: Zero::zero(),
            nominator_id: 1,
            withdraws: vec![
                (35 * SSC, Ok(())),
                (5 * SSC - 100000000000, Ok(())),
                (10 * SSC, Err(StakingError::MinimumNominatorStake)),
            ],
            maybe_deposit: Some(2 * SSC),
            expected_withdraw: Some((39999999898000000000, false)),
            expected_nominator_count_reduced_by: 0,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_nominator_with_rewards_multiple_withdraws_with_error_min_nominator_stake() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: 20 * SSC,
            nominator_id: 1,
            withdraws: vec![
                (35 * SSC, Ok(())),
                (5 * SSC, Ok(())),
                (10 * SSC, Err(StakingError::MinimumNominatorStake)),
            ],
            // given nominator remaining stake goes below minimum
            // we withdraw everything, so for their 50 shares with reward,
            // price would be following
            maybe_deposit: Some(2 * SSC),
            expected_withdraw: Some((43809523819607709753, false)),
            expected_nominator_count_reduced_by: 0,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_nominator_zero_amount() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: Zero::zero(),
            nominator_id: 1,
            withdraws: vec![(0, Err(StakingError::ZeroWithdraw))],
            maybe_deposit: None,
            expected_withdraw: None,
            expected_nominator_count_reduced_by: 0,
            storage_fund_change: (true, 0),
        })
    }

    #[test]
    fn withdraw_stake_nominator_all_with_storage_fee_profit() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: Zero::zero(),
            nominator_id: 1,
            withdraws: vec![(50 * SSC, Ok(()))],
            maybe_deposit: None,
            // The storage fund increased 50% (i.e. 21 * SSC) thus the nominator make 50%
            // storage fee profit i.e. 5 * SSC with rounding dust deducted
            storage_fund_change: (true, 21),
            expected_withdraw: Some((54999999994000000000, true)),
            expected_nominator_count_reduced_by: 1,
        })
    }

    #[test]
    fn withdraw_stake_nominator_all_with_storage_fee_loss() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: Zero::zero(),
            nominator_id: 1,
            withdraws: vec![(50 * SSC, Ok(()))],
            maybe_deposit: None,
            // The storage fund decreased 50% (i.e. 21 * SSC) thus the nominator loss 50%
            // storage fee deposit i.e. 5 * SSC with rounding dust deducted
            storage_fund_change: (false, 21),
            expected_withdraw: Some((44999999998000000000, true)),
            expected_nominator_count_reduced_by: 1,
        })
    }

    #[test]
    fn withdraw_stake_nominator_all_with_storage_fee_loss_all() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: Zero::zero(),
            nominator_id: 1,
            withdraws: vec![(50 * SSC, Ok(()))],
            maybe_deposit: None,
            // The storage fund decreased 100% (i.e. 42 * SSC) thus the nominator loss 100%
            // storage fee deposit i.e. 10 * SSC
            storage_fund_change: (false, 42),
            expected_withdraw: Some((40 * SSC, true)),
            expected_nominator_count_reduced_by: 1,
        })
    }

    #[test]
    fn withdraw_stake_nominator_multiple_withdraws_with_storage_fee_profit() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: Zero::zero(),
            nominator_id: 1,
            withdraws: vec![(5 * SSC, Ok(())), (10 * SSC, Ok(())), (15 * SSC, Ok(()))],
            maybe_deposit: None,
            // The storage fund increased 50% (i.e. 21 * SSC) thus the nominator make 50%
            // storage fee profit i.e. 5 * SSC with rounding dust deducted, withdraw 60% of
            // the stake and the storage fee profit
            storage_fund_change: (true, 21),
            expected_withdraw: Some((30 * SSC + 2999999855527204374, false)),
            expected_nominator_count_reduced_by: 0,
        })
    }

    #[test]
    fn withdraw_stake_nominator_multiple_withdraws_with_storage_fee_loss() {
        withdraw_stake(WithdrawParams {
            minimum_nominator_stake: 10 * SSC,
            nominators: vec![(0, 150 * SSC), (1, 50 * SSC), (2, 10 * SSC)],
            operator_reward: Zero::zero(),
            nominator_id: 1,
            withdraws: vec![(5 * SSC, Ok(())), (5 * SSC, Ok(())), (10 * SSC, Ok(()))],
            maybe_deposit: None,
            // The storage fund increased 50% (i.e. 21 * SSC) thus the nominator loss 50%
            // storage fee i.e. 5 * SSC with rounding dust deducted, withdraw 40% of
            // the stake and 40% of the storage fee loss are deducted
            storage_fund_change: (false, 21),
            expected_withdraw: Some((20 * SSC - 2 * SSC - 33331097576, false)),
            expected_nominator_count_reduced_by: 0,
        })
    }

    #[test]
    fn unlock_multiple_withdrawals() {
        let domain_id = DomainId::new(0);
        let operator_account = 1;
        let operator_free_balance = 250 * SSC;
        let operator_stake = 200 * SSC;
        let pair = OperatorPair::from_seed(&U256::from(0u32).into());
        let nominator_account = 2;
        let nominator_free_balance = 150 * SSC;
        let nominator_stake = 100 * SSC;

        let nominators = vec![
            (operator_account, (operator_free_balance, operator_stake)),
            (nominator_account, (nominator_free_balance, nominator_stake)),
        ];

        let total_deposit = 300 * SSC;
        let init_total_stake = STORAGE_FEE_RESERVE.left_from_one() * total_deposit;
        let init_total_storage_fund = STORAGE_FEE_RESERVE * total_deposit;

        let mut ext = new_test_ext();
        ext.execute_with(|| {
            let (operator_id, _) = register_operator(
                domain_id,
                operator_account,
                operator_free_balance,
                operator_stake,
                10 * SSC,
                pair.public(),
                BTreeMap::from_iter(nominators),
            );

            do_finalize_domain_current_epoch::<Test>(domain_id).unwrap();
            let domain_stake_summary = DomainStakingSummary::<Test>::get(domain_id).unwrap();
            assert_eq!(domain_stake_summary.current_total_stake, init_total_stake);

            let operator = Operators::<Test>::get(operator_id).unwrap();
            assert_eq!(operator.current_total_stake, init_total_stake);
            assert_eq!(operator.total_storage_fee_deposit, init_total_storage_fund);
            assert_eq!(
                operator.total_storage_fee_deposit,
                bundle_storage_fund::total_balance::<Test>(operator_id)
            );

            let amount_per_withdraw = init_total_stake / 100;
            let head_domain_number = HeadDomainNumber::<Test>::get(domain_id);

            // Request `WithdrawalLimit - 1` number of withdrawal
            for _ in 1..<Test as crate::Config>::WithdrawalLimit::get() {
                do_withdraw_stake::<Test>(
                    operator_id,
                    nominator_account,
                    WithdrawStake::Stake(amount_per_withdraw),
                )
                .unwrap();
                do_finalize_domain_current_epoch::<Test>(domain_id).unwrap();
            }
            // Increase the head domain number by 1
            HeadDomainNumber::<Test>::set(domain_id, head_domain_number + 1);

            // All withdrawals of a given nominator submitted in the same epoch will merge into one,
            // so we submit can submit as many as we want even though the withdrawal limit is met
            for _ in 0..5 {
                do_withdraw_stake::<Test>(
                    operator_id,
                    nominator_account,
                    WithdrawStake::Stake(amount_per_withdraw),
                )
                .unwrap();
            }
            do_finalize_domain_current_epoch::<Test>(domain_id).unwrap();

            // After the withdrawal limit is met, any new withdraw will be rejected in the next epoch
            assert_err!(
                do_withdraw_stake::<Test>(
                    operator_id,
                    nominator_account,
                    WithdrawStake::Stake(amount_per_withdraw),
                ),
                StakingError::TooManyWithdrawals
            );
            Withdrawals::<Test>::try_mutate(operator_id, nominator_account, |maybe_withdrawal| {
                let withdrawal = maybe_withdrawal.as_mut().unwrap();
                do_convert_previous_epoch_withdrawal::<Test>(operator_id, withdrawal).unwrap();
                assert_eq!(
                    withdrawal.withdrawals.len() as u32,
                    <Test as crate::Config>::WithdrawalLimit::get()
                );
                Ok::<(), StakingError>(())
            })
            .unwrap();

            // Make the first set of withdrawals pass the unlock period then unlock fund
            HeadDomainNumber::<Test>::set(
                domain_id,
                head_domain_number + <Test as crate::Config>::StakeWithdrawalLockingPeriod::get(),
            );
            let total_balance = Balances::usable_balance(nominator_account);
            assert_ok!(do_unlock_funds::<Test>(operator_id, nominator_account));
            assert_eq!(
                Balances::usable_balance(nominator_account) + 60246126106, // `60246126106` is a minior rounding dust
                total_balance
                    + (<Test as crate::Config>::WithdrawalLimit::get() as u128 - 1) * total_deposit
                        / 100
            );
            let withdrawal = Withdrawals::<Test>::get(operator_id, nominator_account).unwrap();
            assert_eq!(withdrawal.withdrawals.len(), 1);

            // Make the second set of withdrawals pass the unlock period then unlock funds
            HeadDomainNumber::<Test>::set(
                domain_id,
                head_domain_number
                    + <Test as crate::Config>::StakeWithdrawalLockingPeriod::get()
                    + 1,
            );
            let total_balance = Balances::usable_balance(nominator_account);
            assert_ok!(do_unlock_funds::<Test>(operator_id, nominator_account));
            assert_eq!(
                Balances::usable_balance(nominator_account) + 18473897451, // `18473897451` is a minor rounding dust
                total_balance + 5 * total_deposit / 100
            );
            assert!(Withdrawals::<Test>::get(operator_id, nominator_account).is_none());
        });
    }

    #[test]
    fn slash_operator() {
        let domain_id = DomainId::new(0);
        let operator_account = 1;
        let operator_free_balance = 250 * SSC;
        let operator_stake = 200 * SSC;
        let operator_extra_deposit = 40 * SSC;
        let pair = OperatorPair::from_seed(&U256::from(0u32).into());
        let nominator_account = 2;
        let nominator_free_balance = 150 * SSC;
        let nominator_stake = 100 * SSC;
        let nominator_extra_deposit = 40 * SSC;

        let nominators = vec![
            (operator_account, (operator_free_balance, operator_stake)),
            (nominator_account, (nominator_free_balance, nominator_stake)),
        ];

        let unlocking = vec![(operator_account, 10 * SSC), (nominator_account, 10 * SSC)];

        let deposits = vec![
            (operator_account, operator_extra_deposit),
            (nominator_account, nominator_extra_deposit),
        ];

        let init_total_stake = STORAGE_FEE_RESERVE.left_from_one() * 300 * SSC;
        let init_total_storage_fund = STORAGE_FEE_RESERVE * 300 * SSC;

        let mut ext = new_test_ext();
        ext.execute_with(|| {
            let (operator_id, _) = register_operator(
                domain_id,
                operator_account,
                operator_free_balance,
                operator_stake,
                10 * SSC,
                pair.public(),
                BTreeMap::from_iter(nominators),
            );

            do_finalize_domain_current_epoch::<Test>(domain_id).unwrap();
            let domain_stake_summary = DomainStakingSummary::<Test>::get(domain_id).unwrap();
            assert_eq!(domain_stake_summary.current_total_stake, init_total_stake);

            let operator = Operators::<Test>::get(operator_id).unwrap();
            assert_eq!(operator.current_total_stake, init_total_stake);
            assert_eq!(operator.total_storage_fee_deposit, init_total_storage_fund);
            assert_eq!(
                operator.total_storage_fee_deposit,
                bundle_storage_fund::total_balance::<Test>(operator_id)
            );

            for unlock in &unlocking {
                do_withdraw_stake::<Test>(operator_id, unlock.0, WithdrawStake::Share(unlock.1))
                    .unwrap();
            }

            do_reward_operators::<Test>(
                domain_id,
                OperatorRewardSource::Dummy,
                vec![operator_id].into_iter(),
                20 * SSC,
            )
            .unwrap();
            do_finalize_domain_current_epoch::<Test>(domain_id).unwrap();

            // Manually convert previous withdrawal in share to balance
            for id in [operator_account, nominator_account] {
                Withdrawals::<Test>::try_mutate(operator_id, id, |maybe_withdrawal| {
                    do_convert_previous_epoch_withdrawal::<Test>(
                        operator_id,
                        maybe_withdrawal.as_mut().unwrap(),
                    )
                })
                .unwrap();
            }

            // post epoch transition, domain stake has 21.666 amount reduced and storage fund has 5 amount reduced
            // due to withdrawal of 20 shares
            let operator = Operators::<Test>::get(operator_id).unwrap();
            let domain_stake_summary = DomainStakingSummary::<Test>::get(domain_id).unwrap();
            let operator_withdrawal =
                Withdrawals::<Test>::get(operator_id, operator_account).unwrap();
            let nominator_withdrawal =
                Withdrawals::<Test>::get(operator_id, nominator_account).unwrap();

            let total_deposit =
                domain_stake_summary.current_total_stake + operator.total_storage_fee_deposit;
            let total_stake_withdrawal = operator_withdrawal.total_withdrawal_amount
                + nominator_withdrawal.total_withdrawal_amount;
            let total_storage_fee_withdrawal = operator_withdrawal.withdrawals[0]
                .storage_fee_refund
                + nominator_withdrawal.withdrawals[0].storage_fee_refund;
            assert_eq!(293333333331527777778, total_deposit,);
            assert_eq!(21666666668472222222, total_stake_withdrawal);
            assert_eq!(5000000000000000000, total_storage_fee_withdrawal);
            assert_eq!(
                320 * SSC,
                total_deposit + total_stake_withdrawal + total_storage_fee_withdrawal
            );
            assert_eq!(
                operator.total_storage_fee_deposit,
                bundle_storage_fund::total_balance::<Test>(operator_id)
            );

            for deposit in deposits {
                do_nominate_operator::<Test>(operator_id, deposit.0, deposit.1).unwrap();
            }

            do_mark_operators_as_slashed::<Test>(
                vec![operator_id],
                SlashedReason::InvalidBundle(1),
            )
            .unwrap();

            let domain_stake_summary = DomainStakingSummary::<Test>::get(domain_id).unwrap();
            assert!(!domain_stake_summary.next_operators.contains(&operator_id));

            let operator = Operators::<Test>::get(operator_id).unwrap();
            assert_eq!(
                *operator.status::<Test>(operator_id),
                OperatorStatus::Slashed
            );

            let pending_slashes = PendingSlashes::<Test>::get(domain_id).unwrap();
            assert!(pending_slashes.contains(&operator_id));

            assert_eq!(
                Balances::total_balance(&crate::tests::TreasuryAccount::get()),
                0
            );

            do_slash_operator::<Test>(domain_id, MAX_NOMINATORS_TO_SLASH).unwrap();
            assert_eq!(PendingSlashes::<Test>::get(domain_id), None);
            assert_eq!(Operators::<Test>::get(operator_id), None);
            assert_eq!(OperatorIdOwner::<Test>::get(operator_id), None);

            assert_eq!(
                Balances::total_balance(&operator_account),
                operator_free_balance - operator_stake
            );
            assert_eq!(
                Balances::total_balance(&nominator_account),
                nominator_free_balance - nominator_stake
            );

            assert!(Balances::total_balance(&crate::tests::TreasuryAccount::get()) >= 320 * SSC);
            assert_eq!(bundle_storage_fund::total_balance::<Test>(operator_id), 0);
        });
    }

    #[test]
    fn slash_operator_with_more_than_max_nominators_to_slash() {
        let domain_id = DomainId::new(0);
        let operator_account = 1;
        let operator_free_balance = 250 * SSC;
        let operator_stake = 200 * SSC;
        let operator_extra_deposit = 40 * SSC;
        let pair = OperatorPair::from_seed(&U256::from(0u32).into());

        let nominator_accounts: Vec<crate::tests::AccountId> = (2..22).collect();
        let nominator_free_balance = 150 * SSC;
        let nominator_stake = 100 * SSC;
        let nominator_extra_deposit = 40 * SSC;

        let mut nominators = vec![(operator_account, (operator_free_balance, operator_stake))];
        for nominator_account in nominator_accounts.clone() {
            nominators.push((nominator_account, (nominator_free_balance, nominator_stake)))
        }

        let last_nominator_account = nominator_accounts.last().cloned().unwrap();
        let unlocking = vec![
            (operator_account, 10 * SSC),
            (last_nominator_account, 10 * SSC),
        ];

        let deposits = vec![
            (operator_account, operator_extra_deposit),
            (last_nominator_account, nominator_extra_deposit),
        ];

        let init_total_stake = STORAGE_FEE_RESERVE.left_from_one()
            * (200 + (100 * nominator_accounts.len() as u128))
            * SSC;
        let init_total_storage_fund =
            STORAGE_FEE_RESERVE * (200 + (100 * nominator_accounts.len() as u128)) * SSC;

        let mut ext = new_test_ext();
        ext.execute_with(|| {
            let (operator_id, _) = register_operator(
                domain_id,
                operator_account,
                operator_free_balance,
                operator_stake,
                10 * SSC,
                pair.public(),
                BTreeMap::from_iter(nominators),
            );

            do_finalize_domain_current_epoch::<Test>(domain_id).unwrap();
            let domain_stake_summary = DomainStakingSummary::<Test>::get(domain_id).unwrap();
            assert_eq!(domain_stake_summary.current_total_stake, init_total_stake);

            let operator = Operators::<Test>::get(operator_id).unwrap();
            assert_eq!(operator.current_total_stake, init_total_stake);
            assert_eq!(operator.total_storage_fee_deposit, init_total_storage_fund);
            assert_eq!(
                operator.total_storage_fee_deposit,
                bundle_storage_fund::total_balance::<Test>(operator_id)
            );

            for unlock in &unlocking {
                do_withdraw_stake::<Test>(operator_id, unlock.0, WithdrawStake::Share(unlock.1))
                    .unwrap();
            }

            do_reward_operators::<Test>(
                domain_id,
                OperatorRewardSource::Dummy,
                vec![operator_id].into_iter(),
                20 * SSC,
            )
            .unwrap();
            do_finalize_domain_current_epoch::<Test>(domain_id).unwrap();

            // Manually convert previous withdrawal in share to balance
            for id in [operator_account, last_nominator_account] {
                Withdrawals::<Test>::try_mutate(operator_id, id, |maybe_withdrawal| {
                    do_convert_previous_epoch_withdrawal::<Test>(
                        operator_id,
                        maybe_withdrawal.as_mut().unwrap(),
                    )
                })
                .unwrap();
            }

            // post epoch transition, domain stake has 21.666 amount reduced and storage fund has 5 amount reduced
            // due to withdrawal of 20 shares
            let operator = Operators::<Test>::get(operator_id).unwrap();
            let domain_stake_summary = DomainStakingSummary::<Test>::get(domain_id).unwrap();
            let operator_withdrawal =
                Withdrawals::<Test>::get(operator_id, operator_account).unwrap();
            let nominator_withdrawal =
                Withdrawals::<Test>::get(operator_id, last_nominator_account).unwrap();

            let total_deposit =
                domain_stake_summary.current_total_stake + operator.total_storage_fee_deposit;
            let total_stake_withdrawal = operator_withdrawal.total_withdrawal_amount
                + nominator_withdrawal.total_withdrawal_amount;
            let total_storage_fee_withdrawal = operator_withdrawal.withdrawals[0]
                .storage_fee_refund
                + nominator_withdrawal.withdrawals[0].storage_fee_refund;
            assert_eq!(2194772727253419421470, total_deposit,);
            assert_eq!(20227272746580578530, total_stake_withdrawal);
            assert_eq!(5000000000000000000, total_storage_fee_withdrawal);
            assert_eq!(
                2220 * SSC,
                total_deposit + total_stake_withdrawal + total_storage_fee_withdrawal
            );

            assert_eq!(
                operator.total_storage_fee_deposit,
                bundle_storage_fund::total_balance::<Test>(operator_id)
            );

            for deposit in deposits {
                do_nominate_operator::<Test>(operator_id, deposit.0, deposit.1).unwrap();
            }

            do_mark_operators_as_slashed::<Test>(
                vec![operator_id],
                SlashedReason::InvalidBundle(1),
            )
            .unwrap();

            let domain_stake_summary = DomainStakingSummary::<Test>::get(domain_id).unwrap();
            assert!(!domain_stake_summary.next_operators.contains(&operator_id));

            let operator = Operators::<Test>::get(operator_id).unwrap();
            assert_eq!(
                *operator.status::<Test>(operator_id),
                OperatorStatus::Slashed
            );

            let pending_slashes = PendingSlashes::<Test>::get(domain_id).unwrap();
            assert!(pending_slashes.contains(&operator_id));

            assert_eq!(
                Balances::total_balance(&crate::tests::TreasuryAccount::get()),
                0
            );

            // since we only slash 10 nominators a time but we have a total of 21 nominators,
            // do 3 iterations
            do_slash_operator::<Test>(domain_id, MAX_NOMINATORS_TO_SLASH).unwrap();
            do_slash_operator::<Test>(domain_id, MAX_NOMINATORS_TO_SLASH).unwrap();
            do_slash_operator::<Test>(domain_id, MAX_NOMINATORS_TO_SLASH).unwrap();

            assert_eq!(PendingSlashes::<Test>::get(domain_id), None);
            assert_eq!(Operators::<Test>::get(operator_id), None);
            assert_eq!(OperatorIdOwner::<Test>::get(operator_id), None);

            assert_eq!(
                Balances::total_balance(&operator_account),
                operator_free_balance - operator_stake
            );
            for nominator_account in nominator_accounts {
                assert_eq!(
                    Balances::total_balance(&nominator_account),
                    nominator_free_balance - nominator_stake
                );
            }

            assert!(Balances::total_balance(&crate::tests::TreasuryAccount::get()) >= 2220 * SSC);
            assert_eq!(bundle_storage_fund::total_balance::<Test>(operator_id), 0);
        });
    }

    #[test]
    fn slash_operators() {
        let domain_id = DomainId::new(0);
        let operator_free_balance = 250 * SSC;
        let operator_stake = 200 * SSC;

        let operator_account_1 = 1;
        let operator_account_2 = 2;
        let operator_account_3 = 3;

        let pair_1 = OperatorPair::from_seed(&U256::from(0u32).into());
        let pair_2 = OperatorPair::from_seed(&U256::from(1u32).into());
        let pair_3 = OperatorPair::from_seed(&U256::from(2u32).into());

        let mut ext = new_test_ext();
        ext.execute_with(|| {
            let (operator_id_1, _) = register_operator(
                domain_id,
                operator_account_1,
                operator_free_balance,
                operator_stake,
                10 * SSC,
                pair_1.public(),
                Default::default(),
            );

            let (operator_id_2, _) = register_operator(
                domain_id,
                operator_account_2,
                operator_free_balance,
                operator_stake,
                10 * SSC,
                pair_2.public(),
                Default::default(),
            );

            let (operator_id_3, _) = register_operator(
                domain_id,
                operator_account_3,
                operator_free_balance,
                operator_stake,
                10 * SSC,
                pair_3.public(),
                Default::default(),
            );

            do_finalize_domain_current_epoch::<Test>(domain_id).unwrap();
            let domain_stake_summary = DomainStakingSummary::<Test>::get(domain_id).unwrap();
            assert!(domain_stake_summary.next_operators.contains(&operator_id_1));
            assert!(domain_stake_summary.next_operators.contains(&operator_id_2));
            assert!(domain_stake_summary.next_operators.contains(&operator_id_3));
            assert_eq!(
                domain_stake_summary.current_total_stake,
                STORAGE_FEE_RESERVE.left_from_one() * 600 * SSC
            );
            for operator_id in [operator_id_1, operator_id_2, operator_id_3] {
                let operator = Operators::<Test>::get(operator_id).unwrap();
                assert_eq!(
                    operator.total_storage_fee_deposit,
                    STORAGE_FEE_RESERVE * operator_stake
                );
                assert_eq!(
                    operator.total_storage_fee_deposit,
                    bundle_storage_fund::total_balance::<Test>(operator_id)
                );
            }

            do_mark_operators_as_slashed::<Test>(
                vec![operator_id_1],
                SlashedReason::InvalidBundle(1),
            )
            .unwrap();
            do_mark_operators_as_slashed::<Test>(
                vec![operator_id_2],
                SlashedReason::InvalidBundle(2),
            )
            .unwrap();
            do_mark_operators_as_slashed::<Test>(
                vec![operator_id_3],
                SlashedReason::InvalidBundle(3),
            )
            .unwrap();

            let domain_stake_summary = DomainStakingSummary::<Test>::get(domain_id).unwrap();
            assert!(!domain_stake_summary.next_operators.contains(&operator_id_1));
            assert!(!domain_stake_summary.next_operators.contains(&operator_id_2));
            assert!(!domain_stake_summary.next_operators.contains(&operator_id_3));

            let operator = Operators::<Test>::get(operator_id_1).unwrap();
            assert_eq!(
                *operator.status::<Test>(operator_id_1),
                OperatorStatus::Slashed
            );

            let operator = Operators::<Test>::get(operator_id_2).unwrap();
            assert_eq!(
                *operator.status::<Test>(operator_id_2),
                OperatorStatus::Slashed
            );

            let operator = Operators::<Test>::get(operator_id_3).unwrap();
            assert_eq!(
                *operator.status::<Test>(operator_id_3),
                OperatorStatus::Slashed
            );

            assert_eq!(
                Balances::total_balance(&crate::tests::TreasuryAccount::get()),
                0
            );

            let slashed_operators = PendingSlashes::<Test>::get(domain_id).unwrap();
            slashed_operators.into_iter().for_each(|_| {
                do_slash_operator::<Test>(domain_id, MAX_NOMINATORS_TO_SLASH).unwrap();
            });

            assert_eq!(PendingSlashes::<Test>::get(domain_id), None);
            assert_eq!(Operators::<Test>::get(operator_id_1), None);
            assert_eq!(OperatorIdOwner::<Test>::get(operator_id_1), None);
            assert_eq!(Operators::<Test>::get(operator_id_2), None);
            assert_eq!(OperatorIdOwner::<Test>::get(operator_id_2), None);
            assert_eq!(Operators::<Test>::get(operator_id_3), None);
            assert_eq!(OperatorIdOwner::<Test>::get(operator_id_3), None);

            assert_eq!(
                Balances::total_balance(&crate::tests::TreasuryAccount::get()),
                600 * SSC
            );
            for operator_id in [operator_id_1, operator_id_2, operator_id_3] {
                assert_eq!(bundle_storage_fund::total_balance::<Test>(operator_id), 0);
            }
        });
    }

    #[test]
    fn bundle_storage_fund_charged_and_refund_storege_fee() {
        let domain_id = DomainId::new(0);
        let operator_account = 1;
        let operator_free_balance = 150 * SSC;
        let operator_total_stake = 100 * SSC;
        let operator_stake = 80 * SSC;
        let operator_storage_fee_deposit = 20 * SSC;
        let pair = OperatorPair::from_seed(&U256::from(0u32).into());
        let nominator_account = 2;

        let mut ext = new_test_ext();
        ext.execute_with(|| {
            let (operator_id, _) = register_operator(
                domain_id,
                operator_account,
                operator_free_balance,
                operator_total_stake,
                SSC,
                pair.public(),
                BTreeMap::default(),
            );

            let domain_staking_summary = DomainStakingSummary::<Test>::get(domain_id).unwrap();
            assert_eq!(domain_staking_summary.current_total_stake, operator_stake);

            let operator = Operators::<Test>::get(operator_id).unwrap();
            assert_eq!(operator.current_total_stake, operator_stake);
            assert_eq!(operator.current_total_shares, operator_stake);
            assert_eq!(
                operator.total_storage_fee_deposit,
                operator_storage_fee_deposit
            );

            // Drain the bundle storage fund
            bundle_storage_fund::charge_bundle_storage_fee::<Test>(
                operator_id,
                // the transaction fee is one SSC per byte thus div SSC here
                (operator_storage_fee_deposit / SSC) as u32,
            )
            .unwrap();
            assert_eq!(bundle_storage_fund::total_balance::<Test>(operator_id), 0);
            assert_err!(
                bundle_storage_fund::charge_bundle_storage_fee::<Test>(operator_id, 1,),
                bundle_storage_fund::Error::BundleStorageFeePayment
            );

            // The operator add more stake thus add deposit to the bundle storage fund
            do_nominate_operator::<Test>(operator_id, operator_account, 5 * SSC).unwrap();
            assert_eq!(bundle_storage_fund::total_balance::<Test>(operator_id), SSC);

            bundle_storage_fund::charge_bundle_storage_fee::<Test>(operator_id, 1).unwrap();
            assert_eq!(bundle_storage_fund::total_balance::<Test>(operator_id), 0);

            // New nominator add deposit to the bundle storage fund
            Balances::set_balance(&nominator_account, 100 * SSC);
            do_nominate_operator::<Test>(operator_id, nominator_account, 5 * SSC).unwrap();
            assert_eq!(bundle_storage_fund::total_balance::<Test>(operator_id), SSC);

            bundle_storage_fund::charge_bundle_storage_fee::<Test>(operator_id, 1).unwrap();
            assert_eq!(bundle_storage_fund::total_balance::<Test>(operator_id), 0);

            // Refund of the storage fee add deposit to the bundle storage fund
            bundle_storage_fund::refund_storage_fee::<Test>(
                10 * SSC,
                BTreeMap::from_iter([(operator_id, 1), (operator_id + 1, 9)]),
            )
            .unwrap();
            assert_eq!(bundle_storage_fund::total_balance::<Test>(operator_id), SSC);

            // The operator `operator_id + 1` not exist thus the refund storage fee added to treasury
            assert_eq!(
                Balances::total_balance(&crate::tests::TreasuryAccount::get()),
                9 * SSC
            );

            bundle_storage_fund::charge_bundle_storage_fee::<Test>(operator_id, 1).unwrap();
            assert_eq!(bundle_storage_fund::total_balance::<Test>(operator_id), 0);
        });
    }
}
