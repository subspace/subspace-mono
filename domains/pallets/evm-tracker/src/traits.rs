//! Traits that make programming against the runtime easier.

use frame_system::pallet_prelude::OriginFor;

/// Type alias for the runtime's account ID.
pub type AccountIdFor<Runtime> = <Runtime as frame_system::Config>::AccountId;

/// Trait used to convert from a generated `RuntimeCall` type to `pallet_ethereum::Call<Runtime>`.
pub trait MaybeIntoEthCall<Runtime>
where
    Runtime: frame_system::Config + pallet_ethereum::Config,
    Result<pallet_ethereum::RawOrigin, OriginFor<Runtime>>: From<OriginFor<Runtime>>,
{
    /// If this call is a `pallet_ethereum::Call<Runtime>` call, returns the inner call.
    fn maybe_into_eth_call(&self) -> Option<&pallet_ethereum::Call<Runtime>>;
}

/// Trait used to convert from a generated `RuntimeCall` type to `pallet_evm::Call<Runtime>`.
pub trait MaybeIntoEvmCall<Runtime>
where
    Runtime: pallet_evm::Config,
{
    /// If this call is a `pallet_evm::Call<Runtime>` call, returns the inner call.
    fn maybe_into_evm_call(&self) -> Option<&pallet_evm::Call<Runtime>>;
}
