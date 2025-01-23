//! Runtime primitives for pallet-utility.

use scale_info::prelude::collections::VecDeque;

/// Type alias used to get the runtime call type.
pub type RuntimeCallFor<Runtime> = <Runtime as pallet_utility::Config>::RuntimeCall;

/// Trait used to convert from a specific `RuntimeCall` type to `pallet-utility::Call<Runtime>`.
pub trait MaybeIntoUtilityCall<Runtime>
where
    Runtime: pallet_utility::Config,
{
    /// If this call is a `pallet_utility::Call<Runtime>` call, returns the inner call.
    fn maybe_into_utility_call(&self) -> Option<&pallet_utility::Call<Runtime>>;
}

/// Returns an interator over `call`, and any calls nested within it using `pallet-utility`.
///
/// The iterator yields all calls in depth-first order, including calls which contain other calls.
/// `pallet_utility::Call::__Ignore` calls are yielded themsevles, but their contents are not.
// This function doesn't use stack recursion, so there's no need to check the recursion depth.
pub fn nested_utility_call_iter<Runtime>(
    call: &RuntimeCallFor<Runtime>,
) -> impl Iterator<Item = &RuntimeCallFor<Runtime>>
where
    Runtime: pallet_utility::Config,
    RuntimeCallFor<Runtime>: MaybeIntoUtilityCall<Runtime>,
{
    // Instead of using recursion, we allocate references to each call on the heap.
    // TODO: re-use the same memory with an enum for a call ref, a boxed call, or a vec of calls
    let mut new_calls = VecDeque::from([call]);

    core::iter::from_fn(move || {
        let call = new_calls.pop_front()?;

        if let Some(call) = call.maybe_into_utility_call() {
            match call {
                pallet_utility::Call::batch { calls }
                | pallet_utility::Call::batch_all { calls }
                | pallet_utility::Call::force_batch { calls } => calls.iter().for_each(|call| {
                    new_calls.push_front(call);
                }),
                pallet_utility::Call::as_derivative { call, .. }
                | pallet_utility::Call::dispatch_as { call, .. }
                | pallet_utility::Call::with_weight { call, .. } => {
                    new_calls.push_front(call.as_ref())
                }
                pallet_utility::Call::__Ignore(..) => {}
            }
        }

        Some(call)
    })
}
