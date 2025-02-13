(function() {
    var type_impls = Object.fromEntries([["subspace_fake_runtime_api",[]],["subspace_runtime",[]],["subspace_test_runtime",[]]]);
    if (window.register_type_impls) {
        window.register_type_impls(type_impls);
    } else {
        window.pending_type_impls = type_impls;
    }
})()
//{"start":55,"fragment_lengths":[32,24,29]}