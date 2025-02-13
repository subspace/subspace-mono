(function() {
    var type_impls = Object.fromEntries([["domain_service",[]],["domain_test_service",[]],["subspace_service",[]],["subspace_test_client",[]]]);
    if (window.register_type_impls) {
        window.register_type_impls(type_impls);
    } else {
        window.pending_type_impls = type_impls;
    }
})()
//{"start":55,"fragment_lengths":[21,27,24,28]}