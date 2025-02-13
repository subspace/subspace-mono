(function() {
    var implementors = Object.fromEntries([["domain_eth_service",[["impl&lt;Block, RuntimeApi, CT, EC&gt; BlockImportProvider&lt;Block, Client&lt;Backend&lt;Block&gt;, LocalCallExecutor&lt;Block, Backend&lt;Block&gt;, WasmExecutor&lt;(HostFunctions, (HostFunctions, HostFunctions, HostFunctions, HostFunctions, HostFunctions, HostFunctions, HostFunctions, HostFunctions, HostFunctions, HostFunctions, HostFunctions, HostFunctions, HostFunctions), HostFunctions, HostFunctions)&gt;&gt;, Block, RuntimeApi&gt;&gt; for <a class=\"struct\" href=\"domain_eth_service/provider/struct.EthProvider.html\" title=\"struct domain_eth_service::provider::EthProvider\">EthProvider</a>&lt;CT, EC&gt;<div class=\"where\">where\n    Block: BlockT,\n    RuntimeApi: ConstructRuntimeApi&lt;Block, FullClient&lt;Block, RuntimeApi&gt;&gt; + <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a> + 'static,\n    RuntimeApi::RuntimeApi: ApiExt&lt;Block&gt; + Core&lt;Block&gt; + BlockBuilder&lt;Block&gt; + EthereumRuntimeRPCApi&lt;Block&gt;,</div>"]]],["domain_service",[]]]);
    if (window.register_implementors) {
        window.register_implementors(implementors);
    } else {
        window.pending_implementors = implementors;
    }
})()
//{"start":57,"fragment_lengths":[1203,22]}