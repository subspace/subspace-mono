(function() {
    var implementors = Object.fromEntries([["sc_domains",[["impl&lt;CClient, CBlock, Block, Executor&gt; ExtensionsFactory&lt;Block&gt; for <a class=\"struct\" href=\"sc_domains/struct.ExtensionsFactory.html\" title=\"struct sc_domains::ExtensionsFactory\">ExtensionsFactory</a>&lt;CClient, CBlock, Block, Executor&gt;<div class=\"where\">where\n    Block: BlockT,\n    CBlock: BlockT,\n    CBlock::Hash: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;H256&gt; + <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;H256&gt;,\n    CClient: HeaderBackend&lt;CBlock&gt; + ProvideRuntimeApi&lt;CBlock&gt; + 'static,\n    CClient::Api: MmrApi&lt;CBlock, H256, NumberFor&lt;CBlock&gt;&gt; + MessengerApi&lt;CBlock, NumberFor&lt;CBlock&gt;, CBlock::Hash&gt; + DomainsApi&lt;CBlock, Block::Header&gt;,\n    Executor: CodeExecutor + RuntimeVersionOf,</div>"]]]]);
    if (window.register_implementors) {
        window.register_implementors(implementors);
    } else {
        window.pending_implementors = implementors;
    }
})()
//{"start":57,"fragment_lengths":[987]}