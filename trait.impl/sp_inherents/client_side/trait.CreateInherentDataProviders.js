(function() {
    var implementors = Object.fromEntries([["domain_block_preprocessor",[["impl&lt;CClient, CBlock, Block&gt; CreateInherentDataProviders&lt;Block, <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.unit.html\">()</a>&gt; for <a class=\"struct\" href=\"domain_block_preprocessor/inherents/struct.CreateInherentDataProvider.html\" title=\"struct domain_block_preprocessor::inherents::CreateInherentDataProvider\">CreateInherentDataProvider</a>&lt;CClient, CBlock&gt;<div class=\"where\">where\n    Block: BlockT,\n    CBlock: BlockT,\n    CClient: ProvideRuntimeApi&lt;CBlock&gt; + HeaderBackend&lt;CBlock&gt;,\n    CClient::Api: DomainsApi&lt;CBlock, Block::Header&gt; + MessengerApi&lt;CBlock, NumberFor&lt;CBlock&gt;, CBlock::Hash&gt;,</div>"]]]]);
    if (window.register_implementors) {
        window.register_implementors(implementors);
    } else {
        window.pending_implementors = implementors;
    }
})()
//{"start":57,"fragment_lengths":[731]}