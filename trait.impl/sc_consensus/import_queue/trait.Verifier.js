(function() {
    var implementors = Object.fromEntries([["domain_client_consensus_relay_chain",[["impl&lt;Block&gt; Verifier&lt;Block&gt; for <a class=\"struct\" href=\"domain_client_consensus_relay_chain/struct.Verifier.html\" title=\"struct domain_client_consensus_relay_chain::Verifier\">Verifier</a>&lt;Block&gt;<div class=\"where\">where\n    Block: BlockT,</div>"]]],["sc_consensus_subspace",[["impl&lt;PosTable, Block, Client&gt; Verifier&lt;Block&gt; for <a class=\"struct\" href=\"sc_consensus_subspace/verifier/struct.SubspaceVerifier.html\" title=\"struct sc_consensus_subspace::verifier::SubspaceVerifier\">SubspaceVerifier</a>&lt;PosTable, Block, Client&gt;<div class=\"where\">where\n    PosTable: Table,\n    Block: BlockT,\n    BlockNumber: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;NumberFor&lt;Block&gt;&gt;,\n    Client: HeaderBackend&lt;Block&gt; + ProvideRuntimeApi&lt;Block&gt; + AuxStore + 'static,\n    Client::Api: BlockBuilderApi&lt;Block&gt; + SubspaceApi&lt;Block, PublicKey&gt;,</div>"]]]]);
    if (window.register_implementors) {
        window.register_implementors(implementors);
    } else {
        window.pending_implementors = implementors;
    }
})()
//{"start":57,"fragment_lengths":[316,742]}