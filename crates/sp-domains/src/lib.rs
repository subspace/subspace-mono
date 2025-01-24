//! Primitives for domains pallet.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod bundle_producer_election;
pub mod core_api;
pub mod extrinsics;
pub mod merkle_tree;
pub mod proof_provider_and_verifier;
pub mod storage;
#[cfg(any(test, feature = "test-ethereum"))]
pub mod test_ethereum;
#[cfg(any(test, feature = "test-ethereum"))]
pub mod test_ethereum_tx;
#[cfg(test)]
mod tests;
pub mod valued_trie;

#[cfg(not(feature = "std"))]
extern crate alloc;

use crate::storage::{RawGenesis, StorageKey};
#[cfg(not(feature = "std"))]
use alloc::collections::BTreeSet;
#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use bundle_producer_election::{BundleProducerElectionParams, ProofOfElectionError};
use core::num::ParseIntError;
use core::ops::{Add, Sub};
use core::str::FromStr;
use domain_runtime_primitives::{EthereumAccountId, MultiAccountId};
use frame_support::storage::storage_prefix;
use frame_support::{Blake2_128Concat, StorageHasher};
use hexlit::hex;
use parity_scale_codec::{Codec, Decode, Encode, MaxEncodedLen};
use scale_info::TypeInfo;
use serde::{Deserialize, Serialize};
use sp_core::crypto::KeyTypeId;
use sp_core::sr25519::vrf::VrfSignature;
#[cfg(any(feature = "std", feature = "runtime-benchmarks"))]
use sp_core::sr25519::vrf::{VrfPreOutput, VrfProof};
use sp_core::H256;
use sp_runtime::generic::OpaqueDigestItemId;
use sp_runtime::traits::{
    BlakeTwo256, Block as BlockT, CheckedAdd, Hash as HashT, Header as HeaderT, NumberFor, Zero,
};
use sp_runtime::{Digest, DigestItem, OpaqueExtrinsic, Percent};
use sp_runtime_interface::pass_by;
use sp_runtime_interface::pass_by::PassBy;
use sp_std::collections::btree_map::BTreeMap;
use sp_std::fmt::{Display, Formatter};
use sp_trie::TrieLayout;
use sp_version::RuntimeVersion;
use sp_weights::Weight;
#[cfg(feature = "std")]
use std::collections::BTreeSet;
use subspace_core_primitives::hashes::{blake3_hash, Blake3Hash};
use subspace_core_primitives::pot::PotOutput;
use subspace_core_primitives::solutions::bidirectional_distance;
use subspace_core_primitives::{Randomness, U256};
use subspace_runtime_primitives::{Balance, Moment};

/// Key type for Operator.
pub const KEY_TYPE: KeyTypeId = KeyTypeId(*b"oper");

/// Extrinsics shuffling seed
pub const DOMAIN_EXTRINSICS_SHUFFLING_SEED_SUBJECT: &[u8] = b"extrinsics-shuffling-seed";

mod app {
    use super::KEY_TYPE;
    use sp_application_crypto::{app_crypto, sr25519};

    app_crypto!(sr25519, KEY_TYPE);
}

// TODO: this runtime constant is not support to update, see https://github.com/autonomys/subspace/issues/2712
// for more detail about the problem and what we need to do to support it.
//
// The domain storage fee multiplier used to charge a higher storage fee to the domain
// transaction to even out the duplicated/illegal domain transaction storage cost, which
// can not be eliminated right now.
pub const DOMAIN_STORAGE_FEE_MULTIPLIER: Balance = 3;

/// An operator authority signature.
pub type OperatorSignature = app::Signature;

/// An operator authority keypair. Necessarily equivalent to the schnorrkel public key used in
/// the main executor module. If that ever changes, then this must, too.
#[cfg(feature = "std")]
pub type OperatorPair = app::Pair;

/// An operator authority identifier.
pub type OperatorPublicKey = app::Public;

/// A type that implements `BoundToRuntimeAppPublic`, used for operator signing key.
pub struct OperatorKey;

impl sp_runtime::BoundToRuntimeAppPublic for OperatorKey {
    type Public = OperatorPublicKey;
}

/// Stake weight in the domain bundle election.
///
/// Derived from the Balance and can't be smaller than u128.
pub type StakeWeight = u128;

/// The Trie root of all extrinsics included in a bundle.
pub type ExtrinsicsRoot = H256;

/// Type alias for Header Hashing.
pub type HeaderHashingFor<Header> = <Header as HeaderT>::Hashing;
/// Type alias for Header number.
pub type HeaderNumberFor<Header> = <Header as HeaderT>::Number;
/// Type alias for Header hash.
pub type HeaderHashFor<Header> = <Header as HeaderT>::Hash;

/// Unique identifier of a domain.
#[derive(
    Clone,
    Copy,
    Debug,
    Hash,
    Default,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Encode,
    Decode,
    TypeInfo,
    Serialize,
    Deserialize,
    MaxEncodedLen,
)]
pub struct DomainId(u32);

impl From<u32> for DomainId {
    #[inline]
    fn from(x: u32) -> Self {
        Self(x)
    }
}

impl From<DomainId> for u32 {
    #[inline]
    fn from(domain_id: DomainId) -> Self {
        domain_id.0
    }
}

impl FromStr for DomainId {
    type Err = ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<u32>().map(Into::into)
    }
}

impl Add<DomainId> for DomainId {
    type Output = Self;

    fn add(self, other: DomainId) -> Self {
        Self(self.0 + other.0)
    }
}

impl Sub<DomainId> for DomainId {
    type Output = Self;

    fn sub(self, other: DomainId) -> Self {
        Self(self.0 - other.0)
    }
}

impl CheckedAdd for DomainId {
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        self.0.checked_add(rhs.0).map(Self)
    }
}

impl DomainId {
    /// Creates a [`DomainId`].
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Converts the inner integer to little-endian bytes.
    pub fn to_le_bytes(&self) -> [u8; 4] {
        self.0.to_le_bytes()
    }
}

impl Display for DomainId {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

impl PassBy for DomainId {
    type PassBy = pass_by::Codec<Self>;
}

/// Identifier of a chain.
#[derive(
    Clone,
    Copy,
    Debug,
    Hash,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Encode,
    Decode,
    TypeInfo,
    Serialize,
    Deserialize,
    MaxEncodedLen,
)]
pub enum ChainId {
    Consensus,
    Domain(DomainId),
}

impl ChainId {
    #[inline]
    pub fn consensus_chain_id() -> Self {
        Self::Consensus
    }

    #[inline]
    pub fn is_consensus_chain(&self) -> bool {
        match self {
            ChainId::Consensus => true,
            ChainId::Domain(_) => false,
        }
    }

    #[inline]
    pub fn maybe_domain_chain(&self) -> Option<DomainId> {
        match self {
            ChainId::Consensus => None,
            ChainId::Domain(domain_id) => Some(*domain_id),
        }
    }
}

impl From<u32> for ChainId {
    #[inline]
    fn from(x: u32) -> Self {
        Self::Domain(DomainId::new(x))
    }
}

impl From<DomainId> for ChainId {
    #[inline]
    fn from(x: DomainId) -> Self {
        Self::Domain(x)
    }
}

#[derive(Clone, Debug, Decode, Default, Encode, Eq, PartialEq, TypeInfo)]
pub struct BlockFees<Balance> {
    /// The consensus chain storage fee
    pub consensus_storage_fee: Balance,
    /// The domain execution fee including the storage and compute fee on domain chain,
    /// tip, and the XDM reward.
    pub domain_execution_fee: Balance,
    /// Burned balances on domain chain
    pub burned_balance: Balance,
    /// Rewards for the chain.
    pub chain_rewards: BTreeMap<ChainId, Balance>,
}

impl<Balance> BlockFees<Balance>
where
    Balance: CheckedAdd,
{
    pub fn new(
        domain_execution_fee: Balance,
        consensus_storage_fee: Balance,
        burned_balance: Balance,
        chain_rewards: BTreeMap<ChainId, Balance>,
    ) -> Self {
        BlockFees {
            consensus_storage_fee,
            domain_execution_fee,
            burned_balance,
            chain_rewards,
        }
    }

    /// Returns the total fees that was collected and burned on the Domain.
    pub fn total_fees(&self) -> Option<Balance> {
        self.consensus_storage_fee
            .checked_add(&self.domain_execution_fee)
            .and_then(|balance| balance.checked_add(&self.burned_balance))
    }
}

/// Type that holds the transfers(in/out) for a given chain.
#[derive(Debug, Decode, Encode, TypeInfo, PartialEq, Eq, Clone, Default)]
pub struct Transfers<Balance> {
    /// Total transfers that came into the domain.
    pub transfers_in: BTreeMap<ChainId, Balance>,
    /// Total transfers that went out of the domain.
    pub transfers_out: BTreeMap<ChainId, Balance>,
    /// Total transfers from this domain that were reverted.
    pub rejected_transfers_claimed: BTreeMap<ChainId, Balance>,
    /// Total transfers to this domain that were rejected.
    pub transfers_rejected: BTreeMap<ChainId, Balance>,
}

impl<Balance> Transfers<Balance> {
    pub fn is_valid(&self, chain_id: ChainId) -> bool {
        !self.transfers_rejected.contains_key(&chain_id)
            && !self.transfers_in.contains_key(&chain_id)
            && !self.transfers_out.contains_key(&chain_id)
            && !self.rejected_transfers_claimed.contains_key(&chain_id)
    }
}

// TODO: this runtime constant is not support to update, see https://github.com/autonomys/subspace/issues/2712
// for more detail about the problem and what we need to do to support it.
//
/// Initial tx range = U256::MAX / INITIAL_DOMAIN_TX_RANGE.
pub const INITIAL_DOMAIN_TX_RANGE: u64 = 3;

#[derive(Debug, Decode, Encode, TypeInfo, PartialEq, Eq, Clone)]
pub struct BundleHeader<Number, Hash, DomainHeader: HeaderT, Balance> {
    /// Proof of bundle producer election.
    pub proof_of_election: ProofOfElection,
    /// Execution receipt that should extend the receipt chain or add confirmations
    /// to the head receipt.
    pub receipt: ExecutionReceipt<
        Number,
        Hash,
        HeaderNumberFor<DomainHeader>,
        HeaderHashFor<DomainHeader>,
        Balance,
    >,
    /// The total (estimated) weight of all extrinsics in the bundle.
    ///
    /// Used to prevent overloading the bundle with compute.
    pub estimated_bundle_weight: Weight,
    /// The Merkle root of all new extrinsics included in this bundle.
    pub bundle_extrinsics_root: HeaderHashFor<DomainHeader>,
}

impl<Number: Encode, Hash: Encode, DomainHeader: HeaderT, Balance: Encode>
    BundleHeader<Number, Hash, DomainHeader, Balance>
{
    /// Returns the hash of this header.
    pub fn hash(&self) -> HeaderHashFor<DomainHeader> {
        HeaderHashingFor::<DomainHeader>::hash_of(self)
    }
}

/// Header of bundle.
#[derive(Debug, Decode, Encode, TypeInfo, PartialEq, Eq, Clone)]
pub struct SealedBundleHeader<Number, Hash, DomainHeader: HeaderT, Balance> {
    /// Unsealed header.
    pub header: BundleHeader<Number, Hash, DomainHeader, Balance>,
    /// Signature of the bundle.
    pub signature: OperatorSignature,
}

impl<Number: Encode, Hash: Encode, DomainHeader: HeaderT, Balance: Encode>
    SealedBundleHeader<Number, Hash, DomainHeader, Balance>
{
    /// Constructs a new instance of [`SealedBundleHeader`].
    pub fn new(
        header: BundleHeader<Number, Hash, DomainHeader, Balance>,
        signature: OperatorSignature,
    ) -> Self {
        Self { header, signature }
    }

    /// Returns the hash of the inner unsealed header.
    pub fn pre_hash(&self) -> HeaderHashFor<DomainHeader> {
        self.header.hash()
    }

    /// Returns the hash of this header.
    pub fn hash(&self) -> HeaderHashFor<DomainHeader> {
        HeaderHashingFor::<DomainHeader>::hash_of(self)
    }

    pub fn slot_number(&self) -> u64 {
        self.header.proof_of_election.slot_number
    }
}

/// Domain bundle.
#[derive(Debug, Decode, Encode, TypeInfo, PartialEq, Eq, Clone)]
pub struct Bundle<Extrinsic, Number, Hash, DomainHeader: HeaderT, Balance> {
    /// Sealed bundle header.
    pub sealed_header: SealedBundleHeader<Number, Hash, DomainHeader, Balance>,
    /// The accompanying extrinsics.
    pub extrinsics: Vec<Extrinsic>,
}

impl<Extrinsic: Encode, Number: Encode, Hash: Encode, DomainHeader: HeaderT, Balance: Encode>
    Bundle<Extrinsic, Number, Hash, DomainHeader, Balance>
{
    /// Returns the hash of this bundle.
    pub fn hash(&self) -> H256 {
        BlakeTwo256::hash_of(self)
    }

    /// Returns the domain_id of this bundle.
    pub fn domain_id(&self) -> DomainId {
        self.sealed_header.header.proof_of_election.domain_id
    }

    /// Return the `bundle_extrinsics_root`
    pub fn extrinsics_root(&self) -> HeaderHashFor<DomainHeader> {
        self.sealed_header.header.bundle_extrinsics_root
    }

    /// Return the `operator_id`
    pub fn operator_id(&self) -> OperatorId {
        self.sealed_header.header.proof_of_election.operator_id
    }

    /// Return a reference of the execution receipt.
    pub fn receipt(
        &self,
    ) -> &ExecutionReceipt<
        Number,
        Hash,
        HeaderNumberFor<DomainHeader>,
        HeaderHashFor<DomainHeader>,
        Balance,
    > {
        &self.sealed_header.header.receipt
    }

    /// Consumes [`Bundle`] to extract the execution receipt.
    pub fn into_receipt(
        self,
    ) -> ExecutionReceipt<
        Number,
        Hash,
        HeaderNumberFor<DomainHeader>,
        HeaderHashFor<DomainHeader>,
        Balance,
    > {
        self.sealed_header.header.receipt
    }

    /// Return the bundle size (include header and body) in bytes
    pub fn size(&self) -> u32 {
        self.encoded_size() as u32
    }

    /// Return the bundle body size in bytes
    pub fn body_size(&self) -> u32 {
        self.extrinsics
            .iter()
            .map(|tx| tx.encoded_size() as u32)
            .sum::<u32>()
    }

    pub fn estimated_weight(&self) -> Weight {
        self.sealed_header.header.estimated_bundle_weight
    }

    pub fn slot_number(&self) -> u64 {
        self.sealed_header.header.proof_of_election.slot_number
    }
}

/// Bundle with opaque extrinsics.
pub type OpaqueBundle<Number, Hash, DomainHeader, Balance> =
    Bundle<OpaqueExtrinsic, Number, Hash, DomainHeader, Balance>;

/// List of [`OpaqueBundle`].
pub type OpaqueBundles<Block, DomainHeader, Balance> =
    Vec<OpaqueBundle<NumberFor<Block>, <Block as BlockT>::Hash, DomainHeader, Balance>>;

impl<Extrinsic: Encode, Number, Hash, DomainHeader: HeaderT, Balance>
    Bundle<Extrinsic, Number, Hash, DomainHeader, Balance>
{
    /// Convert a bundle with generic extrinsic to a bundle with opaque extrinsic.
    pub fn into_opaque_bundle(self) -> OpaqueBundle<Number, Hash, DomainHeader, Balance> {
        let Bundle {
            sealed_header,
            extrinsics,
        } = self;
        let opaque_extrinsics = extrinsics
            .into_iter()
            .map(|xt| {
                OpaqueExtrinsic::from_bytes(&xt.encode())
                    .expect("We have just encoded a valid extrinsic; qed")
            })
            .collect();
        OpaqueBundle {
            sealed_header,
            extrinsics: opaque_extrinsics,
        }
    }
}

#[cfg(any(feature = "std", feature = "runtime-benchmarks"))]
pub fn dummy_opaque_bundle<
    Number: Encode,
    Hash: Default + Encode,
    DomainHeader: HeaderT,
    Balance: Encode,
>(
    domain_id: DomainId,
    operator_id: OperatorId,
    receipt: ExecutionReceipt<
        Number,
        Hash,
        HeaderNumberFor<DomainHeader>,
        HeaderHashFor<DomainHeader>,
        Balance,
    >,
) -> OpaqueBundle<Number, Hash, DomainHeader, Balance> {
    use sp_core::crypto::UncheckedFrom;

    let header = BundleHeader {
        proof_of_election: ProofOfElection::dummy(domain_id, operator_id),
        receipt,
        estimated_bundle_weight: Default::default(),
        bundle_extrinsics_root: Default::default(),
    };
    let signature = OperatorSignature::unchecked_from([0u8; 64]);

    OpaqueBundle {
        sealed_header: SealedBundleHeader::new(header, signature),
        extrinsics: Vec::new(),
    }
}

/// A digest of the bundle
#[derive(Debug, Decode, Encode, TypeInfo, PartialEq, Eq, Clone)]
pub struct BundleDigest<Hash> {
    /// The hash of the bundle header
    pub header_hash: Hash,
    /// The Merkle root of all new extrinsics included in this bundle.
    pub extrinsics_root: Hash,
    /// The size of the bundle body in bytes.
    pub size: u32,
}

/// Receipt of a domain block execution.
#[derive(Debug, Decode, Encode, TypeInfo, PartialEq, Eq, Clone)]
pub struct ExecutionReceipt<Number, Hash, DomainNumber, DomainHash, Balance> {
    /// The index of the current domain block that forms the basis of this ER.
    pub domain_block_number: DomainNumber,
    /// The block hash corresponding to `domain_block_number`.
    pub domain_block_hash: DomainHash,
    /// Extrinsic root field of the header of domain block referenced by this ER.
    pub domain_block_extrinsic_root: DomainHash,
    /// The hash of the ER for the last domain block.
    pub parent_domain_block_receipt_hash: DomainHash,
    /// A pointer to the consensus block index which contains all of the bundles that were used to derive and
    /// order all extrinsics executed by the current domain block for this ER.
    pub consensus_block_number: Number,
    /// The block hash corresponding to `consensus_block_number`.
    pub consensus_block_hash: Hash,
    /// All the bundles that being included in the consensus block.
    pub inboxed_bundles: Vec<InboxedBundle<DomainHash>>,
    /// The final state root for the current domain block reflected by this ER.
    ///
    /// Used for verifying storage proofs for domains.
    pub final_state_root: DomainHash,
    /// List of storage roots collected during the domain block execution.
    pub execution_trace: Vec<DomainHash>,
    /// The Merkle root of the execution trace for the current domain block.
    ///
    /// Used for verifying fraud proofs.
    pub execution_trace_root: H256,
    /// Compute and Domain storage fees are shared across operators and Consensus
    /// storage fees are given to the consensus block author.
    pub block_fees: BlockFees<Balance>,
    /// List of transfers from this Domain to other chains
    pub transfers: Transfers<Balance>,
}

impl<Number, Hash, DomainNumber, DomainHash, Balance>
    ExecutionReceipt<Number, Hash, DomainNumber, DomainHash, Balance>
{
    pub fn bundles_extrinsics_roots(&self) -> Vec<&DomainHash> {
        self.inboxed_bundles
            .iter()
            .map(|b| &b.extrinsics_root)
            .collect()
    }

    pub fn valid_bundle_digest_at(&self, index: usize) -> Option<DomainHash>
    where
        DomainHash: Copy,
    {
        match self.inboxed_bundles.get(index).map(|ib| &ib.bundle) {
            Some(BundleValidity::Valid(bundle_digest_hash)) => Some(*bundle_digest_hash),
            _ => None,
        }
    }

    pub fn valid_bundle_digests(&self) -> Vec<DomainHash>
    where
        DomainHash: Copy,
    {
        self.inboxed_bundles
            .iter()
            .filter_map(|b| match b.bundle {
                BundleValidity::Valid(bundle_digest_hash) => Some(bundle_digest_hash),
                BundleValidity::Invalid(_) => None,
            })
            .collect()
    }

    pub fn valid_bundle_indexes(&self) -> Vec<u32> {
        self.inboxed_bundles
            .iter()
            .enumerate()
            .filter_map(|(index, b)| match b.bundle {
                BundleValidity::Valid(_) => Some(index as u32),
                BundleValidity::Invalid(_) => None,
            })
            .collect()
    }
}

impl<
        Number: Encode + Zero,
        Hash: Encode + Default,
        DomainNumber: Encode + Zero,
        DomainHash: Clone + Encode + Default,
        Balance: Encode + Zero + Default,
    > ExecutionReceipt<Number, Hash, DomainNumber, DomainHash, Balance>
{
    /// Returns the hash of this execution receipt.
    pub fn hash<DomainHashing: HashT<Output = DomainHash>>(&self) -> DomainHash {
        DomainHashing::hash_of(self)
    }

    pub fn genesis(
        genesis_state_root: DomainHash,
        genesis_extrinsic_root: DomainHash,
        genesis_domain_block_hash: DomainHash,
    ) -> Self {
        ExecutionReceipt {
            domain_block_number: Zero::zero(),
            domain_block_hash: genesis_domain_block_hash,
            domain_block_extrinsic_root: genesis_extrinsic_root,
            parent_domain_block_receipt_hash: Default::default(),
            consensus_block_hash: Default::default(),
            consensus_block_number: Zero::zero(),
            inboxed_bundles: Vec::new(),
            final_state_root: genesis_state_root.clone(),
            execution_trace: sp_std::vec![genesis_state_root],
            execution_trace_root: Default::default(),
            block_fees: Default::default(),
            transfers: Default::default(),
        }
    }

    #[cfg(any(feature = "std", feature = "runtime-benchmarks"))]
    pub fn dummy<DomainHashing>(
        consensus_block_number: Number,
        consensus_block_hash: Hash,
        domain_block_number: DomainNumber,
        parent_domain_block_receipt_hash: DomainHash,
    ) -> ExecutionReceipt<Number, Hash, DomainNumber, DomainHash, Balance>
    where
        DomainHashing: HashT<Output = DomainHash>,
    {
        let execution_trace = sp_std::vec![Default::default(), Default::default()];
        let execution_trace_root = {
            let trace: Vec<[u8; 32]> = execution_trace
                .iter()
                .map(|r: &DomainHash| r.encode().try_into().expect("H256 must fit into [u8; 32]"))
                .collect();
            crate::merkle_tree::MerkleTree::from_leaves(trace.as_slice())
                .root()
                .expect("Compute merkle root of trace should success")
                .into()
        };
        ExecutionReceipt {
            domain_block_number,
            domain_block_hash: Default::default(),
            domain_block_extrinsic_root: Default::default(),
            parent_domain_block_receipt_hash,
            consensus_block_number,
            consensus_block_hash,
            inboxed_bundles: sp_std::vec![InboxedBundle::dummy(Default::default())],
            final_state_root: Default::default(),
            execution_trace,
            execution_trace_root,
            block_fees: Default::default(),
            transfers: Default::default(),
        }
    }
}

#[derive(Debug, Decode, Encode, TypeInfo, PartialEq, Eq, Clone)]
pub struct ProofOfElection {
    /// Domain id.
    pub domain_id: DomainId,
    /// The slot number.
    pub slot_number: u64,
    /// The PoT output for `slot_number`.
    pub proof_of_time: PotOutput,
    /// VRF signature.
    pub vrf_signature: VrfSignature,
    /// Operator index in the OperatorRegistry.
    pub operator_id: OperatorId,
}

impl ProofOfElection {
    pub fn verify_vrf_signature(
        &self,
        operator_signing_key: &OperatorPublicKey,
    ) -> Result<(), ProofOfElectionError> {
        let global_challenge = self
            .proof_of_time
            .derive_global_randomness()
            .derive_global_challenge(self.slot_number);
        bundle_producer_election::verify_vrf_signature(
            self.domain_id,
            operator_signing_key,
            &self.vrf_signature,
            &global_challenge,
        )
    }

    /// Computes the VRF hash.
    pub fn vrf_hash(&self) -> Blake3Hash {
        let mut bytes = self.vrf_signature.pre_output.encode();
        bytes.append(&mut self.vrf_signature.proof.encode());
        blake3_hash(&bytes)
    }

    pub fn slot_number(&self) -> u64 {
        self.slot_number
    }
}

impl ProofOfElection {
    #[cfg(any(feature = "std", feature = "runtime-benchmarks"))]
    pub fn dummy(domain_id: DomainId, operator_id: OperatorId) -> Self {
        let output_bytes = sp_std::vec![0u8; VrfPreOutput::max_encoded_len()];
        let proof_bytes = sp_std::vec![0u8; VrfProof::max_encoded_len()];
        let vrf_signature = VrfSignature {
            pre_output: VrfPreOutput::decode(&mut output_bytes.as_slice()).unwrap(),
            proof: VrfProof::decode(&mut proof_bytes.as_slice()).unwrap(),
        };
        Self {
            domain_id,
            slot_number: 0u64,
            proof_of_time: PotOutput::default(),
            vrf_signature,
            operator_id,
        }
    }
}

/// Singleton receipt submit along when there is a gap between `domain_best_number`
/// and `HeadReceiptNumber`
#[derive(Debug, Decode, Encode, TypeInfo, PartialEq, Eq, Clone)]
pub struct SingletonReceipt<Number, Hash, DomainHeader: HeaderT, Balance> {
    /// Proof of receipt producer election.
    pub proof_of_election: ProofOfElection,
    /// The receipt to submit
    pub receipt: ExecutionReceipt<
        Number,
        Hash,
        HeaderNumberFor<DomainHeader>,
        HeaderHashFor<DomainHeader>,
        Balance,
    >,
}

impl<Number: Encode, Hash: Encode, DomainHeader: HeaderT, Balance: Encode>
    SingletonReceipt<Number, Hash, DomainHeader, Balance>
{
    pub fn hash(&self) -> HeaderHashFor<DomainHeader> {
        HeaderHashingFor::<DomainHeader>::hash_of(&self)
    }
}

#[derive(Debug, Decode, Encode, TypeInfo, PartialEq, Eq, Clone)]
pub struct SealedSingletonReceipt<Number, Hash, DomainHeader: HeaderT, Balance> {
    /// A collection of the receipt.
    pub singleton_receipt: SingletonReceipt<Number, Hash, DomainHeader, Balance>,
    /// Signature of the receipt bundle.
    pub signature: OperatorSignature,
}

impl<Number: Encode, Hash: Encode, DomainHeader: HeaderT, Balance: Encode>
    SealedSingletonReceipt<Number, Hash, DomainHeader, Balance>
{
    /// Returns the `domain_id`
    pub fn domain_id(&self) -> DomainId {
        self.singleton_receipt.proof_of_election.domain_id
    }

    /// Return the `operator_id`
    pub fn operator_id(&self) -> OperatorId {
        self.singleton_receipt.proof_of_election.operator_id
    }

    /// Return the `slot_number` of the `proof_of_election`
    pub fn slot_number(&self) -> u64 {
        self.singleton_receipt.proof_of_election.slot_number
    }

    /// Return the receipt
    pub fn receipt(
        &self,
    ) -> &ExecutionReceipt<
        Number,
        Hash,
        HeaderNumberFor<DomainHeader>,
        HeaderHashFor<DomainHeader>,
        Balance,
    > {
        &self.singleton_receipt.receipt
    }

    /// Consume this `SealedSingletonReceipt` and return the receipt
    pub fn into_receipt(
        self,
    ) -> ExecutionReceipt<
        Number,
        Hash,
        HeaderNumberFor<DomainHeader>,
        HeaderHashFor<DomainHeader>,
        Balance,
    > {
        self.singleton_receipt.receipt
    }

    /// Returns the hash of `SingletonReceipt`
    pub fn pre_hash(&self) -> HeaderHashFor<DomainHeader> {
        HeaderHashingFor::<DomainHeader>::hash_of(&self.singleton_receipt)
    }

    /// Return the encode size of `SealedSingletonReceipt`
    pub fn size(&self) -> u32 {
        self.encoded_size() as u32
    }
}

/// Type that represents an operator allow list for Domains.
#[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperatorAllowList<AccountId: Ord> {
    /// Anyone can operate for this domain.
    Anyone,
    /// Only the specific operators are allowed to operate the domain.
    /// This essentially makes the domain permissioned.
    Operators(BTreeSet<AccountId>),
}

impl<AccountId: Ord> OperatorAllowList<AccountId> {
    /// Returns true if the allow list is either `Anyone` or the operator is part of the allowed operator list.
    pub fn is_operator_allowed(&self, operator: &AccountId) -> bool {
        match self {
            OperatorAllowList::Anyone => true,
            OperatorAllowList::Operators(allowed_operators) => allowed_operators.contains(operator),
        }
    }
}

/// Permissioned actions allowed by either specific accounts or anyone.
#[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermissionedActionAllowedBy<AccountId: Codec + Clone> {
    Accounts(Vec<AccountId>),
    Anyone,
}

impl<AccountId: Codec + PartialEq + Clone> PermissionedActionAllowedBy<AccountId> {
    pub fn is_allowed(&self, who: &AccountId) -> bool {
        match self {
            PermissionedActionAllowedBy::Accounts(accounts) => accounts.contains(who),
            PermissionedActionAllowedBy::Anyone => true,
        }
    }

    pub fn is_anyone_allowed(&self) -> bool {
        matches!(self, PermissionedActionAllowedBy::Anyone)
    }
}

/// EVM-specific domain runtime config.
#[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvmDomainRuntimeConfig {
    /// Accounts initially allowed to create contracts on an EVM domain.
    /// The domain owner can update this list using a sudo call.
    pub initial_contract_creation_allow_list: PermissionedActionAllowedBy<EthereumAccountId>,
}

impl Default for EvmDomainRuntimeConfig {
    fn default() -> Self {
        EvmDomainRuntimeConfig {
            initial_contract_creation_allow_list: PermissionedActionAllowedBy::Anyone,
        }
    }
}

/// AutoId-specific domain runtime config.
#[derive(
    TypeInfo, Debug, Default, Encode, Decode, Clone, PartialEq, Eq, Serialize, Deserialize,
)]
pub struct AutoIdDomainRuntimeConfig {
    // Currently, there is no specific configuration for AutoId.
}

/// Configrations for specific domain runtime kinds.
#[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DomainRuntimeConfig {
    Evm(EvmDomainRuntimeConfig),
    AutoId(AutoIdDomainRuntimeConfig),
}

impl Default for DomainRuntimeConfig {
    fn default() -> Self {
        Self::default_evm()
    }
}

impl From<EvmDomainRuntimeConfig> for DomainRuntimeConfig {
    fn from(evm_config: EvmDomainRuntimeConfig) -> Self {
        DomainRuntimeConfig::Evm(evm_config)
    }
}

impl From<AutoIdDomainRuntimeConfig> for DomainRuntimeConfig {
    fn from(auto_id_config: AutoIdDomainRuntimeConfig) -> Self {
        DomainRuntimeConfig::AutoId(auto_id_config)
    }
}

impl DomainRuntimeConfig {
    pub fn default_evm() -> Self {
        DomainRuntimeConfig::Evm(EvmDomainRuntimeConfig::default())
    }

    pub fn default_auto_id() -> Self {
        DomainRuntimeConfig::AutoId(AutoIdDomainRuntimeConfig::default())
    }

    pub fn is_evm(&self) -> bool {
        matches!(self, DomainRuntimeConfig::Evm(_))
    }

    pub fn is_auto_id(&self) -> bool {
        matches!(self, DomainRuntimeConfig::AutoId(_))
    }

    pub fn evm(&self) -> Option<&EvmDomainRuntimeConfig> {
        match self {
            DomainRuntimeConfig::Evm(evm_config) => Some(evm_config),
            _ => None,
        }
    }

    pub fn initial_contract_creation_allow_list(
        &self,
    ) -> Option<&PermissionedActionAllowedBy<EthereumAccountId>> {
        self.evm()
            .map(|evm_config| &evm_config.initial_contract_creation_allow_list)
    }

    pub fn auto_id(&self) -> Option<&AutoIdDomainRuntimeConfig> {
        match self {
            DomainRuntimeConfig::AutoId(auto_id_config) => Some(auto_id_config),
            _ => None,
        }
    }
}

#[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenesisDomain<AccountId: Ord, Balance> {
    // Domain runtime items
    pub runtime_name: String,
    pub runtime_type: RuntimeType,
    pub runtime_version: RuntimeVersion,
    pub raw_genesis_storage: Vec<u8>,

    // Domain config items
    pub owner_account_id: AccountId,
    pub domain_name: String,
    pub bundle_slot_probability: (u64, u64),
    pub operator_allow_list: OperatorAllowList<AccountId>,
    /// Configurations for a specific type of domain runtime, for example, EVM.
    pub domain_runtime_config: DomainRuntimeConfig,

    // Genesis operator
    pub signing_key: OperatorPublicKey,
    pub minimum_nominator_stake: Balance,
    pub nomination_tax: Percent,

    // initial balances
    pub initial_balances: Vec<(MultiAccountId, Balance)>,
}

/// Types of runtime pallet domains currently supports
#[derive(
    Debug, Default, Encode, Decode, TypeInfo, Copy, Clone, PartialEq, Eq, Serialize, Deserialize,
)]
pub enum RuntimeType {
    #[default]
    Evm,
    AutoId,
}

/// Type representing the runtime ID.
pub type RuntimeId = u32;

/// Type representing domain epoch.
pub type EpochIndex = u32;

/// Type representing operator ID
pub type OperatorId = u64;

/// Channel identity.
pub type ChannelId = sp_core::U256;

/// Domains specific digest item.
#[derive(PartialEq, Eq, Clone, Encode, Decode, TypeInfo)]
pub enum DomainDigestItem {
    DomainRuntimeUpgraded(RuntimeId),
    DomainInstantiated(DomainId),
}

/// Domains specific digest items.
pub trait DomainsDigestItem {
    fn domain_runtime_upgrade(runtime_id: RuntimeId) -> Self;
    fn as_domain_runtime_upgrade(&self) -> Option<RuntimeId>;

    fn domain_instantiation(domain_id: DomainId) -> Self;
    fn as_domain_instantiation(&self) -> Option<DomainId>;
}

impl DomainsDigestItem for DigestItem {
    fn domain_runtime_upgrade(runtime_id: RuntimeId) -> Self {
        Self::Other(DomainDigestItem::DomainRuntimeUpgraded(runtime_id).encode())
    }

    fn as_domain_runtime_upgrade(&self) -> Option<RuntimeId> {
        match self.try_to::<DomainDigestItem>(OpaqueDigestItemId::Other) {
            Some(DomainDigestItem::DomainRuntimeUpgraded(runtime_id)) => Some(runtime_id),
            _ => None,
        }
    }

    fn domain_instantiation(domain_id: DomainId) -> Self {
        Self::Other(DomainDigestItem::DomainInstantiated(domain_id).encode())
    }

    fn as_domain_instantiation(&self) -> Option<DomainId> {
        match self.try_to::<DomainDigestItem>(OpaqueDigestItemId::Other) {
            Some(DomainDigestItem::DomainInstantiated(domain_id)) => Some(domain_id),
            _ => None,
        }
    }
}

/// EVM chain Id storage key.
///
/// This function should ideally use a Host function to fetch the storage key
/// from the domain runtime. But since the Host function is not available at Genesis, we have to
/// assume the storage keys.
/// TODO: once the chain is launched in mainnet, we should use the Host function for all domain instances.
pub(crate) fn evm_chain_id_storage_key() -> StorageKey {
    StorageKey(
        storage_prefix(
            // This is the name used for `pallet_evm_chain_id` in the `construct_runtime` macro
            // i.e. `EVMChainId: pallet_evm_chain_id = 82,`
            "EVMChainId".as_bytes(),
            // This is the storage item name used inside `pallet_evm_chain_id`
            "ChainId".as_bytes(),
        )
        .to_vec(),
    )
}

/// EVM contract creation allow list storage key.
///
/// This function should ideally use a Host function to fetch the storage key
/// from the domain runtime. But since the Host function is not available at Genesis, we have to
/// assume the storage keys.
/// TODO: once the chain is launched in mainnet, we should use the Host function for all domain instances.
pub(crate) fn evm_contract_creation_allowed_by_storage_key() -> StorageKey {
    StorageKey(
        storage_prefix(
            // This is the name used for `pallet_evm_tracker` in the `construct_runtime` macro
            // i.e. `EVMNoncetracker: pallet_evm_tracker = 84,`
            "EVMNoncetracker".as_bytes(),
            // This is the storage item name used inside `pallet_evm_tracker`
            "ContractCreationAllowedBy".as_bytes(),
        )
        .to_vec(),
    )
}

/// Total issuance storage key for Domains.
///
/// This function should ideally use a Host function to fetch the storage key
/// from the domain runtime. But since the Host function is not available at Genesis, we have to
/// assume the storage keys.
/// TODO: once the chain is launched in mainnet, we should use the Host function for all domain instances.
pub fn domain_total_issuance_storage_key() -> StorageKey {
    StorageKey(
        storage_prefix(
            // This is the name used for `pallet_balances` in the `construct_runtime` macro
            "Balances".as_bytes(),
            // This is the storage item name used inside `pallet_balances`
            "TotalIssuance".as_bytes(),
        )
        .to_vec(),
    )
}

/// Account info on frame_system on Domains
///
/// This function should ideally use a Host function to fetch the storage key
/// from the domain runtime. But since the Host function is not available at Genesis, we have to
/// assume the storage keys.
/// TODO: once the chain is launched in mainnet, we should use the Host function for all domain instances.
pub fn domain_account_storage_key<AccountId: Encode>(who: AccountId) -> StorageKey {
    let storage_prefix = storage_prefix("System".as_bytes(), "Account".as_bytes());
    let key_hashed = who.using_encoded(Blake2_128Concat::hash);

    let mut final_key = Vec::with_capacity(storage_prefix.len() + key_hashed.len());

    final_key.extend_from_slice(&storage_prefix);
    final_key.extend_from_slice(key_hashed.as_ref());

    StorageKey(final_key)
}

/// The storage key of the `SelfDomainId` storage item in `pallet-domain-id`
///
/// Any change to the storage item name or `pallet-domain-id` name used in the `construct_runtime`
/// macro must be reflected here.
pub fn self_domain_id_storage_key() -> StorageKey {
    StorageKey(
        frame_support::storage::storage_prefix(
            // This is the name used for `pallet-domain-id` in the `construct_runtime` macro
            // i.e. `SelfDomainId: pallet_domain_id = 90`
            "SelfDomainId".as_bytes(),
            // This is the storage item name used inside `pallet-domain-id`
            "SelfDomainId".as_bytes(),
        )
        .to_vec(),
    )
}

/// `DomainInstanceData` is used to construct the genesis storage of domain instance chain
#[derive(Debug, PartialEq, Eq, Clone, Encode, Decode, TypeInfo)]
pub struct DomainInstanceData {
    pub runtime_type: RuntimeType,
    pub raw_genesis: RawGenesis,
}

#[derive(Debug, Decode, Encode, TypeInfo, Clone, PartialEq, Eq)]
pub struct DomainBundleLimit {
    /// The max bundle size for the domain.
    pub max_bundle_size: u32,
    /// The max bundle weight for the domain.
    pub max_bundle_weight: Weight,
}

/// Calculates the max bundle weight and size
// See https://forum.subspace.network/t/on-bundle-weight-limits-sum/2277 for more details
// about the formula
pub fn calculate_max_bundle_weight_and_size(
    max_domain_block_size: u32,
    max_domain_block_weight: Weight,
    consensus_slot_probability: (u64, u64),
    bundle_slot_probability: (u64, u64),
) -> Option<DomainBundleLimit> {
    // (n1 / d1) / (n2 / d2) is equal to (n1 * d2) / (d1 * n2)
    // This represents: bundle_slot_probability/SLOT_PROBABILITY
    let expected_bundles_per_block = bundle_slot_probability
        .0
        .checked_mul(consensus_slot_probability.1)?
        .checked_div(
            bundle_slot_probability
                .1
                .checked_mul(consensus_slot_probability.0)?,
        )?;

    // set the proof size for bundle to be proof size of max domain weight
    // so that each domain extrinsic can use the full proof size if required
    let max_proof_size = max_domain_block_weight.proof_size();
    let max_bundle_weight = max_domain_block_weight
        .checked_div(expected_bundles_per_block)?
        .set_proof_size(max_proof_size);

    let max_bundle_size =
        (max_domain_block_size as u64).checked_div(expected_bundles_per_block)? as u32;

    Some(DomainBundleLimit {
        max_bundle_size,
        max_bundle_weight,
    })
}

/// Checks if the signer Id hash is within the tx range
pub fn signer_in_tx_range(bundle_vrf_hash: &U256, signer_id_hash: &U256, tx_range: &U256) -> bool {
    let distance_from_vrf_hash = bidirectional_distance(bundle_vrf_hash, signer_id_hash);
    distance_from_vrf_hash <= (*tx_range / 2)
}

/// Receipt invalidity type.
#[derive(Debug, Decode, Encode, TypeInfo, Clone, PartialEq, Eq)]
pub enum InvalidReceipt {
    /// The field `invalid_bundles` in [`ExecutionReceipt`] is invalid.
    InvalidBundles,
}

#[derive(Debug, Decode, Encode, TypeInfo, Clone, PartialEq, Eq)]
pub enum ReceiptValidity {
    Valid,
    Invalid(InvalidReceipt),
}

/// Bundle invalidity type
///
/// Each type contains the index of the first invalid extrinsic within the bundle
#[derive(Debug, Decode, Encode, TypeInfo, Clone, PartialEq, Eq)]
pub enum InvalidBundleType {
    /// Failed to decode the opaque extrinsic.
    #[codec(index = 0)]
    UndecodableTx(u32),
    /// Transaction is out of the tx range.
    #[codec(index = 1)]
    OutOfRangeTx(u32),
    /// Transaction is illegal (unable to pay the fee, etc).
    #[codec(index = 2)]
    IllegalTx(u32),
    /// Transaction is an invalid XDM.
    #[codec(index = 3)]
    InvalidXDM(u32),
    /// Transaction is an inherent extrinsic.
    #[codec(index = 4)]
    InherentExtrinsic(u32),
    /// The `estimated_bundle_weight` in the bundle header is invalid
    #[codec(index = 5)]
    InvalidBundleWeight,
}

impl InvalidBundleType {
    // Return the checking order of the invalid type
    pub fn checking_order(&self) -> u64 {
        // A bundle can contains multiple invalid extrinsics thus consider the first invalid extrinsic
        // as the invalid type
        let extrinsic_order = match self {
            Self::UndecodableTx(i) => *i,
            Self::OutOfRangeTx(i) => *i,
            Self::IllegalTx(i) => *i,
            Self::InvalidXDM(i) => *i,
            Self::InherentExtrinsic(i) => *i,
            // NOTE: the `InvalidBundleWeight` is targeting the whole bundle not a specific
            // single extrinsic, as `extrinsic_index` is used as the order to check the extrinsic
            // in the bundle returning `u32::MAX` indicate `InvalidBundleWeight` is checked after
            // all the extrinsic in the bundle is checked.
            Self::InvalidBundleWeight => u32::MAX,
        };

        // The extrinsic can be considered as invalid due to multiple `invalid_type` (i.e. an extrinsic
        // can be `OutOfRangeTx` and `IllegalTx` at the same time) thus use the following checking order
        // and consider the first check as the invalid type
        //
        // NOTE: Use explicit number as the order instead of the enum discriminant
        // to avoid changing the order accidentally
        let rule_order = match self {
            Self::UndecodableTx(_) => 1,
            Self::OutOfRangeTx(_) => 2,
            Self::InherentExtrinsic(_) => 3,
            Self::InvalidXDM(_) => 4,
            Self::IllegalTx(_) => 5,
            Self::InvalidBundleWeight => 6,
        };

        // The checking order is a combination of the `extrinsic_order` and `rule_order`
        // it presents as an `u64` where the first 32 bits is the `extrinsic_order` and
        // last 32 bits is the `rule_order` meaning the `extrinsic_order` is checked first
        // then the `rule_order`.
        ((extrinsic_order as u64) << 32) | (rule_order as u64)
    }

    // Return the index of the extrinsic that the invalid type points to
    //
    // NOTE: `InvalidBundleWeight` will return `None` since it is targeting the whole bundle not a
    // specific single extrinsic
    pub fn extrinsic_index(&self) -> Option<u32> {
        match self {
            Self::UndecodableTx(i) => Some(*i),
            Self::OutOfRangeTx(i) => Some(*i),
            Self::IllegalTx(i) => Some(*i),
            Self::InvalidXDM(i) => Some(*i),
            Self::InherentExtrinsic(i) => Some(*i),
            Self::InvalidBundleWeight => None,
        }
    }
}

#[derive(Debug, Decode, Encode, TypeInfo, PartialEq, Eq, Clone)]
pub enum BundleValidity<Hash> {
    // The invalid bundle was originally included in the consensus block but subsequently
    // excluded from execution as invalid and holds the `InvalidBundleType`
    Invalid(InvalidBundleType),
    // The valid bundle's hash of `Vec<(tx_signer, tx_hash)>` of all domain extrinsic being
    // included in the bundle.
    // TODO remove this and use host function to fetch above mentioned data
    Valid(Hash),
}

/// [`InboxedBundle`] represents a bundle that was successfully submitted to the consensus chain
#[derive(Debug, Decode, Encode, TypeInfo, PartialEq, Eq, Clone)]
pub struct InboxedBundle<Hash> {
    pub bundle: BundleValidity<Hash>,
    // TODO remove this as the root is already present in the `ExecutionInbox` storage
    pub extrinsics_root: Hash,
}

impl<Hash> InboxedBundle<Hash> {
    pub fn valid(bundle_digest_hash: Hash, extrinsics_root: Hash) -> Self {
        InboxedBundle {
            bundle: BundleValidity::Valid(bundle_digest_hash),
            extrinsics_root,
        }
    }

    pub fn invalid(invalid_bundle_type: InvalidBundleType, extrinsics_root: Hash) -> Self {
        InboxedBundle {
            bundle: BundleValidity::Invalid(invalid_bundle_type),
            extrinsics_root,
        }
    }

    pub fn is_invalid(&self) -> bool {
        matches!(self.bundle, BundleValidity::Invalid(_))
    }

    pub fn invalid_extrinsic_index(&self) -> Option<u32> {
        match &self.bundle {
            BundleValidity::Invalid(invalid_bundle_type) => invalid_bundle_type.extrinsic_index(),
            BundleValidity::Valid(_) => None,
        }
    }

    #[cfg(any(feature = "std", feature = "runtime-benchmarks"))]
    pub fn dummy(extrinsics_root: Hash) -> Self
    where
        Hash: Default,
    {
        InboxedBundle {
            bundle: BundleValidity::Valid(Hash::default()),
            extrinsics_root,
        }
    }
}

/// Empty extrinsics root.
pub const EMPTY_EXTRINSIC_ROOT: ExtrinsicsRoot = ExtrinsicsRoot {
    0: hex!("03170a2e7597b7b7e3d84c05391d139a62b157e78786d8c082f29dcf4c111314"),
};

pub fn derive_domain_block_hash<DomainHeader: HeaderT>(
    domain_block_number: DomainHeader::Number,
    extrinsics_root: DomainHeader::Hash,
    state_root: DomainHeader::Hash,
    parent_domain_block_hash: DomainHeader::Hash,
    digest: Digest,
) -> DomainHeader::Hash {
    let domain_header = DomainHeader::new(
        domain_block_number,
        extrinsics_root,
        state_root,
        parent_domain_block_hash,
        digest,
    );

    domain_header.hash()
}

/// Represents the extrinsic either as full data or hash of the data.
#[derive(Clone, Debug, Decode, Encode, Eq, PartialEq, TypeInfo)]
pub enum ExtrinsicDigest {
    /// Actual extrinsic data that is inlined since it is less than 33 bytes.
    Data(Vec<u8>),
    /// Extrinsic Hash.
    Hash(H256),
}

impl ExtrinsicDigest {
    pub fn new<Layout: TrieLayout>(ext: Vec<u8>) -> Self
    where
        Layout::Hash: HashT,
        <Layout::Hash as HashT>::Output: Into<H256>,
    {
        if let Some(threshold) = Layout::MAX_INLINE_VALUE {
            if ext.len() >= threshold as usize {
                ExtrinsicDigest::Hash(Layout::Hash::hash(&ext).into())
            } else {
                ExtrinsicDigest::Data(ext)
            }
        } else {
            ExtrinsicDigest::Data(ext)
        }
    }
}

/// Trait that tracks the balances on Domains.
pub trait DomainsTransfersTracker<Balance> {
    type Error;

    /// Initializes the domain balance
    fn initialize_domain_balance(domain_id: DomainId, amount: Balance) -> Result<(), Self::Error>;

    /// Notes a transfer between chains.
    /// Balance on from_chain_id is reduced if it is a domain chain
    fn note_transfer(
        from_chain_id: ChainId,
        to_chain_id: ChainId,
        amount: Balance,
    ) -> Result<(), Self::Error>;

    /// Confirms a transfer between chains.
    fn confirm_transfer(
        from_chain_id: ChainId,
        to_chain_id: ChainId,
        amount: Balance,
    ) -> Result<(), Self::Error>;

    /// Claims a rejected transfer between chains.
    fn claim_rejected_transfer(
        from_chain_id: ChainId,
        to_chain_id: ChainId,
        amount: Balance,
    ) -> Result<(), Self::Error>;

    /// Rejects a initiated transfer between chains.
    fn reject_transfer(
        from_chain_id: ChainId,
        to_chain_id: ChainId,
        amount: Balance,
    ) -> Result<(), Self::Error>;

    /// Reduces a given amount from the domain balance
    fn reduce_domain_balance(domain_id: DomainId, amount: Balance) -> Result<(), Self::Error>;
}

/// Trait to check domain owner.
pub trait DomainOwner<AccountId> {
    /// Returns true if the account is the domain owner.
    fn is_domain_owner(domain_id: DomainId, acc: AccountId) -> bool;
}

impl<AccountId> DomainOwner<AccountId> for () {
    fn is_domain_owner(_domain_id: DomainId, _acc: AccountId) -> bool {
        false
    }
}

/// Post hook to know if the domain had bundle submitted in the previous block.
pub trait DomainBundleSubmitted {
    /// Called in the next block initialisation if there was a domain bundle in the previous block.
    /// This hook if called for domain represents that there is a new domain block for parent consensus block.
    fn domain_bundle_submitted(domain_id: DomainId);
}

impl DomainBundleSubmitted for () {
    fn domain_bundle_submitted(_domain_id: DomainId) {}
}

/// A hook to call after a domain is instantiated
pub trait OnDomainInstantiated {
    fn on_domain_instantiated(domain_id: DomainId);
}

impl OnDomainInstantiated for () {
    fn on_domain_instantiated(_domain_id: DomainId) {}
}

pub type ExecutionReceiptFor<DomainHeader, CBlock, Balance> = ExecutionReceipt<
    NumberFor<CBlock>,
    <CBlock as BlockT>::Hash,
    <DomainHeader as HeaderT>::Number,
    <DomainHeader as HeaderT>::Hash,
    Balance,
>;

/// Domain chains allowlist updates.
#[derive(Default, Debug, Encode, Decode, PartialEq, Eq, Clone, TypeInfo)]
pub struct DomainAllowlistUpdates {
    /// Chains that are allowed to open a channel with this chain.
    pub allow_chains: BTreeSet<ChainId>,
    /// Chains that are not allowed to open a channel with this chain.
    pub remove_chains: BTreeSet<ChainId>,
}

impl DomainAllowlistUpdates {
    pub fn is_empty(&self) -> bool {
        self.allow_chains.is_empty() && self.remove_chains.is_empty()
    }

    pub fn clear(&mut self) {
        self.allow_chains.clear();
        self.remove_chains.clear();
    }
}

/// Domain Sudo runtime call.
///
/// This structure exists because we need to generate a storage proof for FP
/// and Storage shouldn't be None. So each domain must always hold this value even if
/// there is an empty runtime call inside
#[derive(Default, Debug, Encode, Decode, PartialEq, Eq, Clone, TypeInfo)]
pub struct DomainSudoCall {
    pub maybe_call: Option<Vec<u8>>,
}

impl DomainSudoCall {
    pub fn clear(&mut self) {
        self.maybe_call.take();
    }
}

#[derive(TypeInfo, Debug, Encode, Decode, Clone, PartialEq, Eq)]
pub struct RuntimeObject<Number, Hash> {
    pub runtime_name: String,
    pub runtime_type: RuntimeType,
    pub runtime_upgrades: u32,
    pub instance_count: u32,
    pub hash: Hash,
    // The raw genesis storage that contains the runtime code.
    // NOTE: don't use this field directly but `into_complete_raw_genesis` instead
    pub raw_genesis: RawGenesis,
    pub version: RuntimeVersion,
    pub created_at: Number,
    pub updated_at: Number,
}

/// Digest storage key in frame_system.
/// Unfortunately, the digest storage is private and not possible to derive the key from it directly.
pub fn system_digest_final_key() -> Vec<u8> {
    frame_support::storage::storage_prefix("System".as_ref(), "Digest".as_ref()).to_vec()
}

/// Hook to handle chain rewards.
pub trait OnChainRewards<Balance> {
    fn on_chain_rewards(chain_id: ChainId, reward: Balance);
}

impl<Balance> OnChainRewards<Balance> for () {
    fn on_chain_rewards(_chain_id: ChainId, _reward: Balance) {}
}

#[derive(Debug, Decode, Encode, TypeInfo, PartialEq, Eq, Clone)]
pub enum OperatorRewardSource<Number> {
    Bundle {
        at_block_number: Number,
    },
    XDMProtocolFees,
    #[cfg(any(feature = "std", feature = "runtime-benchmarks"))]
    Dummy,
}

sp_api::decl_runtime_apis! {
    /// APIs used to access the domains pallet.
    // When updating this version, document new APIs with "Only present in API versions" comments.
    // TODO: when removing this version, also remove "Only present in API versions" comments and
    // deprecated attributes.
    #[api_version(3)]
    pub trait DomainsApi<DomainHeader: HeaderT> {
        /// Submits the transaction bundle via an unsigned extrinsic.
        fn submit_bundle_unsigned(opaque_bundle: OpaqueBundle<NumberFor<Block>, Block::Hash, DomainHeader, Balance>);

        // Submits a singleton receipt via an unsigned extrinsic.
        fn submit_receipt_unsigned(singleton_receipt: SealedSingletonReceipt<NumberFor<Block>, Block::Hash, DomainHeader, Balance>);

        /// Extracts the bundles successfully stored from the given extrinsics.
        fn extract_successful_bundles(
            domain_id: DomainId,
            extrinsics: Vec<Block::Extrinsic>,
        ) -> OpaqueBundles<Block, DomainHeader, Balance>;

        /// Generates a randomness seed for extrinsics shuffling.
        fn extrinsics_shuffling_seed() -> Randomness;

        /// Returns the current WASM bundle for the given `domain_id`.
        fn domain_runtime_code(domain_id: DomainId) -> Option<Vec<u8>>;

        /// Returns the runtime id for the given `domain_id`.
        fn runtime_id(domain_id: DomainId) -> Option<RuntimeId>;

        /// Returns the list of runtime upgrades in the current block.
        /// Only present in API versions 2 and later.
        fn runtime_upgrades() -> Vec<RuntimeId>;

        /// Returns the domain instance data for the given `domain_id`.
        fn domain_instance_data(domain_id: DomainId) -> Option<(DomainInstanceData, NumberFor<Block>)>;

        /// Returns the current timestamp at the current height.
        fn domain_timestamp() -> Moment;

        /// Returns the current timestamp at the current height.
        #[allow(clippy::deprecated_semver)]
        #[deprecated(since = "3", note = "Use `domain_timestamp()` instead")]
        fn timestamp() -> Moment;

        /// Returns the consensus transaction byte fee that will used to charge the domain
        /// transaction for consensus chain storage fees.
        fn consensus_transaction_byte_fee() -> Balance;

        /// Returns the consensus chain byte fee that will used to charge the domain transaction
        /// for consensus chain storage fees.
        #[allow(clippy::deprecated_semver)]
        #[deprecated(since = "3", note = "Use `consensus_transaction_byte_fee()` instead")]
        fn consensus_chain_byte_fee() -> Balance;

        /// Returns the current Tx range for the given domain Id.
        fn domain_tx_range(domain_id: DomainId) -> U256;

        /// Returns the genesis state root if not pruned.
        fn genesis_state_root(domain_id: DomainId) -> Option<H256>;

        /// Returns the best execution chain number.
        fn head_receipt_number(domain_id: DomainId) -> HeaderNumberFor<DomainHeader>;

        /// Returns the block number of oldest unconfirmed execution receipt.
        fn oldest_unconfirmed_receipt_number(domain_id: DomainId) -> Option<HeaderNumberFor<DomainHeader>>;

        /// Returns the domain bundle limit of the given domain.
        fn domain_bundle_limit(domain_id: DomainId) -> Option<DomainBundleLimit>;

        /// Returns true if there are any ERs in the challenge period with non empty extrinsics.
        fn non_empty_er_exists(domain_id: DomainId) -> bool;

        /// Returns the current best block number for the domain.
        fn domain_best_number(domain_id: DomainId) -> Option<HeaderNumberFor<DomainHeader>>;

        /// Returns the execution receipt with the given hash.
        fn execution_receipt(receipt_hash: HeaderHashFor<DomainHeader>) -> Option<ExecutionReceiptFor<DomainHeader, Block, Balance>>;

        /// Returns the current epoch and the next epoch operators of the given domain.
        fn domain_operators(domain_id: DomainId) -> Option<(BTreeMap<OperatorId, Balance>, Vec<OperatorId>)>;

        /// Returns the execution receipt hash of the given domain and domain block number.
        fn receipt_hash(domain_id: DomainId, domain_number: HeaderNumberFor<DomainHeader>) -> Option<HeaderHashFor<DomainHeader>>;

        /// Returns the latest confirmed domain block number and hash.
        fn latest_confirmed_domain_block(domain_id: DomainId) -> Option<(HeaderNumberFor<DomainHeader>, HeaderHashFor<DomainHeader>)>;

        /// Returns if the receipt is exist and pending to prune
        fn is_bad_er_pending_to_prune(domain_id: DomainId, receipt_hash: HeaderHashFor<DomainHeader>) -> bool;

        /// Returns the balance of the storage fund account.
        fn storage_fund_account_balance(operator_id: OperatorId) -> Balance;

        /// Returns true if the given domain's runtime code has been upgraded since `at`.
        fn is_domain_runtime_upgraded_since(domain_id: DomainId, at: NumberFor<Block>) -> Option<bool>;

        /// Returns the domain sudo calls for the given domain, if any.
        fn domain_sudo_call(domain_id: DomainId) -> Option<Vec<u8>>;

        /// Returns the last confirmed domain block execution receipt.
        fn last_confirmed_domain_block_receipt(domain_id: DomainId) ->Option<ExecutionReceiptFor<DomainHeader, Block, Balance>>;
    }

    pub trait BundleProducerElectionApi<Balance: Encode + Decode> {
        fn bundle_producer_election_params(domain_id: DomainId) -> Option<BundleProducerElectionParams<Balance>>;

        fn operator(operator_id: OperatorId) -> Option<(OperatorPublicKey, Balance)>;
    }
}
