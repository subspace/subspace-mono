//! Set of test accounts.
use fp_account::AccountId20;
use sp_core::ecdsa::{Pair, Public, Signature};
use sp_core::{ecdsa, keccak_256, Pair as PairT};

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Keyring {
    Alice,
    Bob,
    Charlie,
    Dave,
    Eve,
    Ferdie,
    One,
    Two,
    N(u32),
}

impl Keyring {
    /// Sign `msg`.
    pub fn sign(self, msg: &[u8]) -> Signature {
        let msg = keccak_256(msg);
        self.pair().sign_prehashed(&msg)
    }

    /// Return key pair.
    pub fn pair(self) -> Pair {
        ecdsa::Pair::from_string(self.to_seed().as_str(), None).unwrap()
    }

    /// Return public key.
    pub fn public(self) -> Public {
        self.pair().public()
    }

    /// Return seed string.
    pub fn to_seed(self) -> String {
        format!("//{:?}", self)
    }

    /// Return account id
    pub fn to_account_id(self) -> AccountId20 {
        self.public().into()
    }
}
