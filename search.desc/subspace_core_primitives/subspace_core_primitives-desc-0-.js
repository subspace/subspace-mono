searchState.loadedDescShard("subspace_core_primitives", 0, "Core primitives for Subspace Network.\nBlock hash in Subspace network.\nBlock number in Subspace network.\nBlockWeight type for fork choice rules.\nHow many bytes Scalar contains physically, use …\nMaximum value.\nThe middle of the piece distance field. The analogue of …\nA Ristretto Schnorr public key as bytes produced by …\nSigning context used for creating reward signatures by …\nType of randomness.\nHow many full bytes can be stored in BLS12-381 scalar (for …\nSize of randomness (in bytes).\nPublic key size in bytes\nSingle BLS12-381 scalar with big-endian representation, …\nSlot number in Subspace network.\n256-bit unsigned integer\nAdds two numbers, checking for overflow. If overflow …\nDivides two numbers, checking for underflow, overflow and …\nMultiplies two numbers, checking for underflow or …\nSubtracts two numbers, checking for underflow. If …\nModule containing wrapper for SCALE encoding/decoding with …\nDerive global slot challenge from global randomness.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCreate from big endian bytes\nCreate from little endian bytes\nPublic key hash.\nHashes-related data structures and functions.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nData structures related to objects (useful data) stored on …\nOne (multiplicative identity) of this type.\nPieces-related data structures.\nProof of space-related data structures.\nProof of time-related data structures.\nSaturating addition. Computes <code>self + other</code>, saturating at …\nSaturating multiplication. Computes <code>self * other</code>, …\nSaturating subtraction. Computes <code>self - other</code>, saturating …\nSectors-related data structures.\nSegments-related data structures.\nSolutions-related data structures and functions.\nConvert to big endian bytes\nConvert to little endian bytes\nZero (additive identity) of this type.\nWrapper data structure that when encoded/decoded will …\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nBLAKE3 hash output transparent wrapper\nSize of BLAKE3 hash output (in bytes).\nBLAKE3 hashing of a single value truncated to 254 bits as …\nBLAKE3 hashing of a single value.\nBLAKE3 hashing of a list of values.\nBLAKE3 keyed hashing of a list of values.\nBLAKE3 hashing of a single value in parallel (only useful …\nBLAKE3 keyed hashing of a single value.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nObject stored inside of the block\nMapping of objects stored inside of the block\nSpace-saving serialization of an object stored in the …\nObject stored in the history of the blockchain\nMapping of objects stored in the history of the blockchain\nV0 of object mapping data structure\nV0 of object mapping data structure.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns a newly created BlockObjectMapping from a list of …\nReturns a newly created GlobalObjectMapping from a list of …\nObject hash\nObject hash. We order by hash, so object hash lookups can …\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nReturns the object mappings\nReturns the object mappings\nReturns the object mappings as a mutable slice\nReturns the object mappings as a mutable slice\nOffset of object in the encoded block.\nRaw record offset of the object in that piece, for use …\nPiece index where object is contained (at least its …\nObjects stored inside of the block\nObjects stored in the history of the blockchain.\nFlat representation of multiple pieces concatenated for …\nNumber of chunks (scalars) within one raw record.\nNumber of chunks (scalars) within one record.\nNumber of s-buckets contained within one record (and by …\nPiece index 1.\nPiece index 1.\nA piece of archival history in Subspace Network.\nA piece of archival history in Subspace Network.\nPiece index in consensus\nPiece offset in sector\nRaw record contained within recorded history segment …\nRecord contained within a piece.\nRecord commitment contained within a piece.\nRecord witness contained within a piece.\nSize in bytes.\nSize of raw record in bytes, is guaranteed to be a …\nSize of a segment record given the global piece size (in …\nSize of record commitment in bytes.\nSize of record witness in bytes.\nSize of a piece (in bytes).\nPiece index 0.\nPiece index 0.\nCommitment contained within a piece.\nMutable commitment contained within a piece.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCreate piece index from bytes.\nReturns the piece index for a source position and segment …\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nIs this piece index a source piece?\nCreate new instance\nAllocate <code>FlatPieces</code> that will hold <code>piece_count</code> pieces …\nCreate boxed value without hitting stack overflow\nCreate boxed value without hitting stack overflow\nCreate boxed value without hitting stack overflow\nCreate vector filled with zeroe records without hitting …\nReturns the next source piece index. Panics if the piece …\nParallel iterator over parity pieces (odd indices)\nMutable parallel iterator over parity pieces (odd indices)\nParallel iterator over source pieces (even indices)\nMutable parallel iterator over source pieces (even indices)\nIterator over parity pieces (odd indices)\nMutable iterator over parity pieces (odd indices)\nIterator over parity pieces (odd indices)\nIterate over all pieces.\nPosition of a piece in a segment\nRecord contained within a piece.\nMutable record contained within a piece.\nSegment index piece index corresponds to\nConvenient conversion from slice of underlying …\nConvenient conversion from slice of underlying …\nConvenient conversion from slice of underlying …\nConvenient conversion from mutable slice of underlying …\nConvenient conversion from mutable slice of underlying …\nConvenient conversion from mutable slice of underlying …\nConvenient conversion from mutable slice of record to …\nConvenient conversion from mutable slice of record to …\nConvenient conversion from mutable slice of piece array to …\nConvenient conversion from slice of record to underlying …\nConvenient conversion from slice of record to underlying …\nConvenient conversion from slice of piece array to …\nIterator over source pieces (even indices)\nMutable iterator over source pieces (even indices)\nIterator over source pieces (even indices)\nPosition of a source piece in the source pieces for a …\nSplit piece into underlying components.\nSplit piece into underlying mutable components.\nConvert piece index to bytes.\nConvert piece offset to bytes.\nConvert from a record to its raw bytes, assumes dealing …\nEnsure piece contains cheaply cloneable shared data.\nEnsure flat pieces contains cheaply cloneable shared data.\nWitness contained within a piece.\nMutable witness contained within a piece.\nConstant K used for proof of space\nProof of space proof bytes.\nProof of space seed.\nSize of proof of space seed in bytes.\nSize of proof of space proof in bytes.\nReturns the argument unchanged.\nReturns the argument unchanged.\nProof hash.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nNumber of PoT checkpoints produced (used to optimize …\nProof of time checkpoints, result of proving\nProof of time key(input to the encryption).\nProof of time output, can be intermediate checkpoint or …\nProof of time seed\nSize of proof of time key in bytes\nSize of proof of time seed in bytes\nSize of proof of time proof in bytes\nDerives the global randomness from the output\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nDerive initial PoT seed from genesis block hash\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nDerive key from proof of time seed\nGet proof of time output out of checkpoints (last …\nDerive seed from proof of time in case entropy injection …\nDerive seed from proof of time with entropy injection\nMax s-bucket index\nS-bucket used in consensus\nData structure representing sector ID in farmer’s plot\nSector index in consensus\nChallenge used for a particular sector for particular slot\nS-bucket 0.\nDerive evaluation seed\nDerive history size when sector created at <code>history_size</code> …\nDerive piece index that should be stored in sector at …\nDerive sector slot challenge for this sector from provided …\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCreate new sector ID by deriving it from public key and …\nIndex of s-bucket within sector to be audited\nProgress of an archived block.\nArchived history segment after archiving is applied.\nThe block has been fully archived.\nErasure coding rate for records during archiving process.\nSize of blockchain history in segments.\nLast archived block\nNumber of pieces in one segment of archived history.\nNumber of raw records in one segment of recorded history.\nSegment index 1.\nHistory size of one\nNumber of partially archived bytes of a block.\nRecorded history segment before archiving is applied.\nSize of segment commitment in bytes.\nSize of recorded history segment in bytes.\nSize of archived history segment in bytes.\nSegment commitment contained within segment header.\nSegment header for a specific segment.\nSegment index type.\nV0 of the segment header data structure\nSegment index 0.\nProgress of an archived block.\nChecked integer subtraction. Computes <code>self - rhs</code>, …\nWe assume a block can always fit into the segment …\nGet the first piece index in this segment.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nHash of the whole segment header\nSize of blockchain history in pieces.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nLast archived block\nGet the last piece index in this segment.\nCreate new instance\nCreate new instance.\nCreate boxed value without hitting stack overflow\nBlock number\nReturn the number of partially archived bytes if the …\nReturns the number of partially archived bytes for a block.\nHash of the segment header of the previous segment\nHistory size at which expiration check for sector happens.\nSegment commitment of the records in a segment.\nSegment index that corresponds to this history size.\nSegment index\nList of piece indexes that belong to this segment.\nList of piece indexes that belong to this segment with …\nSets the archived state of this block to …\nSets new number of partially archived bytes.\nSets new number of partially archived bytes.\nEnsure archived history segment contains cheaply cloneable …\nLast archived block\nHash of the segment header of the previous segment\nRoot of commitments of all records in a segment.\nSegment index\nWitness for chunk contained within a record.\nA Ristretto Schnorr signature as bytes produced by …\nReward signature size in bytes\nSize of chunk witness in bytes.\nFarmer solution for slot challenge.\nType of solution range.\nBidirectional distance metric implemented on top of …\nChunk at above offset\nWitness for above chunk\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nDummy solution for the genesis block\nSize of the blockchain history at time of sector creation\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nTransform solution with one reward address type into …\nPieces offset within sector\nComputes the following:\nProof of space for piece offset\nPublic key of the farmer that created the solution\nRecord commitment that can use used to verify that piece …\nWitness for above record commitment\nAddress for receiving block reward\nIndex of the sector where solution was found\nComputes the following:")