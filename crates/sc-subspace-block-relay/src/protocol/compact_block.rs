//! Compact block implementation.

use crate::utils::NetworkPeerHandle;
use crate::{
    ClientBackend, ProtocolClient, ProtocolServer, ProtocolUnitInfo, RelayError, Resolved,
    ServerBackend, LOG_TARGET,
};
use async_trait::async_trait;
use codec::{Decode, Encode};
use std::collections::BTreeMap;
use std::sync::Arc;
use tracing::{trace, warn};

/// Request messages
#[derive(Encode, Decode)]
pub(crate) enum CompactBlockRequest<DownloadUnitId, ProtocolUnitId> {
    /// Initial request
    Initial,

    /// Request for missing transactions
    MissingEntries(MissingEntriesRequest<DownloadUnitId, ProtocolUnitId>),
}

/// Response messages
#[derive(Encode, Decode)]
pub(crate) enum CompactBlockResponse<DownloadUnitId, ProtocolUnitId, ProtocolUnit> {
    /// Initial/compact response
    Initial(InitialResponse<DownloadUnitId, ProtocolUnitId, ProtocolUnit>),

    /// Response for missing transactions request
    MissingEntries(MissingEntriesResponse<ProtocolUnit>),
}

/// The compact response
#[derive(Encode, Decode)]
pub(crate) struct InitialResponse<DownloadUnitId, ProtocolUnitId, ProtocolUnit> {
    /// The download unit
    download_unit_id: DownloadUnitId,

    /// List of the protocol units Ids.
    protocol_units: Vec<ProtocolUnitInfo<ProtocolUnitId, ProtocolUnit>>,
}

/// Request for missing transactions
#[derive(Encode, Decode)]
pub(crate) struct MissingEntriesRequest<DownloadUnitId, ProtocolUnitId> {
    /// The download unit
    download_unit_id: DownloadUnitId,

    /// Map of missing entry Id ->  protocol unit Id.
    /// The missing entry Id is an opaque identifier used by the client
    /// side. The server side just returns it as is with the response.
    protocol_unit_ids: BTreeMap<u64, ProtocolUnitId>,
}

/// Response for missing transactions
#[derive(Encode, Decode)]
pub(crate) struct MissingEntriesResponse<ProtocolUnit> {
    /// Map of missing entry Id ->  protocol unit.
    protocol_units: BTreeMap<u64, ProtocolUnit>,
}

struct ResolveContext<ProtocolUnitId, ProtocolUnit> {
    resolved: BTreeMap<u64, Resolved<ProtocolUnitId, ProtocolUnit>>,
    local_miss: BTreeMap<u64, ProtocolUnitId>,
}

pub(crate) struct CompactBlockClient<DownloadUnitId, ProtocolUnitId, ProtocolUnit> {
    backend: Arc<dyn ClientBackend<ProtocolUnitId, ProtocolUnit> + Send + Sync + 'static>,
    _phantom_data: std::marker::PhantomData<DownloadUnitId>,
}

impl<DownloadUnitId, ProtocolUnitId, ProtocolUnit>
    CompactBlockClient<DownloadUnitId, ProtocolUnitId, ProtocolUnit>
where
    DownloadUnitId: Send + Sync + Encode + Decode + Clone,
    ProtocolUnitId: Send + Sync + Encode + Decode + Clone,
    ProtocolUnit: Send + Sync + Encode + Decode + Clone,
{
    /// Creates the client.
    pub(crate) fn new(
        backend: Arc<dyn ClientBackend<ProtocolUnitId, ProtocolUnit> + Send + Sync + 'static>,
    ) -> Self {
        Self {
            backend,
            _phantom_data: Default::default(),
        }
    }

    /// Tries to resolve the entries in InitialResponse locally
    fn resolve_local(
        &self,
        compact_response: &InitialResponse<DownloadUnitId, ProtocolUnitId, ProtocolUnit>,
    ) -> Result<ResolveContext<ProtocolUnitId, ProtocolUnit>, RelayError> {
        let mut context = ResolveContext {
            resolved: BTreeMap::new(),
            local_miss: BTreeMap::new(),
        };

        for (index, entry) in compact_response.protocol_units.iter().enumerate() {
            let ProtocolUnitInfo { id, unit } = entry;
            if let Some(unit) = unit {
                // The full protocol unit was returned
                context.resolved.insert(
                    index as u64,
                    Resolved {
                        protocol_unit_id: id.clone(),
                        protocol_unit: unit.clone(),
                        locally_resolved: true,
                    },
                );
                continue;
            }

            match self.backend.protocol_unit(id) {
                Some(ret) => {
                    context.resolved.insert(
                        index as u64,
                        Resolved {
                            protocol_unit_id: id.clone(),
                            protocol_unit: ret,
                            locally_resolved: true,
                        },
                    );
                }
                None => {
                    context.local_miss.insert(index as u64, id.clone());
                }
            }
        }

        Ok(context)
    }

    /// Fetches the missing entries from the server
    async fn resolve_misses<Request>(
        &self,
        compact_response: InitialResponse<DownloadUnitId, ProtocolUnitId, ProtocolUnit>,
        context: ResolveContext<ProtocolUnitId, ProtocolUnit>,
        network_peer_handle: &NetworkPeerHandle,
    ) -> Result<Vec<Resolved<ProtocolUnitId, ProtocolUnit>>, RelayError>
    where
        Request: From<CompactBlockRequest<DownloadUnitId, ProtocolUnitId>> + Encode + Send + Sync,
    {
        let ResolveContext {
            mut resolved,
            local_miss,
        } = context;
        let missing = local_miss.len();
        // Request the missing entries from the server
        let request = CompactBlockRequest::MissingEntries(MissingEntriesRequest {
            download_unit_id: compact_response.download_unit_id.clone(),
            protocol_unit_ids: local_miss.clone(),
        });
        let response: CompactBlockResponse<DownloadUnitId, ProtocolUnitId, ProtocolUnit> =
            network_peer_handle.request(Request::from(request)).await?;
        let missing_entries_response =
            if let CompactBlockResponse::MissingEntries(response) = response {
                response
            } else {
                return Err(RelayError::UnexpectedProtocolRespone);
            };

        if missing_entries_response.protocol_units.len() != missing {
            return Err(RelayError::ResolveMismatch {
                expected: missing,
                actual: missing_entries_response.protocol_units.len(),
            });
        }

        // Merge the resolved entries from the server
        for (missing_key, protocol_unit_id) in local_miss.into_iter() {
            if let Some(protocol_unit) = missing_entries_response.protocol_units.get(&missing_key) {
                resolved.insert(
                    missing_key,
                    Resolved {
                        protocol_unit_id,
                        protocol_unit: protocol_unit.clone(),
                        locally_resolved: false,
                    },
                );
            } else {
                return Err(RelayError::ResolvedNotFound(missing));
            }
        }

        Ok(resolved.into_values().collect())
    }
}

#[async_trait]
impl<DownloadUnitId, ProtocolUnitId, ProtocolUnit>
    ProtocolClient<DownloadUnitId, ProtocolUnitId, ProtocolUnit>
    for CompactBlockClient<DownloadUnitId, ProtocolUnitId, ProtocolUnit>
where
    DownloadUnitId: Send + Sync + Encode + Decode + Clone + std::fmt::Debug + 'static,
    ProtocolUnitId: Send + Sync + Encode + Decode + Clone + 'static,
    ProtocolUnit: Send + Sync + Encode + Decode + Clone + 'static,
{
    type Request = CompactBlockRequest<DownloadUnitId, ProtocolUnitId>;
    type Response = CompactBlockResponse<DownloadUnitId, ProtocolUnitId, ProtocolUnit>;

    fn build_initial_request(&self) -> Self::Request {
        CompactBlockRequest::Initial
    }

    async fn resolve_initial_response<Request>(
        &self,
        response: Self::Response,
        network_peer_handle: &NetworkPeerHandle,
    ) -> Result<(DownloadUnitId, Vec<Resolved<ProtocolUnitId, ProtocolUnit>>), RelayError>
    where
        Request: From<Self::Request> + Encode + Send + Sync,
    {
        let compact_response = match response {
            CompactBlockResponse::Initial(compact_response) => compact_response,
            _ => return Err(RelayError::UnexpectedInitialResponse),
        };

        // Try to resolve the hashes locally first.
        let context = self.resolve_local(&compact_response)?;
        if context.resolved.len() == compact_response.protocol_units.len() {
            trace!(
                target: LOG_TARGET,
                "relay::resolve: {:?}: resolved locally[{}]",
                compact_response.download_unit_id,
                compact_response.protocol_units.len()
            );
            return Ok((
                compact_response.download_unit_id,
                context.resolved.into_values().collect(),
            ));
        }

        // Resolve the misses from the server
        let misses = context.local_miss.len();
        let download_unit_id = compact_response.download_unit_id.clone();
        let resolved = self
            .resolve_misses::<Request>(compact_response, context, network_peer_handle)
            .await?;
        trace!(
            target: LOG_TARGET,
            "relay::resolve: {:?}: resolved by server[{},{}]",
            download_unit_id,
            resolved.len(),
            misses,
        );
        Ok((download_unit_id, resolved))
    }
}

pub(crate) struct CompactBlockServer<DownloadUnitId, ProtocolUnitId, ProtocolUnit> {
    backend: Arc<
        dyn ServerBackend<DownloadUnitId, ProtocolUnitId, ProtocolUnit> + Send + Sync + 'static,
    >,
}

impl<DownloadUnitId, ProtocolUnitId, ProtocolUnit>
    CompactBlockServer<DownloadUnitId, ProtocolUnitId, ProtocolUnit>
{
    /// Creates the server.
    pub(crate) fn new(
        backend: Arc<
            dyn ServerBackend<DownloadUnitId, ProtocolUnitId, ProtocolUnit> + Send + Sync + 'static,
        >,
    ) -> Self {
        Self { backend }
    }
}

#[async_trait]
impl<DownloadUnitId, ProtocolUnitId, ProtocolUnit> ProtocolServer<DownloadUnitId>
    for CompactBlockServer<DownloadUnitId, ProtocolUnitId, ProtocolUnit>
where
    DownloadUnitId: Encode + Decode + Clone,
    ProtocolUnitId: Encode + Decode + Clone,
    ProtocolUnit: Encode + Decode,
{
    type Request = CompactBlockRequest<DownloadUnitId, ProtocolUnitId>;
    type Response = CompactBlockResponse<DownloadUnitId, ProtocolUnitId, ProtocolUnit>;

    fn build_initial_response(
        &self,
        download_unit_id: &DownloadUnitId,
        initial_request: Self::Request,
    ) -> Result<Self::Response, RelayError> {
        if !matches!(initial_request, CompactBlockRequest::Initial) {
            return Err(RelayError::UnexpectedInitialRequest);
        }

        // Return the info of the members in the download unit.
        let response = InitialResponse {
            download_unit_id: download_unit_id.clone(),
            protocol_units: self.backend.download_unit_members(download_unit_id)?,
        };
        Ok(CompactBlockResponse::Initial(response))
    }

    fn on_request(&self, request: Self::Request) -> Result<Self::Response, RelayError> {
        let request = match request {
            CompactBlockRequest::MissingEntries(req) => req,
            _ => return Err(RelayError::UnexpectedProtocolRequest),
        };

        let mut protocol_units = BTreeMap::new();
        let total_len = request.protocol_unit_ids.len();
        for (missing_id, protocol_unit_id) in request.protocol_unit_ids {
            if let Some(protocol_unit) = self
                .backend
                .protocol_unit(&request.download_unit_id, &protocol_unit_id)
            {
                protocol_units.insert(missing_id, protocol_unit);
            } else {
                warn!(
                    target: LOG_TARGET,
                    "relay::on_request: missing entry not found"
                );
            }
        }
        if total_len != protocol_units.len() {
            warn!(
                target: LOG_TARGET,
                "relay::compact_blocks::on_request: could not resolve all entries: {total_len}/{}",
                protocol_units.len()
            );
        }
        Ok(CompactBlockResponse::MissingEntries(
            MissingEntriesResponse { protocol_units },
        ))
    }
}
