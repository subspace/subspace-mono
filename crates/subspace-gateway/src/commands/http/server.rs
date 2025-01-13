//! HTTP server which fetches objects from the DSN based on a hash, using a mapping indexer service.

use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use std::sync::Arc;
use subspace_core_primitives::hashes::Blake3Hash;
use subspace_data_retrieval::object_fetcher::ObjectFetcher;
use subspace_data_retrieval::piece_getter::PieceGetter;
use subspace_rpc_primitives::ObjectMappingResponse;
use tracing::{debug, error, trace};

/// Parameters for the DSN object HTTP server.
pub(crate) struct ServerParameters<PG>
where
    PG: PieceGetter + Send + Sync + 'static,
{
    pub(crate) object_fetcher: ObjectFetcher<PG>,
    pub(crate) indexer_endpoint: String,
    pub(crate) http_endpoint: String,
}

/// Requests the object mappings for `hashes` from the indexer service.
/// Multiple hashes are separated by `+`.
async fn request_object_mapping(
    endpoint: &str,
    hashes: &[Blake3Hash],
) -> anyhow::Result<ObjectMappingResponse> {
    let client = reqwest::Client::new();
    let hash_list = hashes.iter().map(hex::encode).collect::<Vec<_>>();
    let object_mappings_url = format!("{}/objects/{}", endpoint, hash_list.join("+"));

    debug!(
        ?hashes,
        ?object_mappings_url,
        "Requesting object mappings..."
    );

    let response = client.get(&object_mappings_url).send().await?.json().await;

    match &response {
        Ok(json) => {
            trace!(?hashes, ?json, "Received object mappings");
        }
        Err(err) => {
            error!(?hashes, ?err, ?object_mappings_url, "Request failed");
        }
    }

    response.map_err(|err| err.into())
}

/// Fetches the DSN objects with `hashes`, using the mapping indexer service.
/// Multiple hashes are separated by `+`.
async fn serve_object<PG>(
    hashes: web::Path<String>,
    additional_data: web::Data<Arc<ServerParameters<PG>>>,
) -> impl Responder
where
    PG: PieceGetter + Send + Sync + 'static,
{
    let server_params = additional_data.into_inner();
    let hashes = hashes.into_inner();
    let hashes = hashes
        .split('+')
        .map(|s| {
            let mut hash = Blake3Hash::default();
            hex::decode_to_slice(s, hash.as_mut()).map(|()| hash)
        })
        .try_collect::<Vec<_>>();

    let Ok(hashes) = hashes else {
        return HttpResponse::BadRequest().finish();
    };

    let Ok(object_mappings) =
        request_object_mapping(&server_params.indexer_endpoint, &hashes).await
    else {
        return HttpResponse::BadRequest().finish();
    };

    for object_mapping in object_mappings.objects.objects() {
        if !hashes.contains(&object_mapping.hash) {
            error!(
                ?object_mapping,
                ?hashes,
                "Returned object mapping wasn't in requested hashes"
            );
            return HttpResponse::ServiceUnavailable().finish();
        }
    }

    let object_fetcher_result = server_params
        .object_fetcher
        .fetch_objects(object_mappings.objects)
        .await;

    let objects = match object_fetcher_result {
        Ok(objects) => {
            trace!(
                ?hashes,
                count = %objects.len(),
                sizes = ?objects.iter().map(|object| object.len()),
                "Objects fetched successfully"
            );
            objects
        }
        Err(err) => {
            error!(?hashes, ?err, "Failed to fetch objects");
            return HttpResponse::ServiceUnavailable().finish();
        }
    };

    // TODO: return a multi-part response, with one part per object
    HttpResponse::Ok()
        .content_type("application/octet-stream")
        .body(objects.concat())
}

/// Starts the DSN object HTTP server.
pub async fn start_server<PG>(server_params: ServerParameters<PG>) -> std::io::Result<()>
where
    PG: PieceGetter + Send + Sync + 'static,
{
    let server_params = Arc::new(server_params);
    let http_endpoint = server_params.http_endpoint.clone();
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(server_params.clone()))
            .route("/data/{hashes}", web::get().to(serve_object::<PG>))
    })
    .bind(http_endpoint)?
    .run()
    .await
}
