use crate::protocols::peer_info::{protocol, PeerInfo};
use futures::future::BoxFuture;
use futures::prelude::*;
use libp2p::core::upgrade::ReadyUpgrade;
use libp2p::swarm::handler::{
    ConnectionEvent, DialUpgradeError, FullyNegotiatedInbound, FullyNegotiatedOutbound,
    ListenUpgradeError, StreamUpgradeError as ConnectionHandlerUpgrErr,
};
use libp2p::swarm::{
    ConnectionHandler, ConnectionHandlerEvent, Stream as NegotiatedSubstream, SubstreamProtocol,
};
use libp2p::StreamProtocol;
use std::collections::VecDeque;
use std::error::Error;
use std::io;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use std::time::Duration;
use tracing::debug;

/// The configuration for peer-info protocol.
#[derive(Debug, Clone)]
pub struct Config {
    /// Protocol timeout.
    timeout: Duration,

    /// Protocol name.
    protocol_name: &'static str,
}

impl Config {
    /// Creates a new [`Config`] with the following default settings:
    ///
    ///   * [`Config::with_timeout`] 20s
    pub fn new(protocol_name: &'static str) -> Self {
        Self {
            timeout: Duration::from_secs(20),
            protocol_name,
        }
    }

    /// Sets the protocol timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// The successful result of processing an inbound or outbound peer info requests.
#[derive(Debug)]
pub enum PeerInfoSuccess {
    /// Local peer received peer info from a remote peer.
    Received(PeerInfo),
    /// Local peer sent its peer info to a remote peer.
    Sent,
}

/// A peer info protocol failure.
#[derive(Debug, thiserror::Error)]
pub enum PeerInfoError {
    /// The peer does not support the peer info protocol.
    #[error("Peer info protocol is not supported.")]
    #[allow(dead_code)] // We preserve errors on dial upgrades for future use.
    Unsupported,
    /// The peer info request failed.
    #[error("Peer info error: {error}")]
    Other {
        #[source]
        error: Box<dyn Error + Send + 'static>,
    },
}

/// Struct for outbound peer-info requests.
#[derive(Debug, Clone)]
pub struct HandlerInEvent {
    pub peer_info: Arc<PeerInfo>,
}

/// Protocol handler that handles peer-info requests.
///
/// Any protocol failure produces an error that closes the connection.
pub struct Handler {
    /// Configuration options.
    config: Config,
    /// The outbound request state.
    outbound: Option<OutboundState>,
    /// The inbound request future.
    inbound: Option<InPeerInfoFuture>,
    /// Queue of events to return when polled.
    queued_events: VecDeque<
        ConnectionHandlerEvent<
            <Self as ConnectionHandler>::OutboundProtocol,
            <Self as ConnectionHandler>::OutboundOpenInfo,
            <Self as ConnectionHandler>::ToBehaviour,
        >,
    >,
    /// Future waker.
    waker: Option<Waker>,
}

impl Handler {
    /// Builds a new [`Handler`] with the given configuration.
    pub fn new(config: Config) -> Self {
        Handler {
            config,
            outbound: None,
            inbound: None,
            queued_events: VecDeque::default(),
            waker: None,
        }
    }

    fn wake(&self) {
        if let Some(waker) = &self.waker {
            waker.wake_by_ref()
        }
    }
}

impl ConnectionHandler for Handler {
    type FromBehaviour = HandlerInEvent;
    type ToBehaviour = Result<PeerInfoSuccess, PeerInfoError>;
    type InboundProtocol = ReadyUpgrade<StreamProtocol>;
    type OutboundProtocol = ReadyUpgrade<StreamProtocol>;
    type OutboundOpenInfo = Arc<PeerInfo>;
    type InboundOpenInfo = ();

    fn listen_protocol(&self) -> SubstreamProtocol<ReadyUpgrade<StreamProtocol>, ()> {
        SubstreamProtocol::new(
            ReadyUpgrade::new(StreamProtocol::new(self.config.protocol_name)),
            (),
        )
    }

    fn on_behaviour_event(&mut self, event: Self::FromBehaviour) {
        if let Some(OutboundState::Idle(stream)) = self.outbound.take() {
            self.outbound = Some(OutboundState::SendingData(
                protocol::send(stream, event.peer_info).boxed(),
            ));
        } else {
            self.outbound = Some(OutboundState::RequestNewStream(event.peer_info));
        }
        self.wake();
    }

    fn poll(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<
        ConnectionHandlerEvent<
            ReadyUpgrade<StreamProtocol>,
            Self::OutboundOpenInfo,
            Result<PeerInfoSuccess, PeerInfoError>,
        >,
    > {
        // Return queued events.
        if let Some(event) = self.queued_events.pop_front() {
            return Poll::Ready(event);
        }

        // Respond to inbound requests.
        if let Some(fut) = self.inbound.as_mut() {
            match fut.poll_unpin(cx) {
                Poll::Pending => {}
                Poll::Ready(Err(error)) => {
                    debug!(?error, "Peer info handler: inbound peer info error.");

                    self.inbound = None;
                    return Poll::Ready(ConnectionHandlerEvent::NotifyBehaviour(Err(
                        PeerInfoError::Other {
                            error: Box::new(error),
                        },
                    )));
                }
                Poll::Ready(Ok((stream, peer_info))) => {
                    debug!(?peer_info, "Inbound peer info");

                    self.inbound = Some(protocol::recv(stream).boxed());
                    return Poll::Ready(ConnectionHandlerEvent::NotifyBehaviour(Ok(
                        PeerInfoSuccess::Received(peer_info),
                    )));
                }
            }
        }

        // Outbound requests.
        match self.outbound.take() {
            Some(OutboundState::SendingData(mut peer_info_fut)) => {
                match peer_info_fut.poll_unpin(cx) {
                    Poll::Pending => {
                        self.outbound = Some(OutboundState::SendingData(peer_info_fut));
                    }
                    Poll::Ready(Ok(stream)) => {
                        self.outbound = Some(OutboundState::Idle(stream));

                        return Poll::Ready(ConnectionHandlerEvent::NotifyBehaviour(Ok(
                            PeerInfoSuccess::Sent,
                        )));
                    }
                    Poll::Ready(Err(error)) => {
                        debug!(?error, "Outbound peer info error.",);

                        return Poll::Ready(ConnectionHandlerEvent::NotifyBehaviour(Err(
                            PeerInfoError::Other {
                                error: Box::new(error),
                            },
                        )));
                    }
                }
            }
            Some(OutboundState::Idle(stream)) => {
                // Nothing to do but we have a negotiated stream.
                self.outbound = Some(OutboundState::Idle(stream));
            }
            Some(OutboundState::NegotiatingStream) => {
                self.outbound = Some(OutboundState::NegotiatingStream);
            }
            Some(OutboundState::RequestNewStream(peer_info)) => {
                self.outbound = Some(OutboundState::NegotiatingStream);
                let protocol = SubstreamProtocol::new(
                    ReadyUpgrade::new(StreamProtocol::new(self.config.protocol_name)),
                    peer_info,
                )
                .with_timeout(self.config.timeout);
                return Poll::Ready(ConnectionHandlerEvent::OutboundSubstreamRequest { protocol });
            }
            None => {
                // Not initialized yet.
            }
        }

        self.waker = Some(cx.waker().clone());
        Poll::Pending
    }

    fn on_connection_event(
        &mut self,
        event: ConnectionEvent<
            Self::InboundProtocol,
            Self::OutboundProtocol,
            Self::InboundOpenInfo,
            Self::OutboundOpenInfo,
        >,
    ) {
        match event {
            ConnectionEvent::FullyNegotiatedInbound(FullyNegotiatedInbound {
                protocol: stream,
                ..
            }) => {
                self.inbound = Some(protocol::recv(stream).boxed());
            }
            ConnectionEvent::FullyNegotiatedOutbound(FullyNegotiatedOutbound {
                protocol: stream,
                info,
            }) => {
                self.outbound = Some(OutboundState::SendingData(
                    protocol::send(stream, info).boxed(),
                ));
            }
            ConnectionEvent::DialUpgradeError(DialUpgradeError { error, .. }) => {
                match error {
                    ConnectionHandlerUpgrErr::NegotiationFailed
                    | ConnectionHandlerUpgrErr::Apply(..) => {
                        debug!(?error, "Peer-info protocol dial upgrade failed");
                    }
                    error => {
                        self.queued_events
                            .push_back(ConnectionHandlerEvent::NotifyBehaviour(Err(
                                PeerInfoError::Other {
                                    error: Box::new(error),
                                },
                            )));
                    }
                };
            }
            ConnectionEvent::ListenUpgradeError(ListenUpgradeError { error, .. }) => {
                self.queued_events
                    .push_back(ConnectionHandlerEvent::NotifyBehaviour(Err(
                        PeerInfoError::Other {
                            error: Box::new(error),
                        },
                    )));
            }
            _ => {}
        }
        self.wake();
    }
}

type InPeerInfoFuture = BoxFuture<'static, Result<(NegotiatedSubstream, PeerInfo), io::Error>>;
type OutPeerInfoFuture = BoxFuture<'static, Result<NegotiatedSubstream, io::Error>>;

/// The current state w.r.t. outbound peer info requests.
enum OutboundState {
    RequestNewStream(Arc<PeerInfo>),
    /// A new substream is being negotiated for the protocol.
    NegotiatingStream,
    /// A peer info request is being sent and the response awaited.
    SendingData(OutPeerInfoFuture),
    /// The substream is idle, waiting to send the next peer info request.
    Idle(NegotiatedSubstream),
}