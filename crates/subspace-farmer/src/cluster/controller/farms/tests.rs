use crate::cluster::controller::farms::FarmsAddRemoveStreamMap;
use futures::StreamExt;
use std::task::Context;
use std::time::Duration;
use tokio::time::{sleep, timeout};

fn assert_is_terminated<'a, R: 'a>(stream_map: &FarmsAddRemoveStreamMap<'a, R>) {
    assert!(stream_map.in_progress.is_empty());
    assert!(stream_map.farms_to_add_remove.is_empty());
    assert!(stream_map.is_terminated);
}

#[test]
fn test_stream_map_default() {
    let stream_map = FarmsAddRemoveStreamMap::<()>::default();
    assert_is_terminated(&stream_map);
}

#[test]
fn test_stream_map_push() {
    let mut stream_map = FarmsAddRemoveStreamMap::default();

    let farm_index = 1;
    let fut = Box::pin(async { () });
    stream_map.push(farm_index, fut);
    assert!(stream_map.farms_to_add_remove.is_empty());
    assert!(stream_map.in_progress.contains_key(&farm_index));
    assert!(!stream_map.is_terminated);
}

#[test]
fn test_stream_map_poll_next_entry() {
    let mut stream_map = FarmsAddRemoveStreamMap::default();

    let fut = Box::pin(async { () });
    stream_map.push(0, fut);

    let mut cx = Context::from_waker(futures::task::noop_waker_ref());
    let poll_result = stream_map.poll_next_entry(&mut cx);
    assert!(poll_result.is_ready());
    assert_is_terminated(&stream_map);
}

#[tokio::test]
async fn test_stream_map_stream() {
    let mut stream_map = FarmsAddRemoveStreamMap::default();

    let fut00 = Box::pin(async {
        sleep(Duration::from_millis(1)).await;
        0x00
    });
    stream_map.push(0, fut00);

    let next_item = timeout(Duration::from_millis(3), stream_map.next()).await;
    assert_eq!(next_item.unwrap(), Some(0x00));
    assert_is_terminated(&stream_map);

    let fut11 = Box::pin(async {
        sleep(Duration::from_millis(1)).await;
        0x11
    });
    let fut12 = Box::pin(async {
        sleep(Duration::from_millis(1)).await;
        0x12
    });
    let fut13 = Box::pin(async {
        sleep(Duration::from_millis(1)).await;
        0x13
    });
    let fut21 = Box::pin(async {
        sleep(Duration::from_millis(10)).await;
        0x21
    });
    let fut22 = Box::pin(async {
        sleep(Duration::from_millis(1)).await;
        0x22
    });
    stream_map.push(1, fut11);
    stream_map.push(1, fut12);
    assert!(!stream_map.is_terminated);
    assert_eq!(stream_map.in_progress.len(), 1);
    assert!(stream_map.in_progress.contains_key(&1));
    assert_eq!(stream_map.farms_to_add_remove.len(), 1);

    stream_map.push(2, fut21);
    assert_eq!(stream_map.in_progress.len(), 2);
    assert!(stream_map.in_progress.contains_key(&2));
    assert_eq!(stream_map.farms_to_add_remove.len(), 1);

    stream_map.push(2, fut22);
    assert_eq!(stream_map.in_progress.len(), 2);
    assert_eq!(stream_map.farms_to_add_remove.len(), 2);
    assert_eq!(stream_map.farms_to_add_remove[&2].len(), 1);

    stream_map.push(1, fut13);
    assert!(!stream_map.is_terminated);
    assert!(stream_map.in_progress.contains_key(&1));
    assert_eq!(stream_map.in_progress.len(), 2);
    assert_eq!(stream_map.farms_to_add_remove[&1].len(), 2);

    let next_item = stream_map.next().await;
    assert!(!stream_map.is_terminated);
    assert_eq!(next_item.unwrap(), 0x11);
    assert!(stream_map.in_progress.contains_key(&1));
    assert!(stream_map.in_progress.contains_key(&2));
    assert_eq!(stream_map.in_progress.len(), 2);
    assert_eq!(stream_map.farms_to_add_remove[&1].len(), 1);

    let next_item = stream_map.next().await;
    assert!(!stream_map.is_terminated);
    assert_eq!(next_item.unwrap(), 0x12);
    assert_eq!(stream_map.in_progress.len(), 2);
    assert!(stream_map.in_progress.contains_key(&1));
    assert!(stream_map.in_progress.contains_key(&2));
    assert!(stream_map.farms_to_add_remove.get(&1).is_none());

    let next_item = stream_map.next().await;
    assert!(!stream_map.is_terminated);
    assert_eq!(next_item.unwrap(), 0x13);
    assert_eq!(stream_map.in_progress.len(), 1);
    assert!(!stream_map.in_progress.contains_key(&1));
    assert!(stream_map.in_progress.contains_key(&2));
    assert!(stream_map.farms_to_add_remove.get(&1).is_none());
    assert_eq!(stream_map.farms_to_add_remove[&2].len(), 1);

    let next_item = stream_map.next().await;
    assert!(!stream_map.is_terminated);
    assert_eq!(next_item.unwrap(), 0x21);
    assert_eq!(stream_map.in_progress.len(), 1);
    assert!(!stream_map.in_progress.contains_key(&1));
    assert!(stream_map.in_progress.contains_key(&2));
    assert!(stream_map.farms_to_add_remove.get(&1).is_none());
    assert!(stream_map.farms_to_add_remove.get(&2).is_none());

    let next_item = timeout(Duration::from_millis(3), stream_map.next()).await;
    assert_eq!(next_item.unwrap(), Some(0x22));
    assert_is_terminated(&stream_map);
}
