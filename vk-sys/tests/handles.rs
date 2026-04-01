//! Handle round-trip tests,verify `from_raw`, `as_raw`, `null`, `is_null`
//! for representative dispatchable and non-dispatchable handle types.

use vk_sys::handles::*;

// ── Dispatchable handles (usize) ───────────────────────────────────

#[test]
fn instance_round_trip() {
    let h = Instance::from_raw(0xABCD);
    assert_eq!(h.as_raw(), 0xABCD);
    assert!(!h.is_null());
}

#[test]
fn instance_null() {
    let h = Instance::null();
    assert!(h.is_null());
    assert_eq!(h.as_raw(), 0);
}

#[test]
fn device_round_trip() {
    let h = Device::from_raw(42);
    assert_eq!(h.as_raw(), 42);
    assert!(!h.is_null());
}

#[test]
fn physical_device_round_trip() {
    let h = PhysicalDevice::from_raw(0xFF00);
    assert_eq!(h.as_raw(), 0xFF00);
}

#[test]
fn command_buffer_round_trip() {
    let h = CommandBuffer::from_raw(999);
    assert_eq!(h.as_raw(), 999);
}

// ── Non-dispatchable handles (u64) ─────────────────────────────────

#[test]
fn buffer_round_trip() {
    let h = Buffer::from_raw(0xDEAD_BEEF);
    assert_eq!(h.as_raw(), 0xDEAD_BEEF);
    assert!(!h.is_null());
}

#[test]
fn buffer_null() {
    let h = Buffer::null();
    assert!(h.is_null());
    assert_eq!(h.as_raw(), 0);
}

#[test]
fn surface_khr_round_trip() {
    let h = SurfaceKHR::from_raw(0x1234_5678_9ABC_DEF0);
    assert_eq!(h.as_raw(), 0x1234_5678_9ABC_DEF0);
}

#[test]
fn image_round_trip() {
    let h = Image::from_raw(7);
    assert_eq!(h.as_raw(), 7);
}

#[test]
fn fence_round_trip() {
    let h = Fence::from_raw(0);
    assert!(h.is_null());
}

// ── Default impl ───────────────────────────────────────────────────

#[test]
fn default_is_null() {
    assert!(Instance::default().is_null());
    assert!(Device::default().is_null());
    assert!(Buffer::default().is_null());
    assert!(SurfaceKHR::default().is_null());
}

// ── Copy / Eq ──────────────────────────────────────────────────────

#[test]
fn handles_are_copy_and_eq() {
    let a = Buffer::from_raw(42);
    let b = a; // Copy
    assert_eq!(a, b);

    let c = Instance::from_raw(1);
    let d = Instance::from_raw(2);
    assert_ne!(c, d);
}
