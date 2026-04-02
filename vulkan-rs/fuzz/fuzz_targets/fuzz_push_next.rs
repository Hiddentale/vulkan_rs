#![no_main]

use libfuzzer_sys::fuzz_target;
use vulkan_rs::vk::enums::StructureType;
use vulkan_rs::vk::structs::*;

/// Walk a pNext chain starting from `head`, collecting sType values.
/// Returns None if the chain exceeds `max_depth` (likely a cycle).
unsafe fn walk_chain(head: *const core::ffi::c_void, max_depth: usize) -> Option<Vec<i32>> {
    let mut types = Vec::new();
    let mut current = head as *const BaseOutStructure;
    for _ in 0..max_depth {
        if current.is_null() {
            return Some(types);
        }
        let s = unsafe { &*current };
        types.push(s.s_type.as_raw());
        current = s.p_next;
    }
    None
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    // Three structs that all extend DeviceCreateInfo, each with a distinct sType.
    let mut features_v11 = PhysicalDeviceVulkan11Features::default();
    let mut features_v12 = PhysicalDeviceVulkan12Features::default();
    let mut features_v13 = PhysicalDeviceVulkan13Features::default();

    let expected = [
        StructureType::PHYSICAL_DEVICE_VULKAN_1_1_FEATURES.as_raw(),
        StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES.as_raw(),
        StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES.as_raw(),
    ];

    // Verify sTypes are set correctly by builder default.
    assert_eq!(features_v11.s_type.as_raw(), expected[0]);
    assert_eq!(features_v12.s_type.as_raw(), expected[1]);
    assert_eq!(features_v13.s_type.as_raw(), expected[2]);

    // Chain all 3 in fuzzer-controlled order (6 permutations).
    let builder = match data[0] % 6 {
        0 => DeviceCreateInfo::builder()
            .push_next(&mut features_v11)
            .push_next(&mut features_v12)
            .push_next(&mut features_v13),
        1 => DeviceCreateInfo::builder()
            .push_next(&mut features_v11)
            .push_next(&mut features_v13)
            .push_next(&mut features_v12),
        2 => DeviceCreateInfo::builder()
            .push_next(&mut features_v12)
            .push_next(&mut features_v11)
            .push_next(&mut features_v13),
        3 => DeviceCreateInfo::builder()
            .push_next(&mut features_v12)
            .push_next(&mut features_v13)
            .push_next(&mut features_v11),
        4 => DeviceCreateInfo::builder()
            .push_next(&mut features_v13)
            .push_next(&mut features_v11)
            .push_next(&mut features_v12),
        _ => DeviceCreateInfo::builder()
            .push_next(&mut features_v13)
            .push_next(&mut features_v12)
            .push_next(&mut features_v11),
    };

    let info = &*builder;

    // Walk the chain and verify invariants.
    let chain = unsafe { walk_chain(info.p_next, 10) }
        .expect("pNext chain exceeded max depth (cycle?)");

    // All 3 structs must be reachable.
    assert_eq!(chain.len(), 3, "expected 3 structs in pNext chain");

    // Every sType must be one of the three we pushed.
    for stype in &chain {
        assert!(
            expected.contains(stype),
            "unexpected sType {stype} in pNext chain",
        );
    }

    // All three must appear exactly once.
    let mut found = chain.clone();
    found.sort();
    found.dedup();
    assert_eq!(found.len(), 3, "pNext chain has duplicate or missing sTypes");
});
