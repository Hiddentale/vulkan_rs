#[test]
fn cast_aligned_spv_succeeds() {
    // 8 bytes of fake SPIR-V (aligned by repr(align(4)) in static data).
    #[repr(align(4))]
    struct Aligned([u8; 8]);
    let data = Aligned([0x03, 0x02, 0x23, 0x07, 0, 0, 0, 0]);
    let words = vk_engine::cast_to_u32(&data.0).unwrap();
    assert_eq!(words.len(), 2);
    assert_eq!(words[0], 0x07230203); // SPIR-V magic number (little-endian)
}

#[test]
fn cast_rejects_non_multiple_of_four() {
    #[repr(align(4))]
    struct Aligned([u8; 7]);
    let data = Aligned([1, 2, 3, 4, 5, 6, 7]);
    let err = vk_engine::cast_to_u32(&data.0).unwrap_err();
    assert_eq!(err, vk_engine::BytecodeError::InvalidLength(7));
}

#[test]
fn cast_empty_slice_succeeds() {
    let words = vk_engine::cast_to_u32(&[]).unwrap();
    assert!(words.is_empty());
}
