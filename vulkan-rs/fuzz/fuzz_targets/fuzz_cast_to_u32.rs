#![no_main]

use libfuzzer_sys::fuzz_target;
use vulkan_rs::cast_to_u32;

fuzz_target!(|data: &[u8]| {
    match cast_to_u32(data) {
        Ok(words) => {
            // If it succeeded, the invariants must hold:
            // - original length was a multiple of 4
            assert!(data.len() % 4 == 0);
            // - output length is input length / 4
            assert_eq!(words.len(), data.len() / 4);
            // - every u32 is readable without UB
            for &w in words {
                std::hint::black_box(w);
            }
        }
        Err(vulkan_rs::BytecodeError::InvalidLength(len)) => {
            // Length was not a multiple of 4.
            assert_eq!(len, data.len());
            assert!(len % 4 != 0);
        }
        Err(vulkan_rs::BytecodeError::MisalignedPointer) => {
            // Pointer was not 4-byte aligned, length was valid.
            assert!(data.len() % 4 == 0);
            assert!(data.as_ptr() as usize % 4 != 0);
        }
    }
});
