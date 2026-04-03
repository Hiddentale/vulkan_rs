//! Byte-level helpers for Vulkan data: SPIR-V alignment and raw struct access.

use std::fmt;

/// SPIR-V instructions are 32-bit words, so both length and alignment
/// must be multiples of this value.
const SPIRV_WORD_SIZE: usize = 4;

/// Error returned when SPIR-V bytecode has invalid alignment or size.
///
/// # Examples
///
/// ```
/// use vulkan_rust::BytecodeError;
///
/// let err = BytecodeError::InvalidLength(7);
/// assert_eq!(err.to_string(), "SPIR-V byte length 7 is not a multiple of 4");
///
/// let err = BytecodeError::MisalignedPointer;
/// assert!(err.to_string().contains("not 4-byte aligned"));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BytecodeError {
    /// Length is not a multiple of 4.
    InvalidLength(usize),
    /// Pointer is not aligned to 4 bytes.
    MisalignedPointer,
}

impl fmt::Display for BytecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength(len) => {
                write!(f, "SPIR-V byte length {len} is not a multiple of 4")
            }
            Self::MisalignedPointer => {
                write!(f, "SPIR-V byte slice pointer is not 4-byte aligned")
            }
        }
    }
}

impl std::error::Error for BytecodeError {}

/// Reinterprets a reference to a `Copy` type as a byte slice.
///
/// This is useful for passing `#[repr(C)]` structs to Vulkan commands
/// that accept `&[u8]`, such as `cmd_push_constants`.
///
/// # Safety
///
/// `T` must not contain padding bytes. Reading uninitialized padding is
/// undefined behavior. Use `#[repr(C)]` structs whose fields leave no
/// gaps, or verify with `assert_eq!(size_of::<T>(), /* expected */)`.
///
/// # Examples
///
/// ```
/// use vulkan_rust::as_bytes;
///
/// #[repr(C)]
/// #[derive(Clone, Copy)]
/// struct PushData { time: f32, scale: f32 }
///
/// let data = PushData { time: 1.0, scale: 2.0 };
/// let bytes = unsafe { as_bytes(&data) };
/// assert_eq!(bytes.len(), 8);
/// ```
pub unsafe fn as_bytes<T: Copy>(value: &T) -> &[u8] {
    // SAFETY: caller guarantees T has no padding. Casting to u8 is always
    // aligned, and the slice covers exactly size_of::<T>() bytes.
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) }
}

/// Cast a byte slice to a `u32` slice for `ShaderModuleCreateInfo`.
///
/// Returns an error if `bytes.len()` is not a multiple of 4 or the
/// pointer is not 4-byte aligned. Use `include_bytes!` or aligned
/// allocators to ensure correctness.
///
/// # Examples
///
/// ```
/// use vulkan_rust::cast_to_u32;
///
/// #[repr(align(4))]
/// struct Aligned([u8; 8]);
/// let spirv = Aligned([0x03, 0x02, 0x23, 0x07, 0, 0, 0, 0]);
/// let words = cast_to_u32(&spirv.0).expect("alignment error");
/// assert_eq!(words.len(), 2);
/// ```
pub fn cast_to_u32(bytes: &[u8]) -> Result<&[u32], BytecodeError> {
    if bytes.is_empty() {
        return Ok(&[]);
    }
    if bytes.len() % SPIRV_WORD_SIZE != 0 {
        return Err(BytecodeError::InvalidLength(bytes.len()));
    }
    if (bytes.as_ptr() as usize) % SPIRV_WORD_SIZE != 0 {
        return Err(BytecodeError::MisalignedPointer);
    }
    // SAFETY: length and alignment checked above, pointer is valid, aligned to u32, and in-bounds.
    Ok(unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / SPIRV_WORD_SIZE)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn misaligned_pointer_display() {
        let err = BytecodeError::MisalignedPointer;
        assert_eq!(
            err.to_string(),
            "SPIR-V byte slice pointer is not 4-byte aligned"
        );
    }

    #[test]
    fn invalid_length_display() {
        let err = BytecodeError::InvalidLength(7);
        assert_eq!(
            err.to_string(),
            "SPIR-V byte length 7 is not a multiple of 4"
        );
    }

    #[test]
    fn misaligned_pointer_returns_error() {
        // Create a 4-byte-aligned buffer, then take a subslice at offset 1
        // to guarantee misalignment.
        #[repr(align(4))]
        struct Aligned([u8; 8]);
        let data = Aligned([0; 8]);
        let misaligned = &data.0[1..5];
        assert_eq!(
            cast_to_u32(misaligned),
            Err(BytecodeError::MisalignedPointer)
        );
    }

    #[test]
    fn bytecode_error_is_std_error() {
        let err: &dyn std::error::Error = &BytecodeError::InvalidLength(3);
        // source() should return None (no underlying cause).
        assert!(err.source().is_none());
    }
}
