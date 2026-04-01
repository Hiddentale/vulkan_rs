//! SPIR-V bytecode alignment helper.

use std::fmt;

/// Error returned when SPIR-V bytecode has invalid alignment or size.
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

/// Cast a byte slice to a `u32` slice for `ShaderModuleCreateInfo`.
///
/// Returns an error if `bytes.len()` is not a multiple of 4 or the
/// pointer is not 4-byte aligned. Use `include_bytes!` or aligned
/// allocators to ensure correctness.
///
/// # Examples
///
/// ```
/// use vk_engine::cast_to_u32;
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
    if !bytes.len().is_multiple_of(4) {
        return Err(BytecodeError::InvalidLength(bytes.len()));
    }
    if !(bytes.as_ptr() as usize).is_multiple_of(4) {
        return Err(BytecodeError::MisalignedPointer);
    }
    // SAFETY: length and alignment checked above, pointer is valid, aligned to u32, and in-bounds.
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_input_succeeds() {
        #[repr(align(4))]
        struct Aligned([u8; 8]);
        let data = Aligned([0x03, 0x02, 0x23, 0x07, 0, 0, 0, 0]);
        let words = cast_to_u32(&data.0).expect("aligned input should succeed");
        assert_eq!(words.len(), 2);
        assert_eq!(words[0], 0x07230203); // SPIR-V magic number (little-endian)
    }

    #[test]
    fn invalid_length_returns_error() {
        #[repr(align(4))]
        struct Aligned([u8; 5]);
        let data = Aligned([1, 2, 3, 4, 5]);
        assert_eq!(cast_to_u32(&data.0), Err(BytecodeError::InvalidLength(5)));
    }

    #[test]
    fn empty_input_succeeds() {
        let empty: &[u8] = &[];
        let words = cast_to_u32(empty).expect("empty input should succeed");
        assert!(words.is_empty());
    }

    #[test]
    fn misaligned_pointer_display() {
        let err = BytecodeError::MisalignedPointer;
        assert_eq!(
            err.to_string(),
            "SPIR-V byte slice pointer is not 4-byte aligned"
        );
    }
}
