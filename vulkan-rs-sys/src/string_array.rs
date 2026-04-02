//! Fixed-size null-terminated string wrapper for Vulkan `[c_char; N]` fields.
//!
//! [`StringArray<N>`] is a `#[repr(transparent)]` wrapper around `[c_char; N]`
//! that provides `Display`, `Debug`, `PartialEq`, and `Hash` implementations
//! which compare only up to the first null terminator. This makes it safe to
//! compare extension names, layer names, and device names directly.

use core::ffi::{c_char, CStr};
use core::{fmt, hash};

use crate::constants::{
    MAX_DESCRIPTION_SIZE, MAX_DRIVER_INFO_SIZE, MAX_DRIVER_NAME_SIZE, MAX_EXTENSION_NAME_SIZE,
    MAX_PHYSICAL_DEVICE_NAME_SIZE,
};

/// Extension name string, e.g. from `ExtensionProperties::extension_name`.
pub type ExtensionName = StringArray<{ MAX_EXTENSION_NAME_SIZE as usize }>;

/// Layer name string, e.g. from `LayerProperties::layer_name`.
pub type LayerName = StringArray<{ MAX_EXTENSION_NAME_SIZE as usize }>;

/// Physical device name string, from `PhysicalDeviceProperties::device_name`.
pub type DeviceName = StringArray<{ MAX_PHYSICAL_DEVICE_NAME_SIZE as usize }>;

/// Description string, from `LayerProperties::description`.
pub type DescriptionName = StringArray<{ MAX_DESCRIPTION_SIZE as usize }>;

/// Driver name string, from `PhysicalDeviceDriverProperties::driver_name`.
pub type DriverName = StringArray<{ MAX_DRIVER_NAME_SIZE as usize }>;

/// Driver info string, from `PhysicalDeviceDriverProperties::driver_info`.
pub type DriverInfo = StringArray<{ MAX_DRIVER_INFO_SIZE as usize }>;

/// A fixed-size array containing a null-terminated C string.
///
/// Wraps the raw `[c_char; N]` arrays used by Vulkan structs
/// (e.g. `ExtensionProperties::extension_name`). Equality and hashing
/// ignore bytes after the first null terminator.
///
/// # Examples
///
/// ```no_run
/// use vulkan_rs_sys::StringArray;
///
/// let a = StringArray::<256>::from_cstr(c"VK_KHR_swapchain");
/// let b = StringArray::<256>::from_cstr(c"VK_KHR_swapchain");
/// assert_eq!(a, b);
/// ```
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct StringArray<const N: usize>(pub [c_char; N]);

impl<const N: usize> StringArray<N> {
    /// View the contents as a `&CStr`.
    ///
    /// If the array contains no null terminator (shouldn't happen with
    /// well-formed Vulkan data), treats the entire array as the string.
    #[inline]
    pub fn as_cstr(&self) -> &CStr {
        let bytes: &[u8] = unsafe { core::slice::from_raw_parts(self.0.as_ptr().cast(), N) };
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(N.saturating_sub(1));
        unsafe { CStr::from_bytes_with_nul_unchecked(&bytes[..end + 1]) }
    }

    /// Construct from a `&CStr`, truncating if longer than `N - 1`.
    pub fn from_cstr(cstr: &CStr) -> Self {
        let mut array = [0 as c_char; N];
        let bytes = cstr.to_bytes();
        let len = bytes.len().min(N.saturating_sub(1));
        for i in 0..len {
            array[i] = bytes[i] as c_char;
        }
        Self(array)
    }
}

impl<const N: usize> core::ops::Deref for StringArray<N> {
    type Target = [c_char; N];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const N: usize> Default for StringArray<N> {
    #[inline]
    fn default() -> Self {
        Self([0; N])
    }
}

impl<const N: usize> PartialEq for StringArray<N> {
    fn eq(&self, other: &Self) -> bool {
        self.as_cstr() == other.as_cstr()
    }
}

impl<const N: usize> Eq for StringArray<N> {}

impl<const N: usize> hash::Hash for StringArray<N> {
    fn hash<H: hash::Hasher>(&self, hasher: &mut H) {
        self.as_cstr().hash(hasher);
    }
}

impl<const N: usize> PartialEq<&CStr> for StringArray<N> {
    fn eq(&self, other: &&CStr) -> bool {
        self.as_cstr() == *other
    }
}

impl<const N: usize> fmt::Display for StringArray<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.as_cstr().to_str() {
            Ok(s) => f.write_str(s),
            Err(_) => write!(f, "{:?}", self.as_cstr()),
        }
    }
}

impl<const N: usize> fmt::Debug for StringArray<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"{}\"", self)
    }
}

impl<const N: usize> From<[c_char; N]> for StringArray<N> {
    #[inline]
    fn from(array: [c_char; N]) -> Self {
        Self(array)
    }
}

impl<const N: usize> From<StringArray<N>> for [c_char; N] {
    #[inline]
    fn from(array: StringArray<N>) -> Self {
        array.0
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::format;
    use super::*;
    use core::hash::Hasher;

    fn hash_of(v: impl hash::Hash) -> u64 {
        let mut h = SimpleHasher(0);
        v.hash(&mut h);
        h.0
    }

    // Minimal hasher for no_std tests.
    struct SimpleHasher(u64);
    impl Hasher for SimpleHasher {
        fn write(&mut self, bytes: &[u8]) {
            for &b in bytes {
                self.0 = self.0.wrapping_mul(31).wrapping_add(b as u64);
            }
        }
        fn finish(&self) -> u64 {
            self.0
        }
    }

    #[test]
    fn equal_strings_are_equal() {
        let a = StringArray::<256>::from_cstr(c"VK_KHR_swapchain");
        let b = StringArray::<256>::from_cstr(c"VK_KHR_swapchain");
        assert_eq!(a, b);
        assert_eq!(hash_of(a), hash_of(b));
    }

    #[test]
    fn different_strings_are_not_equal() {
        let a = StringArray::<256>::from_cstr(c"VK_KHR_swapchain");
        let b = StringArray::<256>::from_cstr(c"VK_KHR_surface");
        assert_ne!(a, b);
    }

    #[test]
    fn trailing_garbage_ignored() {
        let mut arr_a = [0 as c_char; 8];
        let mut arr_b = [0 as c_char; 8];
        arr_a[0] = b'a' as c_char;
        arr_a[1] = 0;
        arr_a[2] = b'X' as c_char;
        arr_b[0] = b'a' as c_char;
        arr_b[1] = 0;
        arr_b[2] = b'Y' as c_char;
        assert_eq!(StringArray(arr_a), StringArray(arr_b));
        assert_eq!(hash_of(StringArray(arr_a)), hash_of(StringArray(arr_b)));
    }

    #[test]
    fn compare_with_cstr() {
        let a = StringArray::<256>::from_cstr(c"VK_KHR_swapchain");
        assert_eq!(a, c"VK_KHR_swapchain");
    }

    #[test]
    fn display_shows_string() {
        let a = StringArray::<256>::from_cstr(c"hello");
        let s = format!("{a}");
        assert_eq!(s, "hello");
    }

    #[test]
    fn default_is_empty() {
        let a = StringArray::<32>::default();
        assert_eq!(a, c"");
    }

    #[test]
    fn from_cstr_truncates_long_string() {
        let a = StringArray::<4>::from_cstr(c"abcdef");
        assert_eq!(a.as_cstr(), c"abc");
    }

    #[test]
    fn round_trip_array() {
        let orig = [b'h' as c_char, b'i' as c_char, 0, 0];
        let sa = StringArray::from(orig);
        let back: [c_char; 4] = sa.into();
        assert_eq!(orig, back);
    }
}
