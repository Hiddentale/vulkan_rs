//! Platform-aware Vulkan surface creation and extension helpers.

use std::ffi::CStr;

/// Instance extensions required for surface creation on this platform.
///
/// Always includes `VK_KHR_surface`. Adds the platform-specific
/// surface extension based on `#[cfg(target_os)]`.
pub fn required_extensions() -> &'static [&'static CStr] {
    #[cfg(target_os = "windows")]
    {
        &[c"VK_KHR_surface", c"VK_KHR_win32_surface"]
    }
    #[cfg(all(
        unix,
        not(target_os = "android"),
        not(target_os = "macos"),
        not(target_os = "ios"),
    ))]
    {
        // Wayland and X11 — return both; the loader ignores missing ones
        // at enumerate time, and the user can filter to what's available.
        &[
            c"VK_KHR_surface",
            c"VK_KHR_xlib_surface",
            c"VK_KHR_wayland_surface",
        ]
    }
    #[cfg(target_os = "macos")]
    {
        &[c"VK_KHR_surface", c"VK_EXT_metal_surface"]
    }
    #[cfg(target_os = "android")]
    {
        &[c"VK_KHR_surface", c"VK_KHR_android_surface"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_extensions_is_non_empty() {
        assert!(!required_extensions().is_empty());
    }

    #[test]
    fn first_extension_is_khr_surface() {
        assert_eq!(required_extensions()[0], c"VK_KHR_surface");
    }
}
