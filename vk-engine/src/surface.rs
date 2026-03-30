//! Platform-aware Vulkan surface creation and extension helpers.

use std::ffi::CStr;
use std::fmt;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};

use crate::error::check;
use crate::instance::Instance;
use crate::vk;
use vk::handles::Handle;

/// Error returned by surface creation.
#[derive(Debug)]
pub enum SurfaceError {
    /// The display/window handle combination is not supported.
    UnsupportedPlatform,
    /// `raw-window-handle` returned an error.
    HandleError(raw_window_handle::HandleError),
    /// Vulkan error from the surface creation call.
    Vulkan(vk::enums::Result),
}

impl fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => {
                f.write_str("unsupported display/window handle combination for Vulkan surface")
            }
            Self::HandleError(e) => write!(f, "raw-window-handle error: {e}"),
            Self::Vulkan(e) => write!(f, "Vulkan surface creation failed: {e:?}"),
        }
    }
}

impl std::error::Error for SurfaceError {}

impl From<raw_window_handle::HandleError> for SurfaceError {
    fn from(e: raw_window_handle::HandleError) -> Self {
        Self::HandleError(e)
    }
}

impl From<vk::enums::Result> for SurfaceError {
    fn from(e: vk::enums::Result) -> Self {
        Self::Vulkan(e)
    }
}

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

impl Instance {
    /// Create a Vulkan surface from platform window handles.
    ///
    /// The instance must have been created with the extensions returned
    /// by [`required_extensions()`].
    ///
    /// # Safety
    ///
    /// - `display` and `window` must be valid and outlive the returned surface.
    /// - The instance must have enabled the required surface extensions.
    pub unsafe fn create_surface(
        &self,
        display: &dyn HasDisplayHandle,
        window: &dyn HasWindowHandle,
        allocator: Option<&vk::structs::AllocationCallbacks>,
    ) -> Result<vk::handles::SurfaceKHR, SurfaceError> {
        let raw_display = display.display_handle()?.as_raw();
        let raw_window = window.window_handle()?.as_raw();
        let alloc_ptr = allocator.map_or(std::ptr::null(), |a| a as *const _);

        match (raw_display, raw_window) {
            #[cfg(target_os = "windows")]
            (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(h)) => {
                let info = vk::structs::Win32SurfaceCreateInfoKHR {
                    hinstance: h.hinstance.map_or(0, |v| v.get()),
                    hwnd: h.hwnd.get(),
                    ..Default::default()
                };
                let fp = self
                    .commands()
                    .create_win32_surface_khr
                    .expect("VK_KHR_win32_surface not loaded");
                let mut surface = vk::handles::SurfaceKHR::null();
                check(unsafe { fp(self.handle(), &info, alloc_ptr, &mut surface) })?;
                Ok(surface)
            }

            #[cfg(all(
                unix,
                not(target_os = "android"),
                not(target_os = "macos"),
                not(target_os = "ios"),
            ))]
            (RawDisplayHandle::Xlib(d), RawWindowHandle::Xlib(w)) => {
                let info = vk::structs::XlibSurfaceCreateInfoKHR {
                    dpy: d.display.map_or(std::ptr::null_mut(), |p| p.as_ptr()),
                    window: w.window,
                    ..Default::default()
                };
                let fp = self
                    .commands()
                    .create_xlib_surface_khr
                    .expect("VK_KHR_xlib_surface not loaded");
                let mut surface = vk::handles::SurfaceKHR::null();
                check(unsafe { fp(self.handle(), &info, alloc_ptr, &mut surface) })?;
                Ok(surface)
            }

            #[cfg(all(
                unix,
                not(target_os = "android"),
                not(target_os = "macos"),
                not(target_os = "ios"),
            ))]
            (RawDisplayHandle::Xcb(d), RawWindowHandle::Xcb(w)) => {
                let info = vk::structs::XcbSurfaceCreateInfoKHR {
                    connection: d.connection.map_or(std::ptr::null_mut(), |p| p.as_ptr()),
                    window: w.window.get(),
                    ..Default::default()
                };
                let fp = self
                    .commands()
                    .create_xcb_surface_khr
                    .expect("VK_KHR_xcb_surface not loaded");
                let mut surface = vk::handles::SurfaceKHR::null();
                check(unsafe { fp(self.handle(), &info, alloc_ptr, &mut surface) })?;
                Ok(surface)
            }

            #[cfg(all(
                unix,
                not(target_os = "android"),
                not(target_os = "macos"),
                not(target_os = "ios"),
            ))]
            (RawDisplayHandle::Wayland(d), RawWindowHandle::Wayland(w)) => {
                let info = vk::structs::WaylandSurfaceCreateInfoKHR {
                    display: d.display.as_ptr(),
                    surface: w.surface.as_ptr(),
                    ..Default::default()
                };
                let fp = self
                    .commands()
                    .create_wayland_surface_khr
                    .expect("VK_KHR_wayland_surface not loaded");
                let mut surface = vk::handles::SurfaceKHR::null();
                check(unsafe { fp(self.handle(), &info, alloc_ptr, &mut surface) })?;
                Ok(surface)
            }

            #[cfg(target_os = "macos")]
            (RawDisplayHandle::AppKit(_), RawWindowHandle::AppKit(w)) => {
                // The caller's NSView must be backed by a CAMetalLayer
                // (typically set up by the windowing library).
                let info = vk::structs::MetalSurfaceCreateInfoEXT {
                    p_layer: w.ns_view.as_ptr() as *const _,
                    ..Default::default()
                };
                let fp = self
                    .commands()
                    .create_metal_surface_ext
                    .expect("VK_EXT_metal_surface not loaded");
                let mut surface = vk::handles::SurfaceKHR::null();
                check(unsafe { fp(self.handle(), &info, alloc_ptr, &mut surface) })?;
                Ok(surface)
            }

            #[cfg(target_os = "android")]
            (RawDisplayHandle::Android(_), RawWindowHandle::AndroidNdk(w)) => {
                let info = vk::structs::AndroidSurfaceCreateInfoKHR {
                    window: w.a_native_window.as_ptr(),
                    ..Default::default()
                };
                let fp = self
                    .commands()
                    .create_android_surface_khr
                    .expect("VK_KHR_android_surface not loaded");
                let mut surface = vk::handles::SurfaceKHR::null();
                check(unsafe { fp(self.handle(), &info, alloc_ptr, &mut surface) })?;
                Ok(surface)
            }

            _ => Err(SurfaceError::UnsupportedPlatform),
        }
    }

    /// Destroy a Vulkan surface.
    ///
    /// # Safety
    ///
    /// - The surface must not be in use by any swapchain or other object.
    /// - The surface must have been created from this instance.
    pub unsafe fn destroy_surface(
        &self,
        surface: vk::handles::SurfaceKHR,
        allocator: Option<&vk::structs::AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_surface_khr
            .expect("VK_KHR_surface not loaded");
        let alloc_ptr = allocator.map_or(std::ptr::null(), |a| a as *const _);
        unsafe { fp(self.handle(), surface, alloc_ptr) };
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

    #[test]
    fn unsupported_handle_returns_error() {
        use raw_window_handle::{
            DisplayHandle, RawDisplayHandle, RawWindowHandle, WebDisplayHandle,
            WebWindowHandle, WindowHandle,
        };

        let raw_display = RawDisplayHandle::Web(WebDisplayHandle::new());
        let display = unsafe { DisplayHandle::borrow_raw(raw_display) };

        let raw_window = RawWindowHandle::Web(WebWindowHandle::new(1));
        let window = unsafe { WindowHandle::borrow_raw(raw_window) };

        let instance = mock_instance();
        let result = unsafe { instance.create_surface(&display, &window, None) };
        assert!(matches!(result, Err(SurfaceError::UnsupportedPlatform)));
    }

    #[test]
    #[should_panic(expected = "VK_KHR_surface not loaded")]
    fn destroy_surface_panics_when_extension_not_loaded() {
        let instance = mock_instance();
        unsafe {
            instance.destroy_surface(vk::handles::SurfaceKHR::null(), None);
        }
    }

    #[test]
    fn surface_error_display_unsupported() {
        let err = SurfaceError::UnsupportedPlatform;
        assert_eq!(
            err.to_string(),
            "unsupported display/window handle combination for Vulkan surface"
        );
    }

    #[test]
    fn surface_error_display_vulkan() {
        let err = SurfaceError::Vulkan(vk::enums::Result::ERROR_SURFACE_LOST);
        let msg = err.to_string();
        assert!(msg.contains("Vulkan surface creation failed"));
    }

    #[test]
    fn surface_error_display_handle_error() {
        let err = SurfaceError::HandleError(raw_window_handle::HandleError::Unavailable);
        let msg = err.to_string();
        assert!(msg.contains("raw-window-handle error"));
    }

    #[test]
    fn surface_error_from_handle_error() {
        let he = raw_window_handle::HandleError::Unavailable;
        let se: SurfaceError = he.into();
        assert!(matches!(se, SurfaceError::HandleError(_)));
    }

    #[test]
    fn surface_error_from_vk_result() {
        let vk_err = vk::enums::Result::ERROR_SURFACE_LOST;
        let se: SurfaceError = vk_err.into();
        assert!(matches!(se, SurfaceError::Vulkan(_)));
    }

    fn mock_instance() -> Instance {
        use std::ffi::c_char;

        unsafe extern "system" fn mock_get_instance_proc_addr(
            _instance: vk::handles::Instance,
            _name: *const c_char,
        ) -> vk::structs::PFN_vkVoidFunction {
            None
        }

        unsafe {
            Instance::from_raw_parts(
                vk::handles::Instance::from_raw(0xDEAD),
                Some(mock_get_instance_proc_addr),
            )
        }
    }
}
