//! Platform-aware Vulkan surface creation and extension helpers.
//!
//! **Guide:** [Hello Triangle, Part 2](https://hiddentale.github.io/vulkan_rs/getting-started/hello-triangle-2.html)
//! covers surface creation and swapchain setup.

use std::ffi::CStr;
use std::fmt;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};

use crate::error::check;
use crate::instance::Instance;
use crate::vk;
use vk::handles::Handle;

/// Error returned by surface creation.
///
/// # Examples
///
/// ```
/// use vulkan_rs::SurfaceError;
///
/// let err = SurfaceError::UnsupportedPlatform;
/// assert!(err.to_string().contains("unsupported"));
/// ```
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
///
/// # Examples
///
/// ```
/// use vulkan_rs::required_extensions;
///
/// let exts = required_extensions();
/// assert!(exts.iter().any(|e| *e == c"VK_KHR_surface"));
/// ```
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
        // Wayland and X11,return both; the loader ignores missing ones
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
    #[cfg(not(any(
        target_os = "windows",
        target_os = "android",
        target_os = "macos",
        target_os = "ios",
        all(
            unix,
            not(target_os = "android"),
            not(target_os = "macos"),
            not(target_os = "ios"),
        ),
    )))]
    {
        compile_error!(
            "vulkan-rs surface support: unsupported platform. \
             Disable the `surface` feature or open an issue."
        );
    }
}

impl Instance {
    /// Create a Vulkan surface from platform window handles.
    ///
    /// Supports Win32, X11, Wayland, Metal, and Android. The instance must
    /// have been created with the extensions returned by [`required_extensions()`].
    ///
    /// # Safety
    ///
    /// - `display` and `window` must be valid and outlive the returned surface.
    /// - The instance must have enabled the required surface extensions.
    /// - The returned surface must be destroyed with `destroy_surface_khr`
    ///   before the instance is destroyed.
    pub unsafe fn create_surface(
        &self,
        display: &dyn HasDisplayHandle,
        window: &dyn HasWindowHandle,
        allocator: Option<&vk::structs::AllocationCallbacks>,
    ) -> Result<vk::handles::SurfaceKHR, SurfaceError> {
        let raw_display = display.display_handle()?.as_raw();
        let raw_window = window.window_handle()?.as_raw();
        let alloc_ptr = allocator.map_or(std::ptr::null(), |a| a as *const _);

        // SAFETY (all arms): caller guarantees display/window handles are valid and outlive
        // the surface. The instance has the required surface extensions enabled. The create
        // info structs are built from the raw handles and are valid for the duration of the call.
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
        // SAFETY: caller guarantees the surface is not in use and belongs to this instance.
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
            DisplayHandle, RawDisplayHandle, RawWindowHandle, WebDisplayHandle, WebWindowHandle,
            WindowHandle,
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

    #[test]
    fn surface_error_is_std_error() {
        let err: &dyn std::error::Error = &SurfaceError::UnsupportedPlatform;
        assert!(err.source().is_none());
    }

    // -- Platform-specific surface creation tests (Windows) -------------------

    #[cfg(target_os = "windows")]
    mod win32_tests {
        use super::*;
        use std::ffi::c_char;
        use std::num::NonZeroIsize;

        use raw_window_handle::{
            DisplayHandle, RawDisplayHandle, RawWindowHandle, Win32WindowHandle, WindowHandle,
            WindowsDisplayHandle,
        };

        unsafe extern "system" fn mock_create_win32_surface(
            _instance: vk::handles::Instance,
            _p_create_info: *const vk::structs::Win32SurfaceCreateInfoKHR,
            _p_allocator: *const vk::structs::AllocationCallbacks,
            p_surface: *mut vk::handles::SurfaceKHR,
        ) -> vk::enums::Result {
            unsafe { *p_surface = vk::handles::SurfaceKHR::from_raw(0xEF01) };
            vk::enums::Result::SUCCESS
        }

        unsafe extern "system" fn failing_create_win32_surface(
            _instance: vk::handles::Instance,
            _p_create_info: *const vk::structs::Win32SurfaceCreateInfoKHR,
            _p_allocator: *const vk::structs::AllocationCallbacks,
            _p_surface: *mut vk::handles::SurfaceKHR,
        ) -> vk::enums::Result {
            vk::enums::Result::ERROR_INITIALIZATION_FAILED
        }

        unsafe extern "system" fn mock_destroy_surface(
            instance: vk::handles::Instance,
            surface: vk::handles::SurfaceKHR,
            _p_allocator: *const vk::structs::AllocationCallbacks,
        ) {
            assert_eq!(instance.as_raw(), 0xDEAD, "wrong instance handle");
            assert_eq!(surface.as_raw(), 0xABCD, "wrong surface handle");
        }

        unsafe extern "system" fn surface_instance_proc_addr(
            _instance: vk::handles::Instance,
            name: *const c_char,
        ) -> vk::structs::PFN_vkVoidFunction {
            let name = unsafe { CStr::from_ptr(name) };
            match name.to_bytes() {
                b"vkCreateWin32SurfaceKHR" => Some(unsafe {
                    std::mem::transmute::<
                        unsafe extern "system" fn(
                            vk::handles::Instance,
                            *const vk::structs::Win32SurfaceCreateInfoKHR,
                            *const vk::structs::AllocationCallbacks,
                            *mut vk::handles::SurfaceKHR,
                        ) -> vk::enums::Result,
                        unsafe extern "system" fn(),
                    >(mock_create_win32_surface)
                }),
                b"vkDestroySurfaceKHR" => Some(unsafe {
                    std::mem::transmute::<
                        unsafe extern "system" fn(
                            vk::handles::Instance,
                            vk::handles::SurfaceKHR,
                            *const vk::structs::AllocationCallbacks,
                        ),
                        unsafe extern "system" fn(),
                    >(mock_destroy_surface)
                }),
                _ => None,
            }
        }

        unsafe extern "system" fn failing_surface_instance_proc_addr(
            _instance: vk::handles::Instance,
            name: *const c_char,
        ) -> vk::structs::PFN_vkVoidFunction {
            let name = unsafe { CStr::from_ptr(name) };
            match name.to_bytes() {
                b"vkCreateWin32SurfaceKHR" => Some(unsafe {
                    std::mem::transmute::<
                        unsafe extern "system" fn(
                            vk::handles::Instance,
                            *const vk::structs::Win32SurfaceCreateInfoKHR,
                            *const vk::structs::AllocationCallbacks,
                            *mut vk::handles::SurfaceKHR,
                        ) -> vk::enums::Result,
                        unsafe extern "system" fn(),
                    >(failing_create_win32_surface)
                }),
                _ => None,
            }
        }

        fn win32_display() -> DisplayHandle<'static> {
            let raw = RawDisplayHandle::Windows(WindowsDisplayHandle::new());
            unsafe { DisplayHandle::borrow_raw(raw) }
        }

        fn win32_window() -> WindowHandle<'static> {
            let hwnd = NonZeroIsize::new(0x1234).unwrap();
            let raw = RawWindowHandle::Win32(Win32WindowHandle::new(hwnd));
            unsafe { WindowHandle::borrow_raw(raw) }
        }

        fn surface_instance() -> Instance {
            unsafe {
                Instance::from_raw_parts(
                    vk::handles::Instance::from_raw(0xDEAD),
                    Some(surface_instance_proc_addr),
                )
            }
        }

        #[test]
        fn create_surface_win32_succeeds() {
            let instance = surface_instance();
            let display = win32_display();
            let window = win32_window();

            let surface = unsafe { instance.create_surface(&display, &window, None) }
                .expect("create_surface should succeed");
            assert!(!surface.is_null());
        }

        #[test]
        fn create_surface_win32_propagates_vulkan_error() {
            let instance = unsafe {
                Instance::from_raw_parts(
                    vk::handles::Instance::from_raw(0xDEAD),
                    Some(failing_surface_instance_proc_addr),
                )
            };
            let display = win32_display();
            let window = win32_window();

            let result = unsafe { instance.create_surface(&display, &window, None) };
            match result {
                Err(SurfaceError::Vulkan(e)) => {
                    assert_eq!(e, vk::enums::Result::ERROR_INITIALIZATION_FAILED);
                }
                other => panic!("expected SurfaceError::Vulkan, got {other:?}"),
            }
        }

        #[test]
        fn destroy_surface_calls_fp_with_correct_args() {
            let instance = surface_instance();
            let surface = vk::handles::SurfaceKHR::from_raw(0xABCD);
            // The mock asserts handle values internally.
            unsafe { instance.destroy_surface(surface, None) };
        }
    }
}
