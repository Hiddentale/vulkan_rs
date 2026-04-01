/// Maps C type names from vk.xml to Rust type tokens.
pub fn c_type_to_rust(c: &str) -> Option<&'static str> {
    Some(match c {
        // Fixed-width integers
        "uint8_t" => "u8",
        "uint16_t" => "u16",
        "uint32_t" => "u32",
        "uint64_t" => "u64",
        "int32_t" => "i32",
        "int64_t" => "i64",
        "size_t" => "usize",

        // C primitives
        "float" => "f32",
        "double" => "f64",
        "char" => "core::ffi::c_char",
        "void" => "core::ffi::c_void",
        "int" => "core::ffi::c_int",

        // Vulkan typedefs
        "VkBool32" => "u32",
        "VkDeviceSize" => "u64",
        "VkDeviceAddress" => "u64",
        "VkFlags" => "u32",
        "VkFlags64" => "u64",
        "VkSampleMask" => "u32",
        "VkRemoteAddressNV" => "*mut core::ffi::c_void",

        // Platform types,Win32
        "HINSTANCE" => "isize",
        "HWND" => "isize",
        "HMONITOR" => "isize",
        "HANDLE" => "isize",
        "DWORD" => "u32",
        "LPCWSTR" => "*const u16",
        "SECURITY_ATTRIBUTES" => "core::ffi::c_void",

        // Platform types,X11/Xlib
        "Display" => "core::ffi::c_void",
        "Window" => "core::ffi::c_ulong",
        "VisualID" => "core::ffi::c_ulong",
        "RROutput" => "core::ffi::c_ulong",

        // Platform types,XCB
        "xcb_connection_t" => "core::ffi::c_void",
        "xcb_window_t" => "u32",
        "xcb_visualid_t" => "u32",

        // Platform types,Wayland
        "wl_display" => "core::ffi::c_void",
        "wl_surface" => "core::ffi::c_void",

        // Platform types,Android
        "ANativeWindow" => "core::ffi::c_void",
        "AHardwareBuffer" => "core::ffi::c_void",

        // Platform types,Metal/macOS/iOS
        "CAMetalLayer" => "core::ffi::c_void",
        "MTLDevice_id" => "core::ffi::c_void",
        "MTLCommandQueue_id" => "core::ffi::c_void",
        "MTLBuffer_id" => "core::ffi::c_void",
        "MTLTexture_id" => "core::ffi::c_void",
        "MTLSharedEvent_id" => "core::ffi::c_void",
        "IOSurfaceRef" => "core::ffi::c_void",

        // Platform types,DirectFB
        "IDirectFB" => "core::ffi::c_void",
        "IDirectFBSurface" => "core::ffi::c_void",

        // Platform types,Fuchsia
        "zx_handle_t" => "u32",

        // Platform types,QNX Screen
        "_screen_window" => "core::ffi::c_void",
        "_screen_context" => "core::ffi::c_void",
        "_screen_buffer" => "core::ffi::c_void",

        // Platform types,GGP (Stadia)
        "GgpStreamDescriptor" => "u32",
        "GgpFrameToken" => "u32",

        // Platform types,NvSci
        "NvSciSyncObj" => "core::ffi::c_void",
        "NvSciSyncFence" => "core::ffi::c_void",
        "NvSciBufObj" => "core::ffi::c_void",
        "NvSciSyncAttrList" => "core::ffi::c_void",
        "NvSciBufAttrList" => "core::ffi::c_void",

        // Platform types,OHOS (OpenHarmony)
        "OHNativeWindow" => "core::ffi::c_void",
        "OH_NativeBuffer" => "core::ffi::c_void",
        "OHBufferHandle" => "core::ffi::c_void",

        // Platform types,UBM (Samsung)
        "ubm_device" => "core::ffi::c_void",
        "ubm_surface" => "core::ffi::c_void",

        // Vk-prefixed or unknown types are not primitives,caller handles them.
        _ => return None,
    })
}

/// True if this C type maps to a Rust primitive (not a generated Vk type).
pub fn is_primitive(c: &str) -> bool {
    c_type_to_rust(c).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_fixed_width_integers() {
        assert_eq!(c_type_to_rust("uint32_t"), Some("u32"));
        assert_eq!(c_type_to_rust("int64_t"), Some("i64"));
        assert_eq!(c_type_to_rust("size_t"), Some("usize"));
    }

    #[test]
    fn maps_vulkan_typedefs() {
        assert_eq!(c_type_to_rust("VkBool32"), Some("u32"));
        assert_eq!(c_type_to_rust("VkDeviceSize"), Some("u64"));
    }

    #[test]
    fn maps_platform_types() {
        assert_eq!(c_type_to_rust("HWND"), Some("isize"));
        assert_eq!(c_type_to_rust("ANativeWindow"), Some("core::ffi::c_void"));
    }

    #[test]
    fn returns_none_for_unknown() {
        assert_eq!(c_type_to_rust("VkBuffer"), None);
        assert_eq!(c_type_to_rust("SomeUnknownType"), None);
    }

    #[test]
    fn is_primitive_classification() {
        assert!(is_primitive("uint32_t"));
        assert!(is_primitive("VkBool32"));
        assert!(!is_primitive("VkBuffer"));
    }
}
