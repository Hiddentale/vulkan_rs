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
        "char" => "std::ffi::c_char",
        "void" => "std::ffi::c_void",
        "int" => "std::ffi::c_int",

        // Vulkan typedefs
        "VkBool32" => "u32",
        "VkDeviceSize" => "u64",
        "VkDeviceAddress" => "u64",
        "VkFlags" => "u32",
        "VkFlags64" => "u64",
        "VkSampleMask" => "u32",

        // Platform types — Win32
        "HINSTANCE" => "isize",
        "HWND" => "isize",
        "HMONITOR" => "isize",
        "HANDLE" => "isize",
        "DWORD" => "u32",
        "LPCWSTR" => "*const u16",
        "SECURITY_ATTRIBUTES" => "std::ffi::c_void",

        // Platform types — X11/Xlib
        "Display" => "std::ffi::c_void",
        "Window" => "std::ffi::c_ulong",
        "VisualID" => "std::ffi::c_ulong",
        "RROutput" => "std::ffi::c_ulong",

        // Platform types — XCB
        "xcb_connection_t" => "std::ffi::c_void",
        "xcb_window_t" => "u32",
        "xcb_visualid_t" => "u32",

        // Platform types — Wayland
        "wl_display" => "std::ffi::c_void",
        "wl_surface" => "std::ffi::c_void",

        // Platform types — Android
        "ANativeWindow" => "std::ffi::c_void",
        "AHardwareBuffer" => "std::ffi::c_void",

        // Platform types — Metal/macOS/iOS
        "CAMetalLayer" => "std::ffi::c_void",

        // Platform types — DirectFB
        "IDirectFB" => "std::ffi::c_void",
        "IDirectFBSurface" => "std::ffi::c_void",

        // Platform types — Fuchsia
        "zx_handle_t" => "u32",

        // Platform types — QNX Screen
        "_screen_window" => "std::ffi::c_void",
        "_screen_context" => "std::ffi::c_void",
        "_screen_buffer" => "std::ffi::c_void",

        // Platform types — GGP (Stadia)
        "GgpStreamDescriptor" => "u32",
        "GgpFrameToken" => "u32",

        // Platform types — NvSci
        "NvSciSyncObj" => "std::ffi::c_void",
        "NvSciSyncFence" => "std::ffi::c_void",
        "NvSciBufObj" => "std::ffi::c_void",
        "NvSciSyncAttrList" => "std::ffi::c_void",
        "NvSciBufAttrList" => "std::ffi::c_void",

        // Vk-prefixed or unknown types are not primitives — caller handles them.
        _ => return None,
    })
}

/// True if this C type maps to a Rust primitive (not a generated Vk type).
pub fn is_primitive(c: &str) -> bool {
    c_type_to_rust(c).is_some()
}
