//! Resolves C types from vk.xml into Rust type tokens.
//!
//! Handles primitives, Vk-prefixed types, pointers, arrays, and field name conversion.

use heck::ToSnakeCase;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::parse::{MemberDef, ParamDef};
use crate::type_map;

/// Resolve a struct member's C type + pointer/array info into a Rust type token.
pub fn resolve_member_type(member: &MemberDef) -> TokenStream {
    resolve_type_fields(
        &member.type_name,
        member.is_pointer,
        member.is_const,
        member.is_double_pointer,
        member.array_size.as_deref(),
    )
}

/// Resolve a command parameter's C type into a Rust type token.
pub fn resolve_param_type(param: &ParamDef) -> TokenStream {
    resolve_type_fields(
        &param.type_name,
        param.is_pointer,
        param.is_const,
        param.is_double_pointer,
        param.array_size.as_deref(),
    )
}

/// Resolve a command return type string into a Rust type token.
/// Returns `None` for void (no return type needed in the signature).
pub fn resolve_return_type(return_type: &str) -> Option<TokenStream> {
    match return_type {
        "void" => None,
        other => Some(resolve_base_type(other)),
    }
}

/// Core type resolution: shared by member and parameter resolution.
fn resolve_type_fields(
    type_name: &str,
    is_pointer: bool,
    is_const: bool,
    is_double_pointer: bool,
    array_size: Option<&str>,
) -> TokenStream {
    if type_name == "void" && !is_pointer {
        return quote! { core::ffi::c_void };
    }

    // Opaque platform types that map to c_void but aren't marked as pointers.
    if !is_pointer
        && let Some(rust) = type_map::c_type_to_rust(type_name)
        && rust == "core::ffi::c_void"
        && type_name != "void"
    {
        return quote! { *const core::ffi::c_void };
    }

    let base = resolve_base_type(type_name);

    if let Some(size) = array_size {
        return wrap_array(&base, size);
    }

    if is_double_pointer {
        if is_const {
            quote! { *const *const #base }
        } else {
            quote! { *mut *mut #base }
        }
    } else if is_pointer {
        if is_const {
            quote! { *const #base }
        } else {
            quote! { *mut #base }
        }
    } else {
        base
    }
}

/// Resolve a C type name to Rust tokens: either a primitive or a generated type.
pub fn resolve_base_type(c_type: &str) -> TokenStream {
    if let Some(rust) = type_map::c_type_to_rust(c_type) {
        let ty: TokenStream = rust.parse().expect("invalid type_map entry");
        return ty;
    }

    let stripped = c_type.strip_prefix("Vk").unwrap_or(c_type);

    if stripped.starts_with("StdVideo") || c_type.starts_with("StdVideo") {
        let ident = format_ident!("{}", c_type);
        return quote! { #ident };
    }

    if stripped.starts_with("PFN_vk") || c_type.starts_with("PFN_vk") {
        let ident = format_ident!("{}", c_type);
        return quote! { #ident };
    }

    let ident = format_ident!("{}", stripped);
    quote! { #ident }
}

/// Resolve `FooFlags` → `FooFlagBits`, handling extension suffixes.
pub fn resolve_flags_alias(name: &str) -> String {
    use crate::emit_enums::EXTENSION_SUFFIXES;

    for suffix in EXTENSION_SUFFIXES {
        for digit in ["", "2", "3"] {
            let flags_pattern = format!("Flags{digit}{suffix}");
            if let Some(prefix) = name.strip_suffix(flags_pattern.as_str()) {
                return format!("{prefix}FlagBits{digit}{suffix}");
            }
        }
    }

    for digit in ["2", "3"] {
        let pattern = format!("Flags{digit}");
        if let Some(prefix) = name.strip_suffix(pattern.as_str()) {
            return format!("{prefix}FlagBits{digit}");
        }
    }

    if name.ends_with("Flags") && !name.ends_with("FlagBits") {
        let prefix = name.strip_suffix("Flags").unwrap();
        return format!("{prefix}FlagBits");
    }
    name.to_string()
}

fn wrap_array(base: &TokenStream, size: &str) -> TokenStream {
    if let Ok(n) = size.parse::<usize>() {
        quote! { [#base; #n] }
    } else {
        let const_name = size.strip_prefix("VK_").unwrap_or(size);
        let ident = format_ident!("{}", const_name);
        quote! { [#base; #ident as usize] }
    }
}

/// Convert a C member name to snake_case, handling Vulkan quirks.
pub fn member_name(c_name: &str) -> String {
    match c_name {
        "sType" => "s_type".to_string(),
        "pNext" => "p_next".to_string(),
        _ => c_name.to_snake_case(),
    }
}

/// True if this member name is a Rust keyword and needs raw-ident escaping.
pub fn is_rust_keyword(name: &str) -> bool {
    matches!(
        name,
        "type"
            | "ref"
            | "in"
            | "use"
            | "box"
            | "move"
            | "yield"
            | "async"
            | "await"
            | "dyn"
            | "try"
            | "macro"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::{MemberDef, ParamDef};

    fn make_member(name: &str, type_name: &str) -> MemberDef {
        MemberDef {
            name: name.to_string(),
            type_name: type_name.to_string(),
            is_pointer: false,
            is_const: false,
            is_double_pointer: false,
            array_size: None,
            optional: false,
            values: None,
            len: None,
            extern_sync: None,
        }
    }

    fn make_pointer_member(name: &str, type_name: &str, is_const: bool) -> MemberDef {
        MemberDef {
            is_pointer: true,
            is_const,
            ..make_member(name, type_name)
        }
    }

    fn make_double_pointer_member(name: &str, type_name: &str, is_const: bool) -> MemberDef {
        MemberDef {
            is_pointer: true,
            is_double_pointer: true,
            is_const,
            ..make_member(name, type_name)
        }
    }

    fn make_array_member(name: &str, type_name: &str, size: &str) -> MemberDef {
        MemberDef {
            array_size: Some(size.to_string()),
            ..make_member(name, type_name)
        }
    }

    #[test]
    fn resolve_primitive_type() {
        let m = make_member("size", "uint32_t");
        assert_eq!(resolve_member_type(&m).to_string(), "u32");
    }

    #[test]
    fn resolve_vk_type() {
        let m = make_member("format", "VkFormat");
        assert_eq!(resolve_member_type(&m).to_string(), "Format");
    }

    #[test]
    fn resolve_const_pointer() {
        let m = make_pointer_member("pNext", "void", true);
        assert_eq!(
            resolve_member_type(&m).to_string(),
            "* const core :: ffi :: c_void"
        );
    }

    #[test]
    fn resolve_mut_pointer() {
        let m = make_pointer_member("pNext", "void", false);
        assert_eq!(
            resolve_member_type(&m).to_string(),
            "* mut core :: ffi :: c_void"
        );
    }

    #[test]
    fn resolve_const_vk_pointer() {
        let m = make_pointer_member("pCreateInfo", "VkBufferCreateInfo", true);
        assert_eq!(
            resolve_member_type(&m).to_string(),
            "* const BufferCreateInfo"
        );
    }

    #[test]
    fn resolve_double_pointer() {
        let m = make_double_pointer_member("ppData", "void", false);
        assert_eq!(
            resolve_member_type(&m).to_string(),
            "* mut * mut core :: ffi :: c_void"
        );
    }

    #[test]
    fn resolve_const_double_pointer() {
        let m = make_double_pointer_member("ppEnabledLayerNames", "char", true);
        assert_eq!(
            resolve_member_type(&m).to_string(),
            "* const * const core :: ffi :: c_char"
        );
    }

    #[test]
    fn resolve_numeric_array() {
        let m = make_array_member("color", "float", "4");
        assert_eq!(resolve_member_type(&m).to_string(), "[f32 ; 4usize]");
    }

    #[test]
    fn resolve_constant_array() {
        let m = make_array_member("deviceName", "char", "VK_MAX_PHYSICAL_DEVICE_NAME_SIZE");
        assert_eq!(
            resolve_member_type(&m).to_string(),
            "[core :: ffi :: c_char ; MAX_PHYSICAL_DEVICE_NAME_SIZE as usize]"
        );
    }

    #[test]
    fn member_name_stype() {
        assert_eq!(member_name("sType"), "s_type");
    }

    #[test]
    fn member_name_pnext() {
        assert_eq!(member_name("pNext"), "p_next");
    }

    #[test]
    fn member_name_camel_case() {
        assert_eq!(
            member_name("queueFamilyIndexCount"),
            "queue_family_index_count"
        );
    }

    #[test]
    fn member_name_simple() {
        assert_eq!(member_name("flags"), "flags");
        assert_eq!(member_name("size"), "size");
    }

    #[test]
    fn member_name_pp_prefix() {
        assert_eq!(member_name("ppEnabledLayerNames"), "pp_enabled_layer_names");
    }

    #[test]
    fn keyword_detection() {
        assert!(is_rust_keyword("type"));
        assert!(is_rust_keyword("ref"));
        assert!(!is_rust_keyword("flags"));
    }

    // --- ParamDef resolution tests ---

    fn make_param(name: &str, type_name: &str) -> ParamDef {
        ParamDef {
            name: name.to_string(),
            type_name: type_name.to_string(),
            is_pointer: false,
            is_const: false,
            is_double_pointer: false,
            array_size: None,
            optional: false,
            len: None,
            extern_sync: None,
        }
    }

    #[test]
    fn resolve_param_handle_type() {
        let p = ParamDef {
            is_pointer: false,
            ..make_param("instance", "VkInstance")
        };
        assert_eq!(resolve_param_type(&p).to_string(), "Instance");
    }

    #[test]
    fn resolve_param_const_pointer() {
        let p = ParamDef {
            is_pointer: true,
            is_const: true,
            ..make_param("pCreateInfo", "VkInstanceCreateInfo")
        };
        assert_eq!(
            resolve_param_type(&p).to_string(),
            "* const InstanceCreateInfo"
        );
    }

    #[test]
    fn resolve_param_mut_pointer() {
        let p = ParamDef {
            is_pointer: true,
            is_const: false,
            ..make_param("pInstance", "VkInstance")
        };
        assert_eq!(resolve_param_type(&p).to_string(), "* mut Instance");
    }

    #[test]
    fn resolve_param_double_pointer() {
        let p = ParamDef {
            is_pointer: true,
            is_double_pointer: true,
            is_const: false,
            ..make_param("ppData", "void")
        };
        assert_eq!(
            resolve_param_type(&p).to_string(),
            "* mut * mut core :: ffi :: c_void"
        );
    }

    #[test]
    fn resolve_param_primitive() {
        let p = make_param("vertexCount", "uint32_t");
        assert_eq!(resolve_param_type(&p).to_string(), "u32");
    }

    // --- Return type resolution tests ---

    #[test]
    fn resolve_return_void() {
        assert!(resolve_return_type("void").is_none());
    }

    #[test]
    fn resolve_return_vk_result() {
        let ty = resolve_return_type("VkResult").unwrap();
        assert_eq!(ty.to_string(), "Result");
    }

    #[test]
    fn resolve_return_vk_bool() {
        let ty = resolve_return_type("VkBool32").unwrap();
        assert_eq!(ty.to_string(), "u32");
    }

    // --- Flags alias resolution tests ---

    #[test]
    fn flags_alias_bare() {
        assert_eq!(resolve_flags_alias("BufferCreateFlags"), "BufferCreateFlagBits");
    }

    #[test]
    fn flags_alias_extension_suffixed() {
        assert_eq!(
            resolve_flags_alias("SwapchainCreateFlagsKHR"),
            "SwapchainCreateFlagBitsKHR"
        );
    }

    #[test]
    fn flags_alias_versioned() {
        assert_eq!(
            resolve_flags_alias("PipelineCreateFlags2KHR"),
            "PipelineCreateFlagBits2KHR"
        );
    }

    #[test]
    fn flags_alias_noop_for_flagbits() {
        assert_eq!(resolve_flags_alias("BufferCreateFlagBits"), "BufferCreateFlagBits");
    }

    #[test]
    fn flags_alias_noop_for_unrelated() {
        assert_eq!(resolve_flags_alias("SomeRandomType"), "SomeRandomType");
    }

    #[test]
    fn flags_alias_ext_suffix() {
        assert_eq!(
            resolve_flags_alias("DebugUtilsMessageTypeFlagsEXT"),
            "DebugUtilsMessageTypeFlagBitsEXT"
        );
    }

    #[test]
    fn flags_alias_nv_suffix() {
        assert_eq!(
            resolve_flags_alias("ExternalMemoryHandleTypeFlagsNV"),
            "ExternalMemoryHandleTypeFlagBitsNV"
        );
    }
}
