//! Emits `#[repr(C)]` struct and union definitions for all Vulkan types.

use heck::ToSnakeCase;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::parse::{MemberDef, StructDef};
use crate::type_map;

// ---------------------------------------------------------------------------
// StructureType mapping (subtask 5)
// ---------------------------------------------------------------------------

/// Given the `values` field from an sType member (e.g. `VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO`),
/// return a token for the corresponding `StructureType` constant.
fn stype_constant(values: &str) -> TokenStream {
    // Strip VK_STRUCTURE_TYPE_ prefix to get the variant name.
    let variant = values.strip_prefix("VK_STRUCTURE_TYPE_").unwrap_or(values);
    let ident = format_ident!("{}", variant);
    quote! { StructureType::#ident }
}

/// Returns the StructureType constant for a struct, if it has an sType member with a values hint.
fn struct_stype(def: &StructDef) -> Option<TokenStream> {
    def.members.iter().find_map(|m| {
        if m.name == "sType" {
            m.values.as_deref().map(stype_constant)
        } else {
            None
        }
    })
}

// ---------------------------------------------------------------------------
// Field type resolution (subtask 1)
// ---------------------------------------------------------------------------

/// Resolve a struct member's C type + pointer/array info into a Rust type token.
fn resolve_member_type(member: &MemberDef) -> TokenStream {
    // Special case: void without pointer (rare, appears in unions)
    if member.type_name == "void" && !member.is_pointer {
        return quote! { std::ffi::c_void };
    }

    // Resolve the base type.
    let base = resolve_base_type(&member.type_name);

    // Wrap in array if fixed-size.
    if let Some(ref size) = member.array_size {
        let array = wrap_array(&base, size);
        // Arrays are never additionally pointer-wrapped in Vulkan structs.
        return array;
    }

    // Wrap in pointer(s) if needed.
    if member.is_double_pointer {
        if member.is_const {
            quote! { *const *const #base }
        } else {
            quote! { *mut *mut #base }
        }
    } else if member.is_pointer {
        if member.is_const {
            quote! { *const #base }
        } else {
            quote! { *mut #base }
        }
    } else {
        base
    }
}

/// Resolve a C type name to Rust tokens: either a primitive or a generated type.
fn resolve_base_type(c_type: &str) -> TokenStream {
    if let Some(rust) = type_map::c_type_to_rust(c_type) {
        // Primitive or platform type — parse the string into tokens.
        let ty: TokenStream = rust.parse().expect("invalid type_map entry");
        ty
    } else {
        // Vk-prefixed type → strip Vk prefix and use as ident.
        let stripped = c_type.strip_prefix("Vk").unwrap_or(c_type);
        let ident = format_ident!("{}", stripped);
        quote! { #ident }
    }
}

/// Wrap a base type in a fixed-size array: `[base; SIZE]`.
fn wrap_array(base: &TokenStream, size: &str) -> TokenStream {
    // Size may be numeric ("4") or a constant name ("VK_MAX_MEMORY_TYPES").
    if let Ok(n) = size.parse::<usize>() {
        quote! { [#base; #n] }
    } else {
        // Constant reference — strip VK_ prefix to match our generated constant name.
        let const_name = size.strip_prefix("VK_").unwrap_or(size);
        let ident = format_ident!("{}", const_name);
        quote! { [#base; #ident as usize] }
    }
}

// ---------------------------------------------------------------------------
// Field name conversion (subtask 2a)
// ---------------------------------------------------------------------------

/// Convert a C member name to snake_case, handling Vulkan quirks.
fn member_name(c_name: &str) -> String {
    match c_name {
        // Special cases that heck's to_snake_case doesn't handle well.
        "sType" => "s_type".to_string(),
        "pNext" => "p_next".to_string(),
        _ => {
            // heck handles camelCase → snake_case, including sequences like
            // "queueFamilyIndexCount" → "queue_family_index_count".
            c_name.to_snake_case()
        }
    }
}

/// True if this member name is a Rust keyword and needs to be raw-ident escaped.
fn is_rust_keyword(name: &str) -> bool {
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
fn assert_valid_rust(tokens: &TokenStream) {
    syn::parse2::<syn::File>(tokens.clone())
        .unwrap_or_else(|e| panic!("generated code is not valid Rust: {e}\n\n{tokens}"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::{MemberDef, StructDef};

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

    // --- StructureType mapping ---

    #[test]
    fn stype_constant_strips_prefix() {
        let tokens = stype_constant("VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO");
        assert_eq!(tokens.to_string(), "StructureType :: BUFFER_CREATE_INFO");
    }

    #[test]
    fn struct_stype_finds_value() {
        let def = StructDef {
            name: "BufferCreateInfo".to_string(),
            members: vec![
                MemberDef {
                    values: Some("VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO".to_string()),
                    ..make_member("sType", "VkStructureType")
                },
                make_pointer_member("pNext", "void", true),
            ],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        };
        let result = struct_stype(&def);
        assert!(result.is_some());
        assert!(result.unwrap().to_string().contains("BUFFER_CREATE_INFO"));
    }

    #[test]
    fn struct_stype_returns_none_for_plain_struct() {
        let def = StructDef {
            name: "Extent2D".to_string(),
            members: vec![
                make_member("width", "uint32_t"),
                make_member("height", "uint32_t"),
            ],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        };
        assert!(struct_stype(&def).is_none());
    }

    // --- Type resolution ---

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
            "* const std :: ffi :: c_void"
        );
    }

    #[test]
    fn resolve_mut_pointer() {
        let m = make_pointer_member("pNext", "void", false);
        assert_eq!(
            resolve_member_type(&m).to_string(),
            "* mut std :: ffi :: c_void"
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
            "* mut * mut std :: ffi :: c_void"
        );
    }

    #[test]
    fn resolve_const_double_pointer() {
        let m = make_double_pointer_member("ppEnabledLayerNames", "char", true);
        assert_eq!(
            resolve_member_type(&m).to_string(),
            "* const * const std :: ffi :: c_char"
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
            "[std :: ffi :: c_char ; MAX_PHYSICAL_DEVICE_NAME_SIZE as usize]"
        );
    }

    // --- Field name conversion ---

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
}
