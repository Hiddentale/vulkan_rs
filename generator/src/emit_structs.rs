//! Emits `#[repr(C)]` struct and union definitions for all Vulkan types.

use heck::ToSnakeCase;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::parse::{MemberDef, StructDef, VkRegistry};
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
// Marker traits for pNext chains (subtask 6)
// ---------------------------------------------------------------------------

/// Emit marker trait definitions and implementations for type-safe pNext chains.
///
/// For each struct that appears as a `structextends` target, we emit:
///   `pub unsafe trait ExtendsFoo {}`
///
/// For each struct that declares `structextends="VkFoo,VkBar"`, we emit:
///   `unsafe impl ExtendsFoo for ThisStruct {}`
///   `unsafe impl ExtendsBar for ThisStruct {}`
fn emit_marker_traits(registry: &VkRegistry) -> TokenStream {
    use std::collections::BTreeSet;

    // Collect all unique trait names (the targets of structextends).
    let mut trait_names: BTreeSet<String> = BTreeSet::new();
    for s in &registry.structs {
        for extends in &s.extends {
            trait_names.insert(extends.clone());
        }
    }

    // Emit trait definitions.
    let trait_defs: Vec<TokenStream> = trait_names
        .iter()
        .map(|name| {
            let trait_ident = format_ident!("Extends{}", name);
            let vk_name = format!("Vk{}", name);
            quote! {
                /// Marker trait for structs that can appear in the pNext chain of
                #[doc = concat!("[`", #vk_name, "`].")]
                pub unsafe trait #trait_ident {}
            }
        })
        .collect();

    // Emit trait implementations.
    let trait_impls: Vec<TokenStream> = registry
        .structs
        .iter()
        .flat_map(|s| {
            let struct_ident = format_ident!("{}", &s.name);
            s.extends.iter().map(move |extends| {
                let trait_ident = format_ident!("Extends{}", extends);
                quote! {
                    unsafe impl #trait_ident for #struct_ident {}
                }
            })
        })
        .collect();

    quote! {
        #(#trait_defs)*
        #(#trait_impls)*
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Emit all struct and union definitions plus pNext marker traits.
pub fn emit_structs(registry: &VkRegistry) -> TokenStream {
    let base_structs = emit_base_pnext_structs();

    let structs: Vec<TokenStream> = registry
        .structs
        .iter()
        .filter(|s| !is_base_pnext_struct(&s.name))
        .map(emit_struct_or_union)
        .collect();

    let marker_traits = emit_marker_traits(registry);

    quote! {
        use super::enums::*;
        use super::handles::*;
        use super::bitmasks::*;
        use super::constants::*;

        #base_structs
        #(#structs)*
        #marker_traits
    }
}

// ---------------------------------------------------------------------------
// BaseOutStructure / BaseInStructure (subtask 3)
// ---------------------------------------------------------------------------

const BASE_PNEXT_STRUCTS: &[&str] = &["BaseOutStructure", "BaseInStructure"];

fn is_base_pnext_struct(name: &str) -> bool {
    BASE_PNEXT_STRUCTS.contains(&name)
}

fn emit_base_pnext_structs() -> TokenStream {
    quote! {
        /// Structure type used for traversing pNext chains (mutable).
        #[repr(C)]
        #[derive(Copy, Clone)]
        #[doc(alias = "VkBaseOutStructure")]
        pub struct BaseOutStructure {
            pub s_type: StructureType,
            pub p_next: *mut BaseOutStructure,
        }

        impl Default for BaseOutStructure {
            #[inline]
            fn default() -> Self {
                Self {
                    s_type: StructureType::from_raw(0),
                    p_next: std::ptr::null_mut(),
                }
            }
        }

        impl std::fmt::Debug for BaseOutStructure {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("BaseOutStructure")
                    .field("s_type", &self.s_type)
                    .field("p_next", &self.p_next)
                    .finish()
            }
        }

        /// Structure type used for traversing pNext chains (const).
        #[repr(C)]
        #[derive(Copy, Clone)]
        #[doc(alias = "VkBaseInStructure")]
        pub struct BaseInStructure {
            pub s_type: StructureType,
            pub p_next: *const BaseInStructure,
        }

        impl Default for BaseInStructure {
            #[inline]
            fn default() -> Self {
                Self {
                    s_type: StructureType::from_raw(0),
                    p_next: std::ptr::null(),
                }
            }
        }

        impl std::fmt::Debug for BaseInStructure {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("BaseInStructure")
                    .field("s_type", &self.s_type)
                    .field("p_next", &self.p_next)
                    .finish()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Struct and union emission (subtasks 2b, 2c, 2d)
// ---------------------------------------------------------------------------

fn emit_struct_or_union(def: &StructDef) -> TokenStream {
    if def.is_union {
        emit_union(def)
    } else {
        emit_struct(def)
    }
}

fn emit_struct(def: &StructDef) -> TokenStream {
    let name = format_ident!("{}", &def.name);
    let vk_name = format!("Vk{}", &def.name);
    let fields = emit_fields(&def.members);
    let default_impl = emit_default(def);

    quote! {
        #[repr(C)]
        #[derive(Copy, Clone, Debug)]
        #[doc(alias = #vk_name)]
        pub struct #name {
            #(#fields)*
        }

        #default_impl
    }
}

fn emit_union(def: &StructDef) -> TokenStream {
    let name = format_ident!("{}", &def.name);
    let vk_name = format!("Vk{}", &def.name);
    let fields = emit_fields(&def.members);

    quote! {
        #[repr(C)]
        #[derive(Copy, Clone)]
        #[doc(alias = #vk_name)]
        pub union #name {
            #(#fields)*
        }

        impl Default for #name {
            #[inline]
            fn default() -> Self {
                unsafe { std::mem::zeroed() }
            }
        }

        impl std::fmt::Debug for #name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(stringify!(#name))
            }
        }
    }
}

fn emit_fields(members: &[MemberDef]) -> Vec<TokenStream> {
    members.iter().map(emit_field).collect()
}

fn emit_field(member: &MemberDef) -> TokenStream {
    let rust_name = member_name(&member.name);
    let field_ident = if is_rust_keyword(&rust_name) {
        format_ident!("r#{}", rust_name)
    } else {
        format_ident!("{}", rust_name)
    };
    let ty = resolve_member_type(member);

    quote! { pub #field_ident: #ty, }
}

fn emit_default(def: &StructDef) -> TokenStream {
    let name = format_ident!("{}", &def.name);
    let stype = struct_stype(def);
    let has_pnext = def.members.iter().any(|m| m.name == "pNext");

    if stype.is_some() || has_pnext {
        // Struct with sType/pNext: manual Default that fills s_type and nulls pNext.
        let field_defaults: Vec<TokenStream> = def
            .members
            .iter()
            .map(|m| {
                let rust_name = member_name(&m.name);
                let field_ident = if is_rust_keyword(&rust_name) {
                    format_ident!("r#{}", rust_name)
                } else {
                    format_ident!("{}", rust_name)
                };

                if m.name == "sType"
                    && let Some(ref stype_val) = stype
                {
                    return quote! { #field_ident: #stype_val, };
                }

                let default_val = default_value_for(m);
                quote! { #field_ident: #default_val, }
            })
            .collect();

        quote! {
            impl Default for #name {
                #[inline]
                fn default() -> Self {
                    Self {
                        #(#field_defaults)*
                    }
                }
            }
        }
    } else {
        // Plain struct: use zeroed memory. Safe because all Vulkan structs are
        // repr(C) and zero-initialized is a valid state.
        quote! {
            impl Default for #name {
                #[inline]
                fn default() -> Self {
                    unsafe { std::mem::zeroed() }
                }
            }
        }
    }
}

/// Produce a default value expression for a struct member.
fn default_value_for(member: &MemberDef) -> TokenStream {
    // Pointers default to null.
    if member.is_pointer || member.is_double_pointer {
        if member.is_const {
            return quote! { std::ptr::null() };
        } else {
            return quote! { std::ptr::null_mut() };
        }
    }

    // Arrays default to zeroed.
    if member.array_size.is_some() {
        return quote! { unsafe { std::mem::zeroed() } };
    }

    // Everything else: use Default::default() which works for u32, i32, f32,
    // and our generated newtypes (all have Default).
    quote! { Default::default() }
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
    use crate::parse::{MemberDef, StructDef, VkRegistry};
    use std::collections::HashMap;

    fn empty_registry() -> VkRegistry {
        VkRegistry {
            handles: vec![],
            structs: vec![],
            enums: vec![],
            bitmasks: vec![],
            commands: vec![],
            constants: vec![],
            func_pointers: vec![],
            extensions: vec![],
            platforms: vec![],
            aliases: HashMap::new(),
            base_types: HashMap::new(),
        }
    }

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

    // --- Base pNext structs ---

    #[test]
    fn base_pnext_structs_valid_rust() {
        let tokens = emit_base_pnext_structs();
        // Needs StructureType in scope to parse.
        let wrapped = quote! {
            #[repr(transparent)]
            #[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
            pub struct StructureType(i32);
            impl StructureType {
                pub const fn from_raw(value: i32) -> Self { Self(value) }
            }
            #tokens
        };
        assert_valid_rust(&wrapped);
    }

    #[test]
    fn base_pnext_structs_have_self_referential_pointer() {
        let code = emit_base_pnext_structs().to_string();
        assert!(code.contains("p_next : * mut BaseOutStructure"));
        assert!(code.contains("p_next : * const BaseInStructure"));
    }

    // --- Struct emission ---

    #[test]
    fn plain_struct_emits_repr_c() {
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
        let tokens = emit_struct(&def);
        let code = tokens.to_string();
        assert!(code.contains("# [repr (C)]"));
        assert!(code.contains("pub struct Extent2D"));
        assert!(code.contains("pub width : u32"));
        assert!(code.contains("pub height : u32"));
    }

    #[test]
    fn plain_struct_has_zeroed_default() {
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
        let code = emit_struct(&def).to_string();
        assert!(code.contains("std :: mem :: zeroed ()"));
    }

    #[test]
    fn stype_struct_has_manual_default() {
        let def = StructDef {
            name: "BufferCreateInfo".to_string(),
            members: vec![
                MemberDef {
                    values: Some("VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO".to_string()),
                    ..make_member("sType", "VkStructureType")
                },
                make_pointer_member("pNext", "void", true),
                make_member("flags", "VkBufferCreateFlags"),
                make_member("size", "VkDeviceSize"),
            ],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        };
        let code = emit_struct(&def).to_string();
        assert!(code.contains("StructureType :: BUFFER_CREATE_INFO"));
        assert!(code.contains("std :: ptr :: null ()"));
    }

    #[test]
    fn struct_has_doc_alias() {
        let def = StructDef {
            name: "Extent2D".to_string(),
            members: vec![make_member("width", "uint32_t")],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        };
        let code = emit_struct(&def).to_string();
        assert!(code.contains("VkExtent2D"));
    }

    #[test]
    fn keyword_field_gets_raw_ident() {
        let def = StructDef {
            name: "WriteDescriptorSet".to_string(),
            members: vec![make_member("type", "VkDescriptorType")],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        };
        let code = emit_struct(&def).to_string();
        // raw ident shows as `r#type` in token output
        assert!(code.contains("r#type"));
    }

    // --- Union emission ---

    #[test]
    fn union_emits_union_keyword() {
        let def = StructDef {
            name: "ClearColorValue".to_string(),
            members: vec![
                make_array_member("float32", "float", "4"),
                make_array_member("int32", "int32_t", "4"),
                make_array_member("uint32", "uint32_t", "4"),
            ],
            extends: vec![],
            returned_only: false,
            is_union: true,
            provided_by: None,
        };
        let code = emit_union(&def).to_string();
        assert!(code.contains("pub union ClearColorValue"));
        assert!(!code.contains("Debug ,"), "union should not derive Debug");
        assert!(code.contains("impl std :: fmt :: Debug"));
    }

    // --- Marker traits ---

    #[test]
    fn marker_trait_defs_emitted_for_extends_targets() {
        let registry = VkRegistry {
            structs: vec![StructDef {
                name: "PhysicalDeviceVulkan12Features".to_string(),
                members: vec![],
                extends: vec![
                    "PhysicalDeviceFeatures2".to_string(),
                    "DeviceCreateInfo".to_string(),
                ],
                returned_only: false,
                is_union: false,
                provided_by: None,
            }],
            ..empty_registry()
        };
        let code = emit_marker_traits(&registry).to_string();
        assert!(
            code.contains("pub unsafe trait ExtendsPhysicalDeviceFeatures2"),
            "missing trait def for PhysicalDeviceFeatures2"
        );
        assert!(
            code.contains("pub unsafe trait ExtendsDeviceCreateInfo"),
            "missing trait def for DeviceCreateInfo"
        );
    }

    #[test]
    fn marker_trait_impls_emitted() {
        let registry = VkRegistry {
            structs: vec![StructDef {
                name: "PhysicalDeviceVulkan12Features".to_string(),
                members: vec![],
                extends: vec!["DeviceCreateInfo".to_string()],
                returned_only: false,
                is_union: false,
                provided_by: None,
            }],
            ..empty_registry()
        };
        let code = emit_marker_traits(&registry).to_string();
        assert!(
            code.contains("impl ExtendsDeviceCreateInfo for PhysicalDeviceVulkan12Features"),
            "missing trait impl"
        );
    }

    #[test]
    fn marker_traits_deduplicate() {
        let registry = VkRegistry {
            structs: vec![
                StructDef {
                    name: "A".to_string(),
                    members: vec![],
                    extends: vec!["Foo".to_string()],
                    returned_only: false,
                    is_union: false,
                    provided_by: None,
                },
                StructDef {
                    name: "B".to_string(),
                    members: vec![],
                    extends: vec!["Foo".to_string()],
                    returned_only: false,
                    is_union: false,
                    provided_by: None,
                },
            ],
            ..empty_registry()
        };
        let code = emit_marker_traits(&registry).to_string();
        // Trait def should appear exactly once.
        let count = code.matches("pub unsafe trait ExtendsFoo").count();
        assert_eq!(count, 1, "trait ExtendsFoo should be defined exactly once");
        // But two impls.
        let impl_count = code.matches("impl ExtendsFoo for").count();
        assert_eq!(impl_count, 2, "expected two impls of ExtendsFoo");
    }

    // --- Union emission ---

    #[test]
    fn union_has_zeroed_default() {
        let def = StructDef {
            name: "ClearColorValue".to_string(),
            members: vec![make_array_member("float32", "float", "4")],
            extends: vec![],
            returned_only: false,
            is_union: true,
            provided_by: None,
        };
        let code = emit_union(&def).to_string();
        assert!(code.contains("std :: mem :: zeroed ()"));
    }
}
