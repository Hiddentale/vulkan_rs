//! Emits `#[repr(C)]` struct and union definitions for all Vulkan types.

use std::collections::{HashMap, HashSet};

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::emit_aliases;
use crate::parse::{MemberDef, StructDef, VkRegistry};
use crate::resolve_types::{is_rust_keyword, member_name, resolve_member_type};
use crate::stype;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Emit all struct and union definitions plus pNext marker traits.
pub fn emit_structs(registry: &VkRegistry) -> TokenStream {
    let base_structs = emit_base_pnext_structs();
    let func_pointer_stubs = emit_aliases::emit_func_pointer_stubs(registry);
    let stdvideo_stubs = emit_aliases::emit_stdvideo_stubs(registry);
    let flags_aliases = emit_aliases::emit_flags_aliases(registry);
    let type_aliases = emit_aliases::emit_type_aliases(registry);

    let stype_raw = stype::build_raw_map(registry);

    let structs: Vec<TokenStream> = registry
        .structs
        .iter()
        .filter(|s| !is_base_pnext_struct(&s.name))
        .map(|s| emit_struct_or_union(s, &stype_raw))
        .collect();

    let marker_traits = emit_marker_traits(registry);

    quote! {
        use super::enums::*;
        use super::handles::*;
        use super::bitmasks::*;
        use super::constants::*;

        #func_pointer_stubs
        #stdvideo_stubs
        #flags_aliases
        #type_aliases
        #base_structs
        #(#structs)*
        #marker_traits
    }
}

// ---------------------------------------------------------------------------
// BaseOutStructure / BaseInStructure
// ---------------------------------------------------------------------------

const BASE_PNEXT_STRUCTS: &[&str] = &["BaseOutStructure", "BaseInStructure"];

pub fn is_base_pnext_struct(name: &str) -> bool {
    BASE_PNEXT_STRUCTS.contains(&name)
}

/// Returns true if the struct has both sType and pNext members.
pub fn has_stype_pnext(def: &StructDef) -> bool {
    def.members.iter().any(|m| m.name == "sType") && def.members.iter().any(|m| m.name == "pNext")
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
// Struct and union emission
// ---------------------------------------------------------------------------

fn emit_struct_or_union(def: &StructDef, stype_raw: &HashMap<String, i32>) -> TokenStream {
    if def.is_union {
        emit_union(def)
    } else {
        emit_struct(def, stype_raw)
    }
}

fn emit_struct(def: &StructDef, stype_raw: &HashMap<String, i32>) -> TokenStream {
    let name = format_ident!("{}", &def.name);
    let vk_name = format!("Vk{}", &def.name);
    let fields = emit_fields(&def.members);
    let default_impl = emit_default(def, stype_raw);

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
    let mut seen = HashSet::new();
    members
        .iter()
        .filter(|m| seen.insert(m.name.clone()))
        .map(emit_field)
        .collect()
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

fn emit_default(def: &StructDef, stype_raw: &HashMap<String, i32>) -> TokenStream {
    let name = format_ident!("{}", &def.name);
    let stype_val = stype::struct_stype(def, stype_raw);
    let has_pnext = def.members.iter().any(|m| m.name == "pNext");

    if stype_val.is_some() || has_pnext {
        let mut seen = HashSet::new();
        let field_defaults: Vec<TokenStream> = def
            .members
            .iter()
            .filter(|m| seen.insert(m.name.clone()))
            .map(|m| {
                let rust_name = member_name(&m.name);
                let field_ident = if is_rust_keyword(&rust_name) {
                    format_ident!("r#{}", rust_name)
                } else {
                    format_ident!("{}", rust_name)
                };

                if m.name == "sType"
                    && let Some(ref stype_token) = stype_val
                {
                    return quote! { #field_ident: #stype_token, };
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

fn default_value_for(member: &MemberDef) -> TokenStream {
    if member.is_pointer || member.is_double_pointer {
        if member.is_const {
            return quote! { std::ptr::null() };
        } else {
            return quote! { std::ptr::null_mut() };
        }
    }

    if member.array_size.is_some() {
        return quote! { unsafe { std::mem::zeroed() } };
    }

    quote! { Default::default() }
}

// ---------------------------------------------------------------------------
// Marker traits for pNext chains
// ---------------------------------------------------------------------------

fn emit_marker_traits(registry: &VkRegistry) -> TokenStream {
    use std::collections::BTreeSet;

    let mut trait_names: BTreeSet<String> = BTreeSet::new();
    for s in &registry.structs {
        for extends in &s.extends {
            trait_names.insert(extends.clone());
        }
        if !s.returned_only && has_stype_pnext(s) {
            trait_names.insert(s.name.clone());
        }
    }

    let trait_defs: Vec<TokenStream> = trait_names
        .iter()
        .map(|name| {
            let trait_ident = format_ident!("Extends{}", name);
            let vk_name = format!("Vk{}", name);
            quote! {
                /// Marker trait for structs that can appear in the pNext chain of
                #[doc = concat!("[`", #vk_name, "`].")]
                ///
                /// # Safety
                /// Implementors must be valid pNext chain members for the target struct.
                pub unsafe trait #trait_ident {}
            }
        })
        .collect();

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
            aliases: vec![],
            base_types: std::collections::HashMap::new(),
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

    fn make_array_member(name: &str, type_name: &str, size: &str) -> MemberDef {
        MemberDef {
            array_size: Some(size.to_string()),
            ..make_member(name, type_name)
        }
    }

    fn make_stype_raw_map() -> HashMap<String, i32> {
        let mut m = HashMap::new();
        m.insert("VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO".to_string(), 12);
        m
    }

    // --- Base pNext structs ---

    #[test]
    fn base_pnext_structs_valid_rust() {
        let tokens = emit_base_pnext_structs();
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
        let tokens = emit_struct(&def, &HashMap::new());
        let code = tokens.to_string();
        assert!(code.contains("# [repr (C)]"));
        assert!(code.contains("pub struct Extent2D"));
        assert!(code.contains("pub width : u32"));
    }

    #[test]
    fn plain_struct_has_zeroed_default() {
        let def = StructDef {
            name: "Extent2D".to_string(),
            members: vec![make_member("width", "uint32_t")],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        };
        let code = emit_struct(&def, &HashMap::new()).to_string();
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
            ],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        };
        let raw = make_stype_raw_map();
        let code = emit_struct(&def, &raw).to_string();
        assert!(code.contains("from_raw"));
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
        let code = emit_struct(&def, &HashMap::new()).to_string();
        assert!(code.contains("VkExtent2D"));
    }

    #[test]
    fn keyword_field_gets_raw_ident() {
        let def = StructDef {
            name: "Test".to_string(),
            members: vec![make_member("type", "VkDescriptorType")],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        };
        let code = emit_struct(&def, &HashMap::new()).to_string();
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
        assert!(code.contains("pub unsafe trait ExtendsPhysicalDeviceFeatures2"));
        assert!(code.contains("pub unsafe trait ExtendsDeviceCreateInfo"));
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
        assert!(code.contains("impl ExtendsDeviceCreateInfo for PhysicalDeviceVulkan12Features"));
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
        let count = code.matches("pub unsafe trait ExtendsFoo").count();
        assert_eq!(count, 1);
        let impl_count = code.matches("impl ExtendsFoo for").count();
        assert_eq!(impl_count, 2);
    }

    // --- Public helpers ---

    #[test]
    fn has_stype_pnext_true_for_extensible_struct() {
        let def = StructDef {
            name: "BufferCreateInfo".to_string(),
            members: vec![
                make_member("sType", "VkStructureType"),
                make_pointer_member("pNext", "void", true),
                make_member("flags", "uint32_t"),
            ],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        };
        assert!(has_stype_pnext(&def));
    }

    #[test]
    fn has_stype_pnext_false_for_plain_struct() {
        let def = StructDef {
            name: "Extent2D".to_string(),
            members: vec![make_member("width", "uint32_t")],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        };
        assert!(!has_stype_pnext(&def));
    }

    #[test]
    fn is_base_pnext_struct_recognizes_both() {
        assert!(is_base_pnext_struct("BaseOutStructure"));
        assert!(is_base_pnext_struct("BaseInStructure"));
    }

    #[test]
    fn is_base_pnext_struct_rejects_normal() {
        assert!(!is_base_pnext_struct("BufferCreateInfo"));
        assert!(!is_base_pnext_struct("InstanceCreateInfo"));
    }
}
