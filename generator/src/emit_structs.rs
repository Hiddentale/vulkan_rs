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
    let extended_by = build_extended_by_map(registry);

    let structs: Vec<TokenStream> = registry
        .structs
        .iter()
        .filter(|s| !is_base_pnext_struct(&s.name))
        .map(|s| emit_struct_or_union(s, &stype_raw, &extended_by))
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

/// Build reverse map: struct name → list of structs that extend it via pNext.
fn build_extended_by_map(registry: &VkRegistry) -> HashMap<String, Vec<String>> {
    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    for s in &registry.structs {
        for extends in &s.extends {
            map.entry(extends.clone()).or_default().push(s.name.clone());
        }
    }
    // Sort for deterministic output.
    for extenders in map.values_mut() {
        extenders.sort();
    }
    map
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
                    p_next: core::ptr::null_mut(),
                }
            }
        }

        impl core::fmt::Debug for BaseOutStructure {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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
                    p_next: core::ptr::null(),
                }
            }
        }

        impl core::fmt::Debug for BaseInStructure {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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

fn emit_struct_or_union(
    def: &StructDef,
    stype_raw: &HashMap<String, i32>,
    extended_by: &HashMap<String, Vec<String>>,
) -> TokenStream {
    if def.is_union {
        emit_union(def, extended_by)
    } else {
        emit_struct(def, stype_raw, extended_by)
    }
}

/// Build doc comment lines for a struct or union from vk.xml metadata.
fn emit_struct_docs(def: &StructDef, extended_by: &HashMap<String, Vec<String>>) -> TokenStream {
    let vk_name = format!("Vk{}", &def.name);
    let spec_link = format!(
        "[`{vk_name}`](https://registry.khronos.org/vulkan/specs/latest/man/html/{vk_name}.html)"
    );

    let mut doc_lines: Vec<TokenStream> = vec![quote! { #[doc = #spec_link] }];

    // Provided by annotation.
    if let Some(ref provider) = def.provided_by {
        let line = format!("\nProvided by **{provider}**.");
        doc_lines.push(quote! { #[doc = #line] });
    }

    // Returned-only annotation.
    if def.returned_only {
        let line = "\n**Returned only** — filled by Vulkan, not constructed by the application.";
        doc_lines.push(quote! { #[doc = #line] });
    }

    // Extends section (what this struct extends).
    if !def.extends.is_empty() {
        doc_lines.push(quote! { #[doc = ""] });
        doc_lines.push(quote! { #[doc = "# Extends"] });
        for parent in &def.extends {
            let line = format!("- [`{parent}`]");
            doc_lines.push(quote! { #[doc = #line] });
        }
    }

    // Extended By section (what extends this struct).
    if let Some(extenders) = extended_by.get(&def.name) {
        doc_lines.push(quote! { #[doc = ""] });
        doc_lines.push(quote! { #[doc = "# Extended By"] });
        for ext in extenders {
            let line = format!("- [`{ext}`]");
            doc_lines.push(quote! { #[doc = #line] });
        }
    }

    quote! { #(#doc_lines)* }
}

fn emit_struct(
    def: &StructDef,
    stype_raw: &HashMap<String, i32>,
    extended_by: &HashMap<String, Vec<String>>,
) -> TokenStream {
    let name = format_ident!("{}", &def.name);
    let vk_name = format!("Vk{}", &def.name);
    let docs = emit_struct_docs(def, extended_by);
    let fields = emit_fields_with_docs(&def.members);
    let default_impl = emit_default(def, stype_raw);
    let bitfield_accessors = emit_bitfield_accessors(def);

    quote! {
        #docs
        #[repr(C)]
        #[derive(Copy, Clone, Debug)]
        #[doc(alias = #vk_name)]
        pub struct #name {
            #(#fields)*
        }

        #default_impl
        #bitfield_accessors
    }
}

fn emit_union(def: &StructDef, extended_by: &HashMap<String, Vec<String>>) -> TokenStream {
    let name = format_ident!("{}", &def.name);
    let vk_name = format!("Vk{}", &def.name);
    let docs = emit_struct_docs(def, extended_by);
    let fields = emit_fields(&def.members);

    quote! {
        #docs
        #[repr(C)]
        #[derive(Copy, Clone)]
        #[doc(alias = #vk_name)]
        pub union #name {
            #(#fields)*
        }

        impl Default for #name {
            #[inline]
            fn default() -> Self {
                unsafe { core::mem::zeroed() }
            }
        }

        impl core::fmt::Debug for #name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str(stringify!(#name))
            }
        }
    }
}

/// Info about one field within a packed bitfield group.
struct BitfieldEntry {
    name: String,
    offset: u32,
    width: u32,
}

/// Info about a packed bitfield group (one `u32`).
struct BitfieldGroup {
    packed_name: String,
    entries: Vec<BitfieldEntry>,
}

/// Extract bitfield group metadata from members.
fn bitfield_groups(members: &[MemberDef]) -> Vec<BitfieldGroup> {
    let mut groups = Vec::new();
    let mut i = 0;
    let mut counter = 0u32;

    while i < members.len() {
        if !members[i].is_bitfield {
            i += 1;
            continue;
        }

        let mut entries = Vec::new();
        let mut bit_offset = 0u32;
        while i < members.len() && members[i].is_bitfield {
            let width = members[i].bitwidth.unwrap_or(0) as u32;
            entries.push(BitfieldEntry {
                name: members[i].name.clone(),
                offset: bit_offset,
                width,
            });
            bit_offset += width;
            i += 1;
            if bit_offset == 32 {
                break;
            }
        }

        groups.push(BitfieldGroup {
            packed_name: format!("bitfield_{counter}"),
            entries,
        });
        counter += 1;
    }

    groups
}

/// Generate getter and setter methods for packed bitfield groups.
fn emit_bitfield_accessors(def: &StructDef) -> TokenStream {
    let groups = bitfield_groups(&def.members);
    if groups.is_empty() {
        return quote! {};
    }

    let struct_name = format_ident!("{}", &def.name);
    let mut methods = Vec::new();

    for group in &groups {
        let packed_ident = format_ident!("{}", &group.packed_name);

        for entry in &group.entries {
            if entry.name == "reserved" {
                continue;
            }
            let rust_name = member_name(&entry.name);
            let getter = format_ident!("{}", &rust_name);
            let setter = format_ident!("set_{}", &rust_name);
            let offset = entry.offset;
            let width = entry.width;
            let mask = (1u64 << width) - 1;
            let mask_lit = mask as u32;

            if offset == 0 {
                methods.push(quote! {
                    #[inline]
                    pub fn #getter(&self) -> u32 {
                        self.#packed_ident & #mask_lit
                    }

                    #[inline]
                    pub fn #setter(&mut self, val: u32) {
                        self.#packed_ident = (self.#packed_ident & !#mask_lit)
                            | (val & #mask_lit);
                    }
                });
            } else {
                methods.push(quote! {
                    #[inline]
                    pub fn #getter(&self) -> u32 {
                        (self.#packed_ident >> #offset) & #mask_lit
                    }

                    #[inline]
                    pub fn #setter(&mut self, val: u32) {
                        self.#packed_ident = (self.#packed_ident & !(#mask_lit << #offset))
                            | ((val & #mask_lit) << #offset);
                    }
                });
            }
        }
    }

    quote! {
        impl #struct_name {
            #(#methods)*
        }
    }
}

/// Pack consecutive bitfield members into single `u32` fields.
/// `instanceCustomIndex:24` + `mask:8` → `bitfield_0: u32`.
fn pack_bitfields(members: &[MemberDef]) -> Vec<MemberDef> {
    let mut result = Vec::new();
    let mut i = 0;
    let mut bitfield_counter = 0u32;

    while i < members.len() {
        if !members[i].is_bitfield {
            result.push(members[i].clone());
            i += 1;
            continue;
        }

        // Consume consecutive bitfields summing to 32 bits
        let mut total_bits = 0u32;
        while i < members.len() && members[i].is_bitfield {
            let width = members[i].bitwidth.unwrap_or(0) as u32;
            total_bits += width;
            i += 1;
            if total_bits == 32 {
                break;
            }
        }

        let packed_name = format!("bitfield_{bitfield_counter}");
        bitfield_counter += 1;
        result.push(MemberDef {
            name: packed_name,
            type_name: "uint32_t".to_string(),
            is_pointer: false,
            is_const: false,
            is_double_pointer: false,
            array_size: None,
            optional: false,
            values: None,
            len: None,
            extern_sync: None,
            is_bitfield: false,
            bitwidth: None,
        });
    }

    result
}

fn emit_fields(members: &[MemberDef]) -> Vec<TokenStream> {
    let packed = pack_bitfields(members);
    let mut seen = HashSet::new();
    packed
        .iter()
        .filter(|m| seen.insert(m.name.clone()))
        .map(|m| emit_field(m, false))
        .collect()
}

fn emit_fields_with_docs(members: &[MemberDef]) -> Vec<TokenStream> {
    let members = pack_bitfields(members);
    // Build a set of pointer members' len targets so we can annotate count fields.
    let len_targets: HashMap<&str, &str> = members
        .iter()
        .filter(|m| m.is_pointer && m.len.is_some())
        .filter_map(|m| {
            let len = m.len.as_deref()?;
            let count = len.split(',').next()?.trim();
            if count.contains("null-terminated") || count.contains('/') {
                return None;
            }
            Some((count, m.name.as_str()))
        })
        .collect();

    let mut seen = HashSet::new();
    members
        .iter()
        .filter(|m| seen.insert(m.name.clone()))
        .map(|m| {
            let field = emit_field(m, true);
            let mut doc_lines: Vec<TokenStream> = Vec::new();

            // Fixed value (sType).
            if let Some(ref values) = m.values {
                let line = format!("Must be `{values}`.");
                doc_lines.push(quote! { #[doc = #line] });
            }

            // Optional pointer.
            if m.optional && m.is_pointer {
                doc_lines.push(quote! { #[doc = "Optional — may be null."] });
            }

            // Count field paired with a pointer.
            if let Some(ptr_name) = len_targets.get(m.name.as_str()) {
                let ptr_rust = member_name(ptr_name);
                let line = format!("Length of `{ptr_rust}`.");
                doc_lines.push(quote! { #[doc = #line] });
            }

            // Thread safety.
            if m.extern_sync.is_some() {
                doc_lines.push(
                    quote! { #[doc = "**Thread safety:** must be externally synchronized."] },
                );
            }

            quote! {
                #(#doc_lines)*
                #field
            }
        })
        .collect()
}

fn emit_field(member: &MemberDef, _with_docs: bool) -> TokenStream {
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
        let packed = pack_bitfields(&def.members);
        let mut seen = HashSet::new();
        let field_defaults: Vec<TokenStream> = packed
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
                    unsafe { core::mem::zeroed() }
                }
            }
        }
    }
}

fn default_value_for(member: &MemberDef) -> TokenStream {
    if member.is_pointer || member.is_double_pointer {
        if member.is_const {
            return quote! { core::ptr::null() };
        } else {
            return quote! { core::ptr::null_mut() };
        }
    }

    if member.array_size.is_some() {
        return quote! { unsafe { core::mem::zeroed() } };
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
            let rust_name = name.clone();
            quote! {
                /// Marker trait for structs that can appear in the pNext chain of
                #[doc = concat!("[`", #rust_name, "`].")]
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
            is_bitfield: false,
            bitwidth: None,
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
        let tokens = emit_struct(&def, &HashMap::new(), &HashMap::new());
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
        let code = emit_struct(&def, &HashMap::new(), &HashMap::new()).to_string();
        assert!(code.contains("core :: mem :: zeroed ()"));
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
        let code = emit_struct(&def, &raw, &HashMap::new()).to_string();
        assert!(code.contains("from_raw"));
        assert!(code.contains("core :: ptr :: null ()"));
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
        let code = emit_struct(&def, &HashMap::new(), &HashMap::new()).to_string();
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
        let code = emit_struct(&def, &HashMap::new(), &HashMap::new()).to_string();
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
        let code = emit_union(&def, &HashMap::new()).to_string();
        assert!(code.contains("pub union ClearColorValue"));
        assert!(!code.contains("Debug ,"), "union should not derive Debug");
        assert!(code.contains("impl core :: fmt :: Debug"));
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
        let code = emit_union(&def, &HashMap::new()).to_string();
        assert!(code.contains("core :: mem :: zeroed ()"));
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

    // --- Struct documentation (1.1.1–1.1.6) ---

    #[test]
    fn struct_has_spec_link() {
        let def = StructDef {
            name: "Extent2D".to_string(),
            members: vec![make_member("width", "uint32_t")],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        };
        let code = emit_struct(&def, &HashMap::new(), &HashMap::new()).to_string();
        assert!(
            code.contains("registry.khronos.org/vulkan/specs/latest/man/html/VkExtent2D.html"),
            "missing spec link"
        );
    }

    #[test]
    fn struct_has_provided_by_when_present() {
        let def = StructDef {
            name: "SwapchainCreateInfoKHR".to_string(),
            members: vec![make_member("flags", "uint32_t")],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: Some("VK_KHR_swapchain".to_string()),
        };
        let code = emit_struct(&def, &HashMap::new(), &HashMap::new()).to_string();
        assert!(code.contains("VK_KHR_swapchain"), "missing provided_by");
    }

    #[test]
    fn struct_no_provided_by_when_absent() {
        let def = StructDef {
            name: "Extent2D".to_string(),
            members: vec![make_member("width", "uint32_t")],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        };
        let code = emit_struct(&def, &HashMap::new(), &HashMap::new()).to_string();
        assert!(!code.contains("Provided by"), "should have no provided_by");
    }

    #[test]
    fn struct_has_returned_only_annotation() {
        let def = StructDef {
            name: "PhysicalDeviceProperties".to_string(),
            members: vec![make_member("apiVersion", "uint32_t")],
            extends: vec![],
            returned_only: true,
            is_union: false,
            provided_by: None,
        };
        let code = emit_struct(&def, &HashMap::new(), &HashMap::new()).to_string();
        assert!(
            code.contains("Returned only"),
            "missing returned_only annotation"
        );
    }

    #[test]
    fn struct_no_returned_only_when_false() {
        let def = StructDef {
            name: "Extent2D".to_string(),
            members: vec![make_member("width", "uint32_t")],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        };
        let code = emit_struct(&def, &HashMap::new(), &HashMap::new()).to_string();
        assert!(!code.contains("Returned only"));
    }

    #[test]
    fn struct_has_extends_section() {
        let def = StructDef {
            name: "PhysicalDeviceVulkan12Features".to_string(),
            members: vec![],
            extends: vec![
                "PhysicalDeviceFeatures2".to_string(),
                "DeviceCreateInfo".to_string(),
            ],
            returned_only: false,
            is_union: false,
            provided_by: None,
        };
        let code = emit_struct(&def, &HashMap::new(), &HashMap::new()).to_string();
        assert!(code.contains("Extends"), "missing Extends section");
        assert!(code.contains("PhysicalDeviceFeatures2"));
        assert!(code.contains("DeviceCreateInfo"));
    }

    #[test]
    fn struct_has_extended_by_section() {
        let def = StructDef {
            name: "DeviceCreateInfo".to_string(),
            members: vec![],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        };
        let mut extended_by = HashMap::new();
        extended_by.insert(
            "DeviceCreateInfo".to_string(),
            vec!["PhysicalDeviceVulkan12Features".to_string()],
        );
        let code = emit_struct(&def, &HashMap::new(), &extended_by).to_string();
        assert!(code.contains("Extended By"), "missing Extended By section");
        assert!(code.contains("PhysicalDeviceVulkan12Features"));
    }

    #[test]
    fn struct_no_extended_by_when_none() {
        let def = StructDef {
            name: "Extent2D".to_string(),
            members: vec![make_member("width", "uint32_t")],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        };
        let code = emit_struct(&def, &HashMap::new(), &HashMap::new()).to_string();
        assert!(!code.contains("Extended By"));
    }

    #[test]
    fn field_stype_has_values_doc() {
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
        let code = emit_struct(&def, &make_stype_raw_map(), &HashMap::new()).to_string();
        assert!(
            code.contains("VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO"),
            "sType field should document its required value"
        );
    }

    #[test]
    fn field_optional_pointer_has_doc() {
        let def = StructDef {
            name: "Test".to_string(),
            members: vec![MemberDef {
                optional: true,
                ..make_pointer_member("pAllocator", "VkAllocationCallbacks", true)
            }],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        };
        let code = emit_struct(&def, &HashMap::new(), &HashMap::new()).to_string();
        assert!(
            code.contains("Optional"),
            "optional pointer should be annotated"
        );
    }

    #[test]
    fn field_count_has_length_of_doc() {
        let def = StructDef {
            name: "Test".to_string(),
            members: vec![
                make_member("queueFamilyIndexCount", "uint32_t"),
                MemberDef {
                    len: Some("queueFamilyIndexCount".to_string()),
                    ..make_pointer_member("pQueueFamilyIndices", "uint32_t", true)
                },
            ],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        };
        let code = emit_struct(&def, &HashMap::new(), &HashMap::new()).to_string();
        assert!(
            code.contains("Length of"),
            "count field should document its paired pointer"
        );
    }

    #[test]
    fn field_extern_sync_has_thread_safety_doc() {
        let def = StructDef {
            name: "Test".to_string(),
            members: vec![MemberDef {
                extern_sync: Some("true".to_string()),
                ..make_member("commandBuffer", "VkCommandBuffer")
            }],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        };
        let code = emit_struct(&def, &HashMap::new(), &HashMap::new()).to_string();
        assert!(
            code.contains("externally synchronized"),
            "extern_sync field should have thread safety doc"
        );
    }

    #[test]
    fn union_has_spec_link() {
        let def = StructDef {
            name: "ClearColorValue".to_string(),
            members: vec![make_array_member("float32", "float", "4")],
            extends: vec![],
            returned_only: false,
            is_union: true,
            provided_by: None,
        };
        let code = emit_union(&def, &HashMap::new()).to_string();
        assert!(
            code.contains(
                "registry.khronos.org/vulkan/specs/latest/man/html/VkClearColorValue.html"
            ),
            "union missing spec link"
        );
    }
}
