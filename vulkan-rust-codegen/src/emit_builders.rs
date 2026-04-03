//! Emits builder patterns for Vulkan structs.
//!
//! Extensible structs (with sType/pNext) get lifetime-tied builders with
//! `push_next` support. Plain structs get simple builders without lifetimes.

use std::collections::HashSet;

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::emit_structs::{has_stype_pnext, is_base_pnext_struct};
use crate::parse::{MemberDef, StructDef, VkRegistry};
use crate::resolve_types::{is_rust_keyword, member_name, resolve_base_type, resolve_member_type};
use crate::stype;

/// Emit builders for all structs.
pub fn emit_builders(registry: &VkRegistry) -> TokenStream {
    let stype_raw = stype::build_raw_map(registry);
    let builders: Vec<TokenStream> = registry
        .structs
        .iter()
        .filter(|s| !s.is_union && !is_base_pnext_struct(&s.name) && !has_bitfield_members(s))
        .map(|s| {
            if has_stype_pnext(s) {
                emit_builder(s, &stype_raw)
            } else {
                emit_plain_builder(s)
            }
        })
        .collect();

    quote! {
        use super::structs::*;
        use super::enums::*;
        use super::handles::*;
        use super::bitmasks::*;
        use super::constants::*;

        #(#builders)*
    }
}

fn emit_builder(
    def: &StructDef,
    stype_raw: &std::collections::HashMap<String, i32>,
) -> TokenStream {
    let struct_name = format_ident!("{}", &def.name);
    let builder_name = format_ident!("{}Builder", &def.name);

    // Build the set of count fields that are paired with a pointer via `len`.
    let count_fields = collect_count_fields(def);

    let mut seen_setters = std::collections::HashSet::new();
    let setters: Vec<TokenStream> = def
        .members
        .iter()
        .filter(|m| {
            m.name != "sType"
                && m.name != "pNext"
                && !count_fields.contains(&m.name)
                && seen_setters.insert(m.name.clone())
        })
        .map(|m| emit_setter(m, def))
        .collect();

    let stype_val = stype::struct_stype(def, stype_raw);
    let default_stype = stype_val.unwrap_or_else(|| quote! { Default::default() });

    // push_next: only if any struct extends this one.
    let extends_trait = format_ident!("Extends{}", &def.name);
    let pnext_is_mut = def.members.iter().any(|m| m.name == "pNext" && !m.is_const);
    let push_next = if !def.returned_only {
        let assign_pnext = if pnext_is_mut {
            quote! { self.inner.p_next = <*mut BaseOutStructure>::cast::<core::ffi::c_void>(next_ptr); }
        } else {
            quote! { self.inner.p_next = <*mut BaseOutStructure>::cast::<core::ffi::c_void>(next_ptr) as *const _; }
        };
        let push_next_doc = format!(
            "Prepend a struct to the pNext chain. See [`{}`]'s **Extended By** section for valid types.",
            &def.name,
        );
        quote! {
            #[doc = #push_next_doc]
            #[inline]
            pub fn push_next<T: #extends_trait>(mut self, next: &'a mut T) -> Self {
                unsafe {
                    let next_ptr = <*mut T>::cast::<BaseOutStructure>(next);
                    (*next_ptr).p_next = self.inner.p_next as *mut _;
                    #assign_pnext
                }
                self
            }
        }
    } else {
        quote! {}
    };

    let builder_doc = format!(
        "Builder for [`{}`] with lifetime-tied pNext safety.",
        &def.name
    );

    quote! {
        #[doc = #builder_doc]
        pub struct #builder_name<'a> {
            inner: #struct_name,
            _marker: core::marker::PhantomData<&'a ()>,
        }

        impl #struct_name {
            /// Start building this struct; `s_type` is already set to the correct variant.
            #[inline]
            pub fn builder<'a>() -> #builder_name<'a> {
                #builder_name {
                    inner: #struct_name {
                        s_type: #default_stype,
                        ..Default::default()
                    },
                    _marker: core::marker::PhantomData,
                }
            }
        }

        impl<'a> #builder_name<'a> {
            #(#setters)*
            #push_next
        }

        impl<'a> core::ops::Deref for #builder_name<'a> {
            type Target = #struct_name;
            #[inline]
            fn deref(&self) -> &Self::Target { &self.inner }
        }

        impl<'a> core::ops::DerefMut for #builder_name<'a> {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target { &mut self.inner }
        }
    }
}

/// Emit a simple builder for a plain struct (no sType/pNext).
fn emit_plain_builder(def: &StructDef) -> TokenStream {
    let struct_name = format_ident!("{}", &def.name);
    let builder_name = format_ident!("{}Builder", &def.name);

    let count_fields = collect_count_fields(def);
    let needs_lifetime = def.members.iter().any(|m| m.is_pointer);

    let mut seen_setters = std::collections::HashSet::new();
    let setters: Vec<TokenStream> = def
        .members
        .iter()
        .filter(|m| !count_fields.contains(&m.name) && seen_setters.insert(m.name.clone()))
        .map(|m| emit_setter(m, def))
        .collect();

    let builder_doc = format!("Builder for [`{}`].", &def.name);

    if needs_lifetime {
        quote! {
            #[doc = #builder_doc]
            pub struct #builder_name<'a> {
                inner: #struct_name,
                _marker: core::marker::PhantomData<&'a ()>,
            }

            impl #struct_name {
                /// Start building this struct.
                #[inline]
                pub fn builder<'a>() -> #builder_name<'a> {
                    #builder_name {
                        inner: #struct_name { ..Default::default() },
                        _marker: core::marker::PhantomData,
                    }
                }
            }

            impl<'a> #builder_name<'a> {
                #(#setters)*
            }

            impl<'a> core::ops::Deref for #builder_name<'a> {
                type Target = #struct_name;
                #[inline]
                fn deref(&self) -> &Self::Target { &self.inner }
            }

            impl<'a> core::ops::DerefMut for #builder_name<'a> {
                #[inline]
                fn deref_mut(&mut self) -> &mut Self::Target { &mut self.inner }
            }
        }
    } else {
        quote! {
            #[doc = #builder_doc]
            pub struct #builder_name {
                inner: #struct_name,
            }

            impl #struct_name {
                /// Start building this struct.
                #[inline]
                pub fn builder() -> #builder_name {
                    #builder_name {
                        inner: #struct_name { ..Default::default() },
                    }
                }
            }

            impl #builder_name {
                #(#setters)*
            }

            impl core::ops::Deref for #builder_name {
                type Target = #struct_name;
                #[inline]
                fn deref(&self) -> &Self::Target { &self.inner }
            }

            impl core::ops::DerefMut for #builder_name {
                #[inline]
                fn deref_mut(&mut self) -> &mut Self::Target { &mut self.inner }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Setters
// ---------------------------------------------------------------------------

fn emit_setter(member: &MemberDef, parent: &StructDef) -> TokenStream {
    // Check if this pointer member has a paired count field (slice setter).
    if member.is_pointer
        && !member.is_double_pointer
        && let Some(ref len) = member.len
        && let Some(count_member) = find_count_member(parent, len)
    {
        if is_byte_count_len(len) {
            return emit_byte_slice_setter(member, count_member);
        }
        return emit_slice_setter(member, count_member);
    }

    // const char* const* with a count field → &[*const c_char] slice setter.
    // Covers ppEnabledLayerNames, ppEnabledExtensionNames, etc.
    // Some fields (deprecated device layers) lack a `len` attribute in vk.xml,
    // so we also try inferring the count field: ppEnabledFooNames → enabledFooCount.
    if member.is_double_pointer && member.is_const && member.type_name == "char" {
        let count_member = member
            .len
            .as_deref()
            .and_then(|len| find_count_member(parent, len))
            .or_else(|| infer_count_member(parent, &member.name));
        if let Some(count_member) = count_member {
            return emit_cstr_ptr_slice_setter(member, count_member);
        }
    }

    emit_simple_setter(member)
}

fn emit_simple_setter(member: &MemberDef) -> TokenStream {
    let rust_name = member_name(&member.name);
    let ident = if is_rust_keyword(&rust_name) {
        format_ident!("r#{}", rust_name)
    } else {
        format_ident!("{}", rust_name)
    };

    // VkBool32 fields accept `bool` and cast to u32 internally.
    if member.type_name == "VkBool32" && !member.is_pointer {
        return quote! {
            #[inline]
            pub fn #ident(mut self, value: bool) -> Self {
                self.inner.#ident = value as u32;
                self
            }
        };
    }

    // *const char fields accept &CStr and call .as_ptr() internally.
    if member.type_name == "char"
        && member.is_pointer
        && member.is_const
        && !member.is_double_pointer
    {
        return quote! {
            #[inline]
            pub fn #ident(mut self, value: &'a core::ffi::CStr) -> Self {
                self.inner.#ident = value.as_ptr();
                self
            }
        };
    }

    // *const VkFoo fields accept &T instead of raw pointers.
    if member.is_pointer
        && member.is_const
        && !member.is_double_pointer
        && member.type_name.starts_with("Vk")
    {
        let base = resolve_base_type(&member.type_name);
        return quote! {
            #[inline]
            pub fn #ident(mut self, value: &'a #base) -> Self {
                self.inner.#ident = value;
                self
            }
        };
    }

    let ty = resolve_member_type(member);

    quote! {
        #[inline]
        pub fn #ident(mut self, value: #ty) -> Self {
            self.inner.#ident = value;
            self
        }
    }
}

fn emit_slice_setter(ptr_member: &MemberDef, count_member: &MemberDef) -> TokenStream {
    let rust_name = member_name(&ptr_member.name);
    // Strip leading `p_` for the setter name: `p_queue_family_indices` → `queue_family_indices`.
    let setter_name = rust_name.strip_prefix("p_").unwrap_or(&rust_name);
    let setter_ident = format_ident!("{}", setter_name);

    let ptr_field = format_ident!("{}", member_name(&ptr_member.name));
    let count_field = format_ident!("{}", member_name(&count_member.name));

    // The element type is the base type (without pointer wrapping).
    let elem_ty = resolve_base_type(&ptr_member.type_name);

    // Count field type (usually u32). Avoid redundant `as usize` casts.
    let count_ty = resolve_member_type(count_member);
    let is_usize = count_member.type_name == "size_t";
    let set_count = if is_usize {
        quote! { self.inner.#count_field = slice.len(); }
    } else {
        quote! { self.inner.#count_field = slice.len() as #count_ty; }
    };

    if ptr_member.is_const {
        quote! {
            #[inline]
            pub fn #setter_ident(mut self, slice: &'a [#elem_ty]) -> Self {
                #set_count
                self.inner.#ptr_field = slice.as_ptr();
                self
            }
        }
    } else {
        quote! {
            #[inline]
            pub fn #setter_ident(mut self, slice: &'a mut [#elem_ty]) -> Self {
                #set_count
                self.inner.#ptr_field = slice.as_mut_ptr();
                self
            }
        }
    }
}

/// Emit a slice setter for `const char* const*` fields (e.g. `ppEnabledLayerNames`).
/// Accepts `&[*const c_char]` and sets both the pointer and count.
fn emit_cstr_ptr_slice_setter(ptr_member: &MemberDef, count_member: &MemberDef) -> TokenStream {
    let rust_name = member_name(&ptr_member.name);
    // Strip leading `pp_` for the setter name: `pp_enabled_layer_names` → `enabled_layer_names`.
    let setter_name = rust_name
        .strip_prefix("pp_")
        .or_else(|| rust_name.strip_prefix("p_"))
        .unwrap_or(&rust_name);
    let setter_ident = format_ident!("{}", setter_name);

    let ptr_field = format_ident!("{}", member_name(&ptr_member.name));
    let count_field = format_ident!("{}", member_name(&count_member.name));

    let count_ty = resolve_member_type(count_member);

    quote! {
        #[inline]
        pub fn #setter_ident(mut self, slice: &'a [*const core::ffi::c_char]) -> Self {
            self.inner.#count_field = slice.len() as #count_ty;
            self.inner.#ptr_field = slice.as_ptr();
            self
        }
    }
}

/// Emit a slice setter where the count field is in bytes, not elements.
/// For `pCode` with `len="codeSize / 4"`, this emits a setter that takes
/// `&[u32]` and sets `code_size = slice.len() * size_of::<u32>()`.
fn emit_byte_slice_setter(ptr_member: &MemberDef, count_member: &MemberDef) -> TokenStream {
    let rust_name = member_name(&ptr_member.name);
    let setter_name = rust_name.strip_prefix("p_").unwrap_or(&rust_name);
    let setter_ident = format_ident!("{}", setter_name);

    let ptr_field = format_ident!("{}", member_name(&ptr_member.name));
    let count_field = format_ident!("{}", member_name(&count_member.name));
    let elem_ty = resolve_base_type(&ptr_member.type_name);

    if ptr_member.is_const {
        quote! {
            #[inline]
            pub fn #setter_ident(mut self, slice: &'a [#elem_ty]) -> Self {
                self.inner.#count_field = core::mem::size_of_val(slice);
                self.inner.#ptr_field = slice.as_ptr();
                self
            }
        }
    } else {
        quote! {
            #[inline]
            pub fn #setter_ident(mut self, slice: &'a mut [#elem_ty]) -> Self {
                self.inner.#count_field = core::mem::size_of_val(slice);
                self.inner.#ptr_field = slice.as_mut_ptr();
                self
            }
        }
    }
}

fn has_bitfield_members(def: &StructDef) -> bool {
    def.members.iter().any(|m| m.is_bitfield)
}

// ---------------------------------------------------------------------------
// Pointer/count pairing
// ---------------------------------------------------------------------------

/// Collect the names of all count fields that are paired with a pointer via `len`.
fn collect_count_fields(def: &StructDef) -> HashSet<String> {
    let mut counts = HashSet::new();
    for m in &def.members {
        // Single-pointer arrays and const char* const* arrays both have count fields.
        let is_slice_candidate =
            m.is_pointer && (!m.is_double_pointer || (m.is_const && m.type_name == "char"));
        if is_slice_candidate && let Some(ref len) = m.len {
            // When the pointer is optional, the count has independent meaning
            // (e.g. descriptorCount with optional pImmutableSamplers), so keep
            // the count field's standalone setter.
            if m.optional {
                continue;
            }
            // len can be a comma-separated list; take the first element.
            let count_name = len.split(',').next().unwrap_or(len).trim();
            // Skip "null-terminated".
            if count_name.contains("null-terminated") {
                continue;
            }
            // Handle division expressions like "codeSize / 4".
            let base_name = if let Some((name, _)) = count_name.split_once('/') {
                name.trim()
            } else {
                count_name
            };
            if def.members.iter().any(|other| other.name == base_name) {
                counts.insert(base_name.to_string());
            }
        }
        // Also check inferred count for double-pointer char fields without len.
        if m.is_double_pointer
            && m.is_const
            && m.type_name == "char"
            && m.len.is_none()
            && let Some(count) = infer_count_member(def, &m.name)
        {
            counts.insert(count.name.clone());
        }
    }
    counts
}

/// Find the count member referenced by a pointer's `len` attribute.
/// Infer a count member for `ppEnabledFooNames` → `enabledFooCount`.
fn infer_count_member<'a>(def: &'a StructDef, ptr_name: &str) -> Option<&'a MemberDef> {
    let stem = ptr_name.strip_prefix("ppEnabled")?.strip_suffix("Names")?;
    let count_name = format!("enabled{stem}Count");
    def.members.iter().find(|m| m.name == count_name)
}

fn find_count_member<'a>(def: &'a StructDef, len: &str) -> Option<&'a MemberDef> {
    let count_name = len.split(',').next()?.trim();
    if count_name.contains("null-terminated") {
        return None;
    }
    // Handle division expressions like "codeSize / 4".
    let base_name = if let Some((name, _)) = count_name.split_once('/') {
        name.trim()
    } else {
        count_name
    };
    def.members.iter().find(|m| m.name == base_name)
}

/// Check if a len expression contains a division (e.g. "codeSize / 4").
fn is_byte_count_len(len: &str) -> bool {
    let count_name = len.split(',').next().unwrap_or(len).trim();
    count_name.contains('/')
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::MemberDef;

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

    fn make_stype_member() -> MemberDef {
        MemberDef {
            values: Some("VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO".to_string()),
            ..make_member("sType", "VkStructureType")
        }
    }

    fn make_pnext_member() -> MemberDef {
        MemberDef {
            is_pointer: true,
            is_const: false,
            ..make_member("pNext", "void")
        }
    }

    fn make_pointer_member(name: &str, type_name: &str, len: Option<&str>) -> MemberDef {
        MemberDef {
            is_pointer: true,
            is_const: true,
            len: len.map(str::to_string),
            ..make_member(name, type_name)
        }
    }

    fn make_buffer_create_info() -> StructDef {
        StructDef {
            name: "BufferCreateInfo".to_string(),
            members: vec![
                make_stype_member(),
                make_pnext_member(),
                make_member("flags", "VkBufferCreateFlags"),
                make_member("size", "VkDeviceSize"),
                make_member("usage", "VkBufferUsageFlags"),
                make_member("sharingMode", "VkSharingMode"),
                make_member("queueFamilyIndexCount", "uint32_t"),
                make_pointer_member(
                    "pQueueFamilyIndices",
                    "uint32_t",
                    Some("queueFamilyIndexCount"),
                ),
            ],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        }
    }

    #[test]
    fn builder_struct_emitted() {
        let def = make_buffer_create_info();
        let code = emit_builder(&def, &std::collections::HashMap::new()).to_string();
        assert!(code.contains("pub struct BufferCreateInfoBuilder"));
        assert!(code.contains("PhantomData"));
    }

    #[test]
    fn builder_has_builder_fn_on_struct() {
        let def = make_buffer_create_info();
        let mut raw = std::collections::HashMap::new();
        raw.insert("VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO".to_string(), 12i32);
        let code = emit_builder(&def, &raw).to_string();
        assert!(code.contains("impl BufferCreateInfo"));
        assert!(code.contains("fn builder"));
        assert!(code.contains("from_raw"));
    }

    #[test]
    fn builder_skips_stype_and_pnext_setters() {
        let def = make_buffer_create_info();
        let code = emit_builder(&def, &std::collections::HashMap::new()).to_string();
        // Should NOT have a setter named s_type or p_next.
        assert!(!code.contains("fn s_type"), "should skip sType setter");
        assert!(!code.contains("fn p_next"), "should skip pNext setter");
    }

    #[test]
    fn builder_has_simple_setters() {
        let def = make_buffer_create_info();
        let code = emit_builder(&def, &std::collections::HashMap::new()).to_string();
        assert!(code.contains("fn flags"), "missing flags setter");
        assert!(code.contains("fn size"), "missing size setter");
        assert!(code.contains("fn usage"), "missing usage setter");
        assert!(
            code.contains("fn sharing_mode"),
            "missing sharing_mode setter"
        );
    }

    #[test]
    fn builder_has_slice_setter_for_pointer_count_pair() {
        let def = make_buffer_create_info();
        let code = emit_builder(&def, &std::collections::HashMap::new()).to_string();
        assert!(
            code.contains("fn queue_family_indices"),
            "missing slice setter"
        );
        assert!(
            code.contains("slice . len ()"),
            "slice setter should set count from slice.len()"
        );
        assert!(
            code.contains("slice . as_ptr ()"),
            "slice setter should set pointer from slice.as_ptr()"
        );
    }

    #[test]
    fn builder_skips_count_field_setter() {
        let def = make_buffer_create_info();
        let code = emit_builder(&def, &std::collections::HashMap::new()).to_string();
        assert!(
            !code.contains("fn queue_family_index_count"),
            "count field should not have its own setter"
        );
    }

    #[test]
    fn builder_has_push_next() {
        let def = make_buffer_create_info();
        let code = emit_builder(&def, &std::collections::HashMap::new()).to_string();
        assert!(code.contains("fn push_next"), "missing push_next");
        assert!(
            code.contains("ExtendsBufferCreateInfo"),
            "push_next should use extends trait"
        );
    }

    #[test]
    fn builder_has_deref() {
        let def = make_buffer_create_info();
        let code = emit_builder(&def, &std::collections::HashMap::new()).to_string();
        assert!(code.contains("impl < 'a > core :: ops :: Deref for BufferCreateInfoBuilder"));
        assert!(code.contains("type Target = BufferCreateInfo"));
    }

    #[test]
    fn builder_has_deref_mut() {
        let def = make_buffer_create_info();
        let code = emit_builder(&def, &std::collections::HashMap::new()).to_string();
        assert!(code.contains("DerefMut for BufferCreateInfoBuilder"));
    }

    #[test]
    fn returned_only_struct_has_no_push_next() {
        let def = StructDef {
            name: "PhysicalDeviceProperties".to_string(),
            members: vec![make_stype_member(), make_pnext_member()],
            extends: vec![],
            returned_only: true,
            is_union: false,
            provided_by: None,
        };
        let code = emit_builder(&def, &std::collections::HashMap::new()).to_string();
        assert!(
            !code.contains("fn push_next"),
            "returned_only structs should not have push_next"
        );
    }

    #[test]
    fn collect_count_fields_finds_pairs() {
        let def = make_buffer_create_info();
        let counts = collect_count_fields(&def);
        assert!(counts.contains("queueFamilyIndexCount"));
        assert_eq!(counts.len(), 1);
    }

    #[test]
    fn collect_count_fields_skips_null_terminated() {
        let def = StructDef {
            name: "Test".to_string(),
            members: vec![
                make_stype_member(),
                make_pnext_member(),
                make_pointer_member("pName", "char", Some("null-terminated")),
            ],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        };
        let counts = collect_count_fields(&def);
        assert!(counts.is_empty(), "null-terminated should not pair");
    }
}
