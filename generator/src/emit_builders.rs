//! Emits builder patterns for extensible Vulkan structs (those with sType/pNext).

use std::collections::HashSet;

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::emit_structs::{is_rust_keyword, member_name, resolve_member_type, struct_stype};
use crate::parse::{MemberDef, StructDef, VkRegistry};

/// Emit builders for all extensible structs.
pub fn emit_builders(registry: &VkRegistry) -> TokenStream {
    let builders: Vec<TokenStream> = registry
        .structs
        .iter()
        .filter(|s| has_stype_pnext(s))
        .map(emit_builder)
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

fn has_stype_pnext(def: &StructDef) -> bool {
    def.members.iter().any(|m| m.name == "sType") && def.members.iter().any(|m| m.name == "pNext")
}

fn emit_builder(def: &StructDef) -> TokenStream {
    let struct_name = format_ident!("{}", &def.name);
    let builder_name = format_ident!("{}Builder", &def.name);

    // Build the set of count fields that are paired with a pointer via `len`.
    let count_fields = collect_count_fields(def);

    let setters: Vec<TokenStream> = def
        .members
        .iter()
        .filter(|m| m.name != "sType" && m.name != "pNext" && !count_fields.contains(&m.name))
        .map(|m| emit_setter(m, def))
        .collect();

    let stype_val = struct_stype(def);
    let default_stype = stype_val.unwrap_or_else(|| quote! { Default::default() });

    // push_next: only if any struct extends this one.
    let extends_trait = format_ident!("Extends{}", &def.name);
    let push_next = if !def.returned_only {
        quote! {
            /// Prepend a struct to the pNext chain.
            #[inline]
            pub fn push_next<T: #extends_trait>(mut self, next: &'a mut T) -> Self {
                unsafe {
                    let next_ptr = <*mut T>::cast::<BaseOutStructure>(next);
                    (*next_ptr).p_next = self.inner.p_next as *mut _;
                    self.inner.p_next = <*mut BaseOutStructure>::cast::<std::ffi::c_void>(next_ptr);
                }
                self
            }
        }
    } else {
        quote! {}
    };

    quote! {
        pub struct #builder_name<'a> {
            inner: #struct_name,
            _marker: std::marker::PhantomData<&'a ()>,
        }

        impl #struct_name {
            /// Returns a builder for this struct with sType pre-filled.
            #[inline]
            pub fn builder<'a>() -> #builder_name<'a> {
                #builder_name {
                    inner: #struct_name {
                        s_type: #default_stype,
                        p_next: std::ptr::null_mut(),
                        ..Default::default()
                    },
                    _marker: std::marker::PhantomData,
                }
            }
        }

        impl<'a> #builder_name<'a> {
            #(#setters)*
            #push_next
        }

        impl<'a> std::ops::Deref for #builder_name<'a> {
            type Target = #struct_name;
            #[inline]
            fn deref(&self) -> &Self::Target { &self.inner }
        }

        impl<'a> std::ops::DerefMut for #builder_name<'a> {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target { &mut self.inner }
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
        return emit_slice_setter(member, count_member);
    }

    emit_simple_setter(member)
}

fn emit_simple_setter(member: &MemberDef) -> TokenStream {
    let rust_name = member_name(&member.name);
    let setter_ident = if is_rust_keyword(&rust_name) {
        format_ident!("r#{}", rust_name)
    } else {
        format_ident!("{}", rust_name)
    };
    let field_ident = setter_ident.clone();
    let ty = resolve_member_type(member);

    quote! {
        #[inline]
        pub fn #setter_ident(mut self, value: #ty) -> Self {
            self.inner.#field_ident = value;
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
    let elem_ty = crate::emit_structs::resolve_base_type(&ptr_member.type_name);

    // Count field type (usually u32).
    let count_ty = resolve_member_type(count_member);

    quote! {
        #[inline]
        pub fn #setter_ident(mut self, slice: &'a [#elem_ty]) -> Self {
            self.inner.#count_field = slice.len() as #count_ty;
            self.inner.#ptr_field = slice.as_ptr();
            self
        }
    }
}

// ---------------------------------------------------------------------------
// Pointer/count pairing
// ---------------------------------------------------------------------------

/// Collect the names of all count fields that are paired with a pointer via `len`.
fn collect_count_fields(def: &StructDef) -> HashSet<String> {
    let mut counts = HashSet::new();
    for m in &def.members {
        if m.is_pointer
            && !m.is_double_pointer
            && let Some(ref len) = m.len
        {
            // len can be a comma-separated list (e.g. "codeSize/4"); take the first simple name.
            let count_name = len.split(',').next().unwrap_or(len).trim();
            // Skip "null-terminated" and expressions with `/`.
            if !count_name.contains("null-terminated")
                && !count_name.contains('/')
                && def.members.iter().any(|other| other.name == count_name)
            {
                counts.insert(count_name.to_string());
            }
        }
    }
    counts
}

/// Find the count member referenced by a pointer's `len` attribute.
fn find_count_member<'a>(def: &'a StructDef, len: &str) -> Option<&'a MemberDef> {
    let count_name = len.split(',').next()?.trim();
    if count_name.contains("null-terminated") || count_name.contains('/') {
        return None;
    }
    def.members.iter().find(|m| m.name == count_name)
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
        let code = emit_builder(&def).to_string();
        assert!(code.contains("pub struct BufferCreateInfoBuilder"));
        assert!(code.contains("PhantomData"));
    }

    #[test]
    fn builder_has_builder_fn_on_struct() {
        let def = make_buffer_create_info();
        let code = emit_builder(&def).to_string();
        assert!(code.contains("impl BufferCreateInfo"));
        assert!(code.contains("fn builder"));
        assert!(code.contains("BUFFER_CREATE_INFO"));
    }

    #[test]
    fn builder_skips_stype_and_pnext_setters() {
        let def = make_buffer_create_info();
        let code = emit_builder(&def).to_string();
        // Should NOT have a setter named s_type or p_next.
        assert!(!code.contains("fn s_type"), "should skip sType setter");
        assert!(!code.contains("fn p_next"), "should skip pNext setter");
    }

    #[test]
    fn builder_has_simple_setters() {
        let def = make_buffer_create_info();
        let code = emit_builder(&def).to_string();
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
        let code = emit_builder(&def).to_string();
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
        let code = emit_builder(&def).to_string();
        assert!(
            !code.contains("fn queue_family_index_count"),
            "count field should not have its own setter"
        );
    }

    #[test]
    fn builder_has_push_next() {
        let def = make_buffer_create_info();
        let code = emit_builder(&def).to_string();
        assert!(code.contains("fn push_next"), "missing push_next");
        assert!(
            code.contains("ExtendsBufferCreateInfo"),
            "push_next should use extends trait"
        );
    }

    #[test]
    fn builder_has_deref() {
        let def = make_buffer_create_info();
        let code = emit_builder(&def).to_string();
        assert!(code.contains("impl < 'a > std :: ops :: Deref for BufferCreateInfoBuilder"));
        assert!(code.contains("type Target = BufferCreateInfo"));
    }

    #[test]
    fn builder_has_deref_mut() {
        let def = make_buffer_create_info();
        let code = emit_builder(&def).to_string();
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
        let code = emit_builder(&def).to_string();
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
