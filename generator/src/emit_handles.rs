//! Emits `#[repr(transparent)]` handle newtypes and the `Handle` trait.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::parse::{HandleDef, VkRegistry};

/// Parses a TokenStream as a Rust file, panicking with a clear message on failure.
#[cfg(test)]
fn assert_valid_rust(tokens: &TokenStream) {
    syn::parse2::<syn::File>(tokens.clone())
        .unwrap_or_else(|e| panic!("generated code is not valid Rust: {e}\n\n{tokens}"));
}

/// Emit all handle types plus the `Handle` trait definition.
pub fn emit_handles(registry: &VkRegistry) -> TokenStream {
    let trait_def = emit_handle_trait();
    let handles: Vec<TokenStream> = registry.handles.iter().map(emit_handle).collect();

    quote! {
        #trait_def
        #(#handles)*
    }
}

fn emit_handle_trait() -> TokenStream {
    quote! {
        /// Trait implemented by all Vulkan handle types.
        pub trait Handle: Copy + Eq + core::hash::Hash {
            /// The raw representation type (`usize` for dispatchable, `u64` for non-dispatchable).
            type Repr;

            /// Returns the null handle.
            fn null() -> Self;

            /// Constructs a handle from its raw representation.
            fn from_raw(raw: Self::Repr) -> Self;

            /// Returns the raw representation.
            fn as_raw(self) -> Self::Repr;

            /// Returns `true` if this is the null handle.
            fn is_null(self) -> bool;
        }
    }
}

fn emit_handle(handle: &HandleDef) -> TokenStream {
    let name = format_ident!("{}", &handle.name);
    let vk_name = format!("Vk{}", &handle.name);

    let (repr_type, zero_literal) = if handle.dispatchable {
        (quote! { usize }, quote! { 0usize })
    } else {
        (quote! { u64 }, quote! { 0u64 })
    };

    quote! {
        #[repr(transparent)]
        #[derive(Copy, Clone, PartialEq, Eq, Hash)]
        #[doc(alias = #vk_name)]
        pub struct #name(#repr_type);

        impl Handle for #name {
            type Repr = #repr_type;

            #[inline]
            fn null() -> Self { Self(#zero_literal) }

            #[inline]
            fn from_raw(raw: #repr_type) -> Self { Self(raw) }

            #[inline]
            fn as_raw(self) -> #repr_type { self.0 }

            #[inline]
            fn is_null(self) -> bool { self.0 == #zero_literal }
        }

        impl Default for #name {
            #[inline]
            fn default() -> Self { Self::null() }
        }

        impl core::fmt::Debug for #name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "{}({:#x})", stringify!(#name), self.0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_handle(name: &str, dispatchable: bool) -> HandleDef {
        HandleDef {
            name: name.to_string(),
            dispatchable,
            parent: None,
            object_type: None,
            provided_by: None,
        }
    }

    #[test]
    fn dispatchable_handle_uses_usize() {
        let tokens = emit_handle(&make_handle("Instance", true));
        assert_valid_rust(&tokens);
        let code = tokens.to_string();
        assert!(
            code.contains("pub struct Instance (usize)"),
            "expected usize repr"
        );
        assert!(code.contains("type Repr = usize"), "expected Repr = usize");
    }

    #[test]
    fn non_dispatchable_handle_uses_u64() {
        let tokens = emit_handle(&make_handle("Buffer", false));
        assert_valid_rust(&tokens);
        let code = tokens.to_string();
        assert!(
            code.contains("pub struct Buffer (u64)"),
            "expected u64 repr"
        );
        assert!(code.contains("type Repr = u64"), "expected Repr = u64");
    }

    #[test]
    fn handle_has_doc_alias() {
        let tokens = emit_handle(&make_handle("Fence", false));
        let code = tokens.to_string();
        assert!(code.contains("VkFence"), "expected doc(alias) with VkFence");
    }

    #[test]
    fn handle_trait_emitted() {
        let tokens = emit_handle_trait();
        assert_valid_rust(&tokens);
        let code = tokens.to_string();
        assert!(code.contains("pub trait Handle"));
        assert!(code.contains("fn null"));
        assert!(code.contains("fn from_raw"));
        assert!(code.contains("fn as_raw"));
        assert!(code.contains("fn is_null"));
    }

    #[test]
    fn handle_has_default_and_debug() {
        let tokens = emit_handle(&make_handle("Semaphore", false));
        let code = tokens.to_string();
        assert!(code.contains("impl Default for Semaphore"));
        assert!(code.contains("impl core :: fmt :: Debug for Semaphore"));
    }
}
