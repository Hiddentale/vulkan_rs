//! Emits type aliases, Flags→FlagBits aliases, PFN stubs, and StdVideo stubs.
//!
//! These fill gaps between what vk.xml references and what our emitters generate.

use std::collections::BTreeSet;

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::parse::VkRegistry;
use crate::resolve_types::resolve_flags_alias;

/// Emit `pub type FooFlags = FooFlagBits;` aliases for all bitmask types.
pub fn emit_flags_aliases(registry: &VkRegistry) -> TokenStream {
    let existing_bitmasks: BTreeSet<String> =
        registry.bitmasks.iter().map(|b| b.name.clone()).collect();
    let struct_names: BTreeSet<String> = registry.structs.iter().map(|s| s.name.clone()).collect();

    let mut emitted = BTreeSet::new();
    let mut aliases = Vec::new();

    for bm in &registry.bitmasks {
        if bm.flags_name != bm.name
            && !struct_names.contains(&bm.flags_name)
            && emitted.insert(bm.flags_name.clone())
        {
            let flags = format_ident!("{}", &bm.flags_name);
            let bits = format_ident!("{}", &bm.name);
            aliases.push(quote! { pub type #flags = #bits; });
        }
    }

    // Collect all type names referenced by struct members and command params.
    let referenced_types = registry
        .structs
        .iter()
        .flat_map(|s| s.members.iter().map(|m| m.type_name.as_str()))
        .chain(
            registry
                .commands
                .iter()
                .flat_map(|c| c.params.iter().map(|p| p.type_name.as_str())),
        );

    for type_name in referenced_types {
        let stripped = type_name.strip_prefix("Vk").unwrap_or(type_name);
        if stripped.contains("Flags")
            && !stripped.contains("FlagBits")
            && !struct_names.contains(stripped)
            && emitted.insert(stripped.to_string())
        {
            let flag_bits = resolve_flags_alias(stripped);
            let flags_ident = format_ident!("{}", stripped);
            if existing_bitmasks.contains(&flag_bits) || emitted.contains(&flag_bits) {
                let bits_ident = format_ident!("{}", flag_bits);
                aliases.push(quote! { pub type #flags_ident = #bits_ident; });
            } else {
                let is_64 = stripped.contains("Flags2") || stripped.contains("Flags3");
                if is_64 {
                    aliases.push(quote! { pub type #flags_ident = u64; });
                } else {
                    aliases.push(quote! { pub type #flags_ident = u32; });
                }
            }
        }
    }

    quote! { #(#aliases)* }
}

/// Emit `pub type Name = Target;` for Vk type aliases (e.g. promoted extension types).
pub fn emit_type_aliases(registry: &VkRegistry) -> TokenStream {
    let aliases: Vec<TokenStream> = registry
        .aliases
        .iter()
        .filter_map(|(name, target)| {
            if name == target {
                return None;
            }
            if name.starts_with(|c: char| c.is_ascii_lowercase()) || name.contains("vk") {
                return None;
            }

            let clean_name = name.strip_prefix("Vk").unwrap_or(name);
            let clean_target = target.strip_prefix("Vk").unwrap_or(target);
            if clean_name == clean_target {
                return None;
            }
            if clean_name.contains("Flags") || clean_target.contains("Flags") {
                return None;
            }
            let name_ident = format_ident!("{}", clean_name);
            let target_ident = format_ident!("{}", clean_target);
            Some(quote! {
                pub type #name_ident = #target_ident;
            })
        })
        .collect();
    quote! { #(#aliases)* }
}

/// Emit opaque stubs for PFN_vk* function pointer types.
pub fn emit_func_pointer_stubs(registry: &VkRegistry) -> TokenStream {
    let stubs: Vec<TokenStream> = registry
        .func_pointers
        .iter()
        .map(|fp| {
            let ident = format_ident!("{}", &fp.name);
            quote! {
                pub type #ident = Option<unsafe extern "system" fn()>;
            }
        })
        .collect();
    quote! { #(#stubs)* }
}

/// Emit opaque stubs for StdVideo* types from Vulkan video codec headers.
pub fn emit_stdvideo_stubs(registry: &VkRegistry) -> TokenStream {
    let mut names: BTreeSet<String> = BTreeSet::new();
    for s in &registry.structs {
        for m in &s.members {
            if m.type_name.starts_with("StdVideo") {
                names.insert(m.type_name.clone());
            }
        }
    }

    let stubs: Vec<TokenStream> = names
        .iter()
        .map(|name| {
            let ident = format_ident!("{}", name);
            quote! {
                /// Opaque video codec type (defined in vulkan_video_codec headers).
                #[repr(C)]
                #[derive(Debug, Copy, Clone, Default)]
                pub struct #ident {
                    _opaque: [u8; 0],
                }
            }
        })
        .collect();
    quote! { #(#stubs)* }
}
