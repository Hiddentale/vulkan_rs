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
    use crate::parse::AliasKind;
    let aliases: Vec<TokenStream> = registry
        .aliases
        .iter()
        .filter(|a| a.kind == AliasKind::Type)
        .filter_map(|a| {
            let clean_name = a.name.strip_prefix("Vk").unwrap_or(&a.name);
            let clean_target = a.target.strip_prefix("Vk").unwrap_or(&a.target);
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

/// Emit opaque stubs for StdVideo* types referenced by struct members.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::*;
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
            aliases: vec![],
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
    fn flags_alias_from_bitmask() {
        let mut reg = empty_registry();
        reg.bitmasks.push(BitmaskDef {
            name: "BufferCreateFlagBits".to_string(),
            flags_name: "BufferCreateFlags".to_string(),
            bitwidth: 32,
            bits: vec![],
        });
        let code = emit_flags_aliases(&reg).to_string();
        assert!(code.contains("BufferCreateFlags"));
        assert!(code.contains("BufferCreateFlagBits"));
    }

    #[test]
    fn flags_alias_skips_struct_name_conflict() {
        let mut reg = empty_registry();
        reg.bitmasks.push(BitmaskDef {
            name: "FooFlagBits".to_string(),
            flags_name: "FooFlags".to_string(),
            bitwidth: 32,
            bits: vec![],
        });
        reg.structs.push(StructDef {
            name: "FooFlags".to_string(),
            members: vec![],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        });
        let code = emit_flags_aliases(&reg).to_string();
        assert!(
            !code.contains("FooFlags"),
            "should skip alias when struct with same name exists"
        );
    }

    #[test]
    fn flags_alias_falls_back_to_u32_when_no_bitmask() {
        let mut reg = empty_registry();
        reg.structs.push(StructDef {
            name: "Test".to_string(),
            members: vec![make_member("flags", "VkDescriptorPoolResetFlags")],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        });
        let code = emit_flags_aliases(&reg).to_string();
        assert!(code.contains("DescriptorPoolResetFlags"));
        assert!(code.contains("u32"), "should fall back to u32");
    }

    #[test]
    fn flags_alias_scans_command_params() {
        let mut reg = empty_registry();
        reg.commands.push(CommandDef {
            name: "vkTrimCommandPool".to_string(),
            return_type: "void".to_string(),
            params: vec![make_param("flags", "VkCommandPoolTrimFlags")],
            success_codes: vec![],
            error_codes: vec![],
            dispatch_level: DispatchLevel::Device,
            provided_by: None,
        });
        let code = emit_flags_aliases(&reg).to_string();
        assert!(
            code.contains("CommandPoolTrimFlags"),
            "should emit alias for flags type only referenced by command param"
        );
    }

    #[test]
    fn type_alias_skips_self_referential() {
        let mut reg = empty_registry();
        reg.aliases.push(AliasDef {
            name: "Foo".to_string(),
            target: "Foo".to_string(),
            kind: AliasKind::Type,
        });
        let code = emit_type_aliases(&reg).to_string();
        assert!(code.is_empty());
    }

    #[test]
    fn type_alias_skips_flags() {
        let mut reg = empty_registry();
        reg.aliases.push(AliasDef {
            name: "ImageUsageFlags".to_string(),
            target: "ImageUsageFlagBits".to_string(),
            kind: AliasKind::Type,
        });
        let code = emit_type_aliases(&reg).to_string();
        assert!(!code.contains("ImageUsageFlags"));
    }

    #[test]
    fn type_alias_emits_promoted_type() {
        let mut reg = empty_registry();
        reg.aliases.push(AliasDef {
            name: "RenderPassCreateInfo2KHR".to_string(),
            target: "RenderPassCreateInfo2".to_string(),
            kind: AliasKind::Type,
        });
        let code = emit_type_aliases(&reg).to_string();
        assert!(code.contains("RenderPassCreateInfo2KHR"));
        assert!(code.contains("RenderPassCreateInfo2"));
    }

    #[test]
    fn stdvideo_stub_emits_opaque_struct() {
        let mut reg = empty_registry();
        reg.structs.push(StructDef {
            name: "VideoDecodeInfoKHR".to_string(),
            members: vec![make_member("pSlot", "StdVideoDecodeH264PictureInfo")],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        });
        let code = emit_stdvideo_stubs(&reg).to_string();
        assert!(code.contains("StdVideoDecodeH264PictureInfo"));
        assert!(code.contains("repr (C)"));
        assert!(code.contains("_opaque"));
    }

    #[test]
    fn func_pointer_stub_emits_option_fn() {
        let mut reg = empty_registry();
        reg.func_pointers.push(FuncPointerDef {
            name: "PFN_vkAllocationFunction".to_string(),
            return_type: "void*".to_string(),
            is_return_pointer: true,
            params: vec![],
        });
        let code = emit_func_pointer_stubs(&reg).to_string();
        assert!(code.contains("PFN_vkAllocationFunction"));
        assert!(code.contains("Option"));
        assert!(code.contains("extern \"system\""));
    }
}
