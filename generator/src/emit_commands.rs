//! Emits PFN typedefs and dispatch table structs for Vulkan commands.
//!
//! Generates `EntryCommands`, `InstanceCommands`, `DeviceCommands` — each holding
//! `Option<fn>` pointers loaded at runtime via `vkGet*ProcAddr`.

use std::collections::HashMap;

use heck::ToSnakeCase;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::parse::{CommandDef, DispatchLevel, ParamDef, VkRegistry};
use crate::resolve_types::{is_rust_keyword, resolve_param_type, resolve_return_type};

/// Emit the full commands module: PFN typedefs + dispatch structs + load fns.
pub fn emit_commands(registry: &VkRegistry) -> TokenStream {
    let alias_map = build_reverse_alias_map(registry);

    let pfn_typedefs = emit_pfn_typedefs(&registry.commands);
    let pfn_aliases = emit_pfn_aliases(&registry.aliases);

    let deduped = dedup_commands(&registry.commands);
    let entry_cmds: Vec<&CommandDef> = deduped
        .iter()
        .copied()
        .filter(|c| c.dispatch_level == DispatchLevel::Entry)
        .collect();
    let instance_cmds: Vec<&CommandDef> = deduped
        .iter()
        .copied()
        .filter(|c| c.dispatch_level == DispatchLevel::Instance)
        .collect();
    let device_cmds: Vec<&CommandDef> = deduped
        .iter()
        .copied()
        .filter(|c| c.dispatch_level == DispatchLevel::Device)
        .collect();

    let entry_struct = emit_dispatch_struct("EntryCommands", &entry_cmds, &alias_map);
    let instance_struct = emit_dispatch_struct("InstanceCommands", &instance_cmds, &alias_map);
    let device_struct = emit_dispatch_struct("DeviceCommands", &device_cmds, &alias_map);

    quote! {
        #![allow(clippy::manual_c_str_literals)]
        #![allow(clippy::missing_transmute_annotations)]
        #![allow(clippy::field_reassign_with_default)]

        use super::structs::*;
        use super::enums::*;
        use super::handles::*;
        use super::bitmasks::*;

        #pfn_typedefs
        #pfn_aliases

        #entry_struct
        #instance_struct
        #device_struct
    }
}

// ---------------------------------------------------------------------------
// PFN typedefs
// ---------------------------------------------------------------------------

fn emit_pfn_typedefs(commands: &[CommandDef]) -> TokenStream {
    let mut seen = std::collections::HashSet::new();
    let typedefs: Vec<TokenStream> = commands
        .iter()
        .filter(|c| seen.insert(c.name.clone()))
        .map(emit_pfn_typedef)
        .collect();
    quote! { #(#typedefs)* }
}

fn emit_pfn_typedef(cmd: &CommandDef) -> TokenStream {
    let pfn_name = format_ident!("PFN_{}", &cmd.name);
    let params = emit_pfn_params(&cmd.params);
    let ret = resolve_return_type(&cmd.return_type);
    let ret_tokens = ret.map(|ty| quote! { -> #ty }).unwrap_or_default();

    quote! {
        pub type #pfn_name = Option<unsafe extern "system" fn(#params) #ret_tokens>;
    }
}

fn emit_pfn_params(params: &[ParamDef]) -> TokenStream {
    let items: Vec<TokenStream> = params
        .iter()
        .map(|p| {
            let name = param_name(&p.name);
            let name_ident = if is_rust_keyword(&name) {
                format_ident!("r#{}", name)
            } else {
                format_ident!("{}", name)
            };
            let ty = resolve_param_type(p);
            quote! { #name_ident: #ty }
        })
        .collect();
    quote! { #(#items),* }
}

/// Emit `pub type PFN_vkFooKHR = PFN_vkFoo;` for aliased commands.
fn emit_pfn_aliases(aliases: &[crate::parse::AliasDef]) -> TokenStream {
    use crate::parse::AliasKind;
    let items: Vec<TokenStream> = aliases
        .iter()
        .filter(|a| a.kind == AliasKind::Command)
        .map(|a| {
            // Command alias names have the `vk` prefix (strip_vk only strips `Vk`).
            let alias_pfn = format_ident!("PFN_{}", &a.name);
            let canonical_pfn = format_ident!("PFN_{}", &a.target);
            quote! { pub type #alias_pfn = #canonical_pfn; }
        })
        .collect();
    quote! { #(#items)* }
}

// ---------------------------------------------------------------------------
// Dispatch structs
// ---------------------------------------------------------------------------

fn emit_dispatch_struct(
    name: &str,
    commands: &[&CommandDef],
    alias_map: &HashMap<String, Vec<String>>,
) -> TokenStream {
    let struct_name = format_ident!("{}", name);

    let fields: Vec<TokenStream> = commands
        .iter()
        .map(|cmd| {
            let field = format_ident!("{}", command_field_name(&cmd.name));
            let pfn = format_ident!("PFN_{}", &cmd.name);
            quote! { pub #field: #pfn }
        })
        .collect();

    let load_stmts: Vec<TokenStream> = commands
        .iter()
        .map(|cmd| emit_load_stmt(cmd, alias_map))
        .collect();

    quote! {
        pub struct #struct_name {
            #(#fields,)*
        }

        impl Default for #struct_name {
            #[inline]
            fn default() -> Self {
                unsafe { std::mem::zeroed() }
            }
        }

        impl #struct_name {
            /// Load all function pointers from the given loader callback.
            ///
            /// Load all function pointers from the given loader callback.
            ///
            /// # Safety
            ///
            /// The loader must return valid function pointers compatible with
            /// each command's signature, or null for unavailable commands.
            pub unsafe fn load(
                mut f: impl FnMut(&std::ffi::CStr) -> *const std::ffi::c_void,
            ) -> Self {
                unsafe {
                    let mut cmd = Self::default();
                    #(#load_stmts)*
                    cmd
                }
            }
        }
    }
}

fn cstr_literal(name: &str) -> TokenStream {
    let bytes = proc_macro2::Literal::byte_string(format!("{name}\0").as_bytes());
    quote! { std::ffi::CStr::from_bytes_with_nul_unchecked(#bytes) }
}

fn emit_load_stmt(cmd: &CommandDef, alias_map: &HashMap<String, Vec<String>>) -> TokenStream {
    let field = format_ident!("{}", command_field_name(&cmd.name));
    let c_name_expr = cstr_literal(&cmd.name);

    // Primary load: try the canonical name.
    let primary = quote! {
        cmd.#field = std::mem::transmute(f(#c_name_expr));
    };

    // Fallback loads: try alias names if the primary returned None.
    let aliases = alias_map.get(&cmd.name);
    let fallbacks: Vec<TokenStream> = aliases
        .into_iter()
        .flatten()
        .map(|alias| {
            let alias_expr = cstr_literal(alias);
            quote! {
                if cmd.#field.is_none() {
                    cmd.#field = std::mem::transmute(f(#alias_expr));
                }
            }
        })
        .collect();

    quote! {
        #primary
        #(#fallbacks)*
    }
}

// ---------------------------------------------------------------------------
// Name helpers
// ---------------------------------------------------------------------------

/// Convert a C command name to a snake_case field name: `vkCreateInstance` → `create_instance`.
fn command_field_name(c_name: &str) -> String {
    let stripped = c_name.strip_prefix("vk").unwrap_or(c_name);
    stripped.to_snake_case()
}

/// Convert a C parameter name to snake_case.
fn param_name(c_name: &str) -> String {
    c_name.to_snake_case()
}

/// Deduplicate commands that appear multiple times (vulkan vs vulkansc API variants).
fn dedup_commands(commands: &[CommandDef]) -> Vec<&CommandDef> {
    let mut seen = std::collections::HashSet::new();
    commands
        .iter()
        .filter(|c| seen.insert(c.name.as_str()))
        .collect()
}

/// Build a reverse alias map: canonical command name → list of alias names.
/// Filtered to `AliasKind::Command` so no heuristic needed.
fn build_reverse_alias_map(registry: &VkRegistry) -> HashMap<String, Vec<String>> {
    use crate::parse::AliasKind;
    let mut reverse: HashMap<String, Vec<String>> = HashMap::new();
    for a in &registry.aliases {
        if a.kind == AliasKind::Command {
            reverse
                .entry(a.target.clone())
                .or_default()
                .push(a.name.clone());
        }
    }
    reverse
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::ParamDef;

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

    fn make_command(name: &str, return_type: &str, params: Vec<ParamDef>) -> CommandDef {
        CommandDef {
            name: name.to_string(),
            return_type: return_type.to_string(),
            params,
            success_codes: vec![],
            error_codes: vec![],
            dispatch_level: DispatchLevel::Entry,
            provided_by: None,
        }
    }

    #[test]
    fn command_field_name_strips_vk_prefix() {
        assert_eq!(command_field_name("vkCreateInstance"), "create_instance");
        assert_eq!(command_field_name("vkCmdDraw"), "cmd_draw");
        assert_eq!(
            command_field_name("vkGetDeviceProcAddr"),
            "get_device_proc_addr"
        );
    }

    #[test]
    fn param_name_converts_to_snake_case() {
        assert_eq!(param_name("pCreateInfo"), "p_create_info");
        assert_eq!(param_name("instance"), "instance");
        assert_eq!(param_name("vertexCount"), "vertex_count");
    }

    #[test]
    fn pfn_typedef_has_correct_shape() {
        let cmd = make_command(
            "vkCreateInstance",
            "VkResult",
            vec![
                ParamDef {
                    is_pointer: true,
                    is_const: true,
                    ..make_param("pCreateInfo", "VkInstanceCreateInfo")
                },
                ParamDef {
                    is_pointer: true,
                    is_const: true,
                    ..make_param("pAllocator", "VkAllocationCallbacks")
                },
                ParamDef {
                    is_pointer: true,
                    ..make_param("pInstance", "VkInstance")
                },
            ],
        );
        let code = emit_pfn_typedef(&cmd).to_string();
        assert!(code.contains("PFN_vkCreateInstance"));
        assert!(code.contains("unsafe extern \"system\""));
        assert!(code.contains("p_create_info"));
        assert!(code.contains("* const InstanceCreateInfo"));
        assert!(code.contains("-> Result"));
    }

    #[test]
    fn pfn_typedef_void_return_has_no_arrow() {
        let cmd = make_command(
            "vkDestroyInstance",
            "void",
            vec![make_param("instance", "VkInstance")],
        );
        let code = emit_pfn_typedef(&cmd).to_string();
        assert!(code.contains("PFN_vkDestroyInstance"));
        assert!(!code.contains("->"), "void return should have no arrow");
    }

    #[test]
    fn dispatch_struct_has_fields_and_load() {
        let cmd = make_command(
            "vkCreateInstance",
            "VkResult",
            vec![make_param("pCreateInfo", "VkInstanceCreateInfo")],
        );
        let cmds = vec![&cmd];
        let alias_map = HashMap::new();
        let code = emit_dispatch_struct("EntryCommands", &cmds, &alias_map).to_string();
        assert!(code.contains("pub struct EntryCommands"));
        assert!(code.contains("create_instance"));
        assert!(code.contains("PFN_vkCreateInstance"));
        assert!(code.contains("fn load"));
        assert!(code.contains("mem :: zeroed"));
    }

    #[test]
    fn load_stmt_uses_original_c_name() {
        let cmd = make_command("vkCmdDraw", "void", vec![]);
        let alias_map = HashMap::new();
        let code = emit_load_stmt(&cmd, &alias_map).to_string();
        assert!(
            code.contains("vkCmdDraw"),
            "load should use original C name"
        );
    }

    #[test]
    fn load_stmt_has_alias_fallback() {
        let cmd = make_command("vkCreateRenderPass2", "VkResult", vec![]);
        let mut alias_map = HashMap::new();
        alias_map.insert(
            "vkCreateRenderPass2".to_string(),
            vec!["vkCreateRenderPass2KHR".to_string()],
        );
        let code = emit_load_stmt(&cmd, &alias_map).to_string();
        assert!(code.contains("vkCreateRenderPass2"));
        assert!(
            code.contains("vkCreateRenderPass2KHR"),
            "should try alias as fallback"
        );
        assert!(code.contains("is_none"), "fallback should check is_none");
    }

    #[test]
    fn pfn_alias_emits_type_alias() {
        use crate::parse::{AliasDef, AliasKind};
        let aliases = vec![AliasDef {
            name: "vkCreateRenderPass2KHR".to_string(),
            target: "vkCreateRenderPass2".to_string(),
            kind: AliasKind::Command,
        }];
        let code = emit_pfn_aliases(&aliases).to_string();
        assert!(code.contains("PFN_vkCreateRenderPass2KHR"));
        assert!(code.contains("PFN_vkCreateRenderPass2"));
    }

    #[test]
    fn pfn_alias_skips_non_command_aliases() {
        use crate::parse::{AliasDef, AliasKind};
        let aliases = vec![AliasDef {
            name: "ImageUsageFlags".to_string(),
            target: "ImageUsageFlagBits".to_string(),
            kind: AliasKind::Type,
        }];
        let code = emit_pfn_aliases(&aliases).to_string();
        assert!(code.is_empty(), "should not emit PFN alias for type aliases");
    }
}
