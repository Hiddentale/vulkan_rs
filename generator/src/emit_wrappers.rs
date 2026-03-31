//! Emits ergonomic wrapper methods on `Entry`, `Instance`, and `Device`.
//!
//! Generates `impl` blocks with one method per Vulkan command, applying
//! self-handle elision, output-param returns, slice inputs, and optional
//! allocator transforms.

use std::collections::HashSet;

use heck::ToSnakeCase;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::parse::{CommandDef, DispatchLevel, ParamDef, VkRegistry};
use crate::resolve_types::{is_rust_keyword, resolve_base_type, resolve_param_type};
use crate::wrapper_utils::{
    CommandPattern, ParamRole, build_pnext_struct_set, classify_command, classify_params,
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Emit three complete source files (entry, instance, device) containing
/// `impl` blocks with generated wrapper methods.
pub fn emit_wrappers(registry: &VkRegistry) -> (TokenStream, TokenStream, TokenStream) {
    let pnext = build_pnext_struct_set(registry);
    let exclusions = exclusion_set();

    let (entry_methods, ec) = emit_methods(
        DispatchLevel::Entry,
        &registry.commands,
        &pnext,
        &exclusions,
    );
    let (instance_methods, ic) = emit_methods(
        DispatchLevel::Instance,
        &registry.commands,
        &pnext,
        &exclusions,
    );
    let (device_methods, dc) = emit_methods(
        DispatchLevel::Device,
        &registry.commands,
        &pnext,
        &exclusions,
    );

    println!("\n=== wrappers ===");
    println!("  entry methods:    {ec}");
    println!("  instance methods: {ic}");
    println!("  device methods:   {dc}");

    (
        emit_file("Entry", entry_methods),
        emit_file("Instance", instance_methods),
        emit_file("Device", device_methods),
    )
}

/// Count how many wrapper methods would be generated per dispatch level.
/// Used by tests to catch regressions where commands silently get dropped.
#[cfg(test)]
fn wrapper_counts(registry: &VkRegistry) -> (usize, usize, usize) {
    let exclusions = exclusion_set();

    let count = |level: DispatchLevel| -> usize {
        registry
            .commands
            .iter()
            .filter(|c| c.dispatch_level == level && !exclusions.contains(&c.name))
            .count()
    };

    (
        count(DispatchLevel::Entry),
        count(DispatchLevel::Instance),
        count(DispatchLevel::Device),
    )
}

// ---------------------------------------------------------------------------
// Exclusion set
// ---------------------------------------------------------------------------

/// Commands with hand-written wrappers in vk-engine that the generator must
/// not overwrite.
fn exclusion_set() -> HashSet<String> {
    [
        // Entry — special construction / fallback logic
        "vkCreateInstance",
        "vkEnumerateInstanceVersion",
        "vkEnumerateInstanceLayerProperties",
        "vkEnumerateInstanceExtensionProperties",
        // Instance — returns wrapped Device, not raw handle
        "vkCreateDevice",
        // Instance — feature-gated surface methods in surface.rs
        "vkCreateWin32SurfaceKHR",
        "vkCreateXlibSurfaceKHR",
        "vkCreateXcbSurfaceKHR",
        "vkCreateWaylandSurfaceKHR",
        "vkCreateMetalSurfaceEXT",
        "vkCreateAndroidSurfaceKHR",
        "vkDestroySurfaceKHR",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

// ---------------------------------------------------------------------------
// File-level wrapper
// ---------------------------------------------------------------------------

fn emit_file(wrapper_type: &str, methods: TokenStream) -> TokenStream {
    let wrapper = format_ident!("{}", wrapper_type);
    quote! {
        #![allow(unused_imports)]
        #![allow(clippy::too_many_arguments)]
        // Safety docs are now generated from vk.xml metadata.

        use crate::error::{check, enumerate_two_call, fill_two_call, VkResult};
        use crate::vk::bitmasks::*;
        use crate::vk::constants::*;
        use crate::vk::enums::*;
        use crate::vk::handles::*;
        use crate::vk::structs::*;

        impl crate::#wrapper {
            #methods
        }
    }
}

// ---------------------------------------------------------------------------
// Method collection
// ---------------------------------------------------------------------------

fn emit_methods(
    level: DispatchLevel,
    commands: &[CommandDef],
    pnext: &HashSet<String>,
    exclusions: &HashSet<String>,
) -> (TokenStream, usize) {
    let mut methods = TokenStream::new();
    let mut count = 0;

    for cmd in commands {
        if cmd.dispatch_level != level {
            continue;
        }
        if exclusions.contains(&cmd.name) {
            continue;
        }

        let roles = classify_params(cmd, pnext);
        let pattern = classify_command(cmd, &roles);
        methods.extend(emit_method(cmd, &roles, pattern));
        count += 1;
    }

    (methods, count)
}

// ---------------------------------------------------------------------------
// Single method emission
// ---------------------------------------------------------------------------

fn emit_method(cmd: &CommandDef, roles: &[ParamRole], pattern: CommandPattern) -> TokenStream {
    let method_name = format_ident!("{}", command_field_name(&cmd.name));
    let sig_params = emit_signature_params(cmd, roles);
    let ret = emit_return_type(cmd, roles, pattern);
    let body = emit_body(cmd, roles, pattern);
    let docs = emit_wrapper_docs(cmd, roles);

    quote! {
        #docs
        pub unsafe fn #method_name(&self, #sig_params) #ret {
            #body
        }
    }
}

/// Build doc comment lines for a wrapper method.
fn emit_wrapper_docs(cmd: &CommandDef, roles: &[ParamRole]) -> TokenStream {
    let spec_link = format!(
        "Wraps [`{name}`](https://registry.khronos.org/vulkan/specs/latest/man/html/{name}.html).",
        name = &cmd.name,
    );

    let mut doc_lines: Vec<TokenStream> = vec![quote! { #[doc = #spec_link] }];

    // Provided by.
    if let Some(ref provider) = cmd.provided_by {
        let line = format!("\nProvided by **{provider}**.");
        doc_lines.push(quote! { #[doc = #line] });
    }

    // Error codes → # Errors section.
    if !cmd.error_codes.is_empty() {
        doc_lines.push(quote! { #[doc = ""] });
        doc_lines.push(quote! { #[doc = "# Errors"] });
        for code in &cmd.error_codes {
            let line = format!("- `{code}`");
            doc_lines.push(quote! { #[doc = #line] });
        }
    }

    // Safety section.
    let dispatch_handle = cmd.params.first().map(|p| p.name.as_str()).unwrap_or("");
    let handle_safety = format!("- `{dispatch_handle}` (self) must be valid and not destroyed.");
    let sync_lines: Vec<String> = cmd
        .params
        .iter()
        .zip(roles.iter())
        .filter(|(p, _)| p.extern_sync.is_some())
        .map(|(p, _)| format!("- `{}` must be externally synchronized.", p.name))
        .collect();

    doc_lines.push(quote! { #[doc = ""] });
    doc_lines.push(quote! { #[doc = "# Safety"] });
    doc_lines.push(quote! { #[doc = #handle_safety] });
    for line in &sync_lines {
        doc_lines.push(quote! { #[doc = #line] });
    }

    quote! { #(#doc_lines)* }
}

// ---------------------------------------------------------------------------
// Signature parameters
// ---------------------------------------------------------------------------

fn emit_signature_params(cmd: &CommandDef, roles: &[ParamRole]) -> TokenStream {
    let params: Vec<TokenStream> = cmd
        .params
        .iter()
        .zip(roles.iter())
        .filter_map(|(param, role)| emit_signature_param(param, role))
        .collect();
    quote! { #(#params),* }
}

fn emit_signature_param(param: &ParamDef, role: &ParamRole) -> Option<TokenStream> {
    match role {
        ParamRole::SelfHandle
        | ParamRole::Output
        | ParamRole::OutputCount { .. }
        | ParamRole::OutputArray { .. }
        | ParamRole::InputCount { .. } => None,

        ParamRole::Allocator => Some(quote! {
            allocator: Option<&AllocationCallbacks>
        }),

        ParamRole::InputArray { .. } => {
            let name = param_ident(&param.name);
            let elem = resolve_base_type(&param.type_name);
            Some(quote! { #name: &[#elem] })
        }

        ParamRole::Regular => {
            let name = param_ident(&param.name);
            let ty = wrapper_param_type(param);
            Some(quote! { #name: #ty })
        }
    }
}

/// Resolve a Regular parameter's type for the wrapper signature.
///
/// `*const VkFoo` → `&Foo` (or `Option<&Foo>` when optional).
/// Everything else passes through as the raw resolved C type.
fn wrapper_param_type(param: &ParamDef) -> TokenStream {
    if param.is_pointer
        && param.is_const
        && !param.is_double_pointer
        && is_vk_type(&param.type_name)
    {
        let base = resolve_base_type(&param.type_name);
        if param.optional {
            quote! { Option<&#base> }
        } else {
            quote! { &#base }
        }
    } else {
        resolve_param_type(param)
    }
}

fn is_vk_type(type_name: &str) -> bool {
    type_name.starts_with("Vk")
}

// ---------------------------------------------------------------------------
// Return type
// ---------------------------------------------------------------------------

fn emit_return_type(cmd: &CommandDef, roles: &[ParamRole], pattern: CommandPattern) -> TokenStream {
    match pattern {
        CommandPattern::Create => {
            let ty = output_base_type(cmd, roles);
            quote! { -> VkResult<#ty> }
        }
        CommandPattern::Enumerate => {
            let ty = output_array_elem_type(cmd, roles);
            quote! { -> VkResult<Vec<#ty>> }
        }
        CommandPattern::Fill => {
            let ty = output_array_elem_type(cmd, roles);
            quote! { -> Vec<#ty> }
        }
        CommandPattern::Query => {
            let ty = output_base_type(cmd, roles);
            quote! { -> #ty }
        }
        CommandPattern::ResultOnly => quote! { -> VkResult<()> },
        CommandPattern::Destroy | CommandPattern::VoidForward => TokenStream::new(),
    }
}

fn output_base_type(cmd: &CommandDef, roles: &[ParamRole]) -> TokenStream {
    cmd.params
        .iter()
        .zip(roles.iter())
        .find_map(|(p, r)| matches!(r, ParamRole::Output).then(|| resolve_base_type(&p.type_name)))
        .expect("Create/Query pattern must have an Output role")
}

fn output_array_elem_type(cmd: &CommandDef, roles: &[ParamRole]) -> TokenStream {
    cmd.params
        .iter()
        .zip(roles.iter())
        .find_map(|(p, r)| {
            matches!(r, ParamRole::OutputArray { .. }).then(|| resolve_base_type(&p.type_name))
        })
        .expect("Enumerate/Fill pattern must have an OutputArray role")
}

// ---------------------------------------------------------------------------
// Method body
// ---------------------------------------------------------------------------

fn emit_body(cmd: &CommandDef, roles: &[ParamRole], pattern: CommandPattern) -> TokenStream {
    let field = format_ident!("{}", command_field_name(&cmd.name));
    let expect_msg = format!("{} not loaded", cmd.name);
    let fp_load = quote! {
        let fp = self.commands().#field.expect(#expect_msg);
    };
    let bindings = emit_bindings(cmd, roles);

    match pattern {
        CommandPattern::Create => {
            let args = emit_call_args(cmd, roles);
            quote! {
                #fp_load
                #bindings
                let mut out = unsafe { core::mem::zeroed() };
                check(unsafe { fp(#args) })?;
                Ok(out)
            }
        }
        CommandPattern::Destroy | CommandPattern::VoidForward => {
            let args = emit_call_args(cmd, roles);
            quote! {
                #fp_load
                #bindings
                unsafe { fp(#args) };
            }
        }
        CommandPattern::ResultOnly => {
            let args = emit_call_args(cmd, roles);
            quote! {
                #fp_load
                #bindings
                check(unsafe { fp(#args) })
            }
        }
        CommandPattern::Query => {
            let args = emit_call_args(cmd, roles);
            quote! {
                #fp_load
                #bindings
                let mut out = unsafe { core::mem::zeroed() };
                unsafe { fp(#args) };
                out
            }
        }
        CommandPattern::Enumerate => {
            let args = emit_two_call_args(cmd, roles);
            quote! {
                #fp_load
                #bindings
                enumerate_two_call(|count, data| unsafe { fp(#args) })
            }
        }
        CommandPattern::Fill => {
            let args = emit_two_call_args(cmd, roles);
            quote! {
                #fp_load
                #bindings
                fill_two_call(|count, data| unsafe { fp(#args) })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Let bindings for transformed parameters
// ---------------------------------------------------------------------------

fn emit_bindings(cmd: &CommandDef, roles: &[ParamRole]) -> TokenStream {
    let mut bindings = TokenStream::new();

    for (param, role) in cmd.params.iter().zip(roles.iter()) {
        match role {
            ParamRole::Allocator => {
                bindings.extend(quote! {
                    let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
                });
            }
            ParamRole::Regular if is_optional_vk_const_ptr(param) => {
                let ptr_name = format_ident!("{}_ptr", param.name.to_snake_case());
                let name = param_ident(&param.name);
                bindings.extend(quote! {
                    let #ptr_name = #name.map_or(core::ptr::null(), core::ptr::from_ref);
                });
            }
            _ => {}
        }
    }

    bindings
}

fn is_optional_vk_const_ptr(param: &ParamDef) -> bool {
    param.optional
        && param.is_pointer
        && param.is_const
        && !param.is_double_pointer
        && is_vk_type(&param.type_name)
}

// ---------------------------------------------------------------------------
// Call arguments
// ---------------------------------------------------------------------------

/// Build call args for direct patterns (Create, Destroy, Query, ResultOnly,
/// VoidForward). Output params become `&mut out`.
fn emit_call_args(cmd: &CommandDef, roles: &[ParamRole]) -> TokenStream {
    let args: Vec<TokenStream> = (0..cmd.params.len())
        .map(|i| emit_call_arg(i, &cmd.params, roles, false))
        .collect();
    quote! { #(#args),* }
}

/// Build call args for two-call patterns (Enumerate, Fill).
/// OutputCount → `count`, OutputArray → `data` (closure params).
fn emit_two_call_args(cmd: &CommandDef, roles: &[ParamRole]) -> TokenStream {
    let args: Vec<TokenStream> = (0..cmd.params.len())
        .map(|i| emit_call_arg(i, &cmd.params, roles, true))
        .collect();
    quote! { #(#args),* }
}

fn emit_call_arg(
    idx: usize,
    params: &[ParamDef],
    roles: &[ParamRole],
    two_call: bool,
) -> TokenStream {
    let param = &params[idx];
    let role = &roles[idx];

    match role {
        ParamRole::SelfHandle => quote! { self.handle() },

        ParamRole::Output => quote! { &mut out },

        ParamRole::OutputCount { .. } => {
            debug_assert!(two_call, "OutputCount in non-two-call context");
            quote! { count }
        }

        ParamRole::OutputArray { .. } => {
            debug_assert!(two_call, "OutputArray in non-two-call context");
            quote! { data }
        }

        ParamRole::InputCount { partner } => {
            let partner_name = param_ident(&params[*partner].name);
            quote! { #partner_name.len() as u32 }
        }

        ParamRole::InputArray { .. } => {
            let name = param_ident(&param.name);
            quote! { #name.as_ptr() }
        }

        ParamRole::Allocator => quote! { alloc_ptr },

        ParamRole::Regular => {
            if is_optional_vk_const_ptr(param) {
                let ptr_name = format_ident!("{}_ptr", param.name.to_snake_case());
                quote! { #ptr_name }
            } else {
                let name = param_ident(&param.name);
                quote! { #name }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Name helpers
// ---------------------------------------------------------------------------

/// `vkCreateBuffer` → `create_buffer`
fn command_field_name(c_name: &str) -> String {
    c_name.strip_prefix("vk").unwrap_or(c_name).to_snake_case()
}

/// Build an identifier for a parameter, escaping Rust keywords.
fn param_ident(c_name: &str) -> proc_macro2::Ident {
    let snake = c_name.to_snake_case();
    if is_rust_keyword(&snake) {
        format_ident!("r#{}", snake)
    } else {
        format_ident!("{}", snake)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::DispatchLevel;

    // -- Test helpers -------------------------------------------------------

    fn param(name: &str, type_name: &str) -> ParamDef {
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

    fn const_ptr(name: &str, type_name: &str) -> ParamDef {
        ParamDef {
            is_pointer: true,
            is_const: true,
            ..param(name, type_name)
        }
    }

    fn mut_ptr(name: &str, type_name: &str) -> ParamDef {
        ParamDef {
            is_pointer: true,
            ..param(name, type_name)
        }
    }

    fn optional_const_ptr(name: &str, type_name: &str) -> ParamDef {
        ParamDef {
            optional: true,
            ..const_ptr(name, type_name)
        }
    }

    fn mut_ptr_with_len(name: &str, type_name: &str, len: &str) -> ParamDef {
        ParamDef {
            len: Some(len.to_string()),
            ..mut_ptr(name, type_name)
        }
    }

    fn const_ptr_with_len(name: &str, type_name: &str, len: &str) -> ParamDef {
        ParamDef {
            len: Some(len.to_string()),
            ..const_ptr(name, type_name)
        }
    }

    fn double_mut_ptr(name: &str, type_name: &str) -> ParamDef {
        ParamDef {
            is_double_pointer: true,
            ..mut_ptr(name, type_name)
        }
    }

    fn cmd(
        name: &str,
        return_type: &str,
        params: Vec<ParamDef>,
        level: DispatchLevel,
    ) -> CommandDef {
        CommandDef {
            name: name.to_string(),
            return_type: return_type.to_string(),
            params,
            success_codes: vec![],
            error_codes: vec![],
            dispatch_level: level,
            provided_by: None,
        }
    }

    fn empty_pnext() -> HashSet<String> {
        HashSet::new()
    }

    fn stringify(tokens: TokenStream) -> String {
        tokens.to_string()
    }

    // -- Signature tests ----------------------------------------------------

    #[test]
    fn signature_create_buffer() {
        let c = cmd(
            "vkCreateBuffer",
            "VkResult",
            vec![
                param("device", "VkDevice"),
                const_ptr("pCreateInfo", "VkBufferCreateInfo"),
                optional_const_ptr("pAllocator", "VkAllocationCallbacks"),
                mut_ptr("pBuffer", "VkBuffer"),
            ],
            DispatchLevel::Device,
        );
        let roles = classify_params(&c, &empty_pnext());
        let sig = stringify(emit_signature_params(&c, &roles));

        assert!(sig.contains("create_info : & BufferCreateInfo"));
        assert!(sig.contains("allocator : Option < & AllocationCallbacks >"));
        // Output param (pBuffer) must NOT appear
        assert!(!sig.contains("buffer"));
        // Self-handle (device) must NOT appear
        assert!(!sig.contains("device"));
    }

    #[test]
    fn signature_cmd_draw_all_regular() {
        let c = cmd(
            "vkCmdDraw",
            "void",
            vec![
                param("commandBuffer", "VkCommandBuffer"),
                param("vertexCount", "uint32_t"),
                param("instanceCount", "uint32_t"),
                param("firstVertex", "uint32_t"),
                param("firstInstance", "uint32_t"),
            ],
            DispatchLevel::Device,
        );
        let roles = classify_params(&c, &empty_pnext());
        let sig = stringify(emit_signature_params(&c, &roles));

        // VkCommandBuffer is NOT the device self-handle, so it appears
        assert!(sig.contains("command_buffer"));
        assert!(sig.contains("vertex_count"));
        assert!(sig.contains("instance_count"));
    }

    #[test]
    fn signature_enumerate_elides_output_pair() {
        let c = cmd(
            "vkEnumeratePhysicalDevices",
            "VkResult",
            vec![
                param("instance", "VkInstance"),
                mut_ptr("pPhysicalDeviceCount", "uint32_t"),
                mut_ptr_with_len(
                    "pPhysicalDevices",
                    "VkPhysicalDevice",
                    "pPhysicalDeviceCount",
                ),
            ],
            DispatchLevel::Instance,
        );
        let roles = classify_params(&c, &empty_pnext());
        let sig = stringify(emit_signature_params(&c, &roles));

        // Self-handle and output pair all elided — no params
        assert!(sig.is_empty() || sig.trim().is_empty());
    }

    #[test]
    fn signature_input_array_becomes_slice() {
        let c = cmd(
            "vkQueueSubmit",
            "VkResult",
            vec![
                param("queue", "VkQueue"),
                param("submitCount", "uint32_t"),
                const_ptr_with_len("pSubmits", "VkSubmitInfo", "submitCount"),
                param("fence", "VkFence"),
            ],
            DispatchLevel::Device,
        );
        let roles = classify_params(&c, &empty_pnext());
        let sig = stringify(emit_signature_params(&c, &roles));

        // Count param suppressed, array becomes slice
        assert!(!sig.contains("submit_count"));
        assert!(sig.contains("submits : & [SubmitInfo]"));
        assert!(sig.contains("queue"));
        assert!(sig.contains("fence"));
    }

    // -- Return type tests --------------------------------------------------

    #[test]
    fn return_type_create() {
        let c = cmd(
            "vkCreateBuffer",
            "VkResult",
            vec![
                param("device", "VkDevice"),
                const_ptr("pCreateInfo", "VkBufferCreateInfo"),
                optional_const_ptr("pAllocator", "VkAllocationCallbacks"),
                mut_ptr("pBuffer", "VkBuffer"),
            ],
            DispatchLevel::Device,
        );
        let roles = classify_params(&c, &empty_pnext());
        let pattern = classify_command(&c, &roles);
        let ret = stringify(emit_return_type(&c, &roles, pattern));

        assert!(ret.contains("VkResult < Buffer >"));
    }

    #[test]
    fn return_type_enumerate() {
        let c = cmd(
            "vkEnumeratePhysicalDevices",
            "VkResult",
            vec![
                param("instance", "VkInstance"),
                mut_ptr("pPhysicalDeviceCount", "uint32_t"),
                mut_ptr_with_len(
                    "pPhysicalDevices",
                    "VkPhysicalDevice",
                    "pPhysicalDeviceCount",
                ),
            ],
            DispatchLevel::Instance,
        );
        let roles = classify_params(&c, &empty_pnext());
        let pattern = classify_command(&c, &roles);
        let ret = stringify(emit_return_type(&c, &roles, pattern));

        assert!(ret.contains("VkResult < Vec < PhysicalDevice >>"));
    }

    #[test]
    fn return_type_void_forward_is_empty() {
        let c = cmd(
            "vkCmdDraw",
            "void",
            vec![
                param("commandBuffer", "VkCommandBuffer"),
                param("vertexCount", "uint32_t"),
            ],
            DispatchLevel::Device,
        );
        let roles = classify_params(&c, &empty_pnext());
        let pattern = classify_command(&c, &roles);
        let ret = stringify(emit_return_type(&c, &roles, pattern));

        assert!(ret.is_empty());
    }

    // -- Body tests ---------------------------------------------------------

    #[test]
    fn body_create_uses_check_and_returns_ok() {
        let c = cmd(
            "vkCreateFence",
            "VkResult",
            vec![
                param("device", "VkDevice"),
                const_ptr("pCreateInfo", "VkFenceCreateInfo"),
                optional_const_ptr("pAllocator", "VkAllocationCallbacks"),
                mut_ptr("pFence", "VkFence"),
            ],
            DispatchLevel::Device,
        );
        let roles = classify_params(&c, &empty_pnext());
        let pattern = classify_command(&c, &roles);
        let body = stringify(emit_body(&c, &roles, pattern));

        assert!(body.contains("self . commands () . create_fence . expect"));
        assert!(body.contains("let alloc_ptr"));
        assert!(body.contains("let mut out = unsafe { core :: mem :: zeroed () }"));
        assert!(body.contains("check (unsafe { fp (self . handle ()"));
        assert!(body.contains("Ok (out)"));
    }

    #[test]
    fn body_destroy_no_check_no_return() {
        let c = cmd(
            "vkDestroyFence",
            "void",
            vec![
                param("device", "VkDevice"),
                param("fence", "VkFence"),
                optional_const_ptr("pAllocator", "VkAllocationCallbacks"),
            ],
            DispatchLevel::Device,
        );
        let roles = classify_params(&c, &empty_pnext());
        let pattern = classify_command(&c, &roles);
        let body = stringify(emit_body(&c, &roles, pattern));

        assert!(body.contains("unsafe { fp (self . handle () , fence , alloc_ptr) }"));
        assert!(!body.contains("check"));
        assert!(!body.contains("Ok"));
    }

    #[test]
    fn body_enumerate_uses_two_call() {
        let c = cmd(
            "vkEnumeratePhysicalDevices",
            "VkResult",
            vec![
                param("instance", "VkInstance"),
                mut_ptr("pPhysicalDeviceCount", "uint32_t"),
                mut_ptr_with_len(
                    "pPhysicalDevices",
                    "VkPhysicalDevice",
                    "pPhysicalDeviceCount",
                ),
            ],
            DispatchLevel::Instance,
        );
        let roles = classify_params(&c, &empty_pnext());
        let pattern = classify_command(&c, &roles);
        let body = stringify(emit_body(&c, &roles, pattern));

        assert!(body.contains("enumerate_two_call"));
        assert!(body.contains("| count , data |"));
        assert!(body.contains("unsafe { fp (self . handle () , count , data) }"));
    }

    #[test]
    fn body_fill_uses_fill_two_call() {
        let c = cmd(
            "vkGetPhysicalDeviceQueueFamilyProperties",
            "void",
            vec![
                param("physicalDevice", "VkPhysicalDevice"),
                mut_ptr("pQueueFamilyPropertyCount", "uint32_t"),
                mut_ptr_with_len(
                    "pQueueFamilyProperties",
                    "VkQueueFamilyProperties",
                    "pQueueFamilyPropertyCount",
                ),
            ],
            DispatchLevel::Instance,
        );
        let roles = classify_params(&c, &empty_pnext());
        let pattern = classify_command(&c, &roles);
        let body = stringify(emit_body(&c, &roles, pattern));

        assert!(body.contains("fill_two_call"));
        assert!(body.contains("unsafe { fp (physical_device , count , data) }"));
    }

    #[test]
    fn body_query_returns_zeroed_out() {
        let c = cmd(
            "vkGetPhysicalDeviceProperties",
            "void",
            vec![
                param("physicalDevice", "VkPhysicalDevice"),
                mut_ptr("pProperties", "VkPhysicalDeviceProperties"),
            ],
            DispatchLevel::Instance,
        );
        let roles = classify_params(&c, &empty_pnext());
        let pattern = classify_command(&c, &roles);
        let body = stringify(emit_body(&c, &roles, pattern));

        assert!(body.contains("let mut out = unsafe { core :: mem :: zeroed () }"));
        assert!(body.contains("unsafe { fp (physical_device , & mut out) }"));
        assert!(!body.contains("check"));
        // Last expression is `out`
        assert!(body.ends_with("out }") || body.contains("; out"));
    }

    #[test]
    fn body_result_only_uses_check() {
        let c = cmd(
            "vkDeviceWaitIdle",
            "VkResult",
            vec![param("device", "VkDevice")],
            DispatchLevel::Device,
        );
        let roles = classify_params(&c, &empty_pnext());
        let pattern = classify_command(&c, &roles);
        let body = stringify(emit_body(&c, &roles, pattern));

        assert!(body.contains("check (unsafe { fp (self . handle ()) })"));
    }

    #[test]
    fn body_void_forward_calls_fp() {
        let c = cmd(
            "vkCmdDraw",
            "void",
            vec![
                param("commandBuffer", "VkCommandBuffer"),
                param("vertexCount", "uint32_t"),
                param("instanceCount", "uint32_t"),
                param("firstVertex", "uint32_t"),
                param("firstInstance", "uint32_t"),
            ],
            DispatchLevel::Device,
        );
        let roles = classify_params(&c, &empty_pnext());
        let pattern = classify_command(&c, &roles);
        let body = stringify(emit_body(&c, &roles, pattern));

        assert!(body.contains(
            "unsafe { fp (command_buffer , vertex_count , instance_count , first_vertex , first_instance) }"
        ));
    }

    #[test]
    fn body_input_array_uses_len_and_as_ptr() {
        let c = cmd(
            "vkQueueSubmit",
            "VkResult",
            vec![
                param("queue", "VkQueue"),
                param("submitCount", "uint32_t"),
                const_ptr_with_len("pSubmits", "VkSubmitInfo", "submitCount"),
                param("fence", "VkFence"),
            ],
            DispatchLevel::Device,
        );
        let roles = classify_params(&c, &empty_pnext());
        let pattern = classify_command(&c, &roles);
        let body = stringify(emit_body(&c, &roles, pattern));

        assert!(body.contains("submits . len () as u32"));
        assert!(body.contains("submits . as_ptr ()"));
    }

    // -- Exclusion test -----------------------------------------------------

    #[test]
    fn excluded_commands_not_emitted() {
        let commands = vec![
            cmd(
                "vkCreateInstance",
                "VkResult",
                vec![
                    const_ptr("pCreateInfo", "VkInstanceCreateInfo"),
                    optional_const_ptr("pAllocator", "VkAllocationCallbacks"),
                    mut_ptr("pInstance", "VkInstance"),
                ],
                DispatchLevel::Entry,
            ),
            cmd(
                "vkDestroyInstance",
                "void",
                vec![
                    param("instance", "VkInstance"),
                    optional_const_ptr("pAllocator", "VkAllocationCallbacks"),
                ],
                DispatchLevel::Instance,
            ),
        ];
        let exclusions = exclusion_set();

        let (entry_methods, ec) =
            emit_methods(DispatchLevel::Entry, &commands, &empty_pnext(), &exclusions);
        let (instance_methods, ic) = emit_methods(
            DispatchLevel::Instance,
            &commands,
            &empty_pnext(),
            &exclusions,
        );

        assert_eq!(ec, 0); // vkCreateInstance is excluded
        assert_eq!(ic, 1); // vkDestroyInstance is NOT excluded
        assert!(stringify(entry_methods).is_empty());
        assert!(stringify(instance_methods).contains("destroy_instance"));
    }

    // -- Full method emission test ------------------------------------------

    #[test]
    fn full_method_create_buffer() {
        let c = cmd(
            "vkCreateBuffer",
            "VkResult",
            vec![
                param("device", "VkDevice"),
                const_ptr("pCreateInfo", "VkBufferCreateInfo"),
                optional_const_ptr("pAllocator", "VkAllocationCallbacks"),
                mut_ptr("pBuffer", "VkBuffer"),
            ],
            DispatchLevel::Device,
        );
        let roles = classify_params(&c, &empty_pnext());
        let pattern = classify_command(&c, &roles);
        let method = stringify(emit_method(&c, &roles, pattern));

        assert!(method.contains("pub unsafe fn create_buffer"));
        assert!(method.contains("& self"));
        assert!(method.contains("create_info : & BufferCreateInfo"));
        assert!(method.contains("allocator : Option < & AllocationCallbacks >"));
        assert!(method.contains("-> VkResult < Buffer >"));
        assert!(method.contains("Ok (out)"));
    }

    #[test]
    fn full_method_enumerate_with_extra_params() {
        let c = cmd(
            "vkEnumerateDeviceExtensionProperties",
            "VkResult",
            vec![
                param("physicalDevice", "VkPhysicalDevice"),
                optional_const_ptr("pLayerName", "char"),
                mut_ptr("pPropertyCount", "uint32_t"),
                mut_ptr_with_len("pProperties", "VkExtensionProperties", "pPropertyCount"),
            ],
            DispatchLevel::Instance,
        );
        let roles = classify_params(&c, &empty_pnext());
        let pattern = classify_command(&c, &roles);
        let method = stringify(emit_method(&c, &roles, pattern));

        assert!(method.contains("pub unsafe fn enumerate_device_extension_properties"));
        // physicalDevice is Regular (not self-handle for Instance)
        assert!(method.contains("physical_device"));
        // pLayerName is *const char, optional, but not Vk type → raw pointer
        assert!(method.contains("p_layer_name"));
        assert!(method.contains("-> VkResult < Vec < ExtensionProperties >>"));
        assert!(method.contains("enumerate_two_call"));
        // The closure passes physical_device, p_layer_name, then count/data
        assert!(method.contains("unsafe { fp (physical_device , p_layer_name , count , data) }"));
    }

    #[test]
    fn full_method_map_memory_raw_forward() {
        let c = cmd(
            "vkMapMemory",
            "VkResult",
            vec![
                param("device", "VkDevice"),
                param("memory", "VkDeviceMemory"),
                param("offset", "VkDeviceSize"),
                param("size", "VkDeviceSize"),
                param("flags", "VkMemoryMapFlags"),
                double_mut_ptr("ppData", "void"),
            ],
            DispatchLevel::Device,
        );
        let roles = classify_params(&c, &empty_pnext());
        let pattern = classify_command(&c, &roles);
        let method = stringify(emit_method(&c, &roles, pattern));

        assert!(method.contains("pub unsafe fn map_memory"));
        assert!(method.contains("-> VkResult < () >"));
        // Double pointer passed through as raw
        assert!(method.contains("pp_data"));
        assert!(method.contains("check (unsafe { fp (self . handle ()"));
    }

    // -- Method count verification ------------------------------------------

    #[test]
    fn method_counts_match_expected_range() {
        let vk_xml = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("vk.xml");
        let registry = crate::parse::parse_registry(&vk_xml);
        let (entry, instance, device) = super::wrapper_counts(&registry);

        // Entry: 5 total commands minus 4 excluded = 1+
        assert!(entry >= 1, "entry methods too low: {entry}");
        assert!(entry <= 15, "entry methods unexpectedly high: {entry}");

        // Instance: ~109 total minus ~7 excluded = ~99-105
        assert!(instance >= 80, "instance methods too low: {instance}");
        assert!(
            instance <= 130,
            "instance methods unexpectedly high: {instance}"
        );

        // Device: ~628 total minus ~0 excluded = ~620-640
        assert!(device >= 550, "device methods too low: {device}");
        assert!(device <= 700, "device methods unexpectedly high: {device}");

        // Total should be close to the full command count minus exclusions
        let total = entry + instance + device;
        assert!(total >= 650, "total methods too low: {total}");
    }

    // -- Audit report -------------------------------------------------------

    /// Generates `docs/wrapper_audit.md` with every command's classification.
    ///
    /// Run with: `cargo test -p generator -- generate_wrapper_audit --ignored --nocapture`
    #[test]
    #[ignore] // expensive, writes to docs/
    fn generate_wrapper_audit() {
        use std::fmt::Write;

        let vk_xml = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("vk.xml");
        let registry = crate::parse::parse_registry(&vk_xml);
        let pnext = build_pnext_struct_set(&registry);
        let exclusions = super::exclusion_set();

        let mut report = String::new();
        writeln!(report, "# Wrapper Audit Report\n").unwrap();
        writeln!(
            report,
            "Auto-generated. Run `cargo test -p generator -- generate_wrapper_audit --ignored` to regenerate.\n"
        )
        .unwrap();

        let mut pattern_counts: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        let mut raw_forward_commands = Vec::new();

        for (level_name, level) in [
            ("Entry", DispatchLevel::Entry),
            ("Instance", DispatchLevel::Instance),
            ("Device", DispatchLevel::Device),
        ] {
            let cmds: Vec<&CommandDef> = registry
                .commands
                .iter()
                .filter(|c| c.dispatch_level == level)
                .collect();

            writeln!(report, "## {level_name} ({} commands)\n", cmds.len()).unwrap();
            writeln!(report, "| Command | Pattern | Params | Transforms |").unwrap();
            writeln!(report, "|---------|---------|--------|------------|").unwrap();

            for cmd in &cmds {
                let excluded = exclusions.contains(&cmd.name);
                let roles = classify_params(cmd, &pnext);
                let pattern = classify_command(cmd, &roles);
                let pattern_name = match pattern {
                    CommandPattern::Create => "Create",
                    CommandPattern::Destroy => "Destroy",
                    CommandPattern::Enumerate => "Enumerate",
                    CommandPattern::Fill => "Fill",
                    CommandPattern::Query => "Query",
                    CommandPattern::ResultOnly => "ResultOnly",
                    CommandPattern::VoidForward => "VoidForward",
                };

                if !excluded {
                    *pattern_counts.entry(pattern_name).or_default() += 1;
                }

                let mut transforms = Vec::new();
                for (param, role) in cmd.params.iter().zip(roles.iter()) {
                    match role {
                        ParamRole::SelfHandle => {
                            transforms.push(format!("{} → self", param.name));
                        }
                        ParamRole::Output => {
                            transforms.push(format!("{} → return", param.name));
                        }
                        ParamRole::OutputCount { .. } => {
                            transforms.push(format!("{} → two-call count", param.name));
                        }
                        ParamRole::OutputArray { .. } => {
                            transforms.push(format!("{} → two-call data", param.name));
                        }
                        ParamRole::InputCount { partner } => {
                            transforms.push(format!(
                                "{} → {}.len()",
                                param.name, cmd.params[*partner].name
                            ));
                        }
                        ParamRole::InputArray { .. } => {
                            transforms.push(format!("{} → &[T]", param.name));
                        }
                        ParamRole::Allocator => {
                            transforms.push(format!("{} → Option", param.name));
                        }
                        ParamRole::Regular => {}
                    }
                }

                let raw_params: Vec<String> = cmd
                    .params
                    .iter()
                    .zip(roles.iter())
                    .filter(|(_, r)| matches!(r, ParamRole::Regular))
                    .filter(|(p, _)| p.is_pointer)
                    .map(|(p, _)| p.name.clone())
                    .collect();

                let status = if excluded {
                    "**EXCLUDED**"
                } else {
                    pattern_name
                };

                let transform_str = if transforms.is_empty() {
                    "—".to_string()
                } else {
                    transforms.join(", ")
                };

                writeln!(
                    report,
                    "| `{}` | {} | {} | {} |",
                    cmd.name,
                    status,
                    cmd.params.len(),
                    transform_str,
                )
                .unwrap();

                // Track commands with raw pointer params that stayed Regular.
                if !excluded && !raw_params.is_empty() {
                    raw_forward_commands.push((cmd.name.clone(), pattern_name, raw_params));
                }
            }

            writeln!(report).unwrap();
        }

        // Summary section.
        writeln!(report, "## Summary\n").unwrap();
        writeln!(report, "| Pattern | Count |").unwrap();
        writeln!(report, "|---------|-------|").unwrap();
        let mut sorted_patterns: Vec<_> = pattern_counts.into_iter().collect();
        sorted_patterns.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        for (name, count) in &sorted_patterns {
            writeln!(report, "| {name} | {count} |").unwrap();
        }
        let total: usize = sorted_patterns.iter().map(|(_, c)| c).sum();
        writeln!(report, "| **Total** | **{total}** |").unwrap();

        // Raw pointer warnings.
        writeln!(
            report,
            "\n## Commands with raw pointer params ({} total)\n",
            raw_forward_commands.len()
        )
        .unwrap();
        writeln!(
            report,
            "These commands have parameters that passed through as raw pointers"
        )
        .unwrap();
        writeln!(
            report,
            "because they didn't match any ergonomic transform rule.\n"
        )
        .unwrap();
        writeln!(report, "| Command | Pattern | Raw params |").unwrap();
        writeln!(report, "|---------|---------|------------|").unwrap();
        for (name, pattern, params) in &raw_forward_commands {
            writeln!(report, "| `{name}` | {pattern} | {} |", params.join(", ")).unwrap();
        }

        let out_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/wrapper_audit.md");
        std::fs::write(&out_path, &report)
            .unwrap_or_else(|e| panic!("failed to write {}: {e}", out_path.display()));
        println!("Wrote audit report to {}", out_path.display());
    }
}
