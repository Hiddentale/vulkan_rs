//! Parameter role classification and command pattern detection for wrapper generation.
//!
//! Analyses `CommandDef` metadata to determine how each parameter should be
//! transformed in an ergonomic wrapper method, and what overall calling pattern
//! the command follows.

use std::collections::{HashMap, HashSet};

use crate::parse::{CommandDef, DispatchLevel, ParamDef, VkRegistry};

// ---------------------------------------------------------------------------
// Parameter roles
// ---------------------------------------------------------------------------

/// Role of a parameter in a Vulkan command from the wrapper's perspective.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamRole {
    /// First dispatchable handle, becomes `&self`.
    SelfHandle,
    /// Single output value, becomes the return value.
    Output,
    /// Count param for an output array (enumerate/fill pair).
    /// `partner` is the index of the array param.
    OutputCount { partner: usize },
    /// Array param of an output enumerate/fill pair.
    /// `count` is the index of the count param.
    OutputArray { count: usize },
    /// Output array whose length is known before the call.
    /// Elided from the signature; the wrapper allocates a `Vec` and returns it.
    AllocateArrayOutput { count_source: CountSource },
    /// Count param for an input slice, suppressed from the wrapper signature.
    /// `partner` is the index of the array param.
    InputCount { partner: usize },
    /// Input array param with associated count, becomes `&[T]`.
    /// `count` is the index of the count param.
    InputArray { count: usize },
    /// Output param whose type has pNext, becomes `&mut T` in the signature.
    /// The caller initializes the struct (and optional pNext chain) before calling.
    PNextOutput,
    /// Optional allocator (`*const AllocationCallbacks`).
    Allocator,
    /// Regular input parameter, passed through as its resolved C type.
    Regular,
}

/// How the array length is determined for an `AllocateArrayOutput`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CountSource {
    /// Length comes from a field inside an input struct parameter.
    /// Example: `pAllocateInfo->commandBufferCount` becomes
    /// `StructField { param: "pAllocateInfo", field: "commandBufferCount" }`.
    StructField { param: String, field: String },
    /// Length equals an input slice's `.len()` (the count param is `InputCount`).
    /// `partner` is the index of the `InputArray` param.
    InputArrayLen { partner: usize },
}

// ---------------------------------------------------------------------------
// Command patterns
// ---------------------------------------------------------------------------

/// Overall calling pattern of a Vulkan command, determining the method body
/// template used during wrapper emission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandPattern {
    /// VkResult + AllocateArrayOutput → `VkResult<Vec<T>>`
    AllocateArray,
    /// VkResult + single Output param → `VkResult<T>`
    Create,
    /// void + name contains "Destroy"/"Free" + Allocator → `()`
    Destroy,
    /// VkResult + OutputCount/OutputArray pair → `VkResult<Vec<T>>`
    Enumerate,
    /// void + OutputCount/OutputArray pair → `Vec<T>`
    Fill,
    /// void + single Output param (struct) → `T`
    Query,
    /// VkResult + no output params → `VkResult<()>`
    ResultOnly,
    /// void + no output params → `()`
    VoidForward,
}

// ---------------------------------------------------------------------------
// pNext struct detection
// ---------------------------------------------------------------------------

/// Build a set of struct names (Vk-prefix already stripped) that have a `pNext`
/// member. Output params of these types require caller-controlled initialisation
/// and cannot be safely `zeroed()`, so they stay as `Regular` params.
pub fn build_pnext_struct_set(registry: &VkRegistry) -> HashSet<String> {
    registry
        .structs
        .iter()
        .filter(|s| s.members.iter().any(|m| m.name == "pNext"))
        .map(|s| s.name.clone())
        .collect()
}

// ---------------------------------------------------------------------------
// Parameter classification
// ---------------------------------------------------------------------------

/// Each dispatch level implicitly passes its owning handle as the first
/// parameter, which the wrapper receives as `&self`.
fn self_handle_type(level: DispatchLevel) -> Option<&'static str> {
    match level {
        DispatchLevel::Entry => None,
        DispatchLevel::Instance => Some("VkInstance"),
        DispatchLevel::Device => Some("VkDevice"),
    }
}

/// Classify every parameter of a command into a [`ParamRole`].
///
/// `pnext_structs` is the set returned by [`build_pnext_struct_set`], struct
/// names (Vk-prefix stripped) whose instances have `sType`/`pNext` and therefore
/// cannot be used as simple output values.
pub fn assign_param_roles(cmd: &CommandDef, pnext_structs: &HashSet<String>) -> Vec<ParamRole> {
    let params = &cmd.params;
    let mut roles = vec![ParamRole::Regular; params.len()];

    let name_to_idx: HashMap<&str, usize> = params
        .iter()
        .enumerate()
        .map(|(i, p)| (p.name.as_str(), i))
        .collect();

    mark_self_handle(params, cmd.dispatch_level, &mut roles);
    mark_output_pairs(params, &name_to_idx, &mut roles);
    mark_input_pairs(params, &name_to_idx, &mut roles);
    mark_allocators(params, &mut roles);
    mark_allocate_array_outputs(params, &name_to_idx, &mut roles);
    mark_single_output(params, pnext_structs, &mut roles);
    mark_pnext_outputs(params, pnext_structs, &mut roles);

    roles
}

/// Mark the first parameter as `SelfHandle` if it matches the dispatch
/// level's handle type and is passed by value.
fn mark_self_handle(params: &[ParamDef], level: DispatchLevel, roles: &mut [ParamRole]) {
    if let Some(handle_type) = self_handle_type(level)
        && let Some(first) = params.first()
        && first.type_name == handle_type
        && !first.is_pointer
    {
        roles[0] = ParamRole::SelfHandle;
    }
}

/// Resolve a `len` attribute to a parameter index, skipping struct-internal
/// references and null-terminated strings.
fn resolve_len<'a>(
    len: Option<&'a str>,
    name_to_idx: &HashMap<&str, usize>,
) -> Option<(&'a str, usize)> {
    let len = len?;
    if len.contains("->") || len == "null-terminated" {
        return None;
    }
    let &idx = name_to_idx.get(len)?;
    Some((len, idx))
}

/// Identify output count/array pairs: a `*mut uint32_t` count paired with
/// a `*mut T` array whose `len` references the count.
fn mark_output_pairs(
    params: &[ParamDef],
    name_to_idx: &HashMap<&str, usize>,
    roles: &mut [ParamRole],
) {
    for (i, param) in params.iter().enumerate() {
        if roles[i] != ParamRole::Regular {
            continue;
        }
        let Some((_, count_idx)) = resolve_len(param.len.as_deref(), name_to_idx) else {
            continue;
        };
        if roles[count_idx] != ParamRole::Regular {
            continue;
        }
        let count_param = &params[count_idx];

        if param.is_pointer
            && !param.is_const
            && !param.is_double_pointer
            && count_param.is_pointer
            && !count_param.is_const
            && count_param.type_name == "uint32_t"
        {
            roles[count_idx] = ParamRole::OutputCount { partner: i };
            roles[i] = ParamRole::OutputArray { count: count_idx };
        }
    }
}

/// Identify input count/array pairs: a plain `uint32_t` count paired with
/// a `*const T` array whose `len` references the count.
fn mark_input_pairs(
    params: &[ParamDef],
    name_to_idx: &HashMap<&str, usize>,
    roles: &mut [ParamRole],
) {
    for (i, param) in params.iter().enumerate() {
        if roles[i] != ParamRole::Regular {
            continue;
        }
        let Some((_, count_idx)) = resolve_len(param.len.as_deref(), name_to_idx) else {
            continue;
        };
        if roles[count_idx] != ParamRole::Regular {
            continue;
        }
        let count_param = &params[count_idx];

        if param.is_pointer
            && param.is_const
            && !param.is_double_pointer
            && !count_param.is_pointer
            && count_param.type_name == "uint32_t"
        {
            roles[count_idx] = ParamRole::InputCount { partner: i };
            roles[i] = ParamRole::InputArray { count: count_idx };
        }
    }
}

/// Mark `*const VkAllocationCallbacks` parameters as `Allocator`.
fn mark_allocators(params: &[ParamDef], roles: &mut [ParamRole]) {
    for (i, param) in params.iter().enumerate() {
        if roles[i] != ParamRole::Regular {
            continue;
        }
        if param.type_name == "VkAllocationCallbacks" && param.is_pointer && param.is_const {
            roles[i] = ParamRole::Allocator;
        }
    }
}

/// Detect allocate-array outputs: `*mut T` params whose length couldn't form
/// a standard output pair, either because the length references a struct field
/// or a count param already claimed as `InputCount`.
fn mark_allocate_array_outputs(
    params: &[ParamDef],
    name_to_idx: &HashMap<&str, usize>,
    roles: &mut [ParamRole],
) {
    let has_output_pair = roles
        .iter()
        .any(|r| matches!(r, ParamRole::OutputArray { .. }));
    if has_output_pair {
        return;
    }

    for i in 0..params.len() {
        if roles[i] != ParamRole::Regular {
            continue;
        }
        let param = &params[i];
        if !param.is_pointer || param.is_const || param.is_double_pointer {
            continue;
        }
        let len = match param.len.as_deref() {
            Some(l) => l,
            None => continue,
        };

        if let Some((struct_param, field)) = len.split_once("->") {
            roles[i] = ParamRole::AllocateArrayOutput {
                count_source: CountSource::StructField {
                    param: struct_param.to_string(),
                    field: field.to_string(),
                },
            };
        } else if let Some(&count_idx) = name_to_idx.get(len)
            && let ParamRole::InputCount { partner } = roles[count_idx]
        {
            roles[i] = ParamRole::AllocateArrayOutput {
                count_source: CountSource::InputArrayLen { partner },
            };
        }
    }
}

/// If no array output was found, check for a single `*mut T` output param
/// (the last non-pNext, non-double-pointer mutable param).
fn mark_single_output(
    params: &[ParamDef],
    pnext_structs: &HashSet<String>,
    roles: &mut [ParamRole],
) {
    let has_any_output = roles.iter().any(|r| {
        matches!(
            r,
            ParamRole::OutputArray { .. } | ParamRole::AllocateArrayOutput { .. }
        )
    });
    if !has_any_output && let Some(idx) = find_single_output(params, roles, pnext_structs) {
        roles[idx] = ParamRole::Output;
    }
}

/// Mark `*mut T` params whose type has sType/pNext as `PNextOutput`. The
/// caller must initialize the struct (and optional pNext chain) before calling.
fn mark_pnext_outputs(
    params: &[ParamDef],
    pnext_structs: &HashSet<String>,
    roles: &mut [ParamRole],
) {
    for i in 0..params.len() {
        if roles[i] != ParamRole::Regular {
            continue;
        }
        let param = &params[i];
        if !param.is_pointer || param.is_const || param.is_double_pointer {
            continue;
        }
        if param.len.is_some() {
            continue;
        }
        let stripped = param
            .type_name
            .strip_prefix("Vk")
            .unwrap_or(&param.type_name);
        if pnext_structs.contains(stripped) {
            roles[i] = ParamRole::PNextOutput;
        }
    }
}

/// Scan backwards for the last `*mut T` param that qualifies as a single output.
///
/// Disqualified if:
/// - Already classified (SelfHandle, Allocator, etc.)
/// - `const` pointer
/// - Double pointer (`*mut *mut T`,e.g. `vkMapMemory`)
/// - Has a `len` attribute (array output whose count lives elsewhere)
/// - Type is a pNext struct (caller must initialise `sType`/`pNext`)
fn find_single_output(
    params: &[ParamDef],
    roles: &[ParamRole],
    pnext_structs: &HashSet<String>,
) -> Option<usize> {
    for i in (0..params.len()).rev() {
        if roles[i] != ParamRole::Regular {
            continue;
        }
        let param = &params[i];

        if !param.is_pointer || param.is_const {
            continue;
        }
        if param.is_double_pointer {
            continue;
        }
        if param.len.is_some() {
            continue;
        }

        let stripped = param
            .type_name
            .strip_prefix("Vk")
            .unwrap_or(&param.type_name);
        if pnext_structs.contains(stripped) {
            continue;
        }

        return Some(i);
    }
    None
}

// ---------------------------------------------------------------------------
// Command pattern classification
// ---------------------------------------------------------------------------

/// Determine the overall calling pattern for a command based on its return
/// type and the roles assigned to its parameters.
pub fn classify_command(cmd: &CommandDef, roles: &[ParamRole]) -> CommandPattern {
    let has_vk_result = cmd.return_type == "VkResult";
    let has_output = roles.iter().any(|r| matches!(r, ParamRole::Output));
    let has_output_pair = roles
        .iter()
        .any(|r| matches!(r, ParamRole::OutputArray { .. }));
    let has_allocate_array = roles
        .iter()
        .any(|r| matches!(r, ParamRole::AllocateArrayOutput { .. }));
    let is_destroy = cmd.name.contains("Destroy") || cmd.name.contains("Free");

    match (
        has_vk_result,
        has_output_pair,
        has_allocate_array,
        has_output,
        is_destroy,
    ) {
        (true, true, _, _, _) => CommandPattern::Enumerate,
        (true, _, true, _, _) => CommandPattern::AllocateArray,
        (true, false, false, true, _) => CommandPattern::Create,
        (true, false, false, false, _) => CommandPattern::ResultOnly,
        (false, true, _, _, _) => CommandPattern::Fill,
        (false, false, _, true, _) => CommandPattern::Query,
        (false, false, _, false, true) => CommandPattern::Destroy,
        (false, false, _, false, false) => CommandPattern::VoidForward,
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
            is_const: false,
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

    fn mut_ptr_with_struct_len(name: &str, type_name: &str, len: &str) -> ParamDef {
        ParamDef {
            len: Some(len.to_string()),
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

    fn pnext_with(names: &[&str]) -> HashSet<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    // -- assign_param_roles tests ----------------------------------------------

    #[test]
    fn create_buffer_roles() {
        // vkCreateBuffer(VkDevice, *const VkBufferCreateInfo,
        //                *const VkAllocationCallbacks, *mut VkBuffer) -> VkResult
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
        let roles = assign_param_roles(&c, &empty_pnext());
        assert_eq!(
            roles,
            vec![
                ParamRole::SelfHandle,
                ParamRole::Regular,
                ParamRole::Allocator,
                ParamRole::Output,
            ]
        );
    }

    #[test]
    fn destroy_buffer_roles() {
        // vkDestroyBuffer(VkDevice, VkBuffer, *const VkAllocationCallbacks)
        let c = cmd(
            "vkDestroyBuffer",
            "void",
            vec![
                param("device", "VkDevice"),
                param("buffer", "VkBuffer"),
                optional_const_ptr("pAllocator", "VkAllocationCallbacks"),
            ],
            DispatchLevel::Device,
        );
        let roles = assign_param_roles(&c, &empty_pnext());
        assert_eq!(
            roles,
            vec![
                ParamRole::SelfHandle,
                ParamRole::Regular,
                ParamRole::Allocator,
            ]
        );
    }

    #[test]
    fn enumerate_physical_devices_roles() {
        // vkEnumeratePhysicalDevices(VkInstance, *mut uint32_t, *mut VkPhysicalDevice)
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
        let roles = assign_param_roles(&c, &empty_pnext());
        assert_eq!(
            roles,
            vec![
                ParamRole::SelfHandle,
                ParamRole::OutputCount { partner: 2 },
                ParamRole::OutputArray { count: 1 },
            ]
        );
    }

    #[test]
    fn get_physical_device_queue_family_properties_roles() {
        // vkGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice,
        //     *mut uint32_t, *mut VkQueueFamilyProperties) -> void
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
        let roles = assign_param_roles(&c, &empty_pnext());
        // VkPhysicalDevice is NOT the self-handle for Instance (only VkInstance is).
        assert_eq!(
            roles,
            vec![
                ParamRole::Regular,
                ParamRole::OutputCount { partner: 2 },
                ParamRole::OutputArray { count: 1 },
            ]
        );
    }

    #[test]
    fn get_physical_device_properties_roles() {
        // vkGetPhysicalDeviceProperties(VkPhysicalDevice,
        //     *mut VkPhysicalDeviceProperties) -> void
        let c = cmd(
            "vkGetPhysicalDeviceProperties",
            "void",
            vec![
                param("physicalDevice", "VkPhysicalDevice"),
                mut_ptr("pProperties", "VkPhysicalDeviceProperties"),
            ],
            DispatchLevel::Instance,
        );
        let roles = assign_param_roles(&c, &empty_pnext());
        assert_eq!(roles, vec![ParamRole::Regular, ParamRole::Output,]);
    }

    #[test]
    fn device_wait_idle_roles() {
        // vkDeviceWaitIdle(VkDevice) -> VkResult
        let c = cmd(
            "vkDeviceWaitIdle",
            "VkResult",
            vec![param("device", "VkDevice")],
            DispatchLevel::Device,
        );
        let roles = assign_param_roles(&c, &empty_pnext());
        assert_eq!(roles, vec![ParamRole::SelfHandle]);
    }

    #[test]
    fn cmd_draw_roles() {
        // vkCmdDraw(VkCommandBuffer, u32, u32, u32, u32)
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
        let roles = assign_param_roles(&c, &empty_pnext());
        // VkCommandBuffer is NOT the self-handle for Device (only VkDevice is).
        assert_eq!(
            roles,
            vec![
                ParamRole::Regular,
                ParamRole::Regular,
                ParamRole::Regular,
                ParamRole::Regular,
                ParamRole::Regular,
            ]
        );
    }

    #[test]
    fn queue_submit_with_input_array_roles() {
        // vkQueueSubmit(VkQueue, uint32_t, *const VkSubmitInfo, VkFence) -> VkResult
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
        let roles = assign_param_roles(&c, &empty_pnext());
        assert_eq!(
            roles,
            vec![
                ParamRole::Regular, // VkQueue, not VkDevice
                ParamRole::InputCount { partner: 2 },
                ParamRole::InputArray { count: 1 },
                ParamRole::Regular,
            ]
        );
    }

    #[test]
    fn map_memory_double_pointer_stays_regular() {
        // vkMapMemory(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize,
        //     VkMemoryMapFlags, *mut *mut void) -> VkResult
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
        let roles = assign_param_roles(&c, &empty_pnext());
        // Double pointer stays Regular,not classified as Output.
        assert_eq!(
            roles,
            vec![
                ParamRole::SelfHandle,
                ParamRole::Regular,
                ParamRole::Regular,
                ParamRole::Regular,
                ParamRole::Regular,
                ParamRole::Regular,
            ]
        );
    }

    #[test]
    fn allocate_command_buffers_struct_internal_len() {
        // vkAllocateCommandBuffers(VkDevice, *const VkCommandBufferAllocateInfo,
        //     *mut VkCommandBuffer) -> VkResult
        // len on pCommandBuffers = "pAllocateInfo->commandBufferCount"
        let c = cmd(
            "vkAllocateCommandBuffers",
            "VkResult",
            vec![
                param("device", "VkDevice"),
                const_ptr("pAllocateInfo", "VkCommandBufferAllocateInfo"),
                mut_ptr_with_struct_len(
                    "pCommandBuffers",
                    "VkCommandBuffer",
                    "pAllocateInfo->commandBufferCount",
                ),
            ],
            DispatchLevel::Device,
        );
        let roles = assign_param_roles(&c, &empty_pnext());
        assert_eq!(
            roles,
            vec![
                ParamRole::SelfHandle,
                ParamRole::Regular,
                ParamRole::AllocateArrayOutput {
                    count_source: CountSource::StructField {
                        param: "pAllocateInfo".to_string(),
                        field: "commandBufferCount".to_string(),
                    },
                },
            ]
        );
    }

    #[test]
    fn create_graphics_pipelines_array_output_shared_count() {
        // vkCreateGraphicsPipelines(VkDevice, VkPipelineCache, uint32_t,
        //     *const VkGraphicsPipelineCreateInfo, *const VkAllocationCallbacks,
        //     *mut VkPipeline) -> VkResult
        // len on pPipelines = "createInfoCount" (shared with input array)
        let c = cmd(
            "vkCreateGraphicsPipelines",
            "VkResult",
            vec![
                param("device", "VkDevice"),
                param("pipelineCache", "VkPipelineCache"),
                param("createInfoCount", "uint32_t"),
                const_ptr_with_len(
                    "pCreateInfos",
                    "VkGraphicsPipelineCreateInfo",
                    "createInfoCount",
                ),
                optional_const_ptr("pAllocator", "VkAllocationCallbacks"),
                mut_ptr_with_len("pPipelines", "VkPipeline", "createInfoCount"),
            ],
            DispatchLevel::Device,
        );
        let roles = assign_param_roles(&c, &empty_pnext());
        assert_eq!(
            roles,
            vec![
                ParamRole::SelfHandle,
                ParamRole::Regular,
                ParamRole::InputCount { partner: 3 },
                ParamRole::InputArray { count: 2 },
                ParamRole::Allocator,
                ParamRole::AllocateArrayOutput {
                    count_source: CountSource::InputArrayLen { partner: 3 },
                },
            ]
        );
    }

    #[test]
    fn get_physical_device_properties2_pnext_output_becomes_pnext_output() {
        // vkGetPhysicalDeviceProperties2(VkPhysicalDevice,
        //     *mut VkPhysicalDeviceProperties2) -> void
        // PhysicalDeviceProperties2 has sType/pNext → PNextOutput (&mut T).
        let pnext = pnext_with(&["PhysicalDeviceProperties2"]);
        let c = cmd(
            "vkGetPhysicalDeviceProperties2",
            "void",
            vec![
                param("physicalDevice", "VkPhysicalDevice"),
                mut_ptr("pProperties", "VkPhysicalDeviceProperties2"),
            ],
            DispatchLevel::Instance,
        );
        let roles = assign_param_roles(&c, &pnext);
        assert_eq!(roles, vec![ParamRole::Regular, ParamRole::PNextOutput]);
    }

    #[test]
    fn entry_command_no_self_handle() {
        // vkEnumerateInstanceExtensionProperties(*const char, *mut uint32_t,
        //     *mut VkExtensionProperties) -> VkResult
        let c = cmd(
            "vkEnumerateInstanceExtensionProperties",
            "VkResult",
            vec![
                optional_const_ptr("pLayerName", "char"),
                mut_ptr("pPropertyCount", "uint32_t"),
                mut_ptr_with_len("pProperties", "VkExtensionProperties", "pPropertyCount"),
            ],
            DispatchLevel::Entry,
        );
        let roles = assign_param_roles(&c, &empty_pnext());
        // Entry level: no self-handle. First param stays Regular.
        assert_eq!(
            roles,
            vec![
                ParamRole::Regular,
                ParamRole::OutputCount { partner: 2 },
                ParamRole::OutputArray { count: 1 },
            ]
        );
    }

    #[test]
    fn enumerate_with_extra_params_before_pair() {
        // vkEnumerateDeviceExtensionProperties(VkPhysicalDevice, *const char,
        //     *mut uint32_t, *mut VkExtensionProperties) -> VkResult
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
        let roles = assign_param_roles(&c, &empty_pnext());
        assert_eq!(
            roles,
            vec![
                ParamRole::Regular, // VkPhysicalDevice, not VkInstance
                ParamRole::Regular,
                ParamRole::OutputCount { partner: 3 },
                ParamRole::OutputArray { count: 2 },
            ]
        );
    }

    // -- classify_command tests ---------------------------------------------

    #[test]
    fn pattern_create_buffer() {
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
        let roles = assign_param_roles(&c, &empty_pnext());
        assert_eq!(classify_command(&c, &roles), CommandPattern::Create);
    }

    #[test]
    fn pattern_destroy_buffer() {
        let c = cmd(
            "vkDestroyBuffer",
            "void",
            vec![
                param("device", "VkDevice"),
                param("buffer", "VkBuffer"),
                optional_const_ptr("pAllocator", "VkAllocationCallbacks"),
            ],
            DispatchLevel::Device,
        );
        let roles = assign_param_roles(&c, &empty_pnext());
        assert_eq!(classify_command(&c, &roles), CommandPattern::Destroy);
    }

    #[test]
    fn pattern_enumerate_physical_devices() {
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
        let roles = assign_param_roles(&c, &empty_pnext());
        assert_eq!(classify_command(&c, &roles), CommandPattern::Enumerate);
    }

    #[test]
    fn pattern_fill_queue_family_properties() {
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
        let roles = assign_param_roles(&c, &empty_pnext());
        assert_eq!(classify_command(&c, &roles), CommandPattern::Fill);
    }

    #[test]
    fn pattern_query_physical_device_properties() {
        let c = cmd(
            "vkGetPhysicalDeviceProperties",
            "void",
            vec![
                param("physicalDevice", "VkPhysicalDevice"),
                mut_ptr("pProperties", "VkPhysicalDeviceProperties"),
            ],
            DispatchLevel::Instance,
        );
        let roles = assign_param_roles(&c, &empty_pnext());
        assert_eq!(classify_command(&c, &roles), CommandPattern::Query);
    }

    #[test]
    fn pattern_device_wait_idle() {
        let c = cmd(
            "vkDeviceWaitIdle",
            "VkResult",
            vec![param("device", "VkDevice")],
            DispatchLevel::Device,
        );
        let roles = assign_param_roles(&c, &empty_pnext());
        assert_eq!(classify_command(&c, &roles), CommandPattern::ResultOnly);
    }

    #[test]
    fn pattern_cmd_draw() {
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
        let roles = assign_param_roles(&c, &empty_pnext());
        assert_eq!(classify_command(&c, &roles), CommandPattern::VoidForward);
    }

    #[test]
    fn pattern_map_memory_is_result_only() {
        // Double pointer output stays Regular → no Output detected → ResultOnly.
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
        let roles = assign_param_roles(&c, &empty_pnext());
        assert_eq!(classify_command(&c, &roles), CommandPattern::ResultOnly);
    }

    #[test]
    fn pattern_free_command_buffers_is_destroy() {
        // vkFreeCommandBuffers(VkDevice, VkCommandPool, uint32_t,
        //     *const VkCommandBuffer) -> void
        let c = cmd(
            "vkFreeCommandBuffers",
            "void",
            vec![
                param("device", "VkDevice"),
                param("commandPool", "VkCommandPool"),
                param("commandBufferCount", "uint32_t"),
                const_ptr_with_len("pCommandBuffers", "VkCommandBuffer", "commandBufferCount"),
            ],
            DispatchLevel::Device,
        );
        let roles = assign_param_roles(&c, &empty_pnext());
        // Contains "Free" → Destroy pattern.
        assert_eq!(classify_command(&c, &roles), CommandPattern::Destroy);
    }

    #[test]
    fn pattern_get_properties2_pnext_is_void_forward() {
        // pNext output becomes PNextOutput role, but the command pattern is
        // still VoidForward (void return, no Output/OutputArray).
        let pnext = pnext_with(&["PhysicalDeviceProperties2"]);
        let c = cmd(
            "vkGetPhysicalDeviceProperties2",
            "void",
            vec![
                param("physicalDevice", "VkPhysicalDevice"),
                mut_ptr("pProperties", "VkPhysicalDeviceProperties2"),
            ],
            DispatchLevel::Instance,
        );
        let roles = assign_param_roles(&c, &pnext);
        assert_eq!(classify_command(&c, &roles), CommandPattern::VoidForward);
    }

    #[test]
    fn pattern_create_graphics_pipelines_is_allocate_array() {
        let c = cmd(
            "vkCreateGraphicsPipelines",
            "VkResult",
            vec![
                param("device", "VkDevice"),
                param("pipelineCache", "VkPipelineCache"),
                param("createInfoCount", "uint32_t"),
                const_ptr_with_len(
                    "pCreateInfos",
                    "VkGraphicsPipelineCreateInfo",
                    "createInfoCount",
                ),
                optional_const_ptr("pAllocator", "VkAllocationCallbacks"),
                mut_ptr_with_len("pPipelines", "VkPipeline", "createInfoCount"),
            ],
            DispatchLevel::Device,
        );
        let roles = assign_param_roles(&c, &empty_pnext());
        assert_eq!(classify_command(&c, &roles), CommandPattern::AllocateArray);
    }

    #[test]
    fn pattern_allocate_command_buffers_is_allocate_array() {
        let c = cmd(
            "vkAllocateCommandBuffers",
            "VkResult",
            vec![
                param("device", "VkDevice"),
                const_ptr("pAllocateInfo", "VkCommandBufferAllocateInfo"),
                mut_ptr_with_struct_len(
                    "pCommandBuffers",
                    "VkCommandBuffer",
                    "pAllocateInfo->commandBufferCount",
                ),
            ],
            DispatchLevel::Device,
        );
        let roles = assign_param_roles(&c, &empty_pnext());
        assert_eq!(classify_command(&c, &roles), CommandPattern::AllocateArray);
    }

    // -- build_pnext_struct_set test ----------------------------------------

    #[test]
    fn pnext_set_from_real_registry() {
        use crate::parse;
        let vk_xml = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("vk.xml");
        let registry = parse::parse_registry(&vk_xml);
        let pnext = build_pnext_struct_set(&registry);

        // Sanity checks: well-known pNext structs should be present.
        assert!(pnext.contains("PhysicalDeviceProperties2"));
        assert!(pnext.contains("PhysicalDeviceFeatures2"));
        assert!(pnext.contains("DeviceCreateInfo"));

        // Non-pNext structs should be absent.
        assert!(!pnext.contains("Extent2D"));
        assert!(!pnext.contains("Offset2D"));
    }
}
