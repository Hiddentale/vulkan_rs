//! Pre-generation validation: ensures all referenced types are resolvable.
//!
//! Catches missing `type_map` entries and unknown Vk types at generator time
//! instead of surfacing as compile errors in 161k lines of generated output.

use std::collections::BTreeSet;

use crate::parse::VkRegistry;
use crate::type_map;

/// Verify that every type referenced by struct members and command params is
/// resolvable. Panics with a list of unresolved types if any are found.
pub fn check_type_completeness(registry: &VkRegistry) {
    let known_types = collect_known_types(registry);
    let mut unresolved: BTreeSet<String> = BTreeSet::new();

    for s in &registry.structs {
        for m in &s.members {
            check_type(&m.type_name, &known_types, &mut unresolved);
        }
    }
    for c in &registry.commands {
        for p in &c.params {
            check_type(&p.type_name, &known_types, &mut unresolved);
        }
        if c.return_type != "void" {
            check_type(&c.return_type, &known_types, &mut unresolved);
        }
    }

    if !unresolved.is_empty() {
        let list: Vec<&str> = unresolved.iter().map(String::as_str).collect();
        panic!(
            "type completeness check failed, {} unresolved type(s):\n  {}",
            list.len(),
            list.join("\n  ")
        );
    }
}

fn check_type(type_name: &str, known: &BTreeSet<String>, unresolved: &mut BTreeSet<String>) {
    if type_name == "void" {
        return;
    }
    if type_map::c_type_to_rust(type_name).is_some() {
        return;
    }
    let stripped = type_name.strip_prefix("Vk").unwrap_or(type_name);
    if known.contains(stripped) {
        return;
    }
    // StdVideo types are emitted as opaque stubs.
    if type_name.starts_with("StdVideo") || stripped.starts_with("StdVideo") {
        return;
    }
    // PFN callback types are emitted as opaque stubs.
    if type_name.starts_with("PFN_") || stripped.starts_with("PFN_") {
        return;
    }
    unresolved.insert(type_name.to_string());
}

/// Collect all type names the generator knows how to emit.
fn collect_known_types(registry: &VkRegistry) -> BTreeSet<String> {
    let mut known = BTreeSet::new();

    for h in &registry.handles {
        known.insert(h.name.clone());
    }
    for e in &registry.enums {
        known.insert(e.name.clone());
    }
    for b in &registry.bitmasks {
        known.insert(b.name.clone());
        known.insert(b.flags_name.clone());
    }
    for s in &registry.structs {
        known.insert(s.name.clone());
    }
    // Aliases: both the alias name and target are known types.
    for a in &registry.aliases {
        let clean_alias = a.name.strip_prefix("Vk").unwrap_or(&a.name);
        let clean_target = a.target.strip_prefix("Vk").unwrap_or(&a.target);
        known.insert(clean_alias.to_string());
        known.insert(clean_target.to_string());
    }
    // Flags types referenced by members/params get aliased to u32/FlagBits.
    // These are discovered dynamically by emit_aliases, so we pre-populate
    // any `*Flags*` type as known (they always resolve to either a bitmask
    // alias or a u32/u64 fallback).
    for s in &registry.structs {
        for m in &s.members {
            let stripped = m.type_name.strip_prefix("Vk").unwrap_or(&m.type_name);
            if stripped.contains("Flags") {
                known.insert(stripped.to_string());
            }
        }
    }
    for c in &registry.commands {
        for p in &c.params {
            let stripped = p.type_name.strip_prefix("Vk").unwrap_or(&p.type_name);
            if stripped.contains("Flags") {
                known.insert(stripped.to_string());
            }
        }
    }

    known
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

    #[test]
    fn empty_registry_passes() {
        check_type_completeness(&empty_registry());
    }

    #[test]
    fn known_primitive_passes() {
        let mut reg = empty_registry();
        reg.structs.push(StructDef {
            name: "Foo".to_string(),
            members: vec![MemberDef {
                name: "x".to_string(),
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
            }],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        });
        check_type_completeness(&reg);
    }

    #[test]
    fn known_handle_passes() {
        let mut reg = empty_registry();
        reg.handles.push(HandleDef {
            name: "Buffer".to_string(),
            dispatchable: false,
            parent: None,
            object_type: None,
            provided_by: None,
        });
        reg.structs.push(StructDef {
            name: "Foo".to_string(),
            members: vec![MemberDef {
                name: "buf".to_string(),
                type_name: "VkBuffer".to_string(),
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
            }],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        });
        check_type_completeness(&reg);
    }

    #[test]
    #[should_panic(expected = "unresolved type")]
    fn unknown_type_panics() {
        let mut reg = empty_registry();
        reg.structs.push(StructDef {
            name: "Foo".to_string(),
            members: vec![MemberDef {
                name: "x".to_string(),
                type_name: "VkCompletelyFakeType".to_string(),
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
            }],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        });
        check_type_completeness(&reg);
    }

    #[test]
    fn stdvideo_types_pass() {
        let mut reg = empty_registry();
        reg.structs.push(StructDef {
            name: "Foo".to_string(),
            members: vec![MemberDef {
                name: "x".to_string(),
                type_name: "StdVideoH264PictureInfo".to_string(),
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
            }],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        });
        check_type_completeness(&reg);
    }
}
