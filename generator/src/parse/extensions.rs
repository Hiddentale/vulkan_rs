//! Provenance stamping, platform/extension collection, and funcpointer XML parsing.

use std::collections::HashMap;

use vk_parse::{ExtensionChild, FeatureChild, InterfaceItem, Registry, RegistryChild};

use super::{
    ExtensionDef, ExtensionItem, FuncPointerDef, ParamDef, PlatformDef, VkRegistry,
    is_vulkan_extension, strip_vk,
};

pub(super) fn stamp_provenance(registry: &Registry, reg: &mut VkRegistry) {
    let mut provider_map: HashMap<String, String> = HashMap::new();

    for child in &registry.0 {
        match child {
            RegistryChild::Feature(feature) => {
                let provider = feature.name.clone();
                for fc in &feature.children {
                    let items = match fc {
                        FeatureChild::Require { items, .. } => items,
                        _ => continue,
                    };
                    for item in items {
                        match item {
                            InterfaceItem::Type { name, .. } => {
                                provider_map
                                    .entry(strip_vk(name))
                                    .or_insert(provider.clone());
                            }
                            InterfaceItem::Command { name, .. } => {
                                provider_map.entry(name.clone()).or_insert(provider.clone());
                            }
                            _ => {}
                        }
                    }
                }
            }
            RegistryChild::Extensions(extensions) => {
                for ext in &extensions.children {
                    if !is_vulkan_extension(ext) {
                        continue;
                    }
                    for ec in &ext.children {
                        let items = match ec {
                            ExtensionChild::Require { items, .. } => items,
                            _ => continue,
                        };
                        for item in items {
                            match item {
                                InterfaceItem::Type { name, .. } => {
                                    provider_map
                                        .entry(strip_vk(name))
                                        .or_insert(ext.name.clone());
                                }
                                InterfaceItem::Command { name, .. } => {
                                    provider_map.entry(name.clone()).or_insert(ext.name.clone());
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    for h in &mut reg.handles {
        h.provided_by = provider_map.get(&h.name).cloned();
    }
    for s in &mut reg.structs {
        s.provided_by = provider_map.get(&s.name).cloned();
    }
    for c in &mut reg.commands {
        c.provided_by = provider_map.get(&c.name).cloned();
    }
}

pub(super) fn collect_platforms(registry: &Registry, reg: &mut VkRegistry) {
    for child in &registry.0 {
        if let RegistryChild::Platforms(platforms) = child {
            for p in &platforms.children {
                reg.platforms.push(PlatformDef {
                    name: p.name.clone(),
                    protect: p.protect.clone(),
                });
            }
        }
    }
}

pub(super) fn collect_extensions(registry: &Registry, reg: &mut VkRegistry) {
    for child in &registry.0 {
        let extensions = match child {
            RegistryChild::Extensions(exts) => &exts.children,
            _ => continue,
        };
        for ext in extensions {
            if !is_vulkan_extension(ext) {
                continue;
            }

            let mut ext_items = Vec::new();
            for child in &ext.children {
                let req_items = match child {
                    ExtensionChild::Require { items, .. } => items,
                    _ => continue,
                };
                for item in req_items {
                    match item {
                        InterfaceItem::Type { name, .. } => {
                            ext_items.push(ExtensionItem::Type(name.clone()));
                        }
                        InterfaceItem::Command { name, .. } => {
                            ext_items.push(ExtensionItem::Command(name.clone()));
                        }
                        _ => {}
                    }
                }
            }

            reg.extensions.push(ExtensionDef {
                name: ext.name.clone(),
                number: ext.number.unwrap_or(0),
                ext_type: ext.ext_type.clone(),
                platform: ext.platform.clone(),
                depends: ext.depends.clone(),
                promoted_to: ext.promotedto.clone(),
                deprecated_by: ext.deprecatedby.clone(),
                supported: ext.supported.clone().unwrap_or_default(),
                items: ext_items,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Funcpointer parsing (from raw XML, since vk-parse skips proto/param)
// ---------------------------------------------------------------------------

fn extract_tag<'a>(xml: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);
    let start = xml.find(&open)?;
    let after_open = &xml[start..];
    let content_start = after_open.find('>')? + 1;
    let content = &xml[start + content_start..];
    let end = content.find(&close)?;
    Some(&content[..end])
}

fn parse_funcpointer_xml(block: &str) -> Option<FuncPointerDef> {
    let proto = extract_tag(block, "proto")?;
    let name = extract_tag(proto, "name")?.to_string();
    let return_type = extract_tag(proto, "type")?.to_string();
    let is_return_pointer = proto.contains('*');

    let mut params = Vec::new();
    let mut search = block;
    while let Some(param_start) = search.find("<param") {
        let rest = &search[param_start..];
        let param_end = match rest.find("</param>") {
            Some(e) => e + "</param>".len(),
            None => break,
        };
        let param_xml = &rest[..param_end];

        if let (Some(type_name), Some(param_name)) = (
            extract_tag(param_xml, "type"),
            extract_tag(param_xml, "name"),
        ) {
            let code = param_xml;
            let pointer_count = code.matches('*').count();
            let is_const = code.contains("const");

            params.push(ParamDef {
                name: param_name.to_string(),
                type_name: type_name.to_string(),
                is_pointer: pointer_count >= 1,
                is_const,
                is_double_pointer: pointer_count >= 2,
                array_size: None,
                optional: false,
                len: None,
                extern_sync: None,
            });
        }

        search = &search[param_start + param_end..];
    }

    Some(FuncPointerDef {
        name,
        return_type,
        is_return_pointer,
        params,
    })
}

pub fn parse_func_pointers(xml_text: &str) -> Vec<FuncPointerDef> {
    let mut result = Vec::new();

    let marker = "category=\"funcpointer\"";
    let mut rest = xml_text;
    while let Some(marker_pos) = rest.find(marker) {
        let before = &rest[..marker_pos];
        let type_start = match before.rfind("<type") {
            Some(s) => s,
            None => {
                rest = &rest[marker_pos + marker.len()..];
                continue;
            }
        };

        let from_block = &rest[type_start..];

        if let Some(block_end) = find_closing_type_tag(from_block) {
            let block = &from_block[..block_end];
            if let Some(fp) = parse_funcpointer_xml(block) {
                result.push(fp);
            }
        }

        rest = &rest[marker_pos + marker.len()..];
    }

    result
}

fn find_closing_type_tag(xml: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut pos = 0;
    while pos < xml.len() {
        if xml[pos..].starts_with("</type>") {
            depth -= 1;
            if depth == 0 {
                return Some(pos + "</type>".len());
            }
            pos += "</type>".len();
        } else if xml[pos..].starts_with("<type") {
            let after = pos + "<type".len();
            if after < xml.len() {
                let ch = xml.as_bytes()[after];
                if ch == b' ' || ch == b'>' || ch == b'/' {
                    depth += 1;
                }
            }
            pos += 1;
        } else {
            pos += 1;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- extract_tag -----------------------------------------------------------

    #[test]
    fn extract_tag_simple() {
        let xml = "<name>vkFoo</name>";
        assert_eq!(extract_tag(xml, "name"), Some("vkFoo"));
    }

    #[test]
    fn extract_tag_with_attributes() {
        let xml = r#"<type category="funcpointer">void</type>"#;
        assert_eq!(extract_tag(xml, "type"), Some("void"));
    }

    #[test]
    fn extract_tag_nested_content() {
        let xml = "<proto>VKAPI_ATTR <type>VkResult</type> VKAPI_CALL <name>vkFoo</name></proto>";
        let proto = extract_tag(xml, "proto").expect("should extract proto");
        assert!(proto.contains("VkResult"));
        assert!(proto.contains("vkFoo"));
    }

    #[test]
    fn extract_tag_missing_tag_returns_none() {
        assert!(extract_tag("<foo>bar</foo>", "baz").is_none());
    }

    #[test]
    fn extract_tag_missing_close_returns_none() {
        assert!(extract_tag("<name>bar", "name").is_none());
    }

    // -- find_closing_type_tag -------------------------------------------------

    #[test]
    fn find_closing_simple() {
        let xml = "<type>void</type>";
        assert_eq!(find_closing_type_tag(xml), Some(xml.len()));
    }

    #[test]
    fn find_closing_nested() {
        let xml = r#"<type category="funcpointer"><type>VkBool32</type></type>"#;
        assert_eq!(find_closing_type_tag(xml), Some(xml.len()));
    }

    #[test]
    fn find_closing_no_match() {
        assert!(find_closing_type_tag("<other>content</other>").is_none());
    }

    #[test]
    fn find_closing_multiple_inner() {
        let xml = "<type><type>A</type><type>B</type></type>";
        assert_eq!(find_closing_type_tag(xml), Some(xml.len()));
    }

    // -- parse_funcpointer_xml -------------------------------------------------

    #[test]
    fn parse_funcpointer_basic() {
        let xml = r#"<type category="funcpointer">
            <proto><type>void</type> <name>PFN_vkFreeFunction</name></proto>
            <param><type>void</type>* <name>pUserData</name></param>
            <param><type>void</type>* <name>pMemory</name></param>
        </type>"#;
        let fp = parse_funcpointer_xml(xml).expect("should parse funcpointer");
        assert_eq!(fp.name, "PFN_vkFreeFunction");
        assert_eq!(fp.return_type, "void");
        assert!(!fp.is_return_pointer);
        assert_eq!(fp.params.len(), 2);
        assert_eq!(fp.params[0].name, "pUserData");
        assert!(fp.params[0].is_pointer);
        assert_eq!(fp.params[1].name, "pMemory");
    }

    #[test]
    fn parse_funcpointer_return_pointer() {
        let xml = r#"<type category="funcpointer">
            <proto><type>void</type>* <name>PFN_vkAllocationFunction</name></proto>
            <param><type>size_t</type> <name>size</name></param>
        </type>"#;
        let fp = parse_funcpointer_xml(xml).expect("should parse return pointer");
        assert!(fp.is_return_pointer);
        assert_eq!(fp.params.len(), 1);
        assert!(!fp.params[0].is_pointer);
    }

    #[test]
    fn parse_funcpointer_const_param() {
        let xml = r#"<type category="funcpointer">
            <proto><type>void</type> <name>PFN_vkFoo</name></proto>
            <param>const <type>VkFoo</type>* <name>pFoo</name></param>
        </type>"#;
        let fp = parse_funcpointer_xml(xml).expect("should parse const param");
        assert!(fp.params[0].is_const);
        assert!(fp.params[0].is_pointer);
    }

    #[test]
    fn parse_funcpointer_double_pointer() {
        let xml = r#"<type category="funcpointer">
            <proto><type>void</type> <name>PFN_vkFoo</name></proto>
            <param><type>void</type>** <name>ppData</name></param>
        </type>"#;
        let fp = parse_funcpointer_xml(xml).expect("should parse double pointer");
        assert!(fp.params[0].is_double_pointer);
    }

    #[test]
    fn parse_funcpointer_no_params() {
        let xml = r#"<type category="funcpointer">
            <proto><type>VkBool32</type> <name>PFN_vkFoo</name></proto>
        </type>"#;
        let fp = parse_funcpointer_xml(xml).expect("should parse no-param funcpointer");
        assert_eq!(fp.return_type, "VkBool32");
        assert!(fp.params.is_empty());
    }

    #[test]
    fn parse_funcpointer_missing_proto_returns_none() {
        let xml = "<type><name>foo</name></type>";
        assert!(parse_funcpointer_xml(xml).is_none());
    }

    // -- parse_func_pointers (full pipeline) -----------------------------------

    #[test]
    fn parse_func_pointers_finds_markers() {
        let xml = r#"
<types>
    <type category="funcpointer">
        <proto><type>void</type> <name>PFN_vkA</name></proto>
    </type>
    <type category="basetype">not a funcpointer</type>
    <type category="funcpointer">
        <proto><type>uint32_t</type> <name>PFN_vkB</name></proto>
        <param><type>VkDevice</type> <name>device</name></param>
    </type>
</types>"#;
        let fps = parse_func_pointers(xml);
        assert_eq!(fps.len(), 2);
        assert_eq!(fps[0].name, "PFN_vkA");
        assert_eq!(fps[1].name, "PFN_vkB");
        assert_eq!(fps[1].params.len(), 1);
    }

    #[test]
    fn parse_func_pointers_empty_input() {
        assert!(parse_func_pointers("").is_empty());
    }

    #[test]
    fn parse_func_pointers_no_markers() {
        assert!(
            parse_func_pointers("<types><type category=\"struct\">foo</type></types>").is_empty()
        );
    }
}
