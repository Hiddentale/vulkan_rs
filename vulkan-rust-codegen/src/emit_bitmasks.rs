//! Emits `#[repr(transparent)]` bitmask newtypes with bitwise operator impls.

use std::collections::HashSet;

use proc_macro2::{Literal, TokenStream};
use quote::{format_ident, quote};

use crate::emit_enums::strip_extension_suffix;
use crate::parse::{BitmaskBit, BitmaskDef, BitmaskValue, VkRegistry};

use heck::ToShoutySnakeCase;

/// Emit all bitmask types.
pub fn emit_bitmasks(registry: &VkRegistry) -> TokenStream {
    let bitmasks: Vec<TokenStream> = registry.bitmasks.iter().map(emit_bitmask).collect();
    quote! { #(#bitmasks)* }
}

fn emit_bitmask(def: &BitmaskDef) -> TokenStream {
    let name = format_ident!("{}", &def.name);
    let vk_name = format!("Vk{}", &def.name);
    let is_64 = def.bitwidth == 64;
    let prefix = bitmask_bit_prefix(&def.name);

    let (repr, zero_lit) = if is_64 {
        (quote! { u64 }, quote! { 0u64 })
    } else {
        (quote! { u32 }, quote! { 0u32 })
    };

    let mut seen = HashSet::new();
    let constants: Vec<TokenStream> = def
        .bits
        .iter()
        .filter_map(|b| {
            let rust_name = strip_bit_prefix(&b.name, &prefix)?;
            if !seen.insert(rust_name) {
                return None;
            }
            emit_bit_constant(b, &prefix, is_64)
        })
        .collect();

    let all_value = compute_all_value(def, &prefix);
    let all_lit = if is_64 {
        Literal::u64_suffixed(all_value)
    } else {
        Literal::u32_suffixed(all_value as u32)
    };

    let mut seen_debug = HashSet::new();
    let debug_arms: Vec<TokenStream> = def
        .bits
        .iter()
        .filter_map(|b| {
            let rust_name = strip_bit_prefix(&b.name, &prefix)?;
            if !seen_debug.insert(rust_name) {
                return None;
            }
            emit_debug_check(b, &prefix)
        })
        .collect();

    let spec_link = format!(
        "[`{vk_name}`](https://registry.khronos.org/vulkan/specs/latest/man/html/{vk_name}.html)"
    );

    quote! {
        #[doc = #spec_link]
        #[repr(transparent)]
        #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
        #[doc(alias = #vk_name)]
        pub struct #name(#repr);

        impl #name {
            #[inline]
            pub const fn empty() -> Self { Self(#zero_lit) }

            #[inline]
            pub const fn from_raw(value: #repr) -> Self { Self(value) }

            #[inline]
            pub const fn as_raw(self) -> #repr { self.0 }

            #[inline]
            pub const fn is_empty(self) -> bool { self.0 == #zero_lit }

            #[inline]
            pub const fn contains(self, other: Self) -> bool {
                (self.0 & other.0) == other.0
            }

            #[inline]
            pub const fn all() -> Self { Self(#all_lit) }

            #(#constants)*
        }

        impl core::ops::BitOr for #name {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: Self) -> Self { Self(self.0 | rhs.0) }
        }

        impl core::ops::BitOrAssign for #name {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) { self.0 |= rhs.0; }
        }

        impl core::ops::BitAnd for #name {
            type Output = Self;
            #[inline]
            fn bitand(self, rhs: Self) -> Self { Self(self.0 & rhs.0) }
        }

        impl core::ops::BitAndAssign for #name {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) { self.0 &= rhs.0; }
        }

        impl core::ops::BitXor for #name {
            type Output = Self;
            #[inline]
            fn bitxor(self, rhs: Self) -> Self { Self(self.0 ^ rhs.0) }
        }

        impl core::ops::BitXorAssign for #name {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) { self.0 ^= rhs.0; }
        }

        impl core::ops::Not for #name {
            type Output = Self;
            #[inline]
            fn not(self) -> Self { Self(!self.0) }
        }

        impl core::fmt::Debug for #name {
            #[allow(unused_mut, unused_variables)]
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                let mut first = true;
                let mut remaining = self.0;
                #(#debug_arms)*
                if remaining != #zero_lit {
                    if !first { f.write_str(" | ")?; }
                    write!(f, "{:#x}", remaining)?;
                } else if first {
                    f.write_str("(empty)")?;
                }
                Ok(())
            }
        }
    }
}

fn emit_bit_constant(bit: &BitmaskBit, prefix: &str, is_64: bool) -> Option<TokenStream> {
    let rust_name = strip_bit_prefix(&bit.name, prefix)?;
    let ident = format_ident!("{}", rust_name);

    match &bit.value {
        BitmaskValue::Bitpos(pos) => {
            let bit_doc = format!("Bit {pos}.");
            let lit = if is_64 {
                let val = 1u64 << pos;
                Literal::u64_suffixed(val)
            } else {
                let val = 1u32 << pos;
                Literal::u32_suffixed(val)
            };
            Some(quote! {
                #[doc = #bit_doc]
                pub const #ident: Self = Self(#lit);
            })
        }
        BitmaskValue::Value(val) => {
            let lit = if is_64 {
                Literal::u64_suffixed(*val)
            } else {
                Literal::u32_suffixed(*val as u32)
            };
            Some(quote! { pub const #ident: Self = Self(#lit); })
        }
        BitmaskValue::Alias(alias) => {
            let alias_name = strip_bit_prefix(alias, prefix)?;
            if alias_name == rust_name {
                return None;
            }
            let alias_ident = format_ident!("{}", alias_name);
            Some(quote! { pub const #ident: Self = Self::#alias_ident; })
        }
    }
}

fn emit_debug_check(bit: &BitmaskBit, prefix: &str) -> Option<TokenStream> {
    let rust_name = strip_bit_prefix(&bit.name, prefix)?;

    // Only emit debug checks for actual bit values, not aliases or zero values.
    match &bit.value {
        BitmaskValue::Bitpos(_) => {}
        BitmaskValue::Value(v) if *v != 0 && v.is_power_of_two() => {}
        _ => return None,
    };

    let ident = format_ident!("{}", rust_name);
    Some(quote! {
        if remaining & Self::#ident.0 != 0 {
            if !first { f.write_str(" | ")?; }
            f.write_str(#rust_name)?;
            remaining &= !Self::#ident.0;
            first = false;
        }
    })
}

/// Compute the OR of all defined bit values for this bitmask.
fn compute_all_value(def: &BitmaskDef, prefix: &str) -> u64 {
    let mut seen = HashSet::new();
    let mut result: u64 = 0;
    for bit in &def.bits {
        let Some(rust_name) = strip_bit_prefix(&bit.name, prefix) else {
            continue;
        };
        if !seen.insert(rust_name) {
            continue;
        }
        match &bit.value {
            BitmaskValue::Bitpos(pos) => result |= 1u64 << pos,
            BitmaskValue::Value(val) => result |= val,
            BitmaskValue::Alias(_) => {}
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Name stripping
// ---------------------------------------------------------------------------

/// Compute prefix for bitmask bit names.
///
/// `PipelineStageFlagBits` → `VK_PIPELINE_STAGE_`
/// (strips `FlagBits`/`FlagBits2` and extension suffix before converting).
fn bitmask_bit_prefix(rust_type_name: &str) -> String {
    let base = strip_extension_suffix(rust_type_name);
    let base = base
        .strip_suffix("FlagBits2")
        .or_else(|| base.strip_suffix("FlagBits"))
        .unwrap_or(&base);
    format!("VK_{}_", base.to_shouty_snake_case())
}

/// Strip prefix from a C bit name like `VK_PIPELINE_STAGE_VERTEX_SHADER_BIT`.
/// Also removes `_BIT` / `_BIT_KHR` etc. from the result.
fn strip_bit_prefix(c_name: &str, prefix: &str) -> Option<String> {
    let stripped = if let Some(s) = c_name.strip_prefix(prefix) {
        s
    } else {
        c_name.strip_prefix("VK_")?
    };

    if stripped.is_empty() || stripped.contains("MAX_ENUM") {
        return None;
    }

    // Remove _BIT suffix (may appear before extension suffix).
    let mut result = stripped.to_string();
    for suffix in &["_BIT_KHR", "_BIT_EXT", "_BIT_NV", "_BIT_AMD", "_BIT"] {
        if result.ends_with(suffix) {
            result.truncate(result.len() - suffix.len());
            break;
        }
    }

    // Also strip trailing extension suffix from the result.
    result = strip_extension_suffix(&result);

    if result.is_empty() {
        return None;
    }

    if result.starts_with(|c: char| c.is_ascii_digit()) {
        Some(format!("_{result}"))
    } else {
        Some(result)
    }
}

#[cfg(test)]
fn assert_valid_rust(tokens: &TokenStream) {
    syn::parse2::<syn::File>(tokens.clone())
        .unwrap_or_else(|e| panic!("generated code is not valid Rust: {e}\n\n{tokens}"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::{BitmaskBit, BitmaskDef, BitmaskValue};

    fn make_bitmask(name: &str, bitwidth: u32, bits: Vec<(&str, u32)>) -> BitmaskDef {
        BitmaskDef {
            name: name.to_string(),
            flags_name: format!("{}Flags", name.strip_suffix("FlagBits").unwrap_or(name)),
            bitwidth,
            bits: bits
                .into_iter()
                .map(|(bname, pos)| BitmaskBit {
                    name: bname.to_string(),
                    value: BitmaskValue::Bitpos(pos),
                })
                .collect(),
        }
    }

    #[test]
    fn bitmask_emits_valid_rust() {
        let def = make_bitmask(
            "BufferUsageFlagBits",
            32,
            vec![
                ("VK_BUFFER_USAGE_TRANSFER_SRC_BIT", 0),
                ("VK_BUFFER_USAGE_TRANSFER_DST_BIT", 1),
                ("VK_BUFFER_USAGE_VERTEX_BUFFER_BIT", 7),
            ],
        );
        let tokens = emit_bitmask(&def);
        assert_valid_rust(&tokens);
    }

    #[test]
    fn bitmask_has_expected_constants() {
        let def = make_bitmask(
            "BufferUsageFlagBits",
            32,
            vec![
                ("VK_BUFFER_USAGE_TRANSFER_SRC_BIT", 0),
                ("VK_BUFFER_USAGE_VERTEX_BUFFER_BIT", 7),
            ],
        );
        let code = emit_bitmask(&def).to_string();
        assert!(
            code.contains("pub const TRANSFER_SRC"),
            "missing TRANSFER_SRC"
        );
        assert!(
            code.contains("pub const VERTEX_BUFFER"),
            "missing VERTEX_BUFFER"
        );
    }

    #[test]
    fn bitmask_has_bitwise_operators() {
        let def = make_bitmask("CullModeFlagBits", 32, vec![("VK_CULL_MODE_FRONT_BIT", 0)]);
        let code = emit_bitmask(&def).to_string();
        assert!(code.contains("impl core :: ops :: BitOr for"));
        assert!(code.contains("impl core :: ops :: BitAnd for"));
        assert!(code.contains("impl core :: ops :: BitXor for"));
        assert!(code.contains("impl core :: ops :: Not for"));
        assert!(code.contains("impl core :: ops :: BitOrAssign for"));
    }

    #[test]
    fn bitmask_has_contains_and_is_empty() {
        let def = make_bitmask("CullModeFlagBits", 32, vec![]);
        let code = emit_bitmask(&def).to_string();
        assert!(code.contains("fn contains"));
        assert!(code.contains("fn is_empty"));
        assert!(code.contains("fn empty"));
    }

    #[test]
    fn bitmask_64bit_uses_u64() {
        let def = make_bitmask(
            "PipelineStageFlagBits2",
            64,
            vec![("VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT", 0)],
        );
        let code = emit_bitmask(&def).to_string();
        assert!(code.contains("(u64)"), "expected u64 inner type");
    }

    #[test]
    fn bitmask_empty_emits_valid_rust() {
        let def = make_bitmask("PipelineLayoutCreateFlagBits", 32, vec![]);
        let tokens = emit_bitmask(&def);
        assert_valid_rust(&tokens);
    }

    #[test]
    fn bitmask_dedup_alias() {
        let def = BitmaskDef {
            name: "AccessFlagBits".to_string(),
            flags_name: "AccessFlags".to_string(),
            bitwidth: 32,
            bits: vec![
                BitmaskBit {
                    name: "VK_ACCESS_SHADER_READ_BIT".to_string(),
                    value: BitmaskValue::Bitpos(5),
                },
                BitmaskBit {
                    name: "VK_ACCESS_SHADER_READ_BIT_KHR".to_string(),
                    value: BitmaskValue::Alias("VK_ACCESS_SHADER_READ_BIT".to_string()),
                },
            ],
        };
        let tokens = emit_bitmask(&def);
        assert_valid_rust(&tokens);
        // Should only have one SHADER_READ constant (deduped).
        let code = tokens.to_string();
        assert_eq!(
            code.matches("pub const SHADER_READ").count(),
            1,
            "expected exactly one SHADER_READ constant"
        );
    }

    #[test]
    fn test_bitmask_bit_prefix() {
        assert_eq!(
            bitmask_bit_prefix("PipelineStageFlagBits"),
            "VK_PIPELINE_STAGE_"
        );
        assert_eq!(
            bitmask_bit_prefix("PipelineStageFlagBits2"),
            "VK_PIPELINE_STAGE_"
        );
        assert_eq!(
            bitmask_bit_prefix("DebugReportFlagBitsEXT"),
            "VK_DEBUG_REPORT_"
        );
    }

    #[test]
    fn test_strip_bit_prefix() {
        let prefix = "VK_PIPELINE_STAGE_";
        assert_eq!(
            strip_bit_prefix("VK_PIPELINE_STAGE_VERTEX_SHADER_BIT", prefix),
            Some("VERTEX_SHADER".to_string())
        );
        assert_eq!(
            strip_bit_prefix("VK_PIPELINE_STAGE_TRANSFER_BIT_KHR", prefix),
            Some("TRANSFER".to_string())
        );
    }
}
