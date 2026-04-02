///[`VkAccelerationStructureCreateFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureCreateFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkAccelerationStructureCreateFlagBitsKHR")]
pub struct AccelerationStructureCreateFlagBitsKHR(u32);
impl AccelerationStructureCreateFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const DEVICE_ADDRESS_CAPTURE_REPLAY: Self = Self(1u32);
    ///Bit 3.
    pub const DESCRIPTOR_BUFFER_CAPTURE_REPLAY: Self = Self(8u32);
    ///Bit 2.
    pub const MOTION: Self = Self(4u32);
}
impl core::ops::BitOr for AccelerationStructureCreateFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for AccelerationStructureCreateFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for AccelerationStructureCreateFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for AccelerationStructureCreateFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for AccelerationStructureCreateFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for AccelerationStructureCreateFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for AccelerationStructureCreateFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for AccelerationStructureCreateFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::DEVICE_ADDRESS_CAPTURE_REPLAY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEVICE_ADDRESS_CAPTURE_REPLAY")?;
            remaining &= !Self::DEVICE_ADDRESS_CAPTURE_REPLAY.0;
            first = false;
        }
        if remaining & Self::DESCRIPTOR_BUFFER_CAPTURE_REPLAY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DESCRIPTOR_BUFFER_CAPTURE_REPLAY")?;
            remaining &= !Self::DESCRIPTOR_BUFFER_CAPTURE_REPLAY.0;
            first = false;
        }
        if remaining & Self::MOTION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MOTION")?;
            remaining &= !Self::MOTION.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkAccessFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccessFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkAccessFlagBits")]
pub struct AccessFlagBits(u32);
impl AccessFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const INDIRECT_COMMAND_READ: Self = Self(1u32);
    ///Bit 1.
    pub const INDEX_READ: Self = Self(2u32);
    ///Bit 2.
    pub const VERTEX_ATTRIBUTE_READ: Self = Self(4u32);
    ///Bit 3.
    pub const UNIFORM_READ: Self = Self(8u32);
    ///Bit 4.
    pub const INPUT_ATTACHMENT_READ: Self = Self(16u32);
    ///Bit 5.
    pub const SHADER_READ: Self = Self(32u32);
    ///Bit 6.
    pub const SHADER_WRITE: Self = Self(64u32);
    ///Bit 7.
    pub const COLOR_ATTACHMENT_READ: Self = Self(128u32);
    ///Bit 8.
    pub const COLOR_ATTACHMENT_WRITE: Self = Self(256u32);
    ///Bit 9.
    pub const DEPTH_STENCIL_ATTACHMENT_READ: Self = Self(512u32);
    ///Bit 10.
    pub const DEPTH_STENCIL_ATTACHMENT_WRITE: Self = Self(1024u32);
    ///Bit 11.
    pub const TRANSFER_READ: Self = Self(2048u32);
    ///Bit 12.
    pub const TRANSFER_WRITE: Self = Self(4096u32);
    ///Bit 13.
    pub const HOST_READ: Self = Self(8192u32);
    ///Bit 14.
    pub const HOST_WRITE: Self = Self(16384u32);
    ///Bit 15.
    pub const MEMORY_READ: Self = Self(32768u32);
    ///Bit 16.
    pub const MEMORY_WRITE: Self = Self(65536u32);
    pub const NONE: Self = Self(0u32);
    ///Bit 25.
    pub const TRANSFORM_FEEDBACK_WRITE: Self = Self(33554432u32);
    ///Bit 26.
    pub const TRANSFORM_FEEDBACK_COUNTER_READ: Self = Self(67108864u32);
    ///Bit 27.
    pub const TRANSFORM_FEEDBACK_COUNTER_WRITE: Self = Self(134217728u32);
    ///Bit 20.
    pub const CONDITIONAL_RENDERING_READ: Self = Self(1048576u32);
    ///Bit 19.
    pub const COLOR_ATTACHMENT_READ_NONCOHERENT: Self = Self(524288u32);
    ///Bit 21.
    pub const ACCELERATION_STRUCTURE_READ: Self = Self(2097152u32);
    ///Bit 22.
    pub const ACCELERATION_STRUCTURE_WRITE: Self = Self(4194304u32);
    pub const SHADING_RATE_IMAGE_READ: Self = Self::FRAGMENT_SHADING_RATE_ATTACHMENT_READ;
    ///Bit 24.
    pub const FRAGMENT_DENSITY_MAP_READ: Self = Self(16777216u32);
    ///Bit 23.
    pub const FRAGMENT_SHADING_RATE_ATTACHMENT_READ: Self = Self(8388608u32);
}
impl core::ops::BitOr for AccessFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for AccessFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for AccessFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for AccessFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for AccessFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for AccessFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for AccessFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for AccessFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::INDIRECT_COMMAND_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INDIRECT_COMMAND_READ")?;
            remaining &= !Self::INDIRECT_COMMAND_READ.0;
            first = false;
        }
        if remaining & Self::INDEX_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INDEX_READ")?;
            remaining &= !Self::INDEX_READ.0;
            first = false;
        }
        if remaining & Self::VERTEX_ATTRIBUTE_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VERTEX_ATTRIBUTE_READ")?;
            remaining &= !Self::VERTEX_ATTRIBUTE_READ.0;
            first = false;
        }
        if remaining & Self::UNIFORM_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("UNIFORM_READ")?;
            remaining &= !Self::UNIFORM_READ.0;
            first = false;
        }
        if remaining & Self::INPUT_ATTACHMENT_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INPUT_ATTACHMENT_READ")?;
            remaining &= !Self::INPUT_ATTACHMENT_READ.0;
            first = false;
        }
        if remaining & Self::SHADER_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SHADER_READ")?;
            remaining &= !Self::SHADER_READ.0;
            first = false;
        }
        if remaining & Self::SHADER_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SHADER_WRITE")?;
            remaining &= !Self::SHADER_WRITE.0;
            first = false;
        }
        if remaining & Self::COLOR_ATTACHMENT_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("COLOR_ATTACHMENT_READ")?;
            remaining &= !Self::COLOR_ATTACHMENT_READ.0;
            first = false;
        }
        if remaining & Self::COLOR_ATTACHMENT_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("COLOR_ATTACHMENT_WRITE")?;
            remaining &= !Self::COLOR_ATTACHMENT_WRITE.0;
            first = false;
        }
        if remaining & Self::DEPTH_STENCIL_ATTACHMENT_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEPTH_STENCIL_ATTACHMENT_READ")?;
            remaining &= !Self::DEPTH_STENCIL_ATTACHMENT_READ.0;
            first = false;
        }
        if remaining & Self::DEPTH_STENCIL_ATTACHMENT_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEPTH_STENCIL_ATTACHMENT_WRITE")?;
            remaining &= !Self::DEPTH_STENCIL_ATTACHMENT_WRITE.0;
            first = false;
        }
        if remaining & Self::TRANSFER_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFER_READ")?;
            remaining &= !Self::TRANSFER_READ.0;
            first = false;
        }
        if remaining & Self::TRANSFER_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFER_WRITE")?;
            remaining &= !Self::TRANSFER_WRITE.0;
            first = false;
        }
        if remaining & Self::HOST_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("HOST_READ")?;
            remaining &= !Self::HOST_READ.0;
            first = false;
        }
        if remaining & Self::HOST_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("HOST_WRITE")?;
            remaining &= !Self::HOST_WRITE.0;
            first = false;
        }
        if remaining & Self::MEMORY_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MEMORY_READ")?;
            remaining &= !Self::MEMORY_READ.0;
            first = false;
        }
        if remaining & Self::MEMORY_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MEMORY_WRITE")?;
            remaining &= !Self::MEMORY_WRITE.0;
            first = false;
        }
        if remaining & Self::TRANSFORM_FEEDBACK_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFORM_FEEDBACK_WRITE")?;
            remaining &= !Self::TRANSFORM_FEEDBACK_WRITE.0;
            first = false;
        }
        if remaining & Self::TRANSFORM_FEEDBACK_COUNTER_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFORM_FEEDBACK_COUNTER_READ")?;
            remaining &= !Self::TRANSFORM_FEEDBACK_COUNTER_READ.0;
            first = false;
        }
        if remaining & Self::TRANSFORM_FEEDBACK_COUNTER_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFORM_FEEDBACK_COUNTER_WRITE")?;
            remaining &= !Self::TRANSFORM_FEEDBACK_COUNTER_WRITE.0;
            first = false;
        }
        if remaining & Self::CONDITIONAL_RENDERING_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CONDITIONAL_RENDERING_READ")?;
            remaining &= !Self::CONDITIONAL_RENDERING_READ.0;
            first = false;
        }
        if remaining & Self::COLOR_ATTACHMENT_READ_NONCOHERENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("COLOR_ATTACHMENT_READ_NONCOHERENT")?;
            remaining &= !Self::COLOR_ATTACHMENT_READ_NONCOHERENT.0;
            first = false;
        }
        if remaining & Self::ACCELERATION_STRUCTURE_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ACCELERATION_STRUCTURE_READ")?;
            remaining &= !Self::ACCELERATION_STRUCTURE_READ.0;
            first = false;
        }
        if remaining & Self::ACCELERATION_STRUCTURE_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ACCELERATION_STRUCTURE_WRITE")?;
            remaining &= !Self::ACCELERATION_STRUCTURE_WRITE.0;
            first = false;
        }
        if remaining & Self::FRAGMENT_DENSITY_MAP_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRAGMENT_DENSITY_MAP_READ")?;
            remaining &= !Self::FRAGMENT_DENSITY_MAP_READ.0;
            first = false;
        }
        if remaining & Self::FRAGMENT_SHADING_RATE_ATTACHMENT_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRAGMENT_SHADING_RATE_ATTACHMENT_READ")?;
            remaining &= !Self::FRAGMENT_SHADING_RATE_ATTACHMENT_READ.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkAccessFlagBits2`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccessFlagBits2.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkAccessFlagBits2")]
pub struct AccessFlagBits2(u64);
impl AccessFlagBits2 {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u64)
    }
    #[inline]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u64
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const _2_NONE: Self = Self(0u64);
    ///Bit 0.
    pub const _2_INDIRECT_COMMAND_READ: Self = Self(1u64);
    ///Bit 1.
    pub const _2_INDEX_READ: Self = Self(2u64);
    ///Bit 2.
    pub const _2_VERTEX_ATTRIBUTE_READ: Self = Self(4u64);
    ///Bit 3.
    pub const _2_UNIFORM_READ: Self = Self(8u64);
    ///Bit 4.
    pub const _2_INPUT_ATTACHMENT_READ: Self = Self(16u64);
    ///Bit 5.
    pub const _2_SHADER_READ: Self = Self(32u64);
    ///Bit 6.
    pub const _2_SHADER_WRITE: Self = Self(64u64);
    ///Bit 7.
    pub const _2_COLOR_ATTACHMENT_READ: Self = Self(128u64);
    ///Bit 8.
    pub const _2_COLOR_ATTACHMENT_WRITE: Self = Self(256u64);
    ///Bit 9.
    pub const _2_DEPTH_STENCIL_ATTACHMENT_READ: Self = Self(512u64);
    ///Bit 10.
    pub const _2_DEPTH_STENCIL_ATTACHMENT_WRITE: Self = Self(1024u64);
    ///Bit 11.
    pub const _2_TRANSFER_READ: Self = Self(2048u64);
    ///Bit 12.
    pub const _2_TRANSFER_WRITE: Self = Self(4096u64);
    ///Bit 13.
    pub const _2_HOST_READ: Self = Self(8192u64);
    ///Bit 14.
    pub const _2_HOST_WRITE: Self = Self(16384u64);
    ///Bit 15.
    pub const _2_MEMORY_READ: Self = Self(32768u64);
    ///Bit 16.
    pub const _2_MEMORY_WRITE: Self = Self(65536u64);
    ///Bit 32.
    pub const _2_SHADER_SAMPLED_READ: Self = Self(4294967296u64);
    ///Bit 33.
    pub const _2_SHADER_STORAGE_READ: Self = Self(8589934592u64);
    ///Bit 34.
    pub const _2_SHADER_STORAGE_WRITE: Self = Self(17179869184u64);
    ///Bit 35.
    pub const _2_VIDEO_DECODE_READ: Self = Self(34359738368u64);
    ///Bit 36.
    pub const _2_VIDEO_DECODE_WRITE: Self = Self(68719476736u64);
    ///Bit 57.
    pub const _2_SAMPLER_HEAP_READ: Self = Self(144115188075855872u64);
    ///Bit 58.
    pub const _2_RESOURCE_HEAP_READ: Self = Self(288230376151711744u64);
    ///Bit 37.
    pub const _2_VIDEO_ENCODE_READ: Self = Self(137438953472u64);
    ///Bit 38.
    pub const _2_VIDEO_ENCODE_WRITE: Self = Self(274877906944u64);
    ///Bit 51.
    pub const _2_SHADER_TILE_ATTACHMENT_READ_BIT: Self = Self(2251799813685248u64);
    ///Bit 52.
    pub const _2_SHADER_TILE_ATTACHMENT_WRITE_BIT: Self = Self(4503599627370496u64);
    ///Bit 25.
    pub const _2_TRANSFORM_FEEDBACK_WRITE: Self = Self(33554432u64);
    ///Bit 26.
    pub const _2_TRANSFORM_FEEDBACK_COUNTER_READ: Self = Self(67108864u64);
    ///Bit 27.
    pub const _2_TRANSFORM_FEEDBACK_COUNTER_WRITE: Self = Self(134217728u64);
    ///Bit 20.
    pub const _2_CONDITIONAL_RENDERING_READ: Self = Self(1048576u64);
    ///Bit 23.
    pub const _2_FRAGMENT_SHADING_RATE_ATTACHMENT_READ: Self = Self(8388608u64);
    pub const _2_SHADING_RATE_IMAGE_READ: Self = Self::_2_FRAGMENT_SHADING_RATE_ATTACHMENT_READ;
    ///Bit 21.
    pub const _2_ACCELERATION_STRUCTURE_READ: Self = Self(2097152u64);
    ///Bit 22.
    pub const _2_ACCELERATION_STRUCTURE_WRITE: Self = Self(4194304u64);
    ///Bit 24.
    pub const _2_FRAGMENT_DENSITY_MAP_READ: Self = Self(16777216u64);
    ///Bit 19.
    pub const _2_COLOR_ATTACHMENT_READ_NONCOHERENT: Self = Self(524288u64);
    ///Bit 41.
    pub const _2_DESCRIPTOR_BUFFER_READ: Self = Self(2199023255552u64);
    ///Bit 39.
    pub const _2_INVOCATION_MASK_READ_BIT: Self = Self(549755813888u64);
    ///Bit 40.
    pub const _2_SHADER_BINDING_TABLE_READ: Self = Self(1099511627776u64);
    ///Bit 44.
    pub const _2_MICROMAP_READ: Self = Self(17592186044416u64);
    ///Bit 45.
    pub const _2_MICROMAP_WRITE: Self = Self(35184372088832u64);
    ///Bit 42.
    pub const _2_OPTICAL_FLOW_READ: Self = Self(4398046511104u64);
    ///Bit 43.
    pub const _2_OPTICAL_FLOW_WRITE: Self = Self(8796093022208u64);
    ///Bit 47.
    pub const _2_DATA_GRAPH_READ_BIT: Self = Self(140737488355328u64);
    ///Bit 48.
    pub const _2_DATA_GRAPH_WRITE_BIT: Self = Self(281474976710656u64);
    ///Bit 55.
    pub const _2_MEMORY_DECOMPRESSION_READ: Self = Self(36028797018963968u64);
    ///Bit 56.
    pub const _2_MEMORY_DECOMPRESSION_WRITE: Self = Self(72057594037927936u64);
}
impl core::ops::BitOr for AccessFlagBits2 {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for AccessFlagBits2 {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for AccessFlagBits2 {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for AccessFlagBits2 {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for AccessFlagBits2 {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for AccessFlagBits2 {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for AccessFlagBits2 {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for AccessFlagBits2 {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::_2_INDIRECT_COMMAND_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_INDIRECT_COMMAND_READ")?;
            remaining &= !Self::_2_INDIRECT_COMMAND_READ.0;
            first = false;
        }
        if remaining & Self::_2_INDEX_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_INDEX_READ")?;
            remaining &= !Self::_2_INDEX_READ.0;
            first = false;
        }
        if remaining & Self::_2_VERTEX_ATTRIBUTE_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VERTEX_ATTRIBUTE_READ")?;
            remaining &= !Self::_2_VERTEX_ATTRIBUTE_READ.0;
            first = false;
        }
        if remaining & Self::_2_UNIFORM_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_UNIFORM_READ")?;
            remaining &= !Self::_2_UNIFORM_READ.0;
            first = false;
        }
        if remaining & Self::_2_INPUT_ATTACHMENT_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_INPUT_ATTACHMENT_READ")?;
            remaining &= !Self::_2_INPUT_ATTACHMENT_READ.0;
            first = false;
        }
        if remaining & Self::_2_SHADER_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_SHADER_READ")?;
            remaining &= !Self::_2_SHADER_READ.0;
            first = false;
        }
        if remaining & Self::_2_SHADER_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_SHADER_WRITE")?;
            remaining &= !Self::_2_SHADER_WRITE.0;
            first = false;
        }
        if remaining & Self::_2_COLOR_ATTACHMENT_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_COLOR_ATTACHMENT_READ")?;
            remaining &= !Self::_2_COLOR_ATTACHMENT_READ.0;
            first = false;
        }
        if remaining & Self::_2_COLOR_ATTACHMENT_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_COLOR_ATTACHMENT_WRITE")?;
            remaining &= !Self::_2_COLOR_ATTACHMENT_WRITE.0;
            first = false;
        }
        if remaining & Self::_2_DEPTH_STENCIL_ATTACHMENT_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_DEPTH_STENCIL_ATTACHMENT_READ")?;
            remaining &= !Self::_2_DEPTH_STENCIL_ATTACHMENT_READ.0;
            first = false;
        }
        if remaining & Self::_2_DEPTH_STENCIL_ATTACHMENT_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_DEPTH_STENCIL_ATTACHMENT_WRITE")?;
            remaining &= !Self::_2_DEPTH_STENCIL_ATTACHMENT_WRITE.0;
            first = false;
        }
        if remaining & Self::_2_TRANSFER_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_TRANSFER_READ")?;
            remaining &= !Self::_2_TRANSFER_READ.0;
            first = false;
        }
        if remaining & Self::_2_TRANSFER_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_TRANSFER_WRITE")?;
            remaining &= !Self::_2_TRANSFER_WRITE.0;
            first = false;
        }
        if remaining & Self::_2_HOST_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_HOST_READ")?;
            remaining &= !Self::_2_HOST_READ.0;
            first = false;
        }
        if remaining & Self::_2_HOST_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_HOST_WRITE")?;
            remaining &= !Self::_2_HOST_WRITE.0;
            first = false;
        }
        if remaining & Self::_2_MEMORY_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_MEMORY_READ")?;
            remaining &= !Self::_2_MEMORY_READ.0;
            first = false;
        }
        if remaining & Self::_2_MEMORY_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_MEMORY_WRITE")?;
            remaining &= !Self::_2_MEMORY_WRITE.0;
            first = false;
        }
        if remaining & Self::_2_SHADER_SAMPLED_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_SHADER_SAMPLED_READ")?;
            remaining &= !Self::_2_SHADER_SAMPLED_READ.0;
            first = false;
        }
        if remaining & Self::_2_SHADER_STORAGE_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_SHADER_STORAGE_READ")?;
            remaining &= !Self::_2_SHADER_STORAGE_READ.0;
            first = false;
        }
        if remaining & Self::_2_SHADER_STORAGE_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_SHADER_STORAGE_WRITE")?;
            remaining &= !Self::_2_SHADER_STORAGE_WRITE.0;
            first = false;
        }
        if remaining & Self::_2_VIDEO_DECODE_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VIDEO_DECODE_READ")?;
            remaining &= !Self::_2_VIDEO_DECODE_READ.0;
            first = false;
        }
        if remaining & Self::_2_VIDEO_DECODE_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VIDEO_DECODE_WRITE")?;
            remaining &= !Self::_2_VIDEO_DECODE_WRITE.0;
            first = false;
        }
        if remaining & Self::_2_SAMPLER_HEAP_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_SAMPLER_HEAP_READ")?;
            remaining &= !Self::_2_SAMPLER_HEAP_READ.0;
            first = false;
        }
        if remaining & Self::_2_RESOURCE_HEAP_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_RESOURCE_HEAP_READ")?;
            remaining &= !Self::_2_RESOURCE_HEAP_READ.0;
            first = false;
        }
        if remaining & Self::_2_VIDEO_ENCODE_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VIDEO_ENCODE_READ")?;
            remaining &= !Self::_2_VIDEO_ENCODE_READ.0;
            first = false;
        }
        if remaining & Self::_2_VIDEO_ENCODE_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VIDEO_ENCODE_WRITE")?;
            remaining &= !Self::_2_VIDEO_ENCODE_WRITE.0;
            first = false;
        }
        if remaining & Self::_2_SHADER_TILE_ATTACHMENT_READ_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_SHADER_TILE_ATTACHMENT_READ_BIT")?;
            remaining &= !Self::_2_SHADER_TILE_ATTACHMENT_READ_BIT.0;
            first = false;
        }
        if remaining & Self::_2_SHADER_TILE_ATTACHMENT_WRITE_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_SHADER_TILE_ATTACHMENT_WRITE_BIT")?;
            remaining &= !Self::_2_SHADER_TILE_ATTACHMENT_WRITE_BIT.0;
            first = false;
        }
        if remaining & Self::_2_TRANSFORM_FEEDBACK_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_TRANSFORM_FEEDBACK_WRITE")?;
            remaining &= !Self::_2_TRANSFORM_FEEDBACK_WRITE.0;
            first = false;
        }
        if remaining & Self::_2_TRANSFORM_FEEDBACK_COUNTER_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_TRANSFORM_FEEDBACK_COUNTER_READ")?;
            remaining &= !Self::_2_TRANSFORM_FEEDBACK_COUNTER_READ.0;
            first = false;
        }
        if remaining & Self::_2_TRANSFORM_FEEDBACK_COUNTER_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_TRANSFORM_FEEDBACK_COUNTER_WRITE")?;
            remaining &= !Self::_2_TRANSFORM_FEEDBACK_COUNTER_WRITE.0;
            first = false;
        }
        if remaining & Self::_2_CONDITIONAL_RENDERING_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_CONDITIONAL_RENDERING_READ")?;
            remaining &= !Self::_2_CONDITIONAL_RENDERING_READ.0;
            first = false;
        }
        if remaining & Self::_2_FRAGMENT_SHADING_RATE_ATTACHMENT_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_FRAGMENT_SHADING_RATE_ATTACHMENT_READ")?;
            remaining &= !Self::_2_FRAGMENT_SHADING_RATE_ATTACHMENT_READ.0;
            first = false;
        }
        if remaining & Self::_2_ACCELERATION_STRUCTURE_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_ACCELERATION_STRUCTURE_READ")?;
            remaining &= !Self::_2_ACCELERATION_STRUCTURE_READ.0;
            first = false;
        }
        if remaining & Self::_2_ACCELERATION_STRUCTURE_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_ACCELERATION_STRUCTURE_WRITE")?;
            remaining &= !Self::_2_ACCELERATION_STRUCTURE_WRITE.0;
            first = false;
        }
        if remaining & Self::_2_FRAGMENT_DENSITY_MAP_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_FRAGMENT_DENSITY_MAP_READ")?;
            remaining &= !Self::_2_FRAGMENT_DENSITY_MAP_READ.0;
            first = false;
        }
        if remaining & Self::_2_COLOR_ATTACHMENT_READ_NONCOHERENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_COLOR_ATTACHMENT_READ_NONCOHERENT")?;
            remaining &= !Self::_2_COLOR_ATTACHMENT_READ_NONCOHERENT.0;
            first = false;
        }
        if remaining & Self::_2_DESCRIPTOR_BUFFER_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_DESCRIPTOR_BUFFER_READ")?;
            remaining &= !Self::_2_DESCRIPTOR_BUFFER_READ.0;
            first = false;
        }
        if remaining & Self::_2_INVOCATION_MASK_READ_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_INVOCATION_MASK_READ_BIT")?;
            remaining &= !Self::_2_INVOCATION_MASK_READ_BIT.0;
            first = false;
        }
        if remaining & Self::_2_SHADER_BINDING_TABLE_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_SHADER_BINDING_TABLE_READ")?;
            remaining &= !Self::_2_SHADER_BINDING_TABLE_READ.0;
            first = false;
        }
        if remaining & Self::_2_MICROMAP_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_MICROMAP_READ")?;
            remaining &= !Self::_2_MICROMAP_READ.0;
            first = false;
        }
        if remaining & Self::_2_MICROMAP_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_MICROMAP_WRITE")?;
            remaining &= !Self::_2_MICROMAP_WRITE.0;
            first = false;
        }
        if remaining & Self::_2_OPTICAL_FLOW_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_OPTICAL_FLOW_READ")?;
            remaining &= !Self::_2_OPTICAL_FLOW_READ.0;
            first = false;
        }
        if remaining & Self::_2_OPTICAL_FLOW_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_OPTICAL_FLOW_WRITE")?;
            remaining &= !Self::_2_OPTICAL_FLOW_WRITE.0;
            first = false;
        }
        if remaining & Self::_2_DATA_GRAPH_READ_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_DATA_GRAPH_READ_BIT")?;
            remaining &= !Self::_2_DATA_GRAPH_READ_BIT.0;
            first = false;
        }
        if remaining & Self::_2_DATA_GRAPH_WRITE_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_DATA_GRAPH_WRITE_BIT")?;
            remaining &= !Self::_2_DATA_GRAPH_WRITE_BIT.0;
            first = false;
        }
        if remaining & Self::_2_MEMORY_DECOMPRESSION_READ.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_MEMORY_DECOMPRESSION_READ")?;
            remaining &= !Self::_2_MEMORY_DECOMPRESSION_READ.0;
            first = false;
        }
        if remaining & Self::_2_MEMORY_DECOMPRESSION_WRITE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_MEMORY_DECOMPRESSION_WRITE")?;
            remaining &= !Self::_2_MEMORY_DECOMPRESSION_WRITE.0;
            first = false;
        }
        if remaining != 0u64 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkAccessFlagBits3KHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccessFlagBits3KHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkAccessFlagBits3KHR")]
pub struct AccessFlagBits3KHR(u64);
impl AccessFlagBits3KHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u64)
    }
    #[inline]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u64
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const ACCESS_3_NONE: Self = Self(0u64);
}
impl core::ops::BitOr for AccessFlagBits3KHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for AccessFlagBits3KHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for AccessFlagBits3KHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for AccessFlagBits3KHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for AccessFlagBits3KHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for AccessFlagBits3KHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for AccessFlagBits3KHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for AccessFlagBits3KHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining != 0u64 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkAcquireProfilingLockFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkAcquireProfilingLockFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkAcquireProfilingLockFlagBitsKHR")]
pub struct AcquireProfilingLockFlagBitsKHR(u32);
impl AcquireProfilingLockFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}
impl core::ops::BitOr for AcquireProfilingLockFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for AcquireProfilingLockFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for AcquireProfilingLockFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for AcquireProfilingLockFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for AcquireProfilingLockFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for AcquireProfilingLockFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for AcquireProfilingLockFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for AcquireProfilingLockFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkAddressCommandFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkAddressCommandFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkAddressCommandFlagBitsKHR")]
pub struct AddressCommandFlagBitsKHR(u32);
impl AddressCommandFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const PROTECTED: Self = Self(1u32);
    ///Bit 1.
    pub const FULLY_BOUND: Self = Self(2u32);
    ///Bit 2.
    pub const STORAGE_BUFFER_USAGE: Self = Self(4u32);
    ///Bit 3.
    pub const UNKNOWN_STORAGE_BUFFER_USAGE: Self = Self(8u32);
    ///Bit 4.
    pub const TRANSFORM_FEEDBACK_BUFFER_USAGE: Self = Self(16u32);
    ///Bit 5.
    pub const UNKNOWN_TRANSFORM_FEEDBACK_BUFFER_USAGE: Self = Self(32u32);
}
impl core::ops::BitOr for AddressCommandFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for AddressCommandFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for AddressCommandFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for AddressCommandFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for AddressCommandFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for AddressCommandFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for AddressCommandFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for AddressCommandFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::PROTECTED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PROTECTED")?;
            remaining &= !Self::PROTECTED.0;
            first = false;
        }
        if remaining & Self::FULLY_BOUND.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FULLY_BOUND")?;
            remaining &= !Self::FULLY_BOUND.0;
            first = false;
        }
        if remaining & Self::STORAGE_BUFFER_USAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("STORAGE_BUFFER_USAGE")?;
            remaining &= !Self::STORAGE_BUFFER_USAGE.0;
            first = false;
        }
        if remaining & Self::UNKNOWN_STORAGE_BUFFER_USAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("UNKNOWN_STORAGE_BUFFER_USAGE")?;
            remaining &= !Self::UNKNOWN_STORAGE_BUFFER_USAGE.0;
            first = false;
        }
        if remaining & Self::TRANSFORM_FEEDBACK_BUFFER_USAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFORM_FEEDBACK_BUFFER_USAGE")?;
            remaining &= !Self::TRANSFORM_FEEDBACK_BUFFER_USAGE.0;
            first = false;
        }
        if remaining & Self::UNKNOWN_TRANSFORM_FEEDBACK_BUFFER_USAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("UNKNOWN_TRANSFORM_FEEDBACK_BUFFER_USAGE")?;
            remaining &= !Self::UNKNOWN_TRANSFORM_FEEDBACK_BUFFER_USAGE.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkAddressCopyFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkAddressCopyFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkAddressCopyFlagBitsKHR")]
pub struct AddressCopyFlagBitsKHR(u32);
impl AddressCopyFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const DEVICE_LOCAL: Self = Self(1u32);
    ///Bit 1.
    pub const SPARSE: Self = Self(2u32);
    ///Bit 2.
    pub const PROTECTED: Self = Self(4u32);
}
impl core::ops::BitOr for AddressCopyFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for AddressCopyFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for AddressCopyFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for AddressCopyFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for AddressCopyFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for AddressCopyFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for AddressCopyFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for AddressCopyFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::DEVICE_LOCAL.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEVICE_LOCAL")?;
            remaining &= !Self::DEVICE_LOCAL.0;
            first = false;
        }
        if remaining & Self::SPARSE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SPARSE")?;
            remaining &= !Self::SPARSE.0;
            first = false;
        }
        if remaining & Self::PROTECTED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PROTECTED")?;
            remaining &= !Self::PROTECTED.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkAttachmentDescriptionFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkAttachmentDescriptionFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkAttachmentDescriptionFlagBits")]
pub struct AttachmentDescriptionFlagBits(u32);
impl AttachmentDescriptionFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const MAY_ALIAS: Self = Self(1u32);
    ///Bit 1.
    pub const RESOLVE_SKIP_TRANSFER_FUNCTION: Self = Self(2u32);
    ///Bit 2.
    pub const RESOLVE_ENABLE_TRANSFER_FUNCTION: Self = Self(4u32);
}
impl core::ops::BitOr for AttachmentDescriptionFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for AttachmentDescriptionFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for AttachmentDescriptionFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for AttachmentDescriptionFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for AttachmentDescriptionFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for AttachmentDescriptionFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for AttachmentDescriptionFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for AttachmentDescriptionFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::MAY_ALIAS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MAY_ALIAS")?;
            remaining &= !Self::MAY_ALIAS.0;
            first = false;
        }
        if remaining & Self::RESOLVE_SKIP_TRANSFER_FUNCTION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RESOLVE_SKIP_TRANSFER_FUNCTION")?;
            remaining &= !Self::RESOLVE_SKIP_TRANSFER_FUNCTION.0;
            first = false;
        }
        if remaining & Self::RESOLVE_ENABLE_TRANSFER_FUNCTION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RESOLVE_ENABLE_TRANSFER_FUNCTION")?;
            remaining &= !Self::RESOLVE_ENABLE_TRANSFER_FUNCTION.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkBufferCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkBufferCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkBufferCreateFlagBits")]
pub struct BufferCreateFlagBits(u32);
impl BufferCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const SPARSE_BINDING: Self = Self(1u32);
    ///Bit 1.
    pub const SPARSE_RESIDENCY: Self = Self(2u32);
    ///Bit 2.
    pub const SPARSE_ALIASED: Self = Self(4u32);
    ///Bit 3.
    pub const PROTECTED: Self = Self(8u32);
    ///Bit 4.
    pub const DEVICE_ADDRESS_CAPTURE_REPLAY: Self = Self(16u32);
    ///Bit 5.
    pub const DESCRIPTOR_BUFFER_CAPTURE_REPLAY: Self = Self(32u32);
    ///Bit 6.
    pub const VIDEO_PROFILE_INDEPENDENT: Self = Self(64u32);
}
impl core::ops::BitOr for BufferCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for BufferCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for BufferCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for BufferCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for BufferCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for BufferCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for BufferCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for BufferCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::SPARSE_BINDING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SPARSE_BINDING")?;
            remaining &= !Self::SPARSE_BINDING.0;
            first = false;
        }
        if remaining & Self::SPARSE_RESIDENCY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SPARSE_RESIDENCY")?;
            remaining &= !Self::SPARSE_RESIDENCY.0;
            first = false;
        }
        if remaining & Self::SPARSE_ALIASED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SPARSE_ALIASED")?;
            remaining &= !Self::SPARSE_ALIASED.0;
            first = false;
        }
        if remaining & Self::PROTECTED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PROTECTED")?;
            remaining &= !Self::PROTECTED.0;
            first = false;
        }
        if remaining & Self::DEVICE_ADDRESS_CAPTURE_REPLAY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEVICE_ADDRESS_CAPTURE_REPLAY")?;
            remaining &= !Self::DEVICE_ADDRESS_CAPTURE_REPLAY.0;
            first = false;
        }
        if remaining & Self::DESCRIPTOR_BUFFER_CAPTURE_REPLAY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DESCRIPTOR_BUFFER_CAPTURE_REPLAY")?;
            remaining &= !Self::DESCRIPTOR_BUFFER_CAPTURE_REPLAY.0;
            first = false;
        }
        if remaining & Self::VIDEO_PROFILE_INDEPENDENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VIDEO_PROFILE_INDEPENDENT")?;
            remaining &= !Self::VIDEO_PROFILE_INDEPENDENT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkBufferUsageFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkBufferUsageFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkBufferUsageFlagBits")]
pub struct BufferUsageFlagBits(u32);
impl BufferUsageFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const TRANSFER_SRC: Self = Self(1u32);
    ///Bit 1.
    pub const TRANSFER_DST: Self = Self(2u32);
    ///Bit 2.
    pub const UNIFORM_TEXEL_BUFFER: Self = Self(4u32);
    ///Bit 3.
    pub const STORAGE_TEXEL_BUFFER: Self = Self(8u32);
    ///Bit 4.
    pub const UNIFORM_BUFFER: Self = Self(16u32);
    ///Bit 5.
    pub const STORAGE_BUFFER: Self = Self(32u32);
    ///Bit 6.
    pub const INDEX_BUFFER: Self = Self(64u32);
    ///Bit 7.
    pub const VERTEX_BUFFER: Self = Self(128u32);
    ///Bit 8.
    pub const INDIRECT_BUFFER: Self = Self(256u32);
    ///Bit 17.
    pub const SHADER_DEVICE_ADDRESS: Self = Self(131072u32);
    ///Bit 13.
    pub const VIDEO_DECODE_SRC: Self = Self(8192u32);
    ///Bit 14.
    pub const VIDEO_DECODE_DST: Self = Self(16384u32);
    ///Bit 11.
    pub const TRANSFORM_FEEDBACK_BUFFER: Self = Self(2048u32);
    ///Bit 12.
    pub const TRANSFORM_FEEDBACK_COUNTER_BUFFER: Self = Self(4096u32);
    ///Bit 9.
    pub const CONDITIONAL_RENDERING: Self = Self(512u32);
    ///Bit 25.
    pub const EXECUTION_GRAPH_SCRATCH_BIT: Self = Self(33554432u32);
    ///Bit 28.
    pub const DESCRIPTOR_HEAP: Self = Self(268435456u32);
    ///Bit 19.
    pub const ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY: Self = Self(524288u32);
    ///Bit 20.
    pub const ACCELERATION_STRUCTURE_STORAGE: Self = Self(1048576u32);
    ///Bit 10.
    pub const SHADER_BINDING_TABLE: Self = Self(1024u32);
    pub const RAY_TRACING: Self = Self::SHADER_BINDING_TABLE;
    ///Bit 15.
    pub const VIDEO_ENCODE_DST: Self = Self(32768u32);
    ///Bit 16.
    pub const VIDEO_ENCODE_SRC: Self = Self(65536u32);
    ///Bit 21.
    pub const SAMPLER_DESCRIPTOR_BUFFER: Self = Self(2097152u32);
    ///Bit 22.
    pub const RESOURCE_DESCRIPTOR_BUFFER: Self = Self(4194304u32);
    ///Bit 26.
    pub const PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER: Self = Self(67108864u32);
    ///Bit 23.
    pub const MICROMAP_BUILD_INPUT_READ_ONLY: Self = Self(8388608u32);
    ///Bit 24.
    pub const MICROMAP_STORAGE: Self = Self(16777216u32);
    ///Bit 27.
    pub const TILE_MEMORY_BIT: Self = Self(134217728u32);
}
impl core::ops::BitOr for BufferUsageFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for BufferUsageFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for BufferUsageFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for BufferUsageFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for BufferUsageFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for BufferUsageFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for BufferUsageFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for BufferUsageFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::TRANSFER_SRC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFER_SRC")?;
            remaining &= !Self::TRANSFER_SRC.0;
            first = false;
        }
        if remaining & Self::TRANSFER_DST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFER_DST")?;
            remaining &= !Self::TRANSFER_DST.0;
            first = false;
        }
        if remaining & Self::UNIFORM_TEXEL_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("UNIFORM_TEXEL_BUFFER")?;
            remaining &= !Self::UNIFORM_TEXEL_BUFFER.0;
            first = false;
        }
        if remaining & Self::STORAGE_TEXEL_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("STORAGE_TEXEL_BUFFER")?;
            remaining &= !Self::STORAGE_TEXEL_BUFFER.0;
            first = false;
        }
        if remaining & Self::UNIFORM_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("UNIFORM_BUFFER")?;
            remaining &= !Self::UNIFORM_BUFFER.0;
            first = false;
        }
        if remaining & Self::STORAGE_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("STORAGE_BUFFER")?;
            remaining &= !Self::STORAGE_BUFFER.0;
            first = false;
        }
        if remaining & Self::INDEX_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INDEX_BUFFER")?;
            remaining &= !Self::INDEX_BUFFER.0;
            first = false;
        }
        if remaining & Self::VERTEX_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VERTEX_BUFFER")?;
            remaining &= !Self::VERTEX_BUFFER.0;
            first = false;
        }
        if remaining & Self::INDIRECT_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INDIRECT_BUFFER")?;
            remaining &= !Self::INDIRECT_BUFFER.0;
            first = false;
        }
        if remaining & Self::SHADER_DEVICE_ADDRESS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SHADER_DEVICE_ADDRESS")?;
            remaining &= !Self::SHADER_DEVICE_ADDRESS.0;
            first = false;
        }
        if remaining & Self::VIDEO_DECODE_SRC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VIDEO_DECODE_SRC")?;
            remaining &= !Self::VIDEO_DECODE_SRC.0;
            first = false;
        }
        if remaining & Self::VIDEO_DECODE_DST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VIDEO_DECODE_DST")?;
            remaining &= !Self::VIDEO_DECODE_DST.0;
            first = false;
        }
        if remaining & Self::TRANSFORM_FEEDBACK_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFORM_FEEDBACK_BUFFER")?;
            remaining &= !Self::TRANSFORM_FEEDBACK_BUFFER.0;
            first = false;
        }
        if remaining & Self::TRANSFORM_FEEDBACK_COUNTER_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFORM_FEEDBACK_COUNTER_BUFFER")?;
            remaining &= !Self::TRANSFORM_FEEDBACK_COUNTER_BUFFER.0;
            first = false;
        }
        if remaining & Self::CONDITIONAL_RENDERING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CONDITIONAL_RENDERING")?;
            remaining &= !Self::CONDITIONAL_RENDERING.0;
            first = false;
        }
        if remaining & Self::EXECUTION_GRAPH_SCRATCH_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("EXECUTION_GRAPH_SCRATCH_BIT")?;
            remaining &= !Self::EXECUTION_GRAPH_SCRATCH_BIT.0;
            first = false;
        }
        if remaining & Self::DESCRIPTOR_HEAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DESCRIPTOR_HEAP")?;
            remaining &= !Self::DESCRIPTOR_HEAP.0;
            first = false;
        }
        if remaining & Self::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY")?;
            remaining &= !Self::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY.0;
            first = false;
        }
        if remaining & Self::ACCELERATION_STRUCTURE_STORAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ACCELERATION_STRUCTURE_STORAGE")?;
            remaining &= !Self::ACCELERATION_STRUCTURE_STORAGE.0;
            first = false;
        }
        if remaining & Self::SHADER_BINDING_TABLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SHADER_BINDING_TABLE")?;
            remaining &= !Self::SHADER_BINDING_TABLE.0;
            first = false;
        }
        if remaining & Self::VIDEO_ENCODE_DST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VIDEO_ENCODE_DST")?;
            remaining &= !Self::VIDEO_ENCODE_DST.0;
            first = false;
        }
        if remaining & Self::VIDEO_ENCODE_SRC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VIDEO_ENCODE_SRC")?;
            remaining &= !Self::VIDEO_ENCODE_SRC.0;
            first = false;
        }
        if remaining & Self::SAMPLER_DESCRIPTOR_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SAMPLER_DESCRIPTOR_BUFFER")?;
            remaining &= !Self::SAMPLER_DESCRIPTOR_BUFFER.0;
            first = false;
        }
        if remaining & Self::RESOURCE_DESCRIPTOR_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RESOURCE_DESCRIPTOR_BUFFER")?;
            remaining &= !Self::RESOURCE_DESCRIPTOR_BUFFER.0;
            first = false;
        }
        if remaining & Self::PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER")?;
            remaining &= !Self::PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER.0;
            first = false;
        }
        if remaining & Self::MICROMAP_BUILD_INPUT_READ_ONLY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MICROMAP_BUILD_INPUT_READ_ONLY")?;
            remaining &= !Self::MICROMAP_BUILD_INPUT_READ_ONLY.0;
            first = false;
        }
        if remaining & Self::MICROMAP_STORAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MICROMAP_STORAGE")?;
            remaining &= !Self::MICROMAP_STORAGE.0;
            first = false;
        }
        if remaining & Self::TILE_MEMORY_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TILE_MEMORY_BIT")?;
            remaining &= !Self::TILE_MEMORY_BIT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkBufferUsageFlagBits2`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkBufferUsageFlagBits2.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkBufferUsageFlagBits2")]
pub struct BufferUsageFlagBits2(u64);
impl BufferUsageFlagBits2 {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u64)
    }
    #[inline]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u64
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const _2_TRANSFER_SRC: Self = Self(1u64);
    ///Bit 1.
    pub const _2_TRANSFER_DST: Self = Self(2u64);
    ///Bit 2.
    pub const _2_UNIFORM_TEXEL_BUFFER: Self = Self(4u64);
    ///Bit 3.
    pub const _2_STORAGE_TEXEL_BUFFER: Self = Self(8u64);
    ///Bit 4.
    pub const _2_UNIFORM_BUFFER: Self = Self(16u64);
    ///Bit 5.
    pub const _2_STORAGE_BUFFER: Self = Self(32u64);
    ///Bit 6.
    pub const _2_INDEX_BUFFER: Self = Self(64u64);
    ///Bit 7.
    pub const _2_VERTEX_BUFFER: Self = Self(128u64);
    ///Bit 8.
    pub const _2_INDIRECT_BUFFER: Self = Self(256u64);
    ///Bit 17.
    pub const _2_SHADER_DEVICE_ADDRESS: Self = Self(131072u64);
    ///Bit 25.
    pub const _2_EXECUTION_GRAPH_SCRATCH_BIT: Self = Self(33554432u64);
    ///Bit 28.
    pub const _2_DESCRIPTOR_HEAP: Self = Self(268435456u64);
    ///Bit 9.
    pub const _2_CONDITIONAL_RENDERING: Self = Self(512u64);
    ///Bit 10.
    pub const _2_SHADER_BINDING_TABLE: Self = Self(1024u64);
    pub const _2_RAY_TRACING: Self = Self::_2_SHADER_BINDING_TABLE;
    ///Bit 11.
    pub const _2_TRANSFORM_FEEDBACK_BUFFER: Self = Self(2048u64);
    ///Bit 12.
    pub const _2_TRANSFORM_FEEDBACK_COUNTER_BUFFER: Self = Self(4096u64);
    ///Bit 13.
    pub const _2_VIDEO_DECODE_SRC: Self = Self(8192u64);
    ///Bit 14.
    pub const _2_VIDEO_DECODE_DST: Self = Self(16384u64);
    ///Bit 15.
    pub const _2_VIDEO_ENCODE_DST: Self = Self(32768u64);
    ///Bit 16.
    pub const _2_VIDEO_ENCODE_SRC: Self = Self(65536u64);
    ///Bit 19.
    pub const _2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY: Self = Self(524288u64);
    ///Bit 20.
    pub const _2_ACCELERATION_STRUCTURE_STORAGE: Self = Self(1048576u64);
    ///Bit 21.
    pub const _2_SAMPLER_DESCRIPTOR_BUFFER: Self = Self(2097152u64);
    ///Bit 22.
    pub const _2_RESOURCE_DESCRIPTOR_BUFFER: Self = Self(4194304u64);
    ///Bit 26.
    pub const _2_PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER: Self = Self(67108864u64);
    ///Bit 23.
    pub const _2_MICROMAP_BUILD_INPUT_READ_ONLY: Self = Self(8388608u64);
    ///Bit 24.
    pub const _2_MICROMAP_STORAGE: Self = Self(16777216u64);
    ///Bit 33.
    pub const _2_COMPRESSED_DATA_DGF1_BIT: Self = Self(8589934592u64);
    ///Bit 29.
    pub const _2_DATA_GRAPH_FOREIGN_DESCRIPTOR_BIT: Self = Self(536870912u64);
    ///Bit 27.
    pub const _2_TILE_MEMORY_BIT: Self = Self(134217728u64);
    ///Bit 32.
    pub const _2_MEMORY_DECOMPRESSION: Self = Self(4294967296u64);
    ///Bit 31.
    pub const _2_PREPROCESS_BUFFER: Self = Self(2147483648u64);
}
impl core::ops::BitOr for BufferUsageFlagBits2 {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for BufferUsageFlagBits2 {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for BufferUsageFlagBits2 {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for BufferUsageFlagBits2 {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for BufferUsageFlagBits2 {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for BufferUsageFlagBits2 {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for BufferUsageFlagBits2 {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for BufferUsageFlagBits2 {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::_2_TRANSFER_SRC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_TRANSFER_SRC")?;
            remaining &= !Self::_2_TRANSFER_SRC.0;
            first = false;
        }
        if remaining & Self::_2_TRANSFER_DST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_TRANSFER_DST")?;
            remaining &= !Self::_2_TRANSFER_DST.0;
            first = false;
        }
        if remaining & Self::_2_UNIFORM_TEXEL_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_UNIFORM_TEXEL_BUFFER")?;
            remaining &= !Self::_2_UNIFORM_TEXEL_BUFFER.0;
            first = false;
        }
        if remaining & Self::_2_STORAGE_TEXEL_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_STORAGE_TEXEL_BUFFER")?;
            remaining &= !Self::_2_STORAGE_TEXEL_BUFFER.0;
            first = false;
        }
        if remaining & Self::_2_UNIFORM_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_UNIFORM_BUFFER")?;
            remaining &= !Self::_2_UNIFORM_BUFFER.0;
            first = false;
        }
        if remaining & Self::_2_STORAGE_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_STORAGE_BUFFER")?;
            remaining &= !Self::_2_STORAGE_BUFFER.0;
            first = false;
        }
        if remaining & Self::_2_INDEX_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_INDEX_BUFFER")?;
            remaining &= !Self::_2_INDEX_BUFFER.0;
            first = false;
        }
        if remaining & Self::_2_VERTEX_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VERTEX_BUFFER")?;
            remaining &= !Self::_2_VERTEX_BUFFER.0;
            first = false;
        }
        if remaining & Self::_2_INDIRECT_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_INDIRECT_BUFFER")?;
            remaining &= !Self::_2_INDIRECT_BUFFER.0;
            first = false;
        }
        if remaining & Self::_2_SHADER_DEVICE_ADDRESS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_SHADER_DEVICE_ADDRESS")?;
            remaining &= !Self::_2_SHADER_DEVICE_ADDRESS.0;
            first = false;
        }
        if remaining & Self::_2_EXECUTION_GRAPH_SCRATCH_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_EXECUTION_GRAPH_SCRATCH_BIT")?;
            remaining &= !Self::_2_EXECUTION_GRAPH_SCRATCH_BIT.0;
            first = false;
        }
        if remaining & Self::_2_DESCRIPTOR_HEAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_DESCRIPTOR_HEAP")?;
            remaining &= !Self::_2_DESCRIPTOR_HEAP.0;
            first = false;
        }
        if remaining & Self::_2_CONDITIONAL_RENDERING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_CONDITIONAL_RENDERING")?;
            remaining &= !Self::_2_CONDITIONAL_RENDERING.0;
            first = false;
        }
        if remaining & Self::_2_SHADER_BINDING_TABLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_SHADER_BINDING_TABLE")?;
            remaining &= !Self::_2_SHADER_BINDING_TABLE.0;
            first = false;
        }
        if remaining & Self::_2_TRANSFORM_FEEDBACK_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_TRANSFORM_FEEDBACK_BUFFER")?;
            remaining &= !Self::_2_TRANSFORM_FEEDBACK_BUFFER.0;
            first = false;
        }
        if remaining & Self::_2_TRANSFORM_FEEDBACK_COUNTER_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_TRANSFORM_FEEDBACK_COUNTER_BUFFER")?;
            remaining &= !Self::_2_TRANSFORM_FEEDBACK_COUNTER_BUFFER.0;
            first = false;
        }
        if remaining & Self::_2_VIDEO_DECODE_SRC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VIDEO_DECODE_SRC")?;
            remaining &= !Self::_2_VIDEO_DECODE_SRC.0;
            first = false;
        }
        if remaining & Self::_2_VIDEO_DECODE_DST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VIDEO_DECODE_DST")?;
            remaining &= !Self::_2_VIDEO_DECODE_DST.0;
            first = false;
        }
        if remaining & Self::_2_VIDEO_ENCODE_DST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VIDEO_ENCODE_DST")?;
            remaining &= !Self::_2_VIDEO_ENCODE_DST.0;
            first = false;
        }
        if remaining & Self::_2_VIDEO_ENCODE_SRC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VIDEO_ENCODE_SRC")?;
            remaining &= !Self::_2_VIDEO_ENCODE_SRC.0;
            first = false;
        }
        if remaining & Self::_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY")?;
            remaining &= !Self::_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY.0;
            first = false;
        }
        if remaining & Self::_2_ACCELERATION_STRUCTURE_STORAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_ACCELERATION_STRUCTURE_STORAGE")?;
            remaining &= !Self::_2_ACCELERATION_STRUCTURE_STORAGE.0;
            first = false;
        }
        if remaining & Self::_2_SAMPLER_DESCRIPTOR_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_SAMPLER_DESCRIPTOR_BUFFER")?;
            remaining &= !Self::_2_SAMPLER_DESCRIPTOR_BUFFER.0;
            first = false;
        }
        if remaining & Self::_2_RESOURCE_DESCRIPTOR_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_RESOURCE_DESCRIPTOR_BUFFER")?;
            remaining &= !Self::_2_RESOURCE_DESCRIPTOR_BUFFER.0;
            first = false;
        }
        if remaining & Self::_2_PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER")?;
            remaining &= !Self::_2_PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER.0;
            first = false;
        }
        if remaining & Self::_2_MICROMAP_BUILD_INPUT_READ_ONLY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_MICROMAP_BUILD_INPUT_READ_ONLY")?;
            remaining &= !Self::_2_MICROMAP_BUILD_INPUT_READ_ONLY.0;
            first = false;
        }
        if remaining & Self::_2_MICROMAP_STORAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_MICROMAP_STORAGE")?;
            remaining &= !Self::_2_MICROMAP_STORAGE.0;
            first = false;
        }
        if remaining & Self::_2_COMPRESSED_DATA_DGF1_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_COMPRESSED_DATA_DGF1_BIT")?;
            remaining &= !Self::_2_COMPRESSED_DATA_DGF1_BIT.0;
            first = false;
        }
        if remaining & Self::_2_DATA_GRAPH_FOREIGN_DESCRIPTOR_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_DATA_GRAPH_FOREIGN_DESCRIPTOR_BIT")?;
            remaining &= !Self::_2_DATA_GRAPH_FOREIGN_DESCRIPTOR_BIT.0;
            first = false;
        }
        if remaining & Self::_2_TILE_MEMORY_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_TILE_MEMORY_BIT")?;
            remaining &= !Self::_2_TILE_MEMORY_BIT.0;
            first = false;
        }
        if remaining & Self::_2_MEMORY_DECOMPRESSION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_MEMORY_DECOMPRESSION")?;
            remaining &= !Self::_2_MEMORY_DECOMPRESSION.0;
            first = false;
        }
        if remaining & Self::_2_PREPROCESS_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_PREPROCESS_BUFFER")?;
            remaining &= !Self::_2_PREPROCESS_BUFFER.0;
            first = false;
        }
        if remaining != 0u64 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkBuildAccelerationStructureFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkBuildAccelerationStructureFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkBuildAccelerationStructureFlagBitsKHR")]
pub struct BuildAccelerationStructureFlagBitsKHR(u32);
impl BuildAccelerationStructureFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const ALLOW_UPDATE: Self = Self(1u32);
    ///Bit 1.
    pub const ALLOW_COMPACTION: Self = Self(2u32);
    ///Bit 2.
    pub const PREFER_FAST_TRACE: Self = Self(4u32);
    ///Bit 3.
    pub const PREFER_FAST_BUILD: Self = Self(8u32);
    ///Bit 4.
    pub const LOW_MEMORY: Self = Self(16u32);
    ///Bit 5.
    pub const MOTION: Self = Self(32u32);
    ///Bit 6.
    pub const ALLOW_OPACITY_MICROMAP_UPDATE: Self = Self(64u32);
    ///Bit 7.
    pub const ALLOW_DISABLE_OPACITY_MICROMAPS: Self = Self(128u32);
    ///Bit 8.
    pub const ALLOW_OPACITY_MICROMAP_DATA_UPDATE: Self = Self(256u32);
    ///Bit 9.
    pub const ALLOW_DISPLACEMENT_MICROMAP_UPDATE: Self = Self(512u32);
    ///Bit 11.
    pub const ALLOW_DATA_ACCESS: Self = Self(2048u32);
    ///Bit 12.
    pub const ALLOW_CLUSTER_OPACITY_MICROMAPS: Self = Self(4096u32);
}
impl core::ops::BitOr for BuildAccelerationStructureFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for BuildAccelerationStructureFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for BuildAccelerationStructureFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for BuildAccelerationStructureFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for BuildAccelerationStructureFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for BuildAccelerationStructureFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for BuildAccelerationStructureFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for BuildAccelerationStructureFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::ALLOW_UPDATE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALLOW_UPDATE")?;
            remaining &= !Self::ALLOW_UPDATE.0;
            first = false;
        }
        if remaining & Self::ALLOW_COMPACTION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALLOW_COMPACTION")?;
            remaining &= !Self::ALLOW_COMPACTION.0;
            first = false;
        }
        if remaining & Self::PREFER_FAST_TRACE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PREFER_FAST_TRACE")?;
            remaining &= !Self::PREFER_FAST_TRACE.0;
            first = false;
        }
        if remaining & Self::PREFER_FAST_BUILD.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PREFER_FAST_BUILD")?;
            remaining &= !Self::PREFER_FAST_BUILD.0;
            first = false;
        }
        if remaining & Self::LOW_MEMORY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("LOW_MEMORY")?;
            remaining &= !Self::LOW_MEMORY.0;
            first = false;
        }
        if remaining & Self::MOTION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MOTION")?;
            remaining &= !Self::MOTION.0;
            first = false;
        }
        if remaining & Self::ALLOW_OPACITY_MICROMAP_UPDATE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALLOW_OPACITY_MICROMAP_UPDATE")?;
            remaining &= !Self::ALLOW_OPACITY_MICROMAP_UPDATE.0;
            first = false;
        }
        if remaining & Self::ALLOW_DISABLE_OPACITY_MICROMAPS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALLOW_DISABLE_OPACITY_MICROMAPS")?;
            remaining &= !Self::ALLOW_DISABLE_OPACITY_MICROMAPS.0;
            first = false;
        }
        if remaining & Self::ALLOW_OPACITY_MICROMAP_DATA_UPDATE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALLOW_OPACITY_MICROMAP_DATA_UPDATE")?;
            remaining &= !Self::ALLOW_OPACITY_MICROMAP_DATA_UPDATE.0;
            first = false;
        }
        if remaining & Self::ALLOW_DISPLACEMENT_MICROMAP_UPDATE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALLOW_DISPLACEMENT_MICROMAP_UPDATE")?;
            remaining &= !Self::ALLOW_DISPLACEMENT_MICROMAP_UPDATE.0;
            first = false;
        }
        if remaining & Self::ALLOW_DATA_ACCESS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALLOW_DATA_ACCESS")?;
            remaining &= !Self::ALLOW_DATA_ACCESS.0;
            first = false;
        }
        if remaining & Self::ALLOW_CLUSTER_OPACITY_MICROMAPS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALLOW_CLUSTER_OPACITY_MICROMAPS")?;
            remaining &= !Self::ALLOW_CLUSTER_OPACITY_MICROMAPS.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkBuildMicromapFlagBitsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkBuildMicromapFlagBitsEXT.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkBuildMicromapFlagBitsEXT")]
pub struct BuildMicromapFlagBitsEXT(u32);
impl BuildMicromapFlagBitsEXT {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const PREFER_FAST_TRACE: Self = Self(1u32);
    ///Bit 1.
    pub const PREFER_FAST_BUILD: Self = Self(2u32);
    ///Bit 2.
    pub const ALLOW_COMPACTION: Self = Self(4u32);
}
impl core::ops::BitOr for BuildMicromapFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for BuildMicromapFlagBitsEXT {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for BuildMicromapFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for BuildMicromapFlagBitsEXT {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for BuildMicromapFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for BuildMicromapFlagBitsEXT {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for BuildMicromapFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for BuildMicromapFlagBitsEXT {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::PREFER_FAST_TRACE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PREFER_FAST_TRACE")?;
            remaining &= !Self::PREFER_FAST_TRACE.0;
            first = false;
        }
        if remaining & Self::PREFER_FAST_BUILD.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PREFER_FAST_BUILD")?;
            remaining &= !Self::PREFER_FAST_BUILD.0;
            first = false;
        }
        if remaining & Self::ALLOW_COMPACTION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALLOW_COMPACTION")?;
            remaining &= !Self::ALLOW_COMPACTION.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkClusterAccelerationStructureAddressResolutionFlagBitsNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkClusterAccelerationStructureAddressResolutionFlagBitsNV.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkClusterAccelerationStructureAddressResolutionFlagBitsNV")]
pub struct ClusterAccelerationStructureAddressResolutionFlagBitsNV(u32);
impl ClusterAccelerationStructureAddressResolutionFlagBitsNV {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const NONE: Self = Self(0u32);
    ///Bit 0.
    pub const INDIRECTED_DST_IMPLICIT_DATA: Self = Self(1u32);
    ///Bit 1.
    pub const INDIRECTED_SCRATCH_DATA: Self = Self(2u32);
    ///Bit 2.
    pub const INDIRECTED_DST_ADDRESS_ARRAY: Self = Self(4u32);
    ///Bit 3.
    pub const INDIRECTED_DST_SIZES_ARRAY: Self = Self(8u32);
    ///Bit 4.
    pub const INDIRECTED_SRC_INFOS_ARRAY: Self = Self(16u32);
    ///Bit 5.
    pub const INDIRECTED_SRC_INFOS_COUNT: Self = Self(32u32);
}
impl core::ops::BitOr for ClusterAccelerationStructureAddressResolutionFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ClusterAccelerationStructureAddressResolutionFlagBitsNV {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ClusterAccelerationStructureAddressResolutionFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign
for ClusterAccelerationStructureAddressResolutionFlagBitsNV {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ClusterAccelerationStructureAddressResolutionFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign
for ClusterAccelerationStructureAddressResolutionFlagBitsNV {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ClusterAccelerationStructureAddressResolutionFlagBitsNV {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ClusterAccelerationStructureAddressResolutionFlagBitsNV {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::INDIRECTED_DST_IMPLICIT_DATA.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INDIRECTED_DST_IMPLICIT_DATA")?;
            remaining &= !Self::INDIRECTED_DST_IMPLICIT_DATA.0;
            first = false;
        }
        if remaining & Self::INDIRECTED_SCRATCH_DATA.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INDIRECTED_SCRATCH_DATA")?;
            remaining &= !Self::INDIRECTED_SCRATCH_DATA.0;
            first = false;
        }
        if remaining & Self::INDIRECTED_DST_ADDRESS_ARRAY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INDIRECTED_DST_ADDRESS_ARRAY")?;
            remaining &= !Self::INDIRECTED_DST_ADDRESS_ARRAY.0;
            first = false;
        }
        if remaining & Self::INDIRECTED_DST_SIZES_ARRAY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INDIRECTED_DST_SIZES_ARRAY")?;
            remaining &= !Self::INDIRECTED_DST_SIZES_ARRAY.0;
            first = false;
        }
        if remaining & Self::INDIRECTED_SRC_INFOS_ARRAY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INDIRECTED_SRC_INFOS_ARRAY")?;
            remaining &= !Self::INDIRECTED_SRC_INFOS_ARRAY.0;
            first = false;
        }
        if remaining & Self::INDIRECTED_SRC_INFOS_COUNT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INDIRECTED_SRC_INFOS_COUNT")?;
            remaining &= !Self::INDIRECTED_SRC_INFOS_COUNT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkClusterAccelerationStructureClusterFlagBitsNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkClusterAccelerationStructureClusterFlagBitsNV.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkClusterAccelerationStructureClusterFlagBitsNV")]
pub struct ClusterAccelerationStructureClusterFlagBitsNV(u32);
impl ClusterAccelerationStructureClusterFlagBitsNV {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const ALLOW_DISABLE_OPACITY_MICROMAPS: Self = Self(1u32);
}
impl core::ops::BitOr for ClusterAccelerationStructureClusterFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ClusterAccelerationStructureClusterFlagBitsNV {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ClusterAccelerationStructureClusterFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ClusterAccelerationStructureClusterFlagBitsNV {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ClusterAccelerationStructureClusterFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ClusterAccelerationStructureClusterFlagBitsNV {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ClusterAccelerationStructureClusterFlagBitsNV {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ClusterAccelerationStructureClusterFlagBitsNV {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::ALLOW_DISABLE_OPACITY_MICROMAPS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALLOW_DISABLE_OPACITY_MICROMAPS")?;
            remaining &= !Self::ALLOW_DISABLE_OPACITY_MICROMAPS.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkClusterAccelerationStructureGeometryFlagBitsNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkClusterAccelerationStructureGeometryFlagBitsNV.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkClusterAccelerationStructureGeometryFlagBitsNV")]
pub struct ClusterAccelerationStructureGeometryFlagBitsNV(u32);
impl ClusterAccelerationStructureGeometryFlagBitsNV {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const CULL_DISABLE: Self = Self(1u32);
    ///Bit 1.
    pub const NO_DUPLICATE_ANYHIT_INVOCATION: Self = Self(2u32);
    ///Bit 2.
    pub const OPAQUE: Self = Self(4u32);
}
impl core::ops::BitOr for ClusterAccelerationStructureGeometryFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ClusterAccelerationStructureGeometryFlagBitsNV {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ClusterAccelerationStructureGeometryFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ClusterAccelerationStructureGeometryFlagBitsNV {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ClusterAccelerationStructureGeometryFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ClusterAccelerationStructureGeometryFlagBitsNV {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ClusterAccelerationStructureGeometryFlagBitsNV {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ClusterAccelerationStructureGeometryFlagBitsNV {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::CULL_DISABLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CULL_DISABLE")?;
            remaining &= !Self::CULL_DISABLE.0;
            first = false;
        }
        if remaining & Self::NO_DUPLICATE_ANYHIT_INVOCATION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("NO_DUPLICATE_ANYHIT_INVOCATION")?;
            remaining &= !Self::NO_DUPLICATE_ANYHIT_INVOCATION.0;
            first = false;
        }
        if remaining & Self::OPAQUE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("OPAQUE")?;
            remaining &= !Self::OPAQUE.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkClusterAccelerationStructureIndexFormatFlagBitsNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkClusterAccelerationStructureIndexFormatFlagBitsNV.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkClusterAccelerationStructureIndexFormatFlagBitsNV")]
pub struct ClusterAccelerationStructureIndexFormatFlagBitsNV(u32);
impl ClusterAccelerationStructureIndexFormatFlagBitsNV {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const _8BIT: Self = Self(1u32);
    ///Bit 1.
    pub const _16BIT: Self = Self(2u32);
    ///Bit 2.
    pub const _32BIT: Self = Self(4u32);
}
impl core::ops::BitOr for ClusterAccelerationStructureIndexFormatFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ClusterAccelerationStructureIndexFormatFlagBitsNV {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ClusterAccelerationStructureIndexFormatFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ClusterAccelerationStructureIndexFormatFlagBitsNV {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ClusterAccelerationStructureIndexFormatFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ClusterAccelerationStructureIndexFormatFlagBitsNV {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ClusterAccelerationStructureIndexFormatFlagBitsNV {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ClusterAccelerationStructureIndexFormatFlagBitsNV {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::_8BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_8BIT")?;
            remaining &= !Self::_8BIT.0;
            first = false;
        }
        if remaining & Self::_16BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_16BIT")?;
            remaining &= !Self::_16BIT.0;
            first = false;
        }
        if remaining & Self::_32BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_32BIT")?;
            remaining &= !Self::_32BIT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkColorComponentFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkColorComponentFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkColorComponentFlagBits")]
pub struct ColorComponentFlagBits(u32);
impl ColorComponentFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const R: Self = Self(1u32);
    ///Bit 1.
    pub const G: Self = Self(2u32);
    ///Bit 2.
    pub const B: Self = Self(4u32);
    ///Bit 3.
    pub const A: Self = Self(8u32);
}
impl core::ops::BitOr for ColorComponentFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ColorComponentFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ColorComponentFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ColorComponentFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ColorComponentFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ColorComponentFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ColorComponentFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ColorComponentFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::R.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("R")?;
            remaining &= !Self::R.0;
            first = false;
        }
        if remaining & Self::G.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("G")?;
            remaining &= !Self::G.0;
            first = false;
        }
        if remaining & Self::B.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("B")?;
            remaining &= !Self::B.0;
            first = false;
        }
        if remaining & Self::A.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("A")?;
            remaining &= !Self::A.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkCommandBufferResetFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandBufferResetFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkCommandBufferResetFlagBits")]
pub struct CommandBufferResetFlagBits(u32);
impl CommandBufferResetFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const RELEASE_RESOURCES: Self = Self(1u32);
}
impl core::ops::BitOr for CommandBufferResetFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for CommandBufferResetFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for CommandBufferResetFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for CommandBufferResetFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for CommandBufferResetFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for CommandBufferResetFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for CommandBufferResetFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for CommandBufferResetFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::RELEASE_RESOURCES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RELEASE_RESOURCES")?;
            remaining &= !Self::RELEASE_RESOURCES.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkCommandBufferUsageFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandBufferUsageFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkCommandBufferUsageFlagBits")]
pub struct CommandBufferUsageFlagBits(u32);
impl CommandBufferUsageFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const ONE_TIME_SUBMIT: Self = Self(1u32);
    ///Bit 1.
    pub const RENDER_PASS_CONTINUE: Self = Self(2u32);
    ///Bit 2.
    pub const SIMULTANEOUS_USE: Self = Self(4u32);
}
impl core::ops::BitOr for CommandBufferUsageFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for CommandBufferUsageFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for CommandBufferUsageFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for CommandBufferUsageFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for CommandBufferUsageFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for CommandBufferUsageFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for CommandBufferUsageFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for CommandBufferUsageFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::ONE_TIME_SUBMIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ONE_TIME_SUBMIT")?;
            remaining &= !Self::ONE_TIME_SUBMIT.0;
            first = false;
        }
        if remaining & Self::RENDER_PASS_CONTINUE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RENDER_PASS_CONTINUE")?;
            remaining &= !Self::RENDER_PASS_CONTINUE.0;
            first = false;
        }
        if remaining & Self::SIMULTANEOUS_USE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SIMULTANEOUS_USE")?;
            remaining &= !Self::SIMULTANEOUS_USE.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkCommandPoolCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandPoolCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkCommandPoolCreateFlagBits")]
pub struct CommandPoolCreateFlagBits(u32);
impl CommandPoolCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const TRANSIENT: Self = Self(1u32);
    ///Bit 1.
    pub const RESET_COMMAND_BUFFER: Self = Self(2u32);
    ///Bit 2.
    pub const PROTECTED: Self = Self(4u32);
}
impl core::ops::BitOr for CommandPoolCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for CommandPoolCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for CommandPoolCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for CommandPoolCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for CommandPoolCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for CommandPoolCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for CommandPoolCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for CommandPoolCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::TRANSIENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSIENT")?;
            remaining &= !Self::TRANSIENT.0;
            first = false;
        }
        if remaining & Self::RESET_COMMAND_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RESET_COMMAND_BUFFER")?;
            remaining &= !Self::RESET_COMMAND_BUFFER.0;
            first = false;
        }
        if remaining & Self::PROTECTED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PROTECTED")?;
            remaining &= !Self::PROTECTED.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkCommandPoolResetFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandPoolResetFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkCommandPoolResetFlagBits")]
pub struct CommandPoolResetFlagBits(u32);
impl CommandPoolResetFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const RELEASE_RESOURCES: Self = Self(1u32);
}
impl core::ops::BitOr for CommandPoolResetFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for CommandPoolResetFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for CommandPoolResetFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for CommandPoolResetFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for CommandPoolResetFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for CommandPoolResetFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for CommandPoolResetFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for CommandPoolResetFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::RELEASE_RESOURCES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RELEASE_RESOURCES")?;
            remaining &= !Self::RELEASE_RESOURCES.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkCompositeAlphaFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkCompositeAlphaFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkCompositeAlphaFlagBitsKHR")]
pub struct CompositeAlphaFlagBitsKHR(u32);
impl CompositeAlphaFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const OPAQUE: Self = Self(1u32);
    ///Bit 1.
    pub const PRE_MULTIPLIED: Self = Self(2u32);
    ///Bit 2.
    pub const POST_MULTIPLIED: Self = Self(4u32);
    ///Bit 3.
    pub const INHERIT: Self = Self(8u32);
}
impl core::ops::BitOr for CompositeAlphaFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for CompositeAlphaFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for CompositeAlphaFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for CompositeAlphaFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for CompositeAlphaFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for CompositeAlphaFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for CompositeAlphaFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for CompositeAlphaFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::OPAQUE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("OPAQUE")?;
            remaining &= !Self::OPAQUE.0;
            first = false;
        }
        if remaining & Self::PRE_MULTIPLIED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PRE_MULTIPLIED")?;
            remaining &= !Self::PRE_MULTIPLIED.0;
            first = false;
        }
        if remaining & Self::POST_MULTIPLIED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("POST_MULTIPLIED")?;
            remaining &= !Self::POST_MULTIPLIED.0;
            first = false;
        }
        if remaining & Self::INHERIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INHERIT")?;
            remaining &= !Self::INHERIT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkConditionalRenderingFlagBitsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkConditionalRenderingFlagBitsEXT.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkConditionalRenderingFlagBitsEXT")]
pub struct ConditionalRenderingFlagBitsEXT(u32);
impl ConditionalRenderingFlagBitsEXT {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const INVERTED: Self = Self(1u32);
}
impl core::ops::BitOr for ConditionalRenderingFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ConditionalRenderingFlagBitsEXT {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ConditionalRenderingFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ConditionalRenderingFlagBitsEXT {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ConditionalRenderingFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ConditionalRenderingFlagBitsEXT {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ConditionalRenderingFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ConditionalRenderingFlagBitsEXT {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::INVERTED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INVERTED")?;
            remaining &= !Self::INVERTED.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkCullModeFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkCullModeFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkCullModeFlagBits")]
pub struct CullModeFlagBits(u32);
impl CullModeFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const NONE: Self = Self(0u32);
    ///Bit 0.
    pub const FRONT: Self = Self(1u32);
    ///Bit 1.
    pub const BACK: Self = Self(2u32);
    pub const FRONT_AND_BACK: Self = Self(3u32);
}
impl core::ops::BitOr for CullModeFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for CullModeFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for CullModeFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for CullModeFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for CullModeFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for CullModeFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for CullModeFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for CullModeFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::FRONT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRONT")?;
            remaining &= !Self::FRONT.0;
            first = false;
        }
        if remaining & Self::BACK.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("BACK")?;
            remaining &= !Self::BACK.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkDataGraphPipelineDispatchFlagBitsARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkDataGraphPipelineDispatchFlagBitsARM.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkDataGraphPipelineDispatchFlagBitsARM")]
pub struct DataGraphPipelineDispatchFlagBitsARM(u64);
impl DataGraphPipelineDispatchFlagBitsARM {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u64)
    }
    #[inline]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u64
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}
impl core::ops::BitOr for DataGraphPipelineDispatchFlagBitsARM {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for DataGraphPipelineDispatchFlagBitsARM {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for DataGraphPipelineDispatchFlagBitsARM {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for DataGraphPipelineDispatchFlagBitsARM {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for DataGraphPipelineDispatchFlagBitsARM {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for DataGraphPipelineDispatchFlagBitsARM {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for DataGraphPipelineDispatchFlagBitsARM {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for DataGraphPipelineDispatchFlagBitsARM {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining != 0u64 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkDataGraphPipelineSessionCreateFlagBitsARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkDataGraphPipelineSessionCreateFlagBitsARM.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkDataGraphPipelineSessionCreateFlagBitsARM")]
pub struct DataGraphPipelineSessionCreateFlagBitsARM(u64);
impl DataGraphPipelineSessionCreateFlagBitsARM {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u64)
    }
    #[inline]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u64
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const PROTECTED_BIT: Self = Self(1u64);
}
impl core::ops::BitOr for DataGraphPipelineSessionCreateFlagBitsARM {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for DataGraphPipelineSessionCreateFlagBitsARM {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for DataGraphPipelineSessionCreateFlagBitsARM {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for DataGraphPipelineSessionCreateFlagBitsARM {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for DataGraphPipelineSessionCreateFlagBitsARM {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for DataGraphPipelineSessionCreateFlagBitsARM {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for DataGraphPipelineSessionCreateFlagBitsARM {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for DataGraphPipelineSessionCreateFlagBitsARM {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::PROTECTED_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PROTECTED_BIT")?;
            remaining &= !Self::PROTECTED_BIT.0;
            first = false;
        }
        if remaining != 0u64 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkDebugReportFlagBitsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkDebugReportFlagBitsEXT.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkDebugReportFlagBitsEXT")]
pub struct DebugReportFlagBitsEXT(u32);
impl DebugReportFlagBitsEXT {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const INFORMATION: Self = Self(1u32);
    ///Bit 1.
    pub const WARNING: Self = Self(2u32);
    ///Bit 2.
    pub const PERFORMANCE_WARNING: Self = Self(4u32);
    ///Bit 3.
    pub const ERROR: Self = Self(8u32);
    ///Bit 4.
    pub const DEBUG: Self = Self(16u32);
}
impl core::ops::BitOr for DebugReportFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for DebugReportFlagBitsEXT {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for DebugReportFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for DebugReportFlagBitsEXT {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for DebugReportFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for DebugReportFlagBitsEXT {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for DebugReportFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for DebugReportFlagBitsEXT {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::INFORMATION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INFORMATION")?;
            remaining &= !Self::INFORMATION.0;
            first = false;
        }
        if remaining & Self::WARNING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("WARNING")?;
            remaining &= !Self::WARNING.0;
            first = false;
        }
        if remaining & Self::PERFORMANCE_WARNING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PERFORMANCE_WARNING")?;
            remaining &= !Self::PERFORMANCE_WARNING.0;
            first = false;
        }
        if remaining & Self::ERROR.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ERROR")?;
            remaining &= !Self::ERROR.0;
            first = false;
        }
        if remaining & Self::DEBUG.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEBUG")?;
            remaining &= !Self::DEBUG.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkDebugUtilsMessageSeverityFlagBitsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkDebugUtilsMessageSeverityFlagBitsEXT.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkDebugUtilsMessageSeverityFlagBitsEXT")]
pub struct DebugUtilsMessageSeverityFlagBitsEXT(u32);
impl DebugUtilsMessageSeverityFlagBitsEXT {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const VERBOSE: Self = Self(1u32);
    ///Bit 4.
    pub const INFO: Self = Self(16u32);
    ///Bit 8.
    pub const WARNING: Self = Self(256u32);
    ///Bit 12.
    pub const ERROR: Self = Self(4096u32);
}
impl core::ops::BitOr for DebugUtilsMessageSeverityFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for DebugUtilsMessageSeverityFlagBitsEXT {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for DebugUtilsMessageSeverityFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for DebugUtilsMessageSeverityFlagBitsEXT {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for DebugUtilsMessageSeverityFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for DebugUtilsMessageSeverityFlagBitsEXT {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for DebugUtilsMessageSeverityFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for DebugUtilsMessageSeverityFlagBitsEXT {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::VERBOSE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VERBOSE")?;
            remaining &= !Self::VERBOSE.0;
            first = false;
        }
        if remaining & Self::INFO.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INFO")?;
            remaining &= !Self::INFO.0;
            first = false;
        }
        if remaining & Self::WARNING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("WARNING")?;
            remaining &= !Self::WARNING.0;
            first = false;
        }
        if remaining & Self::ERROR.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ERROR")?;
            remaining &= !Self::ERROR.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkDebugUtilsMessageTypeFlagBitsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkDebugUtilsMessageTypeFlagBitsEXT.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkDebugUtilsMessageTypeFlagBitsEXT")]
pub struct DebugUtilsMessageTypeFlagBitsEXT(u32);
impl DebugUtilsMessageTypeFlagBitsEXT {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const GENERAL: Self = Self(1u32);
    ///Bit 1.
    pub const VALIDATION: Self = Self(2u32);
    ///Bit 2.
    pub const PERFORMANCE: Self = Self(4u32);
    ///Bit 3.
    pub const DEVICE_ADDRESS_BINDING: Self = Self(8u32);
}
impl core::ops::BitOr for DebugUtilsMessageTypeFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for DebugUtilsMessageTypeFlagBitsEXT {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for DebugUtilsMessageTypeFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for DebugUtilsMessageTypeFlagBitsEXT {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for DebugUtilsMessageTypeFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for DebugUtilsMessageTypeFlagBitsEXT {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for DebugUtilsMessageTypeFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for DebugUtilsMessageTypeFlagBitsEXT {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::GENERAL.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("GENERAL")?;
            remaining &= !Self::GENERAL.0;
            first = false;
        }
        if remaining & Self::VALIDATION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VALIDATION")?;
            remaining &= !Self::VALIDATION.0;
            first = false;
        }
        if remaining & Self::PERFORMANCE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PERFORMANCE")?;
            remaining &= !Self::PERFORMANCE.0;
            first = false;
        }
        if remaining & Self::DEVICE_ADDRESS_BINDING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEVICE_ADDRESS_BINDING")?;
            remaining &= !Self::DEVICE_ADDRESS_BINDING.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkDependencyFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkDependencyFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkDependencyFlagBits")]
pub struct DependencyFlagBits(u32);
impl DependencyFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const BY_REGION: Self = Self(1u32);
    ///Bit 2.
    pub const DEVICE_GROUP: Self = Self(4u32);
    ///Bit 1.
    pub const VIEW_LOCAL: Self = Self(2u32);
    ///Bit 3.
    pub const FEEDBACK_LOOP: Self = Self(8u32);
    ///Bit 5.
    pub const QUEUE_FAMILY_OWNERSHIP_TRANSFER_USE_ALL_STAGES: Self = Self(32u32);
    ///Bit 6.
    pub const ASYMMETRIC_EVENT: Self = Self(64u32);
}
impl core::ops::BitOr for DependencyFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for DependencyFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for DependencyFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for DependencyFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for DependencyFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for DependencyFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for DependencyFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for DependencyFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::BY_REGION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("BY_REGION")?;
            remaining &= !Self::BY_REGION.0;
            first = false;
        }
        if remaining & Self::DEVICE_GROUP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEVICE_GROUP")?;
            remaining &= !Self::DEVICE_GROUP.0;
            first = false;
        }
        if remaining & Self::VIEW_LOCAL.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VIEW_LOCAL")?;
            remaining &= !Self::VIEW_LOCAL.0;
            first = false;
        }
        if remaining & Self::FEEDBACK_LOOP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FEEDBACK_LOOP")?;
            remaining &= !Self::FEEDBACK_LOOP.0;
            first = false;
        }
        if remaining & Self::QUEUE_FAMILY_OWNERSHIP_TRANSFER_USE_ALL_STAGES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("QUEUE_FAMILY_OWNERSHIP_TRANSFER_USE_ALL_STAGES")?;
            remaining &= !Self::QUEUE_FAMILY_OWNERSHIP_TRANSFER_USE_ALL_STAGES.0;
            first = false;
        }
        if remaining & Self::ASYMMETRIC_EVENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ASYMMETRIC_EVENT")?;
            remaining &= !Self::ASYMMETRIC_EVENT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkDescriptorBindingFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkDescriptorBindingFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkDescriptorBindingFlagBits")]
pub struct DescriptorBindingFlagBits(u32);
impl DescriptorBindingFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const UPDATE_AFTER_BIND: Self = Self(1u32);
    ///Bit 1.
    pub const UPDATE_UNUSED_WHILE_PENDING: Self = Self(2u32);
    ///Bit 2.
    pub const PARTIALLY_BOUND: Self = Self(4u32);
    ///Bit 3.
    pub const VARIABLE_DESCRIPTOR_COUNT: Self = Self(8u32);
}
impl core::ops::BitOr for DescriptorBindingFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for DescriptorBindingFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for DescriptorBindingFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for DescriptorBindingFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for DescriptorBindingFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for DescriptorBindingFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for DescriptorBindingFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for DescriptorBindingFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::UPDATE_AFTER_BIND.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("UPDATE_AFTER_BIND")?;
            remaining &= !Self::UPDATE_AFTER_BIND.0;
            first = false;
        }
        if remaining & Self::UPDATE_UNUSED_WHILE_PENDING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("UPDATE_UNUSED_WHILE_PENDING")?;
            remaining &= !Self::UPDATE_UNUSED_WHILE_PENDING.0;
            first = false;
        }
        if remaining & Self::PARTIALLY_BOUND.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PARTIALLY_BOUND")?;
            remaining &= !Self::PARTIALLY_BOUND.0;
            first = false;
        }
        if remaining & Self::VARIABLE_DESCRIPTOR_COUNT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VARIABLE_DESCRIPTOR_COUNT")?;
            remaining &= !Self::VARIABLE_DESCRIPTOR_COUNT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkDescriptorPoolCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkDescriptorPoolCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkDescriptorPoolCreateFlagBits")]
pub struct DescriptorPoolCreateFlagBits(u32);
impl DescriptorPoolCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const FREE_DESCRIPTOR_SET: Self = Self(1u32);
    ///Bit 1.
    pub const UPDATE_AFTER_BIND: Self = Self(2u32);
    pub const HOST_ONLY_BIT: Self = Self::HOST_ONLY;
    ///Bit 2.
    pub const HOST_ONLY: Self = Self(4u32);
    ///Bit 3.
    pub const ALLOW_OVERALLOCATION_SETS: Self = Self(8u32);
    ///Bit 4.
    pub const ALLOW_OVERALLOCATION_POOLS: Self = Self(16u32);
}
impl core::ops::BitOr for DescriptorPoolCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for DescriptorPoolCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for DescriptorPoolCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for DescriptorPoolCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for DescriptorPoolCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for DescriptorPoolCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for DescriptorPoolCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for DescriptorPoolCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::FREE_DESCRIPTOR_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FREE_DESCRIPTOR_SET")?;
            remaining &= !Self::FREE_DESCRIPTOR_SET.0;
            first = false;
        }
        if remaining & Self::UPDATE_AFTER_BIND.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("UPDATE_AFTER_BIND")?;
            remaining &= !Self::UPDATE_AFTER_BIND.0;
            first = false;
        }
        if remaining & Self::HOST_ONLY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("HOST_ONLY")?;
            remaining &= !Self::HOST_ONLY.0;
            first = false;
        }
        if remaining & Self::ALLOW_OVERALLOCATION_SETS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALLOW_OVERALLOCATION_SETS")?;
            remaining &= !Self::ALLOW_OVERALLOCATION_SETS.0;
            first = false;
        }
        if remaining & Self::ALLOW_OVERALLOCATION_POOLS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALLOW_OVERALLOCATION_POOLS")?;
            remaining &= !Self::ALLOW_OVERALLOCATION_POOLS.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkDescriptorSetLayoutCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkDescriptorSetLayoutCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkDescriptorSetLayoutCreateFlagBits")]
pub struct DescriptorSetLayoutCreateFlagBits(u32);
impl DescriptorSetLayoutCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 1.
    pub const UPDATE_AFTER_BIND_POOL: Self = Self(2u32);
    ///Bit 0.
    pub const PUSH_DESCRIPTOR: Self = Self(1u32);
    ///Bit 4.
    pub const DESCRIPTOR_BUFFER: Self = Self(16u32);
    ///Bit 5.
    pub const EMBEDDED_IMMUTABLE_SAMPLERS: Self = Self(32u32);
    pub const HOST_ONLY_POOL_BIT: Self = Self::HOST_ONLY_POOL;
    ///Bit 7.
    pub const INDIRECT_BINDABLE: Self = Self(128u32);
    ///Bit 2.
    pub const HOST_ONLY_POOL: Self = Self(4u32);
    ///Bit 6.
    pub const PER_STAGE: Self = Self(64u32);
}
impl core::ops::BitOr for DescriptorSetLayoutCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for DescriptorSetLayoutCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for DescriptorSetLayoutCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for DescriptorSetLayoutCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for DescriptorSetLayoutCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for DescriptorSetLayoutCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for DescriptorSetLayoutCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for DescriptorSetLayoutCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::UPDATE_AFTER_BIND_POOL.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("UPDATE_AFTER_BIND_POOL")?;
            remaining &= !Self::UPDATE_AFTER_BIND_POOL.0;
            first = false;
        }
        if remaining & Self::PUSH_DESCRIPTOR.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PUSH_DESCRIPTOR")?;
            remaining &= !Self::PUSH_DESCRIPTOR.0;
            first = false;
        }
        if remaining & Self::DESCRIPTOR_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DESCRIPTOR_BUFFER")?;
            remaining &= !Self::DESCRIPTOR_BUFFER.0;
            first = false;
        }
        if remaining & Self::EMBEDDED_IMMUTABLE_SAMPLERS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("EMBEDDED_IMMUTABLE_SAMPLERS")?;
            remaining &= !Self::EMBEDDED_IMMUTABLE_SAMPLERS.0;
            first = false;
        }
        if remaining & Self::INDIRECT_BINDABLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INDIRECT_BINDABLE")?;
            remaining &= !Self::INDIRECT_BINDABLE.0;
            first = false;
        }
        if remaining & Self::HOST_ONLY_POOL.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("HOST_ONLY_POOL")?;
            remaining &= !Self::HOST_ONLY_POOL.0;
            first = false;
        }
        if remaining & Self::PER_STAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PER_STAGE")?;
            remaining &= !Self::PER_STAGE.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkDeviceAddressBindingFlagBitsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceAddressBindingFlagBitsEXT.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkDeviceAddressBindingFlagBitsEXT")]
pub struct DeviceAddressBindingFlagBitsEXT(u32);
impl DeviceAddressBindingFlagBitsEXT {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const INTERNAL_OBJECT: Self = Self(1u32);
}
impl core::ops::BitOr for DeviceAddressBindingFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for DeviceAddressBindingFlagBitsEXT {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for DeviceAddressBindingFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for DeviceAddressBindingFlagBitsEXT {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for DeviceAddressBindingFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for DeviceAddressBindingFlagBitsEXT {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for DeviceAddressBindingFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for DeviceAddressBindingFlagBitsEXT {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::INTERNAL_OBJECT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INTERNAL_OBJECT")?;
            remaining &= !Self::INTERNAL_OBJECT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkDeviceDiagnosticsConfigFlagBitsNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceDiagnosticsConfigFlagBitsNV.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkDeviceDiagnosticsConfigFlagBitsNV")]
pub struct DeviceDiagnosticsConfigFlagBitsNV(u32);
impl DeviceDiagnosticsConfigFlagBitsNV {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const ENABLE_SHADER_DEBUG_INFO: Self = Self(1u32);
    ///Bit 1.
    pub const ENABLE_RESOURCE_TRACKING: Self = Self(2u32);
    ///Bit 2.
    pub const ENABLE_AUTOMATIC_CHECKPOINTS: Self = Self(4u32);
    ///Bit 3.
    pub const ENABLE_SHADER_ERROR_REPORTING: Self = Self(8u32);
}
impl core::ops::BitOr for DeviceDiagnosticsConfigFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for DeviceDiagnosticsConfigFlagBitsNV {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for DeviceDiagnosticsConfigFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for DeviceDiagnosticsConfigFlagBitsNV {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for DeviceDiagnosticsConfigFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for DeviceDiagnosticsConfigFlagBitsNV {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for DeviceDiagnosticsConfigFlagBitsNV {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for DeviceDiagnosticsConfigFlagBitsNV {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::ENABLE_SHADER_DEBUG_INFO.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ENABLE_SHADER_DEBUG_INFO")?;
            remaining &= !Self::ENABLE_SHADER_DEBUG_INFO.0;
            first = false;
        }
        if remaining & Self::ENABLE_RESOURCE_TRACKING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ENABLE_RESOURCE_TRACKING")?;
            remaining &= !Self::ENABLE_RESOURCE_TRACKING.0;
            first = false;
        }
        if remaining & Self::ENABLE_AUTOMATIC_CHECKPOINTS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ENABLE_AUTOMATIC_CHECKPOINTS")?;
            remaining &= !Self::ENABLE_AUTOMATIC_CHECKPOINTS.0;
            first = false;
        }
        if remaining & Self::ENABLE_SHADER_ERROR_REPORTING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ENABLE_SHADER_ERROR_REPORTING")?;
            remaining &= !Self::ENABLE_SHADER_ERROR_REPORTING.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkDeviceFaultFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceFaultFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkDeviceFaultFlagBitsKHR")]
pub struct DeviceFaultFlagBitsKHR(u32);
impl DeviceFaultFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const FLAG_DEVICE_LOST: Self = Self(1u32);
    ///Bit 1.
    pub const FLAG_MEMORY_ADDRESS: Self = Self(2u32);
    ///Bit 2.
    pub const FLAG_INSTRUCTION_ADDRESS: Self = Self(4u32);
    ///Bit 3.
    pub const FLAG_VENDOR: Self = Self(8u32);
    ///Bit 4.
    pub const FLAG_WATCHDOG_TIMEOUT: Self = Self(16u32);
    ///Bit 5.
    pub const FLAG_OVERFLOW: Self = Self(32u32);
}
impl core::ops::BitOr for DeviceFaultFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for DeviceFaultFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for DeviceFaultFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for DeviceFaultFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for DeviceFaultFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for DeviceFaultFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for DeviceFaultFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for DeviceFaultFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::FLAG_DEVICE_LOST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FLAG_DEVICE_LOST")?;
            remaining &= !Self::FLAG_DEVICE_LOST.0;
            first = false;
        }
        if remaining & Self::FLAG_MEMORY_ADDRESS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FLAG_MEMORY_ADDRESS")?;
            remaining &= !Self::FLAG_MEMORY_ADDRESS.0;
            first = false;
        }
        if remaining & Self::FLAG_INSTRUCTION_ADDRESS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FLAG_INSTRUCTION_ADDRESS")?;
            remaining &= !Self::FLAG_INSTRUCTION_ADDRESS.0;
            first = false;
        }
        if remaining & Self::FLAG_VENDOR.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FLAG_VENDOR")?;
            remaining &= !Self::FLAG_VENDOR.0;
            first = false;
        }
        if remaining & Self::FLAG_WATCHDOG_TIMEOUT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FLAG_WATCHDOG_TIMEOUT")?;
            remaining &= !Self::FLAG_WATCHDOG_TIMEOUT.0;
            first = false;
        }
        if remaining & Self::FLAG_OVERFLOW.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FLAG_OVERFLOW")?;
            remaining &= !Self::FLAG_OVERFLOW.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkDeviceGroupPresentModeFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceGroupPresentModeFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkDeviceGroupPresentModeFlagBitsKHR")]
pub struct DeviceGroupPresentModeFlagBitsKHR(u32);
impl DeviceGroupPresentModeFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const LOCAL: Self = Self(1u32);
    ///Bit 1.
    pub const REMOTE: Self = Self(2u32);
    ///Bit 2.
    pub const SUM: Self = Self(4u32);
    ///Bit 3.
    pub const LOCAL_MULTI_DEVICE: Self = Self(8u32);
}
impl core::ops::BitOr for DeviceGroupPresentModeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for DeviceGroupPresentModeFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for DeviceGroupPresentModeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for DeviceGroupPresentModeFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for DeviceGroupPresentModeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for DeviceGroupPresentModeFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for DeviceGroupPresentModeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for DeviceGroupPresentModeFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::LOCAL.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("LOCAL")?;
            remaining &= !Self::LOCAL.0;
            first = false;
        }
        if remaining & Self::REMOTE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("REMOTE")?;
            remaining &= !Self::REMOTE.0;
            first = false;
        }
        if remaining & Self::SUM.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SUM")?;
            remaining &= !Self::SUM.0;
            first = false;
        }
        if remaining & Self::LOCAL_MULTI_DEVICE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("LOCAL_MULTI_DEVICE")?;
            remaining &= !Self::LOCAL_MULTI_DEVICE.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkDeviceQueueCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceQueueCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkDeviceQueueCreateFlagBits")]
pub struct DeviceQueueCreateFlagBits(u32);
impl DeviceQueueCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const PROTECTED: Self = Self(1u32);
    ///Bit 2.
    pub const INTERNALLY_SYNCHRONIZED: Self = Self(4u32);
}
impl core::ops::BitOr for DeviceQueueCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for DeviceQueueCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for DeviceQueueCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for DeviceQueueCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for DeviceQueueCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for DeviceQueueCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for DeviceQueueCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for DeviceQueueCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::PROTECTED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PROTECTED")?;
            remaining &= !Self::PROTECTED.0;
            first = false;
        }
        if remaining & Self::INTERNALLY_SYNCHRONIZED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INTERNALLY_SYNCHRONIZED")?;
            remaining &= !Self::INTERNALLY_SYNCHRONIZED.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkDisplayPlaneAlphaFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkDisplayPlaneAlphaFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkDisplayPlaneAlphaFlagBitsKHR")]
pub struct DisplayPlaneAlphaFlagBitsKHR(u32);
impl DisplayPlaneAlphaFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const OPAQUE: Self = Self(1u32);
    ///Bit 1.
    pub const GLOBAL: Self = Self(2u32);
    ///Bit 2.
    pub const PER_PIXEL: Self = Self(4u32);
    ///Bit 3.
    pub const PER_PIXEL_PREMULTIPLIED: Self = Self(8u32);
}
impl core::ops::BitOr for DisplayPlaneAlphaFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for DisplayPlaneAlphaFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for DisplayPlaneAlphaFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for DisplayPlaneAlphaFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for DisplayPlaneAlphaFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for DisplayPlaneAlphaFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for DisplayPlaneAlphaFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for DisplayPlaneAlphaFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::OPAQUE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("OPAQUE")?;
            remaining &= !Self::OPAQUE.0;
            first = false;
        }
        if remaining & Self::GLOBAL.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("GLOBAL")?;
            remaining &= !Self::GLOBAL.0;
            first = false;
        }
        if remaining & Self::PER_PIXEL.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PER_PIXEL")?;
            remaining &= !Self::PER_PIXEL.0;
            first = false;
        }
        if remaining & Self::PER_PIXEL_PREMULTIPLIED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PER_PIXEL_PREMULTIPLIED")?;
            remaining &= !Self::PER_PIXEL_PREMULTIPLIED.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkEventCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkEventCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkEventCreateFlagBits")]
pub struct EventCreateFlagBits(u32);
impl EventCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const DEVICE_ONLY: Self = Self(1u32);
}
impl core::ops::BitOr for EventCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for EventCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for EventCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for EventCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for EventCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for EventCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for EventCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for EventCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::DEVICE_ONLY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEVICE_ONLY")?;
            remaining &= !Self::DEVICE_ONLY.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkExportMetalObjectTypeFlagBitsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkExportMetalObjectTypeFlagBitsEXT.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkExportMetalObjectTypeFlagBitsEXT")]
pub struct ExportMetalObjectTypeFlagBitsEXT(u32);
impl ExportMetalObjectTypeFlagBitsEXT {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const METAL_DEVICE: Self = Self(1u32);
    ///Bit 1.
    pub const METAL_COMMAND_QUEUE: Self = Self(2u32);
    ///Bit 2.
    pub const METAL_BUFFER: Self = Self(4u32);
    ///Bit 3.
    pub const METAL_TEXTURE: Self = Self(8u32);
    ///Bit 4.
    pub const METAL_IOSURFACE: Self = Self(16u32);
    ///Bit 5.
    pub const METAL_SHARED_EVENT: Self = Self(32u32);
}
impl core::ops::BitOr for ExportMetalObjectTypeFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ExportMetalObjectTypeFlagBitsEXT {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ExportMetalObjectTypeFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ExportMetalObjectTypeFlagBitsEXT {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ExportMetalObjectTypeFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ExportMetalObjectTypeFlagBitsEXT {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ExportMetalObjectTypeFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ExportMetalObjectTypeFlagBitsEXT {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::METAL_DEVICE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("METAL_DEVICE")?;
            remaining &= !Self::METAL_DEVICE.0;
            first = false;
        }
        if remaining & Self::METAL_COMMAND_QUEUE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("METAL_COMMAND_QUEUE")?;
            remaining &= !Self::METAL_COMMAND_QUEUE.0;
            first = false;
        }
        if remaining & Self::METAL_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("METAL_BUFFER")?;
            remaining &= !Self::METAL_BUFFER.0;
            first = false;
        }
        if remaining & Self::METAL_TEXTURE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("METAL_TEXTURE")?;
            remaining &= !Self::METAL_TEXTURE.0;
            first = false;
        }
        if remaining & Self::METAL_IOSURFACE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("METAL_IOSURFACE")?;
            remaining &= !Self::METAL_IOSURFACE.0;
            first = false;
        }
        if remaining & Self::METAL_SHARED_EVENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("METAL_SHARED_EVENT")?;
            remaining &= !Self::METAL_SHARED_EVENT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkExternalFenceFeatureFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalFenceFeatureFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkExternalFenceFeatureFlagBits")]
pub struct ExternalFenceFeatureFlagBits(u32);
impl ExternalFenceFeatureFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const EXPORTABLE: Self = Self(1u32);
    ///Bit 1.
    pub const IMPORTABLE: Self = Self(2u32);
}
impl core::ops::BitOr for ExternalFenceFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ExternalFenceFeatureFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ExternalFenceFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ExternalFenceFeatureFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ExternalFenceFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ExternalFenceFeatureFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ExternalFenceFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ExternalFenceFeatureFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::EXPORTABLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("EXPORTABLE")?;
            remaining &= !Self::EXPORTABLE.0;
            first = false;
        }
        if remaining & Self::IMPORTABLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("IMPORTABLE")?;
            remaining &= !Self::IMPORTABLE.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkExternalFenceHandleTypeFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalFenceHandleTypeFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkExternalFenceHandleTypeFlagBits")]
pub struct ExternalFenceHandleTypeFlagBits(u32);
impl ExternalFenceHandleTypeFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const OPAQUE_FD: Self = Self(1u32);
    ///Bit 1.
    pub const OPAQUE_WIN32: Self = Self(2u32);
    ///Bit 2.
    pub const OPAQUE_WIN32_KMT: Self = Self(4u32);
    ///Bit 3.
    pub const SYNC_FD: Self = Self(8u32);
}
impl core::ops::BitOr for ExternalFenceHandleTypeFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ExternalFenceHandleTypeFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ExternalFenceHandleTypeFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ExternalFenceHandleTypeFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ExternalFenceHandleTypeFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ExternalFenceHandleTypeFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ExternalFenceHandleTypeFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ExternalFenceHandleTypeFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::OPAQUE_FD.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("OPAQUE_FD")?;
            remaining &= !Self::OPAQUE_FD.0;
            first = false;
        }
        if remaining & Self::OPAQUE_WIN32.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("OPAQUE_WIN32")?;
            remaining &= !Self::OPAQUE_WIN32.0;
            first = false;
        }
        if remaining & Self::OPAQUE_WIN32_KMT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("OPAQUE_WIN32_KMT")?;
            remaining &= !Self::OPAQUE_WIN32_KMT.0;
            first = false;
        }
        if remaining & Self::SYNC_FD.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SYNC_FD")?;
            remaining &= !Self::SYNC_FD.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkExternalMemoryFeatureFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalMemoryFeatureFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkExternalMemoryFeatureFlagBits")]
pub struct ExternalMemoryFeatureFlagBits(u32);
impl ExternalMemoryFeatureFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const DEDICATED_ONLY: Self = Self(1u32);
    ///Bit 1.
    pub const EXPORTABLE: Self = Self(2u32);
    ///Bit 2.
    pub const IMPORTABLE: Self = Self(4u32);
}
impl core::ops::BitOr for ExternalMemoryFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ExternalMemoryFeatureFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ExternalMemoryFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ExternalMemoryFeatureFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ExternalMemoryFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ExternalMemoryFeatureFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ExternalMemoryFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ExternalMemoryFeatureFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::DEDICATED_ONLY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEDICATED_ONLY")?;
            remaining &= !Self::DEDICATED_ONLY.0;
            first = false;
        }
        if remaining & Self::EXPORTABLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("EXPORTABLE")?;
            remaining &= !Self::EXPORTABLE.0;
            first = false;
        }
        if remaining & Self::IMPORTABLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("IMPORTABLE")?;
            remaining &= !Self::IMPORTABLE.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkExternalMemoryFeatureFlagBitsNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalMemoryFeatureFlagBitsNV.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkExternalMemoryFeatureFlagBitsNV")]
pub struct ExternalMemoryFeatureFlagBitsNV(u32);
impl ExternalMemoryFeatureFlagBitsNV {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const DEDICATED_ONLY: Self = Self(1u32);
    ///Bit 1.
    pub const EXPORTABLE: Self = Self(2u32);
    ///Bit 2.
    pub const IMPORTABLE: Self = Self(4u32);
}
impl core::ops::BitOr for ExternalMemoryFeatureFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ExternalMemoryFeatureFlagBitsNV {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ExternalMemoryFeatureFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ExternalMemoryFeatureFlagBitsNV {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ExternalMemoryFeatureFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ExternalMemoryFeatureFlagBitsNV {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ExternalMemoryFeatureFlagBitsNV {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ExternalMemoryFeatureFlagBitsNV {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::DEDICATED_ONLY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEDICATED_ONLY")?;
            remaining &= !Self::DEDICATED_ONLY.0;
            first = false;
        }
        if remaining & Self::EXPORTABLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("EXPORTABLE")?;
            remaining &= !Self::EXPORTABLE.0;
            first = false;
        }
        if remaining & Self::IMPORTABLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("IMPORTABLE")?;
            remaining &= !Self::IMPORTABLE.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkExternalMemoryHandleTypeFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalMemoryHandleTypeFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkExternalMemoryHandleTypeFlagBits")]
pub struct ExternalMemoryHandleTypeFlagBits(u32);
impl ExternalMemoryHandleTypeFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const OPAQUE_FD: Self = Self(1u32);
    ///Bit 1.
    pub const OPAQUE_WIN32: Self = Self(2u32);
    ///Bit 2.
    pub const OPAQUE_WIN32_KMT: Self = Self(4u32);
    ///Bit 3.
    pub const D3D11_TEXTURE: Self = Self(8u32);
    ///Bit 4.
    pub const D3D11_TEXTURE_KMT: Self = Self(16u32);
    ///Bit 5.
    pub const D3D12_HEAP: Self = Self(32u32);
    ///Bit 6.
    pub const D3D12_RESOURCE: Self = Self(64u32);
    ///Bit 9.
    pub const DMA_BUF: Self = Self(512u32);
    ///Bit 10.
    pub const ANDROID_HARDWARE_BUFFER_BIT: Self = Self(1024u32);
    ///Bit 7.
    pub const HOST_ALLOCATION: Self = Self(128u32);
    ///Bit 8.
    pub const HOST_MAPPED_FOREIGN_MEMORY: Self = Self(256u32);
    ///Bit 11.
    pub const ZIRCON_VMO_BIT: Self = Self(2048u32);
    ///Bit 12.
    pub const RDMA_ADDRESS: Self = Self(4096u32);
    ///Bit 15.
    pub const OH_NATIVE_BUFFER_BIT: Self = Self(32768u32);
    ///Bit 14.
    pub const SCREEN_BUFFER_BIT: Self = Self(16384u32);
    ///Bit 16.
    pub const MTLBUFFER: Self = Self(65536u32);
    ///Bit 17.
    pub const MTLTEXTURE: Self = Self(131072u32);
    ///Bit 18.
    pub const MTLHEAP: Self = Self(262144u32);
}
impl core::ops::BitOr for ExternalMemoryHandleTypeFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ExternalMemoryHandleTypeFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ExternalMemoryHandleTypeFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ExternalMemoryHandleTypeFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ExternalMemoryHandleTypeFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ExternalMemoryHandleTypeFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ExternalMemoryHandleTypeFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ExternalMemoryHandleTypeFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::OPAQUE_FD.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("OPAQUE_FD")?;
            remaining &= !Self::OPAQUE_FD.0;
            first = false;
        }
        if remaining & Self::OPAQUE_WIN32.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("OPAQUE_WIN32")?;
            remaining &= !Self::OPAQUE_WIN32.0;
            first = false;
        }
        if remaining & Self::OPAQUE_WIN32_KMT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("OPAQUE_WIN32_KMT")?;
            remaining &= !Self::OPAQUE_WIN32_KMT.0;
            first = false;
        }
        if remaining & Self::D3D11_TEXTURE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("D3D11_TEXTURE")?;
            remaining &= !Self::D3D11_TEXTURE.0;
            first = false;
        }
        if remaining & Self::D3D11_TEXTURE_KMT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("D3D11_TEXTURE_KMT")?;
            remaining &= !Self::D3D11_TEXTURE_KMT.0;
            first = false;
        }
        if remaining & Self::D3D12_HEAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("D3D12_HEAP")?;
            remaining &= !Self::D3D12_HEAP.0;
            first = false;
        }
        if remaining & Self::D3D12_RESOURCE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("D3D12_RESOURCE")?;
            remaining &= !Self::D3D12_RESOURCE.0;
            first = false;
        }
        if remaining & Self::DMA_BUF.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DMA_BUF")?;
            remaining &= !Self::DMA_BUF.0;
            first = false;
        }
        if remaining & Self::ANDROID_HARDWARE_BUFFER_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ANDROID_HARDWARE_BUFFER_BIT")?;
            remaining &= !Self::ANDROID_HARDWARE_BUFFER_BIT.0;
            first = false;
        }
        if remaining & Self::HOST_ALLOCATION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("HOST_ALLOCATION")?;
            remaining &= !Self::HOST_ALLOCATION.0;
            first = false;
        }
        if remaining & Self::HOST_MAPPED_FOREIGN_MEMORY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("HOST_MAPPED_FOREIGN_MEMORY")?;
            remaining &= !Self::HOST_MAPPED_FOREIGN_MEMORY.0;
            first = false;
        }
        if remaining & Self::ZIRCON_VMO_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ZIRCON_VMO_BIT")?;
            remaining &= !Self::ZIRCON_VMO_BIT.0;
            first = false;
        }
        if remaining & Self::RDMA_ADDRESS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RDMA_ADDRESS")?;
            remaining &= !Self::RDMA_ADDRESS.0;
            first = false;
        }
        if remaining & Self::OH_NATIVE_BUFFER_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("OH_NATIVE_BUFFER_BIT")?;
            remaining &= !Self::OH_NATIVE_BUFFER_BIT.0;
            first = false;
        }
        if remaining & Self::SCREEN_BUFFER_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SCREEN_BUFFER_BIT")?;
            remaining &= !Self::SCREEN_BUFFER_BIT.0;
            first = false;
        }
        if remaining & Self::MTLBUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MTLBUFFER")?;
            remaining &= !Self::MTLBUFFER.0;
            first = false;
        }
        if remaining & Self::MTLTEXTURE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MTLTEXTURE")?;
            remaining &= !Self::MTLTEXTURE.0;
            first = false;
        }
        if remaining & Self::MTLHEAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MTLHEAP")?;
            remaining &= !Self::MTLHEAP.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkExternalMemoryHandleTypeFlagBitsNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalMemoryHandleTypeFlagBitsNV.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkExternalMemoryHandleTypeFlagBitsNV")]
pub struct ExternalMemoryHandleTypeFlagBitsNV(u32);
impl ExternalMemoryHandleTypeFlagBitsNV {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const OPAQUE_WIN32: Self = Self(1u32);
    ///Bit 1.
    pub const OPAQUE_WIN32_KMT: Self = Self(2u32);
    ///Bit 2.
    pub const D3D11_IMAGE: Self = Self(4u32);
    ///Bit 3.
    pub const D3D11_IMAGE_KMT: Self = Self(8u32);
}
impl core::ops::BitOr for ExternalMemoryHandleTypeFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ExternalMemoryHandleTypeFlagBitsNV {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ExternalMemoryHandleTypeFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ExternalMemoryHandleTypeFlagBitsNV {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ExternalMemoryHandleTypeFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ExternalMemoryHandleTypeFlagBitsNV {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ExternalMemoryHandleTypeFlagBitsNV {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ExternalMemoryHandleTypeFlagBitsNV {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::OPAQUE_WIN32.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("OPAQUE_WIN32")?;
            remaining &= !Self::OPAQUE_WIN32.0;
            first = false;
        }
        if remaining & Self::OPAQUE_WIN32_KMT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("OPAQUE_WIN32_KMT")?;
            remaining &= !Self::OPAQUE_WIN32_KMT.0;
            first = false;
        }
        if remaining & Self::D3D11_IMAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("D3D11_IMAGE")?;
            remaining &= !Self::D3D11_IMAGE.0;
            first = false;
        }
        if remaining & Self::D3D11_IMAGE_KMT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("D3D11_IMAGE_KMT")?;
            remaining &= !Self::D3D11_IMAGE_KMT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkExternalSemaphoreFeatureFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalSemaphoreFeatureFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkExternalSemaphoreFeatureFlagBits")]
pub struct ExternalSemaphoreFeatureFlagBits(u32);
impl ExternalSemaphoreFeatureFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const EXPORTABLE: Self = Self(1u32);
    ///Bit 1.
    pub const IMPORTABLE: Self = Self(2u32);
}
impl core::ops::BitOr for ExternalSemaphoreFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ExternalSemaphoreFeatureFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ExternalSemaphoreFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ExternalSemaphoreFeatureFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ExternalSemaphoreFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ExternalSemaphoreFeatureFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ExternalSemaphoreFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ExternalSemaphoreFeatureFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::EXPORTABLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("EXPORTABLE")?;
            remaining &= !Self::EXPORTABLE.0;
            first = false;
        }
        if remaining & Self::IMPORTABLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("IMPORTABLE")?;
            remaining &= !Self::IMPORTABLE.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkExternalSemaphoreHandleTypeFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalSemaphoreHandleTypeFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkExternalSemaphoreHandleTypeFlagBits")]
pub struct ExternalSemaphoreHandleTypeFlagBits(u32);
impl ExternalSemaphoreHandleTypeFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const OPAQUE_FD: Self = Self(1u32);
    ///Bit 1.
    pub const OPAQUE_WIN32: Self = Self(2u32);
    ///Bit 2.
    pub const OPAQUE_WIN32_KMT: Self = Self(4u32);
    ///Bit 3.
    pub const D3D12_FENCE: Self = Self(8u32);
    pub const D3D11_FENCE: Self = Self::D3D12_FENCE;
    ///Bit 4.
    pub const SYNC_FD: Self = Self(16u32);
    ///Bit 7.
    pub const ZIRCON_EVENT_BIT: Self = Self(128u32);
}
impl core::ops::BitOr for ExternalSemaphoreHandleTypeFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ExternalSemaphoreHandleTypeFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ExternalSemaphoreHandleTypeFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ExternalSemaphoreHandleTypeFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ExternalSemaphoreHandleTypeFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ExternalSemaphoreHandleTypeFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ExternalSemaphoreHandleTypeFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ExternalSemaphoreHandleTypeFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::OPAQUE_FD.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("OPAQUE_FD")?;
            remaining &= !Self::OPAQUE_FD.0;
            first = false;
        }
        if remaining & Self::OPAQUE_WIN32.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("OPAQUE_WIN32")?;
            remaining &= !Self::OPAQUE_WIN32.0;
            first = false;
        }
        if remaining & Self::OPAQUE_WIN32_KMT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("OPAQUE_WIN32_KMT")?;
            remaining &= !Self::OPAQUE_WIN32_KMT.0;
            first = false;
        }
        if remaining & Self::D3D12_FENCE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("D3D12_FENCE")?;
            remaining &= !Self::D3D12_FENCE.0;
            first = false;
        }
        if remaining & Self::SYNC_FD.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SYNC_FD")?;
            remaining &= !Self::SYNC_FD.0;
            first = false;
        }
        if remaining & Self::ZIRCON_EVENT_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ZIRCON_EVENT_BIT")?;
            remaining &= !Self::ZIRCON_EVENT_BIT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkFenceCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkFenceCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkFenceCreateFlagBits")]
pub struct FenceCreateFlagBits(u32);
impl FenceCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const SIGNALED: Self = Self(1u32);
}
impl core::ops::BitOr for FenceCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for FenceCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for FenceCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for FenceCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for FenceCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for FenceCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for FenceCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for FenceCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::SIGNALED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SIGNALED")?;
            remaining &= !Self::SIGNALED.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkFenceImportFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkFenceImportFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkFenceImportFlagBits")]
pub struct FenceImportFlagBits(u32);
impl FenceImportFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const TEMPORARY: Self = Self(1u32);
}
impl core::ops::BitOr for FenceImportFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for FenceImportFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for FenceImportFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for FenceImportFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for FenceImportFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for FenceImportFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for FenceImportFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for FenceImportFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::TEMPORARY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TEMPORARY")?;
            remaining &= !Self::TEMPORARY.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkFormatFeatureFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkFormatFeatureFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkFormatFeatureFlagBits")]
pub struct FormatFeatureFlagBits(u32);
impl FormatFeatureFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const SAMPLED_IMAGE: Self = Self(1u32);
    ///Bit 1.
    pub const STORAGE_IMAGE: Self = Self(2u32);
    ///Bit 2.
    pub const STORAGE_IMAGE_ATOMIC: Self = Self(4u32);
    ///Bit 3.
    pub const UNIFORM_TEXEL_BUFFER: Self = Self(8u32);
    ///Bit 4.
    pub const STORAGE_TEXEL_BUFFER: Self = Self(16u32);
    ///Bit 5.
    pub const STORAGE_TEXEL_BUFFER_ATOMIC: Self = Self(32u32);
    ///Bit 6.
    pub const VERTEX_BUFFER: Self = Self(64u32);
    ///Bit 7.
    pub const COLOR_ATTACHMENT: Self = Self(128u32);
    ///Bit 8.
    pub const COLOR_ATTACHMENT_BLEND: Self = Self(256u32);
    ///Bit 9.
    pub const DEPTH_STENCIL_ATTACHMENT: Self = Self(512u32);
    ///Bit 10.
    pub const BLIT_SRC: Self = Self(1024u32);
    ///Bit 11.
    pub const BLIT_DST: Self = Self(2048u32);
    ///Bit 12.
    pub const SAMPLED_IMAGE_FILTER_LINEAR: Self = Self(4096u32);
    ///Bit 14.
    pub const TRANSFER_SRC: Self = Self(16384u32);
    ///Bit 15.
    pub const TRANSFER_DST: Self = Self(32768u32);
    ///Bit 17.
    pub const MIDPOINT_CHROMA_SAMPLES: Self = Self(131072u32);
    ///Bit 18.
    pub const SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER: Self = Self(262144u32);
    ///Bit 19.
    pub const SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER: Self = Self(
        524288u32,
    );
    ///Bit 20.
    pub const SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT: Self = Self(
        1048576u32,
    );
    ///Bit 21.
    pub const SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE: Self = Self(
        2097152u32,
    );
    ///Bit 22.
    pub const DISJOINT: Self = Self(4194304u32);
    ///Bit 23.
    pub const COSITED_CHROMA_SAMPLES: Self = Self(8388608u32);
    ///Bit 16.
    pub const SAMPLED_IMAGE_FILTER_MINMAX: Self = Self(65536u32);
    pub const SAMPLED_IMAGE_FILTER_CUBIC_BIT: Self = Self::SAMPLED_IMAGE_FILTER_CUBIC;
    ///Bit 25.
    pub const VIDEO_DECODE_OUTPUT: Self = Self(33554432u32);
    ///Bit 26.
    pub const VIDEO_DECODE_DPB: Self = Self(67108864u32);
    ///Bit 29.
    pub const ACCELERATION_STRUCTURE_VERTEX_BUFFER: Self = Self(536870912u32);
    ///Bit 13.
    pub const SAMPLED_IMAGE_FILTER_CUBIC: Self = Self(8192u32);
    ///Bit 24.
    pub const FRAGMENT_DENSITY_MAP: Self = Self(16777216u32);
    ///Bit 30.
    pub const FRAGMENT_SHADING_RATE_ATTACHMENT: Self = Self(1073741824u32);
    ///Bit 27.
    pub const VIDEO_ENCODE_INPUT: Self = Self(134217728u32);
    ///Bit 28.
    pub const VIDEO_ENCODE_DPB: Self = Self(268435456u32);
}
impl core::ops::BitOr for FormatFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for FormatFeatureFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for FormatFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for FormatFeatureFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for FormatFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for FormatFeatureFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for FormatFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for FormatFeatureFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::SAMPLED_IMAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SAMPLED_IMAGE")?;
            remaining &= !Self::SAMPLED_IMAGE.0;
            first = false;
        }
        if remaining & Self::STORAGE_IMAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("STORAGE_IMAGE")?;
            remaining &= !Self::STORAGE_IMAGE.0;
            first = false;
        }
        if remaining & Self::STORAGE_IMAGE_ATOMIC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("STORAGE_IMAGE_ATOMIC")?;
            remaining &= !Self::STORAGE_IMAGE_ATOMIC.0;
            first = false;
        }
        if remaining & Self::UNIFORM_TEXEL_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("UNIFORM_TEXEL_BUFFER")?;
            remaining &= !Self::UNIFORM_TEXEL_BUFFER.0;
            first = false;
        }
        if remaining & Self::STORAGE_TEXEL_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("STORAGE_TEXEL_BUFFER")?;
            remaining &= !Self::STORAGE_TEXEL_BUFFER.0;
            first = false;
        }
        if remaining & Self::STORAGE_TEXEL_BUFFER_ATOMIC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("STORAGE_TEXEL_BUFFER_ATOMIC")?;
            remaining &= !Self::STORAGE_TEXEL_BUFFER_ATOMIC.0;
            first = false;
        }
        if remaining & Self::VERTEX_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VERTEX_BUFFER")?;
            remaining &= !Self::VERTEX_BUFFER.0;
            first = false;
        }
        if remaining & Self::COLOR_ATTACHMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("COLOR_ATTACHMENT")?;
            remaining &= !Self::COLOR_ATTACHMENT.0;
            first = false;
        }
        if remaining & Self::COLOR_ATTACHMENT_BLEND.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("COLOR_ATTACHMENT_BLEND")?;
            remaining &= !Self::COLOR_ATTACHMENT_BLEND.0;
            first = false;
        }
        if remaining & Self::DEPTH_STENCIL_ATTACHMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEPTH_STENCIL_ATTACHMENT")?;
            remaining &= !Self::DEPTH_STENCIL_ATTACHMENT.0;
            first = false;
        }
        if remaining & Self::BLIT_SRC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("BLIT_SRC")?;
            remaining &= !Self::BLIT_SRC.0;
            first = false;
        }
        if remaining & Self::BLIT_DST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("BLIT_DST")?;
            remaining &= !Self::BLIT_DST.0;
            first = false;
        }
        if remaining & Self::SAMPLED_IMAGE_FILTER_LINEAR.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SAMPLED_IMAGE_FILTER_LINEAR")?;
            remaining &= !Self::SAMPLED_IMAGE_FILTER_LINEAR.0;
            first = false;
        }
        if remaining & Self::TRANSFER_SRC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFER_SRC")?;
            remaining &= !Self::TRANSFER_SRC.0;
            first = false;
        }
        if remaining & Self::TRANSFER_DST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFER_DST")?;
            remaining &= !Self::TRANSFER_DST.0;
            first = false;
        }
        if remaining & Self::MIDPOINT_CHROMA_SAMPLES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MIDPOINT_CHROMA_SAMPLES")?;
            remaining &= !Self::MIDPOINT_CHROMA_SAMPLES.0;
            first = false;
        }
        if remaining & Self::SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER")?;
            remaining &= !Self::SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER.0;
            first = false;
        }
        if remaining
            & Self::SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER.0 != 0
        {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str(
                "SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER",
            )?;
            remaining
                &= !Self::SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER
                    .0;
            first = false;
        }
        if remaining
            & Self::SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT.0 != 0
        {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str(
                "SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT",
            )?;
            remaining
                &= !Self::SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT
                    .0;
            first = false;
        }
        if remaining
            & Self::SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE
                .0 != 0
        {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str(
                "SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE",
            )?;
            remaining
                &= !Self::SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE
                    .0;
            first = false;
        }
        if remaining & Self::DISJOINT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DISJOINT")?;
            remaining &= !Self::DISJOINT.0;
            first = false;
        }
        if remaining & Self::COSITED_CHROMA_SAMPLES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("COSITED_CHROMA_SAMPLES")?;
            remaining &= !Self::COSITED_CHROMA_SAMPLES.0;
            first = false;
        }
        if remaining & Self::SAMPLED_IMAGE_FILTER_MINMAX.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SAMPLED_IMAGE_FILTER_MINMAX")?;
            remaining &= !Self::SAMPLED_IMAGE_FILTER_MINMAX.0;
            first = false;
        }
        if remaining & Self::VIDEO_DECODE_OUTPUT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VIDEO_DECODE_OUTPUT")?;
            remaining &= !Self::VIDEO_DECODE_OUTPUT.0;
            first = false;
        }
        if remaining & Self::VIDEO_DECODE_DPB.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VIDEO_DECODE_DPB")?;
            remaining &= !Self::VIDEO_DECODE_DPB.0;
            first = false;
        }
        if remaining & Self::ACCELERATION_STRUCTURE_VERTEX_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ACCELERATION_STRUCTURE_VERTEX_BUFFER")?;
            remaining &= !Self::ACCELERATION_STRUCTURE_VERTEX_BUFFER.0;
            first = false;
        }
        if remaining & Self::SAMPLED_IMAGE_FILTER_CUBIC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SAMPLED_IMAGE_FILTER_CUBIC")?;
            remaining &= !Self::SAMPLED_IMAGE_FILTER_CUBIC.0;
            first = false;
        }
        if remaining & Self::FRAGMENT_DENSITY_MAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRAGMENT_DENSITY_MAP")?;
            remaining &= !Self::FRAGMENT_DENSITY_MAP.0;
            first = false;
        }
        if remaining & Self::FRAGMENT_SHADING_RATE_ATTACHMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRAGMENT_SHADING_RATE_ATTACHMENT")?;
            remaining &= !Self::FRAGMENT_SHADING_RATE_ATTACHMENT.0;
            first = false;
        }
        if remaining & Self::VIDEO_ENCODE_INPUT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VIDEO_ENCODE_INPUT")?;
            remaining &= !Self::VIDEO_ENCODE_INPUT.0;
            first = false;
        }
        if remaining & Self::VIDEO_ENCODE_DPB.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VIDEO_ENCODE_DPB")?;
            remaining &= !Self::VIDEO_ENCODE_DPB.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkFormatFeatureFlagBits2`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkFormatFeatureFlagBits2.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkFormatFeatureFlagBits2")]
pub struct FormatFeatureFlagBits2(u64);
impl FormatFeatureFlagBits2 {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u64)
    }
    #[inline]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u64
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const _2_SAMPLED_IMAGE: Self = Self(1u64);
    ///Bit 1.
    pub const _2_STORAGE_IMAGE: Self = Self(2u64);
    ///Bit 2.
    pub const _2_STORAGE_IMAGE_ATOMIC: Self = Self(4u64);
    ///Bit 3.
    pub const _2_UNIFORM_TEXEL_BUFFER: Self = Self(8u64);
    ///Bit 4.
    pub const _2_STORAGE_TEXEL_BUFFER: Self = Self(16u64);
    ///Bit 5.
    pub const _2_STORAGE_TEXEL_BUFFER_ATOMIC: Self = Self(32u64);
    ///Bit 6.
    pub const _2_VERTEX_BUFFER: Self = Self(64u64);
    ///Bit 7.
    pub const _2_COLOR_ATTACHMENT: Self = Self(128u64);
    ///Bit 8.
    pub const _2_COLOR_ATTACHMENT_BLEND: Self = Self(256u64);
    ///Bit 9.
    pub const _2_DEPTH_STENCIL_ATTACHMENT: Self = Self(512u64);
    ///Bit 10.
    pub const _2_BLIT_SRC: Self = Self(1024u64);
    ///Bit 11.
    pub const _2_BLIT_DST: Self = Self(2048u64);
    ///Bit 12.
    pub const _2_SAMPLED_IMAGE_FILTER_LINEAR: Self = Self(4096u64);
    ///Bit 14.
    pub const _2_TRANSFER_SRC: Self = Self(16384u64);
    ///Bit 15.
    pub const _2_TRANSFER_DST: Self = Self(32768u64);
    ///Bit 16.
    pub const _2_SAMPLED_IMAGE_FILTER_MINMAX: Self = Self(65536u64);
    ///Bit 17.
    pub const _2_MIDPOINT_CHROMA_SAMPLES: Self = Self(131072u64);
    ///Bit 18.
    pub const _2_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER: Self = Self(262144u64);
    ///Bit 19.
    pub const _2_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER: Self = Self(
        524288u64,
    );
    ///Bit 20.
    pub const _2_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT: Self = Self(
        1048576u64,
    );
    ///Bit 21.
    pub const _2_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE: Self = Self(
        2097152u64,
    );
    ///Bit 22.
    pub const _2_DISJOINT: Self = Self(4194304u64);
    ///Bit 23.
    pub const _2_COSITED_CHROMA_SAMPLES: Self = Self(8388608u64);
    ///Bit 31.
    pub const _2_STORAGE_READ_WITHOUT_FORMAT: Self = Self(2147483648u64);
    ///Bit 32.
    pub const _2_STORAGE_WRITE_WITHOUT_FORMAT: Self = Self(4294967296u64);
    ///Bit 33.
    pub const _2_SAMPLED_IMAGE_DEPTH_COMPARISON: Self = Self(8589934592u64);
    ///Bit 13.
    pub const _2_SAMPLED_IMAGE_FILTER_CUBIC: Self = Self(8192u64);
    ///Bit 46.
    pub const _2_HOST_IMAGE_TRANSFER: Self = Self(70368744177664u64);
    ///Bit 25.
    pub const _2_VIDEO_DECODE_OUTPUT: Self = Self(33554432u64);
    ///Bit 26.
    pub const _2_VIDEO_DECODE_DPB: Self = Self(67108864u64);
    ///Bit 29.
    pub const _2_ACCELERATION_STRUCTURE_VERTEX_BUFFER: Self = Self(536870912u64);
    ///Bit 24.
    pub const _2_FRAGMENT_DENSITY_MAP: Self = Self(16777216u64);
    ///Bit 30.
    pub const _2_FRAGMENT_SHADING_RATE_ATTACHMENT: Self = Self(1073741824u64);
    ///Bit 27.
    pub const _2_VIDEO_ENCODE_INPUT: Self = Self(134217728u64);
    ///Bit 28.
    pub const _2_VIDEO_ENCODE_DPB: Self = Self(268435456u64);
    ///Bit 51.
    pub const _2_ACCELERATION_STRUCTURE_RADIUS_BUFFER: Self = Self(2251799813685248u64);
    ///Bit 38.
    pub const _2_LINEAR_COLOR_ATTACHMENT: Self = Self(274877906944u64);
    ///Bit 34.
    pub const _2_WEIGHT_IMAGE_BIT: Self = Self(17179869184u64);
    ///Bit 35.
    pub const _2_WEIGHT_SAMPLED_IMAGE_BIT: Self = Self(34359738368u64);
    ///Bit 36.
    pub const _2_BLOCK_MATCHING_BIT: Self = Self(68719476736u64);
    ///Bit 37.
    pub const _2_BOX_FILTER_SAMPLED_BIT: Self = Self(137438953472u64);
    ///Bit 39.
    pub const _2_TENSOR_SHADER_BIT: Self = Self(549755813888u64);
    ///Bit 43.
    pub const _2_TENSOR_IMAGE_ALIASING_BIT: Self = Self(8796093022208u64);
    ///Bit 40.
    pub const _2_OPTICAL_FLOW_IMAGE: Self = Self(1099511627776u64);
    ///Bit 41.
    pub const _2_OPTICAL_FLOW_VECTOR: Self = Self(2199023255552u64);
    ///Bit 42.
    pub const _2_OPTICAL_FLOW_COST: Self = Self(4398046511104u64);
    ///Bit 48.
    pub const _2_TENSOR_DATA_GRAPH_BIT: Self = Self(281474976710656u64);
    ///Bit 59.
    pub const _2_COPY_IMAGE_INDIRECT_DST: Self = Self(576460752303423488u64);
    ///Bit 49.
    pub const _2_VIDEO_ENCODE_QUANTIZATION_DELTA_MAP: Self = Self(562949953421312u64);
    ///Bit 50.
    pub const _2_VIDEO_ENCODE_EMPHASIS_MAP: Self = Self(1125899906842624u64);
    ///Bit 52.
    pub const _2_DEPTH_COPY_ON_COMPUTE_QUEUE: Self = Self(4503599627370496u64);
    ///Bit 53.
    pub const _2_DEPTH_COPY_ON_TRANSFER_QUEUE: Self = Self(9007199254740992u64);
    ///Bit 54.
    pub const _2_STENCIL_COPY_ON_COMPUTE_QUEUE: Self = Self(18014398509481984u64);
    ///Bit 55.
    pub const _2_STENCIL_COPY_ON_TRANSFER_QUEUE: Self = Self(36028797018963968u64);
}
impl core::ops::BitOr for FormatFeatureFlagBits2 {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for FormatFeatureFlagBits2 {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for FormatFeatureFlagBits2 {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for FormatFeatureFlagBits2 {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for FormatFeatureFlagBits2 {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for FormatFeatureFlagBits2 {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for FormatFeatureFlagBits2 {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for FormatFeatureFlagBits2 {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::_2_SAMPLED_IMAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_SAMPLED_IMAGE")?;
            remaining &= !Self::_2_SAMPLED_IMAGE.0;
            first = false;
        }
        if remaining & Self::_2_STORAGE_IMAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_STORAGE_IMAGE")?;
            remaining &= !Self::_2_STORAGE_IMAGE.0;
            first = false;
        }
        if remaining & Self::_2_STORAGE_IMAGE_ATOMIC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_STORAGE_IMAGE_ATOMIC")?;
            remaining &= !Self::_2_STORAGE_IMAGE_ATOMIC.0;
            first = false;
        }
        if remaining & Self::_2_UNIFORM_TEXEL_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_UNIFORM_TEXEL_BUFFER")?;
            remaining &= !Self::_2_UNIFORM_TEXEL_BUFFER.0;
            first = false;
        }
        if remaining & Self::_2_STORAGE_TEXEL_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_STORAGE_TEXEL_BUFFER")?;
            remaining &= !Self::_2_STORAGE_TEXEL_BUFFER.0;
            first = false;
        }
        if remaining & Self::_2_STORAGE_TEXEL_BUFFER_ATOMIC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_STORAGE_TEXEL_BUFFER_ATOMIC")?;
            remaining &= !Self::_2_STORAGE_TEXEL_BUFFER_ATOMIC.0;
            first = false;
        }
        if remaining & Self::_2_VERTEX_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VERTEX_BUFFER")?;
            remaining &= !Self::_2_VERTEX_BUFFER.0;
            first = false;
        }
        if remaining & Self::_2_COLOR_ATTACHMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_COLOR_ATTACHMENT")?;
            remaining &= !Self::_2_COLOR_ATTACHMENT.0;
            first = false;
        }
        if remaining & Self::_2_COLOR_ATTACHMENT_BLEND.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_COLOR_ATTACHMENT_BLEND")?;
            remaining &= !Self::_2_COLOR_ATTACHMENT_BLEND.0;
            first = false;
        }
        if remaining & Self::_2_DEPTH_STENCIL_ATTACHMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_DEPTH_STENCIL_ATTACHMENT")?;
            remaining &= !Self::_2_DEPTH_STENCIL_ATTACHMENT.0;
            first = false;
        }
        if remaining & Self::_2_BLIT_SRC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_BLIT_SRC")?;
            remaining &= !Self::_2_BLIT_SRC.0;
            first = false;
        }
        if remaining & Self::_2_BLIT_DST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_BLIT_DST")?;
            remaining &= !Self::_2_BLIT_DST.0;
            first = false;
        }
        if remaining & Self::_2_SAMPLED_IMAGE_FILTER_LINEAR.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_SAMPLED_IMAGE_FILTER_LINEAR")?;
            remaining &= !Self::_2_SAMPLED_IMAGE_FILTER_LINEAR.0;
            first = false;
        }
        if remaining & Self::_2_TRANSFER_SRC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_TRANSFER_SRC")?;
            remaining &= !Self::_2_TRANSFER_SRC.0;
            first = false;
        }
        if remaining & Self::_2_TRANSFER_DST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_TRANSFER_DST")?;
            remaining &= !Self::_2_TRANSFER_DST.0;
            first = false;
        }
        if remaining & Self::_2_SAMPLED_IMAGE_FILTER_MINMAX.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_SAMPLED_IMAGE_FILTER_MINMAX")?;
            remaining &= !Self::_2_SAMPLED_IMAGE_FILTER_MINMAX.0;
            first = false;
        }
        if remaining & Self::_2_MIDPOINT_CHROMA_SAMPLES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_MIDPOINT_CHROMA_SAMPLES")?;
            remaining &= !Self::_2_MIDPOINT_CHROMA_SAMPLES.0;
            first = false;
        }
        if remaining & Self::_2_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER")?;
            remaining &= !Self::_2_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER.0;
            first = false;
        }
        if remaining
            & Self::_2_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER.0
            != 0
        {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str(
                "_2_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER",
            )?;
            remaining
                &= !Self::_2_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER
                    .0;
            first = false;
        }
        if remaining
            & Self::_2_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT.0
            != 0
        {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str(
                "_2_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT",
            )?;
            remaining
                &= !Self::_2_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT
                    .0;
            first = false;
        }
        if remaining
            & Self::_2_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE
                .0 != 0
        {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str(
                "_2_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE",
            )?;
            remaining
                &= !Self::_2_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE
                    .0;
            first = false;
        }
        if remaining & Self::_2_DISJOINT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_DISJOINT")?;
            remaining &= !Self::_2_DISJOINT.0;
            first = false;
        }
        if remaining & Self::_2_COSITED_CHROMA_SAMPLES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_COSITED_CHROMA_SAMPLES")?;
            remaining &= !Self::_2_COSITED_CHROMA_SAMPLES.0;
            first = false;
        }
        if remaining & Self::_2_STORAGE_READ_WITHOUT_FORMAT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_STORAGE_READ_WITHOUT_FORMAT")?;
            remaining &= !Self::_2_STORAGE_READ_WITHOUT_FORMAT.0;
            first = false;
        }
        if remaining & Self::_2_STORAGE_WRITE_WITHOUT_FORMAT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_STORAGE_WRITE_WITHOUT_FORMAT")?;
            remaining &= !Self::_2_STORAGE_WRITE_WITHOUT_FORMAT.0;
            first = false;
        }
        if remaining & Self::_2_SAMPLED_IMAGE_DEPTH_COMPARISON.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_SAMPLED_IMAGE_DEPTH_COMPARISON")?;
            remaining &= !Self::_2_SAMPLED_IMAGE_DEPTH_COMPARISON.0;
            first = false;
        }
        if remaining & Self::_2_SAMPLED_IMAGE_FILTER_CUBIC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_SAMPLED_IMAGE_FILTER_CUBIC")?;
            remaining &= !Self::_2_SAMPLED_IMAGE_FILTER_CUBIC.0;
            first = false;
        }
        if remaining & Self::_2_HOST_IMAGE_TRANSFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_HOST_IMAGE_TRANSFER")?;
            remaining &= !Self::_2_HOST_IMAGE_TRANSFER.0;
            first = false;
        }
        if remaining & Self::_2_VIDEO_DECODE_OUTPUT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VIDEO_DECODE_OUTPUT")?;
            remaining &= !Self::_2_VIDEO_DECODE_OUTPUT.0;
            first = false;
        }
        if remaining & Self::_2_VIDEO_DECODE_DPB.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VIDEO_DECODE_DPB")?;
            remaining &= !Self::_2_VIDEO_DECODE_DPB.0;
            first = false;
        }
        if remaining & Self::_2_ACCELERATION_STRUCTURE_VERTEX_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_ACCELERATION_STRUCTURE_VERTEX_BUFFER")?;
            remaining &= !Self::_2_ACCELERATION_STRUCTURE_VERTEX_BUFFER.0;
            first = false;
        }
        if remaining & Self::_2_FRAGMENT_DENSITY_MAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_FRAGMENT_DENSITY_MAP")?;
            remaining &= !Self::_2_FRAGMENT_DENSITY_MAP.0;
            first = false;
        }
        if remaining & Self::_2_FRAGMENT_SHADING_RATE_ATTACHMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_FRAGMENT_SHADING_RATE_ATTACHMENT")?;
            remaining &= !Self::_2_FRAGMENT_SHADING_RATE_ATTACHMENT.0;
            first = false;
        }
        if remaining & Self::_2_VIDEO_ENCODE_INPUT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VIDEO_ENCODE_INPUT")?;
            remaining &= !Self::_2_VIDEO_ENCODE_INPUT.0;
            first = false;
        }
        if remaining & Self::_2_VIDEO_ENCODE_DPB.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VIDEO_ENCODE_DPB")?;
            remaining &= !Self::_2_VIDEO_ENCODE_DPB.0;
            first = false;
        }
        if remaining & Self::_2_ACCELERATION_STRUCTURE_RADIUS_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_ACCELERATION_STRUCTURE_RADIUS_BUFFER")?;
            remaining &= !Self::_2_ACCELERATION_STRUCTURE_RADIUS_BUFFER.0;
            first = false;
        }
        if remaining & Self::_2_LINEAR_COLOR_ATTACHMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_LINEAR_COLOR_ATTACHMENT")?;
            remaining &= !Self::_2_LINEAR_COLOR_ATTACHMENT.0;
            first = false;
        }
        if remaining & Self::_2_WEIGHT_IMAGE_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_WEIGHT_IMAGE_BIT")?;
            remaining &= !Self::_2_WEIGHT_IMAGE_BIT.0;
            first = false;
        }
        if remaining & Self::_2_WEIGHT_SAMPLED_IMAGE_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_WEIGHT_SAMPLED_IMAGE_BIT")?;
            remaining &= !Self::_2_WEIGHT_SAMPLED_IMAGE_BIT.0;
            first = false;
        }
        if remaining & Self::_2_BLOCK_MATCHING_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_BLOCK_MATCHING_BIT")?;
            remaining &= !Self::_2_BLOCK_MATCHING_BIT.0;
            first = false;
        }
        if remaining & Self::_2_BOX_FILTER_SAMPLED_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_BOX_FILTER_SAMPLED_BIT")?;
            remaining &= !Self::_2_BOX_FILTER_SAMPLED_BIT.0;
            first = false;
        }
        if remaining & Self::_2_TENSOR_SHADER_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_TENSOR_SHADER_BIT")?;
            remaining &= !Self::_2_TENSOR_SHADER_BIT.0;
            first = false;
        }
        if remaining & Self::_2_TENSOR_IMAGE_ALIASING_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_TENSOR_IMAGE_ALIASING_BIT")?;
            remaining &= !Self::_2_TENSOR_IMAGE_ALIASING_BIT.0;
            first = false;
        }
        if remaining & Self::_2_OPTICAL_FLOW_IMAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_OPTICAL_FLOW_IMAGE")?;
            remaining &= !Self::_2_OPTICAL_FLOW_IMAGE.0;
            first = false;
        }
        if remaining & Self::_2_OPTICAL_FLOW_VECTOR.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_OPTICAL_FLOW_VECTOR")?;
            remaining &= !Self::_2_OPTICAL_FLOW_VECTOR.0;
            first = false;
        }
        if remaining & Self::_2_OPTICAL_FLOW_COST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_OPTICAL_FLOW_COST")?;
            remaining &= !Self::_2_OPTICAL_FLOW_COST.0;
            first = false;
        }
        if remaining & Self::_2_TENSOR_DATA_GRAPH_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_TENSOR_DATA_GRAPH_BIT")?;
            remaining &= !Self::_2_TENSOR_DATA_GRAPH_BIT.0;
            first = false;
        }
        if remaining & Self::_2_COPY_IMAGE_INDIRECT_DST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_COPY_IMAGE_INDIRECT_DST")?;
            remaining &= !Self::_2_COPY_IMAGE_INDIRECT_DST.0;
            first = false;
        }
        if remaining & Self::_2_VIDEO_ENCODE_QUANTIZATION_DELTA_MAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VIDEO_ENCODE_QUANTIZATION_DELTA_MAP")?;
            remaining &= !Self::_2_VIDEO_ENCODE_QUANTIZATION_DELTA_MAP.0;
            first = false;
        }
        if remaining & Self::_2_VIDEO_ENCODE_EMPHASIS_MAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VIDEO_ENCODE_EMPHASIS_MAP")?;
            remaining &= !Self::_2_VIDEO_ENCODE_EMPHASIS_MAP.0;
            first = false;
        }
        if remaining & Self::_2_DEPTH_COPY_ON_COMPUTE_QUEUE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_DEPTH_COPY_ON_COMPUTE_QUEUE")?;
            remaining &= !Self::_2_DEPTH_COPY_ON_COMPUTE_QUEUE.0;
            first = false;
        }
        if remaining & Self::_2_DEPTH_COPY_ON_TRANSFER_QUEUE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_DEPTH_COPY_ON_TRANSFER_QUEUE")?;
            remaining &= !Self::_2_DEPTH_COPY_ON_TRANSFER_QUEUE.0;
            first = false;
        }
        if remaining & Self::_2_STENCIL_COPY_ON_COMPUTE_QUEUE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_STENCIL_COPY_ON_COMPUTE_QUEUE")?;
            remaining &= !Self::_2_STENCIL_COPY_ON_COMPUTE_QUEUE.0;
            first = false;
        }
        if remaining & Self::_2_STENCIL_COPY_ON_TRANSFER_QUEUE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_STENCIL_COPY_ON_TRANSFER_QUEUE")?;
            remaining &= !Self::_2_STENCIL_COPY_ON_TRANSFER_QUEUE.0;
            first = false;
        }
        if remaining != 0u64 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkFrameBoundaryFlagBitsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkFrameBoundaryFlagBitsEXT.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkFrameBoundaryFlagBitsEXT")]
pub struct FrameBoundaryFlagBitsEXT(u32);
impl FrameBoundaryFlagBitsEXT {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const FRAME_END: Self = Self(1u32);
}
impl core::ops::BitOr for FrameBoundaryFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for FrameBoundaryFlagBitsEXT {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for FrameBoundaryFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for FrameBoundaryFlagBitsEXT {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for FrameBoundaryFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for FrameBoundaryFlagBitsEXT {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for FrameBoundaryFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for FrameBoundaryFlagBitsEXT {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::FRAME_END.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRAME_END")?;
            remaining &= !Self::FRAME_END.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkFramebufferCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkFramebufferCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkFramebufferCreateFlagBits")]
pub struct FramebufferCreateFlagBits(u32);
impl FramebufferCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const IMAGELESS: Self = Self(1u32);
}
impl core::ops::BitOr for FramebufferCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for FramebufferCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for FramebufferCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for FramebufferCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for FramebufferCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for FramebufferCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for FramebufferCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for FramebufferCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::IMAGELESS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("IMAGELESS")?;
            remaining &= !Self::IMAGELESS.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkGeometryFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkGeometryFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkGeometryFlagBitsKHR")]
pub struct GeometryFlagBitsKHR(u32);
impl GeometryFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const OPAQUE: Self = Self(1u32);
    ///Bit 1.
    pub const NO_DUPLICATE_ANY_HIT_INVOCATION: Self = Self(2u32);
}
impl core::ops::BitOr for GeometryFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for GeometryFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for GeometryFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for GeometryFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for GeometryFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for GeometryFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for GeometryFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for GeometryFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::OPAQUE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("OPAQUE")?;
            remaining &= !Self::OPAQUE.0;
            first = false;
        }
        if remaining & Self::NO_DUPLICATE_ANY_HIT_INVOCATION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("NO_DUPLICATE_ANY_HIT_INVOCATION")?;
            remaining &= !Self::NO_DUPLICATE_ANY_HIT_INVOCATION.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkGeometryInstanceFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkGeometryInstanceFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkGeometryInstanceFlagBitsKHR")]
pub struct GeometryInstanceFlagBitsKHR(u32);
impl GeometryInstanceFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const TRIANGLE_FACING_CULL_DISABLE: Self = Self(1u32);
    ///Bit 1.
    pub const TRIANGLE_FLIP_FACING: Self = Self(2u32);
    ///Bit 2.
    pub const FORCE_OPAQUE: Self = Self(4u32);
    ///Bit 3.
    pub const FORCE_NO_OPAQUE: Self = Self(8u32);
    pub const TRIANGLE_FRONT_COUNTERCLOCKWISE: Self = Self::TRIANGLE_FLIP_FACING;
    pub const TRIANGLE_CULL_DISABLE: Self = Self::TRIANGLE_FACING_CULL_DISABLE;
    ///Bit 4.
    pub const FORCE_OPACITY_MICROMAP_2_STATE: Self = Self(16u32);
    ///Bit 5.
    pub const DISABLE_OPACITY_MICROMAPS: Self = Self(32u32);
}
impl core::ops::BitOr for GeometryInstanceFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for GeometryInstanceFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for GeometryInstanceFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for GeometryInstanceFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for GeometryInstanceFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for GeometryInstanceFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for GeometryInstanceFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for GeometryInstanceFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::TRIANGLE_FACING_CULL_DISABLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRIANGLE_FACING_CULL_DISABLE")?;
            remaining &= !Self::TRIANGLE_FACING_CULL_DISABLE.0;
            first = false;
        }
        if remaining & Self::TRIANGLE_FLIP_FACING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRIANGLE_FLIP_FACING")?;
            remaining &= !Self::TRIANGLE_FLIP_FACING.0;
            first = false;
        }
        if remaining & Self::FORCE_OPAQUE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FORCE_OPAQUE")?;
            remaining &= !Self::FORCE_OPAQUE.0;
            first = false;
        }
        if remaining & Self::FORCE_NO_OPAQUE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FORCE_NO_OPAQUE")?;
            remaining &= !Self::FORCE_NO_OPAQUE.0;
            first = false;
        }
        if remaining & Self::FORCE_OPACITY_MICROMAP_2_STATE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FORCE_OPACITY_MICROMAP_2_STATE")?;
            remaining &= !Self::FORCE_OPACITY_MICROMAP_2_STATE.0;
            first = false;
        }
        if remaining & Self::DISABLE_OPACITY_MICROMAPS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DISABLE_OPACITY_MICROMAPS")?;
            remaining &= !Self::DISABLE_OPACITY_MICROMAPS.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkGraphicsPipelineLibraryFlagBitsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkGraphicsPipelineLibraryFlagBitsEXT.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkGraphicsPipelineLibraryFlagBitsEXT")]
pub struct GraphicsPipelineLibraryFlagBitsEXT(u32);
impl GraphicsPipelineLibraryFlagBitsEXT {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const VERTEX_INPUT_INTERFACE: Self = Self(1u32);
    ///Bit 1.
    pub const PRE_RASTERIZATION_SHADERS: Self = Self(2u32);
    ///Bit 2.
    pub const FRAGMENT_SHADER: Self = Self(4u32);
    ///Bit 3.
    pub const FRAGMENT_OUTPUT_INTERFACE: Self = Self(8u32);
}
impl core::ops::BitOr for GraphicsPipelineLibraryFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for GraphicsPipelineLibraryFlagBitsEXT {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for GraphicsPipelineLibraryFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for GraphicsPipelineLibraryFlagBitsEXT {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for GraphicsPipelineLibraryFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for GraphicsPipelineLibraryFlagBitsEXT {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for GraphicsPipelineLibraryFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for GraphicsPipelineLibraryFlagBitsEXT {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::VERTEX_INPUT_INTERFACE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VERTEX_INPUT_INTERFACE")?;
            remaining &= !Self::VERTEX_INPUT_INTERFACE.0;
            first = false;
        }
        if remaining & Self::PRE_RASTERIZATION_SHADERS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PRE_RASTERIZATION_SHADERS")?;
            remaining &= !Self::PRE_RASTERIZATION_SHADERS.0;
            first = false;
        }
        if remaining & Self::FRAGMENT_SHADER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRAGMENT_SHADER")?;
            remaining &= !Self::FRAGMENT_SHADER.0;
            first = false;
        }
        if remaining & Self::FRAGMENT_OUTPUT_INTERFACE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRAGMENT_OUTPUT_INTERFACE")?;
            remaining &= !Self::FRAGMENT_OUTPUT_INTERFACE.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkHostImageCopyFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkHostImageCopyFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkHostImageCopyFlagBits")]
pub struct HostImageCopyFlagBits(u32);
impl HostImageCopyFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const MEMCPY: Self = Self(1u32);
}
impl core::ops::BitOr for HostImageCopyFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for HostImageCopyFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for HostImageCopyFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for HostImageCopyFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for HostImageCopyFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for HostImageCopyFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for HostImageCopyFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for HostImageCopyFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::MEMCPY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MEMCPY")?;
            remaining &= !Self::MEMCPY.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkImageAspectFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageAspectFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkImageAspectFlagBits")]
pub struct ImageAspectFlagBits(u32);
impl ImageAspectFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const COLOR: Self = Self(1u32);
    ///Bit 1.
    pub const DEPTH: Self = Self(2u32);
    ///Bit 2.
    pub const STENCIL: Self = Self(4u32);
    ///Bit 3.
    pub const METADATA: Self = Self(8u32);
    ///Bit 4.
    pub const PLANE_0: Self = Self(16u32);
    ///Bit 5.
    pub const PLANE_1: Self = Self(32u32);
    ///Bit 6.
    pub const PLANE_2: Self = Self(64u32);
    pub const NONE: Self = Self(0u32);
    ///Bit 7.
    pub const MEMORY_PLANE_0: Self = Self(128u32);
    ///Bit 8.
    pub const MEMORY_PLANE_1: Self = Self(256u32);
    ///Bit 9.
    pub const MEMORY_PLANE_2: Self = Self(512u32);
    ///Bit 10.
    pub const MEMORY_PLANE_3: Self = Self(1024u32);
}
impl core::ops::BitOr for ImageAspectFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ImageAspectFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ImageAspectFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ImageAspectFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ImageAspectFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ImageAspectFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ImageAspectFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ImageAspectFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::COLOR.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("COLOR")?;
            remaining &= !Self::COLOR.0;
            first = false;
        }
        if remaining & Self::DEPTH.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEPTH")?;
            remaining &= !Self::DEPTH.0;
            first = false;
        }
        if remaining & Self::STENCIL.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("STENCIL")?;
            remaining &= !Self::STENCIL.0;
            first = false;
        }
        if remaining & Self::METADATA.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("METADATA")?;
            remaining &= !Self::METADATA.0;
            first = false;
        }
        if remaining & Self::PLANE_0.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PLANE_0")?;
            remaining &= !Self::PLANE_0.0;
            first = false;
        }
        if remaining & Self::PLANE_1.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PLANE_1")?;
            remaining &= !Self::PLANE_1.0;
            first = false;
        }
        if remaining & Self::PLANE_2.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PLANE_2")?;
            remaining &= !Self::PLANE_2.0;
            first = false;
        }
        if remaining & Self::MEMORY_PLANE_0.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MEMORY_PLANE_0")?;
            remaining &= !Self::MEMORY_PLANE_0.0;
            first = false;
        }
        if remaining & Self::MEMORY_PLANE_1.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MEMORY_PLANE_1")?;
            remaining &= !Self::MEMORY_PLANE_1.0;
            first = false;
        }
        if remaining & Self::MEMORY_PLANE_2.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MEMORY_PLANE_2")?;
            remaining &= !Self::MEMORY_PLANE_2.0;
            first = false;
        }
        if remaining & Self::MEMORY_PLANE_3.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MEMORY_PLANE_3")?;
            remaining &= !Self::MEMORY_PLANE_3.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkImageCompressionFixedRateFlagBitsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageCompressionFixedRateFlagBitsEXT.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkImageCompressionFixedRateFlagBitsEXT")]
pub struct ImageCompressionFixedRateFlagBitsEXT(u32);
impl ImageCompressionFixedRateFlagBitsEXT {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const NONE: Self = Self(0u32);
    ///Bit 0.
    pub const _1BPC: Self = Self(1u32);
    ///Bit 1.
    pub const _2BPC: Self = Self(2u32);
    ///Bit 2.
    pub const _3BPC: Self = Self(4u32);
    ///Bit 3.
    pub const _4BPC: Self = Self(8u32);
    ///Bit 4.
    pub const _5BPC: Self = Self(16u32);
    ///Bit 5.
    pub const _6BPC: Self = Self(32u32);
    ///Bit 6.
    pub const _7BPC: Self = Self(64u32);
    ///Bit 7.
    pub const _8BPC: Self = Self(128u32);
    ///Bit 8.
    pub const _9BPC: Self = Self(256u32);
    ///Bit 9.
    pub const _10BPC: Self = Self(512u32);
    ///Bit 10.
    pub const _11BPC: Self = Self(1024u32);
    ///Bit 11.
    pub const _12BPC: Self = Self(2048u32);
    ///Bit 12.
    pub const _13BPC: Self = Self(4096u32);
    ///Bit 13.
    pub const _14BPC: Self = Self(8192u32);
    ///Bit 14.
    pub const _15BPC: Self = Self(16384u32);
    ///Bit 15.
    pub const _16BPC: Self = Self(32768u32);
    ///Bit 16.
    pub const _17BPC: Self = Self(65536u32);
    ///Bit 17.
    pub const _18BPC: Self = Self(131072u32);
    ///Bit 18.
    pub const _19BPC: Self = Self(262144u32);
    ///Bit 19.
    pub const _20BPC: Self = Self(524288u32);
    ///Bit 20.
    pub const _21BPC: Self = Self(1048576u32);
    ///Bit 21.
    pub const _22BPC: Self = Self(2097152u32);
    ///Bit 22.
    pub const _23BPC: Self = Self(4194304u32);
    ///Bit 23.
    pub const _24BPC: Self = Self(8388608u32);
}
impl core::ops::BitOr for ImageCompressionFixedRateFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ImageCompressionFixedRateFlagBitsEXT {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ImageCompressionFixedRateFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ImageCompressionFixedRateFlagBitsEXT {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ImageCompressionFixedRateFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ImageCompressionFixedRateFlagBitsEXT {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ImageCompressionFixedRateFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ImageCompressionFixedRateFlagBitsEXT {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::_1BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_1BPC")?;
            remaining &= !Self::_1BPC.0;
            first = false;
        }
        if remaining & Self::_2BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2BPC")?;
            remaining &= !Self::_2BPC.0;
            first = false;
        }
        if remaining & Self::_3BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_3BPC")?;
            remaining &= !Self::_3BPC.0;
            first = false;
        }
        if remaining & Self::_4BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_4BPC")?;
            remaining &= !Self::_4BPC.0;
            first = false;
        }
        if remaining & Self::_5BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_5BPC")?;
            remaining &= !Self::_5BPC.0;
            first = false;
        }
        if remaining & Self::_6BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_6BPC")?;
            remaining &= !Self::_6BPC.0;
            first = false;
        }
        if remaining & Self::_7BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_7BPC")?;
            remaining &= !Self::_7BPC.0;
            first = false;
        }
        if remaining & Self::_8BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_8BPC")?;
            remaining &= !Self::_8BPC.0;
            first = false;
        }
        if remaining & Self::_9BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_9BPC")?;
            remaining &= !Self::_9BPC.0;
            first = false;
        }
        if remaining & Self::_10BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_10BPC")?;
            remaining &= !Self::_10BPC.0;
            first = false;
        }
        if remaining & Self::_11BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_11BPC")?;
            remaining &= !Self::_11BPC.0;
            first = false;
        }
        if remaining & Self::_12BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_12BPC")?;
            remaining &= !Self::_12BPC.0;
            first = false;
        }
        if remaining & Self::_13BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_13BPC")?;
            remaining &= !Self::_13BPC.0;
            first = false;
        }
        if remaining & Self::_14BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_14BPC")?;
            remaining &= !Self::_14BPC.0;
            first = false;
        }
        if remaining & Self::_15BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_15BPC")?;
            remaining &= !Self::_15BPC.0;
            first = false;
        }
        if remaining & Self::_16BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_16BPC")?;
            remaining &= !Self::_16BPC.0;
            first = false;
        }
        if remaining & Self::_17BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_17BPC")?;
            remaining &= !Self::_17BPC.0;
            first = false;
        }
        if remaining & Self::_18BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_18BPC")?;
            remaining &= !Self::_18BPC.0;
            first = false;
        }
        if remaining & Self::_19BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_19BPC")?;
            remaining &= !Self::_19BPC.0;
            first = false;
        }
        if remaining & Self::_20BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_20BPC")?;
            remaining &= !Self::_20BPC.0;
            first = false;
        }
        if remaining & Self::_21BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_21BPC")?;
            remaining &= !Self::_21BPC.0;
            first = false;
        }
        if remaining & Self::_22BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_22BPC")?;
            remaining &= !Self::_22BPC.0;
            first = false;
        }
        if remaining & Self::_23BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_23BPC")?;
            remaining &= !Self::_23BPC.0;
            first = false;
        }
        if remaining & Self::_24BPC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_24BPC")?;
            remaining &= !Self::_24BPC.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkImageCompressionFlagBitsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageCompressionFlagBitsEXT.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkImageCompressionFlagBitsEXT")]
pub struct ImageCompressionFlagBitsEXT(u32);
impl ImageCompressionFlagBitsEXT {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const DEFAULT: Self = Self(0u32);
    ///Bit 0.
    pub const FIXED_RATE_DEFAULT: Self = Self(1u32);
    ///Bit 1.
    pub const FIXED_RATE_EXPLICIT: Self = Self(2u32);
    ///Bit 2.
    pub const DISABLED: Self = Self(4u32);
}
impl core::ops::BitOr for ImageCompressionFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ImageCompressionFlagBitsEXT {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ImageCompressionFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ImageCompressionFlagBitsEXT {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ImageCompressionFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ImageCompressionFlagBitsEXT {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ImageCompressionFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ImageCompressionFlagBitsEXT {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::FIXED_RATE_DEFAULT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FIXED_RATE_DEFAULT")?;
            remaining &= !Self::FIXED_RATE_DEFAULT.0;
            first = false;
        }
        if remaining & Self::FIXED_RATE_EXPLICIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FIXED_RATE_EXPLICIT")?;
            remaining &= !Self::FIXED_RATE_EXPLICIT.0;
            first = false;
        }
        if remaining & Self::DISABLED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DISABLED")?;
            remaining &= !Self::DISABLED.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkImageConstraintsInfoFlagBitsFUCHSIA`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageConstraintsInfoFlagBitsFUCHSIA.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkImageConstraintsInfoFlagBitsFUCHSIA")]
pub struct ImageConstraintsInfoFlagBitsFUCHSIA(u32);
impl ImageConstraintsInfoFlagBitsFUCHSIA {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const CPU_READ_RARELY: Self = Self(1u32);
    ///Bit 1.
    pub const CPU_READ_OFTEN: Self = Self(2u32);
    ///Bit 2.
    pub const CPU_WRITE_RARELY: Self = Self(4u32);
    ///Bit 3.
    pub const CPU_WRITE_OFTEN: Self = Self(8u32);
    ///Bit 4.
    pub const PROTECTED_OPTIONAL: Self = Self(16u32);
}
impl core::ops::BitOr for ImageConstraintsInfoFlagBitsFUCHSIA {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ImageConstraintsInfoFlagBitsFUCHSIA {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ImageConstraintsInfoFlagBitsFUCHSIA {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ImageConstraintsInfoFlagBitsFUCHSIA {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ImageConstraintsInfoFlagBitsFUCHSIA {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ImageConstraintsInfoFlagBitsFUCHSIA {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ImageConstraintsInfoFlagBitsFUCHSIA {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ImageConstraintsInfoFlagBitsFUCHSIA {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::CPU_READ_RARELY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CPU_READ_RARELY")?;
            remaining &= !Self::CPU_READ_RARELY.0;
            first = false;
        }
        if remaining & Self::CPU_READ_OFTEN.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CPU_READ_OFTEN")?;
            remaining &= !Self::CPU_READ_OFTEN.0;
            first = false;
        }
        if remaining & Self::CPU_WRITE_RARELY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CPU_WRITE_RARELY")?;
            remaining &= !Self::CPU_WRITE_RARELY.0;
            first = false;
        }
        if remaining & Self::CPU_WRITE_OFTEN.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CPU_WRITE_OFTEN")?;
            remaining &= !Self::CPU_WRITE_OFTEN.0;
            first = false;
        }
        if remaining & Self::PROTECTED_OPTIONAL.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PROTECTED_OPTIONAL")?;
            remaining &= !Self::PROTECTED_OPTIONAL.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkImageCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkImageCreateFlagBits")]
pub struct ImageCreateFlagBits(u32);
impl ImageCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const SPARSE_BINDING: Self = Self(1u32);
    ///Bit 1.
    pub const SPARSE_RESIDENCY: Self = Self(2u32);
    ///Bit 2.
    pub const SPARSE_ALIASED: Self = Self(4u32);
    ///Bit 3.
    pub const MUTABLE_FORMAT: Self = Self(8u32);
    ///Bit 4.
    pub const CUBE_COMPATIBLE: Self = Self(16u32);
    ///Bit 10.
    pub const ALIAS: Self = Self(1024u32);
    ///Bit 6.
    pub const SPLIT_INSTANCE_BIND_REGIONS: Self = Self(64u32);
    ///Bit 5.
    pub const _2D_ARRAY_COMPATIBLE: Self = Self(32u32);
    ///Bit 7.
    pub const BLOCK_TEXEL_VIEW_COMPATIBLE: Self = Self(128u32);
    ///Bit 8.
    pub const EXTENDED_USAGE: Self = Self(256u32);
    ///Bit 11.
    pub const PROTECTED: Self = Self(2048u32);
    ///Bit 9.
    pub const DISJOINT: Self = Self(512u32);
    ///Bit 13.
    pub const CORNER_SAMPLED: Self = Self(8192u32);
    ///Bit 16.
    pub const DESCRIPTOR_HEAP_CAPTURE_REPLAY: Self = Self(65536u32);
    ///Bit 12.
    pub const SAMPLE_LOCATIONS_COMPATIBLE_DEPTH: Self = Self(4096u32);
    ///Bit 14.
    pub const SUBSAMPLED: Self = Self(16384u32);
    pub const DESCRIPTOR_BUFFER_CAPTURE_REPLAY: Self = Self::DESCRIPTOR_HEAP_CAPTURE_REPLAY;
    ///Bit 18.
    pub const MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED: Self = Self(262144u32);
    ///Bit 17.
    pub const _2D_VIEW_COMPATIBLE: Self = Self(131072u32);
    pub const FRAGMENT_DENSITY_MAP_OFFSET_BIT: Self = Self::FRAGMENT_DENSITY_MAP_OFFSET;
    ///Bit 20.
    pub const VIDEO_PROFILE_INDEPENDENT: Self = Self(1048576u32);
    ///Bit 15.
    pub const FRAGMENT_DENSITY_MAP_OFFSET: Self = Self(32768u32);
}
impl core::ops::BitOr for ImageCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ImageCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ImageCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ImageCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ImageCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ImageCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ImageCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ImageCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::SPARSE_BINDING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SPARSE_BINDING")?;
            remaining &= !Self::SPARSE_BINDING.0;
            first = false;
        }
        if remaining & Self::SPARSE_RESIDENCY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SPARSE_RESIDENCY")?;
            remaining &= !Self::SPARSE_RESIDENCY.0;
            first = false;
        }
        if remaining & Self::SPARSE_ALIASED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SPARSE_ALIASED")?;
            remaining &= !Self::SPARSE_ALIASED.0;
            first = false;
        }
        if remaining & Self::MUTABLE_FORMAT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MUTABLE_FORMAT")?;
            remaining &= !Self::MUTABLE_FORMAT.0;
            first = false;
        }
        if remaining & Self::CUBE_COMPATIBLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CUBE_COMPATIBLE")?;
            remaining &= !Self::CUBE_COMPATIBLE.0;
            first = false;
        }
        if remaining & Self::ALIAS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALIAS")?;
            remaining &= !Self::ALIAS.0;
            first = false;
        }
        if remaining & Self::SPLIT_INSTANCE_BIND_REGIONS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SPLIT_INSTANCE_BIND_REGIONS")?;
            remaining &= !Self::SPLIT_INSTANCE_BIND_REGIONS.0;
            first = false;
        }
        if remaining & Self::_2D_ARRAY_COMPATIBLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2D_ARRAY_COMPATIBLE")?;
            remaining &= !Self::_2D_ARRAY_COMPATIBLE.0;
            first = false;
        }
        if remaining & Self::BLOCK_TEXEL_VIEW_COMPATIBLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("BLOCK_TEXEL_VIEW_COMPATIBLE")?;
            remaining &= !Self::BLOCK_TEXEL_VIEW_COMPATIBLE.0;
            first = false;
        }
        if remaining & Self::EXTENDED_USAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("EXTENDED_USAGE")?;
            remaining &= !Self::EXTENDED_USAGE.0;
            first = false;
        }
        if remaining & Self::PROTECTED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PROTECTED")?;
            remaining &= !Self::PROTECTED.0;
            first = false;
        }
        if remaining & Self::DISJOINT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DISJOINT")?;
            remaining &= !Self::DISJOINT.0;
            first = false;
        }
        if remaining & Self::CORNER_SAMPLED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CORNER_SAMPLED")?;
            remaining &= !Self::CORNER_SAMPLED.0;
            first = false;
        }
        if remaining & Self::DESCRIPTOR_HEAP_CAPTURE_REPLAY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DESCRIPTOR_HEAP_CAPTURE_REPLAY")?;
            remaining &= !Self::DESCRIPTOR_HEAP_CAPTURE_REPLAY.0;
            first = false;
        }
        if remaining & Self::SAMPLE_LOCATIONS_COMPATIBLE_DEPTH.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SAMPLE_LOCATIONS_COMPATIBLE_DEPTH")?;
            remaining &= !Self::SAMPLE_LOCATIONS_COMPATIBLE_DEPTH.0;
            first = false;
        }
        if remaining & Self::SUBSAMPLED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SUBSAMPLED")?;
            remaining &= !Self::SUBSAMPLED.0;
            first = false;
        }
        if remaining & Self::MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED")?;
            remaining &= !Self::MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED.0;
            first = false;
        }
        if remaining & Self::_2D_VIEW_COMPATIBLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2D_VIEW_COMPATIBLE")?;
            remaining &= !Self::_2D_VIEW_COMPATIBLE.0;
            first = false;
        }
        if remaining & Self::VIDEO_PROFILE_INDEPENDENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VIDEO_PROFILE_INDEPENDENT")?;
            remaining &= !Self::VIDEO_PROFILE_INDEPENDENT.0;
            first = false;
        }
        if remaining & Self::FRAGMENT_DENSITY_MAP_OFFSET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRAGMENT_DENSITY_MAP_OFFSET")?;
            remaining &= !Self::FRAGMENT_DENSITY_MAP_OFFSET.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkImageFormatConstraintsFlagBitsFUCHSIA`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageFormatConstraintsFlagBitsFUCHSIA.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkImageFormatConstraintsFlagBitsFUCHSIA")]
pub struct ImageFormatConstraintsFlagBitsFUCHSIA(u32);
impl ImageFormatConstraintsFlagBitsFUCHSIA {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}
impl core::ops::BitOr for ImageFormatConstraintsFlagBitsFUCHSIA {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ImageFormatConstraintsFlagBitsFUCHSIA {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ImageFormatConstraintsFlagBitsFUCHSIA {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ImageFormatConstraintsFlagBitsFUCHSIA {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ImageFormatConstraintsFlagBitsFUCHSIA {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ImageFormatConstraintsFlagBitsFUCHSIA {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ImageFormatConstraintsFlagBitsFUCHSIA {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ImageFormatConstraintsFlagBitsFUCHSIA {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkImageUsageFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageUsageFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkImageUsageFlagBits")]
pub struct ImageUsageFlagBits(u32);
impl ImageUsageFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const TRANSFER_SRC: Self = Self(1u32);
    ///Bit 1.
    pub const TRANSFER_DST: Self = Self(2u32);
    ///Bit 2.
    pub const SAMPLED: Self = Self(4u32);
    ///Bit 3.
    pub const STORAGE: Self = Self(8u32);
    ///Bit 4.
    pub const COLOR_ATTACHMENT: Self = Self(16u32);
    ///Bit 5.
    pub const DEPTH_STENCIL_ATTACHMENT: Self = Self(32u32);
    ///Bit 6.
    pub const TRANSIENT_ATTACHMENT: Self = Self(64u32);
    ///Bit 7.
    pub const INPUT_ATTACHMENT: Self = Self(128u32);
    ///Bit 22.
    pub const HOST_TRANSFER: Self = Self(4194304u32);
    ///Bit 10.
    pub const VIDEO_DECODE_DST: Self = Self(1024u32);
    ///Bit 11.
    pub const VIDEO_DECODE_SRC: Self = Self(2048u32);
    ///Bit 12.
    pub const VIDEO_DECODE_DPB: Self = Self(4096u32);
    pub const SHADING_RATE_IMAGE: Self = Self::FRAGMENT_SHADING_RATE_ATTACHMENT;
    ///Bit 9.
    pub const FRAGMENT_DENSITY_MAP: Self = Self(512u32);
    ///Bit 8.
    pub const FRAGMENT_SHADING_RATE_ATTACHMENT: Self = Self(256u32);
    ///Bit 13.
    pub const VIDEO_ENCODE_DST: Self = Self(8192u32);
    ///Bit 14.
    pub const VIDEO_ENCODE_SRC: Self = Self(16384u32);
    ///Bit 15.
    pub const VIDEO_ENCODE_DPB: Self = Self(32768u32);
    ///Bit 19.
    pub const ATTACHMENT_FEEDBACK_LOOP: Self = Self(524288u32);
    ///Bit 18.
    pub const INVOCATION_MASK_BIT: Self = Self(262144u32);
    ///Bit 20.
    pub const SAMPLE_WEIGHT_BIT: Self = Self(1048576u32);
    ///Bit 21.
    pub const SAMPLE_BLOCK_MATCH_BIT: Self = Self(2097152u32);
    ///Bit 23.
    pub const TENSOR_ALIASING_BIT: Self = Self(8388608u32);
    ///Bit 27.
    pub const TILE_MEMORY_BIT: Self = Self(134217728u32);
    ///Bit 25.
    pub const VIDEO_ENCODE_QUANTIZATION_DELTA_MAP: Self = Self(33554432u32);
    ///Bit 26.
    pub const VIDEO_ENCODE_EMPHASIS_MAP: Self = Self(67108864u32);
}
impl core::ops::BitOr for ImageUsageFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ImageUsageFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ImageUsageFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ImageUsageFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ImageUsageFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ImageUsageFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ImageUsageFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ImageUsageFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::TRANSFER_SRC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFER_SRC")?;
            remaining &= !Self::TRANSFER_SRC.0;
            first = false;
        }
        if remaining & Self::TRANSFER_DST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFER_DST")?;
            remaining &= !Self::TRANSFER_DST.0;
            first = false;
        }
        if remaining & Self::SAMPLED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SAMPLED")?;
            remaining &= !Self::SAMPLED.0;
            first = false;
        }
        if remaining & Self::STORAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("STORAGE")?;
            remaining &= !Self::STORAGE.0;
            first = false;
        }
        if remaining & Self::COLOR_ATTACHMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("COLOR_ATTACHMENT")?;
            remaining &= !Self::COLOR_ATTACHMENT.0;
            first = false;
        }
        if remaining & Self::DEPTH_STENCIL_ATTACHMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEPTH_STENCIL_ATTACHMENT")?;
            remaining &= !Self::DEPTH_STENCIL_ATTACHMENT.0;
            first = false;
        }
        if remaining & Self::TRANSIENT_ATTACHMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSIENT_ATTACHMENT")?;
            remaining &= !Self::TRANSIENT_ATTACHMENT.0;
            first = false;
        }
        if remaining & Self::INPUT_ATTACHMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INPUT_ATTACHMENT")?;
            remaining &= !Self::INPUT_ATTACHMENT.0;
            first = false;
        }
        if remaining & Self::HOST_TRANSFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("HOST_TRANSFER")?;
            remaining &= !Self::HOST_TRANSFER.0;
            first = false;
        }
        if remaining & Self::VIDEO_DECODE_DST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VIDEO_DECODE_DST")?;
            remaining &= !Self::VIDEO_DECODE_DST.0;
            first = false;
        }
        if remaining & Self::VIDEO_DECODE_SRC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VIDEO_DECODE_SRC")?;
            remaining &= !Self::VIDEO_DECODE_SRC.0;
            first = false;
        }
        if remaining & Self::VIDEO_DECODE_DPB.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VIDEO_DECODE_DPB")?;
            remaining &= !Self::VIDEO_DECODE_DPB.0;
            first = false;
        }
        if remaining & Self::FRAGMENT_DENSITY_MAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRAGMENT_DENSITY_MAP")?;
            remaining &= !Self::FRAGMENT_DENSITY_MAP.0;
            first = false;
        }
        if remaining & Self::FRAGMENT_SHADING_RATE_ATTACHMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRAGMENT_SHADING_RATE_ATTACHMENT")?;
            remaining &= !Self::FRAGMENT_SHADING_RATE_ATTACHMENT.0;
            first = false;
        }
        if remaining & Self::VIDEO_ENCODE_DST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VIDEO_ENCODE_DST")?;
            remaining &= !Self::VIDEO_ENCODE_DST.0;
            first = false;
        }
        if remaining & Self::VIDEO_ENCODE_SRC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VIDEO_ENCODE_SRC")?;
            remaining &= !Self::VIDEO_ENCODE_SRC.0;
            first = false;
        }
        if remaining & Self::VIDEO_ENCODE_DPB.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VIDEO_ENCODE_DPB")?;
            remaining &= !Self::VIDEO_ENCODE_DPB.0;
            first = false;
        }
        if remaining & Self::ATTACHMENT_FEEDBACK_LOOP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ATTACHMENT_FEEDBACK_LOOP")?;
            remaining &= !Self::ATTACHMENT_FEEDBACK_LOOP.0;
            first = false;
        }
        if remaining & Self::INVOCATION_MASK_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INVOCATION_MASK_BIT")?;
            remaining &= !Self::INVOCATION_MASK_BIT.0;
            first = false;
        }
        if remaining & Self::SAMPLE_WEIGHT_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SAMPLE_WEIGHT_BIT")?;
            remaining &= !Self::SAMPLE_WEIGHT_BIT.0;
            first = false;
        }
        if remaining & Self::SAMPLE_BLOCK_MATCH_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SAMPLE_BLOCK_MATCH_BIT")?;
            remaining &= !Self::SAMPLE_BLOCK_MATCH_BIT.0;
            first = false;
        }
        if remaining & Self::TENSOR_ALIASING_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TENSOR_ALIASING_BIT")?;
            remaining &= !Self::TENSOR_ALIASING_BIT.0;
            first = false;
        }
        if remaining & Self::TILE_MEMORY_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TILE_MEMORY_BIT")?;
            remaining &= !Self::TILE_MEMORY_BIT.0;
            first = false;
        }
        if remaining & Self::VIDEO_ENCODE_QUANTIZATION_DELTA_MAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VIDEO_ENCODE_QUANTIZATION_DELTA_MAP")?;
            remaining &= !Self::VIDEO_ENCODE_QUANTIZATION_DELTA_MAP.0;
            first = false;
        }
        if remaining & Self::VIDEO_ENCODE_EMPHASIS_MAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VIDEO_ENCODE_EMPHASIS_MAP")?;
            remaining &= !Self::VIDEO_ENCODE_EMPHASIS_MAP.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkImageViewCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageViewCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkImageViewCreateFlagBits")]
pub struct ImageViewCreateFlagBits(u32);
impl ImageViewCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const FRAGMENT_DENSITY_MAP_DYNAMIC: Self = Self(1u32);
    ///Bit 2.
    pub const DESCRIPTOR_BUFFER_CAPTURE_REPLAY: Self = Self(4u32);
    ///Bit 1.
    pub const FRAGMENT_DENSITY_MAP_DEFERRED: Self = Self(2u32);
}
impl core::ops::BitOr for ImageViewCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ImageViewCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ImageViewCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ImageViewCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ImageViewCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ImageViewCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ImageViewCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ImageViewCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::FRAGMENT_DENSITY_MAP_DYNAMIC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRAGMENT_DENSITY_MAP_DYNAMIC")?;
            remaining &= !Self::FRAGMENT_DENSITY_MAP_DYNAMIC.0;
            first = false;
        }
        if remaining & Self::DESCRIPTOR_BUFFER_CAPTURE_REPLAY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DESCRIPTOR_BUFFER_CAPTURE_REPLAY")?;
            remaining &= !Self::DESCRIPTOR_BUFFER_CAPTURE_REPLAY.0;
            first = false;
        }
        if remaining & Self::FRAGMENT_DENSITY_MAP_DEFERRED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRAGMENT_DENSITY_MAP_DEFERRED")?;
            remaining &= !Self::FRAGMENT_DENSITY_MAP_DEFERRED.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkIndirectCommandsInputModeFlagBitsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkIndirectCommandsInputModeFlagBitsEXT.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkIndirectCommandsInputModeFlagBitsEXT")]
pub struct IndirectCommandsInputModeFlagBitsEXT(u32);
impl IndirectCommandsInputModeFlagBitsEXT {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const VULKAN_INDEX_BUFFER: Self = Self(1u32);
    ///Bit 1.
    pub const DXGI_INDEX_BUFFER: Self = Self(2u32);
}
impl core::ops::BitOr for IndirectCommandsInputModeFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for IndirectCommandsInputModeFlagBitsEXT {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for IndirectCommandsInputModeFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for IndirectCommandsInputModeFlagBitsEXT {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for IndirectCommandsInputModeFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for IndirectCommandsInputModeFlagBitsEXT {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for IndirectCommandsInputModeFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for IndirectCommandsInputModeFlagBitsEXT {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::VULKAN_INDEX_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VULKAN_INDEX_BUFFER")?;
            remaining &= !Self::VULKAN_INDEX_BUFFER.0;
            first = false;
        }
        if remaining & Self::DXGI_INDEX_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DXGI_INDEX_BUFFER")?;
            remaining &= !Self::DXGI_INDEX_BUFFER.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkIndirectCommandsLayoutUsageFlagBitsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkIndirectCommandsLayoutUsageFlagBitsEXT.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkIndirectCommandsLayoutUsageFlagBitsEXT")]
pub struct IndirectCommandsLayoutUsageFlagBitsEXT(u32);
impl IndirectCommandsLayoutUsageFlagBitsEXT {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const EXPLICIT_PREPROCESS: Self = Self(1u32);
    ///Bit 1.
    pub const UNORDERED_SEQUENCES: Self = Self(2u32);
}
impl core::ops::BitOr for IndirectCommandsLayoutUsageFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for IndirectCommandsLayoutUsageFlagBitsEXT {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for IndirectCommandsLayoutUsageFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for IndirectCommandsLayoutUsageFlagBitsEXT {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for IndirectCommandsLayoutUsageFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for IndirectCommandsLayoutUsageFlagBitsEXT {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for IndirectCommandsLayoutUsageFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for IndirectCommandsLayoutUsageFlagBitsEXT {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::EXPLICIT_PREPROCESS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("EXPLICIT_PREPROCESS")?;
            remaining &= !Self::EXPLICIT_PREPROCESS.0;
            first = false;
        }
        if remaining & Self::UNORDERED_SEQUENCES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("UNORDERED_SEQUENCES")?;
            remaining &= !Self::UNORDERED_SEQUENCES.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkIndirectCommandsLayoutUsageFlagBitsNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkIndirectCommandsLayoutUsageFlagBitsNV.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkIndirectCommandsLayoutUsageFlagBitsNV")]
pub struct IndirectCommandsLayoutUsageFlagBitsNV(u32);
impl IndirectCommandsLayoutUsageFlagBitsNV {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const EXPLICIT_PREPROCESS: Self = Self(1u32);
    ///Bit 1.
    pub const INDEXED_SEQUENCES: Self = Self(2u32);
    ///Bit 2.
    pub const UNORDERED_SEQUENCES: Self = Self(4u32);
}
impl core::ops::BitOr for IndirectCommandsLayoutUsageFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for IndirectCommandsLayoutUsageFlagBitsNV {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for IndirectCommandsLayoutUsageFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for IndirectCommandsLayoutUsageFlagBitsNV {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for IndirectCommandsLayoutUsageFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for IndirectCommandsLayoutUsageFlagBitsNV {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for IndirectCommandsLayoutUsageFlagBitsNV {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for IndirectCommandsLayoutUsageFlagBitsNV {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::EXPLICIT_PREPROCESS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("EXPLICIT_PREPROCESS")?;
            remaining &= !Self::EXPLICIT_PREPROCESS.0;
            first = false;
        }
        if remaining & Self::INDEXED_SEQUENCES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INDEXED_SEQUENCES")?;
            remaining &= !Self::INDEXED_SEQUENCES.0;
            first = false;
        }
        if remaining & Self::UNORDERED_SEQUENCES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("UNORDERED_SEQUENCES")?;
            remaining &= !Self::UNORDERED_SEQUENCES.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkIndirectStateFlagBitsNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkIndirectStateFlagBitsNV.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkIndirectStateFlagBitsNV")]
pub struct IndirectStateFlagBitsNV(u32);
impl IndirectStateFlagBitsNV {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const FLAG_FRONTFACE: Self = Self(1u32);
}
impl core::ops::BitOr for IndirectStateFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for IndirectStateFlagBitsNV {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for IndirectStateFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for IndirectStateFlagBitsNV {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for IndirectStateFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for IndirectStateFlagBitsNV {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for IndirectStateFlagBitsNV {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for IndirectStateFlagBitsNV {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::FLAG_FRONTFACE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FLAG_FRONTFACE")?;
            remaining &= !Self::FLAG_FRONTFACE.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkInstanceCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkInstanceCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkInstanceCreateFlagBits")]
pub struct InstanceCreateFlagBits(u32);
impl InstanceCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const ENUMERATE_PORTABILITY: Self = Self(1u32);
}
impl core::ops::BitOr for InstanceCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for InstanceCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for InstanceCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for InstanceCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for InstanceCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for InstanceCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for InstanceCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for InstanceCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::ENUMERATE_PORTABILITY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ENUMERATE_PORTABILITY")?;
            remaining &= !Self::ENUMERATE_PORTABILITY.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkMemoryAllocateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryAllocateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkMemoryAllocateFlagBits")]
pub struct MemoryAllocateFlagBits(u32);
impl MemoryAllocateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const DEVICE_MASK: Self = Self(1u32);
    ///Bit 1.
    pub const DEVICE_ADDRESS: Self = Self(2u32);
    ///Bit 2.
    pub const DEVICE_ADDRESS_CAPTURE_REPLAY: Self = Self(4u32);
    ///Bit 3.
    pub const ZERO_INITIALIZE: Self = Self(8u32);
}
impl core::ops::BitOr for MemoryAllocateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for MemoryAllocateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for MemoryAllocateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for MemoryAllocateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for MemoryAllocateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for MemoryAllocateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for MemoryAllocateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for MemoryAllocateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::DEVICE_MASK.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEVICE_MASK")?;
            remaining &= !Self::DEVICE_MASK.0;
            first = false;
        }
        if remaining & Self::DEVICE_ADDRESS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEVICE_ADDRESS")?;
            remaining &= !Self::DEVICE_ADDRESS.0;
            first = false;
        }
        if remaining & Self::DEVICE_ADDRESS_CAPTURE_REPLAY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEVICE_ADDRESS_CAPTURE_REPLAY")?;
            remaining &= !Self::DEVICE_ADDRESS_CAPTURE_REPLAY.0;
            first = false;
        }
        if remaining & Self::ZERO_INITIALIZE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ZERO_INITIALIZE")?;
            remaining &= !Self::ZERO_INITIALIZE.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkMemoryDecompressionMethodFlagBitsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryDecompressionMethodFlagBitsEXT.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkMemoryDecompressionMethodFlagBitsEXT")]
pub struct MemoryDecompressionMethodFlagBitsEXT(u64);
impl MemoryDecompressionMethodFlagBitsEXT {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u64)
    }
    #[inline]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u64
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const GDEFLATE_1_0: Self = Self(1u64);
}
impl core::ops::BitOr for MemoryDecompressionMethodFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for MemoryDecompressionMethodFlagBitsEXT {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for MemoryDecompressionMethodFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for MemoryDecompressionMethodFlagBitsEXT {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for MemoryDecompressionMethodFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for MemoryDecompressionMethodFlagBitsEXT {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for MemoryDecompressionMethodFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for MemoryDecompressionMethodFlagBitsEXT {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::GDEFLATE_1_0.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("GDEFLATE_1_0")?;
            remaining &= !Self::GDEFLATE_1_0.0;
            first = false;
        }
        if remaining != 0u64 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkMemoryHeapFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryHeapFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkMemoryHeapFlagBits")]
pub struct MemoryHeapFlagBits(u32);
impl MemoryHeapFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const DEVICE_LOCAL: Self = Self(1u32);
    ///Bit 1.
    pub const MULTI_INSTANCE: Self = Self(2u32);
    ///Bit 2.
    pub const SEU_SAFE: Self = Self(4u32);
    ///Bit 3.
    pub const TILE_MEMORY_BIT: Self = Self(8u32);
}
impl core::ops::BitOr for MemoryHeapFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for MemoryHeapFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for MemoryHeapFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for MemoryHeapFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for MemoryHeapFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for MemoryHeapFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for MemoryHeapFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for MemoryHeapFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::DEVICE_LOCAL.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEVICE_LOCAL")?;
            remaining &= !Self::DEVICE_LOCAL.0;
            first = false;
        }
        if remaining & Self::MULTI_INSTANCE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MULTI_INSTANCE")?;
            remaining &= !Self::MULTI_INSTANCE.0;
            first = false;
        }
        if remaining & Self::SEU_SAFE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SEU_SAFE")?;
            remaining &= !Self::SEU_SAFE.0;
            first = false;
        }
        if remaining & Self::TILE_MEMORY_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TILE_MEMORY_BIT")?;
            remaining &= !Self::TILE_MEMORY_BIT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkMemoryMapFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryMapFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkMemoryMapFlagBits")]
pub struct MemoryMapFlagBits(u32);
impl MemoryMapFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const PLACED: Self = Self(1u32);
}
impl core::ops::BitOr for MemoryMapFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for MemoryMapFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for MemoryMapFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for MemoryMapFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for MemoryMapFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for MemoryMapFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for MemoryMapFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for MemoryMapFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::PLACED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PLACED")?;
            remaining &= !Self::PLACED.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkMemoryPropertyFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryPropertyFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkMemoryPropertyFlagBits")]
pub struct MemoryPropertyFlagBits(u32);
impl MemoryPropertyFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const DEVICE_LOCAL: Self = Self(1u32);
    ///Bit 1.
    pub const HOST_VISIBLE: Self = Self(2u32);
    ///Bit 2.
    pub const HOST_COHERENT: Self = Self(4u32);
    ///Bit 3.
    pub const HOST_CACHED: Self = Self(8u32);
    ///Bit 4.
    pub const LAZILY_ALLOCATED: Self = Self(16u32);
    ///Bit 5.
    pub const PROTECTED: Self = Self(32u32);
    ///Bit 6.
    pub const DEVICE_COHERENT: Self = Self(64u32);
    ///Bit 7.
    pub const DEVICE_UNCACHED: Self = Self(128u32);
    ///Bit 8.
    pub const RDMA_CAPABLE: Self = Self(256u32);
}
impl core::ops::BitOr for MemoryPropertyFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for MemoryPropertyFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for MemoryPropertyFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for MemoryPropertyFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for MemoryPropertyFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for MemoryPropertyFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for MemoryPropertyFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for MemoryPropertyFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::DEVICE_LOCAL.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEVICE_LOCAL")?;
            remaining &= !Self::DEVICE_LOCAL.0;
            first = false;
        }
        if remaining & Self::HOST_VISIBLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("HOST_VISIBLE")?;
            remaining &= !Self::HOST_VISIBLE.0;
            first = false;
        }
        if remaining & Self::HOST_COHERENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("HOST_COHERENT")?;
            remaining &= !Self::HOST_COHERENT.0;
            first = false;
        }
        if remaining & Self::HOST_CACHED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("HOST_CACHED")?;
            remaining &= !Self::HOST_CACHED.0;
            first = false;
        }
        if remaining & Self::LAZILY_ALLOCATED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("LAZILY_ALLOCATED")?;
            remaining &= !Self::LAZILY_ALLOCATED.0;
            first = false;
        }
        if remaining & Self::PROTECTED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PROTECTED")?;
            remaining &= !Self::PROTECTED.0;
            first = false;
        }
        if remaining & Self::DEVICE_COHERENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEVICE_COHERENT")?;
            remaining &= !Self::DEVICE_COHERENT.0;
            first = false;
        }
        if remaining & Self::DEVICE_UNCACHED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEVICE_UNCACHED")?;
            remaining &= !Self::DEVICE_UNCACHED.0;
            first = false;
        }
        if remaining & Self::RDMA_CAPABLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RDMA_CAPABLE")?;
            remaining &= !Self::RDMA_CAPABLE.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkMemoryUnmapFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryUnmapFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkMemoryUnmapFlagBits")]
pub struct MemoryUnmapFlagBits(u32);
impl MemoryUnmapFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const RESERVE: Self = Self(1u32);
}
impl core::ops::BitOr for MemoryUnmapFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for MemoryUnmapFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for MemoryUnmapFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for MemoryUnmapFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for MemoryUnmapFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for MemoryUnmapFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for MemoryUnmapFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for MemoryUnmapFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::RESERVE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RESERVE")?;
            remaining &= !Self::RESERVE.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkMicromapCreateFlagBitsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkMicromapCreateFlagBitsEXT.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkMicromapCreateFlagBitsEXT")]
pub struct MicromapCreateFlagBitsEXT(u32);
impl MicromapCreateFlagBitsEXT {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const DEVICE_ADDRESS_CAPTURE_REPLAY: Self = Self(1u32);
}
impl core::ops::BitOr for MicromapCreateFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for MicromapCreateFlagBitsEXT {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for MicromapCreateFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for MicromapCreateFlagBitsEXT {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for MicromapCreateFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for MicromapCreateFlagBitsEXT {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for MicromapCreateFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for MicromapCreateFlagBitsEXT {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::DEVICE_ADDRESS_CAPTURE_REPLAY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEVICE_ADDRESS_CAPTURE_REPLAY")?;
            remaining &= !Self::DEVICE_ADDRESS_CAPTURE_REPLAY.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkOpticalFlowExecuteFlagBitsNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkOpticalFlowExecuteFlagBitsNV.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkOpticalFlowExecuteFlagBitsNV")]
pub struct OpticalFlowExecuteFlagBitsNV(u32);
impl OpticalFlowExecuteFlagBitsNV {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const DISABLE_TEMPORAL_HINTS: Self = Self(1u32);
}
impl core::ops::BitOr for OpticalFlowExecuteFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for OpticalFlowExecuteFlagBitsNV {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for OpticalFlowExecuteFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for OpticalFlowExecuteFlagBitsNV {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for OpticalFlowExecuteFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for OpticalFlowExecuteFlagBitsNV {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for OpticalFlowExecuteFlagBitsNV {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for OpticalFlowExecuteFlagBitsNV {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::DISABLE_TEMPORAL_HINTS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DISABLE_TEMPORAL_HINTS")?;
            remaining &= !Self::DISABLE_TEMPORAL_HINTS.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkOpticalFlowGridSizeFlagBitsNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkOpticalFlowGridSizeFlagBitsNV.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkOpticalFlowGridSizeFlagBitsNV")]
pub struct OpticalFlowGridSizeFlagBitsNV(u32);
impl OpticalFlowGridSizeFlagBitsNV {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const UNKNOWN: Self = Self(0u32);
    ///Bit 0.
    pub const _1X1: Self = Self(1u32);
    ///Bit 1.
    pub const _2X2: Self = Self(2u32);
    ///Bit 2.
    pub const _4X4: Self = Self(4u32);
    ///Bit 3.
    pub const _8X8: Self = Self(8u32);
}
impl core::ops::BitOr for OpticalFlowGridSizeFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for OpticalFlowGridSizeFlagBitsNV {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for OpticalFlowGridSizeFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for OpticalFlowGridSizeFlagBitsNV {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for OpticalFlowGridSizeFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for OpticalFlowGridSizeFlagBitsNV {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for OpticalFlowGridSizeFlagBitsNV {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for OpticalFlowGridSizeFlagBitsNV {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::_1X1.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_1X1")?;
            remaining &= !Self::_1X1.0;
            first = false;
        }
        if remaining & Self::_2X2.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2X2")?;
            remaining &= !Self::_2X2.0;
            first = false;
        }
        if remaining & Self::_4X4.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_4X4")?;
            remaining &= !Self::_4X4.0;
            first = false;
        }
        if remaining & Self::_8X8.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_8X8")?;
            remaining &= !Self::_8X8.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkOpticalFlowSessionCreateFlagBitsNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkOpticalFlowSessionCreateFlagBitsNV.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkOpticalFlowSessionCreateFlagBitsNV")]
pub struct OpticalFlowSessionCreateFlagBitsNV(u32);
impl OpticalFlowSessionCreateFlagBitsNV {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const ENABLE_HINT: Self = Self(1u32);
    ///Bit 1.
    pub const ENABLE_COST: Self = Self(2u32);
    ///Bit 2.
    pub const ENABLE_GLOBAL_FLOW: Self = Self(4u32);
    ///Bit 3.
    pub const ALLOW_REGIONS: Self = Self(8u32);
    ///Bit 4.
    pub const BOTH_DIRECTIONS: Self = Self(16u32);
}
impl core::ops::BitOr for OpticalFlowSessionCreateFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for OpticalFlowSessionCreateFlagBitsNV {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for OpticalFlowSessionCreateFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for OpticalFlowSessionCreateFlagBitsNV {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for OpticalFlowSessionCreateFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for OpticalFlowSessionCreateFlagBitsNV {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for OpticalFlowSessionCreateFlagBitsNV {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for OpticalFlowSessionCreateFlagBitsNV {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::ENABLE_HINT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ENABLE_HINT")?;
            remaining &= !Self::ENABLE_HINT.0;
            first = false;
        }
        if remaining & Self::ENABLE_COST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ENABLE_COST")?;
            remaining &= !Self::ENABLE_COST.0;
            first = false;
        }
        if remaining & Self::ENABLE_GLOBAL_FLOW.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ENABLE_GLOBAL_FLOW")?;
            remaining &= !Self::ENABLE_GLOBAL_FLOW.0;
            first = false;
        }
        if remaining & Self::ALLOW_REGIONS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALLOW_REGIONS")?;
            remaining &= !Self::ALLOW_REGIONS.0;
            first = false;
        }
        if remaining & Self::BOTH_DIRECTIONS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("BOTH_DIRECTIONS")?;
            remaining &= !Self::BOTH_DIRECTIONS.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkOpticalFlowUsageFlagBitsNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkOpticalFlowUsageFlagBitsNV.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkOpticalFlowUsageFlagBitsNV")]
pub struct OpticalFlowUsageFlagBitsNV(u32);
impl OpticalFlowUsageFlagBitsNV {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const UNKNOWN: Self = Self(0u32);
    ///Bit 0.
    pub const INPUT: Self = Self(1u32);
    ///Bit 1.
    pub const OUTPUT: Self = Self(2u32);
    ///Bit 2.
    pub const HINT: Self = Self(4u32);
    ///Bit 3.
    pub const COST: Self = Self(8u32);
    ///Bit 4.
    pub const GLOBAL_FLOW: Self = Self(16u32);
}
impl core::ops::BitOr for OpticalFlowUsageFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for OpticalFlowUsageFlagBitsNV {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for OpticalFlowUsageFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for OpticalFlowUsageFlagBitsNV {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for OpticalFlowUsageFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for OpticalFlowUsageFlagBitsNV {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for OpticalFlowUsageFlagBitsNV {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for OpticalFlowUsageFlagBitsNV {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::INPUT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INPUT")?;
            remaining &= !Self::INPUT.0;
            first = false;
        }
        if remaining & Self::OUTPUT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("OUTPUT")?;
            remaining &= !Self::OUTPUT.0;
            first = false;
        }
        if remaining & Self::HINT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("HINT")?;
            remaining &= !Self::HINT.0;
            first = false;
        }
        if remaining & Self::COST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("COST")?;
            remaining &= !Self::COST.0;
            first = false;
        }
        if remaining & Self::GLOBAL_FLOW.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("GLOBAL_FLOW")?;
            remaining &= !Self::GLOBAL_FLOW.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkPartitionedAccelerationStructureInstanceFlagBitsNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPartitionedAccelerationStructureInstanceFlagBitsNV.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkPartitionedAccelerationStructureInstanceFlagBitsNV")]
pub struct PartitionedAccelerationStructureInstanceFlagBitsNV(u32);
impl PartitionedAccelerationStructureInstanceFlagBitsNV {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const FLAG_TRIANGLE_FACING_CULL_DISABLE: Self = Self(1u32);
    ///Bit 1.
    pub const FLAG_TRIANGLE_FLIP_FACING: Self = Self(2u32);
    ///Bit 2.
    pub const FLAG_FORCE_OPAQUE: Self = Self(4u32);
    ///Bit 3.
    pub const FLAG_FORCE_NO_OPAQUE: Self = Self(8u32);
    ///Bit 4.
    pub const FLAG_ENABLE_EXPLICIT_BOUNDING_BOX: Self = Self(16u32);
}
impl core::ops::BitOr for PartitionedAccelerationStructureInstanceFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for PartitionedAccelerationStructureInstanceFlagBitsNV {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for PartitionedAccelerationStructureInstanceFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for PartitionedAccelerationStructureInstanceFlagBitsNV {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for PartitionedAccelerationStructureInstanceFlagBitsNV {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for PartitionedAccelerationStructureInstanceFlagBitsNV {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for PartitionedAccelerationStructureInstanceFlagBitsNV {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for PartitionedAccelerationStructureInstanceFlagBitsNV {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::FLAG_TRIANGLE_FACING_CULL_DISABLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FLAG_TRIANGLE_FACING_CULL_DISABLE")?;
            remaining &= !Self::FLAG_TRIANGLE_FACING_CULL_DISABLE.0;
            first = false;
        }
        if remaining & Self::FLAG_TRIANGLE_FLIP_FACING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FLAG_TRIANGLE_FLIP_FACING")?;
            remaining &= !Self::FLAG_TRIANGLE_FLIP_FACING.0;
            first = false;
        }
        if remaining & Self::FLAG_FORCE_OPAQUE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FLAG_FORCE_OPAQUE")?;
            remaining &= !Self::FLAG_FORCE_OPAQUE.0;
            first = false;
        }
        if remaining & Self::FLAG_FORCE_NO_OPAQUE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FLAG_FORCE_NO_OPAQUE")?;
            remaining &= !Self::FLAG_FORCE_NO_OPAQUE.0;
            first = false;
        }
        if remaining & Self::FLAG_ENABLE_EXPLICIT_BOUNDING_BOX.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FLAG_ENABLE_EXPLICIT_BOUNDING_BOX")?;
            remaining &= !Self::FLAG_ENABLE_EXPLICIT_BOUNDING_BOX.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkPastPresentationTimingFlagBitsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPastPresentationTimingFlagBitsEXT.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkPastPresentationTimingFlagBitsEXT")]
pub struct PastPresentationTimingFlagBitsEXT(u32);
impl PastPresentationTimingFlagBitsEXT {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const ALLOW_PARTIAL_RESULTS: Self = Self(1u32);
    ///Bit 1.
    pub const ALLOW_OUT_OF_ORDER_RESULTS: Self = Self(2u32);
}
impl core::ops::BitOr for PastPresentationTimingFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for PastPresentationTimingFlagBitsEXT {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for PastPresentationTimingFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for PastPresentationTimingFlagBitsEXT {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for PastPresentationTimingFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for PastPresentationTimingFlagBitsEXT {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for PastPresentationTimingFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for PastPresentationTimingFlagBitsEXT {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::ALLOW_PARTIAL_RESULTS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALLOW_PARTIAL_RESULTS")?;
            remaining &= !Self::ALLOW_PARTIAL_RESULTS.0;
            first = false;
        }
        if remaining & Self::ALLOW_OUT_OF_ORDER_RESULTS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALLOW_OUT_OF_ORDER_RESULTS")?;
            remaining &= !Self::ALLOW_OUT_OF_ORDER_RESULTS.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkPeerMemoryFeatureFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPeerMemoryFeatureFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkPeerMemoryFeatureFlagBits")]
pub struct PeerMemoryFeatureFlagBits(u32);
impl PeerMemoryFeatureFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const COPY_SRC: Self = Self(1u32);
    ///Bit 1.
    pub const COPY_DST: Self = Self(2u32);
    ///Bit 2.
    pub const GENERIC_SRC: Self = Self(4u32);
    ///Bit 3.
    pub const GENERIC_DST: Self = Self(8u32);
}
impl core::ops::BitOr for PeerMemoryFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for PeerMemoryFeatureFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for PeerMemoryFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for PeerMemoryFeatureFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for PeerMemoryFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for PeerMemoryFeatureFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for PeerMemoryFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for PeerMemoryFeatureFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::COPY_SRC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("COPY_SRC")?;
            remaining &= !Self::COPY_SRC.0;
            first = false;
        }
        if remaining & Self::COPY_DST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("COPY_DST")?;
            remaining &= !Self::COPY_DST.0;
            first = false;
        }
        if remaining & Self::GENERIC_SRC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("GENERIC_SRC")?;
            remaining &= !Self::GENERIC_SRC.0;
            first = false;
        }
        if remaining & Self::GENERIC_DST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("GENERIC_DST")?;
            remaining &= !Self::GENERIC_DST.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkPerformanceCounterDescriptionFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPerformanceCounterDescriptionFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkPerformanceCounterDescriptionFlagBitsKHR")]
pub struct PerformanceCounterDescriptionFlagBitsKHR(u32);
impl PerformanceCounterDescriptionFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const PERFORMANCE_IMPACTING: Self = Self(1u32);
    ///Bit 1.
    pub const CONCURRENTLY_IMPACTED: Self = Self(2u32);
}
impl core::ops::BitOr for PerformanceCounterDescriptionFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for PerformanceCounterDescriptionFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for PerformanceCounterDescriptionFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for PerformanceCounterDescriptionFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for PerformanceCounterDescriptionFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for PerformanceCounterDescriptionFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for PerformanceCounterDescriptionFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for PerformanceCounterDescriptionFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::PERFORMANCE_IMPACTING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PERFORMANCE_IMPACTING")?;
            remaining &= !Self::PERFORMANCE_IMPACTING.0;
            first = false;
        }
        if remaining & Self::CONCURRENTLY_IMPACTED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CONCURRENTLY_IMPACTED")?;
            remaining &= !Self::CONCURRENTLY_IMPACTED.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkPhysicalDeviceSchedulingControlsFlagBitsARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceSchedulingControlsFlagBitsARM.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkPhysicalDeviceSchedulingControlsFlagBitsARM")]
pub struct PhysicalDeviceSchedulingControlsFlagBitsARM(u64);
impl PhysicalDeviceSchedulingControlsFlagBitsARM {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u64)
    }
    #[inline]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u64
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const SHADER_CORE_COUNT: Self = Self(1u64);
}
impl core::ops::BitOr for PhysicalDeviceSchedulingControlsFlagBitsARM {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for PhysicalDeviceSchedulingControlsFlagBitsARM {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for PhysicalDeviceSchedulingControlsFlagBitsARM {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for PhysicalDeviceSchedulingControlsFlagBitsARM {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for PhysicalDeviceSchedulingControlsFlagBitsARM {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for PhysicalDeviceSchedulingControlsFlagBitsARM {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for PhysicalDeviceSchedulingControlsFlagBitsARM {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for PhysicalDeviceSchedulingControlsFlagBitsARM {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::SHADER_CORE_COUNT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SHADER_CORE_COUNT")?;
            remaining &= !Self::SHADER_CORE_COUNT.0;
            first = false;
        }
        if remaining != 0u64 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkPipelineCacheCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineCacheCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkPipelineCacheCreateFlagBits")]
pub struct PipelineCacheCreateFlagBits(u32);
impl PipelineCacheCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const EXTERNALLY_SYNCHRONIZED: Self = Self(1u32);
    ///Bit 1.
    pub const READ_ONLY: Self = Self(2u32);
    ///Bit 2.
    pub const USE_APPLICATION_STORAGE: Self = Self(4u32);
    ///Bit 3.
    pub const INTERNALLY_SYNCHRONIZED_MERGE: Self = Self(8u32);
}
impl core::ops::BitOr for PipelineCacheCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for PipelineCacheCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for PipelineCacheCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for PipelineCacheCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for PipelineCacheCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for PipelineCacheCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for PipelineCacheCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for PipelineCacheCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::EXTERNALLY_SYNCHRONIZED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("EXTERNALLY_SYNCHRONIZED")?;
            remaining &= !Self::EXTERNALLY_SYNCHRONIZED.0;
            first = false;
        }
        if remaining & Self::READ_ONLY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("READ_ONLY")?;
            remaining &= !Self::READ_ONLY.0;
            first = false;
        }
        if remaining & Self::USE_APPLICATION_STORAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("USE_APPLICATION_STORAGE")?;
            remaining &= !Self::USE_APPLICATION_STORAGE.0;
            first = false;
        }
        if remaining & Self::INTERNALLY_SYNCHRONIZED_MERGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INTERNALLY_SYNCHRONIZED_MERGE")?;
            remaining &= !Self::INTERNALLY_SYNCHRONIZED_MERGE.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkPipelineColorBlendStateCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineColorBlendStateCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkPipelineColorBlendStateCreateFlagBits")]
pub struct PipelineColorBlendStateCreateFlagBits(u32);
impl PipelineColorBlendStateCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const RASTERIZATION_ORDER_ATTACHMENT_ACCESS_BIT: Self = Self::RASTERIZATION_ORDER_ATTACHMENT_ACCESS;
    ///Bit 0.
    pub const RASTERIZATION_ORDER_ATTACHMENT_ACCESS: Self = Self(1u32);
}
impl core::ops::BitOr for PipelineColorBlendStateCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for PipelineColorBlendStateCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for PipelineColorBlendStateCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for PipelineColorBlendStateCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for PipelineColorBlendStateCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for PipelineColorBlendStateCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for PipelineColorBlendStateCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for PipelineColorBlendStateCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::RASTERIZATION_ORDER_ATTACHMENT_ACCESS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RASTERIZATION_ORDER_ATTACHMENT_ACCESS")?;
            remaining &= !Self::RASTERIZATION_ORDER_ATTACHMENT_ACCESS.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkPipelineCompilerControlFlagBitsAMD`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineCompilerControlFlagBitsAMD.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkPipelineCompilerControlFlagBitsAMD")]
pub struct PipelineCompilerControlFlagBitsAMD(u32);
impl PipelineCompilerControlFlagBitsAMD {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}
impl core::ops::BitOr for PipelineCompilerControlFlagBitsAMD {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for PipelineCompilerControlFlagBitsAMD {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for PipelineCompilerControlFlagBitsAMD {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for PipelineCompilerControlFlagBitsAMD {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for PipelineCompilerControlFlagBitsAMD {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for PipelineCompilerControlFlagBitsAMD {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for PipelineCompilerControlFlagBitsAMD {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for PipelineCompilerControlFlagBitsAMD {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkPipelineCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkPipelineCreateFlagBits")]
pub struct PipelineCreateFlagBits(u32);
impl PipelineCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const DISABLE_OPTIMIZATION: Self = Self(1u32);
    ///Bit 1.
    pub const ALLOW_DERIVATIVES: Self = Self(2u32);
    ///Bit 2.
    pub const DERIVATIVE: Self = Self(4u32);
    ///Bit 4.
    pub const DISPATCH_BASE: Self = Self(16u32);
    ///Bit 3.
    pub const VIEW_INDEX_FROM_DEVICE_INDEX: Self = Self(8u32);
    ///Bit 8.
    pub const FAIL_ON_PIPELINE_COMPILE_REQUIRED: Self = Self(256u32);
    ///Bit 9.
    pub const EARLY_RETURN_ON_FAILURE: Self = Self(512u32);
    ///Bit 27.
    pub const NO_PROTECTED_ACCESS: Self = Self(134217728u32);
    ///Bit 30.
    pub const PROTECTED_ACCESS_ONLY: Self = Self(1073741824u32);
    ///Bit 14.
    pub const RAY_TRACING_NO_NULL_ANY_HIT_SHADERS: Self = Self(16384u32);
    ///Bit 15.
    pub const RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS: Self = Self(32768u32);
    ///Bit 16.
    pub const RAY_TRACING_NO_NULL_MISS_SHADERS: Self = Self(65536u32);
    ///Bit 17.
    pub const RAY_TRACING_NO_NULL_INTERSECTION_SHADERS: Self = Self(131072u32);
    ///Bit 12.
    pub const RAY_TRACING_SKIP_TRIANGLES: Self = Self(4096u32);
    ///Bit 13.
    pub const RAY_TRACING_SKIP_AABBS: Self = Self(8192u32);
    ///Bit 19.
    pub const RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY: Self = Self(524288u32);
    ///Bit 5.
    pub const DEFER_COMPILE: Self = Self(32u32);
    ///Bit 22.
    pub const RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT: Self = Self(4194304u32);
    pub const PIPELINE_RASTERIZATION_STATE_CREATE_FRAGMENT_DENSITY_MAP_ATTACHMENT: Self = Self::RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT;
    ///Bit 21.
    pub const RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT: Self = Self(2097152u32);
    pub const PIPELINE_RASTERIZATION_STATE_CREATE_FRAGMENT_SHADING_RATE_ATTACHMENT: Self = Self::RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT;
    ///Bit 6.
    pub const CAPTURE_STATISTICS: Self = Self(64u32);
    ///Bit 7.
    pub const CAPTURE_INTERNAL_REPRESENTATIONS: Self = Self(128u32);
    ///Bit 18.
    pub const INDIRECT_BINDABLE: Self = Self(262144u32);
    ///Bit 11.
    pub const LIBRARY: Self = Self(2048u32);
    ///Bit 29.
    pub const DESCRIPTOR_BUFFER: Self = Self(536870912u32);
    ///Bit 23.
    pub const RETAIN_LINK_TIME_OPTIMIZATION_INFO: Self = Self(8388608u32);
    ///Bit 10.
    pub const LINK_TIME_OPTIMIZATION: Self = Self(1024u32);
    ///Bit 20.
    pub const RAY_TRACING_ALLOW_MOTION: Self = Self(1048576u32);
    ///Bit 25.
    pub const COLOR_ATTACHMENT_FEEDBACK_LOOP: Self = Self(33554432u32);
    ///Bit 26.
    pub const DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP: Self = Self(67108864u32);
    ///Bit 24.
    pub const RAY_TRACING_OPACITY_MICROMAP: Self = Self(16777216u32);
    ///Bit 28.
    pub const RAY_TRACING_DISPLACEMENT_MICROMAP: Self = Self(268435456u32);
}
impl core::ops::BitOr for PipelineCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for PipelineCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for PipelineCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for PipelineCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for PipelineCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for PipelineCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for PipelineCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for PipelineCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::DISABLE_OPTIMIZATION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DISABLE_OPTIMIZATION")?;
            remaining &= !Self::DISABLE_OPTIMIZATION.0;
            first = false;
        }
        if remaining & Self::ALLOW_DERIVATIVES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALLOW_DERIVATIVES")?;
            remaining &= !Self::ALLOW_DERIVATIVES.0;
            first = false;
        }
        if remaining & Self::DERIVATIVE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DERIVATIVE")?;
            remaining &= !Self::DERIVATIVE.0;
            first = false;
        }
        if remaining & Self::DISPATCH_BASE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DISPATCH_BASE")?;
            remaining &= !Self::DISPATCH_BASE.0;
            first = false;
        }
        if remaining & Self::VIEW_INDEX_FROM_DEVICE_INDEX.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VIEW_INDEX_FROM_DEVICE_INDEX")?;
            remaining &= !Self::VIEW_INDEX_FROM_DEVICE_INDEX.0;
            first = false;
        }
        if remaining & Self::FAIL_ON_PIPELINE_COMPILE_REQUIRED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FAIL_ON_PIPELINE_COMPILE_REQUIRED")?;
            remaining &= !Self::FAIL_ON_PIPELINE_COMPILE_REQUIRED.0;
            first = false;
        }
        if remaining & Self::EARLY_RETURN_ON_FAILURE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("EARLY_RETURN_ON_FAILURE")?;
            remaining &= !Self::EARLY_RETURN_ON_FAILURE.0;
            first = false;
        }
        if remaining & Self::NO_PROTECTED_ACCESS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("NO_PROTECTED_ACCESS")?;
            remaining &= !Self::NO_PROTECTED_ACCESS.0;
            first = false;
        }
        if remaining & Self::PROTECTED_ACCESS_ONLY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PROTECTED_ACCESS_ONLY")?;
            remaining &= !Self::PROTECTED_ACCESS_ONLY.0;
            first = false;
        }
        if remaining & Self::RAY_TRACING_NO_NULL_ANY_HIT_SHADERS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RAY_TRACING_NO_NULL_ANY_HIT_SHADERS")?;
            remaining &= !Self::RAY_TRACING_NO_NULL_ANY_HIT_SHADERS.0;
            first = false;
        }
        if remaining & Self::RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS")?;
            remaining &= !Self::RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS.0;
            first = false;
        }
        if remaining & Self::RAY_TRACING_NO_NULL_MISS_SHADERS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RAY_TRACING_NO_NULL_MISS_SHADERS")?;
            remaining &= !Self::RAY_TRACING_NO_NULL_MISS_SHADERS.0;
            first = false;
        }
        if remaining & Self::RAY_TRACING_NO_NULL_INTERSECTION_SHADERS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RAY_TRACING_NO_NULL_INTERSECTION_SHADERS")?;
            remaining &= !Self::RAY_TRACING_NO_NULL_INTERSECTION_SHADERS.0;
            first = false;
        }
        if remaining & Self::RAY_TRACING_SKIP_TRIANGLES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RAY_TRACING_SKIP_TRIANGLES")?;
            remaining &= !Self::RAY_TRACING_SKIP_TRIANGLES.0;
            first = false;
        }
        if remaining & Self::RAY_TRACING_SKIP_AABBS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RAY_TRACING_SKIP_AABBS")?;
            remaining &= !Self::RAY_TRACING_SKIP_AABBS.0;
            first = false;
        }
        if remaining & Self::RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY")?;
            remaining &= !Self::RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY.0;
            first = false;
        }
        if remaining & Self::DEFER_COMPILE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEFER_COMPILE")?;
            remaining &= !Self::DEFER_COMPILE.0;
            first = false;
        }
        if remaining & Self::RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT")?;
            remaining &= !Self::RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT.0;
            first = false;
        }
        if remaining & Self::RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT")?;
            remaining &= !Self::RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT.0;
            first = false;
        }
        if remaining & Self::CAPTURE_STATISTICS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CAPTURE_STATISTICS")?;
            remaining &= !Self::CAPTURE_STATISTICS.0;
            first = false;
        }
        if remaining & Self::CAPTURE_INTERNAL_REPRESENTATIONS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CAPTURE_INTERNAL_REPRESENTATIONS")?;
            remaining &= !Self::CAPTURE_INTERNAL_REPRESENTATIONS.0;
            first = false;
        }
        if remaining & Self::INDIRECT_BINDABLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INDIRECT_BINDABLE")?;
            remaining &= !Self::INDIRECT_BINDABLE.0;
            first = false;
        }
        if remaining & Self::LIBRARY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("LIBRARY")?;
            remaining &= !Self::LIBRARY.0;
            first = false;
        }
        if remaining & Self::DESCRIPTOR_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DESCRIPTOR_BUFFER")?;
            remaining &= !Self::DESCRIPTOR_BUFFER.0;
            first = false;
        }
        if remaining & Self::RETAIN_LINK_TIME_OPTIMIZATION_INFO.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RETAIN_LINK_TIME_OPTIMIZATION_INFO")?;
            remaining &= !Self::RETAIN_LINK_TIME_OPTIMIZATION_INFO.0;
            first = false;
        }
        if remaining & Self::LINK_TIME_OPTIMIZATION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("LINK_TIME_OPTIMIZATION")?;
            remaining &= !Self::LINK_TIME_OPTIMIZATION.0;
            first = false;
        }
        if remaining & Self::RAY_TRACING_ALLOW_MOTION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RAY_TRACING_ALLOW_MOTION")?;
            remaining &= !Self::RAY_TRACING_ALLOW_MOTION.0;
            first = false;
        }
        if remaining & Self::COLOR_ATTACHMENT_FEEDBACK_LOOP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("COLOR_ATTACHMENT_FEEDBACK_LOOP")?;
            remaining &= !Self::COLOR_ATTACHMENT_FEEDBACK_LOOP.0;
            first = false;
        }
        if remaining & Self::DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP")?;
            remaining &= !Self::DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP.0;
            first = false;
        }
        if remaining & Self::RAY_TRACING_OPACITY_MICROMAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RAY_TRACING_OPACITY_MICROMAP")?;
            remaining &= !Self::RAY_TRACING_OPACITY_MICROMAP.0;
            first = false;
        }
        if remaining & Self::RAY_TRACING_DISPLACEMENT_MICROMAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RAY_TRACING_DISPLACEMENT_MICROMAP")?;
            remaining &= !Self::RAY_TRACING_DISPLACEMENT_MICROMAP.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkPipelineCreateFlagBits2`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineCreateFlagBits2.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkPipelineCreateFlagBits2")]
pub struct PipelineCreateFlagBits2(u64);
impl PipelineCreateFlagBits2 {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u64)
    }
    #[inline]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u64
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const _2_DISABLE_OPTIMIZATION: Self = Self(1u64);
    ///Bit 1.
    pub const _2_ALLOW_DERIVATIVES: Self = Self(2u64);
    ///Bit 2.
    pub const _2_DERIVATIVE: Self = Self(4u64);
    ///Bit 3.
    pub const _2_VIEW_INDEX_FROM_DEVICE_INDEX: Self = Self(8u64);
    ///Bit 4.
    pub const _2_DISPATCH_BASE: Self = Self(16u64);
    ///Bit 8.
    pub const _2_FAIL_ON_PIPELINE_COMPILE_REQUIRED: Self = Self(256u64);
    ///Bit 9.
    pub const _2_EARLY_RETURN_ON_FAILURE: Self = Self(512u64);
    ///Bit 27.
    pub const _2_NO_PROTECTED_ACCESS: Self = Self(134217728u64);
    ///Bit 30.
    pub const _2_PROTECTED_ACCESS_ONLY: Self = Self(1073741824u64);
    ///Bit 32.
    pub const _2_EXECUTION_GRAPH_BIT: Self = Self(4294967296u64);
    ///Bit 36.
    pub const _2_DESCRIPTOR_HEAP: Self = Self(68719476736u64);
    pub const _2_RAY_TRACING_SKIP_BUILT_IN_PRIMITIVES: Self = Self::_2_RAY_TRACING_SKIP_TRIANGLES;
    ///Bit 33.
    pub const _2_RAY_TRACING_ALLOW_SPHERES_AND_LINEAR_SWEPT_SPHERES: Self = Self(
        8589934592u64,
    );
    ///Bit 34.
    pub const _2_ENABLE_LEGACY_DITHERING: Self = Self(17179869184u64);
    ///Bit 5.
    pub const _2_DEFER_COMPILE: Self = Self(32u64);
    ///Bit 6.
    pub const _2_CAPTURE_STATISTICS: Self = Self(64u64);
    ///Bit 7.
    pub const _2_CAPTURE_INTERNAL_REPRESENTATIONS: Self = Self(128u64);
    ///Bit 10.
    pub const _2_LINK_TIME_OPTIMIZATION: Self = Self(1024u64);
    ///Bit 23.
    pub const _2_RETAIN_LINK_TIME_OPTIMIZATION_INFO: Self = Self(8388608u64);
    ///Bit 11.
    pub const _2_LIBRARY: Self = Self(2048u64);
    ///Bit 12.
    pub const _2_RAY_TRACING_SKIP_TRIANGLES: Self = Self(4096u64);
    ///Bit 13.
    pub const _2_RAY_TRACING_SKIP_AABBS: Self = Self(8192u64);
    ///Bit 14.
    pub const _2_RAY_TRACING_NO_NULL_ANY_HIT_SHADERS: Self = Self(16384u64);
    ///Bit 15.
    pub const _2_RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS: Self = Self(32768u64);
    ///Bit 16.
    pub const _2_RAY_TRACING_NO_NULL_MISS_SHADERS: Self = Self(65536u64);
    ///Bit 17.
    pub const _2_RAY_TRACING_NO_NULL_INTERSECTION_SHADERS: Self = Self(131072u64);
    ///Bit 19.
    pub const _2_RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY: Self = Self(524288u64);
    ///Bit 18.
    pub const _2_INDIRECT_BINDABLE: Self = Self(262144u64);
    ///Bit 20.
    pub const _2_RAY_TRACING_ALLOW_MOTION: Self = Self(1048576u64);
    ///Bit 21.
    pub const _2_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT: Self = Self(2097152u64);
    ///Bit 22.
    pub const _2_RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT: Self = Self(4194304u64);
    ///Bit 24.
    pub const _2_RAY_TRACING_OPACITY_MICROMAP: Self = Self(16777216u64);
    ///Bit 25.
    pub const _2_COLOR_ATTACHMENT_FEEDBACK_LOOP: Self = Self(33554432u64);
    ///Bit 26.
    pub const _2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP: Self = Self(67108864u64);
    ///Bit 28.
    pub const _2_RAY_TRACING_DISPLACEMENT_MICROMAP: Self = Self(268435456u64);
    ///Bit 29.
    pub const _2_DESCRIPTOR_BUFFER: Self = Self(536870912u64);
    ///Bit 37.
    pub const _2_DISALLOW_OPACITY_MICROMAP_BIT: Self = Self(137438953472u64);
    ///Bit 39.
    pub const _2_INSTRUMENT_SHADERS_BIT: Self = Self(549755813888u64);
    ///Bit 31.
    pub const _2_CAPTURE_DATA: Self = Self(2147483648u64);
    ///Bit 40.
    pub const _2_PER_LAYER_FRAGMENT_DENSITY_BIT: Self = Self(1099511627776u64);
    ///Bit 43.
    pub const _2_64_BIT_INDEXING: Self = Self(8796093022208u64);
}
impl core::ops::BitOr for PipelineCreateFlagBits2 {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for PipelineCreateFlagBits2 {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for PipelineCreateFlagBits2 {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for PipelineCreateFlagBits2 {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for PipelineCreateFlagBits2 {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for PipelineCreateFlagBits2 {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for PipelineCreateFlagBits2 {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for PipelineCreateFlagBits2 {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::_2_DISABLE_OPTIMIZATION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_DISABLE_OPTIMIZATION")?;
            remaining &= !Self::_2_DISABLE_OPTIMIZATION.0;
            first = false;
        }
        if remaining & Self::_2_ALLOW_DERIVATIVES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_ALLOW_DERIVATIVES")?;
            remaining &= !Self::_2_ALLOW_DERIVATIVES.0;
            first = false;
        }
        if remaining & Self::_2_DERIVATIVE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_DERIVATIVE")?;
            remaining &= !Self::_2_DERIVATIVE.0;
            first = false;
        }
        if remaining & Self::_2_VIEW_INDEX_FROM_DEVICE_INDEX.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VIEW_INDEX_FROM_DEVICE_INDEX")?;
            remaining &= !Self::_2_VIEW_INDEX_FROM_DEVICE_INDEX.0;
            first = false;
        }
        if remaining & Self::_2_DISPATCH_BASE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_DISPATCH_BASE")?;
            remaining &= !Self::_2_DISPATCH_BASE.0;
            first = false;
        }
        if remaining & Self::_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED")?;
            remaining &= !Self::_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED.0;
            first = false;
        }
        if remaining & Self::_2_EARLY_RETURN_ON_FAILURE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_EARLY_RETURN_ON_FAILURE")?;
            remaining &= !Self::_2_EARLY_RETURN_ON_FAILURE.0;
            first = false;
        }
        if remaining & Self::_2_NO_PROTECTED_ACCESS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_NO_PROTECTED_ACCESS")?;
            remaining &= !Self::_2_NO_PROTECTED_ACCESS.0;
            first = false;
        }
        if remaining & Self::_2_PROTECTED_ACCESS_ONLY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_PROTECTED_ACCESS_ONLY")?;
            remaining &= !Self::_2_PROTECTED_ACCESS_ONLY.0;
            first = false;
        }
        if remaining & Self::_2_EXECUTION_GRAPH_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_EXECUTION_GRAPH_BIT")?;
            remaining &= !Self::_2_EXECUTION_GRAPH_BIT.0;
            first = false;
        }
        if remaining & Self::_2_DESCRIPTOR_HEAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_DESCRIPTOR_HEAP")?;
            remaining &= !Self::_2_DESCRIPTOR_HEAP.0;
            first = false;
        }
        if remaining & Self::_2_RAY_TRACING_ALLOW_SPHERES_AND_LINEAR_SWEPT_SPHERES.0 != 0
        {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_RAY_TRACING_ALLOW_SPHERES_AND_LINEAR_SWEPT_SPHERES")?;
            remaining &= !Self::_2_RAY_TRACING_ALLOW_SPHERES_AND_LINEAR_SWEPT_SPHERES.0;
            first = false;
        }
        if remaining & Self::_2_ENABLE_LEGACY_DITHERING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_ENABLE_LEGACY_DITHERING")?;
            remaining &= !Self::_2_ENABLE_LEGACY_DITHERING.0;
            first = false;
        }
        if remaining & Self::_2_DEFER_COMPILE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_DEFER_COMPILE")?;
            remaining &= !Self::_2_DEFER_COMPILE.0;
            first = false;
        }
        if remaining & Self::_2_CAPTURE_STATISTICS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_CAPTURE_STATISTICS")?;
            remaining &= !Self::_2_CAPTURE_STATISTICS.0;
            first = false;
        }
        if remaining & Self::_2_CAPTURE_INTERNAL_REPRESENTATIONS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_CAPTURE_INTERNAL_REPRESENTATIONS")?;
            remaining &= !Self::_2_CAPTURE_INTERNAL_REPRESENTATIONS.0;
            first = false;
        }
        if remaining & Self::_2_LINK_TIME_OPTIMIZATION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_LINK_TIME_OPTIMIZATION")?;
            remaining &= !Self::_2_LINK_TIME_OPTIMIZATION.0;
            first = false;
        }
        if remaining & Self::_2_RETAIN_LINK_TIME_OPTIMIZATION_INFO.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_RETAIN_LINK_TIME_OPTIMIZATION_INFO")?;
            remaining &= !Self::_2_RETAIN_LINK_TIME_OPTIMIZATION_INFO.0;
            first = false;
        }
        if remaining & Self::_2_LIBRARY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_LIBRARY")?;
            remaining &= !Self::_2_LIBRARY.0;
            first = false;
        }
        if remaining & Self::_2_RAY_TRACING_SKIP_TRIANGLES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_RAY_TRACING_SKIP_TRIANGLES")?;
            remaining &= !Self::_2_RAY_TRACING_SKIP_TRIANGLES.0;
            first = false;
        }
        if remaining & Self::_2_RAY_TRACING_SKIP_AABBS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_RAY_TRACING_SKIP_AABBS")?;
            remaining &= !Self::_2_RAY_TRACING_SKIP_AABBS.0;
            first = false;
        }
        if remaining & Self::_2_RAY_TRACING_NO_NULL_ANY_HIT_SHADERS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_RAY_TRACING_NO_NULL_ANY_HIT_SHADERS")?;
            remaining &= !Self::_2_RAY_TRACING_NO_NULL_ANY_HIT_SHADERS.0;
            first = false;
        }
        if remaining & Self::_2_RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS")?;
            remaining &= !Self::_2_RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS.0;
            first = false;
        }
        if remaining & Self::_2_RAY_TRACING_NO_NULL_MISS_SHADERS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_RAY_TRACING_NO_NULL_MISS_SHADERS")?;
            remaining &= !Self::_2_RAY_TRACING_NO_NULL_MISS_SHADERS.0;
            first = false;
        }
        if remaining & Self::_2_RAY_TRACING_NO_NULL_INTERSECTION_SHADERS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_RAY_TRACING_NO_NULL_INTERSECTION_SHADERS")?;
            remaining &= !Self::_2_RAY_TRACING_NO_NULL_INTERSECTION_SHADERS.0;
            first = false;
        }
        if remaining & Self::_2_RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY")?;
            remaining &= !Self::_2_RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY.0;
            first = false;
        }
        if remaining & Self::_2_INDIRECT_BINDABLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_INDIRECT_BINDABLE")?;
            remaining &= !Self::_2_INDIRECT_BINDABLE.0;
            first = false;
        }
        if remaining & Self::_2_RAY_TRACING_ALLOW_MOTION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_RAY_TRACING_ALLOW_MOTION")?;
            remaining &= !Self::_2_RAY_TRACING_ALLOW_MOTION.0;
            first = false;
        }
        if remaining & Self::_2_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT")?;
            remaining &= !Self::_2_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT.0;
            first = false;
        }
        if remaining & Self::_2_RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT")?;
            remaining &= !Self::_2_RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT.0;
            first = false;
        }
        if remaining & Self::_2_RAY_TRACING_OPACITY_MICROMAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_RAY_TRACING_OPACITY_MICROMAP")?;
            remaining &= !Self::_2_RAY_TRACING_OPACITY_MICROMAP.0;
            first = false;
        }
        if remaining & Self::_2_COLOR_ATTACHMENT_FEEDBACK_LOOP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_COLOR_ATTACHMENT_FEEDBACK_LOOP")?;
            remaining &= !Self::_2_COLOR_ATTACHMENT_FEEDBACK_LOOP.0;
            first = false;
        }
        if remaining & Self::_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP")?;
            remaining &= !Self::_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP.0;
            first = false;
        }
        if remaining & Self::_2_RAY_TRACING_DISPLACEMENT_MICROMAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_RAY_TRACING_DISPLACEMENT_MICROMAP")?;
            remaining &= !Self::_2_RAY_TRACING_DISPLACEMENT_MICROMAP.0;
            first = false;
        }
        if remaining & Self::_2_DESCRIPTOR_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_DESCRIPTOR_BUFFER")?;
            remaining &= !Self::_2_DESCRIPTOR_BUFFER.0;
            first = false;
        }
        if remaining & Self::_2_DISALLOW_OPACITY_MICROMAP_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_DISALLOW_OPACITY_MICROMAP_BIT")?;
            remaining &= !Self::_2_DISALLOW_OPACITY_MICROMAP_BIT.0;
            first = false;
        }
        if remaining & Self::_2_INSTRUMENT_SHADERS_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_INSTRUMENT_SHADERS_BIT")?;
            remaining &= !Self::_2_INSTRUMENT_SHADERS_BIT.0;
            first = false;
        }
        if remaining & Self::_2_CAPTURE_DATA.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_CAPTURE_DATA")?;
            remaining &= !Self::_2_CAPTURE_DATA.0;
            first = false;
        }
        if remaining & Self::_2_PER_LAYER_FRAGMENT_DENSITY_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_PER_LAYER_FRAGMENT_DENSITY_BIT")?;
            remaining &= !Self::_2_PER_LAYER_FRAGMENT_DENSITY_BIT.0;
            first = false;
        }
        if remaining & Self::_2_64_BIT_INDEXING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_64_BIT_INDEXING")?;
            remaining &= !Self::_2_64_BIT_INDEXING.0;
            first = false;
        }
        if remaining != 0u64 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkPipelineCreationFeedbackFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineCreationFeedbackFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkPipelineCreationFeedbackFlagBits")]
pub struct PipelineCreationFeedbackFlagBits(u32);
impl PipelineCreationFeedbackFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const VALID: Self = Self(1u32);
    ///Bit 1.
    pub const APPLICATION_PIPELINE_CACHE_HIT: Self = Self(2u32);
    ///Bit 2.
    pub const BASE_PIPELINE_ACCELERATION: Self = Self(4u32);
}
impl core::ops::BitOr for PipelineCreationFeedbackFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for PipelineCreationFeedbackFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for PipelineCreationFeedbackFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for PipelineCreationFeedbackFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for PipelineCreationFeedbackFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for PipelineCreationFeedbackFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for PipelineCreationFeedbackFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for PipelineCreationFeedbackFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::VALID.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VALID")?;
            remaining &= !Self::VALID.0;
            first = false;
        }
        if remaining & Self::APPLICATION_PIPELINE_CACHE_HIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("APPLICATION_PIPELINE_CACHE_HIT")?;
            remaining &= !Self::APPLICATION_PIPELINE_CACHE_HIT.0;
            first = false;
        }
        if remaining & Self::BASE_PIPELINE_ACCELERATION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("BASE_PIPELINE_ACCELERATION")?;
            remaining &= !Self::BASE_PIPELINE_ACCELERATION.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkPipelineDepthStencilStateCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineDepthStencilStateCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkPipelineDepthStencilStateCreateFlagBits")]
pub struct PipelineDepthStencilStateCreateFlagBits(u32);
impl PipelineDepthStencilStateCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_BIT: Self = Self::RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS;
    pub const RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_BIT: Self = Self::RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS;
    ///Bit 0.
    pub const RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS: Self = Self(1u32);
    ///Bit 1.
    pub const RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS: Self = Self(2u32);
}
impl core::ops::BitOr for PipelineDepthStencilStateCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for PipelineDepthStencilStateCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for PipelineDepthStencilStateCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for PipelineDepthStencilStateCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for PipelineDepthStencilStateCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for PipelineDepthStencilStateCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for PipelineDepthStencilStateCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for PipelineDepthStencilStateCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS")?;
            remaining &= !Self::RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS.0;
            first = false;
        }
        if remaining & Self::RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS")?;
            remaining &= !Self::RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkPipelineLayoutCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineLayoutCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkPipelineLayoutCreateFlagBits")]
pub struct PipelineLayoutCreateFlagBits(u32);
impl PipelineLayoutCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 1.
    pub const INDEPENDENT_SETS: Self = Self(2u32);
}
impl core::ops::BitOr for PipelineLayoutCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for PipelineLayoutCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for PipelineLayoutCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for PipelineLayoutCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for PipelineLayoutCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for PipelineLayoutCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for PipelineLayoutCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for PipelineLayoutCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::INDEPENDENT_SETS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INDEPENDENT_SETS")?;
            remaining &= !Self::INDEPENDENT_SETS.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkPipelineShaderStageCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineShaderStageCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkPipelineShaderStageCreateFlagBits")]
pub struct PipelineShaderStageCreateFlagBits(u32);
impl PipelineShaderStageCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const ALLOW_VARYING_SUBGROUP_SIZE: Self = Self(1u32);
    ///Bit 1.
    pub const REQUIRE_FULL_SUBGROUPS: Self = Self(2u32);
}
impl core::ops::BitOr for PipelineShaderStageCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for PipelineShaderStageCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for PipelineShaderStageCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for PipelineShaderStageCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for PipelineShaderStageCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for PipelineShaderStageCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for PipelineShaderStageCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for PipelineShaderStageCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::ALLOW_VARYING_SUBGROUP_SIZE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALLOW_VARYING_SUBGROUP_SIZE")?;
            remaining &= !Self::ALLOW_VARYING_SUBGROUP_SIZE.0;
            first = false;
        }
        if remaining & Self::REQUIRE_FULL_SUBGROUPS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("REQUIRE_FULL_SUBGROUPS")?;
            remaining &= !Self::REQUIRE_FULL_SUBGROUPS.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkPipelineStageFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineStageFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkPipelineStageFlagBits")]
pub struct PipelineStageFlagBits(u32);
impl PipelineStageFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const TOP_OF_PIPE: Self = Self(1u32);
    ///Bit 1.
    pub const DRAW_INDIRECT: Self = Self(2u32);
    ///Bit 2.
    pub const VERTEX_INPUT: Self = Self(4u32);
    ///Bit 3.
    pub const VERTEX_SHADER: Self = Self(8u32);
    ///Bit 4.
    pub const TESSELLATION_CONTROL_SHADER: Self = Self(16u32);
    ///Bit 5.
    pub const TESSELLATION_EVALUATION_SHADER: Self = Self(32u32);
    ///Bit 6.
    pub const GEOMETRY_SHADER: Self = Self(64u32);
    ///Bit 7.
    pub const FRAGMENT_SHADER: Self = Self(128u32);
    ///Bit 8.
    pub const EARLY_FRAGMENT_TESTS: Self = Self(256u32);
    ///Bit 9.
    pub const LATE_FRAGMENT_TESTS: Self = Self(512u32);
    ///Bit 10.
    pub const COLOR_ATTACHMENT_OUTPUT: Self = Self(1024u32);
    ///Bit 11.
    pub const COMPUTE_SHADER: Self = Self(2048u32);
    ///Bit 12.
    pub const TRANSFER: Self = Self(4096u32);
    ///Bit 13.
    pub const BOTTOM_OF_PIPE: Self = Self(8192u32);
    ///Bit 14.
    pub const HOST: Self = Self(16384u32);
    ///Bit 15.
    pub const ALL_GRAPHICS: Self = Self(32768u32);
    ///Bit 16.
    pub const ALL_COMMANDS: Self = Self(65536u32);
    pub const NONE: Self = Self(0u32);
    ///Bit 24.
    pub const TRANSFORM_FEEDBACK: Self = Self(16777216u32);
    ///Bit 18.
    pub const CONDITIONAL_RENDERING: Self = Self(262144u32);
    ///Bit 25.
    pub const ACCELERATION_STRUCTURE_BUILD: Self = Self(33554432u32);
    ///Bit 21.
    pub const RAY_TRACING_SHADER: Self = Self(2097152u32);
    pub const SHADING_RATE_IMAGE: Self = Self::FRAGMENT_SHADING_RATE_ATTACHMENT;
    ///Bit 23.
    pub const FRAGMENT_DENSITY_PROCESS: Self = Self(8388608u32);
    ///Bit 22.
    pub const FRAGMENT_SHADING_RATE_ATTACHMENT: Self = Self(4194304u32);
}
impl core::ops::BitOr for PipelineStageFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for PipelineStageFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for PipelineStageFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for PipelineStageFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for PipelineStageFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for PipelineStageFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for PipelineStageFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for PipelineStageFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::TOP_OF_PIPE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TOP_OF_PIPE")?;
            remaining &= !Self::TOP_OF_PIPE.0;
            first = false;
        }
        if remaining & Self::DRAW_INDIRECT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DRAW_INDIRECT")?;
            remaining &= !Self::DRAW_INDIRECT.0;
            first = false;
        }
        if remaining & Self::VERTEX_INPUT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VERTEX_INPUT")?;
            remaining &= !Self::VERTEX_INPUT.0;
            first = false;
        }
        if remaining & Self::VERTEX_SHADER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VERTEX_SHADER")?;
            remaining &= !Self::VERTEX_SHADER.0;
            first = false;
        }
        if remaining & Self::TESSELLATION_CONTROL_SHADER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TESSELLATION_CONTROL_SHADER")?;
            remaining &= !Self::TESSELLATION_CONTROL_SHADER.0;
            first = false;
        }
        if remaining & Self::TESSELLATION_EVALUATION_SHADER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TESSELLATION_EVALUATION_SHADER")?;
            remaining &= !Self::TESSELLATION_EVALUATION_SHADER.0;
            first = false;
        }
        if remaining & Self::GEOMETRY_SHADER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("GEOMETRY_SHADER")?;
            remaining &= !Self::GEOMETRY_SHADER.0;
            first = false;
        }
        if remaining & Self::FRAGMENT_SHADER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRAGMENT_SHADER")?;
            remaining &= !Self::FRAGMENT_SHADER.0;
            first = false;
        }
        if remaining & Self::EARLY_FRAGMENT_TESTS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("EARLY_FRAGMENT_TESTS")?;
            remaining &= !Self::EARLY_FRAGMENT_TESTS.0;
            first = false;
        }
        if remaining & Self::LATE_FRAGMENT_TESTS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("LATE_FRAGMENT_TESTS")?;
            remaining &= !Self::LATE_FRAGMENT_TESTS.0;
            first = false;
        }
        if remaining & Self::COLOR_ATTACHMENT_OUTPUT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("COLOR_ATTACHMENT_OUTPUT")?;
            remaining &= !Self::COLOR_ATTACHMENT_OUTPUT.0;
            first = false;
        }
        if remaining & Self::COMPUTE_SHADER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("COMPUTE_SHADER")?;
            remaining &= !Self::COMPUTE_SHADER.0;
            first = false;
        }
        if remaining & Self::TRANSFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFER")?;
            remaining &= !Self::TRANSFER.0;
            first = false;
        }
        if remaining & Self::BOTTOM_OF_PIPE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("BOTTOM_OF_PIPE")?;
            remaining &= !Self::BOTTOM_OF_PIPE.0;
            first = false;
        }
        if remaining & Self::HOST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("HOST")?;
            remaining &= !Self::HOST.0;
            first = false;
        }
        if remaining & Self::ALL_GRAPHICS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALL_GRAPHICS")?;
            remaining &= !Self::ALL_GRAPHICS.0;
            first = false;
        }
        if remaining & Self::ALL_COMMANDS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALL_COMMANDS")?;
            remaining &= !Self::ALL_COMMANDS.0;
            first = false;
        }
        if remaining & Self::TRANSFORM_FEEDBACK.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFORM_FEEDBACK")?;
            remaining &= !Self::TRANSFORM_FEEDBACK.0;
            first = false;
        }
        if remaining & Self::CONDITIONAL_RENDERING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CONDITIONAL_RENDERING")?;
            remaining &= !Self::CONDITIONAL_RENDERING.0;
            first = false;
        }
        if remaining & Self::ACCELERATION_STRUCTURE_BUILD.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ACCELERATION_STRUCTURE_BUILD")?;
            remaining &= !Self::ACCELERATION_STRUCTURE_BUILD.0;
            first = false;
        }
        if remaining & Self::RAY_TRACING_SHADER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RAY_TRACING_SHADER")?;
            remaining &= !Self::RAY_TRACING_SHADER.0;
            first = false;
        }
        if remaining & Self::FRAGMENT_DENSITY_PROCESS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRAGMENT_DENSITY_PROCESS")?;
            remaining &= !Self::FRAGMENT_DENSITY_PROCESS.0;
            first = false;
        }
        if remaining & Self::FRAGMENT_SHADING_RATE_ATTACHMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRAGMENT_SHADING_RATE_ATTACHMENT")?;
            remaining &= !Self::FRAGMENT_SHADING_RATE_ATTACHMENT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkPipelineStageFlagBits2`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineStageFlagBits2.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkPipelineStageFlagBits2")]
pub struct PipelineStageFlagBits2(u64);
impl PipelineStageFlagBits2 {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u64)
    }
    #[inline]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u64
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const _2_NONE: Self = Self(0u64);
    ///Bit 0.
    pub const _2_TOP_OF_PIPE: Self = Self(1u64);
    ///Bit 1.
    pub const _2_DRAW_INDIRECT: Self = Self(2u64);
    ///Bit 2.
    pub const _2_VERTEX_INPUT: Self = Self(4u64);
    ///Bit 3.
    pub const _2_VERTEX_SHADER: Self = Self(8u64);
    ///Bit 4.
    pub const _2_TESSELLATION_CONTROL_SHADER: Self = Self(16u64);
    ///Bit 5.
    pub const _2_TESSELLATION_EVALUATION_SHADER: Self = Self(32u64);
    ///Bit 6.
    pub const _2_GEOMETRY_SHADER: Self = Self(64u64);
    ///Bit 7.
    pub const _2_FRAGMENT_SHADER: Self = Self(128u64);
    ///Bit 8.
    pub const _2_EARLY_FRAGMENT_TESTS: Self = Self(256u64);
    ///Bit 9.
    pub const _2_LATE_FRAGMENT_TESTS: Self = Self(512u64);
    ///Bit 10.
    pub const _2_COLOR_ATTACHMENT_OUTPUT: Self = Self(1024u64);
    ///Bit 11.
    pub const _2_COMPUTE_SHADER: Self = Self(2048u64);
    ///Bit 12.
    pub const _2_ALL_TRANSFER: Self = Self(4096u64);
    pub const _2_TRANSFER: Self = Self::_2_ALL_TRANSFER;
    ///Bit 13.
    pub const _2_BOTTOM_OF_PIPE: Self = Self(8192u64);
    ///Bit 14.
    pub const _2_HOST: Self = Self(16384u64);
    ///Bit 15.
    pub const _2_ALL_GRAPHICS: Self = Self(32768u64);
    ///Bit 16.
    pub const _2_ALL_COMMANDS: Self = Self(65536u64);
    ///Bit 32.
    pub const _2_COPY: Self = Self(4294967296u64);
    ///Bit 33.
    pub const _2_RESOLVE: Self = Self(8589934592u64);
    ///Bit 34.
    pub const _2_BLIT: Self = Self(17179869184u64);
    ///Bit 35.
    pub const _2_CLEAR: Self = Self(34359738368u64);
    ///Bit 36.
    pub const _2_INDEX_INPUT: Self = Self(68719476736u64);
    ///Bit 37.
    pub const _2_VERTEX_ATTRIBUTE_INPUT: Self = Self(137438953472u64);
    ///Bit 38.
    pub const _2_PRE_RASTERIZATION_SHADERS: Self = Self(274877906944u64);
    ///Bit 26.
    pub const _2_VIDEO_DECODE: Self = Self(67108864u64);
    ///Bit 27.
    pub const _2_VIDEO_ENCODE: Self = Self(134217728u64);
    ///Bit 24.
    pub const _2_TRANSFORM_FEEDBACK: Self = Self(16777216u64);
    ///Bit 18.
    pub const _2_CONDITIONAL_RENDERING: Self = Self(262144u64);
    ///Bit 22.
    pub const _2_FRAGMENT_SHADING_RATE_ATTACHMENT: Self = Self(4194304u64);
    pub const _2_SHADING_RATE_IMAGE: Self = Self::_2_FRAGMENT_SHADING_RATE_ATTACHMENT;
    ///Bit 25.
    pub const _2_ACCELERATION_STRUCTURE_BUILD: Self = Self(33554432u64);
    ///Bit 21.
    pub const _2_RAY_TRACING_SHADER: Self = Self(2097152u64);
    ///Bit 23.
    pub const _2_FRAGMENT_DENSITY_PROCESS: Self = Self(8388608u64);
    ///Bit 39.
    pub const _2_SUBPASS_SHADER_BIT: Self = Self(549755813888u64);
    pub const _2_SUBPASS_SHADING_BIT: Self = Self::_2_SUBPASS_SHADER_BIT;
    ///Bit 40.
    pub const _2_INVOCATION_MASK_BIT: Self = Self(1099511627776u64);
    ///Bit 28.
    pub const _2_ACCELERATION_STRUCTURE_COPY: Self = Self(268435456u64);
    ///Bit 30.
    pub const _2_MICROMAP_BUILD: Self = Self(1073741824u64);
    ///Bit 41.
    pub const _2_CLUSTER_CULLING_SHADER_BIT: Self = Self(2199023255552u64);
    ///Bit 29.
    pub const _2_OPTICAL_FLOW: Self = Self(536870912u64);
    ///Bit 44.
    pub const _2_CONVERT_COOPERATIVE_VECTOR_MATRIX: Self = Self(17592186044416u64);
    ///Bit 42.
    pub const _2_DATA_GRAPH_BIT: Self = Self(4398046511104u64);
    ///Bit 46.
    pub const _2_COPY_INDIRECT: Self = Self(70368744177664u64);
    ///Bit 45.
    pub const _2_MEMORY_DECOMPRESSION: Self = Self(35184372088832u64);
}
impl core::ops::BitOr for PipelineStageFlagBits2 {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for PipelineStageFlagBits2 {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for PipelineStageFlagBits2 {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for PipelineStageFlagBits2 {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for PipelineStageFlagBits2 {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for PipelineStageFlagBits2 {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for PipelineStageFlagBits2 {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for PipelineStageFlagBits2 {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::_2_TOP_OF_PIPE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_TOP_OF_PIPE")?;
            remaining &= !Self::_2_TOP_OF_PIPE.0;
            first = false;
        }
        if remaining & Self::_2_DRAW_INDIRECT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_DRAW_INDIRECT")?;
            remaining &= !Self::_2_DRAW_INDIRECT.0;
            first = false;
        }
        if remaining & Self::_2_VERTEX_INPUT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VERTEX_INPUT")?;
            remaining &= !Self::_2_VERTEX_INPUT.0;
            first = false;
        }
        if remaining & Self::_2_VERTEX_SHADER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VERTEX_SHADER")?;
            remaining &= !Self::_2_VERTEX_SHADER.0;
            first = false;
        }
        if remaining & Self::_2_TESSELLATION_CONTROL_SHADER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_TESSELLATION_CONTROL_SHADER")?;
            remaining &= !Self::_2_TESSELLATION_CONTROL_SHADER.0;
            first = false;
        }
        if remaining & Self::_2_TESSELLATION_EVALUATION_SHADER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_TESSELLATION_EVALUATION_SHADER")?;
            remaining &= !Self::_2_TESSELLATION_EVALUATION_SHADER.0;
            first = false;
        }
        if remaining & Self::_2_GEOMETRY_SHADER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_GEOMETRY_SHADER")?;
            remaining &= !Self::_2_GEOMETRY_SHADER.0;
            first = false;
        }
        if remaining & Self::_2_FRAGMENT_SHADER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_FRAGMENT_SHADER")?;
            remaining &= !Self::_2_FRAGMENT_SHADER.0;
            first = false;
        }
        if remaining & Self::_2_EARLY_FRAGMENT_TESTS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_EARLY_FRAGMENT_TESTS")?;
            remaining &= !Self::_2_EARLY_FRAGMENT_TESTS.0;
            first = false;
        }
        if remaining & Self::_2_LATE_FRAGMENT_TESTS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_LATE_FRAGMENT_TESTS")?;
            remaining &= !Self::_2_LATE_FRAGMENT_TESTS.0;
            first = false;
        }
        if remaining & Self::_2_COLOR_ATTACHMENT_OUTPUT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_COLOR_ATTACHMENT_OUTPUT")?;
            remaining &= !Self::_2_COLOR_ATTACHMENT_OUTPUT.0;
            first = false;
        }
        if remaining & Self::_2_COMPUTE_SHADER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_COMPUTE_SHADER")?;
            remaining &= !Self::_2_COMPUTE_SHADER.0;
            first = false;
        }
        if remaining & Self::_2_ALL_TRANSFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_ALL_TRANSFER")?;
            remaining &= !Self::_2_ALL_TRANSFER.0;
            first = false;
        }
        if remaining & Self::_2_BOTTOM_OF_PIPE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_BOTTOM_OF_PIPE")?;
            remaining &= !Self::_2_BOTTOM_OF_PIPE.0;
            first = false;
        }
        if remaining & Self::_2_HOST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_HOST")?;
            remaining &= !Self::_2_HOST.0;
            first = false;
        }
        if remaining & Self::_2_ALL_GRAPHICS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_ALL_GRAPHICS")?;
            remaining &= !Self::_2_ALL_GRAPHICS.0;
            first = false;
        }
        if remaining & Self::_2_ALL_COMMANDS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_ALL_COMMANDS")?;
            remaining &= !Self::_2_ALL_COMMANDS.0;
            first = false;
        }
        if remaining & Self::_2_COPY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_COPY")?;
            remaining &= !Self::_2_COPY.0;
            first = false;
        }
        if remaining & Self::_2_RESOLVE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_RESOLVE")?;
            remaining &= !Self::_2_RESOLVE.0;
            first = false;
        }
        if remaining & Self::_2_BLIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_BLIT")?;
            remaining &= !Self::_2_BLIT.0;
            first = false;
        }
        if remaining & Self::_2_CLEAR.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_CLEAR")?;
            remaining &= !Self::_2_CLEAR.0;
            first = false;
        }
        if remaining & Self::_2_INDEX_INPUT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_INDEX_INPUT")?;
            remaining &= !Self::_2_INDEX_INPUT.0;
            first = false;
        }
        if remaining & Self::_2_VERTEX_ATTRIBUTE_INPUT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VERTEX_ATTRIBUTE_INPUT")?;
            remaining &= !Self::_2_VERTEX_ATTRIBUTE_INPUT.0;
            first = false;
        }
        if remaining & Self::_2_PRE_RASTERIZATION_SHADERS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_PRE_RASTERIZATION_SHADERS")?;
            remaining &= !Self::_2_PRE_RASTERIZATION_SHADERS.0;
            first = false;
        }
        if remaining & Self::_2_VIDEO_DECODE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VIDEO_DECODE")?;
            remaining &= !Self::_2_VIDEO_DECODE.0;
            first = false;
        }
        if remaining & Self::_2_VIDEO_ENCODE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_VIDEO_ENCODE")?;
            remaining &= !Self::_2_VIDEO_ENCODE.0;
            first = false;
        }
        if remaining & Self::_2_TRANSFORM_FEEDBACK.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_TRANSFORM_FEEDBACK")?;
            remaining &= !Self::_2_TRANSFORM_FEEDBACK.0;
            first = false;
        }
        if remaining & Self::_2_CONDITIONAL_RENDERING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_CONDITIONAL_RENDERING")?;
            remaining &= !Self::_2_CONDITIONAL_RENDERING.0;
            first = false;
        }
        if remaining & Self::_2_FRAGMENT_SHADING_RATE_ATTACHMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_FRAGMENT_SHADING_RATE_ATTACHMENT")?;
            remaining &= !Self::_2_FRAGMENT_SHADING_RATE_ATTACHMENT.0;
            first = false;
        }
        if remaining & Self::_2_ACCELERATION_STRUCTURE_BUILD.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_ACCELERATION_STRUCTURE_BUILD")?;
            remaining &= !Self::_2_ACCELERATION_STRUCTURE_BUILD.0;
            first = false;
        }
        if remaining & Self::_2_RAY_TRACING_SHADER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_RAY_TRACING_SHADER")?;
            remaining &= !Self::_2_RAY_TRACING_SHADER.0;
            first = false;
        }
        if remaining & Self::_2_FRAGMENT_DENSITY_PROCESS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_FRAGMENT_DENSITY_PROCESS")?;
            remaining &= !Self::_2_FRAGMENT_DENSITY_PROCESS.0;
            first = false;
        }
        if remaining & Self::_2_SUBPASS_SHADER_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_SUBPASS_SHADER_BIT")?;
            remaining &= !Self::_2_SUBPASS_SHADER_BIT.0;
            first = false;
        }
        if remaining & Self::_2_INVOCATION_MASK_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_INVOCATION_MASK_BIT")?;
            remaining &= !Self::_2_INVOCATION_MASK_BIT.0;
            first = false;
        }
        if remaining & Self::_2_ACCELERATION_STRUCTURE_COPY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_ACCELERATION_STRUCTURE_COPY")?;
            remaining &= !Self::_2_ACCELERATION_STRUCTURE_COPY.0;
            first = false;
        }
        if remaining & Self::_2_MICROMAP_BUILD.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_MICROMAP_BUILD")?;
            remaining &= !Self::_2_MICROMAP_BUILD.0;
            first = false;
        }
        if remaining & Self::_2_CLUSTER_CULLING_SHADER_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_CLUSTER_CULLING_SHADER_BIT")?;
            remaining &= !Self::_2_CLUSTER_CULLING_SHADER_BIT.0;
            first = false;
        }
        if remaining & Self::_2_OPTICAL_FLOW.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_OPTICAL_FLOW")?;
            remaining &= !Self::_2_OPTICAL_FLOW.0;
            first = false;
        }
        if remaining & Self::_2_CONVERT_COOPERATIVE_VECTOR_MATRIX.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_CONVERT_COOPERATIVE_VECTOR_MATRIX")?;
            remaining &= !Self::_2_CONVERT_COOPERATIVE_VECTOR_MATRIX.0;
            first = false;
        }
        if remaining & Self::_2_DATA_GRAPH_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_DATA_GRAPH_BIT")?;
            remaining &= !Self::_2_DATA_GRAPH_BIT.0;
            first = false;
        }
        if remaining & Self::_2_COPY_INDIRECT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_COPY_INDIRECT")?;
            remaining &= !Self::_2_COPY_INDIRECT.0;
            first = false;
        }
        if remaining & Self::_2_MEMORY_DECOMPRESSION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2_MEMORY_DECOMPRESSION")?;
            remaining &= !Self::_2_MEMORY_DECOMPRESSION.0;
            first = false;
        }
        if remaining != 0u64 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkPresentGravityFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPresentGravityFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkPresentGravityFlagBitsKHR")]
pub struct PresentGravityFlagBitsKHR(u32);
impl PresentGravityFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const MIN: Self = Self(1u32);
    ///Bit 1.
    pub const MAX: Self = Self(2u32);
    ///Bit 2.
    pub const CENTERED: Self = Self(4u32);
}
impl core::ops::BitOr for PresentGravityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for PresentGravityFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for PresentGravityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for PresentGravityFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for PresentGravityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for PresentGravityFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for PresentGravityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for PresentGravityFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::MIN.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MIN")?;
            remaining &= !Self::MIN.0;
            first = false;
        }
        if remaining & Self::MAX.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MAX")?;
            remaining &= !Self::MAX.0;
            first = false;
        }
        if remaining & Self::CENTERED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CENTERED")?;
            remaining &= !Self::CENTERED.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkPresentScalingFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPresentScalingFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkPresentScalingFlagBitsKHR")]
pub struct PresentScalingFlagBitsKHR(u32);
impl PresentScalingFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const ONE_TO_ONE: Self = Self(1u32);
    ///Bit 1.
    pub const ASPECT_RATIO_STRETCH: Self = Self(2u32);
    ///Bit 2.
    pub const STRETCH: Self = Self(4u32);
}
impl core::ops::BitOr for PresentScalingFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for PresentScalingFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for PresentScalingFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for PresentScalingFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for PresentScalingFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for PresentScalingFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for PresentScalingFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for PresentScalingFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::ONE_TO_ONE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ONE_TO_ONE")?;
            remaining &= !Self::ONE_TO_ONE.0;
            first = false;
        }
        if remaining & Self::ASPECT_RATIO_STRETCH.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ASPECT_RATIO_STRETCH")?;
            remaining &= !Self::ASPECT_RATIO_STRETCH.0;
            first = false;
        }
        if remaining & Self::STRETCH.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("STRETCH")?;
            remaining &= !Self::STRETCH.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkPresentStageFlagBitsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPresentStageFlagBitsEXT.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkPresentStageFlagBitsEXT")]
pub struct PresentStageFlagBitsEXT(u32);
impl PresentStageFlagBitsEXT {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const QUEUE_OPERATIONS_END: Self = Self(1u32);
    ///Bit 1.
    pub const REQUEST_DEQUEUED: Self = Self(2u32);
    ///Bit 2.
    pub const IMAGE_FIRST_PIXEL_OUT: Self = Self(4u32);
    ///Bit 3.
    pub const IMAGE_FIRST_PIXEL_VISIBLE: Self = Self(8u32);
}
impl core::ops::BitOr for PresentStageFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for PresentStageFlagBitsEXT {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for PresentStageFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for PresentStageFlagBitsEXT {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for PresentStageFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for PresentStageFlagBitsEXT {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for PresentStageFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for PresentStageFlagBitsEXT {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::QUEUE_OPERATIONS_END.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("QUEUE_OPERATIONS_END")?;
            remaining &= !Self::QUEUE_OPERATIONS_END.0;
            first = false;
        }
        if remaining & Self::REQUEST_DEQUEUED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("REQUEST_DEQUEUED")?;
            remaining &= !Self::REQUEST_DEQUEUED.0;
            first = false;
        }
        if remaining & Self::IMAGE_FIRST_PIXEL_OUT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("IMAGE_FIRST_PIXEL_OUT")?;
            remaining &= !Self::IMAGE_FIRST_PIXEL_OUT.0;
            first = false;
        }
        if remaining & Self::IMAGE_FIRST_PIXEL_VISIBLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("IMAGE_FIRST_PIXEL_VISIBLE")?;
            remaining &= !Self::IMAGE_FIRST_PIXEL_VISIBLE.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkPresentTimingInfoFlagBitsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPresentTimingInfoFlagBitsEXT.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkPresentTimingInfoFlagBitsEXT")]
pub struct PresentTimingInfoFlagBitsEXT(u32);
impl PresentTimingInfoFlagBitsEXT {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const PRESENT_AT_RELATIVE_TIME: Self = Self(1u32);
    ///Bit 1.
    pub const PRESENT_AT_NEAREST_REFRESH_CYCLE: Self = Self(2u32);
}
impl core::ops::BitOr for PresentTimingInfoFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for PresentTimingInfoFlagBitsEXT {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for PresentTimingInfoFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for PresentTimingInfoFlagBitsEXT {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for PresentTimingInfoFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for PresentTimingInfoFlagBitsEXT {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for PresentTimingInfoFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for PresentTimingInfoFlagBitsEXT {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::PRESENT_AT_RELATIVE_TIME.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PRESENT_AT_RELATIVE_TIME")?;
            remaining &= !Self::PRESENT_AT_RELATIVE_TIME.0;
            first = false;
        }
        if remaining & Self::PRESENT_AT_NEAREST_REFRESH_CYCLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PRESENT_AT_NEAREST_REFRESH_CYCLE")?;
            remaining &= !Self::PRESENT_AT_NEAREST_REFRESH_CYCLE.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkPrivateDataSlotCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPrivateDataSlotCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkPrivateDataSlotCreateFlagBits")]
pub struct PrivateDataSlotCreateFlagBits(u32);
impl PrivateDataSlotCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}
impl core::ops::BitOr for PrivateDataSlotCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for PrivateDataSlotCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for PrivateDataSlotCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for PrivateDataSlotCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for PrivateDataSlotCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for PrivateDataSlotCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for PrivateDataSlotCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for PrivateDataSlotCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkQueryControlFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueryControlFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkQueryControlFlagBits")]
pub struct QueryControlFlagBits(u32);
impl QueryControlFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const PRECISE: Self = Self(1u32);
}
impl core::ops::BitOr for QueryControlFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for QueryControlFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for QueryControlFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for QueryControlFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for QueryControlFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for QueryControlFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for QueryControlFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for QueryControlFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::PRECISE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PRECISE")?;
            remaining &= !Self::PRECISE.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkQueryPipelineStatisticFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueryPipelineStatisticFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkQueryPipelineStatisticFlagBits")]
pub struct QueryPipelineStatisticFlagBits(u32);
impl QueryPipelineStatisticFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const INPUT_ASSEMBLY_VERTICES: Self = Self(1u32);
    ///Bit 1.
    pub const INPUT_ASSEMBLY_PRIMITIVES: Self = Self(2u32);
    ///Bit 2.
    pub const VERTEX_SHADER_INVOCATIONS: Self = Self(4u32);
    ///Bit 3.
    pub const GEOMETRY_SHADER_INVOCATIONS: Self = Self(8u32);
    ///Bit 4.
    pub const GEOMETRY_SHADER_PRIMITIVES: Self = Self(16u32);
    ///Bit 5.
    pub const CLIPPING_INVOCATIONS: Self = Self(32u32);
    ///Bit 6.
    pub const CLIPPING_PRIMITIVES: Self = Self(64u32);
    ///Bit 7.
    pub const FRAGMENT_SHADER_INVOCATIONS: Self = Self(128u32);
    ///Bit 8.
    pub const TESSELLATION_CONTROL_SHADER_PATCHES: Self = Self(256u32);
    ///Bit 9.
    pub const TESSELLATION_EVALUATION_SHADER_INVOCATIONS: Self = Self(512u32);
    ///Bit 10.
    pub const COMPUTE_SHADER_INVOCATIONS: Self = Self(1024u32);
    ///Bit 11.
    pub const TASK_SHADER_INVOCATIONS: Self = Self(2048u32);
    ///Bit 12.
    pub const MESH_SHADER_INVOCATIONS: Self = Self(4096u32);
    ///Bit 13.
    pub const CLUSTER_CULLING_SHADER_INVOCATIONS_BIT: Self = Self(8192u32);
}
impl core::ops::BitOr for QueryPipelineStatisticFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for QueryPipelineStatisticFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for QueryPipelineStatisticFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for QueryPipelineStatisticFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for QueryPipelineStatisticFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for QueryPipelineStatisticFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for QueryPipelineStatisticFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for QueryPipelineStatisticFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::INPUT_ASSEMBLY_VERTICES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INPUT_ASSEMBLY_VERTICES")?;
            remaining &= !Self::INPUT_ASSEMBLY_VERTICES.0;
            first = false;
        }
        if remaining & Self::INPUT_ASSEMBLY_PRIMITIVES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INPUT_ASSEMBLY_PRIMITIVES")?;
            remaining &= !Self::INPUT_ASSEMBLY_PRIMITIVES.0;
            first = false;
        }
        if remaining & Self::VERTEX_SHADER_INVOCATIONS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VERTEX_SHADER_INVOCATIONS")?;
            remaining &= !Self::VERTEX_SHADER_INVOCATIONS.0;
            first = false;
        }
        if remaining & Self::GEOMETRY_SHADER_INVOCATIONS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("GEOMETRY_SHADER_INVOCATIONS")?;
            remaining &= !Self::GEOMETRY_SHADER_INVOCATIONS.0;
            first = false;
        }
        if remaining & Self::GEOMETRY_SHADER_PRIMITIVES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("GEOMETRY_SHADER_PRIMITIVES")?;
            remaining &= !Self::GEOMETRY_SHADER_PRIMITIVES.0;
            first = false;
        }
        if remaining & Self::CLIPPING_INVOCATIONS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CLIPPING_INVOCATIONS")?;
            remaining &= !Self::CLIPPING_INVOCATIONS.0;
            first = false;
        }
        if remaining & Self::CLIPPING_PRIMITIVES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CLIPPING_PRIMITIVES")?;
            remaining &= !Self::CLIPPING_PRIMITIVES.0;
            first = false;
        }
        if remaining & Self::FRAGMENT_SHADER_INVOCATIONS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRAGMENT_SHADER_INVOCATIONS")?;
            remaining &= !Self::FRAGMENT_SHADER_INVOCATIONS.0;
            first = false;
        }
        if remaining & Self::TESSELLATION_CONTROL_SHADER_PATCHES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TESSELLATION_CONTROL_SHADER_PATCHES")?;
            remaining &= !Self::TESSELLATION_CONTROL_SHADER_PATCHES.0;
            first = false;
        }
        if remaining & Self::TESSELLATION_EVALUATION_SHADER_INVOCATIONS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TESSELLATION_EVALUATION_SHADER_INVOCATIONS")?;
            remaining &= !Self::TESSELLATION_EVALUATION_SHADER_INVOCATIONS.0;
            first = false;
        }
        if remaining & Self::COMPUTE_SHADER_INVOCATIONS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("COMPUTE_SHADER_INVOCATIONS")?;
            remaining &= !Self::COMPUTE_SHADER_INVOCATIONS.0;
            first = false;
        }
        if remaining & Self::TASK_SHADER_INVOCATIONS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TASK_SHADER_INVOCATIONS")?;
            remaining &= !Self::TASK_SHADER_INVOCATIONS.0;
            first = false;
        }
        if remaining & Self::MESH_SHADER_INVOCATIONS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MESH_SHADER_INVOCATIONS")?;
            remaining &= !Self::MESH_SHADER_INVOCATIONS.0;
            first = false;
        }
        if remaining & Self::CLUSTER_CULLING_SHADER_INVOCATIONS_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CLUSTER_CULLING_SHADER_INVOCATIONS_BIT")?;
            remaining &= !Self::CLUSTER_CULLING_SHADER_INVOCATIONS_BIT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkQueryPoolCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueryPoolCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkQueryPoolCreateFlagBits")]
pub struct QueryPoolCreateFlagBits(u32);
impl QueryPoolCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const RESET: Self = Self(1u32);
}
impl core::ops::BitOr for QueryPoolCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for QueryPoolCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for QueryPoolCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for QueryPoolCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for QueryPoolCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for QueryPoolCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for QueryPoolCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for QueryPoolCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::RESET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RESET")?;
            remaining &= !Self::RESET.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkQueryResultFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueryResultFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkQueryResultFlagBits")]
pub struct QueryResultFlagBits(u32);
impl QueryResultFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const _64: Self = Self(1u32);
    ///Bit 1.
    pub const WAIT: Self = Self(2u32);
    ///Bit 2.
    pub const WITH_AVAILABILITY: Self = Self(4u32);
    ///Bit 3.
    pub const PARTIAL: Self = Self(8u32);
    ///Bit 4.
    pub const WITH_STATUS: Self = Self(16u32);
}
impl core::ops::BitOr for QueryResultFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for QueryResultFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for QueryResultFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for QueryResultFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for QueryResultFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for QueryResultFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for QueryResultFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for QueryResultFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::_64.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_64")?;
            remaining &= !Self::_64.0;
            first = false;
        }
        if remaining & Self::WAIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("WAIT")?;
            remaining &= !Self::WAIT.0;
            first = false;
        }
        if remaining & Self::WITH_AVAILABILITY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("WITH_AVAILABILITY")?;
            remaining &= !Self::WITH_AVAILABILITY.0;
            first = false;
        }
        if remaining & Self::PARTIAL.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PARTIAL")?;
            remaining &= !Self::PARTIAL.0;
            first = false;
        }
        if remaining & Self::WITH_STATUS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("WITH_STATUS")?;
            remaining &= !Self::WITH_STATUS.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkQueueFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueueFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkQueueFlagBits")]
pub struct QueueFlagBits(u32);
impl QueueFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const GRAPHICS: Self = Self(1u32);
    ///Bit 1.
    pub const COMPUTE: Self = Self(2u32);
    ///Bit 2.
    pub const TRANSFER: Self = Self(4u32);
    ///Bit 3.
    pub const SPARSE_BINDING: Self = Self(8u32);
    ///Bit 4.
    pub const PROTECTED: Self = Self(16u32);
    ///Bit 5.
    pub const VIDEO_DECODE: Self = Self(32u32);
    ///Bit 6.
    pub const VIDEO_ENCODE: Self = Self(64u32);
    ///Bit 8.
    pub const OPTICAL_FLOW: Self = Self(256u32);
    ///Bit 10.
    pub const DATA_GRAPH_BIT: Self = Self(1024u32);
}
impl core::ops::BitOr for QueueFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for QueueFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for QueueFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for QueueFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for QueueFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for QueueFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for QueueFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for QueueFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::GRAPHICS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("GRAPHICS")?;
            remaining &= !Self::GRAPHICS.0;
            first = false;
        }
        if remaining & Self::COMPUTE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("COMPUTE")?;
            remaining &= !Self::COMPUTE.0;
            first = false;
        }
        if remaining & Self::TRANSFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFER")?;
            remaining &= !Self::TRANSFER.0;
            first = false;
        }
        if remaining & Self::SPARSE_BINDING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SPARSE_BINDING")?;
            remaining &= !Self::SPARSE_BINDING.0;
            first = false;
        }
        if remaining & Self::PROTECTED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PROTECTED")?;
            remaining &= !Self::PROTECTED.0;
            first = false;
        }
        if remaining & Self::VIDEO_DECODE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VIDEO_DECODE")?;
            remaining &= !Self::VIDEO_DECODE.0;
            first = false;
        }
        if remaining & Self::VIDEO_ENCODE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VIDEO_ENCODE")?;
            remaining &= !Self::VIDEO_ENCODE.0;
            first = false;
        }
        if remaining & Self::OPTICAL_FLOW.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("OPTICAL_FLOW")?;
            remaining &= !Self::OPTICAL_FLOW.0;
            first = false;
        }
        if remaining & Self::DATA_GRAPH_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DATA_GRAPH_BIT")?;
            remaining &= !Self::DATA_GRAPH_BIT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkRefreshObjectFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkRefreshObjectFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkRefreshObjectFlagBitsKHR")]
pub struct RefreshObjectFlagBitsKHR(u32);
impl RefreshObjectFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}
impl core::ops::BitOr for RefreshObjectFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for RefreshObjectFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for RefreshObjectFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for RefreshObjectFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for RefreshObjectFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for RefreshObjectFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for RefreshObjectFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for RefreshObjectFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkRenderPassCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkRenderPassCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkRenderPassCreateFlagBits")]
pub struct RenderPassCreateFlagBits(u32);
impl RenderPassCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 1.
    pub const TRANSFORM_BIT: Self = Self(2u32);
    ///Bit 2.
    pub const PER_LAYER_FRAGMENT_DENSITY_BIT: Self = Self(4u32);
}
impl core::ops::BitOr for RenderPassCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for RenderPassCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for RenderPassCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for RenderPassCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for RenderPassCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for RenderPassCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for RenderPassCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for RenderPassCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::TRANSFORM_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFORM_BIT")?;
            remaining &= !Self::TRANSFORM_BIT.0;
            first = false;
        }
        if remaining & Self::PER_LAYER_FRAGMENT_DENSITY_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PER_LAYER_FRAGMENT_DENSITY_BIT")?;
            remaining &= !Self::PER_LAYER_FRAGMENT_DENSITY_BIT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkRenderingAttachmentFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkRenderingAttachmentFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkRenderingAttachmentFlagBitsKHR")]
pub struct RenderingAttachmentFlagBitsKHR(u32);
impl RenderingAttachmentFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const INPUT_ATTACHMENT_FEEDBACK: Self = Self(1u32);
    ///Bit 1.
    pub const RESOLVE_SKIP_TRANSFER_FUNCTION: Self = Self(2u32);
    ///Bit 2.
    pub const RESOLVE_ENABLE_TRANSFER_FUNCTION: Self = Self(4u32);
}
impl core::ops::BitOr for RenderingAttachmentFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for RenderingAttachmentFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for RenderingAttachmentFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for RenderingAttachmentFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for RenderingAttachmentFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for RenderingAttachmentFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for RenderingAttachmentFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for RenderingAttachmentFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::INPUT_ATTACHMENT_FEEDBACK.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INPUT_ATTACHMENT_FEEDBACK")?;
            remaining &= !Self::INPUT_ATTACHMENT_FEEDBACK.0;
            first = false;
        }
        if remaining & Self::RESOLVE_SKIP_TRANSFER_FUNCTION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RESOLVE_SKIP_TRANSFER_FUNCTION")?;
            remaining &= !Self::RESOLVE_SKIP_TRANSFER_FUNCTION.0;
            first = false;
        }
        if remaining & Self::RESOLVE_ENABLE_TRANSFER_FUNCTION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RESOLVE_ENABLE_TRANSFER_FUNCTION")?;
            remaining &= !Self::RESOLVE_ENABLE_TRANSFER_FUNCTION.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkRenderingFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkRenderingFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkRenderingFlagBits")]
pub struct RenderingFlagBits(u32);
impl RenderingFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const CONTENTS_SECONDARY_COMMAND_BUFFERS: Self = Self(1u32);
    ///Bit 1.
    pub const SUSPENDING: Self = Self(2u32);
    ///Bit 2.
    pub const RESUMING: Self = Self(4u32);
    ///Bit 3.
    pub const ENABLE_LEGACY_DITHERING: Self = Self(8u32);
    ///Bit 5.
    pub const PER_LAYER_FRAGMENT_DENSITY_BIT: Self = Self(32u32);
    ///Bit 6.
    pub const FRAGMENT_REGION: Self = Self(64u32);
    ///Bit 7.
    pub const CUSTOM_RESOLVE: Self = Self(128u32);
    ///Bit 8.
    pub const LOCAL_READ_CONCURRENT_ACCESS_CONTROL: Self = Self(256u32);
}
impl core::ops::BitOr for RenderingFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for RenderingFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for RenderingFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for RenderingFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for RenderingFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for RenderingFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for RenderingFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for RenderingFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::CONTENTS_SECONDARY_COMMAND_BUFFERS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CONTENTS_SECONDARY_COMMAND_BUFFERS")?;
            remaining &= !Self::CONTENTS_SECONDARY_COMMAND_BUFFERS.0;
            first = false;
        }
        if remaining & Self::SUSPENDING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SUSPENDING")?;
            remaining &= !Self::SUSPENDING.0;
            first = false;
        }
        if remaining & Self::RESUMING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RESUMING")?;
            remaining &= !Self::RESUMING.0;
            first = false;
        }
        if remaining & Self::ENABLE_LEGACY_DITHERING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ENABLE_LEGACY_DITHERING")?;
            remaining &= !Self::ENABLE_LEGACY_DITHERING.0;
            first = false;
        }
        if remaining & Self::PER_LAYER_FRAGMENT_DENSITY_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PER_LAYER_FRAGMENT_DENSITY_BIT")?;
            remaining &= !Self::PER_LAYER_FRAGMENT_DENSITY_BIT.0;
            first = false;
        }
        if remaining & Self::FRAGMENT_REGION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRAGMENT_REGION")?;
            remaining &= !Self::FRAGMENT_REGION.0;
            first = false;
        }
        if remaining & Self::CUSTOM_RESOLVE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CUSTOM_RESOLVE")?;
            remaining &= !Self::CUSTOM_RESOLVE.0;
            first = false;
        }
        if remaining & Self::LOCAL_READ_CONCURRENT_ACCESS_CONTROL.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("LOCAL_READ_CONCURRENT_ACCESS_CONTROL")?;
            remaining &= !Self::LOCAL_READ_CONCURRENT_ACCESS_CONTROL.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkResolveImageFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkResolveImageFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkResolveImageFlagBitsKHR")]
pub struct ResolveImageFlagBitsKHR(u32);
impl ResolveImageFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const SKIP_TRANSFER_FUNCTION: Self = Self(1u32);
    ///Bit 1.
    pub const ENABLE_TRANSFER_FUNCTION: Self = Self(2u32);
}
impl core::ops::BitOr for ResolveImageFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ResolveImageFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ResolveImageFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ResolveImageFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ResolveImageFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ResolveImageFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ResolveImageFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ResolveImageFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::SKIP_TRANSFER_FUNCTION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SKIP_TRANSFER_FUNCTION")?;
            remaining &= !Self::SKIP_TRANSFER_FUNCTION.0;
            first = false;
        }
        if remaining & Self::ENABLE_TRANSFER_FUNCTION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ENABLE_TRANSFER_FUNCTION")?;
            remaining &= !Self::ENABLE_TRANSFER_FUNCTION.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkResolveModeFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkResolveModeFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkResolveModeFlagBits")]
pub struct ResolveModeFlagBits(u32);
impl ResolveModeFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const NONE: Self = Self(0u32);
    ///Bit 0.
    pub const SAMPLE_ZERO: Self = Self(1u32);
    ///Bit 1.
    pub const AVERAGE: Self = Self(2u32);
    ///Bit 2.
    pub const MIN: Self = Self(4u32);
    ///Bit 3.
    pub const MAX: Self = Self(8u32);
    ///Bit 4.
    pub const EXTERNAL_FORMAT_DOWNSAMPLE_BIT: Self = Self(16u32);
    pub const EXTERNAL_FORMAT_DOWNSAMPLE: Self = Self::EXTERNAL_FORMAT_DOWNSAMPLE_BIT;
    ///Bit 5.
    pub const CUSTOM: Self = Self(32u32);
}
impl core::ops::BitOr for ResolveModeFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ResolveModeFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ResolveModeFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ResolveModeFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ResolveModeFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ResolveModeFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ResolveModeFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ResolveModeFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::SAMPLE_ZERO.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SAMPLE_ZERO")?;
            remaining &= !Self::SAMPLE_ZERO.0;
            first = false;
        }
        if remaining & Self::AVERAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("AVERAGE")?;
            remaining &= !Self::AVERAGE.0;
            first = false;
        }
        if remaining & Self::MIN.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MIN")?;
            remaining &= !Self::MIN.0;
            first = false;
        }
        if remaining & Self::MAX.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MAX")?;
            remaining &= !Self::MAX.0;
            first = false;
        }
        if remaining & Self::EXTERNAL_FORMAT_DOWNSAMPLE_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("EXTERNAL_FORMAT_DOWNSAMPLE_BIT")?;
            remaining &= !Self::EXTERNAL_FORMAT_DOWNSAMPLE_BIT.0;
            first = false;
        }
        if remaining & Self::CUSTOM.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CUSTOM")?;
            remaining &= !Self::CUSTOM.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkSampleCountFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkSampleCountFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkSampleCountFlagBits")]
pub struct SampleCountFlagBits(u32);
impl SampleCountFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const _1: Self = Self(1u32);
    ///Bit 1.
    pub const _2: Self = Self(2u32);
    ///Bit 2.
    pub const _4: Self = Self(4u32);
    ///Bit 3.
    pub const _8: Self = Self(8u32);
    ///Bit 4.
    pub const _16: Self = Self(16u32);
    ///Bit 5.
    pub const _32: Self = Self(32u32);
    ///Bit 6.
    pub const _64: Self = Self(64u32);
}
impl core::ops::BitOr for SampleCountFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for SampleCountFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for SampleCountFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for SampleCountFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for SampleCountFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for SampleCountFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for SampleCountFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for SampleCountFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::_1.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_1")?;
            remaining &= !Self::_1.0;
            first = false;
        }
        if remaining & Self::_2.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_2")?;
            remaining &= !Self::_2.0;
            first = false;
        }
        if remaining & Self::_4.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_4")?;
            remaining &= !Self::_4.0;
            first = false;
        }
        if remaining & Self::_8.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_8")?;
            remaining &= !Self::_8.0;
            first = false;
        }
        if remaining & Self::_16.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_16")?;
            remaining &= !Self::_16.0;
            first = false;
        }
        if remaining & Self::_32.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_32")?;
            remaining &= !Self::_32.0;
            first = false;
        }
        if remaining & Self::_64.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_64")?;
            remaining &= !Self::_64.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkSamplerCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkSamplerCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkSamplerCreateFlagBits")]
pub struct SamplerCreateFlagBits(u32);
impl SamplerCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const SUBSAMPLED: Self = Self(1u32);
    ///Bit 1.
    pub const SUBSAMPLED_COARSE_RECONSTRUCTION: Self = Self(2u32);
    ///Bit 3.
    pub const DESCRIPTOR_BUFFER_CAPTURE_REPLAY: Self = Self(8u32);
    ///Bit 2.
    pub const NON_SEAMLESS_CUBE_MAP: Self = Self(4u32);
    ///Bit 4.
    pub const IMAGE_PROCESSING_BIT: Self = Self(16u32);
}
impl core::ops::BitOr for SamplerCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for SamplerCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for SamplerCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for SamplerCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for SamplerCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for SamplerCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for SamplerCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for SamplerCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::SUBSAMPLED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SUBSAMPLED")?;
            remaining &= !Self::SUBSAMPLED.0;
            first = false;
        }
        if remaining & Self::SUBSAMPLED_COARSE_RECONSTRUCTION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SUBSAMPLED_COARSE_RECONSTRUCTION")?;
            remaining &= !Self::SUBSAMPLED_COARSE_RECONSTRUCTION.0;
            first = false;
        }
        if remaining & Self::DESCRIPTOR_BUFFER_CAPTURE_REPLAY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DESCRIPTOR_BUFFER_CAPTURE_REPLAY")?;
            remaining &= !Self::DESCRIPTOR_BUFFER_CAPTURE_REPLAY.0;
            first = false;
        }
        if remaining & Self::NON_SEAMLESS_CUBE_MAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("NON_SEAMLESS_CUBE_MAP")?;
            remaining &= !Self::NON_SEAMLESS_CUBE_MAP.0;
            first = false;
        }
        if remaining & Self::IMAGE_PROCESSING_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("IMAGE_PROCESSING_BIT")?;
            remaining &= !Self::IMAGE_PROCESSING_BIT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkSemaphoreCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkSemaphoreCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkSemaphoreCreateFlagBits")]
pub struct SemaphoreCreateFlagBits(u32);
impl SemaphoreCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}
impl core::ops::BitOr for SemaphoreCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for SemaphoreCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for SemaphoreCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for SemaphoreCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for SemaphoreCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for SemaphoreCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for SemaphoreCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for SemaphoreCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkSemaphoreImportFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkSemaphoreImportFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkSemaphoreImportFlagBits")]
pub struct SemaphoreImportFlagBits(u32);
impl SemaphoreImportFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const TEMPORARY: Self = Self(1u32);
}
impl core::ops::BitOr for SemaphoreImportFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for SemaphoreImportFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for SemaphoreImportFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for SemaphoreImportFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for SemaphoreImportFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for SemaphoreImportFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for SemaphoreImportFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for SemaphoreImportFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::TEMPORARY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TEMPORARY")?;
            remaining &= !Self::TEMPORARY.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkSemaphoreWaitFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkSemaphoreWaitFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkSemaphoreWaitFlagBits")]
pub struct SemaphoreWaitFlagBits(u32);
impl SemaphoreWaitFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const ANY: Self = Self(1u32);
}
impl core::ops::BitOr for SemaphoreWaitFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for SemaphoreWaitFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for SemaphoreWaitFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for SemaphoreWaitFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for SemaphoreWaitFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for SemaphoreWaitFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for SemaphoreWaitFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for SemaphoreWaitFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::ANY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ANY")?;
            remaining &= !Self::ANY.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkShaderCorePropertiesFlagBitsAMD`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkShaderCorePropertiesFlagBitsAMD.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkShaderCorePropertiesFlagBitsAMD")]
pub struct ShaderCorePropertiesFlagBitsAMD(u32);
impl ShaderCorePropertiesFlagBitsAMD {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}
impl core::ops::BitOr for ShaderCorePropertiesFlagBitsAMD {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ShaderCorePropertiesFlagBitsAMD {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ShaderCorePropertiesFlagBitsAMD {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ShaderCorePropertiesFlagBitsAMD {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ShaderCorePropertiesFlagBitsAMD {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ShaderCorePropertiesFlagBitsAMD {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ShaderCorePropertiesFlagBitsAMD {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ShaderCorePropertiesFlagBitsAMD {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkShaderCreateFlagBitsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkShaderCreateFlagBitsEXT.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkShaderCreateFlagBitsEXT")]
pub struct ShaderCreateFlagBitsEXT(u32);
impl ShaderCreateFlagBitsEXT {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const LINK_STAGE: Self = Self(1u32);
    ///Bit 10.
    pub const DESCRIPTOR_HEAP: Self = Self(1024u32);
    ///Bit 11.
    pub const INSTRUMENT_SHADER_BIT: Self = Self(2048u32);
    ///Bit 1.
    pub const ALLOW_VARYING_SUBGROUP_SIZE: Self = Self(2u32);
    ///Bit 2.
    pub const REQUIRE_FULL_SUBGROUPS: Self = Self(4u32);
    ///Bit 3.
    pub const NO_TASK_SHADER: Self = Self(8u32);
    ///Bit 4.
    pub const DISPATCH_BASE: Self = Self(16u32);
    ///Bit 5.
    pub const FRAGMENT_SHADING_RATE_ATTACHMENT: Self = Self(32u32);
    ///Bit 6.
    pub const FRAGMENT_DENSITY_MAP_ATTACHMENT: Self = Self(64u32);
    ///Bit 7.
    pub const INDIRECT_BINDABLE: Self = Self(128u32);
    ///Bit 15.
    pub const _64_BIT_INDEXING: Self = Self(32768u32);
}
impl core::ops::BitOr for ShaderCreateFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ShaderCreateFlagBitsEXT {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ShaderCreateFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ShaderCreateFlagBitsEXT {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ShaderCreateFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ShaderCreateFlagBitsEXT {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ShaderCreateFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ShaderCreateFlagBitsEXT {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::LINK_STAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("LINK_STAGE")?;
            remaining &= !Self::LINK_STAGE.0;
            first = false;
        }
        if remaining & Self::DESCRIPTOR_HEAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DESCRIPTOR_HEAP")?;
            remaining &= !Self::DESCRIPTOR_HEAP.0;
            first = false;
        }
        if remaining & Self::INSTRUMENT_SHADER_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INSTRUMENT_SHADER_BIT")?;
            remaining &= !Self::INSTRUMENT_SHADER_BIT.0;
            first = false;
        }
        if remaining & Self::ALLOW_VARYING_SUBGROUP_SIZE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALLOW_VARYING_SUBGROUP_SIZE")?;
            remaining &= !Self::ALLOW_VARYING_SUBGROUP_SIZE.0;
            first = false;
        }
        if remaining & Self::REQUIRE_FULL_SUBGROUPS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("REQUIRE_FULL_SUBGROUPS")?;
            remaining &= !Self::REQUIRE_FULL_SUBGROUPS.0;
            first = false;
        }
        if remaining & Self::NO_TASK_SHADER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("NO_TASK_SHADER")?;
            remaining &= !Self::NO_TASK_SHADER.0;
            first = false;
        }
        if remaining & Self::DISPATCH_BASE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DISPATCH_BASE")?;
            remaining &= !Self::DISPATCH_BASE.0;
            first = false;
        }
        if remaining & Self::FRAGMENT_SHADING_RATE_ATTACHMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRAGMENT_SHADING_RATE_ATTACHMENT")?;
            remaining &= !Self::FRAGMENT_SHADING_RATE_ATTACHMENT.0;
            first = false;
        }
        if remaining & Self::FRAGMENT_DENSITY_MAP_ATTACHMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRAGMENT_DENSITY_MAP_ATTACHMENT")?;
            remaining &= !Self::FRAGMENT_DENSITY_MAP_ATTACHMENT.0;
            first = false;
        }
        if remaining & Self::INDIRECT_BINDABLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INDIRECT_BINDABLE")?;
            remaining &= !Self::INDIRECT_BINDABLE.0;
            first = false;
        }
        if remaining & Self::_64_BIT_INDEXING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_64_BIT_INDEXING")?;
            remaining &= !Self::_64_BIT_INDEXING.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkShaderModuleCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkShaderModuleCreateFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkShaderModuleCreateFlagBits")]
pub struct ShaderModuleCreateFlagBits(u32);
impl ShaderModuleCreateFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}
impl core::ops::BitOr for ShaderModuleCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ShaderModuleCreateFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ShaderModuleCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ShaderModuleCreateFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ShaderModuleCreateFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ShaderModuleCreateFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ShaderModuleCreateFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ShaderModuleCreateFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkShaderStageFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkShaderStageFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkShaderStageFlagBits")]
pub struct ShaderStageFlagBits(u32);
impl ShaderStageFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const VERTEX: Self = Self(1u32);
    ///Bit 1.
    pub const TESSELLATION_CONTROL: Self = Self(2u32);
    ///Bit 2.
    pub const TESSELLATION_EVALUATION: Self = Self(4u32);
    ///Bit 3.
    pub const GEOMETRY: Self = Self(8u32);
    ///Bit 4.
    pub const FRAGMENT: Self = Self(16u32);
    ///Bit 5.
    pub const COMPUTE: Self = Self(32u32);
    pub const ALL_GRAPHICS: Self = Self(31u32);
    pub const ALL: Self = Self(2147483647u32);
    ///Bit 8.
    pub const RAYGEN: Self = Self(256u32);
    ///Bit 9.
    pub const ANY_HIT: Self = Self(512u32);
    ///Bit 10.
    pub const CLOSEST_HIT: Self = Self(1024u32);
    ///Bit 11.
    pub const MISS: Self = Self(2048u32);
    ///Bit 12.
    pub const INTERSECTION: Self = Self(4096u32);
    ///Bit 13.
    pub const CALLABLE: Self = Self(8192u32);
    ///Bit 14.
    pub const SUBPASS_SHADING_BIT: Self = Self(16384u32);
    ///Bit 19.
    pub const CLUSTER_CULLING_BIT: Self = Self(524288u32);
}
impl core::ops::BitOr for ShaderStageFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ShaderStageFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ShaderStageFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ShaderStageFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ShaderStageFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ShaderStageFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ShaderStageFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ShaderStageFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::VERTEX.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VERTEX")?;
            remaining &= !Self::VERTEX.0;
            first = false;
        }
        if remaining & Self::TESSELLATION_CONTROL.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TESSELLATION_CONTROL")?;
            remaining &= !Self::TESSELLATION_CONTROL.0;
            first = false;
        }
        if remaining & Self::TESSELLATION_EVALUATION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TESSELLATION_EVALUATION")?;
            remaining &= !Self::TESSELLATION_EVALUATION.0;
            first = false;
        }
        if remaining & Self::GEOMETRY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("GEOMETRY")?;
            remaining &= !Self::GEOMETRY.0;
            first = false;
        }
        if remaining & Self::FRAGMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRAGMENT")?;
            remaining &= !Self::FRAGMENT.0;
            first = false;
        }
        if remaining & Self::COMPUTE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("COMPUTE")?;
            remaining &= !Self::COMPUTE.0;
            first = false;
        }
        if remaining & Self::RAYGEN.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RAYGEN")?;
            remaining &= !Self::RAYGEN.0;
            first = false;
        }
        if remaining & Self::ANY_HIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ANY_HIT")?;
            remaining &= !Self::ANY_HIT.0;
            first = false;
        }
        if remaining & Self::CLOSEST_HIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CLOSEST_HIT")?;
            remaining &= !Self::CLOSEST_HIT.0;
            first = false;
        }
        if remaining & Self::MISS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MISS")?;
            remaining &= !Self::MISS.0;
            first = false;
        }
        if remaining & Self::INTERSECTION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INTERSECTION")?;
            remaining &= !Self::INTERSECTION.0;
            first = false;
        }
        if remaining & Self::CALLABLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CALLABLE")?;
            remaining &= !Self::CALLABLE.0;
            first = false;
        }
        if remaining & Self::SUBPASS_SHADING_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SUBPASS_SHADING_BIT")?;
            remaining &= !Self::SUBPASS_SHADING_BIT.0;
            first = false;
        }
        if remaining & Self::CLUSTER_CULLING_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CLUSTER_CULLING_BIT")?;
            remaining &= !Self::CLUSTER_CULLING_BIT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkSparseImageFormatFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkSparseImageFormatFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkSparseImageFormatFlagBits")]
pub struct SparseImageFormatFlagBits(u32);
impl SparseImageFormatFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const SINGLE_MIPTAIL: Self = Self(1u32);
    ///Bit 1.
    pub const ALIGNED_MIP_SIZE: Self = Self(2u32);
    ///Bit 2.
    pub const NONSTANDARD_BLOCK_SIZE: Self = Self(4u32);
}
impl core::ops::BitOr for SparseImageFormatFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for SparseImageFormatFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for SparseImageFormatFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for SparseImageFormatFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for SparseImageFormatFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for SparseImageFormatFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for SparseImageFormatFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for SparseImageFormatFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::SINGLE_MIPTAIL.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SINGLE_MIPTAIL")?;
            remaining &= !Self::SINGLE_MIPTAIL.0;
            first = false;
        }
        if remaining & Self::ALIGNED_MIP_SIZE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALIGNED_MIP_SIZE")?;
            remaining &= !Self::ALIGNED_MIP_SIZE.0;
            first = false;
        }
        if remaining & Self::NONSTANDARD_BLOCK_SIZE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("NONSTANDARD_BLOCK_SIZE")?;
            remaining &= !Self::NONSTANDARD_BLOCK_SIZE.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkSparseMemoryBindFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkSparseMemoryBindFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkSparseMemoryBindFlagBits")]
pub struct SparseMemoryBindFlagBits(u32);
impl SparseMemoryBindFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const METADATA: Self = Self(1u32);
}
impl core::ops::BitOr for SparseMemoryBindFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for SparseMemoryBindFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for SparseMemoryBindFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for SparseMemoryBindFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for SparseMemoryBindFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for SparseMemoryBindFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for SparseMemoryBindFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for SparseMemoryBindFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::METADATA.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("METADATA")?;
            remaining &= !Self::METADATA.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkSpirvResourceTypeFlagBitsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkSpirvResourceTypeFlagBitsEXT.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkSpirvResourceTypeFlagBitsEXT")]
pub struct SpirvResourceTypeFlagBitsEXT(u32);
impl SpirvResourceTypeFlagBitsEXT {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const ALL: Self = Self(2147483647u32);
    ///Bit 0.
    pub const SAMPLER: Self = Self(1u32);
    ///Bit 1.
    pub const SAMPLED_IMAGE: Self = Self(2u32);
    ///Bit 2.
    pub const READ_ONLY_IMAGE: Self = Self(4u32);
    ///Bit 3.
    pub const READ_WRITE_IMAGE: Self = Self(8u32);
    ///Bit 4.
    pub const COMBINED_SAMPLED_IMAGE: Self = Self(16u32);
    ///Bit 5.
    pub const UNIFORM_BUFFER: Self = Self(32u32);
    ///Bit 6.
    pub const READ_ONLY_STORAGE_BUFFER: Self = Self(64u32);
    ///Bit 7.
    pub const READ_WRITE_STORAGE_BUFFER: Self = Self(128u32);
    ///Bit 8.
    pub const ACCELERATION_STRUCTURE: Self = Self(256u32);
    ///Bit 9.
    pub const TENSOR_BIT: Self = Self(512u32);
}
impl core::ops::BitOr for SpirvResourceTypeFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for SpirvResourceTypeFlagBitsEXT {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for SpirvResourceTypeFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for SpirvResourceTypeFlagBitsEXT {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for SpirvResourceTypeFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for SpirvResourceTypeFlagBitsEXT {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for SpirvResourceTypeFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for SpirvResourceTypeFlagBitsEXT {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::SAMPLER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SAMPLER")?;
            remaining &= !Self::SAMPLER.0;
            first = false;
        }
        if remaining & Self::SAMPLED_IMAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SAMPLED_IMAGE")?;
            remaining &= !Self::SAMPLED_IMAGE.0;
            first = false;
        }
        if remaining & Self::READ_ONLY_IMAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("READ_ONLY_IMAGE")?;
            remaining &= !Self::READ_ONLY_IMAGE.0;
            first = false;
        }
        if remaining & Self::READ_WRITE_IMAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("READ_WRITE_IMAGE")?;
            remaining &= !Self::READ_WRITE_IMAGE.0;
            first = false;
        }
        if remaining & Self::COMBINED_SAMPLED_IMAGE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("COMBINED_SAMPLED_IMAGE")?;
            remaining &= !Self::COMBINED_SAMPLED_IMAGE.0;
            first = false;
        }
        if remaining & Self::UNIFORM_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("UNIFORM_BUFFER")?;
            remaining &= !Self::UNIFORM_BUFFER.0;
            first = false;
        }
        if remaining & Self::READ_ONLY_STORAGE_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("READ_ONLY_STORAGE_BUFFER")?;
            remaining &= !Self::READ_ONLY_STORAGE_BUFFER.0;
            first = false;
        }
        if remaining & Self::READ_WRITE_STORAGE_BUFFER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("READ_WRITE_STORAGE_BUFFER")?;
            remaining &= !Self::READ_WRITE_STORAGE_BUFFER.0;
            first = false;
        }
        if remaining & Self::ACCELERATION_STRUCTURE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ACCELERATION_STRUCTURE")?;
            remaining &= !Self::ACCELERATION_STRUCTURE.0;
            first = false;
        }
        if remaining & Self::TENSOR_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TENSOR_BIT")?;
            remaining &= !Self::TENSOR_BIT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkStencilFaceFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkStencilFaceFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkStencilFaceFlagBits")]
pub struct StencilFaceFlagBits(u32);
impl StencilFaceFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const FRONT: Self = Self(1u32);
    ///Bit 1.
    pub const BACK: Self = Self(2u32);
    pub const FRONT_AND_BACK: Self = Self(3u32);
    pub const STENCIL_FRONT_AND_BACK: Self = Self::FRONT_AND_BACK;
}
impl core::ops::BitOr for StencilFaceFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for StencilFaceFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for StencilFaceFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for StencilFaceFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for StencilFaceFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for StencilFaceFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for StencilFaceFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for StencilFaceFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::FRONT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRONT")?;
            remaining &= !Self::FRONT.0;
            first = false;
        }
        if remaining & Self::BACK.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("BACK")?;
            remaining &= !Self::BACK.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkSubgroupFeatureFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkSubgroupFeatureFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkSubgroupFeatureFlagBits")]
pub struct SubgroupFeatureFlagBits(u32);
impl SubgroupFeatureFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const BASIC: Self = Self(1u32);
    ///Bit 1.
    pub const VOTE: Self = Self(2u32);
    ///Bit 2.
    pub const ARITHMETIC: Self = Self(4u32);
    ///Bit 3.
    pub const BALLOT: Self = Self(8u32);
    ///Bit 4.
    pub const SHUFFLE: Self = Self(16u32);
    ///Bit 5.
    pub const SHUFFLE_RELATIVE: Self = Self(32u32);
    ///Bit 6.
    pub const CLUSTERED: Self = Self(64u32);
    ///Bit 7.
    pub const QUAD: Self = Self(128u32);
    ///Bit 9.
    pub const ROTATE: Self = Self(512u32);
    ///Bit 10.
    pub const ROTATE_CLUSTERED: Self = Self(1024u32);
}
impl core::ops::BitOr for SubgroupFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for SubgroupFeatureFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for SubgroupFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for SubgroupFeatureFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for SubgroupFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for SubgroupFeatureFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for SubgroupFeatureFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for SubgroupFeatureFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::BASIC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("BASIC")?;
            remaining &= !Self::BASIC.0;
            first = false;
        }
        if remaining & Self::VOTE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VOTE")?;
            remaining &= !Self::VOTE.0;
            first = false;
        }
        if remaining & Self::ARITHMETIC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ARITHMETIC")?;
            remaining &= !Self::ARITHMETIC.0;
            first = false;
        }
        if remaining & Self::BALLOT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("BALLOT")?;
            remaining &= !Self::BALLOT.0;
            first = false;
        }
        if remaining & Self::SHUFFLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SHUFFLE")?;
            remaining &= !Self::SHUFFLE.0;
            first = false;
        }
        if remaining & Self::SHUFFLE_RELATIVE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SHUFFLE_RELATIVE")?;
            remaining &= !Self::SHUFFLE_RELATIVE.0;
            first = false;
        }
        if remaining & Self::CLUSTERED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CLUSTERED")?;
            remaining &= !Self::CLUSTERED.0;
            first = false;
        }
        if remaining & Self::QUAD.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("QUAD")?;
            remaining &= !Self::QUAD.0;
            first = false;
        }
        if remaining & Self::ROTATE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ROTATE")?;
            remaining &= !Self::ROTATE.0;
            first = false;
        }
        if remaining & Self::ROTATE_CLUSTERED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ROTATE_CLUSTERED")?;
            remaining &= !Self::ROTATE_CLUSTERED.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkSubmitFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkSubmitFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkSubmitFlagBits")]
pub struct SubmitFlagBits(u32);
impl SubmitFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const PROTECTED: Self = Self(1u32);
}
impl core::ops::BitOr for SubmitFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for SubmitFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for SubmitFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for SubmitFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for SubmitFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for SubmitFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for SubmitFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for SubmitFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::PROTECTED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PROTECTED")?;
            remaining &= !Self::PROTECTED.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkSubpassDescriptionFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkSubpassDescriptionFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkSubpassDescriptionFlagBits")]
pub struct SubpassDescriptionFlagBits(u32);
impl SubpassDescriptionFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const PER_VIEW_ATTRIBUTES_BIT: Self = Self(1u32);
    ///Bit 1.
    pub const PER_VIEW_POSITION_X_ONLY_BIT: Self = Self(2u32);
    pub const FRAGMENT_REGION_BIT: Self = Self::FRAGMENT_REGION;
    pub const SHADER_RESOLVE_BIT: Self = Self::CUSTOM_RESOLVE;
    ///Bit 8.
    pub const TILE_SHADING_APRON_BIT: Self = Self(256u32);
    pub const RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS_BIT: Self = Self::RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS;
    pub const RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_BIT: Self = Self::RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS;
    pub const RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_BIT: Self = Self::RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS;
    ///Bit 4.
    pub const RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS: Self = Self(16u32);
    ///Bit 5.
    pub const RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS: Self = Self(32u32);
    ///Bit 6.
    pub const RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS: Self = Self(64u32);
    ///Bit 7.
    pub const ENABLE_LEGACY_DITHERING: Self = Self(128u32);
    ///Bit 2.
    pub const FRAGMENT_REGION: Self = Self(4u32);
    ///Bit 3.
    pub const CUSTOM_RESOLVE: Self = Self(8u32);
}
impl core::ops::BitOr for SubpassDescriptionFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for SubpassDescriptionFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for SubpassDescriptionFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for SubpassDescriptionFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for SubpassDescriptionFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for SubpassDescriptionFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for SubpassDescriptionFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for SubpassDescriptionFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::PER_VIEW_ATTRIBUTES_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PER_VIEW_ATTRIBUTES_BIT")?;
            remaining &= !Self::PER_VIEW_ATTRIBUTES_BIT.0;
            first = false;
        }
        if remaining & Self::PER_VIEW_POSITION_X_ONLY_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PER_VIEW_POSITION_X_ONLY_BIT")?;
            remaining &= !Self::PER_VIEW_POSITION_X_ONLY_BIT.0;
            first = false;
        }
        if remaining & Self::TILE_SHADING_APRON_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TILE_SHADING_APRON_BIT")?;
            remaining &= !Self::TILE_SHADING_APRON_BIT.0;
            first = false;
        }
        if remaining & Self::RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS")?;
            remaining &= !Self::RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS.0;
            first = false;
        }
        if remaining & Self::RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS")?;
            remaining &= !Self::RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS.0;
            first = false;
        }
        if remaining & Self::RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS")?;
            remaining &= !Self::RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS.0;
            first = false;
        }
        if remaining & Self::ENABLE_LEGACY_DITHERING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ENABLE_LEGACY_DITHERING")?;
            remaining &= !Self::ENABLE_LEGACY_DITHERING.0;
            first = false;
        }
        if remaining & Self::FRAGMENT_REGION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRAGMENT_REGION")?;
            remaining &= !Self::FRAGMENT_REGION.0;
            first = false;
        }
        if remaining & Self::CUSTOM_RESOLVE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CUSTOM_RESOLVE")?;
            remaining &= !Self::CUSTOM_RESOLVE.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkSurfaceCounterFlagBitsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkSurfaceCounterFlagBitsEXT.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkSurfaceCounterFlagBitsEXT")]
pub struct SurfaceCounterFlagBitsEXT(u32);
impl SurfaceCounterFlagBitsEXT {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const VBLANK: Self = Self(1u32);
}
impl core::ops::BitOr for SurfaceCounterFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for SurfaceCounterFlagBitsEXT {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for SurfaceCounterFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for SurfaceCounterFlagBitsEXT {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for SurfaceCounterFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for SurfaceCounterFlagBitsEXT {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for SurfaceCounterFlagBitsEXT {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for SurfaceCounterFlagBitsEXT {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::VBLANK.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VBLANK")?;
            remaining &= !Self::VBLANK.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkSurfaceTransformFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkSurfaceTransformFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkSurfaceTransformFlagBitsKHR")]
pub struct SurfaceTransformFlagBitsKHR(u32);
impl SurfaceTransformFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const IDENTITY: Self = Self(1u32);
    ///Bit 1.
    pub const ROTATE_90: Self = Self(2u32);
    ///Bit 2.
    pub const ROTATE_180: Self = Self(4u32);
    ///Bit 3.
    pub const ROTATE_270: Self = Self(8u32);
    ///Bit 4.
    pub const HORIZONTAL_MIRROR: Self = Self(16u32);
    ///Bit 5.
    pub const HORIZONTAL_MIRROR_ROTATE_90: Self = Self(32u32);
    ///Bit 6.
    pub const HORIZONTAL_MIRROR_ROTATE_180: Self = Self(64u32);
    ///Bit 7.
    pub const HORIZONTAL_MIRROR_ROTATE_270: Self = Self(128u32);
    ///Bit 8.
    pub const INHERIT: Self = Self(256u32);
}
impl core::ops::BitOr for SurfaceTransformFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for SurfaceTransformFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for SurfaceTransformFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for SurfaceTransformFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for SurfaceTransformFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for SurfaceTransformFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for SurfaceTransformFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for SurfaceTransformFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::IDENTITY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("IDENTITY")?;
            remaining &= !Self::IDENTITY.0;
            first = false;
        }
        if remaining & Self::ROTATE_90.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ROTATE_90")?;
            remaining &= !Self::ROTATE_90.0;
            first = false;
        }
        if remaining & Self::ROTATE_180.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ROTATE_180")?;
            remaining &= !Self::ROTATE_180.0;
            first = false;
        }
        if remaining & Self::ROTATE_270.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ROTATE_270")?;
            remaining &= !Self::ROTATE_270.0;
            first = false;
        }
        if remaining & Self::HORIZONTAL_MIRROR.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("HORIZONTAL_MIRROR")?;
            remaining &= !Self::HORIZONTAL_MIRROR.0;
            first = false;
        }
        if remaining & Self::HORIZONTAL_MIRROR_ROTATE_90.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("HORIZONTAL_MIRROR_ROTATE_90")?;
            remaining &= !Self::HORIZONTAL_MIRROR_ROTATE_90.0;
            first = false;
        }
        if remaining & Self::HORIZONTAL_MIRROR_ROTATE_180.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("HORIZONTAL_MIRROR_ROTATE_180")?;
            remaining &= !Self::HORIZONTAL_MIRROR_ROTATE_180.0;
            first = false;
        }
        if remaining & Self::HORIZONTAL_MIRROR_ROTATE_270.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("HORIZONTAL_MIRROR_ROTATE_270")?;
            remaining &= !Self::HORIZONTAL_MIRROR_ROTATE_270.0;
            first = false;
        }
        if remaining & Self::INHERIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INHERIT")?;
            remaining &= !Self::INHERIT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkSwapchainCreateFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkSwapchainCreateFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkSwapchainCreateFlagBitsKHR")]
pub struct SwapchainCreateFlagBitsKHR(u32);
impl SwapchainCreateFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const SPLIT_INSTANCE_BIND_REGIONS: Self = Self(1u32);
    ///Bit 1.
    pub const PROTECTED: Self = Self(2u32);
    ///Bit 2.
    pub const MUTABLE_FORMAT: Self = Self(4u32);
    ///Bit 9.
    pub const PRESENT_TIMING: Self = Self(512u32);
    ///Bit 6.
    pub const PRESENT_ID_2: Self = Self(64u32);
    ///Bit 7.
    pub const PRESENT_WAIT_2: Self = Self(128u32);
}
impl core::ops::BitOr for SwapchainCreateFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for SwapchainCreateFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for SwapchainCreateFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for SwapchainCreateFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for SwapchainCreateFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for SwapchainCreateFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for SwapchainCreateFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for SwapchainCreateFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::SPLIT_INSTANCE_BIND_REGIONS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SPLIT_INSTANCE_BIND_REGIONS")?;
            remaining &= !Self::SPLIT_INSTANCE_BIND_REGIONS.0;
            first = false;
        }
        if remaining & Self::PROTECTED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PROTECTED")?;
            remaining &= !Self::PROTECTED.0;
            first = false;
        }
        if remaining & Self::MUTABLE_FORMAT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MUTABLE_FORMAT")?;
            remaining &= !Self::MUTABLE_FORMAT.0;
            first = false;
        }
        if remaining & Self::PRESENT_TIMING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PRESENT_TIMING")?;
            remaining &= !Self::PRESENT_TIMING.0;
            first = false;
        }
        if remaining & Self::PRESENT_ID_2.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PRESENT_ID_2")?;
            remaining &= !Self::PRESENT_ID_2.0;
            first = false;
        }
        if remaining & Self::PRESENT_WAIT_2.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PRESENT_WAIT_2")?;
            remaining &= !Self::PRESENT_WAIT_2.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkSwapchainImageUsageFlagBitsANDROID`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkSwapchainImageUsageFlagBitsANDROID.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkSwapchainImageUsageFlagBitsANDROID")]
pub struct SwapchainImageUsageFlagBitsANDROID(u32);
impl SwapchainImageUsageFlagBitsANDROID {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const SHARED_BIT: Self = Self(1u32);
}
impl core::ops::BitOr for SwapchainImageUsageFlagBitsANDROID {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for SwapchainImageUsageFlagBitsANDROID {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for SwapchainImageUsageFlagBitsANDROID {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for SwapchainImageUsageFlagBitsANDROID {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for SwapchainImageUsageFlagBitsANDROID {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for SwapchainImageUsageFlagBitsANDROID {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for SwapchainImageUsageFlagBitsANDROID {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for SwapchainImageUsageFlagBitsANDROID {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::SHARED_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SHARED_BIT")?;
            remaining &= !Self::SHARED_BIT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkSwapchainImageUsageFlagBitsOHOS`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkSwapchainImageUsageFlagBitsOHOS.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkSwapchainImageUsageFlagBitsOHOS")]
pub struct SwapchainImageUsageFlagBitsOHOS(u32);
impl SwapchainImageUsageFlagBitsOHOS {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const SHARED_BIT: Self = Self(1u32);
}
impl core::ops::BitOr for SwapchainImageUsageFlagBitsOHOS {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for SwapchainImageUsageFlagBitsOHOS {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for SwapchainImageUsageFlagBitsOHOS {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for SwapchainImageUsageFlagBitsOHOS {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for SwapchainImageUsageFlagBitsOHOS {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for SwapchainImageUsageFlagBitsOHOS {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for SwapchainImageUsageFlagBitsOHOS {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for SwapchainImageUsageFlagBitsOHOS {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::SHARED_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SHARED_BIT")?;
            remaining &= !Self::SHARED_BIT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkTensorCreateFlagBitsARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkTensorCreateFlagBitsARM.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkTensorCreateFlagBitsARM")]
pub struct TensorCreateFlagBitsARM(u64);
impl TensorCreateFlagBitsARM {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u64)
    }
    #[inline]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u64
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const MUTABLE_FORMAT_BIT: Self = Self(1u64);
    ///Bit 1.
    pub const PROTECTED_BIT: Self = Self(2u64);
    ///Bit 3.
    pub const DESCRIPTOR_HEAP_CAPTURE_REPLAY_BIT: Self = Self(8u64);
    ///Bit 2.
    pub const DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT: Self = Self(4u64);
}
impl core::ops::BitOr for TensorCreateFlagBitsARM {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for TensorCreateFlagBitsARM {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for TensorCreateFlagBitsARM {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for TensorCreateFlagBitsARM {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for TensorCreateFlagBitsARM {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for TensorCreateFlagBitsARM {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for TensorCreateFlagBitsARM {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for TensorCreateFlagBitsARM {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::MUTABLE_FORMAT_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MUTABLE_FORMAT_BIT")?;
            remaining &= !Self::MUTABLE_FORMAT_BIT.0;
            first = false;
        }
        if remaining & Self::PROTECTED_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PROTECTED_BIT")?;
            remaining &= !Self::PROTECTED_BIT.0;
            first = false;
        }
        if remaining & Self::DESCRIPTOR_HEAP_CAPTURE_REPLAY_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DESCRIPTOR_HEAP_CAPTURE_REPLAY_BIT")?;
            remaining &= !Self::DESCRIPTOR_HEAP_CAPTURE_REPLAY_BIT.0;
            first = false;
        }
        if remaining & Self::DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT")?;
            remaining &= !Self::DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT.0;
            first = false;
        }
        if remaining != 0u64 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkTensorUsageFlagBitsARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkTensorUsageFlagBitsARM.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkTensorUsageFlagBitsARM")]
pub struct TensorUsageFlagBitsARM(u64);
impl TensorUsageFlagBitsARM {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u64)
    }
    #[inline]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u64
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 1.
    pub const SHADER_BIT: Self = Self(2u64);
    ///Bit 2.
    pub const TRANSFER_SRC_BIT: Self = Self(4u64);
    ///Bit 3.
    pub const TRANSFER_DST_BIT: Self = Self(8u64);
    ///Bit 4.
    pub const IMAGE_ALIASING_BIT: Self = Self(16u64);
    ///Bit 5.
    pub const DATA_GRAPH_BIT: Self = Self(32u64);
}
impl core::ops::BitOr for TensorUsageFlagBitsARM {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for TensorUsageFlagBitsARM {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for TensorUsageFlagBitsARM {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for TensorUsageFlagBitsARM {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for TensorUsageFlagBitsARM {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for TensorUsageFlagBitsARM {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for TensorUsageFlagBitsARM {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for TensorUsageFlagBitsARM {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::SHADER_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SHADER_BIT")?;
            remaining &= !Self::SHADER_BIT.0;
            first = false;
        }
        if remaining & Self::TRANSFER_SRC_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFER_SRC_BIT")?;
            remaining &= !Self::TRANSFER_SRC_BIT.0;
            first = false;
        }
        if remaining & Self::TRANSFER_DST_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFER_DST_BIT")?;
            remaining &= !Self::TRANSFER_DST_BIT.0;
            first = false;
        }
        if remaining & Self::IMAGE_ALIASING_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("IMAGE_ALIASING_BIT")?;
            remaining &= !Self::IMAGE_ALIASING_BIT.0;
            first = false;
        }
        if remaining & Self::DATA_GRAPH_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DATA_GRAPH_BIT")?;
            remaining &= !Self::DATA_GRAPH_BIT.0;
            first = false;
        }
        if remaining != 0u64 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkTensorViewCreateFlagBitsARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkTensorViewCreateFlagBitsARM.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkTensorViewCreateFlagBitsARM")]
pub struct TensorViewCreateFlagBitsARM(u64);
impl TensorViewCreateFlagBitsARM {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u64)
    }
    #[inline]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u64
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT: Self = Self(1u64);
}
impl core::ops::BitOr for TensorViewCreateFlagBitsARM {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for TensorViewCreateFlagBitsARM {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for TensorViewCreateFlagBitsARM {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for TensorViewCreateFlagBitsARM {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for TensorViewCreateFlagBitsARM {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for TensorViewCreateFlagBitsARM {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for TensorViewCreateFlagBitsARM {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for TensorViewCreateFlagBitsARM {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT")?;
            remaining &= !Self::DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT.0;
            first = false;
        }
        if remaining != 0u64 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkTileShadingRenderPassFlagBitsQCOM`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkTileShadingRenderPassFlagBitsQCOM.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkTileShadingRenderPassFlagBitsQCOM")]
pub struct TileShadingRenderPassFlagBitsQCOM(u32);
impl TileShadingRenderPassFlagBitsQCOM {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const ENABLE_BIT: Self = Self(1u32);
    ///Bit 1.
    pub const PER_TILE_EXECUTION_BIT: Self = Self(2u32);
}
impl core::ops::BitOr for TileShadingRenderPassFlagBitsQCOM {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for TileShadingRenderPassFlagBitsQCOM {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for TileShadingRenderPassFlagBitsQCOM {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for TileShadingRenderPassFlagBitsQCOM {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for TileShadingRenderPassFlagBitsQCOM {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for TileShadingRenderPassFlagBitsQCOM {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for TileShadingRenderPassFlagBitsQCOM {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for TileShadingRenderPassFlagBitsQCOM {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::ENABLE_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ENABLE_BIT")?;
            remaining &= !Self::ENABLE_BIT.0;
            first = false;
        }
        if remaining & Self::PER_TILE_EXECUTION_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PER_TILE_EXECUTION_BIT")?;
            remaining &= !Self::PER_TILE_EXECUTION_BIT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkToolPurposeFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkToolPurposeFlagBits.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkToolPurposeFlagBits")]
pub struct ToolPurposeFlagBits(u32);
impl ToolPurposeFlagBits {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const VALIDATION: Self = Self(1u32);
    ///Bit 1.
    pub const PROFILING: Self = Self(2u32);
    ///Bit 2.
    pub const TRACING: Self = Self(4u32);
    ///Bit 3.
    pub const ADDITIONAL_FEATURES: Self = Self(8u32);
    ///Bit 4.
    pub const MODIFYING_FEATURES: Self = Self(16u32);
    ///Bit 5.
    pub const DEBUG_REPORTING: Self = Self(32u32);
    ///Bit 6.
    pub const DEBUG_MARKERS: Self = Self(64u32);
}
impl core::ops::BitOr for ToolPurposeFlagBits {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for ToolPurposeFlagBits {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for ToolPurposeFlagBits {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for ToolPurposeFlagBits {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for ToolPurposeFlagBits {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for ToolPurposeFlagBits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for ToolPurposeFlagBits {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for ToolPurposeFlagBits {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::VALIDATION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VALIDATION")?;
            remaining &= !Self::VALIDATION.0;
            first = false;
        }
        if remaining & Self::PROFILING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PROFILING")?;
            remaining &= !Self::PROFILING.0;
            first = false;
        }
        if remaining & Self::TRACING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRACING")?;
            remaining &= !Self::TRACING.0;
            first = false;
        }
        if remaining & Self::ADDITIONAL_FEATURES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ADDITIONAL_FEATURES")?;
            remaining &= !Self::ADDITIONAL_FEATURES.0;
            first = false;
        }
        if remaining & Self::MODIFYING_FEATURES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MODIFYING_FEATURES")?;
            remaining &= !Self::MODIFYING_FEATURES.0;
            first = false;
        }
        if remaining & Self::DEBUG_REPORTING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEBUG_REPORTING")?;
            remaining &= !Self::DEBUG_REPORTING.0;
            first = false;
        }
        if remaining & Self::DEBUG_MARKERS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEBUG_MARKERS")?;
            remaining &= !Self::DEBUG_MARKERS.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoCapabilityFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoCapabilityFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoCapabilityFlagBitsKHR")]
pub struct VideoCapabilityFlagBitsKHR(u32);
impl VideoCapabilityFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const PROTECTED_CONTENT: Self = Self(1u32);
    ///Bit 1.
    pub const SEPARATE_REFERENCE_IMAGES: Self = Self(2u32);
}
impl core::ops::BitOr for VideoCapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoCapabilityFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoCapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoCapabilityFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoCapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoCapabilityFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoCapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoCapabilityFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::PROTECTED_CONTENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PROTECTED_CONTENT")?;
            remaining &= !Self::PROTECTED_CONTENT.0;
            first = false;
        }
        if remaining & Self::SEPARATE_REFERENCE_IMAGES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SEPARATE_REFERENCE_IMAGES")?;
            remaining &= !Self::SEPARATE_REFERENCE_IMAGES.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoChromaSubsamplingFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoChromaSubsamplingFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoChromaSubsamplingFlagBitsKHR")]
pub struct VideoChromaSubsamplingFlagBitsKHR(u32);
impl VideoChromaSubsamplingFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const INVALID: Self = Self(0u32);
    ///Bit 0.
    pub const MONOCHROME: Self = Self(1u32);
    ///Bit 1.
    pub const _420: Self = Self(2u32);
    ///Bit 2.
    pub const _422: Self = Self(4u32);
    ///Bit 3.
    pub const _444: Self = Self(8u32);
}
impl core::ops::BitOr for VideoChromaSubsamplingFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoChromaSubsamplingFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoChromaSubsamplingFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoChromaSubsamplingFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoChromaSubsamplingFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoChromaSubsamplingFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoChromaSubsamplingFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoChromaSubsamplingFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::MONOCHROME.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MONOCHROME")?;
            remaining &= !Self::MONOCHROME.0;
            first = false;
        }
        if remaining & Self::_420.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_420")?;
            remaining &= !Self::_420.0;
            first = false;
        }
        if remaining & Self::_422.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_422")?;
            remaining &= !Self::_422.0;
            first = false;
        }
        if remaining & Self::_444.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_444")?;
            remaining &= !Self::_444.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoCodecOperationFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoCodecOperationFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoCodecOperationFlagBitsKHR")]
pub struct VideoCodecOperationFlagBitsKHR(u32);
impl VideoCodecOperationFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const NONE: Self = Self(0u32);
    ///Bit 16.
    pub const ENCODE_H264: Self = Self(65536u32);
    ///Bit 17.
    pub const ENCODE_H265: Self = Self(131072u32);
    ///Bit 0.
    pub const DECODE_H264: Self = Self(1u32);
    ///Bit 1.
    pub const DECODE_H265: Self = Self(2u32);
    ///Bit 2.
    pub const DECODE_AV1: Self = Self(4u32);
    ///Bit 18.
    pub const ENCODE_AV1: Self = Self(262144u32);
    ///Bit 3.
    pub const DECODE_VP9: Self = Self(8u32);
}
impl core::ops::BitOr for VideoCodecOperationFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoCodecOperationFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoCodecOperationFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoCodecOperationFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoCodecOperationFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoCodecOperationFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoCodecOperationFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoCodecOperationFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::ENCODE_H264.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ENCODE_H264")?;
            remaining &= !Self::ENCODE_H264.0;
            first = false;
        }
        if remaining & Self::ENCODE_H265.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ENCODE_H265")?;
            remaining &= !Self::ENCODE_H265.0;
            first = false;
        }
        if remaining & Self::DECODE_H264.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DECODE_H264")?;
            remaining &= !Self::DECODE_H264.0;
            first = false;
        }
        if remaining & Self::DECODE_H265.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DECODE_H265")?;
            remaining &= !Self::DECODE_H265.0;
            first = false;
        }
        if remaining & Self::DECODE_AV1.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DECODE_AV1")?;
            remaining &= !Self::DECODE_AV1.0;
            first = false;
        }
        if remaining & Self::ENCODE_AV1.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ENCODE_AV1")?;
            remaining &= !Self::ENCODE_AV1.0;
            first = false;
        }
        if remaining & Self::DECODE_VP9.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DECODE_VP9")?;
            remaining &= !Self::DECODE_VP9.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoCodingControlFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoCodingControlFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoCodingControlFlagBitsKHR")]
pub struct VideoCodingControlFlagBitsKHR(u32);
impl VideoCodingControlFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const RESET: Self = Self(1u32);
    ///Bit 1.
    pub const ENCODE_RATE_CONTROL: Self = Self(2u32);
    ///Bit 2.
    pub const ENCODE_QUALITY_LEVEL: Self = Self(4u32);
}
impl core::ops::BitOr for VideoCodingControlFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoCodingControlFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoCodingControlFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoCodingControlFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoCodingControlFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoCodingControlFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoCodingControlFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoCodingControlFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::RESET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RESET")?;
            remaining &= !Self::RESET.0;
            first = false;
        }
        if remaining & Self::ENCODE_RATE_CONTROL.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ENCODE_RATE_CONTROL")?;
            remaining &= !Self::ENCODE_RATE_CONTROL.0;
            first = false;
        }
        if remaining & Self::ENCODE_QUALITY_LEVEL.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ENCODE_QUALITY_LEVEL")?;
            remaining &= !Self::ENCODE_QUALITY_LEVEL.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoComponentBitDepthFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoComponentBitDepthFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoComponentBitDepthFlagBitsKHR")]
pub struct VideoComponentBitDepthFlagBitsKHR(u32);
impl VideoComponentBitDepthFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const INVALID: Self = Self(0u32);
    ///Bit 0.
    pub const _8: Self = Self(1u32);
    ///Bit 2.
    pub const _10: Self = Self(4u32);
    ///Bit 4.
    pub const _12: Self = Self(16u32);
}
impl core::ops::BitOr for VideoComponentBitDepthFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoComponentBitDepthFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoComponentBitDepthFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoComponentBitDepthFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoComponentBitDepthFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoComponentBitDepthFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoComponentBitDepthFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoComponentBitDepthFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::_8.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_8")?;
            remaining &= !Self::_8.0;
            first = false;
        }
        if remaining & Self::_10.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_10")?;
            remaining &= !Self::_10.0;
            first = false;
        }
        if remaining & Self::_12.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_12")?;
            remaining &= !Self::_12.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoDecodeCapabilityFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoDecodeCapabilityFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoDecodeCapabilityFlagBitsKHR")]
pub struct VideoDecodeCapabilityFlagBitsKHR(u32);
impl VideoDecodeCapabilityFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const DPB_AND_OUTPUT_COINCIDE: Self = Self(1u32);
    ///Bit 1.
    pub const DPB_AND_OUTPUT_DISTINCT: Self = Self(2u32);
}
impl core::ops::BitOr for VideoDecodeCapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoDecodeCapabilityFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoDecodeCapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoDecodeCapabilityFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoDecodeCapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoDecodeCapabilityFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoDecodeCapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoDecodeCapabilityFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::DPB_AND_OUTPUT_COINCIDE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DPB_AND_OUTPUT_COINCIDE")?;
            remaining &= !Self::DPB_AND_OUTPUT_COINCIDE.0;
            first = false;
        }
        if remaining & Self::DPB_AND_OUTPUT_DISTINCT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DPB_AND_OUTPUT_DISTINCT")?;
            remaining &= !Self::DPB_AND_OUTPUT_DISTINCT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoDecodeH264PictureLayoutFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoDecodeH264PictureLayoutFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoDecodeH264PictureLayoutFlagBitsKHR")]
pub struct VideoDecodeH264PictureLayoutFlagBitsKHR(u32);
impl VideoDecodeH264PictureLayoutFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const PROGRESSIVE: Self = Self(0u32);
    ///Bit 0.
    pub const INTERLACED_INTERLEAVED_LINES: Self = Self(1u32);
    ///Bit 1.
    pub const INTERLACED_SEPARATE_PLANES: Self = Self(2u32);
}
impl core::ops::BitOr for VideoDecodeH264PictureLayoutFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoDecodeH264PictureLayoutFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoDecodeH264PictureLayoutFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoDecodeH264PictureLayoutFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoDecodeH264PictureLayoutFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoDecodeH264PictureLayoutFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoDecodeH264PictureLayoutFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoDecodeH264PictureLayoutFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::INTERLACED_INTERLEAVED_LINES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INTERLACED_INTERLEAVED_LINES")?;
            remaining &= !Self::INTERLACED_INTERLEAVED_LINES.0;
            first = false;
        }
        if remaining & Self::INTERLACED_SEPARATE_PLANES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INTERLACED_SEPARATE_PLANES")?;
            remaining &= !Self::INTERLACED_SEPARATE_PLANES.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoDecodeUsageFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoDecodeUsageFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoDecodeUsageFlagBitsKHR")]
pub struct VideoDecodeUsageFlagBitsKHR(u32);
impl VideoDecodeUsageFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const DEFAULT: Self = Self(0u32);
    ///Bit 0.
    pub const TRANSCODING: Self = Self(1u32);
    ///Bit 1.
    pub const OFFLINE: Self = Self(2u32);
    ///Bit 2.
    pub const STREAMING: Self = Self(4u32);
}
impl core::ops::BitOr for VideoDecodeUsageFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoDecodeUsageFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoDecodeUsageFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoDecodeUsageFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoDecodeUsageFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoDecodeUsageFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoDecodeUsageFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoDecodeUsageFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::TRANSCODING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSCODING")?;
            remaining &= !Self::TRANSCODING.0;
            first = false;
        }
        if remaining & Self::OFFLINE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("OFFLINE")?;
            remaining &= !Self::OFFLINE.0;
            first = false;
        }
        if remaining & Self::STREAMING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("STREAMING")?;
            remaining &= !Self::STREAMING.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoEncodeAV1CapabilityFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeAV1CapabilityFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoEncodeAV1CapabilityFlagBitsKHR")]
pub struct VideoEncodeAV1CapabilityFlagBitsKHR(u32);
impl VideoEncodeAV1CapabilityFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const PER_RATE_CONTROL_GROUP_MIN_MAX_Q_INDEX: Self = Self(1u32);
    ///Bit 1.
    pub const GENERATE_OBU_EXTENSION_HEADER: Self = Self(2u32);
    ///Bit 2.
    pub const PRIMARY_REFERENCE_CDF_ONLY: Self = Self(4u32);
    ///Bit 3.
    pub const FRAME_SIZE_OVERRIDE: Self = Self(8u32);
    ///Bit 4.
    pub const MOTION_VECTOR_SCALING: Self = Self(16u32);
    ///Bit 5.
    pub const COMPOUND_PREDICTION_INTRA_REFRESH: Self = Self(32u32);
}
impl core::ops::BitOr for VideoEncodeAV1CapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoEncodeAV1CapabilityFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoEncodeAV1CapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoEncodeAV1CapabilityFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoEncodeAV1CapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoEncodeAV1CapabilityFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoEncodeAV1CapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoEncodeAV1CapabilityFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::PER_RATE_CONTROL_GROUP_MIN_MAX_Q_INDEX.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PER_RATE_CONTROL_GROUP_MIN_MAX_Q_INDEX")?;
            remaining &= !Self::PER_RATE_CONTROL_GROUP_MIN_MAX_Q_INDEX.0;
            first = false;
        }
        if remaining & Self::GENERATE_OBU_EXTENSION_HEADER.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("GENERATE_OBU_EXTENSION_HEADER")?;
            remaining &= !Self::GENERATE_OBU_EXTENSION_HEADER.0;
            first = false;
        }
        if remaining & Self::PRIMARY_REFERENCE_CDF_ONLY.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PRIMARY_REFERENCE_CDF_ONLY")?;
            remaining &= !Self::PRIMARY_REFERENCE_CDF_ONLY.0;
            first = false;
        }
        if remaining & Self::FRAME_SIZE_OVERRIDE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FRAME_SIZE_OVERRIDE")?;
            remaining &= !Self::FRAME_SIZE_OVERRIDE.0;
            first = false;
        }
        if remaining & Self::MOTION_VECTOR_SCALING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MOTION_VECTOR_SCALING")?;
            remaining &= !Self::MOTION_VECTOR_SCALING.0;
            first = false;
        }
        if remaining & Self::COMPOUND_PREDICTION_INTRA_REFRESH.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("COMPOUND_PREDICTION_INTRA_REFRESH")?;
            remaining &= !Self::COMPOUND_PREDICTION_INTRA_REFRESH.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoEncodeAV1RateControlFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeAV1RateControlFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoEncodeAV1RateControlFlagBitsKHR")]
pub struct VideoEncodeAV1RateControlFlagBitsKHR(u32);
impl VideoEncodeAV1RateControlFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const REGULAR_GOP: Self = Self(1u32);
    ///Bit 1.
    pub const TEMPORAL_LAYER_PATTERN_DYADIC: Self = Self(2u32);
    ///Bit 2.
    pub const REFERENCE_PATTERN_FLAT: Self = Self(4u32);
    ///Bit 3.
    pub const REFERENCE_PATTERN_DYADIC: Self = Self(8u32);
}
impl core::ops::BitOr for VideoEncodeAV1RateControlFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoEncodeAV1RateControlFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoEncodeAV1RateControlFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoEncodeAV1RateControlFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoEncodeAV1RateControlFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoEncodeAV1RateControlFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoEncodeAV1RateControlFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoEncodeAV1RateControlFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::REGULAR_GOP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("REGULAR_GOP")?;
            remaining &= !Self::REGULAR_GOP.0;
            first = false;
        }
        if remaining & Self::TEMPORAL_LAYER_PATTERN_DYADIC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TEMPORAL_LAYER_PATTERN_DYADIC")?;
            remaining &= !Self::TEMPORAL_LAYER_PATTERN_DYADIC.0;
            first = false;
        }
        if remaining & Self::REFERENCE_PATTERN_FLAT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("REFERENCE_PATTERN_FLAT")?;
            remaining &= !Self::REFERENCE_PATTERN_FLAT.0;
            first = false;
        }
        if remaining & Self::REFERENCE_PATTERN_DYADIC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("REFERENCE_PATTERN_DYADIC")?;
            remaining &= !Self::REFERENCE_PATTERN_DYADIC.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoEncodeAV1StdFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeAV1StdFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoEncodeAV1StdFlagBitsKHR")]
pub struct VideoEncodeAV1StdFlagBitsKHR(u32);
impl VideoEncodeAV1StdFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const UNIFORM_TILE_SPACING_FLAG_SET: Self = Self(1u32);
    ///Bit 1.
    pub const SKIP_MODE_PRESENT_UNSET: Self = Self(2u32);
    ///Bit 2.
    pub const PRIMARY_REF_FRAME: Self = Self(4u32);
    ///Bit 3.
    pub const DELTA_Q: Self = Self(8u32);
}
impl core::ops::BitOr for VideoEncodeAV1StdFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoEncodeAV1StdFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoEncodeAV1StdFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoEncodeAV1StdFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoEncodeAV1StdFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoEncodeAV1StdFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoEncodeAV1StdFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoEncodeAV1StdFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::UNIFORM_TILE_SPACING_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("UNIFORM_TILE_SPACING_FLAG_SET")?;
            remaining &= !Self::UNIFORM_TILE_SPACING_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::SKIP_MODE_PRESENT_UNSET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SKIP_MODE_PRESENT_UNSET")?;
            remaining &= !Self::SKIP_MODE_PRESENT_UNSET.0;
            first = false;
        }
        if remaining & Self::PRIMARY_REF_FRAME.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PRIMARY_REF_FRAME")?;
            remaining &= !Self::PRIMARY_REF_FRAME.0;
            first = false;
        }
        if remaining & Self::DELTA_Q.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DELTA_Q")?;
            remaining &= !Self::DELTA_Q.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoEncodeAV1SuperblockSizeFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeAV1SuperblockSizeFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoEncodeAV1SuperblockSizeFlagBitsKHR")]
pub struct VideoEncodeAV1SuperblockSizeFlagBitsKHR(u32);
impl VideoEncodeAV1SuperblockSizeFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const _64: Self = Self(1u32);
    ///Bit 1.
    pub const _128: Self = Self(2u32);
}
impl core::ops::BitOr for VideoEncodeAV1SuperblockSizeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoEncodeAV1SuperblockSizeFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoEncodeAV1SuperblockSizeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoEncodeAV1SuperblockSizeFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoEncodeAV1SuperblockSizeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoEncodeAV1SuperblockSizeFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoEncodeAV1SuperblockSizeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoEncodeAV1SuperblockSizeFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::_64.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_64")?;
            remaining &= !Self::_64.0;
            first = false;
        }
        if remaining & Self::_128.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_128")?;
            remaining &= !Self::_128.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoEncodeCapabilityFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeCapabilityFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoEncodeCapabilityFlagBitsKHR")]
pub struct VideoEncodeCapabilityFlagBitsKHR(u32);
impl VideoEncodeCapabilityFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const PRECEDING_EXTERNALLY_ENCODED_BYTES: Self = Self(1u32);
    ///Bit 1.
    pub const INSUFFICIENT_BITSTREAM_BUFFER_RANGE_DETECTION: Self = Self(2u32);
    ///Bit 2.
    pub const QUANTIZATION_DELTA_MAP: Self = Self(4u32);
    ///Bit 3.
    pub const EMPHASIS_MAP: Self = Self(8u32);
}
impl core::ops::BitOr for VideoEncodeCapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoEncodeCapabilityFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoEncodeCapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoEncodeCapabilityFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoEncodeCapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoEncodeCapabilityFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoEncodeCapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoEncodeCapabilityFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::PRECEDING_EXTERNALLY_ENCODED_BYTES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PRECEDING_EXTERNALLY_ENCODED_BYTES")?;
            remaining &= !Self::PRECEDING_EXTERNALLY_ENCODED_BYTES.0;
            first = false;
        }
        if remaining & Self::INSUFFICIENT_BITSTREAM_BUFFER_RANGE_DETECTION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INSUFFICIENT_BITSTREAM_BUFFER_RANGE_DETECTION")?;
            remaining &= !Self::INSUFFICIENT_BITSTREAM_BUFFER_RANGE_DETECTION.0;
            first = false;
        }
        if remaining & Self::QUANTIZATION_DELTA_MAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("QUANTIZATION_DELTA_MAP")?;
            remaining &= !Self::QUANTIZATION_DELTA_MAP.0;
            first = false;
        }
        if remaining & Self::EMPHASIS_MAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("EMPHASIS_MAP")?;
            remaining &= !Self::EMPHASIS_MAP.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoEncodeContentFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeContentFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoEncodeContentFlagBitsKHR")]
pub struct VideoEncodeContentFlagBitsKHR(u32);
impl VideoEncodeContentFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const DEFAULT: Self = Self(0u32);
    ///Bit 0.
    pub const CAMERA: Self = Self(1u32);
    ///Bit 1.
    pub const DESKTOP: Self = Self(2u32);
    ///Bit 2.
    pub const RENDERED: Self = Self(4u32);
}
impl core::ops::BitOr for VideoEncodeContentFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoEncodeContentFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoEncodeContentFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoEncodeContentFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoEncodeContentFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoEncodeContentFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoEncodeContentFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoEncodeContentFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::CAMERA.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CAMERA")?;
            remaining &= !Self::CAMERA.0;
            first = false;
        }
        if remaining & Self::DESKTOP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DESKTOP")?;
            remaining &= !Self::DESKTOP.0;
            first = false;
        }
        if remaining & Self::RENDERED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RENDERED")?;
            remaining &= !Self::RENDERED.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoEncodeFeedbackFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeFeedbackFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoEncodeFeedbackFlagBitsKHR")]
pub struct VideoEncodeFeedbackFlagBitsKHR(u32);
impl VideoEncodeFeedbackFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const BITSTREAM_BUFFER_OFFSET: Self = Self(1u32);
    ///Bit 1.
    pub const BITSTREAM_BYTES_WRITTEN: Self = Self(2u32);
    ///Bit 2.
    pub const BITSTREAM_HAS_OVERRIDES: Self = Self(4u32);
}
impl core::ops::BitOr for VideoEncodeFeedbackFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoEncodeFeedbackFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoEncodeFeedbackFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoEncodeFeedbackFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoEncodeFeedbackFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoEncodeFeedbackFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoEncodeFeedbackFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoEncodeFeedbackFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::BITSTREAM_BUFFER_OFFSET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("BITSTREAM_BUFFER_OFFSET")?;
            remaining &= !Self::BITSTREAM_BUFFER_OFFSET.0;
            first = false;
        }
        if remaining & Self::BITSTREAM_BYTES_WRITTEN.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("BITSTREAM_BYTES_WRITTEN")?;
            remaining &= !Self::BITSTREAM_BYTES_WRITTEN.0;
            first = false;
        }
        if remaining & Self::BITSTREAM_HAS_OVERRIDES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("BITSTREAM_HAS_OVERRIDES")?;
            remaining &= !Self::BITSTREAM_HAS_OVERRIDES.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoEncodeFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoEncodeFlagBitsKHR")]
pub struct VideoEncodeFlagBitsKHR(u32);
impl VideoEncodeFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 2.
    pub const INTRA_REFRESH: Self = Self(4u32);
    ///Bit 0.
    pub const WITH_QUANTIZATION_DELTA_MAP: Self = Self(1u32);
    ///Bit 1.
    pub const WITH_EMPHASIS_MAP: Self = Self(2u32);
}
impl core::ops::BitOr for VideoEncodeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoEncodeFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoEncodeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoEncodeFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoEncodeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoEncodeFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoEncodeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoEncodeFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::INTRA_REFRESH.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INTRA_REFRESH")?;
            remaining &= !Self::INTRA_REFRESH.0;
            first = false;
        }
        if remaining & Self::WITH_QUANTIZATION_DELTA_MAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("WITH_QUANTIZATION_DELTA_MAP")?;
            remaining &= !Self::WITH_QUANTIZATION_DELTA_MAP.0;
            first = false;
        }
        if remaining & Self::WITH_EMPHASIS_MAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("WITH_EMPHASIS_MAP")?;
            remaining &= !Self::WITH_EMPHASIS_MAP.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoEncodeH264CapabilityFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH264CapabilityFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoEncodeH264CapabilityFlagBitsKHR")]
pub struct VideoEncodeH264CapabilityFlagBitsKHR(u32);
impl VideoEncodeH264CapabilityFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const HRD_COMPLIANCE: Self = Self(1u32);
    ///Bit 1.
    pub const PREDICTION_WEIGHT_TABLE_GENERATED: Self = Self(2u32);
    ///Bit 2.
    pub const ROW_UNALIGNED_SLICE: Self = Self(4u32);
    ///Bit 3.
    pub const DIFFERENT_SLICE_TYPE: Self = Self(8u32);
    ///Bit 4.
    pub const B_FRAME_IN_L0_LIST: Self = Self(16u32);
    ///Bit 5.
    pub const B_FRAME_IN_L1_LIST: Self = Self(32u32);
    ///Bit 6.
    pub const PER_PICTURE_TYPE_MIN_MAX_QP: Self = Self(64u32);
    ///Bit 7.
    pub const PER_SLICE_CONSTANT_QP: Self = Self(128u32);
    ///Bit 8.
    pub const GENERATE_PREFIX_NALU: Self = Self(256u32);
    ///Bit 10.
    pub const B_PICTURE_INTRA_REFRESH: Self = Self(1024u32);
    ///Bit 9.
    pub const MB_QP_DIFF_WRAPAROUND: Self = Self(512u32);
}
impl core::ops::BitOr for VideoEncodeH264CapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoEncodeH264CapabilityFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoEncodeH264CapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoEncodeH264CapabilityFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoEncodeH264CapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoEncodeH264CapabilityFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoEncodeH264CapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoEncodeH264CapabilityFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::HRD_COMPLIANCE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("HRD_COMPLIANCE")?;
            remaining &= !Self::HRD_COMPLIANCE.0;
            first = false;
        }
        if remaining & Self::PREDICTION_WEIGHT_TABLE_GENERATED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PREDICTION_WEIGHT_TABLE_GENERATED")?;
            remaining &= !Self::PREDICTION_WEIGHT_TABLE_GENERATED.0;
            first = false;
        }
        if remaining & Self::ROW_UNALIGNED_SLICE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ROW_UNALIGNED_SLICE")?;
            remaining &= !Self::ROW_UNALIGNED_SLICE.0;
            first = false;
        }
        if remaining & Self::DIFFERENT_SLICE_TYPE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DIFFERENT_SLICE_TYPE")?;
            remaining &= !Self::DIFFERENT_SLICE_TYPE.0;
            first = false;
        }
        if remaining & Self::B_FRAME_IN_L0_LIST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("B_FRAME_IN_L0_LIST")?;
            remaining &= !Self::B_FRAME_IN_L0_LIST.0;
            first = false;
        }
        if remaining & Self::B_FRAME_IN_L1_LIST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("B_FRAME_IN_L1_LIST")?;
            remaining &= !Self::B_FRAME_IN_L1_LIST.0;
            first = false;
        }
        if remaining & Self::PER_PICTURE_TYPE_MIN_MAX_QP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PER_PICTURE_TYPE_MIN_MAX_QP")?;
            remaining &= !Self::PER_PICTURE_TYPE_MIN_MAX_QP.0;
            first = false;
        }
        if remaining & Self::PER_SLICE_CONSTANT_QP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PER_SLICE_CONSTANT_QP")?;
            remaining &= !Self::PER_SLICE_CONSTANT_QP.0;
            first = false;
        }
        if remaining & Self::GENERATE_PREFIX_NALU.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("GENERATE_PREFIX_NALU")?;
            remaining &= !Self::GENERATE_PREFIX_NALU.0;
            first = false;
        }
        if remaining & Self::B_PICTURE_INTRA_REFRESH.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("B_PICTURE_INTRA_REFRESH")?;
            remaining &= !Self::B_PICTURE_INTRA_REFRESH.0;
            first = false;
        }
        if remaining & Self::MB_QP_DIFF_WRAPAROUND.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MB_QP_DIFF_WRAPAROUND")?;
            remaining &= !Self::MB_QP_DIFF_WRAPAROUND.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoEncodeH264RateControlFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH264RateControlFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoEncodeH264RateControlFlagBitsKHR")]
pub struct VideoEncodeH264RateControlFlagBitsKHR(u32);
impl VideoEncodeH264RateControlFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const ATTEMPT_HRD_COMPLIANCE: Self = Self(1u32);
    ///Bit 1.
    pub const REGULAR_GOP: Self = Self(2u32);
    ///Bit 2.
    pub const REFERENCE_PATTERN_FLAT: Self = Self(4u32);
    ///Bit 3.
    pub const REFERENCE_PATTERN_DYADIC: Self = Self(8u32);
    ///Bit 4.
    pub const TEMPORAL_LAYER_PATTERN_DYADIC: Self = Self(16u32);
}
impl core::ops::BitOr for VideoEncodeH264RateControlFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoEncodeH264RateControlFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoEncodeH264RateControlFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoEncodeH264RateControlFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoEncodeH264RateControlFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoEncodeH264RateControlFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoEncodeH264RateControlFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoEncodeH264RateControlFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::ATTEMPT_HRD_COMPLIANCE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ATTEMPT_HRD_COMPLIANCE")?;
            remaining &= !Self::ATTEMPT_HRD_COMPLIANCE.0;
            first = false;
        }
        if remaining & Self::REGULAR_GOP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("REGULAR_GOP")?;
            remaining &= !Self::REGULAR_GOP.0;
            first = false;
        }
        if remaining & Self::REFERENCE_PATTERN_FLAT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("REFERENCE_PATTERN_FLAT")?;
            remaining &= !Self::REFERENCE_PATTERN_FLAT.0;
            first = false;
        }
        if remaining & Self::REFERENCE_PATTERN_DYADIC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("REFERENCE_PATTERN_DYADIC")?;
            remaining &= !Self::REFERENCE_PATTERN_DYADIC.0;
            first = false;
        }
        if remaining & Self::TEMPORAL_LAYER_PATTERN_DYADIC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TEMPORAL_LAYER_PATTERN_DYADIC")?;
            remaining &= !Self::TEMPORAL_LAYER_PATTERN_DYADIC.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoEncodeH264StdFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH264StdFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoEncodeH264StdFlagBitsKHR")]
pub struct VideoEncodeH264StdFlagBitsKHR(u32);
impl VideoEncodeH264StdFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const SEPARATE_COLOR_PLANE_FLAG_SET: Self = Self(1u32);
    ///Bit 1.
    pub const QPPRIME_Y_ZERO_TRANSFORM_BYPASS_FLAG_SET: Self = Self(2u32);
    ///Bit 2.
    pub const SCALING_MATRIX_PRESENT_FLAG_SET: Self = Self(4u32);
    ///Bit 3.
    pub const CHROMA_QP_INDEX_OFFSET: Self = Self(8u32);
    ///Bit 4.
    pub const SECOND_CHROMA_QP_INDEX_OFFSET: Self = Self(16u32);
    ///Bit 5.
    pub const PIC_INIT_QP_MINUS26: Self = Self(32u32);
    ///Bit 6.
    pub const WEIGHTED_PRED_FLAG_SET: Self = Self(64u32);
    ///Bit 7.
    pub const WEIGHTED_BIPRED_IDC_EXPLICIT: Self = Self(128u32);
    ///Bit 8.
    pub const WEIGHTED_BIPRED_IDC_IMPLICIT: Self = Self(256u32);
    ///Bit 9.
    pub const TRANSFORM_8X8_MODE_FLAG_SET: Self = Self(512u32);
    ///Bit 10.
    pub const DIRECT_SPATIAL_MV_PRED_FLAG_UNSET: Self = Self(1024u32);
    ///Bit 11.
    pub const ENTROPY_CODING_MODE_FLAG_UNSET: Self = Self(2048u32);
    ///Bit 12.
    pub const ENTROPY_CODING_MODE_FLAG_SET: Self = Self(4096u32);
    ///Bit 13.
    pub const DIRECT_8X8_INFERENCE_FLAG_UNSET: Self = Self(8192u32);
    ///Bit 14.
    pub const CONSTRAINED_INTRA_PRED_FLAG_SET: Self = Self(16384u32);
    ///Bit 15.
    pub const DEBLOCKING_FILTER_DISABLED: Self = Self(32768u32);
    ///Bit 16.
    pub const DEBLOCKING_FILTER_ENABLED: Self = Self(65536u32);
    ///Bit 17.
    pub const DEBLOCKING_FILTER_PARTIAL: Self = Self(131072u32);
    ///Bit 19.
    pub const SLICE_QP_DELTA: Self = Self(524288u32);
    ///Bit 20.
    pub const DIFFERENT_SLICE_QP_DELTA: Self = Self(1048576u32);
}
impl core::ops::BitOr for VideoEncodeH264StdFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoEncodeH264StdFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoEncodeH264StdFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoEncodeH264StdFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoEncodeH264StdFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoEncodeH264StdFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoEncodeH264StdFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoEncodeH264StdFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::SEPARATE_COLOR_PLANE_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SEPARATE_COLOR_PLANE_FLAG_SET")?;
            remaining &= !Self::SEPARATE_COLOR_PLANE_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::QPPRIME_Y_ZERO_TRANSFORM_BYPASS_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("QPPRIME_Y_ZERO_TRANSFORM_BYPASS_FLAG_SET")?;
            remaining &= !Self::QPPRIME_Y_ZERO_TRANSFORM_BYPASS_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::SCALING_MATRIX_PRESENT_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SCALING_MATRIX_PRESENT_FLAG_SET")?;
            remaining &= !Self::SCALING_MATRIX_PRESENT_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::CHROMA_QP_INDEX_OFFSET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CHROMA_QP_INDEX_OFFSET")?;
            remaining &= !Self::CHROMA_QP_INDEX_OFFSET.0;
            first = false;
        }
        if remaining & Self::SECOND_CHROMA_QP_INDEX_OFFSET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SECOND_CHROMA_QP_INDEX_OFFSET")?;
            remaining &= !Self::SECOND_CHROMA_QP_INDEX_OFFSET.0;
            first = false;
        }
        if remaining & Self::PIC_INIT_QP_MINUS26.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PIC_INIT_QP_MINUS26")?;
            remaining &= !Self::PIC_INIT_QP_MINUS26.0;
            first = false;
        }
        if remaining & Self::WEIGHTED_PRED_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("WEIGHTED_PRED_FLAG_SET")?;
            remaining &= !Self::WEIGHTED_PRED_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::WEIGHTED_BIPRED_IDC_EXPLICIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("WEIGHTED_BIPRED_IDC_EXPLICIT")?;
            remaining &= !Self::WEIGHTED_BIPRED_IDC_EXPLICIT.0;
            first = false;
        }
        if remaining & Self::WEIGHTED_BIPRED_IDC_IMPLICIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("WEIGHTED_BIPRED_IDC_IMPLICIT")?;
            remaining &= !Self::WEIGHTED_BIPRED_IDC_IMPLICIT.0;
            first = false;
        }
        if remaining & Self::TRANSFORM_8X8_MODE_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFORM_8X8_MODE_FLAG_SET")?;
            remaining &= !Self::TRANSFORM_8X8_MODE_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::DIRECT_SPATIAL_MV_PRED_FLAG_UNSET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DIRECT_SPATIAL_MV_PRED_FLAG_UNSET")?;
            remaining &= !Self::DIRECT_SPATIAL_MV_PRED_FLAG_UNSET.0;
            first = false;
        }
        if remaining & Self::ENTROPY_CODING_MODE_FLAG_UNSET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ENTROPY_CODING_MODE_FLAG_UNSET")?;
            remaining &= !Self::ENTROPY_CODING_MODE_FLAG_UNSET.0;
            first = false;
        }
        if remaining & Self::ENTROPY_CODING_MODE_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ENTROPY_CODING_MODE_FLAG_SET")?;
            remaining &= !Self::ENTROPY_CODING_MODE_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::DIRECT_8X8_INFERENCE_FLAG_UNSET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DIRECT_8X8_INFERENCE_FLAG_UNSET")?;
            remaining &= !Self::DIRECT_8X8_INFERENCE_FLAG_UNSET.0;
            first = false;
        }
        if remaining & Self::CONSTRAINED_INTRA_PRED_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CONSTRAINED_INTRA_PRED_FLAG_SET")?;
            remaining &= !Self::CONSTRAINED_INTRA_PRED_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::DEBLOCKING_FILTER_DISABLED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEBLOCKING_FILTER_DISABLED")?;
            remaining &= !Self::DEBLOCKING_FILTER_DISABLED.0;
            first = false;
        }
        if remaining & Self::DEBLOCKING_FILTER_ENABLED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEBLOCKING_FILTER_ENABLED")?;
            remaining &= !Self::DEBLOCKING_FILTER_ENABLED.0;
            first = false;
        }
        if remaining & Self::DEBLOCKING_FILTER_PARTIAL.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEBLOCKING_FILTER_PARTIAL")?;
            remaining &= !Self::DEBLOCKING_FILTER_PARTIAL.0;
            first = false;
        }
        if remaining & Self::SLICE_QP_DELTA.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SLICE_QP_DELTA")?;
            remaining &= !Self::SLICE_QP_DELTA.0;
            first = false;
        }
        if remaining & Self::DIFFERENT_SLICE_QP_DELTA.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DIFFERENT_SLICE_QP_DELTA")?;
            remaining &= !Self::DIFFERENT_SLICE_QP_DELTA.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoEncodeH265CapabilityFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH265CapabilityFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoEncodeH265CapabilityFlagBitsKHR")]
pub struct VideoEncodeH265CapabilityFlagBitsKHR(u32);
impl VideoEncodeH265CapabilityFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const HRD_COMPLIANCE: Self = Self(1u32);
    ///Bit 1.
    pub const PREDICTION_WEIGHT_TABLE_GENERATED: Self = Self(2u32);
    ///Bit 2.
    pub const ROW_UNALIGNED_SLICE_SEGMENT: Self = Self(4u32);
    ///Bit 3.
    pub const DIFFERENT_SLICE_SEGMENT_TYPE: Self = Self(8u32);
    ///Bit 4.
    pub const B_FRAME_IN_L0_LIST: Self = Self(16u32);
    ///Bit 5.
    pub const B_FRAME_IN_L1_LIST: Self = Self(32u32);
    ///Bit 6.
    pub const PER_PICTURE_TYPE_MIN_MAX_QP: Self = Self(64u32);
    ///Bit 7.
    pub const PER_SLICE_SEGMENT_CONSTANT_QP: Self = Self(128u32);
    ///Bit 8.
    pub const MULTIPLE_TILES_PER_SLICE_SEGMENT: Self = Self(256u32);
    ///Bit 9.
    pub const MULTIPLE_SLICE_SEGMENTS_PER_TILE: Self = Self(512u32);
    ///Bit 11.
    pub const B_PICTURE_INTRA_REFRESH: Self = Self(2048u32);
    ///Bit 10.
    pub const CU_QP_DIFF_WRAPAROUND: Self = Self(1024u32);
}
impl core::ops::BitOr for VideoEncodeH265CapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoEncodeH265CapabilityFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoEncodeH265CapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoEncodeH265CapabilityFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoEncodeH265CapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoEncodeH265CapabilityFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoEncodeH265CapabilityFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoEncodeH265CapabilityFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::HRD_COMPLIANCE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("HRD_COMPLIANCE")?;
            remaining &= !Self::HRD_COMPLIANCE.0;
            first = false;
        }
        if remaining & Self::PREDICTION_WEIGHT_TABLE_GENERATED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PREDICTION_WEIGHT_TABLE_GENERATED")?;
            remaining &= !Self::PREDICTION_WEIGHT_TABLE_GENERATED.0;
            first = false;
        }
        if remaining & Self::ROW_UNALIGNED_SLICE_SEGMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ROW_UNALIGNED_SLICE_SEGMENT")?;
            remaining &= !Self::ROW_UNALIGNED_SLICE_SEGMENT.0;
            first = false;
        }
        if remaining & Self::DIFFERENT_SLICE_SEGMENT_TYPE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DIFFERENT_SLICE_SEGMENT_TYPE")?;
            remaining &= !Self::DIFFERENT_SLICE_SEGMENT_TYPE.0;
            first = false;
        }
        if remaining & Self::B_FRAME_IN_L0_LIST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("B_FRAME_IN_L0_LIST")?;
            remaining &= !Self::B_FRAME_IN_L0_LIST.0;
            first = false;
        }
        if remaining & Self::B_FRAME_IN_L1_LIST.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("B_FRAME_IN_L1_LIST")?;
            remaining &= !Self::B_FRAME_IN_L1_LIST.0;
            first = false;
        }
        if remaining & Self::PER_PICTURE_TYPE_MIN_MAX_QP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PER_PICTURE_TYPE_MIN_MAX_QP")?;
            remaining &= !Self::PER_PICTURE_TYPE_MIN_MAX_QP.0;
            first = false;
        }
        if remaining & Self::PER_SLICE_SEGMENT_CONSTANT_QP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PER_SLICE_SEGMENT_CONSTANT_QP")?;
            remaining &= !Self::PER_SLICE_SEGMENT_CONSTANT_QP.0;
            first = false;
        }
        if remaining & Self::MULTIPLE_TILES_PER_SLICE_SEGMENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MULTIPLE_TILES_PER_SLICE_SEGMENT")?;
            remaining &= !Self::MULTIPLE_TILES_PER_SLICE_SEGMENT.0;
            first = false;
        }
        if remaining & Self::MULTIPLE_SLICE_SEGMENTS_PER_TILE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MULTIPLE_SLICE_SEGMENTS_PER_TILE")?;
            remaining &= !Self::MULTIPLE_SLICE_SEGMENTS_PER_TILE.0;
            first = false;
        }
        if remaining & Self::B_PICTURE_INTRA_REFRESH.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("B_PICTURE_INTRA_REFRESH")?;
            remaining &= !Self::B_PICTURE_INTRA_REFRESH.0;
            first = false;
        }
        if remaining & Self::CU_QP_DIFF_WRAPAROUND.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CU_QP_DIFF_WRAPAROUND")?;
            remaining &= !Self::CU_QP_DIFF_WRAPAROUND.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoEncodeH265CtbSizeFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH265CtbSizeFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoEncodeH265CtbSizeFlagBitsKHR")]
pub struct VideoEncodeH265CtbSizeFlagBitsKHR(u32);
impl VideoEncodeH265CtbSizeFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const _16: Self = Self(1u32);
    ///Bit 1.
    pub const _32: Self = Self(2u32);
    ///Bit 2.
    pub const _64: Self = Self(4u32);
}
impl core::ops::BitOr for VideoEncodeH265CtbSizeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoEncodeH265CtbSizeFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoEncodeH265CtbSizeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoEncodeH265CtbSizeFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoEncodeH265CtbSizeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoEncodeH265CtbSizeFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoEncodeH265CtbSizeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoEncodeH265CtbSizeFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::_16.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_16")?;
            remaining &= !Self::_16.0;
            first = false;
        }
        if remaining & Self::_32.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_32")?;
            remaining &= !Self::_32.0;
            first = false;
        }
        if remaining & Self::_64.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_64")?;
            remaining &= !Self::_64.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoEncodeH265RateControlFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH265RateControlFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoEncodeH265RateControlFlagBitsKHR")]
pub struct VideoEncodeH265RateControlFlagBitsKHR(u32);
impl VideoEncodeH265RateControlFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const ATTEMPT_HRD_COMPLIANCE: Self = Self(1u32);
    ///Bit 1.
    pub const REGULAR_GOP: Self = Self(2u32);
    ///Bit 2.
    pub const REFERENCE_PATTERN_FLAT: Self = Self(4u32);
    ///Bit 3.
    pub const REFERENCE_PATTERN_DYADIC: Self = Self(8u32);
    ///Bit 4.
    pub const TEMPORAL_SUB_LAYER_PATTERN_DYADIC: Self = Self(16u32);
}
impl core::ops::BitOr for VideoEncodeH265RateControlFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoEncodeH265RateControlFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoEncodeH265RateControlFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoEncodeH265RateControlFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoEncodeH265RateControlFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoEncodeH265RateControlFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoEncodeH265RateControlFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoEncodeH265RateControlFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::ATTEMPT_HRD_COMPLIANCE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ATTEMPT_HRD_COMPLIANCE")?;
            remaining &= !Self::ATTEMPT_HRD_COMPLIANCE.0;
            first = false;
        }
        if remaining & Self::REGULAR_GOP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("REGULAR_GOP")?;
            remaining &= !Self::REGULAR_GOP.0;
            first = false;
        }
        if remaining & Self::REFERENCE_PATTERN_FLAT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("REFERENCE_PATTERN_FLAT")?;
            remaining &= !Self::REFERENCE_PATTERN_FLAT.0;
            first = false;
        }
        if remaining & Self::REFERENCE_PATTERN_DYADIC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("REFERENCE_PATTERN_DYADIC")?;
            remaining &= !Self::REFERENCE_PATTERN_DYADIC.0;
            first = false;
        }
        if remaining & Self::TEMPORAL_SUB_LAYER_PATTERN_DYADIC.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TEMPORAL_SUB_LAYER_PATTERN_DYADIC")?;
            remaining &= !Self::TEMPORAL_SUB_LAYER_PATTERN_DYADIC.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoEncodeH265StdFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH265StdFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoEncodeH265StdFlagBitsKHR")]
pub struct VideoEncodeH265StdFlagBitsKHR(u32);
impl VideoEncodeH265StdFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const SEPARATE_COLOR_PLANE_FLAG_SET: Self = Self(1u32);
    ///Bit 1.
    pub const SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG_SET: Self = Self(2u32);
    ///Bit 2.
    pub const SCALING_LIST_DATA_PRESENT_FLAG_SET: Self = Self(4u32);
    ///Bit 3.
    pub const PCM_ENABLED_FLAG_SET: Self = Self(8u32);
    ///Bit 4.
    pub const SPS_TEMPORAL_MVP_ENABLED_FLAG_SET: Self = Self(16u32);
    ///Bit 5.
    pub const INIT_QP_MINUS26: Self = Self(32u32);
    ///Bit 6.
    pub const WEIGHTED_PRED_FLAG_SET: Self = Self(64u32);
    ///Bit 7.
    pub const WEIGHTED_BIPRED_FLAG_SET: Self = Self(128u32);
    ///Bit 8.
    pub const LOG2_PARALLEL_MERGE_LEVEL_MINUS2: Self = Self(256u32);
    ///Bit 9.
    pub const SIGN_DATA_HIDING_ENABLED_FLAG_SET: Self = Self(512u32);
    ///Bit 10.
    pub const TRANSFORM_SKIP_ENABLED_FLAG_SET: Self = Self(1024u32);
    ///Bit 11.
    pub const TRANSFORM_SKIP_ENABLED_FLAG_UNSET: Self = Self(2048u32);
    ///Bit 12.
    pub const PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT_FLAG_SET: Self = Self(4096u32);
    ///Bit 13.
    pub const TRANSQUANT_BYPASS_ENABLED_FLAG_SET: Self = Self(8192u32);
    ///Bit 14.
    pub const CONSTRAINED_INTRA_PRED_FLAG_SET: Self = Self(16384u32);
    ///Bit 15.
    pub const ENTROPY_CODING_SYNC_ENABLED_FLAG_SET: Self = Self(32768u32);
    ///Bit 16.
    pub const DEBLOCKING_FILTER_OVERRIDE_ENABLED_FLAG_SET: Self = Self(65536u32);
    ///Bit 17.
    pub const DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG_SET: Self = Self(131072u32);
    ///Bit 18.
    pub const DEPENDENT_SLICE_SEGMENT_FLAG_SET: Self = Self(262144u32);
    ///Bit 19.
    pub const SLICE_QP_DELTA: Self = Self(524288u32);
    ///Bit 20.
    pub const DIFFERENT_SLICE_QP_DELTA: Self = Self(1048576u32);
}
impl core::ops::BitOr for VideoEncodeH265StdFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoEncodeH265StdFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoEncodeH265StdFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoEncodeH265StdFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoEncodeH265StdFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoEncodeH265StdFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoEncodeH265StdFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoEncodeH265StdFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::SEPARATE_COLOR_PLANE_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SEPARATE_COLOR_PLANE_FLAG_SET")?;
            remaining &= !Self::SEPARATE_COLOR_PLANE_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG_SET")?;
            remaining &= !Self::SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::SCALING_LIST_DATA_PRESENT_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SCALING_LIST_DATA_PRESENT_FLAG_SET")?;
            remaining &= !Self::SCALING_LIST_DATA_PRESENT_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::PCM_ENABLED_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PCM_ENABLED_FLAG_SET")?;
            remaining &= !Self::PCM_ENABLED_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::SPS_TEMPORAL_MVP_ENABLED_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SPS_TEMPORAL_MVP_ENABLED_FLAG_SET")?;
            remaining &= !Self::SPS_TEMPORAL_MVP_ENABLED_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::INIT_QP_MINUS26.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INIT_QP_MINUS26")?;
            remaining &= !Self::INIT_QP_MINUS26.0;
            first = false;
        }
        if remaining & Self::WEIGHTED_PRED_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("WEIGHTED_PRED_FLAG_SET")?;
            remaining &= !Self::WEIGHTED_PRED_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::WEIGHTED_BIPRED_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("WEIGHTED_BIPRED_FLAG_SET")?;
            remaining &= !Self::WEIGHTED_BIPRED_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::LOG2_PARALLEL_MERGE_LEVEL_MINUS2.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("LOG2_PARALLEL_MERGE_LEVEL_MINUS2")?;
            remaining &= !Self::LOG2_PARALLEL_MERGE_LEVEL_MINUS2.0;
            first = false;
        }
        if remaining & Self::SIGN_DATA_HIDING_ENABLED_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SIGN_DATA_HIDING_ENABLED_FLAG_SET")?;
            remaining &= !Self::SIGN_DATA_HIDING_ENABLED_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::TRANSFORM_SKIP_ENABLED_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFORM_SKIP_ENABLED_FLAG_SET")?;
            remaining &= !Self::TRANSFORM_SKIP_ENABLED_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::TRANSFORM_SKIP_ENABLED_FLAG_UNSET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSFORM_SKIP_ENABLED_FLAG_UNSET")?;
            remaining &= !Self::TRANSFORM_SKIP_ENABLED_FLAG_UNSET.0;
            first = false;
        }
        if remaining & Self::PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT_FLAG_SET")?;
            remaining &= !Self::PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::TRANSQUANT_BYPASS_ENABLED_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSQUANT_BYPASS_ENABLED_FLAG_SET")?;
            remaining &= !Self::TRANSQUANT_BYPASS_ENABLED_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::CONSTRAINED_INTRA_PRED_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CONSTRAINED_INTRA_PRED_FLAG_SET")?;
            remaining &= !Self::CONSTRAINED_INTRA_PRED_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::ENTROPY_CODING_SYNC_ENABLED_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ENTROPY_CODING_SYNC_ENABLED_FLAG_SET")?;
            remaining &= !Self::ENTROPY_CODING_SYNC_ENABLED_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::DEBLOCKING_FILTER_OVERRIDE_ENABLED_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEBLOCKING_FILTER_OVERRIDE_ENABLED_FLAG_SET")?;
            remaining &= !Self::DEBLOCKING_FILTER_OVERRIDE_ENABLED_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG_SET")?;
            remaining &= !Self::DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::DEPENDENT_SLICE_SEGMENT_FLAG_SET.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DEPENDENT_SLICE_SEGMENT_FLAG_SET")?;
            remaining &= !Self::DEPENDENT_SLICE_SEGMENT_FLAG_SET.0;
            first = false;
        }
        if remaining & Self::SLICE_QP_DELTA.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("SLICE_QP_DELTA")?;
            remaining &= !Self::SLICE_QP_DELTA.0;
            first = false;
        }
        if remaining & Self::DIFFERENT_SLICE_QP_DELTA.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DIFFERENT_SLICE_QP_DELTA")?;
            remaining &= !Self::DIFFERENT_SLICE_QP_DELTA.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoEncodeH265TransformBlockSizeFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH265TransformBlockSizeFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoEncodeH265TransformBlockSizeFlagBitsKHR")]
pub struct VideoEncodeH265TransformBlockSizeFlagBitsKHR(u32);
impl VideoEncodeH265TransformBlockSizeFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const _4: Self = Self(1u32);
    ///Bit 1.
    pub const _8: Self = Self(2u32);
    ///Bit 2.
    pub const _16: Self = Self(4u32);
    ///Bit 3.
    pub const _32: Self = Self(8u32);
}
impl core::ops::BitOr for VideoEncodeH265TransformBlockSizeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoEncodeH265TransformBlockSizeFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoEncodeH265TransformBlockSizeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoEncodeH265TransformBlockSizeFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoEncodeH265TransformBlockSizeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoEncodeH265TransformBlockSizeFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoEncodeH265TransformBlockSizeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoEncodeH265TransformBlockSizeFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::_4.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_4")?;
            remaining &= !Self::_4.0;
            first = false;
        }
        if remaining & Self::_8.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_8")?;
            remaining &= !Self::_8.0;
            first = false;
        }
        if remaining & Self::_16.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_16")?;
            remaining &= !Self::_16.0;
            first = false;
        }
        if remaining & Self::_32.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("_32")?;
            remaining &= !Self::_32.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoEncodeIntraRefreshModeFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeIntraRefreshModeFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoEncodeIntraRefreshModeFlagBitsKHR")]
pub struct VideoEncodeIntraRefreshModeFlagBitsKHR(u32);
impl VideoEncodeIntraRefreshModeFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const NONE: Self = Self(0u32);
    ///Bit 0.
    pub const PER_PICTURE_PARTITION: Self = Self(1u32);
    ///Bit 1.
    pub const BLOCK_BASED: Self = Self(2u32);
    ///Bit 2.
    pub const BLOCK_ROW_BASED: Self = Self(4u32);
    ///Bit 3.
    pub const BLOCK_COLUMN_BASED: Self = Self(8u32);
}
impl core::ops::BitOr for VideoEncodeIntraRefreshModeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoEncodeIntraRefreshModeFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoEncodeIntraRefreshModeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoEncodeIntraRefreshModeFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoEncodeIntraRefreshModeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoEncodeIntraRefreshModeFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoEncodeIntraRefreshModeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoEncodeIntraRefreshModeFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::PER_PICTURE_PARTITION.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PER_PICTURE_PARTITION")?;
            remaining &= !Self::PER_PICTURE_PARTITION.0;
            first = false;
        }
        if remaining & Self::BLOCK_BASED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("BLOCK_BASED")?;
            remaining &= !Self::BLOCK_BASED.0;
            first = false;
        }
        if remaining & Self::BLOCK_ROW_BASED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("BLOCK_ROW_BASED")?;
            remaining &= !Self::BLOCK_ROW_BASED.0;
            first = false;
        }
        if remaining & Self::BLOCK_COLUMN_BASED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("BLOCK_COLUMN_BASED")?;
            remaining &= !Self::BLOCK_COLUMN_BASED.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoEncodeRateControlModeFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeRateControlModeFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoEncodeRateControlModeFlagBitsKHR")]
pub struct VideoEncodeRateControlModeFlagBitsKHR(u32);
impl VideoEncodeRateControlModeFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const DEFAULT: Self = Self(0u32);
    ///Bit 0.
    pub const DISABLED: Self = Self(1u32);
    ///Bit 1.
    pub const CBR: Self = Self(2u32);
    ///Bit 2.
    pub const VBR: Self = Self(4u32);
}
impl core::ops::BitOr for VideoEncodeRateControlModeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoEncodeRateControlModeFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoEncodeRateControlModeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoEncodeRateControlModeFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoEncodeRateControlModeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoEncodeRateControlModeFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoEncodeRateControlModeFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoEncodeRateControlModeFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::DISABLED.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("DISABLED")?;
            remaining &= !Self::DISABLED.0;
            first = false;
        }
        if remaining & Self::CBR.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CBR")?;
            remaining &= !Self::CBR.0;
            first = false;
        }
        if remaining & Self::VBR.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("VBR")?;
            remaining &= !Self::VBR.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoEncodeRgbChromaOffsetFlagBitsVALVE`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeRgbChromaOffsetFlagBitsVALVE.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoEncodeRgbChromaOffsetFlagBitsVALVE")]
pub struct VideoEncodeRgbChromaOffsetFlagBitsVALVE(u32);
impl VideoEncodeRgbChromaOffsetFlagBitsVALVE {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const COSITED_EVEN_BIT: Self = Self(1u32);
    ///Bit 1.
    pub const MIDPOINT_BIT: Self = Self(2u32);
}
impl core::ops::BitOr for VideoEncodeRgbChromaOffsetFlagBitsVALVE {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoEncodeRgbChromaOffsetFlagBitsVALVE {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoEncodeRgbChromaOffsetFlagBitsVALVE {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoEncodeRgbChromaOffsetFlagBitsVALVE {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoEncodeRgbChromaOffsetFlagBitsVALVE {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoEncodeRgbChromaOffsetFlagBitsVALVE {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoEncodeRgbChromaOffsetFlagBitsVALVE {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoEncodeRgbChromaOffsetFlagBitsVALVE {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::COSITED_EVEN_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("COSITED_EVEN_BIT")?;
            remaining &= !Self::COSITED_EVEN_BIT.0;
            first = false;
        }
        if remaining & Self::MIDPOINT_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("MIDPOINT_BIT")?;
            remaining &= !Self::MIDPOINT_BIT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoEncodeRgbModelConversionFlagBitsVALVE`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeRgbModelConversionFlagBitsVALVE.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoEncodeRgbModelConversionFlagBitsVALVE")]
pub struct VideoEncodeRgbModelConversionFlagBitsVALVE(u32);
impl VideoEncodeRgbModelConversionFlagBitsVALVE {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const RGB_IDENTITY_BIT: Self = Self(1u32);
    ///Bit 1.
    pub const YCBCR_IDENTITY_BIT: Self = Self(2u32);
    ///Bit 2.
    pub const YCBCR_709_BIT: Self = Self(4u32);
    ///Bit 3.
    pub const YCBCR_601_BIT: Self = Self(8u32);
    ///Bit 4.
    pub const YCBCR_2020_BIT: Self = Self(16u32);
}
impl core::ops::BitOr for VideoEncodeRgbModelConversionFlagBitsVALVE {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoEncodeRgbModelConversionFlagBitsVALVE {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoEncodeRgbModelConversionFlagBitsVALVE {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoEncodeRgbModelConversionFlagBitsVALVE {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoEncodeRgbModelConversionFlagBitsVALVE {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoEncodeRgbModelConversionFlagBitsVALVE {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoEncodeRgbModelConversionFlagBitsVALVE {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoEncodeRgbModelConversionFlagBitsVALVE {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::RGB_IDENTITY_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RGB_IDENTITY_BIT")?;
            remaining &= !Self::RGB_IDENTITY_BIT.0;
            first = false;
        }
        if remaining & Self::YCBCR_IDENTITY_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("YCBCR_IDENTITY_BIT")?;
            remaining &= !Self::YCBCR_IDENTITY_BIT.0;
            first = false;
        }
        if remaining & Self::YCBCR_709_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("YCBCR_709_BIT")?;
            remaining &= !Self::YCBCR_709_BIT.0;
            first = false;
        }
        if remaining & Self::YCBCR_601_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("YCBCR_601_BIT")?;
            remaining &= !Self::YCBCR_601_BIT.0;
            first = false;
        }
        if remaining & Self::YCBCR_2020_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("YCBCR_2020_BIT")?;
            remaining &= !Self::YCBCR_2020_BIT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoEncodeRgbRangeCompressionFlagBitsVALVE`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeRgbRangeCompressionFlagBitsVALVE.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoEncodeRgbRangeCompressionFlagBitsVALVE")]
pub struct VideoEncodeRgbRangeCompressionFlagBitsVALVE(u32);
impl VideoEncodeRgbRangeCompressionFlagBitsVALVE {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const FULL_RANGE_BIT: Self = Self(1u32);
    ///Bit 1.
    pub const NARROW_RANGE_BIT: Self = Self(2u32);
}
impl core::ops::BitOr for VideoEncodeRgbRangeCompressionFlagBitsVALVE {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoEncodeRgbRangeCompressionFlagBitsVALVE {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoEncodeRgbRangeCompressionFlagBitsVALVE {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoEncodeRgbRangeCompressionFlagBitsVALVE {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoEncodeRgbRangeCompressionFlagBitsVALVE {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoEncodeRgbRangeCompressionFlagBitsVALVE {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoEncodeRgbRangeCompressionFlagBitsVALVE {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoEncodeRgbRangeCompressionFlagBitsVALVE {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::FULL_RANGE_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("FULL_RANGE_BIT")?;
            remaining &= !Self::FULL_RANGE_BIT.0;
            first = false;
        }
        if remaining & Self::NARROW_RANGE_BIT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("NARROW_RANGE_BIT")?;
            remaining &= !Self::NARROW_RANGE_BIT.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoEncodeUsageFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeUsageFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoEncodeUsageFlagBitsKHR")]
pub struct VideoEncodeUsageFlagBitsKHR(u32);
impl VideoEncodeUsageFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const DEFAULT: Self = Self(0u32);
    ///Bit 0.
    pub const TRANSCODING: Self = Self(1u32);
    ///Bit 1.
    pub const STREAMING: Self = Self(2u32);
    ///Bit 2.
    pub const RECORDING: Self = Self(4u32);
    ///Bit 3.
    pub const CONFERENCING: Self = Self(8u32);
}
impl core::ops::BitOr for VideoEncodeUsageFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoEncodeUsageFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoEncodeUsageFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoEncodeUsageFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoEncodeUsageFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoEncodeUsageFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoEncodeUsageFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoEncodeUsageFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::TRANSCODING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("TRANSCODING")?;
            remaining &= !Self::TRANSCODING.0;
            first = false;
        }
        if remaining & Self::STREAMING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("STREAMING")?;
            remaining &= !Self::STREAMING.0;
            first = false;
        }
        if remaining & Self::RECORDING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("RECORDING")?;
            remaining &= !Self::RECORDING.0;
            first = false;
        }
        if remaining & Self::CONFERENCING.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("CONFERENCING")?;
            remaining &= !Self::CONFERENCING.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoSessionCreateFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoSessionCreateFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoSessionCreateFlagBitsKHR")]
pub struct VideoSessionCreateFlagBitsKHR(u32);
impl VideoSessionCreateFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const PROTECTED_CONTENT: Self = Self(1u32);
    ///Bit 1.
    pub const ALLOW_ENCODE_PARAMETER_OPTIMIZATIONS: Self = Self(2u32);
    ///Bit 2.
    pub const INLINE_QUERIES: Self = Self(4u32);
    ///Bit 3.
    pub const ALLOW_ENCODE_QUANTIZATION_DELTA_MAP: Self = Self(8u32);
    ///Bit 4.
    pub const ALLOW_ENCODE_EMPHASIS_MAP: Self = Self(16u32);
    ///Bit 5.
    pub const INLINE_SESSION_PARAMETERS: Self = Self(32u32);
}
impl core::ops::BitOr for VideoSessionCreateFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoSessionCreateFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoSessionCreateFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoSessionCreateFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoSessionCreateFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoSessionCreateFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoSessionCreateFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoSessionCreateFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::PROTECTED_CONTENT.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("PROTECTED_CONTENT")?;
            remaining &= !Self::PROTECTED_CONTENT.0;
            first = false;
        }
        if remaining & Self::ALLOW_ENCODE_PARAMETER_OPTIMIZATIONS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALLOW_ENCODE_PARAMETER_OPTIMIZATIONS")?;
            remaining &= !Self::ALLOW_ENCODE_PARAMETER_OPTIMIZATIONS.0;
            first = false;
        }
        if remaining & Self::INLINE_QUERIES.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INLINE_QUERIES")?;
            remaining &= !Self::INLINE_QUERIES.0;
            first = false;
        }
        if remaining & Self::ALLOW_ENCODE_QUANTIZATION_DELTA_MAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALLOW_ENCODE_QUANTIZATION_DELTA_MAP")?;
            remaining &= !Self::ALLOW_ENCODE_QUANTIZATION_DELTA_MAP.0;
            first = false;
        }
        if remaining & Self::ALLOW_ENCODE_EMPHASIS_MAP.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("ALLOW_ENCODE_EMPHASIS_MAP")?;
            remaining &= !Self::ALLOW_ENCODE_EMPHASIS_MAP.0;
            first = false;
        }
        if remaining & Self::INLINE_SESSION_PARAMETERS.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("INLINE_SESSION_PARAMETERS")?;
            remaining &= !Self::INLINE_SESSION_PARAMETERS.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkVideoSessionParametersCreateFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoSessionParametersCreateFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkVideoSessionParametersCreateFlagBitsKHR")]
pub struct VideoSessionParametersCreateFlagBitsKHR(u32);
impl VideoSessionParametersCreateFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    ///Bit 0.
    pub const QUANTIZATION_MAP_COMPATIBLE: Self = Self(1u32);
}
impl core::ops::BitOr for VideoSessionParametersCreateFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for VideoSessionParametersCreateFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for VideoSessionParametersCreateFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for VideoSessionParametersCreateFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for VideoSessionParametersCreateFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for VideoSessionParametersCreateFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for VideoSessionParametersCreateFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for VideoSessionParametersCreateFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining & Self::QUANTIZATION_MAP_COMPATIBLE.0 != 0 {
            if !first {
                f.write_str(" | ")?;
            }
            f.write_str("QUANTIZATION_MAP_COMPATIBLE")?;
            remaining &= !Self::QUANTIZATION_MAP_COMPATIBLE.0;
            first = false;
        }
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
///[`VkWaylandSurfaceCreateFlagBitsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkWaylandSurfaceCreateFlagBitsKHR.html)
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
#[doc(alias = "VkWaylandSurfaceCreateFlagBitsKHR")]
pub struct WaylandSurfaceCreateFlagBitsKHR(u32);
impl WaylandSurfaceCreateFlagBitsKHR {
    #[inline]
    pub const fn empty() -> Self {
        Self(0u32)
    }
    #[inline]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0u32
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}
impl core::ops::BitOr for WaylandSurfaceCreateFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for WaylandSurfaceCreateFlagBitsKHR {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl core::ops::BitAnd for WaylandSurfaceCreateFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl core::ops::BitAndAssign for WaylandSurfaceCreateFlagBitsKHR {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl core::ops::BitXor for WaylandSurfaceCreateFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}
impl core::ops::BitXorAssign for WaylandSurfaceCreateFlagBitsKHR {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl core::ops::Not for WaylandSurfaceCreateFlagBitsKHR {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl core::fmt::Debug for WaylandSurfaceCreateFlagBitsKHR {
    #[allow(unused_mut, unused_variables)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut remaining = self.0;
        if remaining != 0u32 {
            if !first {
                f.write_str(" | ")?;
            }
            write!(f, "{:#x}", remaining)?;
        } else if first {
            f.write_str("(empty)")?;
        }
        Ok(())
    }
}
