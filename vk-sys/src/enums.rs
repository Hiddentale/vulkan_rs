#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkScopeKHR")]
pub struct ScopeKHR(i32);
impl ScopeKHR {
    pub const DEVICE: Self = Self(1i32);
    pub const WORKGROUP: Self = Self(2i32);
    pub const SUBGROUP: Self = Self(3i32);
    pub const QUEUE_FAMILY: Self = Self(5i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ScopeKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            1i32 => f.write_str("DEVICE"),
            2i32 => f.write_str("WORKGROUP"),
            3i32 => f.write_str("SUBGROUP"),
            5i32 => f.write_str("QUEUE_FAMILY"),
            other => write!(f, "{}({})", stringify!(ScopeKHR), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkLatencyMarkerNV")]
pub struct LatencyMarkerNV(i32);
impl LatencyMarkerNV {
    pub const SIMULATION_START: Self = Self(0i32);
    pub const SIMULATION_END: Self = Self(1i32);
    pub const RENDERSUBMIT_START: Self = Self(2i32);
    pub const RENDERSUBMIT_END: Self = Self(3i32);
    pub const PRESENT_START: Self = Self(4i32);
    pub const PRESENT_END: Self = Self(5i32);
    pub const INPUT_SAMPLE: Self = Self(6i32);
    pub const TRIGGER_FLASH: Self = Self(7i32);
    pub const OUT_OF_BAND_RENDERSUBMIT_START: Self = Self(8i32);
    pub const OUT_OF_BAND_RENDERSUBMIT_END: Self = Self(9i32);
    pub const OUT_OF_BAND_PRESENT_START: Self = Self(10i32);
    pub const OUT_OF_BAND_PRESENT_END: Self = Self(11i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for LatencyMarkerNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("SIMULATION_START"),
            1i32 => f.write_str("SIMULATION_END"),
            2i32 => f.write_str("RENDERSUBMIT_START"),
            3i32 => f.write_str("RENDERSUBMIT_END"),
            4i32 => f.write_str("PRESENT_START"),
            5i32 => f.write_str("PRESENT_END"),
            6i32 => f.write_str("INPUT_SAMPLE"),
            7i32 => f.write_str("TRIGGER_FLASH"),
            8i32 => f.write_str("OUT_OF_BAND_RENDERSUBMIT_START"),
            9i32 => f.write_str("OUT_OF_BAND_RENDERSUBMIT_END"),
            10i32 => f.write_str("OUT_OF_BAND_PRESENT_START"),
            11i32 => f.write_str("OUT_OF_BAND_PRESENT_END"),
            other => write!(f, "{}({})", stringify!(LatencyMarkerNV), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkLayerSettingTypeEXT")]
pub struct LayerSettingTypeEXT(i32);
impl LayerSettingTypeEXT {
    pub const BOOL32: Self = Self(0i32);
    pub const INT32: Self = Self(1i32);
    pub const INT64: Self = Self(2i32);
    pub const UINT32: Self = Self(3i32);
    pub const UINT64: Self = Self(4i32);
    pub const FLOAT32: Self = Self(5i32);
    pub const FLOAT64: Self = Self(6i32);
    pub const STRING: Self = Self(7i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for LayerSettingTypeEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("BOOL32"),
            1i32 => f.write_str("INT32"),
            2i32 => f.write_str("INT64"),
            3i32 => f.write_str("UINT32"),
            4i32 => f.write_str("UINT64"),
            5i32 => f.write_str("FLOAT32"),
            6i32 => f.write_str("FLOAT64"),
            7i32 => f.write_str("STRING"),
            other => write!(f, "{}({})", stringify!(LayerSettingTypeEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkCoverageModulationModeNV")]
pub struct CoverageModulationModeNV(i32);
impl CoverageModulationModeNV {
    pub const NONE: Self = Self(0i32);
    pub const RGB: Self = Self(1i32);
    pub const ALPHA: Self = Self(2i32);
    pub const RGBA: Self = Self(3i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for CoverageModulationModeNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("NONE"),
            1i32 => f.write_str("RGB"),
            2i32 => f.write_str("ALPHA"),
            3i32 => f.write_str("RGBA"),
            other => write!(f, "{}({})", stringify!(CoverageModulationModeNV), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkFaultType")]
pub struct FaultType(i32);
impl FaultType {
    pub const INVALID: Self = Self(0i32);
    pub const UNASSIGNED: Self = Self(1i32);
    pub const IMPLEMENTATION: Self = Self(2i32);
    pub const SYSTEM: Self = Self(3i32);
    pub const PHYSICAL_DEVICE: Self = Self(4i32);
    pub const COMMAND_BUFFER_FULL: Self = Self(5i32);
    pub const INVALID_API_USAGE: Self = Self(6i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for FaultType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("INVALID"),
            1i32 => f.write_str("UNASSIGNED"),
            2i32 => f.write_str("IMPLEMENTATION"),
            3i32 => f.write_str("SYSTEM"),
            4i32 => f.write_str("PHYSICAL_DEVICE"),
            5i32 => f.write_str("COMMAND_BUFFER_FULL"),
            6i32 => f.write_str("INVALID_API_USAGE"),
            other => write!(f, "{}({})", stringify!(FaultType), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkQueueGlobalPriority")]
pub struct QueueGlobalPriority(i32);
impl QueueGlobalPriority {
    pub const LOW: Self = Self(128i32);
    pub const MEDIUM: Self = Self(256i32);
    pub const HIGH: Self = Self(512i32);
    pub const REALTIME: Self = Self(1024i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for QueueGlobalPriority {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            128i32 => f.write_str("LOW"),
            256i32 => f.write_str("MEDIUM"),
            512i32 => f.write_str("HIGH"),
            1024i32 => f.write_str("REALTIME"),
            other => write!(f, "{}({})", stringify!(QueueGlobalPriority), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkRayTracingShaderGroupTypeKHR")]
pub struct RayTracingShaderGroupTypeKHR(i32);
impl RayTracingShaderGroupTypeKHR {
    pub const GENERAL: Self = Self(0i32);
    pub const TRIANGLES_HIT_GROUP: Self = Self(1i32);
    pub const PROCEDURAL_HIT_GROUP: Self = Self(2i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for RayTracingShaderGroupTypeKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("GENERAL"),
            1i32 => f.write_str("TRIANGLES_HIT_GROUP"),
            2i32 => f.write_str("PROCEDURAL_HIT_GROUP"),
            other => write!(f, "{}({})", stringify!(RayTracingShaderGroupTypeKHR), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkSamplerAddressMode")]
pub struct SamplerAddressMode(i32);
impl SamplerAddressMode {
    pub const REPEAT: Self = Self(0i32);
    pub const MIRRORED_REPEAT: Self = Self(1i32);
    pub const CLAMP_TO_EDGE: Self = Self(2i32);
    pub const CLAMP_TO_BORDER: Self = Self(3i32);
    pub const MIRROR_CLAMP_TO_EDGE: Self = Self(4i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for SamplerAddressMode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("REPEAT"),
            1i32 => f.write_str("MIRRORED_REPEAT"),
            2i32 => f.write_str("CLAMP_TO_EDGE"),
            3i32 => f.write_str("CLAMP_TO_BORDER"),
            4i32 => f.write_str("MIRROR_CLAMP_TO_EDGE"),
            other => write!(f, "{}({})", stringify!(SamplerAddressMode), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkVertexInputRate")]
pub struct VertexInputRate(i32);
impl VertexInputRate {
    pub const VERTEX: Self = Self(0i32);
    pub const INSTANCE: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for VertexInputRate {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("VERTEX"),
            1i32 => f.write_str("INSTANCE"),
            other => write!(f, "{}({})", stringify!(VertexInputRate), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDiscardRectangleModeEXT")]
pub struct DiscardRectangleModeEXT(i32);
impl DiscardRectangleModeEXT {
    pub const INCLUSIVE: Self = Self(0i32);
    pub const EXCLUSIVE: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DiscardRectangleModeEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("INCLUSIVE"),
            1i32 => f.write_str("EXCLUSIVE"),
            other => write!(f, "{}({})", stringify!(DiscardRectangleModeEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkLogicOp")]
pub struct LogicOp(i32);
impl LogicOp {
    pub const CLEAR: Self = Self(0i32);
    pub const AND: Self = Self(1i32);
    pub const AND_REVERSE: Self = Self(2i32);
    pub const COPY: Self = Self(3i32);
    pub const AND_INVERTED: Self = Self(4i32);
    pub const NO_OP: Self = Self(5i32);
    pub const XOR: Self = Self(6i32);
    pub const OR: Self = Self(7i32);
    pub const NOR: Self = Self(8i32);
    pub const EQUIVALENT: Self = Self(9i32);
    pub const INVERT: Self = Self(10i32);
    pub const OR_REVERSE: Self = Self(11i32);
    pub const COPY_INVERTED: Self = Self(12i32);
    pub const OR_INVERTED: Self = Self(13i32);
    pub const NAND: Self = Self(14i32);
    pub const SET: Self = Self(15i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for LogicOp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("CLEAR"),
            1i32 => f.write_str("AND"),
            2i32 => f.write_str("AND_REVERSE"),
            3i32 => f.write_str("COPY"),
            4i32 => f.write_str("AND_INVERTED"),
            5i32 => f.write_str("NO_OP"),
            6i32 => f.write_str("XOR"),
            7i32 => f.write_str("OR"),
            8i32 => f.write_str("NOR"),
            9i32 => f.write_str("EQUIVALENT"),
            10i32 => f.write_str("INVERT"),
            11i32 => f.write_str("OR_REVERSE"),
            12i32 => f.write_str("COPY_INVERTED"),
            13i32 => f.write_str("OR_INVERTED"),
            14i32 => f.write_str("NAND"),
            15i32 => f.write_str("SET"),
            other => write!(f, "{}({})", stringify!(LogicOp), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkOpticalFlowPerformanceLevelNV")]
pub struct OpticalFlowPerformanceLevelNV(i32);
impl OpticalFlowPerformanceLevelNV {
    pub const UNKNOWN: Self = Self(0i32);
    pub const SLOW: Self = Self(1i32);
    pub const MEDIUM: Self = Self(2i32);
    pub const FAST: Self = Self(3i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for OpticalFlowPerformanceLevelNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("UNKNOWN"),
            1i32 => f.write_str("SLOW"),
            2i32 => f.write_str("MEDIUM"),
            3i32 => f.write_str("FAST"),
            other => {
                write!(f, "{}({})", stringify!(OpticalFlowPerformanceLevelNV), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkColorSpaceKHR")]
pub struct ColorSpaceKHR(i32);
impl ColorSpaceKHR {
    pub const SRGB_NONLINEAR: Self = Self(0i32);
    pub const COLORSPACE_SRGB_NONLINEAR: Self = Self::SRGB_NONLINEAR;
    pub const DISPLAY_P3_NONLINEAR: Self = Self(1000104001i32);
    pub const EXTENDED_SRGB_LINEAR: Self = Self(1000104002i32);
    pub const DISPLAY_P3_LINEAR: Self = Self(1000104003i32);
    pub const DCI_P3_NONLINEAR: Self = Self(1000104004i32);
    pub const BT709_LINEAR: Self = Self(1000104005i32);
    pub const BT709_NONLINEAR: Self = Self(1000104006i32);
    pub const BT2020_LINEAR: Self = Self(1000104007i32);
    pub const HDR10_ST2084: Self = Self(1000104008i32);
    pub const DOLBYVISION: Self = Self(1000104009i32);
    pub const HDR10_HLG: Self = Self(1000104010i32);
    pub const ADOBERGB_LINEAR: Self = Self(1000104011i32);
    pub const ADOBERGB_NONLINEAR: Self = Self(1000104012i32);
    pub const PASS_THROUGH: Self = Self(1000104013i32);
    pub const EXTENDED_SRGB_NONLINEAR: Self = Self(1000104014i32);
    pub const DCI_P3_LINEAR: Self = Self::DISPLAY_P3_LINEAR;
    pub const DISPLAY_NATIVE: Self = Self(1000213000i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ColorSpaceKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("SRGB_NONLINEAR"),
            1000104001i32 => f.write_str("DISPLAY_P3_NONLINEAR"),
            1000104002i32 => f.write_str("EXTENDED_SRGB_LINEAR"),
            1000104003i32 => f.write_str("DISPLAY_P3_LINEAR"),
            1000104004i32 => f.write_str("DCI_P3_NONLINEAR"),
            1000104005i32 => f.write_str("BT709_LINEAR"),
            1000104006i32 => f.write_str("BT709_NONLINEAR"),
            1000104007i32 => f.write_str("BT2020_LINEAR"),
            1000104008i32 => f.write_str("HDR10_ST2084"),
            1000104009i32 => f.write_str("DOLBYVISION"),
            1000104010i32 => f.write_str("HDR10_HLG"),
            1000104011i32 => f.write_str("ADOBERGB_LINEAR"),
            1000104012i32 => f.write_str("ADOBERGB_NONLINEAR"),
            1000104013i32 => f.write_str("PASS_THROUGH"),
            1000104014i32 => f.write_str("EXTENDED_SRGB_NONLINEAR"),
            1000213000i32 => f.write_str("DISPLAY_NATIVE"),
            other => write!(f, "{}({})", stringify!(ColorSpaceKHR), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPerformanceCounterStorageKHR")]
pub struct PerformanceCounterStorageKHR(i32);
impl PerformanceCounterStorageKHR {
    pub const INT32: Self = Self(0i32);
    pub const INT64: Self = Self(1i32);
    pub const UINT32: Self = Self(2i32);
    pub const UINT64: Self = Self(3i32);
    pub const FLOAT32: Self = Self(4i32);
    pub const FLOAT64: Self = Self(5i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PerformanceCounterStorageKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("INT32"),
            1i32 => f.write_str("INT64"),
            2i32 => f.write_str("UINT32"),
            3i32 => f.write_str("UINT64"),
            4i32 => f.write_str("FLOAT32"),
            5i32 => f.write_str("FLOAT64"),
            other => write!(f, "{}({})", stringify!(PerformanceCounterStorageKHR), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkBlendOverlapEXT")]
pub struct BlendOverlapEXT(i32);
impl BlendOverlapEXT {
    pub const UNCORRELATED: Self = Self(0i32);
    pub const DISJOINT: Self = Self(1i32);
    pub const CONJOINT: Self = Self(2i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for BlendOverlapEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("UNCORRELATED"),
            1i32 => f.write_str("DISJOINT"),
            2i32 => f.write_str("CONJOINT"),
            other => write!(f, "{}({})", stringify!(BlendOverlapEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDataGraphPipelineSessionBindPointARM")]
pub struct DataGraphPipelineSessionBindPointARM(i32);
impl DataGraphPipelineSessionBindPointARM {
    pub const TRANSIENT: Self = Self(0i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DataGraphPipelineSessionBindPointARM {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("TRANSIENT"),
            other => {
                write!(
                    f, "{}({})", stringify!(DataGraphPipelineSessionBindPointARM), other
                )
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPerformanceCounterScopeKHR")]
pub struct PerformanceCounterScopeKHR(i32);
impl PerformanceCounterScopeKHR {
    pub const COMMAND_BUFFER: Self = Self(0i32);
    pub const RENDER_PASS: Self = Self(1i32);
    pub const COMMAND: Self = Self(2i32);
    pub const QUERY_SCOPE_COMMAND_BUFFER: Self = Self::COMMAND_BUFFER;
    pub const QUERY_SCOPE_RENDER_PASS: Self = Self::RENDER_PASS;
    pub const QUERY_SCOPE_COMMAND: Self = Self::COMMAND;
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PerformanceCounterScopeKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("COMMAND_BUFFER"),
            1i32 => f.write_str("RENDER_PASS"),
            2i32 => f.write_str("COMMAND"),
            other => write!(f, "{}({})", stringify!(PerformanceCounterScopeKHR), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPipelineMatchControl")]
pub struct PipelineMatchControl(i32);
impl PipelineMatchControl {
    pub const APPLICATION_UUID_EXACT_MATCH: Self = Self(0i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PipelineMatchControl {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("APPLICATION_UUID_EXACT_MATCH"),
            other => write!(f, "{}({})", stringify!(PipelineMatchControl), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkVideoEncodeTuningModeKHR")]
pub struct VideoEncodeTuningModeKHR(i32);
impl VideoEncodeTuningModeKHR {
    pub const DEFAULT: Self = Self(0i32);
    pub const HIGH_QUALITY: Self = Self(1i32);
    pub const LOW_LATENCY: Self = Self(2i32);
    pub const ULTRA_LOW_LATENCY: Self = Self(3i32);
    pub const LOSSLESS: Self = Self(4i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for VideoEncodeTuningModeKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("DEFAULT"),
            1i32 => f.write_str("HIGH_QUALITY"),
            2i32 => f.write_str("LOW_LATENCY"),
            3i32 => f.write_str("ULTRA_LOW_LATENCY"),
            4i32 => f.write_str("LOSSLESS"),
            other => write!(f, "{}({})", stringify!(VideoEncodeTuningModeKHR), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkSemaphoreType")]
pub struct SemaphoreType(i32);
impl SemaphoreType {
    pub const BINARY: Self = Self(0i32);
    pub const TIMELINE: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for SemaphoreType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("BINARY"),
            1i32 => f.write_str("TIMELINE"),
            other => write!(f, "{}({})", stringify!(SemaphoreType), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkAccelerationStructureMemoryRequirementsTypeNV")]
pub struct AccelerationStructureMemoryRequirementsTypeNV(i32);
impl AccelerationStructureMemoryRequirementsTypeNV {
    pub const OBJECT: Self = Self(0i32);
    pub const BUILD_SCRATCH: Self = Self(1i32);
    pub const UPDATE_SCRATCH: Self = Self(2i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for AccelerationStructureMemoryRequirementsTypeNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("OBJECT"),
            1i32 => f.write_str("BUILD_SCRATCH"),
            2i32 => f.write_str("UPDATE_SCRATCH"),
            other => {
                write!(
                    f, "{}({})",
                    stringify!(AccelerationStructureMemoryRequirementsTypeNV), other
                )
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkAccelerationStructureMotionInstanceTypeNV")]
pub struct AccelerationStructureMotionInstanceTypeNV(i32);
impl AccelerationStructureMotionInstanceTypeNV {
    pub const STATIC: Self = Self(0i32);
    pub const MATRIX_MOTION: Self = Self(1i32);
    pub const SRT_MOTION: Self = Self(2i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for AccelerationStructureMotionInstanceTypeNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("STATIC"),
            1i32 => f.write_str("MATRIX_MOTION"),
            2i32 => f.write_str("SRT_MOTION"),
            other => {
                write!(
                    f, "{}({})", stringify!(AccelerationStructureMotionInstanceTypeNV),
                    other
                )
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkAntiLagStageAMD")]
pub struct AntiLagStageAMD(i32);
impl AntiLagStageAMD {
    pub const INPUT: Self = Self(0i32);
    pub const PRESENT: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for AntiLagStageAMD {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("INPUT"),
            1i32 => f.write_str("PRESENT"),
            other => write!(f, "{}({})", stringify!(AntiLagStageAMD), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPipelineRobustnessImageBehavior")]
pub struct PipelineRobustnessImageBehavior(i32);
impl PipelineRobustnessImageBehavior {
    pub const DEVICE_DEFAULT: Self = Self(0i32);
    pub const DISABLED: Self = Self(1i32);
    pub const ROBUST_IMAGE_ACCESS: Self = Self(2i32);
    pub const ROBUST_IMAGE_ACCESS_2: Self = Self(3i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PipelineRobustnessImageBehavior {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("DEVICE_DEFAULT"),
            1i32 => f.write_str("DISABLED"),
            2i32 => f.write_str("ROBUST_IMAGE_ACCESS"),
            3i32 => f.write_str("ROBUST_IMAGE_ACCESS_2"),
            other => {
                write!(f, "{}({})", stringify!(PipelineRobustnessImageBehavior), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkShaderFloatControlsIndependence")]
pub struct ShaderFloatControlsIndependence(i32);
impl ShaderFloatControlsIndependence {
    pub const _32_BIT_ONLY: Self = Self(0i32);
    pub const ALL: Self = Self(1i32);
    pub const NONE: Self = Self(2i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ShaderFloatControlsIndependence {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("_32_BIT_ONLY"),
            1i32 => f.write_str("ALL"),
            2i32 => f.write_str("NONE"),
            other => {
                write!(f, "{}({})", stringify!(ShaderFloatControlsIndependence), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkValidationFeatureEnableEXT")]
pub struct ValidationFeatureEnableEXT(i32);
impl ValidationFeatureEnableEXT {
    pub const GPU_ASSISTED: Self = Self(0i32);
    pub const GPU_ASSISTED_RESERVE_BINDING_SLOT: Self = Self(1i32);
    pub const BEST_PRACTICES: Self = Self(2i32);
    pub const DEBUG_PRINTF: Self = Self(3i32);
    pub const SYNCHRONIZATION_VALIDATION: Self = Self(4i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ValidationFeatureEnableEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("GPU_ASSISTED"),
            1i32 => f.write_str("GPU_ASSISTED_RESERVE_BINDING_SLOT"),
            2i32 => f.write_str("BEST_PRACTICES"),
            3i32 => f.write_str("DEBUG_PRINTF"),
            4i32 => f.write_str("SYNCHRONIZATION_VALIDATION"),
            other => write!(f, "{}({})", stringify!(ValidationFeatureEnableEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkFilter")]
pub struct Filter(i32);
impl Filter {
    pub const NEAREST: Self = Self(0i32);
    pub const LINEAR: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for Filter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("NEAREST"),
            1i32 => f.write_str("LINEAR"),
            other => write!(f, "{}({})", stringify!(Filter), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDisplayEventTypeEXT")]
pub struct DisplayEventTypeEXT(i32);
impl DisplayEventTypeEXT {
    pub const FIRST_PIXEL_OUT: Self = Self(0i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DisplayEventTypeEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("FIRST_PIXEL_OUT"),
            other => write!(f, "{}({})", stringify!(DisplayEventTypeEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkQueryResultStatusKHR")]
pub struct QueryResultStatusKHR(i32);
impl QueryResultStatusKHR {
    pub const ERROR: Self = Self(-1i32);
    pub const NOT_READY: Self = Self(0i32);
    pub const COMPLETE: Self = Self(1i32);
    pub const INSUFFICIENT_BITSTREAM_BUFFER_RANGE: Self = Self(-1000299000i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for QueryResultStatusKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            -1i32 => f.write_str("ERROR"),
            0i32 => f.write_str("NOT_READY"),
            1i32 => f.write_str("COMPLETE"),
            -1000299000i32 => f.write_str("INSUFFICIENT_BITSTREAM_BUFFER_RANGE"),
            other => write!(f, "{}({})", stringify!(QueryResultStatusKHR), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPerformanceOverrideTypeINTEL")]
pub struct PerformanceOverrideTypeINTEL(i32);
impl PerformanceOverrideTypeINTEL {
    pub const NULL_HARDWARE: Self = Self(0i32);
    pub const FLUSH_GPU_CACHES: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PerformanceOverrideTypeINTEL {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("NULL_HARDWARE"),
            1i32 => f.write_str("FLUSH_GPU_CACHES"),
            other => write!(f, "{}({})", stringify!(PerformanceOverrideTypeINTEL), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPerformanceParameterTypeINTEL")]
pub struct PerformanceParameterTypeINTEL(i32);
impl PerformanceParameterTypeINTEL {
    pub const HW_COUNTERS_SUPPORTED: Self = Self(0i32);
    pub const STREAM_MARKER_VALID_BITS: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PerformanceParameterTypeINTEL {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("HW_COUNTERS_SUPPORTED"),
            1i32 => f.write_str("STREAM_MARKER_VALID_BITS"),
            other => {
                write!(f, "{}({})", stringify!(PerformanceParameterTypeINTEL), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkRasterizationOrderAMD")]
pub struct RasterizationOrderAMD(i32);
impl RasterizationOrderAMD {
    pub const STRICT: Self = Self(0i32);
    pub const RELAXED: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for RasterizationOrderAMD {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("STRICT"),
            1i32 => f.write_str("RELAXED"),
            other => write!(f, "{}({})", stringify!(RasterizationOrderAMD), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkStructureType")]
pub struct StructureType(i32);
impl StructureType {
    pub const APPLICATION_INFO: Self = Self(0i32);
    pub const INSTANCE_CREATE_INFO: Self = Self(1i32);
    pub const DEVICE_QUEUE_CREATE_INFO: Self = Self(2i32);
    pub const DEVICE_CREATE_INFO: Self = Self(3i32);
    pub const SUBMIT_INFO: Self = Self(4i32);
    pub const MEMORY_ALLOCATE_INFO: Self = Self(5i32);
    pub const MAPPED_MEMORY_RANGE: Self = Self(6i32);
    pub const BIND_SPARSE_INFO: Self = Self(7i32);
    pub const FENCE_CREATE_INFO: Self = Self(8i32);
    pub const SEMAPHORE_CREATE_INFO: Self = Self(9i32);
    pub const EVENT_CREATE_INFO: Self = Self(10i32);
    pub const QUERY_POOL_CREATE_INFO: Self = Self(11i32);
    pub const BUFFER_CREATE_INFO: Self = Self(12i32);
    pub const BUFFER_VIEW_CREATE_INFO: Self = Self(13i32);
    pub const IMAGE_CREATE_INFO: Self = Self(14i32);
    pub const IMAGE_VIEW_CREATE_INFO: Self = Self(15i32);
    pub const SHADER_MODULE_CREATE_INFO: Self = Self(16i32);
    pub const PIPELINE_CACHE_CREATE_INFO: Self = Self(17i32);
    pub const PIPELINE_SHADER_STAGE_CREATE_INFO: Self = Self(18i32);
    pub const PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO: Self = Self(19i32);
    pub const PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO: Self = Self(20i32);
    pub const PIPELINE_TESSELLATION_STATE_CREATE_INFO: Self = Self(21i32);
    pub const PIPELINE_VIEWPORT_STATE_CREATE_INFO: Self = Self(22i32);
    pub const PIPELINE_RASTERIZATION_STATE_CREATE_INFO: Self = Self(23i32);
    pub const PIPELINE_MULTISAMPLE_STATE_CREATE_INFO: Self = Self(24i32);
    pub const PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO: Self = Self(25i32);
    pub const PIPELINE_COLOR_BLEND_STATE_CREATE_INFO: Self = Self(26i32);
    pub const PIPELINE_DYNAMIC_STATE_CREATE_INFO: Self = Self(27i32);
    pub const GRAPHICS_PIPELINE_CREATE_INFO: Self = Self(28i32);
    pub const COMPUTE_PIPELINE_CREATE_INFO: Self = Self(29i32);
    pub const PIPELINE_LAYOUT_CREATE_INFO: Self = Self(30i32);
    pub const SAMPLER_CREATE_INFO: Self = Self(31i32);
    pub const DESCRIPTOR_SET_LAYOUT_CREATE_INFO: Self = Self(32i32);
    pub const DESCRIPTOR_POOL_CREATE_INFO: Self = Self(33i32);
    pub const DESCRIPTOR_SET_ALLOCATE_INFO: Self = Self(34i32);
    pub const WRITE_DESCRIPTOR_SET: Self = Self(35i32);
    pub const COPY_DESCRIPTOR_SET: Self = Self(36i32);
    pub const FRAMEBUFFER_CREATE_INFO: Self = Self(37i32);
    pub const RENDER_PASS_CREATE_INFO: Self = Self(38i32);
    pub const COMMAND_POOL_CREATE_INFO: Self = Self(39i32);
    pub const COMMAND_BUFFER_ALLOCATE_INFO: Self = Self(40i32);
    pub const COMMAND_BUFFER_INHERITANCE_INFO: Self = Self(41i32);
    pub const COMMAND_BUFFER_BEGIN_INFO: Self = Self(42i32);
    pub const RENDER_PASS_BEGIN_INFO: Self = Self(43i32);
    pub const BUFFER_MEMORY_BARRIER: Self = Self(44i32);
    pub const IMAGE_MEMORY_BARRIER: Self = Self(45i32);
    pub const MEMORY_BARRIER: Self = Self(46i32);
    pub const LOADER_INSTANCE_CREATE_INFO: Self = Self(47i32);
    pub const LOADER_DEVICE_CREATE_INFO: Self = Self(48i32);
    pub const BIND_BUFFER_MEMORY_INFO: Self = Self(1000157000i32);
    pub const BIND_IMAGE_MEMORY_INFO: Self = Self(1000157001i32);
    pub const MEMORY_DEDICATED_REQUIREMENTS: Self = Self(1000127000i32);
    pub const MEMORY_DEDICATED_ALLOCATE_INFO: Self = Self(1000127001i32);
    pub const MEMORY_ALLOCATE_FLAGS_INFO: Self = Self(1000060000i32);
    pub const DEVICE_GROUP_COMMAND_BUFFER_BEGIN_INFO: Self = Self(1000060004i32);
    pub const DEVICE_GROUP_SUBMIT_INFO: Self = Self(1000060005i32);
    pub const DEVICE_GROUP_BIND_SPARSE_INFO: Self = Self(1000060006i32);
    pub const BIND_BUFFER_MEMORY_DEVICE_GROUP_INFO: Self = Self(1000060013i32);
    pub const BIND_IMAGE_MEMORY_DEVICE_GROUP_INFO: Self = Self(1000060014i32);
    pub const PHYSICAL_DEVICE_GROUP_PROPERTIES: Self = Self(1000070000i32);
    pub const DEVICE_GROUP_DEVICE_CREATE_INFO: Self = Self(1000070001i32);
    pub const BUFFER_MEMORY_REQUIREMENTS_INFO_2: Self = Self(1000146000i32);
    pub const IMAGE_MEMORY_REQUIREMENTS_INFO_2: Self = Self(1000146001i32);
    pub const IMAGE_SPARSE_MEMORY_REQUIREMENTS_INFO_2: Self = Self(1000146002i32);
    pub const MEMORY_REQUIREMENTS_2: Self = Self(1000146003i32);
    pub const SPARSE_IMAGE_MEMORY_REQUIREMENTS_2: Self = Self(1000146004i32);
    pub const PHYSICAL_DEVICE_FEATURES_2: Self = Self(1000059000i32);
    pub const PHYSICAL_DEVICE_PROPERTIES_2: Self = Self(1000059001i32);
    pub const FORMAT_PROPERTIES_2: Self = Self(1000059002i32);
    pub const IMAGE_FORMAT_PROPERTIES_2: Self = Self(1000059003i32);
    pub const PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2: Self = Self(1000059004i32);
    pub const QUEUE_FAMILY_PROPERTIES_2: Self = Self(1000059005i32);
    pub const PHYSICAL_DEVICE_MEMORY_PROPERTIES_2: Self = Self(1000059006i32);
    pub const SPARSE_IMAGE_FORMAT_PROPERTIES_2: Self = Self(1000059007i32);
    pub const PHYSICAL_DEVICE_SPARSE_IMAGE_FORMAT_INFO_2: Self = Self(1000059008i32);
    pub const IMAGE_VIEW_USAGE_CREATE_INFO: Self = Self(1000117002i32);
    pub const PROTECTED_SUBMIT_INFO: Self = Self(1000145000i32);
    pub const PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES: Self = Self(1000145001i32);
    pub const PHYSICAL_DEVICE_PROTECTED_MEMORY_PROPERTIES: Self = Self(1000145002i32);
    pub const DEVICE_QUEUE_INFO_2: Self = Self(1000145003i32);
    pub const PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO: Self = Self(1000071000i32);
    pub const EXTERNAL_IMAGE_FORMAT_PROPERTIES: Self = Self(1000071001i32);
    pub const PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO: Self = Self(1000071002i32);
    pub const EXTERNAL_BUFFER_PROPERTIES: Self = Self(1000071003i32);
    pub const PHYSICAL_DEVICE_ID_PROPERTIES: Self = Self(1000071004i32);
    pub const EXTERNAL_MEMORY_BUFFER_CREATE_INFO: Self = Self(1000072000i32);
    pub const EXTERNAL_MEMORY_IMAGE_CREATE_INFO: Self = Self(1000072001i32);
    pub const EXPORT_MEMORY_ALLOCATE_INFO: Self = Self(1000072002i32);
    pub const PHYSICAL_DEVICE_EXTERNAL_FENCE_INFO: Self = Self(1000112000i32);
    pub const EXTERNAL_FENCE_PROPERTIES: Self = Self(1000112001i32);
    pub const EXPORT_FENCE_CREATE_INFO: Self = Self(1000113000i32);
    pub const EXPORT_SEMAPHORE_CREATE_INFO: Self = Self(1000077000i32);
    pub const PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO: Self = Self(1000076000i32);
    pub const EXTERNAL_SEMAPHORE_PROPERTIES: Self = Self(1000076001i32);
    pub const PHYSICAL_DEVICE_SUBGROUP_PROPERTIES: Self = Self(1000094000i32);
    pub const PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES: Self = Self(1000083000i32);
    pub const PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES: Self = Self(1000120000i32);
    pub const PHYSICAL_DEVICE_VARIABLE_POINTER_FEATURES: Self = Self::PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES;
    pub const DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO: Self = Self(1000085000i32);
    pub const PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES: Self = Self(1000168000i32);
    pub const DESCRIPTOR_SET_LAYOUT_SUPPORT: Self = Self(1000168001i32);
    pub const SAMPLER_YCBCR_CONVERSION_CREATE_INFO: Self = Self(1000156000i32);
    pub const SAMPLER_YCBCR_CONVERSION_INFO: Self = Self(1000156001i32);
    pub const BIND_IMAGE_PLANE_MEMORY_INFO: Self = Self(1000156002i32);
    pub const IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO: Self = Self(1000156003i32);
    pub const PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES: Self = Self(
        1000156004i32,
    );
    pub const SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES: Self = Self(
        1000156005i32,
    );
    pub const DEVICE_GROUP_RENDER_PASS_BEGIN_INFO: Self = Self(1000060003i32);
    pub const PHYSICAL_DEVICE_POINT_CLIPPING_PROPERTIES: Self = Self(1000117000i32);
    pub const RENDER_PASS_INPUT_ATTACHMENT_ASPECT_CREATE_INFO: Self = Self(
        1000117001i32,
    );
    pub const PIPELINE_TESSELLATION_DOMAIN_ORIGIN_STATE_CREATE_INFO: Self = Self(
        1000117003i32,
    );
    pub const RENDER_PASS_MULTIVIEW_CREATE_INFO: Self = Self(1000053000i32);
    pub const PHYSICAL_DEVICE_MULTIVIEW_FEATURES: Self = Self(1000053001i32);
    pub const PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES: Self = Self(1000053002i32);
    pub const PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES: Self = Self(
        1000063000i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_DRAW_PARAMETER_FEATURES: Self = Self::PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES;
    pub const PHYSICAL_DEVICE_DRIVER_PROPERTIES: Self = Self(1000196000i32);
    pub const PHYSICAL_DEVICE_VULKAN_1_1_FEATURES: Self = Self(49i32);
    pub const PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES: Self = Self(50i32);
    pub const PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: Self = Self(51i32);
    pub const PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES: Self = Self(52i32);
    pub const IMAGE_FORMAT_LIST_CREATE_INFO: Self = Self(1000147000i32);
    pub const PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES: Self = Self(1000211000i32);
    pub const PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES: Self = Self(1000261000i32);
    pub const PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES: Self = Self(1000207000i32);
    pub const PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES: Self = Self(1000207001i32);
    pub const SEMAPHORE_TYPE_CREATE_INFO: Self = Self(1000207002i32);
    pub const TIMELINE_SEMAPHORE_SUBMIT_INFO: Self = Self(1000207003i32);
    pub const SEMAPHORE_WAIT_INFO: Self = Self(1000207004i32);
    pub const SEMAPHORE_SIGNAL_INFO: Self = Self(1000207005i32);
    pub const PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES: Self = Self(1000257000i32);
    pub const BUFFER_DEVICE_ADDRESS_INFO: Self = Self(1000244001i32);
    pub const BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO: Self = Self(1000257002i32);
    pub const MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO: Self = Self(1000257003i32);
    pub const DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO: Self = Self(1000257004i32);
    pub const PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES: Self = Self(1000177000i32);
    pub const PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES: Self = Self(1000180000i32);
    pub const PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES: Self = Self(1000082000i32);
    pub const PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES: Self = Self(1000197000i32);
    pub const DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO: Self = Self(
        1000161000i32,
    );
    pub const PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES: Self = Self(1000161001i32);
    pub const PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES: Self = Self(1000161002i32);
    pub const DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO: Self = Self(
        1000161003i32,
    );
    pub const DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_LAYOUT_SUPPORT: Self = Self(
        1000161004i32,
    );
    pub const PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES: Self = Self(1000221000i32);
    pub const PHYSICAL_DEVICE_SAMPLER_FILTER_MINMAX_PROPERTIES: Self = Self(
        1000130000i32,
    );
    pub const SAMPLER_REDUCTION_MODE_CREATE_INFO: Self = Self(1000130001i32);
    pub const PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES: Self = Self(
        1000253000i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES: Self = Self(
        1000175000i32,
    );
    pub const ATTACHMENT_DESCRIPTION_2: Self = Self(1000109000i32);
    pub const ATTACHMENT_REFERENCE_2: Self = Self(1000109001i32);
    pub const SUBPASS_DESCRIPTION_2: Self = Self(1000109002i32);
    pub const SUBPASS_DEPENDENCY_2: Self = Self(1000109003i32);
    pub const RENDER_PASS_CREATE_INFO_2: Self = Self(1000109004i32);
    pub const SUBPASS_BEGIN_INFO: Self = Self(1000109005i32);
    pub const SUBPASS_END_INFO: Self = Self(1000109006i32);
    pub const PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES: Self = Self(
        1000199000i32,
    );
    pub const SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE: Self = Self(1000199001i32);
    pub const IMAGE_STENCIL_USAGE_CREATE_INFO: Self = Self(1000246000i32);
    pub const PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES: Self = Self(1000108000i32);
    pub const FRAMEBUFFER_ATTACHMENTS_CREATE_INFO: Self = Self(1000108001i32);
    pub const FRAMEBUFFER_ATTACHMENT_IMAGE_INFO: Self = Self(1000108002i32);
    pub const RENDER_PASS_ATTACHMENT_BEGIN_INFO: Self = Self(1000108003i32);
    pub const PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES: Self = Self(
        1000241000i32,
    );
    pub const ATTACHMENT_REFERENCE_STENCIL_LAYOUT: Self = Self(1000241001i32);
    pub const ATTACHMENT_DESCRIPTION_STENCIL_LAYOUT: Self = Self(1000241002i32);
    pub const PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: Self = Self(53i32);
    pub const PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES: Self = Self(54i32);
    pub const PHYSICAL_DEVICE_TOOL_PROPERTIES: Self = Self(1000245000i32);
    pub const PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES: Self = Self(1000295000i32);
    pub const DEVICE_PRIVATE_DATA_CREATE_INFO: Self = Self(1000295001i32);
    pub const PRIVATE_DATA_SLOT_CREATE_INFO: Self = Self(1000295002i32);
    pub const MEMORY_BARRIER_2: Self = Self(1000314000i32);
    pub const BUFFER_MEMORY_BARRIER_2: Self = Self(1000314001i32);
    pub const IMAGE_MEMORY_BARRIER_2: Self = Self(1000314002i32);
    pub const DEPENDENCY_INFO: Self = Self(1000314003i32);
    pub const SUBMIT_INFO_2: Self = Self(1000314004i32);
    pub const SEMAPHORE_SUBMIT_INFO: Self = Self(1000314005i32);
    pub const COMMAND_BUFFER_SUBMIT_INFO: Self = Self(1000314006i32);
    pub const PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES: Self = Self(1000314007i32);
    pub const COPY_BUFFER_INFO_2: Self = Self(1000337000i32);
    pub const COPY_IMAGE_INFO_2: Self = Self(1000337001i32);
    pub const COPY_BUFFER_TO_IMAGE_INFO_2: Self = Self(1000337002i32);
    pub const COPY_IMAGE_TO_BUFFER_INFO_2: Self = Self(1000337003i32);
    pub const BUFFER_COPY_2: Self = Self(1000337006i32);
    pub const IMAGE_COPY_2: Self = Self(1000337007i32);
    pub const BUFFER_IMAGE_COPY_2: Self = Self(1000337009i32);
    pub const PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES: Self = Self(
        1000066000i32,
    );
    pub const FORMAT_PROPERTIES_3: Self = Self(1000360000i32);
    pub const PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES: Self = Self(1000413000i32);
    pub const PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES: Self = Self(1000413001i32);
    pub const DEVICE_BUFFER_MEMORY_REQUIREMENTS: Self = Self(1000413002i32);
    pub const DEVICE_IMAGE_MEMORY_REQUIREMENTS: Self = Self(1000413003i32);
    pub const PIPELINE_CREATION_FEEDBACK_CREATE_INFO: Self = Self(1000192000i32);
    pub const PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES: Self = Self(
        1000215000i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES: Self = Self(
        1000276000i32,
    );
    pub const PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES: Self = Self(
        1000297000i32,
    );
    pub const PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES: Self = Self(
        1000325000i32,
    );
    pub const PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES: Self = Self(1000335000i32);
    pub const PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES: Self = Self(
        1000225000i32,
    );
    pub const PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO: Self = Self(
        1000225001i32,
    );
    pub const PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES: Self = Self(1000225002i32);
    pub const PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES: Self = Self(1000138000i32);
    pub const PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_PROPERTIES: Self = Self(
        1000138001i32,
    );
    pub const WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK: Self = Self(1000138002i32);
    pub const DESCRIPTOR_POOL_INLINE_UNIFORM_BLOCK_CREATE_INFO: Self = Self(
        1000138003i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES: Self = Self(
        1000280000i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES: Self = Self(
        1000280001i32,
    );
    pub const PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_PROPERTIES: Self = Self(
        1000281001i32,
    );
    pub const BLIT_IMAGE_INFO_2: Self = Self(1000337004i32);
    pub const RESOLVE_IMAGE_INFO_2: Self = Self(1000337005i32);
    pub const IMAGE_BLIT_2: Self = Self(1000337008i32);
    pub const IMAGE_RESOLVE_2: Self = Self(1000337010i32);
    pub const RENDERING_INFO: Self = Self(1000044000i32);
    pub const RENDERING_ATTACHMENT_INFO: Self = Self(1000044001i32);
    pub const PIPELINE_RENDERING_CREATE_INFO: Self = Self(1000044002i32);
    pub const PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES: Self = Self(1000044003i32);
    pub const COMMAND_BUFFER_INHERITANCE_RENDERING_INFO: Self = Self(1000044004i32);
    pub const PHYSICAL_DEVICE_VULKAN_1_4_FEATURES: Self = Self(55i32);
    pub const PHYSICAL_DEVICE_VULKAN_1_4_PROPERTIES: Self = Self(56i32);
    pub const DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO: Self = Self(1000174000i32);
    pub const PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES: Self = Self(1000388000i32);
    pub const QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES: Self = Self(1000388001i32);
    pub const PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES: Self = Self(1000265000i32);
    pub const MEMORY_MAP_INFO: Self = Self(1000271000i32);
    pub const MEMORY_UNMAP_INFO: Self = Self(1000271001i32);
    pub const PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES: Self = Self(1000470000i32);
    pub const PHYSICAL_DEVICE_MAINTENANCE_5_PROPERTIES: Self = Self(1000470001i32);
    pub const DEVICE_IMAGE_SUBRESOURCE_INFO: Self = Self(1000470004i32);
    pub const SUBRESOURCE_LAYOUT_2: Self = Self(1000338002i32);
    pub const IMAGE_SUBRESOURCE_2: Self = Self(1000338003i32);
    pub const BUFFER_USAGE_FLAGS_2_CREATE_INFO: Self = Self(1000470006i32);
    pub const PHYSICAL_DEVICE_MAINTENANCE_6_FEATURES: Self = Self(1000545000i32);
    pub const PHYSICAL_DEVICE_MAINTENANCE_6_PROPERTIES: Self = Self(1000545001i32);
    pub const BIND_MEMORY_STATUS: Self = Self(1000545002i32);
    pub const PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES: Self = Self(1000270000i32);
    pub const PHYSICAL_DEVICE_HOST_IMAGE_COPY_PROPERTIES: Self = Self(1000270001i32);
    pub const MEMORY_TO_IMAGE_COPY: Self = Self(1000270002i32);
    pub const IMAGE_TO_MEMORY_COPY: Self = Self(1000270003i32);
    pub const COPY_IMAGE_TO_MEMORY_INFO: Self = Self(1000270004i32);
    pub const COPY_MEMORY_TO_IMAGE_INFO: Self = Self(1000270005i32);
    pub const HOST_IMAGE_LAYOUT_TRANSITION_INFO: Self = Self(1000270006i32);
    pub const COPY_IMAGE_TO_IMAGE_INFO: Self = Self(1000270007i32);
    pub const SUBRESOURCE_HOST_MEMCPY_SIZE: Self = Self(1000270008i32);
    pub const HOST_IMAGE_COPY_DEVICE_PERFORMANCE_QUERY: Self = Self(1000270009i32);
    pub const PHYSICAL_DEVICE_SHADER_SUBGROUP_ROTATE_FEATURES: Self = Self(
        1000416000i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES: Self = Self(
        1000528000i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_EXPECT_ASSUME_FEATURES: Self = Self(1000544000i32);
    pub const PIPELINE_CREATE_FLAGS_2_CREATE_INFO: Self = Self(1000470005i32);
    pub const PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES: Self = Self(1000080000i32);
    pub const BIND_DESCRIPTOR_SETS_INFO: Self = Self(1000545003i32);
    pub const PUSH_CONSTANTS_INFO: Self = Self(1000545004i32);
    pub const PUSH_DESCRIPTOR_SET_INFO: Self = Self(1000545005i32);
    pub const PUSH_DESCRIPTOR_SET_WITH_TEMPLATE_INFO: Self = Self(1000545006i32);
    pub const PHYSICAL_DEVICE_PIPELINE_PROTECTED_ACCESS_FEATURES: Self = Self(
        1000466000i32,
    );
    pub const PIPELINE_ROBUSTNESS_CREATE_INFO: Self = Self(1000068000i32);
    pub const PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES: Self = Self(1000068001i32);
    pub const PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_PROPERTIES: Self = Self(1000068002i32);
    pub const PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES: Self = Self(1000259000i32);
    pub const PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO: Self = Self(1000259001i32);
    pub const PHYSICAL_DEVICE_LINE_RASTERIZATION_PROPERTIES: Self = Self(1000259002i32);
    pub const PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_PROPERTIES: Self = Self(
        1000525000i32,
    );
    pub const PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO: Self = Self(
        1000190001i32,
    );
    pub const PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES: Self = Self(
        1000190002i32,
    );
    pub const RENDERING_AREA_INFO: Self = Self(1000470003i32);
    pub const PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES: Self = Self(
        1000232000i32,
    );
    pub const RENDERING_ATTACHMENT_LOCATION_INFO: Self = Self(1000232001i32);
    pub const RENDERING_INPUT_ATTACHMENT_INDEX_INFO: Self = Self(1000232002i32);
    pub const PHYSICAL_DEVICE_VULKAN_SC_1_0_FEATURES: Self = Self(1000298000i32);
    pub const PHYSICAL_DEVICE_VULKAN_SC_1_0_PROPERTIES: Self = Self(1000298001i32);
    pub const DEVICE_OBJECT_RESERVATION_CREATE_INFO: Self = Self(1000298002i32);
    pub const COMMAND_POOL_MEMORY_RESERVATION_CREATE_INFO: Self = Self(1000298003i32);
    pub const COMMAND_POOL_MEMORY_CONSUMPTION: Self = Self(1000298004i32);
    pub const PIPELINE_POOL_SIZE: Self = Self(1000298005i32);
    pub const FAULT_DATA: Self = Self(1000298007i32);
    pub const FAULT_CALLBACK_INFO: Self = Self(1000298008i32);
    pub const PIPELINE_OFFLINE_CREATE_INFO: Self = Self(1000298010i32);
    pub const SWAPCHAIN_CREATE_INFO: Self = Self(1000001000i32);
    pub const PRESENT_INFO: Self = Self(1000001001i32);
    pub const DEVICE_GROUP_PRESENT_CAPABILITIES: Self = Self(1000060007i32);
    pub const IMAGE_SWAPCHAIN_CREATE_INFO: Self = Self(1000060008i32);
    pub const BIND_IMAGE_MEMORY_SWAPCHAIN_INFO: Self = Self(1000060009i32);
    pub const ACQUIRE_NEXT_IMAGE_INFO: Self = Self(1000060010i32);
    pub const DEVICE_GROUP_PRESENT_INFO: Self = Self(1000060011i32);
    pub const DEVICE_GROUP_SWAPCHAIN_CREATE_INFO: Self = Self(1000060012i32);
    pub const DISPLAY_MODE_CREATE_INFO: Self = Self(1000002000i32);
    pub const DISPLAY_SURFACE_CREATE_INFO: Self = Self(1000002001i32);
    pub const DISPLAY_PRESENT_INFO: Self = Self(1000003000i32);
    pub const XLIB_SURFACE_CREATE_INFO: Self = Self(1000004000i32);
    pub const XCB_SURFACE_CREATE_INFO: Self = Self(1000005000i32);
    pub const WAYLAND_SURFACE_CREATE_INFO: Self = Self(1000006000i32);
    pub const ANDROID_SURFACE_CREATE_INFO: Self = Self(1000008000i32);
    pub const WIN32_SURFACE_CREATE_INFO: Self = Self(1000009000i32);
    pub const DEBUG_REPORT_CALLBACK_CREATE_INFO: Self = Self(1000011000i32);
    pub const DEBUG_REPORT_CREATE_INFO: Self = Self::DEBUG_REPORT_CALLBACK_CREATE_INFO;
    pub const PIPELINE_RASTERIZATION_STATE_RASTERIZATION_ORDER: Self = Self(
        1000018000i32,
    );
    pub const DEBUG_MARKER_OBJECT_NAME_INFO: Self = Self(1000022000i32);
    pub const DEBUG_MARKER_OBJECT_TAG_INFO: Self = Self(1000022001i32);
    pub const DEBUG_MARKER_MARKER_INFO: Self = Self(1000022002i32);
    pub const VIDEO_PROFILE_INFO: Self = Self(1000023000i32);
    pub const VIDEO_CAPABILITIES: Self = Self(1000023001i32);
    pub const VIDEO_PICTURE_RESOURCE_INFO: Self = Self(1000023002i32);
    pub const VIDEO_SESSION_MEMORY_REQUIREMENTS: Self = Self(1000023003i32);
    pub const BIND_VIDEO_SESSION_MEMORY_INFO: Self = Self(1000023004i32);
    pub const VIDEO_SESSION_CREATE_INFO: Self = Self(1000023005i32);
    pub const VIDEO_SESSION_PARAMETERS_CREATE_INFO: Self = Self(1000023006i32);
    pub const VIDEO_SESSION_PARAMETERS_UPDATE_INFO: Self = Self(1000023007i32);
    pub const VIDEO_BEGIN_CODING_INFO: Self = Self(1000023008i32);
    pub const VIDEO_END_CODING_INFO: Self = Self(1000023009i32);
    pub const VIDEO_CODING_CONTROL_INFO: Self = Self(1000023010i32);
    pub const VIDEO_REFERENCE_SLOT_INFO: Self = Self(1000023011i32);
    pub const QUEUE_FAMILY_VIDEO_PROPERTIES: Self = Self(1000023012i32);
    pub const VIDEO_PROFILE_LIST_INFO: Self = Self(1000023013i32);
    pub const PHYSICAL_DEVICE_VIDEO_FORMAT_INFO: Self = Self(1000023014i32);
    pub const VIDEO_FORMAT_PROPERTIES: Self = Self(1000023015i32);
    pub const QUEUE_FAMILY_QUERY_RESULT_STATUS_PROPERTIES: Self = Self(1000023016i32);
    pub const VIDEO_DECODE_INFO: Self = Self(1000024000i32);
    pub const VIDEO_DECODE_CAPABILITIES: Self = Self(1000024001i32);
    pub const VIDEO_DECODE_USAGE_INFO: Self = Self(1000024002i32);
    pub const DEDICATED_ALLOCATION_IMAGE_CREATE_INFO: Self = Self(1000026000i32);
    pub const DEDICATED_ALLOCATION_BUFFER_CREATE_INFO: Self = Self(1000026001i32);
    pub const DEDICATED_ALLOCATION_MEMORY_ALLOCATE_INFO: Self = Self(1000026002i32);
    pub const PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES: Self = Self(1000028000i32);
    pub const PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_PROPERTIES: Self = Self(1000028001i32);
    pub const PIPELINE_RASTERIZATION_STATE_STREAM_CREATE_INFO: Self = Self(
        1000028002i32,
    );
    pub const CU_MODULE_CREATE_INFO: Self = Self(1000029000i32);
    pub const CU_FUNCTION_CREATE_INFO: Self = Self(1000029001i32);
    pub const CU_LAUNCH_INFO: Self = Self(1000029002i32);
    pub const CU_MODULE_TEXTURING_MODE_CREATE_INFO: Self = Self(1000029004i32);
    pub const IMAGE_VIEW_HANDLE_INFO: Self = Self(1000030000i32);
    pub const IMAGE_VIEW_ADDRESS_PROPERTIES: Self = Self(1000030001i32);
    pub const VIDEO_ENCODE_H264_CAPABILITIES: Self = Self(1000038000i32);
    pub const VIDEO_ENCODE_H264_SESSION_PARAMETERS_CREATE_INFO: Self = Self(
        1000038001i32,
    );
    pub const VIDEO_ENCODE_H264_SESSION_PARAMETERS_ADD_INFO: Self = Self(1000038002i32);
    pub const VIDEO_ENCODE_H264_PICTURE_INFO: Self = Self(1000038003i32);
    pub const VIDEO_ENCODE_H264_DPB_SLOT_INFO: Self = Self(1000038004i32);
    pub const VIDEO_ENCODE_H264_NALU_SLICE_INFO: Self = Self(1000038005i32);
    pub const VIDEO_ENCODE_H264_GOP_REMAINING_FRAME_INFO: Self = Self(1000038006i32);
    pub const VIDEO_ENCODE_H264_PROFILE_INFO: Self = Self(1000038007i32);
    pub const VIDEO_ENCODE_H264_RATE_CONTROL_INFO: Self = Self(1000038008i32);
    pub const VIDEO_ENCODE_H264_RATE_CONTROL_LAYER_INFO: Self = Self(1000038009i32);
    pub const VIDEO_ENCODE_H264_SESSION_CREATE_INFO: Self = Self(1000038010i32);
    pub const VIDEO_ENCODE_H264_QUALITY_LEVEL_PROPERTIES: Self = Self(1000038011i32);
    pub const VIDEO_ENCODE_H264_SESSION_PARAMETERS_GET_INFO: Self = Self(1000038012i32);
    pub const VIDEO_ENCODE_H264_SESSION_PARAMETERS_FEEDBACK_INFO: Self = Self(
        1000038013i32,
    );
    pub const VIDEO_ENCODE_H265_CAPABILITIES: Self = Self(1000039000i32);
    pub const VIDEO_ENCODE_H265_SESSION_PARAMETERS_CREATE_INFO: Self = Self(
        1000039001i32,
    );
    pub const VIDEO_ENCODE_H265_SESSION_PARAMETERS_ADD_INFO: Self = Self(1000039002i32);
    pub const VIDEO_ENCODE_H265_PICTURE_INFO: Self = Self(1000039003i32);
    pub const VIDEO_ENCODE_H265_DPB_SLOT_INFO: Self = Self(1000039004i32);
    pub const VIDEO_ENCODE_H265_NALU_SLICE_SEGMENT_INFO: Self = Self(1000039005i32);
    pub const VIDEO_ENCODE_H265_GOP_REMAINING_FRAME_INFO: Self = Self(1000039006i32);
    pub const VIDEO_ENCODE_H265_PROFILE_INFO: Self = Self(1000039007i32);
    pub const VIDEO_ENCODE_H265_RATE_CONTROL_INFO: Self = Self(1000039009i32);
    pub const VIDEO_ENCODE_H265_RATE_CONTROL_LAYER_INFO: Self = Self(1000039010i32);
    pub const VIDEO_ENCODE_H265_SESSION_CREATE_INFO: Self = Self(1000039011i32);
    pub const VIDEO_ENCODE_H265_QUALITY_LEVEL_PROPERTIES: Self = Self(1000039012i32);
    pub const VIDEO_ENCODE_H265_SESSION_PARAMETERS_GET_INFO: Self = Self(1000039013i32);
    pub const VIDEO_ENCODE_H265_SESSION_PARAMETERS_FEEDBACK_INFO: Self = Self(
        1000039014i32,
    );
    pub const VIDEO_DECODE_H264_CAPABILITIES: Self = Self(1000040000i32);
    pub const VIDEO_DECODE_H264_PICTURE_INFO: Self = Self(1000040001i32);
    pub const VIDEO_DECODE_H264_PROFILE_INFO: Self = Self(1000040003i32);
    pub const VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO: Self = Self(
        1000040004i32,
    );
    pub const VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO: Self = Self(1000040005i32);
    pub const VIDEO_DECODE_H264_DPB_SLOT_INFO: Self = Self(1000040006i32);
    pub const TEXTURE_LOD_GATHER_FORMAT_PROPERTIES: Self = Self(1000041000i32);
    pub const STREAM_DESCRIPTOR_SURFACE_CREATE_INFO: Self = Self(1000049000i32);
    pub const PHYSICAL_DEVICE_CORNER_SAMPLED_IMAGE_FEATURES: Self = Self(1000050000i32);
    pub const PRIVATE_VENDOR_INFO_PLACEHOLDER_OFFSET_0: Self = Self(1000051000i32);
    pub const IMPORT_MEMORY_WIN32_HANDLE_INFO: Self = Self(1000057000i32);
    pub const EXPORT_MEMORY_WIN32_HANDLE_INFO: Self = Self(1000057001i32);
    pub const WIN32_KEYED_MUTEX_ACQUIRE_RELEASE_INFO: Self = Self(1000058000i32);
    pub const VALIDATION_FLAGS: Self = Self(1000061000i32);
    pub const VI_SURFACE_CREATE_INFO: Self = Self(1000062000i32);
    pub const IMAGE_VIEW_ASTC_DECODE_MODE: Self = Self(1000067000i32);
    pub const PHYSICAL_DEVICE_ASTC_DECODE_FEATURES: Self = Self(1000067001i32);
    pub const MEMORY_WIN32_HANDLE_PROPERTIES: Self = Self(1000073002i32);
    pub const MEMORY_GET_WIN32_HANDLE_INFO: Self = Self(1000073003i32);
    pub const IMPORT_MEMORY_FD_INFO: Self = Self(1000074000i32);
    pub const MEMORY_FD_PROPERTIES: Self = Self(1000074001i32);
    pub const MEMORY_GET_FD_INFO: Self = Self(1000074002i32);
    pub const IMPORT_SEMAPHORE_WIN32_HANDLE_INFO: Self = Self(1000078000i32);
    pub const EXPORT_SEMAPHORE_WIN32_HANDLE_INFO: Self = Self(1000078001i32);
    pub const D3D12_FENCE_SUBMIT_INFO: Self = Self(1000078002i32);
    pub const SEMAPHORE_GET_WIN32_HANDLE_INFO: Self = Self(1000078003i32);
    pub const IMPORT_SEMAPHORE_FD_INFO: Self = Self(1000079000i32);
    pub const SEMAPHORE_GET_FD_INFO: Self = Self(1000079001i32);
    pub const COMMAND_BUFFER_INHERITANCE_CONDITIONAL_RENDERING_INFO: Self = Self(
        1000081000i32,
    );
    pub const PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES: Self = Self(1000081001i32);
    pub const CONDITIONAL_RENDERING_BEGIN_INFO: Self = Self(1000081002i32);
    pub const PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES: Self = Self::PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    pub const PRESENT_REGIONS: Self = Self(1000084000i32);
    pub const PIPELINE_VIEWPORT_W_SCALING_STATE_CREATE_INFO: Self = Self(1000087000i32);
    pub const SURFACE_CAPABILITIES_2: Self = Self(1000090000i32);
    pub const SURFACE_CAPABILITIES2: Self = Self::SURFACE_CAPABILITIES_2;
    pub const DISPLAY_POWER_INFO: Self = Self(1000091000i32);
    pub const DEVICE_EVENT_INFO: Self = Self(1000091001i32);
    pub const DISPLAY_EVENT_INFO: Self = Self(1000091002i32);
    pub const SWAPCHAIN_COUNTER_CREATE_INFO: Self = Self(1000091003i32);
    pub const PRESENT_TIMES_INFO: Self = Self(1000092000i32);
    pub const PHYSICAL_DEVICE_MULTIVIEW_PER_VIEW_ATTRIBUTES_PROPERTIES: Self = Self(
        1000097000i32,
    );
    pub const MULTIVIEW_PER_VIEW_ATTRIBUTES_INFO: Self = Self(1000044009i32);
    pub const PIPELINE_VIEWPORT_SWIZZLE_STATE_CREATE_INFO: Self = Self(1000098000i32);
    pub const PHYSICAL_DEVICE_DISCARD_RECTANGLE_PROPERTIES: Self = Self(1000099000i32);
    pub const PIPELINE_DISCARD_RECTANGLE_STATE_CREATE_INFO: Self = Self(1000099001i32);
    pub const PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES: Self = Self(
        1000101000i32,
    );
    pub const PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO: Self = Self(
        1000101001i32,
    );
    pub const PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES: Self = Self(1000102000i32);
    pub const PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO: Self = Self(
        1000102001i32,
    );
    pub const HDR_METADATA: Self = Self(1000105000i32);
    pub const PHYSICAL_DEVICE_RELAXED_LINE_RASTERIZATION_FEATURES: Self = Self(
        1000110000i32,
    );
    pub const SHARED_PRESENT_SURFACE_CAPABILITIES: Self = Self(1000111000i32);
    pub const IMPORT_FENCE_WIN32_HANDLE_INFO: Self = Self(1000114000i32);
    pub const EXPORT_FENCE_WIN32_HANDLE_INFO: Self = Self(1000114001i32);
    pub const FENCE_GET_WIN32_HANDLE_INFO: Self = Self(1000114002i32);
    pub const IMPORT_FENCE_FD_INFO: Self = Self(1000115000i32);
    pub const FENCE_GET_FD_INFO: Self = Self(1000115001i32);
    pub const PHYSICAL_DEVICE_PERFORMANCE_QUERY_FEATURES: Self = Self(1000116000i32);
    pub const PHYSICAL_DEVICE_PERFORMANCE_QUERY_PROPERTIES: Self = Self(1000116001i32);
    pub const QUERY_POOL_PERFORMANCE_CREATE_INFO: Self = Self(1000116002i32);
    pub const PERFORMANCE_QUERY_SUBMIT_INFO: Self = Self(1000116003i32);
    pub const ACQUIRE_PROFILING_LOCK_INFO: Self = Self(1000116004i32);
    pub const PERFORMANCE_COUNTER: Self = Self(1000116005i32);
    pub const PERFORMANCE_COUNTER_DESCRIPTION: Self = Self(1000116006i32);
    pub const PERFORMANCE_QUERY_RESERVATION_INFO: Self = Self(1000116007i32);
    pub const PHYSICAL_DEVICE_SURFACE_INFO_2: Self = Self(1000119000i32);
    pub const SURFACE_FORMAT_2: Self = Self(1000119002i32);
    pub const DISPLAY_PROPERTIES_2: Self = Self(1000121000i32);
    pub const DISPLAY_PLANE_PROPERTIES_2: Self = Self(1000121001i32);
    pub const DISPLAY_MODE_PROPERTIES_2: Self = Self(1000121002i32);
    pub const DISPLAY_PLANE_INFO_2: Self = Self(1000121003i32);
    pub const DISPLAY_PLANE_CAPABILITIES_2: Self = Self(1000121004i32);
    pub const IOS_SURFACE_CREATE_INFO: Self = Self(1000122000i32);
    pub const MACOS_SURFACE_CREATE_INFO: Self = Self(1000123000i32);
    pub const DEBUG_UTILS_OBJECT_NAME_INFO: Self = Self(1000128000i32);
    pub const DEBUG_UTILS_OBJECT_TAG_INFO: Self = Self(1000128001i32);
    pub const DEBUG_UTILS_LABEL: Self = Self(1000128002i32);
    pub const DEBUG_UTILS_MESSENGER_CALLBACK_DATA: Self = Self(1000128003i32);
    pub const DEBUG_UTILS_MESSENGER_CREATE_INFO: Self = Self(1000128004i32);
    pub const ANDROID_HARDWARE_BUFFER_USAGE: Self = Self(1000129000i32);
    pub const ANDROID_HARDWARE_BUFFER_PROPERTIES: Self = Self(1000129001i32);
    pub const ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES: Self = Self(1000129002i32);
    pub const IMPORT_ANDROID_HARDWARE_BUFFER_INFO: Self = Self(1000129003i32);
    pub const MEMORY_GET_ANDROID_HARDWARE_BUFFER_INFO: Self = Self(1000129004i32);
    pub const EXTERNAL_FORMAT: Self = Self(1000129005i32);
    pub const ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_2: Self = Self(1000129006i32);
    pub const PHYSICAL_DEVICE_SHADER_ENQUEUE_FEATURES: Self = Self(1000134000i32);
    pub const PHYSICAL_DEVICE_SHADER_ENQUEUE_PROPERTIES: Self = Self(1000134001i32);
    pub const EXECUTION_GRAPH_PIPELINE_SCRATCH_SIZE: Self = Self(1000134002i32);
    pub const EXECUTION_GRAPH_PIPELINE_CREATE_INFO: Self = Self(1000134003i32);
    pub const PIPELINE_SHADER_STAGE_NODE_CREATE_INFO: Self = Self(1000134004i32);
    pub const TEXEL_BUFFER_DESCRIPTOR_INFO: Self = Self(1000135000i32);
    pub const IMAGE_DESCRIPTOR_INFO: Self = Self(1000135001i32);
    pub const RESOURCE_DESCRIPTOR_INFO: Self = Self(1000135002i32);
    pub const BIND_HEAP_INFO: Self = Self(1000135003i32);
    pub const PUSH_DATA_INFO: Self = Self(1000135004i32);
    pub const DESCRIPTOR_SET_AND_BINDING_MAPPING: Self = Self(1000135005i32);
    pub const SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO: Self = Self(1000135006i32);
    pub const OPAQUE_CAPTURE_DATA_CREATE_INFO: Self = Self(1000135007i32);
    pub const PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES: Self = Self(1000135008i32);
    pub const PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES: Self = Self(1000135009i32);
    pub const COMMAND_BUFFER_INHERITANCE_DESCRIPTOR_HEAP_INFO: Self = Self(
        1000135010i32,
    );
    pub const SAMPLER_CUSTOM_BORDER_COLOR_INDEX_CREATE_INFO: Self = Self(1000135011i32);
    pub const INDIRECT_COMMANDS_LAYOUT_PUSH_DATA_TOKEN: Self = Self(1000135012i32);
    pub const SUBSAMPLED_IMAGE_FORMAT_PROPERTIES: Self = Self(1000135013i32);
    pub const PHYSICAL_DEVICE_DESCRIPTOR_HEAP_TENSOR_PROPERTIES: Self = Self(
        1000135014i32,
    );
    pub const ATTACHMENT_SAMPLE_COUNT_INFO: Self = Self(1000044008i32);
    pub const PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES: Self = Self(1000141000i32);
    pub const SAMPLE_LOCATIONS_INFO: Self = Self(1000143000i32);
    pub const RENDER_PASS_SAMPLE_LOCATIONS_BEGIN_INFO: Self = Self(1000143001i32);
    pub const PIPELINE_SAMPLE_LOCATIONS_STATE_CREATE_INFO: Self = Self(1000143002i32);
    pub const PHYSICAL_DEVICE_SAMPLE_LOCATIONS_PROPERTIES: Self = Self(1000143003i32);
    pub const MULTISAMPLE_PROPERTIES: Self = Self(1000143004i32);
    pub const PHYSICAL_DEVICE_BLEND_OPERATION_ADVANCED_FEATURES: Self = Self(
        1000148000i32,
    );
    pub const PHYSICAL_DEVICE_BLEND_OPERATION_ADVANCED_PROPERTIES: Self = Self(
        1000148001i32,
    );
    pub const PIPELINE_COLOR_BLEND_ADVANCED_STATE_CREATE_INFO: Self = Self(
        1000148002i32,
    );
    pub const PIPELINE_COVERAGE_TO_COLOR_STATE_CREATE_INFO: Self = Self(1000149000i32);
    pub const WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE: Self = Self(1000150007i32);
    pub const ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO: Self = Self(1000150000i32);
    pub const ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO: Self = Self(1000150002i32);
    pub const ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA: Self = Self(1000150003i32);
    pub const ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA: Self = Self(1000150004i32);
    pub const ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA: Self = Self(1000150005i32);
    pub const ACCELERATION_STRUCTURE_GEOMETRY: Self = Self(1000150006i32);
    pub const ACCELERATION_STRUCTURE_VERSION_INFO: Self = Self(1000150009i32);
    pub const COPY_ACCELERATION_STRUCTURE_INFO: Self = Self(1000150010i32);
    pub const COPY_ACCELERATION_STRUCTURE_TO_MEMORY_INFO: Self = Self(1000150011i32);
    pub const COPY_MEMORY_TO_ACCELERATION_STRUCTURE_INFO: Self = Self(1000150012i32);
    pub const PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES: Self = Self(
        1000150013i32,
    );
    pub const PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES: Self = Self(
        1000150014i32,
    );
    pub const ACCELERATION_STRUCTURE_CREATE_INFO: Self = Self(1000150017i32);
    pub const ACCELERATION_STRUCTURE_BUILD_SIZES_INFO: Self = Self(1000150020i32);
    pub const PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES: Self = Self(1000347000i32);
    pub const PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES: Self = Self(
        1000347001i32,
    );
    pub const RAY_TRACING_PIPELINE_CREATE_INFO: Self = Self(1000150015i32);
    pub const RAY_TRACING_SHADER_GROUP_CREATE_INFO: Self = Self(1000150016i32);
    pub const RAY_TRACING_PIPELINE_INTERFACE_CREATE_INFO: Self = Self(1000150018i32);
    pub const PHYSICAL_DEVICE_RAY_QUERY_FEATURES: Self = Self(1000348013i32);
    pub const PIPELINE_COVERAGE_MODULATION_STATE_CREATE_INFO: Self = Self(1000152000i32);
    pub const PHYSICAL_DEVICE_SHADER_SM_BUILTINS_FEATURES: Self = Self(1000154000i32);
    pub const PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES: Self = Self(1000154001i32);
    pub const DRM_FORMAT_MODIFIER_PROPERTIES_LIST: Self = Self(1000158000i32);
    pub const PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO: Self = Self(1000158002i32);
    pub const IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO: Self = Self(1000158003i32);
    pub const IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO: Self = Self(1000158004i32);
    pub const IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES: Self = Self(1000158005i32);
    pub const DRM_FORMAT_MODIFIER_PROPERTIES_LIST_2: Self = Self(1000158006i32);
    pub const VALIDATION_CACHE_CREATE_INFO: Self = Self(1000160000i32);
    pub const SHADER_MODULE_VALIDATION_CACHE_CREATE_INFO: Self = Self(1000160001i32);
    pub const PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES: Self = Self(1000163000i32);
    pub const PHYSICAL_DEVICE_PORTABILITY_SUBSET_PROPERTIES: Self = Self(1000163001i32);
    pub const PIPELINE_VIEWPORT_SHADING_RATE_IMAGE_STATE_CREATE_INFO: Self = Self(
        1000164000i32,
    );
    pub const PHYSICAL_DEVICE_SHADING_RATE_IMAGE_FEATURES: Self = Self(1000164001i32);
    pub const PHYSICAL_DEVICE_SHADING_RATE_IMAGE_PROPERTIES: Self = Self(1000164002i32);
    pub const PIPELINE_VIEWPORT_COARSE_SAMPLE_ORDER_STATE_CREATE_INFO: Self = Self(
        1000164005i32,
    );
    pub const GEOMETRY: Self = Self(1000165003i32);
    pub const GEOMETRY_TRIANGLES: Self = Self(1000165004i32);
    pub const GEOMETRY_AABB: Self = Self(1000165005i32);
    pub const BIND_ACCELERATION_STRUCTURE_MEMORY_INFO: Self = Self(1000165006i32);
    pub const ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO: Self = Self(
        1000165008i32,
    );
    pub const PHYSICAL_DEVICE_RAY_TRACING_PROPERTIES: Self = Self(1000165009i32);
    pub const ACCELERATION_STRUCTURE_INFO: Self = Self(1000165012i32);
    pub const PHYSICAL_DEVICE_REPRESENTATIVE_FRAGMENT_TEST_FEATURES: Self = Self(
        1000166000i32,
    );
    pub const PIPELINE_REPRESENTATIVE_FRAGMENT_TEST_STATE_CREATE_INFO: Self = Self(
        1000166001i32,
    );
    pub const PHYSICAL_DEVICE_IMAGE_VIEW_IMAGE_FORMAT_INFO: Self = Self(1000170000i32);
    pub const FILTER_CUBIC_IMAGE_VIEW_IMAGE_FORMAT_PROPERTIES: Self = Self(
        1000170001i32,
    );
    pub const PHYSICAL_DEVICE_COOPERATIVE_MATRIX_CONVERSION_FEATURES: Self = Self(
        1000172000i32,
    );
    pub const IMPORT_MEMORY_HOST_POINTER_INFO: Self = Self(1000178000i32);
    pub const MEMORY_HOST_POINTER_PROPERTIES: Self = Self(1000178001i32);
    pub const PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES: Self = Self(
        1000178002i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES: Self = Self(1000181000i32);
    pub const PIPELINE_COMPILER_CONTROL_CREATE_INFO: Self = Self(1000183000i32);
    pub const PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES: Self = Self(1000185000i32);
    pub const VIDEO_DECODE_H265_CAPABILITIES: Self = Self(1000187000i32);
    pub const VIDEO_DECODE_H265_SESSION_PARAMETERS_CREATE_INFO: Self = Self(
        1000187001i32,
    );
    pub const VIDEO_DECODE_H265_SESSION_PARAMETERS_ADD_INFO: Self = Self(1000187002i32);
    pub const VIDEO_DECODE_H265_PROFILE_INFO: Self = Self(1000187003i32);
    pub const VIDEO_DECODE_H265_PICTURE_INFO: Self = Self(1000187004i32);
    pub const VIDEO_DECODE_H265_DPB_SLOT_INFO: Self = Self(1000187005i32);
    pub const DEVICE_MEMORY_OVERALLOCATION_CREATE_INFO: Self = Self(1000189000i32);
    pub const PRESENT_FRAME_TOKEN: Self = Self(1000191000i32);
    pub const PHYSICAL_DEVICE_MESH_SHADER_FEATURES: Self = Self(1000202000i32);
    pub const PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES: Self = Self(1000202001i32);
    pub const PHYSICAL_DEVICE_SHADER_IMAGE_FOOTPRINT_FEATURES: Self = Self(
        1000204000i32,
    );
    pub const PIPELINE_VIEWPORT_EXCLUSIVE_SCISSOR_STATE_CREATE_INFO: Self = Self(
        1000205000i32,
    );
    pub const PHYSICAL_DEVICE_EXCLUSIVE_SCISSOR_FEATURES: Self = Self(1000205002i32);
    pub const CHECKPOINT_DATA: Self = Self(1000206000i32);
    pub const QUEUE_FAMILY_CHECKPOINT_PROPERTIES: Self = Self(1000206001i32);
    pub const QUEUE_FAMILY_CHECKPOINT_PROPERTIES_2: Self = Self(1000314008i32);
    pub const CHECKPOINT_DATA_2: Self = Self(1000314009i32);
    pub const PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES: Self = Self(1000208000i32);
    pub const SWAPCHAIN_TIMING_PROPERTIES: Self = Self(1000208001i32);
    pub const SWAPCHAIN_TIME_DOMAIN_PROPERTIES: Self = Self(1000208002i32);
    pub const PRESENT_TIMINGS_INFO: Self = Self(1000208003i32);
    pub const PRESENT_TIMING_INFO: Self = Self(1000208004i32);
    pub const PAST_PRESENTATION_TIMING_INFO: Self = Self(1000208005i32);
    pub const PAST_PRESENTATION_TIMING_PROPERTIES: Self = Self(1000208006i32);
    pub const PAST_PRESENTATION_TIMING: Self = Self(1000208007i32);
    pub const PRESENT_TIMING_SURFACE_CAPABILITIES: Self = Self(1000208008i32);
    pub const SWAPCHAIN_CALIBRATED_TIMESTAMP_INFO: Self = Self(1000208009i32);
    pub const PHYSICAL_DEVICE_SHADER_INTEGER_FUNCTIONS_2_FEATURES: Self = Self(
        1000209000i32,
    );
    pub const QUERY_POOL_PERFORMANCE_QUERY_CREATE_INFO: Self = Self(1000210000i32);
    pub const INITIALIZE_PERFORMANCE_API_INFO: Self = Self(1000210001i32);
    pub const PERFORMANCE_MARKER_INFO: Self = Self(1000210002i32);
    pub const PERFORMANCE_STREAM_MARKER_INFO: Self = Self(1000210003i32);
    pub const PERFORMANCE_OVERRIDE_INFO: Self = Self(1000210004i32);
    pub const PERFORMANCE_CONFIGURATION_ACQUIRE_INFO: Self = Self(1000210005i32);
    pub const PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES: Self = Self(1000212000i32);
    pub const DISPLAY_NATIVE_HDR_SURFACE_CAPABILITIES: Self = Self(1000213000i32);
    pub const SWAPCHAIN_DISPLAY_NATIVE_HDR_CREATE_INFO: Self = Self(1000213001i32);
    pub const IMAGEPIPE_SURFACE_CREATE_INFO: Self = Self(1000214000i32);
    pub const METAL_SURFACE_CREATE_INFO: Self = Self(1000217000i32);
    pub const PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_FEATURES: Self = Self(1000218000i32);
    pub const PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_PROPERTIES: Self = Self(
        1000218001i32,
    );
    pub const RENDER_PASS_FRAGMENT_DENSITY_MAP_CREATE_INFO: Self = Self(1000218002i32);
    pub const RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT_INFO: Self = Self(1000044007i32);
    pub const FRAGMENT_SHADING_RATE_ATTACHMENT_INFO: Self = Self(1000226000i32);
    pub const PIPELINE_FRAGMENT_SHADING_RATE_STATE_CREATE_INFO: Self = Self(
        1000226001i32,
    );
    pub const PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_PROPERTIES: Self = Self(
        1000226002i32,
    );
    pub const PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES: Self = Self(1000226003i32);
    pub const PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE: Self = Self(1000226004i32);
    pub const RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO: Self = Self(
        1000044006i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_2: Self = Self(1000227000i32);
    pub const PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES: Self = Self(1000229000i32);
    pub const PHYSICAL_DEVICE_SHADER_CONSTANT_DATA_FEATURES: Self = Self(1000231000i32);
    pub const PHYSICAL_DEVICE_SHADER_ABORT_FEATURES: Self = Self(1000233000i32);
    pub const DEVICE_FAULT_SHADER_ABORT_MESSAGE_INFO: Self = Self(1000233001i32);
    pub const PHYSICAL_DEVICE_SHADER_ABORT_PROPERTIES: Self = Self(1000233002i32);
    pub const PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES: Self = Self(
        1000234000i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_QUAD_CONTROL_FEATURES: Self = Self(1000235000i32);
    pub const PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES: Self = Self(1000237000i32);
    pub const PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES: Self = Self(1000238000i32);
    pub const MEMORY_PRIORITY_ALLOCATE_INFO: Self = Self(1000238001i32);
    pub const SURFACE_PROTECTED_CAPABILITIES: Self = Self(1000239000i32);
    pub const PHYSICAL_DEVICE_DEDICATED_ALLOCATION_IMAGE_ALIASING_FEATURES: Self = Self(
        1000240000i32,
    );
    pub const PHYSICAL_DEVICE_BUFFER_ADDRESS_FEATURES: Self = Self::PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    pub const BUFFER_DEVICE_ADDRESS_CREATE_INFO: Self = Self(1000244002i32);
    pub const VALIDATION_FEATURES: Self = Self(1000247000i32);
    pub const PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES: Self = Self(1000248000i32);
    pub const PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES: Self = Self(1000249000i32);
    pub const COOPERATIVE_MATRIX_PROPERTIES: Self = Self(1000249001i32);
    pub const PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES: Self = Self(1000249002i32);
    pub const PHYSICAL_DEVICE_COVERAGE_REDUCTION_MODE_FEATURES: Self = Self(
        1000250000i32,
    );
    pub const PIPELINE_COVERAGE_REDUCTION_STATE_CREATE_INFO: Self = Self(1000250001i32);
    pub const FRAMEBUFFER_MIXED_SAMPLES_COMBINATION: Self = Self(1000250002i32);
    pub const PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES: Self = Self(
        1000251000i32,
    );
    pub const PHYSICAL_DEVICE_YCBCR_IMAGE_ARRAYS_FEATURES: Self = Self(1000252000i32);
    pub const PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES: Self = Self(1000254000i32);
    pub const PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO: Self = Self(
        1000254001i32,
    );
    pub const PHYSICAL_DEVICE_PROVOKING_VERTEX_PROPERTIES: Self = Self(1000254002i32);
    pub const SURFACE_FULL_SCREEN_EXCLUSIVE_INFO: Self = Self(1000255000i32);
    pub const SURFACE_CAPABILITIES_FULL_SCREEN_EXCLUSIVE: Self = Self(1000255002i32);
    pub const SURFACE_FULL_SCREEN_EXCLUSIVE_WIN32_INFO: Self = Self(1000255001i32);
    pub const HEADLESS_SURFACE_CREATE_INFO: Self = Self(1000256000i32);
    pub const PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES: Self = Self(1000260000i32);
    pub const PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES: Self = Self(
        1000267000i32,
    );
    pub const PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES: Self = Self(
        1000269000i32,
    );
    pub const PIPELINE_INFO: Self = Self(1000269001i32);
    pub const PIPELINE_EXECUTABLE_PROPERTIES: Self = Self(1000269002i32);
    pub const PIPELINE_EXECUTABLE_INFO: Self = Self(1000269003i32);
    pub const PIPELINE_EXECUTABLE_STATISTIC: Self = Self(1000269004i32);
    pub const PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION: Self = Self(1000269005i32);
    pub const PHYSICAL_DEVICE_MAP_MEMORY_PLACED_FEATURES: Self = Self(1000272000i32);
    pub const PHYSICAL_DEVICE_MAP_MEMORY_PLACED_PROPERTIES: Self = Self(1000272001i32);
    pub const MEMORY_MAP_PLACED_INFO: Self = Self(1000272002i32);
    pub const PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES: Self = Self(1000273000i32);
    pub const PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_PROPERTIES: Self = Self(
        1000277000i32,
    );
    pub const GRAPHICS_SHADER_GROUP_CREATE_INFO: Self = Self(1000277001i32);
    pub const GRAPHICS_PIPELINE_SHADER_GROUPS_CREATE_INFO: Self = Self(1000277002i32);
    pub const INDIRECT_COMMANDS_LAYOUT_TOKEN: Self = Self(1000277003i32);
    pub const INDIRECT_COMMANDS_LAYOUT_CREATE_INFO: Self = Self(1000277004i32);
    pub const GENERATED_COMMANDS_INFO: Self = Self(1000277005i32);
    pub const GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO: Self = Self(1000277006i32);
    pub const PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES: Self = Self(
        1000277007i32,
    );
    pub const PHYSICAL_DEVICE_INHERITED_VIEWPORT_SCISSOR_FEATURES: Self = Self(
        1000278000i32,
    );
    pub const COMMAND_BUFFER_INHERITANCE_VIEWPORT_SCISSOR_INFO: Self = Self(
        1000278001i32,
    );
    pub const PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_FEATURES: Self = Self(
        1000281000i32,
    );
    pub const COMMAND_BUFFER_INHERITANCE_RENDER_PASS_TRANSFORM_INFO: Self = Self(
        1000282000i32,
    );
    pub const RENDER_PASS_TRANSFORM_BEGIN_INFO: Self = Self(1000282001i32);
    pub const PHYSICAL_DEVICE_DEPTH_BIAS_CONTROL_FEATURES: Self = Self(1000283000i32);
    pub const DEPTH_BIAS_INFO: Self = Self(1000283001i32);
    pub const DEPTH_BIAS_REPRESENTATION_INFO: Self = Self(1000283002i32);
    pub const PHYSICAL_DEVICE_DEVICE_MEMORY_REPORT_FEATURES: Self = Self(1000284000i32);
    pub const DEVICE_DEVICE_MEMORY_REPORT_CREATE_INFO: Self = Self(1000284001i32);
    pub const DEVICE_MEMORY_REPORT_CALLBACK_DATA: Self = Self(1000284002i32);
    pub const SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO: Self = Self(1000287000i32);
    pub const PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_PROPERTIES: Self = Self(1000287001i32);
    pub const PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES: Self = Self(1000287002i32);
    pub const PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_3D_FEATURES: Self = Self(
        1000288000i32,
    );
    pub const PIPELINE_LIBRARY_CREATE_INFO: Self = Self(1000290000i32);
    pub const PHYSICAL_DEVICE_PRESENT_BARRIER_FEATURES: Self = Self(1000292000i32);
    pub const SURFACE_CAPABILITIES_PRESENT_BARRIER: Self = Self(1000292001i32);
    pub const SWAPCHAIN_PRESENT_BARRIER_CREATE_INFO: Self = Self(1000292002i32);
    pub const PRESENT_ID: Self = Self(1000294000i32);
    pub const PHYSICAL_DEVICE_PRESENT_ID_FEATURES: Self = Self(1000294001i32);
    pub const VIDEO_ENCODE_INFO: Self = Self(1000299000i32);
    pub const VIDEO_ENCODE_RATE_CONTROL_INFO: Self = Self(1000299001i32);
    pub const VIDEO_ENCODE_RATE_CONTROL_LAYER_INFO: Self = Self(1000299002i32);
    pub const VIDEO_ENCODE_CAPABILITIES: Self = Self(1000299003i32);
    pub const VIDEO_ENCODE_USAGE_INFO: Self = Self(1000299004i32);
    pub const QUERY_POOL_VIDEO_ENCODE_FEEDBACK_CREATE_INFO: Self = Self(1000299005i32);
    pub const PHYSICAL_DEVICE_VIDEO_ENCODE_QUALITY_LEVEL_INFO: Self = Self(
        1000299006i32,
    );
    pub const VIDEO_ENCODE_QUALITY_LEVEL_PROPERTIES: Self = Self(1000299007i32);
    pub const VIDEO_ENCODE_QUALITY_LEVEL_INFO: Self = Self(1000299008i32);
    pub const VIDEO_ENCODE_SESSION_PARAMETERS_GET_INFO: Self = Self(1000299009i32);
    pub const VIDEO_ENCODE_SESSION_PARAMETERS_FEEDBACK_INFO: Self = Self(1000299010i32);
    pub const PHYSICAL_DEVICE_DIAGNOSTICS_CONFIG_FEATURES: Self = Self(1000300000i32);
    pub const DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO: Self = Self(1000300001i32);
    pub const CUDA_MODULE_CREATE_INFO: Self = Self(1000307000i32);
    pub const CUDA_FUNCTION_CREATE_INFO: Self = Self(1000307001i32);
    pub const CUDA_LAUNCH_INFO: Self = Self(1000307002i32);
    pub const PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES: Self = Self(1000307003i32);
    pub const PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_PROPERTIES: Self = Self(1000307004i32);
    pub const REFRESH_OBJECT_LIST: Self = Self(1000308000i32);
    pub const PHYSICAL_DEVICE_TILE_SHADING_FEATURES: Self = Self(1000309000i32);
    pub const PHYSICAL_DEVICE_TILE_SHADING_PROPERTIES: Self = Self(1000309001i32);
    pub const RENDER_PASS_TILE_SHADING_CREATE_INFO: Self = Self(1000309002i32);
    pub const PER_TILE_BEGIN_INFO: Self = Self(1000309003i32);
    pub const PER_TILE_END_INFO: Self = Self(1000309004i32);
    pub const DISPATCH_TILE_INFO: Self = Self(1000309005i32);
    pub const QUERY_LOW_LATENCY_SUPPORT: Self = Self(1000310000i32);
    pub const EXPORT_METAL_OBJECT_CREATE_INFO: Self = Self(1000311000i32);
    pub const EXPORT_METAL_OBJECTS_INFO: Self = Self(1000311001i32);
    pub const EXPORT_METAL_DEVICE_INFO: Self = Self(1000311002i32);
    pub const EXPORT_METAL_COMMAND_QUEUE_INFO: Self = Self(1000311003i32);
    pub const EXPORT_METAL_BUFFER_INFO: Self = Self(1000311004i32);
    pub const IMPORT_METAL_BUFFER_INFO: Self = Self(1000311005i32);
    pub const EXPORT_METAL_TEXTURE_INFO: Self = Self(1000311006i32);
    pub const IMPORT_METAL_TEXTURE_INFO: Self = Self(1000311007i32);
    pub const EXPORT_METAL_IO_SURFACE_INFO: Self = Self(1000311008i32);
    pub const IMPORT_METAL_IO_SURFACE_INFO: Self = Self(1000311009i32);
    pub const EXPORT_METAL_SHARED_EVENT_INFO: Self = Self(1000311010i32);
    pub const IMPORT_METAL_SHARED_EVENT_INFO: Self = Self(1000311011i32);
    pub const PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES: Self = Self(1000316000i32);
    pub const PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_DENSITY_MAP_PROPERTIES: Self = Self(
        1000316001i32,
    );
    pub const PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES: Self = Self(1000316002i32);
    pub const DESCRIPTOR_ADDRESS_INFO: Self = Self(1000316003i32);
    pub const DESCRIPTOR_GET_INFO: Self = Self(1000316004i32);
    pub const BUFFER_CAPTURE_DESCRIPTOR_DATA_INFO: Self = Self(1000316005i32);
    pub const IMAGE_CAPTURE_DESCRIPTOR_DATA_INFO: Self = Self(1000316006i32);
    pub const IMAGE_VIEW_CAPTURE_DESCRIPTOR_DATA_INFO: Self = Self(1000316007i32);
    pub const SAMPLER_CAPTURE_DESCRIPTOR_DATA_INFO: Self = Self(1000316008i32);
    pub const OPAQUE_CAPTURE_DESCRIPTOR_DATA_CREATE_INFO: Self = Self(1000316010i32);
    pub const DESCRIPTOR_BUFFER_BINDING_INFO: Self = Self(1000316011i32);
    pub const DESCRIPTOR_BUFFER_BINDING_PUSH_DESCRIPTOR_BUFFER_HANDLE: Self = Self(
        1000316012i32,
    );
    pub const ACCELERATION_STRUCTURE_CAPTURE_DESCRIPTOR_DATA_INFO: Self = Self(
        1000316009i32,
    );
    pub const DEVICE_MEMORY_COPY: Self = Self(1000318000i32);
    pub const COPY_DEVICE_MEMORY_INFO: Self = Self(1000318001i32);
    pub const DEVICE_MEMORY_IMAGE_COPY: Self = Self(1000318002i32);
    pub const COPY_DEVICE_MEMORY_IMAGE_INFO: Self = Self(1000318003i32);
    pub const MEMORY_RANGE_BARRIERS_INFO: Self = Self(1000318004i32);
    pub const MEMORY_RANGE_BARRIER: Self = Self(1000318005i32);
    pub const PHYSICAL_DEVICE_DEVICE_ADDRESS_COMMANDS_FEATURES: Self = Self(
        1000318006i32,
    );
    pub const BIND_INDEX_BUFFER_3_INFO: Self = Self(1000318007i32);
    pub const BIND_VERTEX_BUFFER_3_INFO: Self = Self(1000318008i32);
    pub const DRAW_INDIRECT_2_INFO: Self = Self(1000318009i32);
    pub const DRAW_INDIRECT_COUNT_2_INFO: Self = Self(1000318010i32);
    pub const DISPATCH_INDIRECT_2_INFO: Self = Self(1000318011i32);
    pub const CONDITIONAL_RENDERING_BEGIN_INFO_2: Self = Self(1000318012i32);
    pub const BIND_TRANSFORM_FEEDBACK_BUFFER_2_INFO: Self = Self(1000318013i32);
    pub const MEMORY_MARKER_INFO: Self = Self(1000318014i32);
    pub const ACCELERATION_STRUCTURE_CREATE_INFO_2: Self = Self(1000318015i32);
    pub const PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES: Self = Self(
        1000320000i32,
    );
    pub const PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES: Self = Self(
        1000320001i32,
    );
    pub const GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO: Self = Self(1000320002i32);
    pub const PHYSICAL_DEVICE_SHADER_EARLY_AND_LATE_FRAGMENT_TESTS_FEATURES: Self = Self(
        1000321000i32,
    );
    pub const PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_PROPERTIES: Self = Self(
        1000322000i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_FEATURES: Self = Self(
        1000323000i32,
    );
    pub const PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_ENUMS_PROPERTIES: Self = Self(
        1000326000i32,
    );
    pub const PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_ENUMS_FEATURES: Self = Self(
        1000326001i32,
    );
    pub const PIPELINE_FRAGMENT_SHADING_RATE_ENUM_STATE_CREATE_INFO: Self = Self(
        1000326002i32,
    );
    pub const ACCELERATION_STRUCTURE_GEOMETRY_MOTION_TRIANGLES_DATA: Self = Self(
        1000327000i32,
    );
    pub const PHYSICAL_DEVICE_RAY_TRACING_MOTION_BLUR_FEATURES: Self = Self(
        1000327001i32,
    );
    pub const ACCELERATION_STRUCTURE_MOTION_INFO: Self = Self(1000327002i32);
    pub const PHYSICAL_DEVICE_YCBCR_2_PLANE_444_FORMATS_FEATURES: Self = Self(
        1000330000i32,
    );
    pub const PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_2_FEATURES: Self = Self(
        1000332000i32,
    );
    pub const PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_2_PROPERTIES: Self = Self(
        1000332001i32,
    );
    pub const COPY_COMMAND_TRANSFORM_INFO: Self = Self(1000333000i32);
    pub const PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES: Self = Self(
        1000336000i32,
    );
    pub const PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_FEATURES: Self = Self(
        1000338000i32,
    );
    pub const IMAGE_COMPRESSION_CONTROL: Self = Self(1000338001i32);
    pub const IMAGE_COMPRESSION_PROPERTIES: Self = Self(1000338004i32);
    pub const PHYSICAL_DEVICE_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_FEATURES: Self = Self(
        1000339000i32,
    );
    pub const PHYSICAL_DEVICE_4444_FORMATS_FEATURES: Self = Self(1000340000i32);
    pub const PHYSICAL_DEVICE_FAULT_FEATURES: Self = Self(1000341000i32);
    pub const DEVICE_FAULT_COUNTS: Self = Self(1000341001i32);
    pub const DEVICE_FAULT_INFO: Self = Self(1000341002i32);
    pub const PHYSICAL_DEVICE_RGBA10X6_FORMATS_FEATURES: Self = Self(1000344000i32);
    pub const DIRECTFB_SURFACE_CREATE_INFO: Self = Self(1000346000i32);
    pub const PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES: Self = Self(
        1000352000i32,
    );
    pub const VERTEX_INPUT_BINDING_DESCRIPTION_2: Self = Self(1000352001i32);
    pub const VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2: Self = Self(1000352002i32);
    pub const PHYSICAL_DEVICE_DRM_PROPERTIES: Self = Self(1000353000i32);
    pub const PHYSICAL_DEVICE_ADDRESS_BINDING_REPORT_FEATURES: Self = Self(
        1000354000i32,
    );
    pub const DEVICE_ADDRESS_BINDING_CALLBACK_DATA: Self = Self(1000354001i32);
    pub const PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES: Self = Self(1000355000i32);
    pub const PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO: Self = Self(
        1000355001i32,
    );
    pub const PHYSICAL_DEVICE_PRIMITIVE_TOPOLOGY_LIST_RESTART_FEATURES: Self = Self(
        1000356000i32,
    );
    pub const IMPORT_MEMORY_ZIRCON_HANDLE_INFO: Self = Self(1000364000i32);
    pub const MEMORY_ZIRCON_HANDLE_PROPERTIES: Self = Self(1000364001i32);
    pub const MEMORY_GET_ZIRCON_HANDLE_INFO: Self = Self(1000364002i32);
    pub const IMPORT_SEMAPHORE_ZIRCON_HANDLE_INFO: Self = Self(1000365000i32);
    pub const SEMAPHORE_GET_ZIRCON_HANDLE_INFO: Self = Self(1000365001i32);
    pub const BUFFER_COLLECTION_CREATE_INFO: Self = Self(1000366000i32);
    pub const IMPORT_MEMORY_BUFFER_COLLECTION: Self = Self(1000366001i32);
    pub const BUFFER_COLLECTION_IMAGE_CREATE_INFO: Self = Self(1000366002i32);
    pub const BUFFER_COLLECTION_PROPERTIES: Self = Self(1000366003i32);
    pub const BUFFER_CONSTRAINTS_INFO: Self = Self(1000366004i32);
    pub const BUFFER_COLLECTION_BUFFER_CREATE_INFO: Self = Self(1000366005i32);
    pub const IMAGE_CONSTRAINTS_INFO: Self = Self(1000366006i32);
    pub const IMAGE_FORMAT_CONSTRAINTS_INFO: Self = Self(1000366007i32);
    pub const SYSMEM_COLOR_SPACE: Self = Self(1000366008i32);
    pub const BUFFER_COLLECTION_CONSTRAINTS_INFO: Self = Self(1000366009i32);
    pub const SUBPASS_SHADING_PIPELINE_CREATE_INFO: Self = Self(1000369000i32);
    pub const PHYSICAL_DEVICE_SUBPASS_SHADING_FEATURES: Self = Self(1000369001i32);
    pub const PHYSICAL_DEVICE_SUBPASS_SHADING_PROPERTIES: Self = Self(1000369002i32);
    pub const PHYSICAL_DEVICE_INVOCATION_MASK_FEATURES: Self = Self(1000370000i32);
    pub const MEMORY_GET_REMOTE_ADDRESS_INFO: Self = Self(1000371000i32);
    pub const PHYSICAL_DEVICE_EXTERNAL_MEMORY_RDMA_FEATURES: Self = Self(1000371001i32);
    pub const PIPELINE_PROPERTIES_IDENTIFIER: Self = Self(1000372000i32);
    pub const PHYSICAL_DEVICE_PIPELINE_PROPERTIES_FEATURES: Self = Self(1000372001i32);
    pub const IMPORT_FENCE_SCI_SYNC_INFO: Self = Self(1000373000i32);
    pub const EXPORT_FENCE_SCI_SYNC_INFO: Self = Self(1000373001i32);
    pub const FENCE_GET_SCI_SYNC_INFO: Self = Self(1000373002i32);
    pub const SCI_SYNC_ATTRIBUTES_INFO: Self = Self(1000373003i32);
    pub const IMPORT_SEMAPHORE_SCI_SYNC_INFO: Self = Self(1000373004i32);
    pub const EXPORT_SEMAPHORE_SCI_SYNC_INFO: Self = Self(1000373005i32);
    pub const SEMAPHORE_GET_SCI_SYNC_INFO: Self = Self(1000373006i32);
    pub const PHYSICAL_DEVICE_EXTERNAL_SCI_SYNC_FEATURES: Self = Self(1000373007i32);
    pub const IMPORT_MEMORY_SCI_BUF_INFO: Self = Self(1000374000i32);
    pub const EXPORT_MEMORY_SCI_BUF_INFO: Self = Self(1000374001i32);
    pub const MEMORY_GET_SCI_BUF_INFO: Self = Self(1000374002i32);
    pub const MEMORY_SCI_BUF_PROPERTIES: Self = Self(1000374003i32);
    pub const PHYSICAL_DEVICE_EXTERNAL_MEMORY_SCI_BUF_FEATURES: Self = Self(
        1000374004i32,
    );
    pub const PHYSICAL_DEVICE_EXTERNAL_SCI_BUF_FEATURES: Self = Self::PHYSICAL_DEVICE_EXTERNAL_MEMORY_SCI_BUF_FEATURES;
    pub const PHYSICAL_DEVICE_FRAME_BOUNDARY_FEATURES: Self = Self(1000375000i32);
    pub const FRAME_BOUNDARY: Self = Self(1000375001i32);
    pub const PHYSICAL_DEVICE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_FEATURES: Self = Self(
        1000376000i32,
    );
    pub const SUBPASS_RESOLVE_PERFORMANCE_QUERY: Self = Self(1000376001i32);
    pub const MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_INFO: Self = Self(1000376002i32);
    pub const PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES: Self = Self(
        1000377000i32,
    );
    pub const SCREEN_SURFACE_CREATE_INFO: Self = Self(1000378000i32);
    pub const PHYSICAL_DEVICE_COLOR_WRITE_ENABLE_FEATURES: Self = Self(1000381000i32);
    pub const PIPELINE_COLOR_WRITE_CREATE_INFO: Self = Self(1000381001i32);
    pub const PHYSICAL_DEVICE_PRIMITIVES_GENERATED_QUERY_FEATURES: Self = Self(
        1000382000i32,
    );
    pub const PHYSICAL_DEVICE_RAY_TRACING_MAINTENANCE_1_FEATURES: Self = Self(
        1000386000i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES: Self = Self(
        1000387000i32,
    );
    pub const PHYSICAL_DEVICE_VIDEO_ENCODE_RGB_CONVERSION_FEATURES: Self = Self(
        1000390000i32,
    );
    pub const VIDEO_ENCODE_RGB_CONVERSION_CAPABILITIES: Self = Self(1000390001i32);
    pub const VIDEO_ENCODE_PROFILE_RGB_CONVERSION_INFO: Self = Self(1000390002i32);
    pub const VIDEO_ENCODE_SESSION_RGB_CONVERSION_CREATE_INFO: Self = Self(
        1000390003i32,
    );
    pub const PHYSICAL_DEVICE_IMAGE_VIEW_MIN_LOD_FEATURES: Self = Self(1000391000i32);
    pub const IMAGE_VIEW_MIN_LOD_CREATE_INFO: Self = Self(1000391001i32);
    pub const PHYSICAL_DEVICE_MULTI_DRAW_FEATURES: Self = Self(1000392000i32);
    pub const PHYSICAL_DEVICE_MULTI_DRAW_PROPERTIES: Self = Self(1000392001i32);
    pub const PHYSICAL_DEVICE_IMAGE_2D_VIEW_OF_3D_FEATURES: Self = Self(1000393000i32);
    pub const PHYSICAL_DEVICE_SHADER_TILE_IMAGE_FEATURES: Self = Self(1000395000i32);
    pub const PHYSICAL_DEVICE_SHADER_TILE_IMAGE_PROPERTIES: Self = Self(1000395001i32);
    pub const MICROMAP_BUILD_INFO: Self = Self(1000396000i32);
    pub const MICROMAP_VERSION_INFO: Self = Self(1000396001i32);
    pub const COPY_MICROMAP_INFO: Self = Self(1000396002i32);
    pub const COPY_MICROMAP_TO_MEMORY_INFO: Self = Self(1000396003i32);
    pub const COPY_MEMORY_TO_MICROMAP_INFO: Self = Self(1000396004i32);
    pub const PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES: Self = Self(1000396005i32);
    pub const PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES: Self = Self(1000396006i32);
    pub const MICROMAP_CREATE_INFO: Self = Self(1000396007i32);
    pub const MICROMAP_BUILD_SIZES_INFO: Self = Self(1000396008i32);
    pub const ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP: Self = Self(
        1000396009i32,
    );
    pub const PHYSICAL_DEVICE_DISPLACEMENT_MICROMAP_FEATURES: Self = Self(1000397000i32);
    pub const PHYSICAL_DEVICE_DISPLACEMENT_MICROMAP_PROPERTIES: Self = Self(
        1000397001i32,
    );
    pub const ACCELERATION_STRUCTURE_TRIANGLES_DISPLACEMENT_MICROMAP: Self = Self(
        1000397002i32,
    );
    pub const PHYSICAL_DEVICE_CLUSTER_CULLING_SHADER_FEATURES: Self = Self(
        1000404000i32,
    );
    pub const PHYSICAL_DEVICE_CLUSTER_CULLING_SHADER_PROPERTIES: Self = Self(
        1000404001i32,
    );
    pub const PHYSICAL_DEVICE_CLUSTER_CULLING_SHADER_VRS_FEATURES: Self = Self(
        1000404002i32,
    );
    pub const PHYSICAL_DEVICE_BORDER_COLOR_SWIZZLE_FEATURES: Self = Self(1000411000i32);
    pub const SAMPLER_BORDER_COLOR_COMPONENT_MAPPING_CREATE_INFO: Self = Self(
        1000411001i32,
    );
    pub const PHYSICAL_DEVICE_PAGEABLE_DEVICE_LOCAL_MEMORY_FEATURES: Self = Self(
        1000412000i32,
    );
    pub const DEVICE_QUEUE_SHADER_CORE_CONTROL_CREATE_INFO: Self = Self(1000417000i32);
    pub const PHYSICAL_DEVICE_SCHEDULING_CONTROLS_FEATURES: Self = Self(1000417001i32);
    pub const PHYSICAL_DEVICE_SCHEDULING_CONTROLS_PROPERTIES: Self = Self(1000417002i32);
    pub const PHYSICAL_DEVICE_IMAGE_SLICED_VIEW_OF_3D_FEATURES: Self = Self(
        1000418000i32,
    );
    pub const IMAGE_VIEW_SLICED_CREATE_INFO: Self = Self(1000418001i32);
    pub const PHYSICAL_DEVICE_DESCRIPTOR_SET_HOST_MAPPING_FEATURES: Self = Self(
        1000420000i32,
    );
    pub const DESCRIPTOR_SET_BINDING_REFERENCE: Self = Self(1000420001i32);
    pub const DESCRIPTOR_SET_LAYOUT_HOST_MAPPING_INFO: Self = Self(1000420002i32);
    pub const PHYSICAL_DEVICE_NON_SEAMLESS_CUBE_MAP_FEATURES: Self = Self(1000422000i32);
    pub const PHYSICAL_DEVICE_RENDER_PASS_STRIPED_FEATURES: Self = Self(1000424000i32);
    pub const PHYSICAL_DEVICE_RENDER_PASS_STRIPED_PROPERTIES: Self = Self(1000424001i32);
    pub const RENDER_PASS_STRIPE_BEGIN_INFO: Self = Self(1000424002i32);
    pub const RENDER_PASS_STRIPE_INFO: Self = Self(1000424003i32);
    pub const RENDER_PASS_STRIPE_SUBMIT_INFO: Self = Self(1000424004i32);
    pub const SUBPASS_FRAGMENT_DENSITY_MAP_OFFSET_END_INFO: Self = Self::RENDER_PASS_FRAGMENT_DENSITY_MAP_OFFSET_END_INFO;
    pub const PHYSICAL_DEVICE_COPY_MEMORY_INDIRECT_FEATURES: Self = Self(1000426000i32);
    pub const PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_COMPUTE_FEATURES: Self = Self(
        1000428000i32,
    );
    pub const COMPUTE_PIPELINE_INDIRECT_BUFFER_INFO: Self = Self(1000428001i32);
    pub const PIPELINE_INDIRECT_DEVICE_ADDRESS_INFO: Self = Self(1000428002i32);
    pub const PHYSICAL_DEVICE_RAY_TRACING_LINEAR_SWEPT_SPHERES_FEATURES: Self = Self(
        1000429008i32,
    );
    pub const ACCELERATION_STRUCTURE_GEOMETRY_LINEAR_SWEPT_SPHERES_DATA: Self = Self(
        1000429009i32,
    );
    pub const ACCELERATION_STRUCTURE_GEOMETRY_SPHERES_DATA: Self = Self(1000429010i32);
    pub const PHYSICAL_DEVICE_LINEAR_COLOR_ATTACHMENT_FEATURES: Self = Self(
        1000430000i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES: Self = Self(
        1000434000i32,
    );
    pub const APPLICATION_PARAMETERS: Self = Self(1000435000i32);
    pub const PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_SWAPCHAIN_FEATURES: Self = Self(
        1000437000i32,
    );
    pub const PHYSICAL_DEVICE_IMAGE_PROCESSING_FEATURES: Self = Self(1000440000i32);
    pub const PHYSICAL_DEVICE_IMAGE_PROCESSING_PROPERTIES: Self = Self(1000440001i32);
    pub const IMAGE_VIEW_SAMPLE_WEIGHT_CREATE_INFO: Self = Self(1000440002i32);
    pub const PHYSICAL_DEVICE_NESTED_COMMAND_BUFFER_FEATURES: Self = Self(1000451000i32);
    pub const PHYSICAL_DEVICE_NESTED_COMMAND_BUFFER_PROPERTIES: Self = Self(
        1000451001i32,
    );
    pub const NATIVE_BUFFER_USAGE: Self = Self(1000452000i32);
    pub const NATIVE_BUFFER_PROPERTIES: Self = Self(1000452001i32);
    pub const NATIVE_BUFFER_FORMAT_PROPERTIES: Self = Self(1000452002i32);
    pub const IMPORT_NATIVE_BUFFER_INFO: Self = Self(1000452003i32);
    pub const MEMORY_GET_NATIVE_BUFFER_INFO: Self = Self(1000452004i32);
    pub const EXTERNAL_MEMORY_ACQUIRE_UNMODIFIED: Self = Self(1000453000i32);
    pub const PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES: Self = Self(
        1000455000i32,
    );
    pub const PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_PROPERTIES: Self = Self(
        1000455001i32,
    );
    pub const PHYSICAL_DEVICE_SUBPASS_MERGE_FEEDBACK_FEATURES: Self = Self(
        1000458000i32,
    );
    pub const RENDER_PASS_CREATION_CONTROL: Self = Self(1000458001i32);
    pub const RENDER_PASS_CREATION_FEEDBACK_CREATE_INFO: Self = Self(1000458002i32);
    pub const RENDER_PASS_SUBPASS_FEEDBACK_CREATE_INFO: Self = Self(1000458003i32);
    pub const DIRECT_DRIVER_LOADING_INFO: Self = Self(1000459000i32);
    pub const DIRECT_DRIVER_LOADING_LIST: Self = Self(1000459001i32);
    pub const TENSOR_CREATE_INFO: Self = Self(1000460000i32);
    pub const TENSOR_VIEW_CREATE_INFO: Self = Self(1000460001i32);
    pub const BIND_TENSOR_MEMORY_INFO: Self = Self(1000460002i32);
    pub const WRITE_DESCRIPTOR_SET_TENSOR: Self = Self(1000460003i32);
    pub const PHYSICAL_DEVICE_TENSOR_PROPERTIES: Self = Self(1000460004i32);
    pub const TENSOR_FORMAT_PROPERTIES: Self = Self(1000460005i32);
    pub const TENSOR_DESCRIPTION: Self = Self(1000460006i32);
    pub const TENSOR_MEMORY_REQUIREMENTS_INFO: Self = Self(1000460007i32);
    pub const TENSOR_MEMORY_BARRIER: Self = Self(1000460008i32);
    pub const PHYSICAL_DEVICE_TENSOR_FEATURES: Self = Self(1000460009i32);
    pub const DEVICE_TENSOR_MEMORY_REQUIREMENTS: Self = Self(1000460010i32);
    pub const COPY_TENSOR_INFO: Self = Self(1000460011i32);
    pub const TENSOR_COPY: Self = Self(1000460012i32);
    pub const TENSOR_DEPENDENCY_INFO: Self = Self(1000460013i32);
    pub const MEMORY_DEDICATED_ALLOCATE_INFO_TENSOR: Self = Self(1000460014i32);
    pub const PHYSICAL_DEVICE_EXTERNAL_TENSOR_INFO: Self = Self(1000460015i32);
    pub const EXTERNAL_TENSOR_PROPERTIES: Self = Self(1000460016i32);
    pub const EXTERNAL_MEMORY_TENSOR_CREATE_INFO: Self = Self(1000460017i32);
    pub const PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_TENSOR_FEATURES: Self = Self(
        1000460018i32,
    );
    pub const PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_TENSOR_PROPERTIES: Self = Self(
        1000460019i32,
    );
    pub const DESCRIPTOR_GET_TENSOR_INFO: Self = Self(1000460020i32);
    pub const TENSOR_CAPTURE_DESCRIPTOR_DATA_INFO: Self = Self(1000460021i32);
    pub const TENSOR_VIEW_CAPTURE_DESCRIPTOR_DATA_INFO: Self = Self(1000460022i32);
    pub const FRAME_BOUNDARY_TENSORS: Self = Self(1000460023i32);
    pub const PHYSICAL_DEVICE_SHADER_MODULE_IDENTIFIER_FEATURES: Self = Self(
        1000462000i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_MODULE_IDENTIFIER_PROPERTIES: Self = Self(
        1000462001i32,
    );
    pub const PIPELINE_SHADER_STAGE_MODULE_IDENTIFIER_CREATE_INFO: Self = Self(
        1000462002i32,
    );
    pub const SHADER_MODULE_IDENTIFIER: Self = Self(1000462003i32);
    pub const PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES: Self = Self(1000464000i32);
    pub const PHYSICAL_DEVICE_OPTICAL_FLOW_PROPERTIES: Self = Self(1000464001i32);
    pub const OPTICAL_FLOW_IMAGE_FORMAT_INFO: Self = Self(1000464002i32);
    pub const OPTICAL_FLOW_IMAGE_FORMAT_PROPERTIES: Self = Self(1000464003i32);
    pub const OPTICAL_FLOW_SESSION_CREATE_INFO: Self = Self(1000464004i32);
    pub const OPTICAL_FLOW_EXECUTE_INFO: Self = Self(1000464005i32);
    pub const OPTICAL_FLOW_SESSION_CREATE_PRIVATE_DATA_INFO: Self = Self(1000464010i32);
    pub const PHYSICAL_DEVICE_LEGACY_DITHERING_FEATURES: Self = Self(1000465000i32);
    pub const PHYSICAL_DEVICE_EXTERNAL_FORMAT_RESOLVE_FEATURES: Self = Self(
        1000468000i32,
    );
    pub const PHYSICAL_DEVICE_EXTERNAL_FORMAT_RESOLVE_PROPERTIES: Self = Self(
        1000468001i32,
    );
    pub const ANDROID_HARDWARE_BUFFER_FORMAT_RESOLVE_PROPERTIES: Self = Self(
        1000468002i32,
    );
    pub const PHYSICAL_DEVICE_ANTI_LAG_FEATURES: Self = Self(1000476000i32);
    pub const ANTI_LAG_DATA: Self = Self(1000476001i32);
    pub const ANTI_LAG_PRESENTATION_INFO: Self = Self(1000476002i32);
    pub const PHYSICAL_DEVICE_DENSE_GEOMETRY_FORMAT_FEATURES: Self = Self(1000478000i32);
    pub const ACCELERATION_STRUCTURE_DENSE_GEOMETRY_FORMAT_TRIANGLES_DATA: Self = Self(
        1000478001i32,
    );
    pub const SURFACE_CAPABILITIES_PRESENT_ID_2: Self = Self(1000479000i32);
    pub const PRESENT_ID_2: Self = Self(1000479001i32);
    pub const PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES: Self = Self(1000479002i32);
    pub const SURFACE_CAPABILITIES_PRESENT_WAIT_2: Self = Self(1000480000i32);
    pub const PHYSICAL_DEVICE_PRESENT_WAIT_2_FEATURES: Self = Self(1000480001i32);
    pub const PRESENT_WAIT_2_INFO: Self = Self(1000480002i32);
    pub const PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES: Self = Self(
        1000481000i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES: Self = Self(1000482000i32);
    pub const PHYSICAL_DEVICE_SHADER_OBJECT_PROPERTIES: Self = Self(1000482001i32);
    pub const SHADER_CREATE_INFO: Self = Self(1000482002i32);
    pub const SHADER_REQUIRED_SUBGROUP_SIZE_CREATE_INFO: Self = Self::PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    pub const PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES: Self = Self(1000483000i32);
    pub const PIPELINE_BINARY_CREATE_INFO: Self = Self(1000483001i32);
    pub const PIPELINE_BINARY_INFO: Self = Self(1000483002i32);
    pub const PIPELINE_BINARY_KEY: Self = Self(1000483003i32);
    pub const PHYSICAL_DEVICE_PIPELINE_BINARY_PROPERTIES: Self = Self(1000483004i32);
    pub const RELEASE_CAPTURED_PIPELINE_DATA_INFO: Self = Self(1000483005i32);
    pub const PIPELINE_BINARY_DATA_INFO: Self = Self(1000483006i32);
    pub const PIPELINE_CREATE_INFO: Self = Self(1000483007i32);
    pub const DEVICE_PIPELINE_BINARY_INTERNAL_CACHE_CONTROL: Self = Self(1000483008i32);
    pub const PIPELINE_BINARY_HANDLES_INFO: Self = Self(1000483009i32);
    pub const PHYSICAL_DEVICE_TILE_PROPERTIES_FEATURES: Self = Self(1000484000i32);
    pub const TILE_PROPERTIES: Self = Self(1000484001i32);
    pub const PHYSICAL_DEVICE_AMIGO_PROFILING_FEATURES: Self = Self(1000485000i32);
    pub const AMIGO_PROFILING_SUBMIT_INFO: Self = Self(1000485001i32);
    pub const PHYSICAL_DEVICE_MULTIVIEW_PER_VIEW_VIEWPORTS_FEATURES: Self = Self(
        1000488000i32,
    );
    pub const SEMAPHORE_SCI_SYNC_POOL_CREATE_INFO: Self = Self(1000489000i32);
    pub const SEMAPHORE_SCI_SYNC_CREATE_INFO: Self = Self(1000489001i32);
    pub const PHYSICAL_DEVICE_EXTERNAL_SCI_SYNC_2_FEATURES: Self = Self(1000489002i32);
    pub const DEVICE_SEMAPHORE_SCI_SYNC_POOL_RESERVATION_CREATE_INFO: Self = Self(
        1000489003i32,
    );
    pub const PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES: Self = Self(
        1000490000i32,
    );
    pub const PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES: Self = Self(
        1000490001i32,
    );
    pub const PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES: Self = Self(1000491000i32);
    pub const PHYSICAL_DEVICE_COOPERATIVE_VECTOR_PROPERTIES: Self = Self(1000491001i32);
    pub const COOPERATIVE_VECTOR_PROPERTIES: Self = Self(1000491002i32);
    pub const CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO: Self = Self(1000491004i32);
    pub const PHYSICAL_DEVICE_EXTENDED_SPARSE_ADDRESS_SPACE_FEATURES: Self = Self(
        1000492000i32,
    );
    pub const PHYSICAL_DEVICE_EXTENDED_SPARSE_ADDRESS_SPACE_PROPERTIES: Self = Self(
        1000492001i32,
    );
    pub const PHYSICAL_DEVICE_LEGACY_VERTEX_ATTRIBUTES_FEATURES: Self = Self(
        1000495000i32,
    );
    pub const PHYSICAL_DEVICE_LEGACY_VERTEX_ATTRIBUTES_PROPERTIES: Self = Self(
        1000495001i32,
    );
    pub const LAYER_SETTINGS_CREATE_INFO: Self = Self(1000496000i32);
    pub const PHYSICAL_DEVICE_SHADER_CORE_BUILTINS_FEATURES: Self = Self(1000497000i32);
    pub const PHYSICAL_DEVICE_SHADER_CORE_BUILTINS_PROPERTIES: Self = Self(
        1000497001i32,
    );
    pub const PHYSICAL_DEVICE_PIPELINE_LIBRARY_GROUP_HANDLES_FEATURES: Self = Self(
        1000498000i32,
    );
    pub const PHYSICAL_DEVICE_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_FEATURES: Self = Self(
        1000499000i32,
    );
    pub const PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES: Self = Self(
        1000504000i32,
    );
    pub const LATENCY_SLEEP_MODE_INFO: Self = Self(1000505000i32);
    pub const LATENCY_SLEEP_INFO: Self = Self(1000505001i32);
    pub const SET_LATENCY_MARKER_INFO: Self = Self(1000505002i32);
    pub const GET_LATENCY_MARKER_INFO: Self = Self(1000505003i32);
    pub const LATENCY_TIMINGS_FRAME_REPORT: Self = Self(1000505004i32);
    pub const LATENCY_SUBMISSION_PRESENT_ID: Self = Self(1000505005i32);
    pub const OUT_OF_BAND_QUEUE_TYPE_INFO: Self = Self(1000505006i32);
    pub const SWAPCHAIN_LATENCY_CREATE_INFO: Self = Self(1000505007i32);
    pub const LATENCY_SURFACE_CAPABILITIES: Self = Self(1000505008i32);
    pub const DATA_GRAPH_PIPELINE_CREATE_INFO: Self = Self(1000507000i32);
    pub const DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO: Self = Self(1000507001i32);
    pub const DATA_GRAPH_PIPELINE_RESOURCE_INFO: Self = Self(1000507002i32);
    pub const DATA_GRAPH_PIPELINE_CONSTANT: Self = Self(1000507003i32);
    pub const DATA_GRAPH_PIPELINE_SESSION_MEMORY_REQUIREMENTS_INFO: Self = Self(
        1000507004i32,
    );
    pub const BIND_DATA_GRAPH_PIPELINE_SESSION_MEMORY_INFO: Self = Self(1000507005i32);
    pub const PHYSICAL_DEVICE_DATA_GRAPH_FEATURES: Self = Self(1000507006i32);
    pub const DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO: Self = Self(1000507007i32);
    pub const DATA_GRAPH_PIPELINE_PROPERTY_QUERY_RESULT: Self = Self(1000507008i32);
    pub const DATA_GRAPH_PIPELINE_INFO: Self = Self(1000507009i32);
    pub const DATA_GRAPH_PIPELINE_COMPILER_CONTROL_CREATE_INFO: Self = Self(
        1000507010i32,
    );
    pub const DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENTS_INFO: Self = Self(
        1000507011i32,
    );
    pub const DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENT: Self = Self(
        1000507012i32,
    );
    pub const DATA_GRAPH_PIPELINE_IDENTIFIER_CREATE_INFO: Self = Self(1000507013i32);
    pub const DATA_GRAPH_PIPELINE_DISPATCH_INFO: Self = Self(1000507014i32);
    pub const DATA_GRAPH_PROCESSING_ENGINE_CREATE_INFO: Self = Self(1000507016i32);
    pub const QUEUE_FAMILY_DATA_GRAPH_PROCESSING_ENGINE_PROPERTIES: Self = Self(
        1000507017i32,
    );
    pub const QUEUE_FAMILY_DATA_GRAPH_PROPERTIES: Self = Self(1000507018i32);
    pub const PHYSICAL_DEVICE_QUEUE_FAMILY_DATA_GRAPH_PROCESSING_ENGINE_INFO: Self = Self(
        1000507019i32,
    );
    pub const DATA_GRAPH_PIPELINE_CONSTANT_TENSOR_SEMI_STRUCTURED_SPARSITY_INFO: Self = Self(
        1000507015i32,
    );
    pub const PHYSICAL_DEVICE_MULTIVIEW_PER_VIEW_RENDER_AREAS_FEATURES: Self = Self(
        1000510000i32,
    );
    pub const MULTIVIEW_PER_VIEW_RENDER_AREAS_RENDER_PASS_BEGIN_INFO: Self = Self(
        1000510001i32,
    );
    pub const PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_PROPERTIES: Self = Self(
        1000511000i32,
    );
    pub const VIDEO_DECODE_AV1_CAPABILITIES: Self = Self(1000512000i32);
    pub const VIDEO_DECODE_AV1_PICTURE_INFO: Self = Self(1000512001i32);
    pub const VIDEO_DECODE_AV1_PROFILE_INFO: Self = Self(1000512003i32);
    pub const VIDEO_DECODE_AV1_SESSION_PARAMETERS_CREATE_INFO: Self = Self(
        1000512004i32,
    );
    pub const VIDEO_DECODE_AV1_DPB_SLOT_INFO: Self = Self(1000512005i32);
    pub const VIDEO_ENCODE_AV1_CAPABILITIES: Self = Self(1000513000i32);
    pub const VIDEO_ENCODE_AV1_SESSION_PARAMETERS_CREATE_INFO: Self = Self(
        1000513001i32,
    );
    pub const VIDEO_ENCODE_AV1_PICTURE_INFO: Self = Self(1000513002i32);
    pub const VIDEO_ENCODE_AV1_DPB_SLOT_INFO: Self = Self(1000513003i32);
    pub const PHYSICAL_DEVICE_VIDEO_ENCODE_AV1_FEATURES: Self = Self(1000513004i32);
    pub const VIDEO_ENCODE_AV1_PROFILE_INFO: Self = Self(1000513005i32);
    pub const VIDEO_ENCODE_AV1_RATE_CONTROL_INFO: Self = Self(1000513006i32);
    pub const VIDEO_ENCODE_AV1_RATE_CONTROL_LAYER_INFO: Self = Self(1000513007i32);
    pub const VIDEO_ENCODE_AV1_QUALITY_LEVEL_PROPERTIES: Self = Self(1000513008i32);
    pub const VIDEO_ENCODE_AV1_SESSION_CREATE_INFO: Self = Self(1000513009i32);
    pub const VIDEO_ENCODE_AV1_GOP_REMAINING_FRAME_INFO: Self = Self(1000513010i32);
    pub const PHYSICAL_DEVICE_VIDEO_DECODE_VP9_FEATURES: Self = Self(1000514000i32);
    pub const VIDEO_DECODE_VP9_CAPABILITIES: Self = Self(1000514001i32);
    pub const VIDEO_DECODE_VP9_PICTURE_INFO: Self = Self(1000514002i32);
    pub const VIDEO_DECODE_VP9_PROFILE_INFO: Self = Self(1000514003i32);
    pub const PHYSICAL_DEVICE_VIDEO_MAINTENANCE_1_FEATURES: Self = Self(1000515000i32);
    pub const VIDEO_INLINE_QUERY_INFO: Self = Self(1000515001i32);
    pub const PHYSICAL_DEVICE_PER_STAGE_DESCRIPTOR_SET_FEATURES: Self = Self(
        1000516000i32,
    );
    pub const PHYSICAL_DEVICE_IMAGE_PROCESSING_2_FEATURES: Self = Self(1000518000i32);
    pub const PHYSICAL_DEVICE_IMAGE_PROCESSING_2_PROPERTIES: Self = Self(1000518001i32);
    pub const SAMPLER_BLOCK_MATCH_WINDOW_CREATE_INFO: Self = Self(1000518002i32);
    pub const SAMPLER_CUBIC_WEIGHTS_CREATE_INFO: Self = Self(1000519000i32);
    pub const PHYSICAL_DEVICE_CUBIC_WEIGHTS_FEATURES: Self = Self(1000519001i32);
    pub const BLIT_IMAGE_CUBIC_WEIGHTS_INFO: Self = Self(1000519002i32);
    pub const PHYSICAL_DEVICE_YCBCR_DEGAMMA_FEATURES: Self = Self(1000520000i32);
    pub const SAMPLER_YCBCR_CONVERSION_YCBCR_DEGAMMA_CREATE_INFO: Self = Self(
        1000520001i32,
    );
    pub const PHYSICAL_DEVICE_CUBIC_CLAMP_FEATURES: Self = Self(1000521000i32);
    pub const PHYSICAL_DEVICE_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_FEATURES: Self = Self(
        1000524000i32,
    );
    pub const PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES: Self = Self(1000527000i32);
    pub const ATTACHMENT_FEEDBACK_LOOP_INFO: Self = Self(1000527001i32);
    pub const SCREEN_BUFFER_PROPERTIES: Self = Self(1000529000i32);
    pub const SCREEN_BUFFER_FORMAT_PROPERTIES: Self = Self(1000529001i32);
    pub const IMPORT_SCREEN_BUFFER_INFO: Self = Self(1000529002i32);
    pub const PHYSICAL_DEVICE_EXTERNAL_MEMORY_SCREEN_BUFFER_FEATURES: Self = Self(
        1000529004i32,
    );
    pub const PHYSICAL_DEVICE_LAYERED_DRIVER_PROPERTIES: Self = Self(1000530000i32);
    pub const SET_DESCRIPTOR_BUFFER_OFFSETS_INFO: Self = Self(1000545007i32);
    pub const BIND_DESCRIPTOR_BUFFER_EMBEDDED_SAMPLERS_INFO: Self = Self(1000545008i32);
    pub const PHYSICAL_DEVICE_DESCRIPTOR_POOL_OVERALLOCATION_FEATURES: Self = Self(
        1000546000i32,
    );
    pub const PHYSICAL_DEVICE_TILE_MEMORY_HEAP_FEATURES: Self = Self(1000547000i32);
    pub const PHYSICAL_DEVICE_TILE_MEMORY_HEAP_PROPERTIES: Self = Self(1000547001i32);
    pub const TILE_MEMORY_REQUIREMENTS: Self = Self(1000547002i32);
    pub const TILE_MEMORY_BIND_INFO: Self = Self(1000547003i32);
    pub const TILE_MEMORY_SIZE_INFO: Self = Self(1000547004i32);
    pub const COPY_MEMORY_INDIRECT_INFO: Self = Self(1000549002i32);
    pub const COPY_MEMORY_TO_IMAGE_INDIRECT_INFO: Self = Self(1000549003i32);
    pub const DECOMPRESS_MEMORY_INFO: Self = Self(1000550002i32);
    pub const DISPLAY_SURFACE_STEREO_CREATE_INFO: Self = Self(1000551000i32);
    pub const DISPLAY_MODE_STEREO_PROPERTIES: Self = Self(1000551001i32);
    pub const VIDEO_ENCODE_INTRA_REFRESH_CAPABILITIES: Self = Self(1000552000i32);
    pub const VIDEO_ENCODE_SESSION_INTRA_REFRESH_CREATE_INFO: Self = Self(1000552001i32);
    pub const VIDEO_ENCODE_INTRA_REFRESH_INFO: Self = Self(1000552002i32);
    pub const VIDEO_REFERENCE_INTRA_REFRESH_INFO: Self = Self(1000552003i32);
    pub const PHYSICAL_DEVICE_VIDEO_ENCODE_INTRA_REFRESH_FEATURES: Self = Self(
        1000552004i32,
    );
    pub const VIDEO_ENCODE_QUANTIZATION_MAP_CAPABILITIES: Self = Self(1000553000i32);
    pub const VIDEO_FORMAT_QUANTIZATION_MAP_PROPERTIES: Self = Self(1000553001i32);
    pub const VIDEO_ENCODE_QUANTIZATION_MAP_INFO: Self = Self(1000553002i32);
    pub const VIDEO_ENCODE_QUANTIZATION_MAP_SESSION_PARAMETERS_CREATE_INFO: Self = Self(
        1000553005i32,
    );
    pub const PHYSICAL_DEVICE_VIDEO_ENCODE_QUANTIZATION_MAP_FEATURES: Self = Self(
        1000553009i32,
    );
    pub const VIDEO_ENCODE_H264_QUANTIZATION_MAP_CAPABILITIES: Self = Self(
        1000553003i32,
    );
    pub const VIDEO_ENCODE_H265_QUANTIZATION_MAP_CAPABILITIES: Self = Self(
        1000553004i32,
    );
    pub const VIDEO_FORMAT_H265_QUANTIZATION_MAP_PROPERTIES: Self = Self(1000553006i32);
    pub const VIDEO_ENCODE_AV1_QUANTIZATION_MAP_CAPABILITIES: Self = Self(1000553007i32);
    pub const VIDEO_FORMAT_AV1_QUANTIZATION_MAP_PROPERTIES: Self = Self(1000553008i32);
    pub const PHYSICAL_DEVICE_RAW_ACCESS_CHAINS_FEATURES: Self = Self(1000555000i32);
    pub const EXTERNAL_COMPUTE_QUEUE_DEVICE_CREATE_INFO: Self = Self(1000556000i32);
    pub const EXTERNAL_COMPUTE_QUEUE_CREATE_INFO: Self = Self(1000556001i32);
    pub const EXTERNAL_COMPUTE_QUEUE_DATA_PARAMS: Self = Self(1000556002i32);
    pub const PHYSICAL_DEVICE_EXTERNAL_COMPUTE_QUEUE_PROPERTIES: Self = Self(
        1000556003i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_RELAXED_EXTENDED_INSTRUCTION_FEATURES: Self = Self(
        1000558000i32,
    );
    pub const PHYSICAL_DEVICE_COMMAND_BUFFER_INHERITANCE_FEATURES: Self = Self(
        1000559000i32,
    );
    pub const PHYSICAL_DEVICE_MAINTENANCE_7_FEATURES: Self = Self(1000562000i32);
    pub const PHYSICAL_DEVICE_MAINTENANCE_7_PROPERTIES: Self = Self(1000562001i32);
    pub const PHYSICAL_DEVICE_LAYERED_API_PROPERTIES_LIST: Self = Self(1000562002i32);
    pub const PHYSICAL_DEVICE_LAYERED_API_PROPERTIES: Self = Self(1000562003i32);
    pub const PHYSICAL_DEVICE_LAYERED_API_VULKAN_PROPERTIES: Self = Self(1000562004i32);
    pub const PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT16_VECTOR_FEATURES: Self = Self(
        1000563000i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_REPLICATED_COMPOSITES_FEATURES: Self = Self(
        1000564000i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES: Self = Self(1000567000i32);
    pub const PHYSICAL_DEVICE_RAY_TRACING_VALIDATION_FEATURES: Self = Self(
        1000568000i32,
    );
    pub const PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES: Self = Self(
        1000569000i32,
    );
    pub const PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES: Self = Self(
        1000569001i32,
    );
    pub const CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT: Self = Self(
        1000569002i32,
    );
    pub const CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT: Self = Self(
        1000569003i32,
    );
    pub const CLUSTER_ACCELERATION_STRUCTURE_MOVE_OBJECTS_INPUT: Self = Self(
        1000569004i32,
    );
    pub const CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO: Self = Self(1000569005i32);
    pub const CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO: Self = Self(1000569006i32);
    pub const RAY_TRACING_PIPELINE_CLUSTER_ACCELERATION_STRUCTURE_CREATE_INFO: Self = Self(
        1000569007i32,
    );
    pub const PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_FEATURES: Self = Self(
        1000570000i32,
    );
    pub const PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_PROPERTIES: Self = Self(
        1000570001i32,
    );
    pub const WRITE_DESCRIPTOR_SET_PARTITIONED_ACCELERATION_STRUCTURE: Self = Self(
        1000570002i32,
    );
    pub const PARTITIONED_ACCELERATION_STRUCTURE_INSTANCES_INPUT: Self = Self(
        1000570003i32,
    );
    pub const BUILD_PARTITIONED_ACCELERATION_STRUCTURE_INFO: Self = Self(1000570004i32);
    pub const PARTITIONED_ACCELERATION_STRUCTURE_FLAGS: Self = Self(1000570005i32);
    pub const INDIRECT_EXECUTION_SET_CREATE_INFO: Self = Self(1000572003i32);
    pub const WRITE_INDIRECT_EXECUTION_SET_PIPELINE: Self = Self(1000572008i32);
    pub const WRITE_INDIRECT_EXECUTION_SET_SHADER: Self = Self(1000572009i32);
    pub const INDIRECT_EXECUTION_SET_PIPELINE_INFO: Self = Self(1000572010i32);
    pub const INDIRECT_EXECUTION_SET_SHADER_INFO: Self = Self(1000572011i32);
    pub const INDIRECT_EXECUTION_SET_SHADER_LAYOUT_INFO: Self = Self(1000572012i32);
    pub const GENERATED_COMMANDS_PIPELINE_INFO: Self = Self(1000572013i32);
    pub const GENERATED_COMMANDS_SHADER_INFO: Self = Self(1000572014i32);
    pub const PHYSICAL_DEVICE_FAULT_PROPERTIES: Self = Self(1000573001i32);
    pub const DEVICE_FAULT_DEBUG_INFO: Self = Self(1000573003i32);
    pub const PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES: Self = Self(1000574000i32);
    pub const MEMORY_BARRIER_ACCESS_FLAGS_3: Self = Self(1000574002i32);
    pub const PHYSICAL_DEVICE_IMAGE_ALIGNMENT_CONTROL_FEATURES: Self = Self(
        1000575000i32,
    );
    pub const PHYSICAL_DEVICE_IMAGE_ALIGNMENT_CONTROL_PROPERTIES: Self = Self(
        1000575001i32,
    );
    pub const IMAGE_ALIGNMENT_CONTROL_CREATE_INFO: Self = Self(1000575002i32);
    pub const PHYSICAL_DEVICE_SHADER_FMA_FEATURES: Self = Self(1000579000i32);
    pub const PUSH_CONSTANT_BANK_INFO: Self = Self(1000580000i32);
    pub const PHYSICAL_DEVICE_PUSH_CONSTANT_BANK_FEATURES: Self = Self(1000580001i32);
    pub const PHYSICAL_DEVICE_PUSH_CONSTANT_BANK_PROPERTIES: Self = Self(1000580002i32);
    pub const PHYSICAL_DEVICE_DEPTH_CLAMP_CONTROL_FEATURES: Self = Self(1000582000i32);
    pub const PIPELINE_VIEWPORT_DEPTH_CLAMP_CONTROL_CREATE_INFO: Self = Self(
        1000582001i32,
    );
    pub const PHYSICAL_DEVICE_MAINTENANCE_9_FEATURES: Self = Self(1000584000i32);
    pub const PHYSICAL_DEVICE_MAINTENANCE_9_PROPERTIES: Self = Self(1000584001i32);
    pub const QUEUE_FAMILY_OWNERSHIP_TRANSFER_PROPERTIES: Self = Self(1000584002i32);
    pub const PHYSICAL_DEVICE_VIDEO_MAINTENANCE_2_FEATURES: Self = Self(1000586000i32);
    pub const VIDEO_DECODE_H264_INLINE_SESSION_PARAMETERS_INFO: Self = Self(
        1000586001i32,
    );
    pub const VIDEO_DECODE_H265_INLINE_SESSION_PARAMETERS_INFO: Self = Self(
        1000586002i32,
    );
    pub const VIDEO_DECODE_AV1_INLINE_SESSION_PARAMETERS_INFO: Self = Self(
        1000586003i32,
    );
    pub const SURFACE_CREATE_INFO: Self = Self(1000685000i32);
    pub const PHYSICAL_DEVICE_HDR_VIVID_FEATURES: Self = Self(1000590000i32);
    pub const HDR_VIVID_DYNAMIC_METADATA: Self = Self(1000590001i32);
    pub const PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES: Self = Self(1000593000i32);
    pub const COOPERATIVE_MATRIX_FLEXIBLE_DIMENSIONS_PROPERTIES: Self = Self(
        1000593001i32,
    );
    pub const PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_PROPERTIES: Self = Self(
        1000593002i32,
    );
    pub const PHYSICAL_DEVICE_PIPELINE_OPACITY_MICROMAP_FEATURES: Self = Self(
        1000596000i32,
    );
    pub const IMPORT_MEMORY_METAL_HANDLE_INFO: Self = Self(1000602000i32);
    pub const MEMORY_METAL_HANDLE_PROPERTIES: Self = Self(1000602001i32);
    pub const MEMORY_GET_METAL_HANDLE_INFO: Self = Self(1000602002i32);
    pub const PHYSICAL_DEVICE_PERFORMANCE_COUNTERS_BY_REGION_FEATURES: Self = Self(
        1000605000i32,
    );
    pub const PHYSICAL_DEVICE_PERFORMANCE_COUNTERS_BY_REGION_PROPERTIES: Self = Self(
        1000605001i32,
    );
    pub const RENDER_PASS_PERFORMANCE_COUNTERS_BY_REGION_BEGIN_INFO: Self = Self(
        1000605004i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_INSTRUMENTATION_FEATURES: Self = Self(
        1000607000i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_INSTRUMENTATION_PROPERTIES: Self = Self(
        1000607001i32,
    );
    pub const SHADER_INSTRUMENTATION_CREATE_INFO: Self = Self(1000607002i32);
    pub const SHADER_INSTRUMENTATION_METRIC_DESCRIPTION: Self = Self(1000607003i32);
    pub const PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_ROBUSTNESS_FEATURES: Self = Self(
        1000608000i32,
    );
    pub const PHYSICAL_DEVICE_FORMAT_PACK_FEATURES: Self = Self(1000609000i32);
    pub const PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_LAYERED_FEATURES: Self = Self(
        1000611000i32,
    );
    pub const PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_LAYERED_PROPERTIES: Self = Self(
        1000611001i32,
    );
    pub const PIPELINE_FRAGMENT_DENSITY_MAP_LAYERED_CREATE_INFO: Self = Self(
        1000611002i32,
    );
    pub const SET_PRESENT_CONFIG: Self = Self(1000613000i32);
    pub const PHYSICAL_DEVICE_PRESENT_METERING_FEATURES: Self = Self(1000613001i32);
    pub const RENDER_PASS_FRAGMENT_DENSITY_MAP_OFFSET_END_INFO: Self = Self(
        1000425002i32,
    );
    pub const PHYSICAL_DEVICE_ZERO_INITIALIZE_DEVICE_MEMORY_FEATURES: Self = Self(
        1000620000i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_64_BIT_INDEXING_FEATURES: Self = Self(
        1000627000i32,
    );
    pub const PHYSICAL_DEVICE_CUSTOM_RESOLVE_FEATURES: Self = Self(1000628000i32);
    pub const BEGIN_CUSTOM_RESOLVE_INFO: Self = Self(1000628001i32);
    pub const CUSTOM_RESOLVE_CREATE_INFO: Self = Self(1000628002i32);
    pub const PHYSICAL_DEVICE_DATA_GRAPH_MODEL_FEATURES: Self = Self(1000629000i32);
    pub const DATA_GRAPH_PIPELINE_BUILTIN_MODEL_CREATE_INFO: Self = Self(1000629001i32);
    pub const PHYSICAL_DEVICE_MAINTENANCE_10_FEATURES: Self = Self(1000630000i32);
    pub const PHYSICAL_DEVICE_MAINTENANCE_10_PROPERTIES: Self = Self(1000630001i32);
    pub const RENDERING_ATTACHMENT_FLAGS_INFO: Self = Self(1000630002i32);
    pub const RESOLVE_IMAGE_MODE_INFO: Self = Self(1000630004i32);
    pub const PHYSICAL_DEVICE_SHADER_LONG_VECTOR_FEATURES: Self = Self(1000635000i32);
    pub const PHYSICAL_DEVICE_SHADER_LONG_VECTOR_PROPERTIES: Self = Self(1000635001i32);
    pub const PHYSICAL_DEVICE_PIPELINE_CACHE_INCREMENTAL_MODE_FEATURES: Self = Self(
        1000637000i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_UNIFORM_BUFFER_UNSIZED_ARRAY_FEATURES: Self = Self(
        1000642000i32,
    );
    pub const COMPUTE_OCCUPANCY_PRIORITY_PARAMETERS: Self = Self(1000645000i32);
    pub const PHYSICAL_DEVICE_COMPUTE_OCCUPANCY_PRIORITY_FEATURES: Self = Self(
        1000645001i32,
    );
    pub const PHYSICAL_DEVICE_SHADER_SUBGROUP_PARTITIONED_FEATURES: Self = Self(
        1000662000i32,
    );
    pub const UBM_SURFACE_CREATE_INFO: Self = Self(1000664000i32);
    pub const PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_PRODUCT_FEATURES: Self = Self(
        1000673000i32,
    );
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for StructureType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("APPLICATION_INFO"),
            1i32 => f.write_str("INSTANCE_CREATE_INFO"),
            2i32 => f.write_str("DEVICE_QUEUE_CREATE_INFO"),
            3i32 => f.write_str("DEVICE_CREATE_INFO"),
            4i32 => f.write_str("SUBMIT_INFO"),
            5i32 => f.write_str("MEMORY_ALLOCATE_INFO"),
            6i32 => f.write_str("MAPPED_MEMORY_RANGE"),
            7i32 => f.write_str("BIND_SPARSE_INFO"),
            8i32 => f.write_str("FENCE_CREATE_INFO"),
            9i32 => f.write_str("SEMAPHORE_CREATE_INFO"),
            10i32 => f.write_str("EVENT_CREATE_INFO"),
            11i32 => f.write_str("QUERY_POOL_CREATE_INFO"),
            12i32 => f.write_str("BUFFER_CREATE_INFO"),
            13i32 => f.write_str("BUFFER_VIEW_CREATE_INFO"),
            14i32 => f.write_str("IMAGE_CREATE_INFO"),
            15i32 => f.write_str("IMAGE_VIEW_CREATE_INFO"),
            16i32 => f.write_str("SHADER_MODULE_CREATE_INFO"),
            17i32 => f.write_str("PIPELINE_CACHE_CREATE_INFO"),
            18i32 => f.write_str("PIPELINE_SHADER_STAGE_CREATE_INFO"),
            19i32 => f.write_str("PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO"),
            20i32 => f.write_str("PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO"),
            21i32 => f.write_str("PIPELINE_TESSELLATION_STATE_CREATE_INFO"),
            22i32 => f.write_str("PIPELINE_VIEWPORT_STATE_CREATE_INFO"),
            23i32 => f.write_str("PIPELINE_RASTERIZATION_STATE_CREATE_INFO"),
            24i32 => f.write_str("PIPELINE_MULTISAMPLE_STATE_CREATE_INFO"),
            25i32 => f.write_str("PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO"),
            26i32 => f.write_str("PIPELINE_COLOR_BLEND_STATE_CREATE_INFO"),
            27i32 => f.write_str("PIPELINE_DYNAMIC_STATE_CREATE_INFO"),
            28i32 => f.write_str("GRAPHICS_PIPELINE_CREATE_INFO"),
            29i32 => f.write_str("COMPUTE_PIPELINE_CREATE_INFO"),
            30i32 => f.write_str("PIPELINE_LAYOUT_CREATE_INFO"),
            31i32 => f.write_str("SAMPLER_CREATE_INFO"),
            32i32 => f.write_str("DESCRIPTOR_SET_LAYOUT_CREATE_INFO"),
            33i32 => f.write_str("DESCRIPTOR_POOL_CREATE_INFO"),
            34i32 => f.write_str("DESCRIPTOR_SET_ALLOCATE_INFO"),
            35i32 => f.write_str("WRITE_DESCRIPTOR_SET"),
            36i32 => f.write_str("COPY_DESCRIPTOR_SET"),
            37i32 => f.write_str("FRAMEBUFFER_CREATE_INFO"),
            38i32 => f.write_str("RENDER_PASS_CREATE_INFO"),
            39i32 => f.write_str("COMMAND_POOL_CREATE_INFO"),
            40i32 => f.write_str("COMMAND_BUFFER_ALLOCATE_INFO"),
            41i32 => f.write_str("COMMAND_BUFFER_INHERITANCE_INFO"),
            42i32 => f.write_str("COMMAND_BUFFER_BEGIN_INFO"),
            43i32 => f.write_str("RENDER_PASS_BEGIN_INFO"),
            44i32 => f.write_str("BUFFER_MEMORY_BARRIER"),
            45i32 => f.write_str("IMAGE_MEMORY_BARRIER"),
            46i32 => f.write_str("MEMORY_BARRIER"),
            47i32 => f.write_str("LOADER_INSTANCE_CREATE_INFO"),
            48i32 => f.write_str("LOADER_DEVICE_CREATE_INFO"),
            1000157000i32 => f.write_str("BIND_BUFFER_MEMORY_INFO"),
            1000157001i32 => f.write_str("BIND_IMAGE_MEMORY_INFO"),
            1000127000i32 => f.write_str("MEMORY_DEDICATED_REQUIREMENTS"),
            1000127001i32 => f.write_str("MEMORY_DEDICATED_ALLOCATE_INFO"),
            1000060000i32 => f.write_str("MEMORY_ALLOCATE_FLAGS_INFO"),
            1000060004i32 => f.write_str("DEVICE_GROUP_COMMAND_BUFFER_BEGIN_INFO"),
            1000060005i32 => f.write_str("DEVICE_GROUP_SUBMIT_INFO"),
            1000060006i32 => f.write_str("DEVICE_GROUP_BIND_SPARSE_INFO"),
            1000060013i32 => f.write_str("BIND_BUFFER_MEMORY_DEVICE_GROUP_INFO"),
            1000060014i32 => f.write_str("BIND_IMAGE_MEMORY_DEVICE_GROUP_INFO"),
            1000070000i32 => f.write_str("PHYSICAL_DEVICE_GROUP_PROPERTIES"),
            1000070001i32 => f.write_str("DEVICE_GROUP_DEVICE_CREATE_INFO"),
            1000146000i32 => f.write_str("BUFFER_MEMORY_REQUIREMENTS_INFO_2"),
            1000146001i32 => f.write_str("IMAGE_MEMORY_REQUIREMENTS_INFO_2"),
            1000146002i32 => f.write_str("IMAGE_SPARSE_MEMORY_REQUIREMENTS_INFO_2"),
            1000146003i32 => f.write_str("MEMORY_REQUIREMENTS_2"),
            1000146004i32 => f.write_str("SPARSE_IMAGE_MEMORY_REQUIREMENTS_2"),
            1000059000i32 => f.write_str("PHYSICAL_DEVICE_FEATURES_2"),
            1000059001i32 => f.write_str("PHYSICAL_DEVICE_PROPERTIES_2"),
            1000059002i32 => f.write_str("FORMAT_PROPERTIES_2"),
            1000059003i32 => f.write_str("IMAGE_FORMAT_PROPERTIES_2"),
            1000059004i32 => f.write_str("PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2"),
            1000059005i32 => f.write_str("QUEUE_FAMILY_PROPERTIES_2"),
            1000059006i32 => f.write_str("PHYSICAL_DEVICE_MEMORY_PROPERTIES_2"),
            1000059007i32 => f.write_str("SPARSE_IMAGE_FORMAT_PROPERTIES_2"),
            1000059008i32 => f.write_str("PHYSICAL_DEVICE_SPARSE_IMAGE_FORMAT_INFO_2"),
            1000117002i32 => f.write_str("IMAGE_VIEW_USAGE_CREATE_INFO"),
            1000145000i32 => f.write_str("PROTECTED_SUBMIT_INFO"),
            1000145001i32 => f.write_str("PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES"),
            1000145002i32 => f.write_str("PHYSICAL_DEVICE_PROTECTED_MEMORY_PROPERTIES"),
            1000145003i32 => f.write_str("DEVICE_QUEUE_INFO_2"),
            1000071000i32 => f.write_str("PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO"),
            1000071001i32 => f.write_str("EXTERNAL_IMAGE_FORMAT_PROPERTIES"),
            1000071002i32 => f.write_str("PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO"),
            1000071003i32 => f.write_str("EXTERNAL_BUFFER_PROPERTIES"),
            1000071004i32 => f.write_str("PHYSICAL_DEVICE_ID_PROPERTIES"),
            1000072000i32 => f.write_str("EXTERNAL_MEMORY_BUFFER_CREATE_INFO"),
            1000072001i32 => f.write_str("EXTERNAL_MEMORY_IMAGE_CREATE_INFO"),
            1000072002i32 => f.write_str("EXPORT_MEMORY_ALLOCATE_INFO"),
            1000112000i32 => f.write_str("PHYSICAL_DEVICE_EXTERNAL_FENCE_INFO"),
            1000112001i32 => f.write_str("EXTERNAL_FENCE_PROPERTIES"),
            1000113000i32 => f.write_str("EXPORT_FENCE_CREATE_INFO"),
            1000077000i32 => f.write_str("EXPORT_SEMAPHORE_CREATE_INFO"),
            1000076000i32 => f.write_str("PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO"),
            1000076001i32 => f.write_str("EXTERNAL_SEMAPHORE_PROPERTIES"),
            1000094000i32 => f.write_str("PHYSICAL_DEVICE_SUBGROUP_PROPERTIES"),
            1000083000i32 => f.write_str("PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES"),
            1000120000i32 => f.write_str("PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES"),
            1000085000i32 => f.write_str("DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO"),
            1000168000i32 => f.write_str("PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES"),
            1000168001i32 => f.write_str("DESCRIPTOR_SET_LAYOUT_SUPPORT"),
            1000156000i32 => f.write_str("SAMPLER_YCBCR_CONVERSION_CREATE_INFO"),
            1000156001i32 => f.write_str("SAMPLER_YCBCR_CONVERSION_INFO"),
            1000156002i32 => f.write_str("BIND_IMAGE_PLANE_MEMORY_INFO"),
            1000156003i32 => f.write_str("IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO"),
            1000156004i32 => {
                f.write_str("PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES")
            }
            1000156005i32 => {
                f.write_str("SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES")
            }
            1000060003i32 => f.write_str("DEVICE_GROUP_RENDER_PASS_BEGIN_INFO"),
            1000117000i32 => f.write_str("PHYSICAL_DEVICE_POINT_CLIPPING_PROPERTIES"),
            1000117001i32 => {
                f.write_str("RENDER_PASS_INPUT_ATTACHMENT_ASPECT_CREATE_INFO")
            }
            1000117003i32 => {
                f.write_str("PIPELINE_TESSELLATION_DOMAIN_ORIGIN_STATE_CREATE_INFO")
            }
            1000053000i32 => f.write_str("RENDER_PASS_MULTIVIEW_CREATE_INFO"),
            1000053001i32 => f.write_str("PHYSICAL_DEVICE_MULTIVIEW_FEATURES"),
            1000053002i32 => f.write_str("PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES"),
            1000063000i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES")
            }
            1000196000i32 => f.write_str("PHYSICAL_DEVICE_DRIVER_PROPERTIES"),
            49i32 => f.write_str("PHYSICAL_DEVICE_VULKAN_1_1_FEATURES"),
            50i32 => f.write_str("PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES"),
            51i32 => f.write_str("PHYSICAL_DEVICE_VULKAN_1_2_FEATURES"),
            52i32 => f.write_str("PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES"),
            1000147000i32 => f.write_str("IMAGE_FORMAT_LIST_CREATE_INFO"),
            1000211000i32 => f.write_str("PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES"),
            1000261000i32 => f.write_str("PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES"),
            1000207000i32 => f.write_str("PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES"),
            1000207001i32 => f.write_str("PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES"),
            1000207002i32 => f.write_str("SEMAPHORE_TYPE_CREATE_INFO"),
            1000207003i32 => f.write_str("TIMELINE_SEMAPHORE_SUBMIT_INFO"),
            1000207004i32 => f.write_str("SEMAPHORE_WAIT_INFO"),
            1000207005i32 => f.write_str("SEMAPHORE_SIGNAL_INFO"),
            1000257000i32 => {
                f.write_str("PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES")
            }
            1000244001i32 => f.write_str("BUFFER_DEVICE_ADDRESS_INFO"),
            1000257002i32 => f.write_str("BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO"),
            1000257003i32 => f.write_str("MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO"),
            1000257004i32 => f.write_str("DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO"),
            1000177000i32 => f.write_str("PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES"),
            1000180000i32 => f.write_str("PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES"),
            1000082000i32 => f.write_str("PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES"),
            1000197000i32 => f.write_str("PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES"),
            1000161000i32 => {
                f.write_str("DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO")
            }
            1000161001i32 => f.write_str("PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES"),
            1000161002i32 => {
                f.write_str("PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES")
            }
            1000161003i32 => {
                f.write_str("DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO")
            }
            1000161004i32 => {
                f.write_str("DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_LAYOUT_SUPPORT")
            }
            1000221000i32 => f.write_str("PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES"),
            1000130000i32 => {
                f.write_str("PHYSICAL_DEVICE_SAMPLER_FILTER_MINMAX_PROPERTIES")
            }
            1000130001i32 => f.write_str("SAMPLER_REDUCTION_MODE_CREATE_INFO"),
            1000253000i32 => {
                f.write_str("PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES")
            }
            1000175000i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES")
            }
            1000109000i32 => f.write_str("ATTACHMENT_DESCRIPTION_2"),
            1000109001i32 => f.write_str("ATTACHMENT_REFERENCE_2"),
            1000109002i32 => f.write_str("SUBPASS_DESCRIPTION_2"),
            1000109003i32 => f.write_str("SUBPASS_DEPENDENCY_2"),
            1000109004i32 => f.write_str("RENDER_PASS_CREATE_INFO_2"),
            1000109005i32 => f.write_str("SUBPASS_BEGIN_INFO"),
            1000109006i32 => f.write_str("SUBPASS_END_INFO"),
            1000199000i32 => {
                f.write_str("PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES")
            }
            1000199001i32 => f.write_str("SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE"),
            1000246000i32 => f.write_str("IMAGE_STENCIL_USAGE_CREATE_INFO"),
            1000108000i32 => {
                f.write_str("PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES")
            }
            1000108001i32 => f.write_str("FRAMEBUFFER_ATTACHMENTS_CREATE_INFO"),
            1000108002i32 => f.write_str("FRAMEBUFFER_ATTACHMENT_IMAGE_INFO"),
            1000108003i32 => f.write_str("RENDER_PASS_ATTACHMENT_BEGIN_INFO"),
            1000241000i32 => {
                f.write_str("PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES")
            }
            1000241001i32 => f.write_str("ATTACHMENT_REFERENCE_STENCIL_LAYOUT"),
            1000241002i32 => f.write_str("ATTACHMENT_DESCRIPTION_STENCIL_LAYOUT"),
            53i32 => f.write_str("PHYSICAL_DEVICE_VULKAN_1_3_FEATURES"),
            54i32 => f.write_str("PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES"),
            1000245000i32 => f.write_str("PHYSICAL_DEVICE_TOOL_PROPERTIES"),
            1000295000i32 => f.write_str("PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES"),
            1000295001i32 => f.write_str("DEVICE_PRIVATE_DATA_CREATE_INFO"),
            1000295002i32 => f.write_str("PRIVATE_DATA_SLOT_CREATE_INFO"),
            1000314000i32 => f.write_str("MEMORY_BARRIER_2"),
            1000314001i32 => f.write_str("BUFFER_MEMORY_BARRIER_2"),
            1000314002i32 => f.write_str("IMAGE_MEMORY_BARRIER_2"),
            1000314003i32 => f.write_str("DEPENDENCY_INFO"),
            1000314004i32 => f.write_str("SUBMIT_INFO_2"),
            1000314005i32 => f.write_str("SEMAPHORE_SUBMIT_INFO"),
            1000314006i32 => f.write_str("COMMAND_BUFFER_SUBMIT_INFO"),
            1000314007i32 => f.write_str("PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES"),
            1000337000i32 => f.write_str("COPY_BUFFER_INFO_2"),
            1000337001i32 => f.write_str("COPY_IMAGE_INFO_2"),
            1000337002i32 => f.write_str("COPY_BUFFER_TO_IMAGE_INFO_2"),
            1000337003i32 => f.write_str("COPY_IMAGE_TO_BUFFER_INFO_2"),
            1000337006i32 => f.write_str("BUFFER_COPY_2"),
            1000337007i32 => f.write_str("IMAGE_COPY_2"),
            1000337009i32 => f.write_str("BUFFER_IMAGE_COPY_2"),
            1000066000i32 => {
                f.write_str("PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES")
            }
            1000360000i32 => f.write_str("FORMAT_PROPERTIES_3"),
            1000413000i32 => f.write_str("PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES"),
            1000413001i32 => f.write_str("PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES"),
            1000413002i32 => f.write_str("DEVICE_BUFFER_MEMORY_REQUIREMENTS"),
            1000413003i32 => f.write_str("DEVICE_IMAGE_MEMORY_REQUIREMENTS"),
            1000192000i32 => f.write_str("PIPELINE_CREATION_FEEDBACK_CREATE_INFO"),
            1000215000i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES")
            }
            1000276000i32 => {
                f.write_str(
                    "PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES",
                )
            }
            1000297000i32 => {
                f.write_str("PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES")
            }
            1000325000i32 => {
                f.write_str("PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES")
            }
            1000335000i32 => f.write_str("PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES"),
            1000225000i32 => {
                f.write_str("PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES")
            }
            1000225001i32 => {
                f.write_str("PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO")
            }
            1000225002i32 => {
                f.write_str("PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES")
            }
            1000138000i32 => f.write_str("PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES"),
            1000138001i32 => {
                f.write_str("PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_PROPERTIES")
            }
            1000138002i32 => f.write_str("WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK"),
            1000138003i32 => {
                f.write_str("DESCRIPTOR_POOL_INLINE_UNIFORM_BLOCK_CREATE_INFO")
            }
            1000280000i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES")
            }
            1000280001i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES")
            }
            1000281001i32 => {
                f.write_str("PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_PROPERTIES")
            }
            1000337004i32 => f.write_str("BLIT_IMAGE_INFO_2"),
            1000337005i32 => f.write_str("RESOLVE_IMAGE_INFO_2"),
            1000337008i32 => f.write_str("IMAGE_BLIT_2"),
            1000337010i32 => f.write_str("IMAGE_RESOLVE_2"),
            1000044000i32 => f.write_str("RENDERING_INFO"),
            1000044001i32 => f.write_str("RENDERING_ATTACHMENT_INFO"),
            1000044002i32 => f.write_str("PIPELINE_RENDERING_CREATE_INFO"),
            1000044003i32 => f.write_str("PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES"),
            1000044004i32 => f.write_str("COMMAND_BUFFER_INHERITANCE_RENDERING_INFO"),
            55i32 => f.write_str("PHYSICAL_DEVICE_VULKAN_1_4_FEATURES"),
            56i32 => f.write_str("PHYSICAL_DEVICE_VULKAN_1_4_PROPERTIES"),
            1000174000i32 => f.write_str("DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO"),
            1000388000i32 => {
                f.write_str("PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES")
            }
            1000388001i32 => f.write_str("QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES"),
            1000265000i32 => f.write_str("PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES"),
            1000271000i32 => f.write_str("MEMORY_MAP_INFO"),
            1000271001i32 => f.write_str("MEMORY_UNMAP_INFO"),
            1000470000i32 => f.write_str("PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES"),
            1000470001i32 => f.write_str("PHYSICAL_DEVICE_MAINTENANCE_5_PROPERTIES"),
            1000470004i32 => f.write_str("DEVICE_IMAGE_SUBRESOURCE_INFO"),
            1000338002i32 => f.write_str("SUBRESOURCE_LAYOUT_2"),
            1000338003i32 => f.write_str("IMAGE_SUBRESOURCE_2"),
            1000470006i32 => f.write_str("BUFFER_USAGE_FLAGS_2_CREATE_INFO"),
            1000545000i32 => f.write_str("PHYSICAL_DEVICE_MAINTENANCE_6_FEATURES"),
            1000545001i32 => f.write_str("PHYSICAL_DEVICE_MAINTENANCE_6_PROPERTIES"),
            1000545002i32 => f.write_str("BIND_MEMORY_STATUS"),
            1000270000i32 => f.write_str("PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES"),
            1000270001i32 => f.write_str("PHYSICAL_DEVICE_HOST_IMAGE_COPY_PROPERTIES"),
            1000270002i32 => f.write_str("MEMORY_TO_IMAGE_COPY"),
            1000270003i32 => f.write_str("IMAGE_TO_MEMORY_COPY"),
            1000270004i32 => f.write_str("COPY_IMAGE_TO_MEMORY_INFO"),
            1000270005i32 => f.write_str("COPY_MEMORY_TO_IMAGE_INFO"),
            1000270006i32 => f.write_str("HOST_IMAGE_LAYOUT_TRANSITION_INFO"),
            1000270007i32 => f.write_str("COPY_IMAGE_TO_IMAGE_INFO"),
            1000270008i32 => f.write_str("SUBRESOURCE_HOST_MEMCPY_SIZE"),
            1000270009i32 => f.write_str("HOST_IMAGE_COPY_DEVICE_PERFORMANCE_QUERY"),
            1000416000i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_SUBGROUP_ROTATE_FEATURES")
            }
            1000528000i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES")
            }
            1000544000i32 => f.write_str("PHYSICAL_DEVICE_SHADER_EXPECT_ASSUME_FEATURES"),
            1000470005i32 => f.write_str("PIPELINE_CREATE_FLAGS_2_CREATE_INFO"),
            1000080000i32 => f.write_str("PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES"),
            1000545003i32 => f.write_str("BIND_DESCRIPTOR_SETS_INFO"),
            1000545004i32 => f.write_str("PUSH_CONSTANTS_INFO"),
            1000545005i32 => f.write_str("PUSH_DESCRIPTOR_SET_INFO"),
            1000545006i32 => f.write_str("PUSH_DESCRIPTOR_SET_WITH_TEMPLATE_INFO"),
            1000466000i32 => {
                f.write_str("PHYSICAL_DEVICE_PIPELINE_PROTECTED_ACCESS_FEATURES")
            }
            1000068000i32 => f.write_str("PIPELINE_ROBUSTNESS_CREATE_INFO"),
            1000068001i32 => f.write_str("PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES"),
            1000068002i32 => {
                f.write_str("PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_PROPERTIES")
            }
            1000259000i32 => f.write_str("PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES"),
            1000259001i32 => f.write_str("PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO"),
            1000259002i32 => f.write_str("PHYSICAL_DEVICE_LINE_RASTERIZATION_PROPERTIES"),
            1000525000i32 => {
                f.write_str("PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_PROPERTIES")
            }
            1000190001i32 => {
                f.write_str("PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO")
            }
            1000190002i32 => {
                f.write_str("PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES")
            }
            1000470003i32 => f.write_str("RENDERING_AREA_INFO"),
            1000232000i32 => {
                f.write_str("PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES")
            }
            1000232001i32 => f.write_str("RENDERING_ATTACHMENT_LOCATION_INFO"),
            1000232002i32 => f.write_str("RENDERING_INPUT_ATTACHMENT_INDEX_INFO"),
            1000298000i32 => f.write_str("PHYSICAL_DEVICE_VULKAN_SC_1_0_FEATURES"),
            1000298001i32 => f.write_str("PHYSICAL_DEVICE_VULKAN_SC_1_0_PROPERTIES"),
            1000298002i32 => f.write_str("DEVICE_OBJECT_RESERVATION_CREATE_INFO"),
            1000298003i32 => f.write_str("COMMAND_POOL_MEMORY_RESERVATION_CREATE_INFO"),
            1000298004i32 => f.write_str("COMMAND_POOL_MEMORY_CONSUMPTION"),
            1000298005i32 => f.write_str("PIPELINE_POOL_SIZE"),
            1000298007i32 => f.write_str("FAULT_DATA"),
            1000298008i32 => f.write_str("FAULT_CALLBACK_INFO"),
            1000298010i32 => f.write_str("PIPELINE_OFFLINE_CREATE_INFO"),
            1000001000i32 => f.write_str("SWAPCHAIN_CREATE_INFO"),
            1000001001i32 => f.write_str("PRESENT_INFO"),
            1000060007i32 => f.write_str("DEVICE_GROUP_PRESENT_CAPABILITIES"),
            1000060008i32 => f.write_str("IMAGE_SWAPCHAIN_CREATE_INFO"),
            1000060009i32 => f.write_str("BIND_IMAGE_MEMORY_SWAPCHAIN_INFO"),
            1000060010i32 => f.write_str("ACQUIRE_NEXT_IMAGE_INFO"),
            1000060011i32 => f.write_str("DEVICE_GROUP_PRESENT_INFO"),
            1000060012i32 => f.write_str("DEVICE_GROUP_SWAPCHAIN_CREATE_INFO"),
            1000002000i32 => f.write_str("DISPLAY_MODE_CREATE_INFO"),
            1000002001i32 => f.write_str("DISPLAY_SURFACE_CREATE_INFO"),
            1000003000i32 => f.write_str("DISPLAY_PRESENT_INFO"),
            1000004000i32 => f.write_str("XLIB_SURFACE_CREATE_INFO"),
            1000005000i32 => f.write_str("XCB_SURFACE_CREATE_INFO"),
            1000006000i32 => f.write_str("WAYLAND_SURFACE_CREATE_INFO"),
            1000008000i32 => f.write_str("ANDROID_SURFACE_CREATE_INFO"),
            1000009000i32 => f.write_str("WIN32_SURFACE_CREATE_INFO"),
            1000011000i32 => f.write_str("DEBUG_REPORT_CALLBACK_CREATE_INFO"),
            1000018000i32 => {
                f.write_str("PIPELINE_RASTERIZATION_STATE_RASTERIZATION_ORDER")
            }
            1000022000i32 => f.write_str("DEBUG_MARKER_OBJECT_NAME_INFO"),
            1000022001i32 => f.write_str("DEBUG_MARKER_OBJECT_TAG_INFO"),
            1000022002i32 => f.write_str("DEBUG_MARKER_MARKER_INFO"),
            1000023000i32 => f.write_str("VIDEO_PROFILE_INFO"),
            1000023001i32 => f.write_str("VIDEO_CAPABILITIES"),
            1000023002i32 => f.write_str("VIDEO_PICTURE_RESOURCE_INFO"),
            1000023003i32 => f.write_str("VIDEO_SESSION_MEMORY_REQUIREMENTS"),
            1000023004i32 => f.write_str("BIND_VIDEO_SESSION_MEMORY_INFO"),
            1000023005i32 => f.write_str("VIDEO_SESSION_CREATE_INFO"),
            1000023006i32 => f.write_str("VIDEO_SESSION_PARAMETERS_CREATE_INFO"),
            1000023007i32 => f.write_str("VIDEO_SESSION_PARAMETERS_UPDATE_INFO"),
            1000023008i32 => f.write_str("VIDEO_BEGIN_CODING_INFO"),
            1000023009i32 => f.write_str("VIDEO_END_CODING_INFO"),
            1000023010i32 => f.write_str("VIDEO_CODING_CONTROL_INFO"),
            1000023011i32 => f.write_str("VIDEO_REFERENCE_SLOT_INFO"),
            1000023012i32 => f.write_str("QUEUE_FAMILY_VIDEO_PROPERTIES"),
            1000023013i32 => f.write_str("VIDEO_PROFILE_LIST_INFO"),
            1000023014i32 => f.write_str("PHYSICAL_DEVICE_VIDEO_FORMAT_INFO"),
            1000023015i32 => f.write_str("VIDEO_FORMAT_PROPERTIES"),
            1000023016i32 => f.write_str("QUEUE_FAMILY_QUERY_RESULT_STATUS_PROPERTIES"),
            1000024000i32 => f.write_str("VIDEO_DECODE_INFO"),
            1000024001i32 => f.write_str("VIDEO_DECODE_CAPABILITIES"),
            1000024002i32 => f.write_str("VIDEO_DECODE_USAGE_INFO"),
            1000026000i32 => f.write_str("DEDICATED_ALLOCATION_IMAGE_CREATE_INFO"),
            1000026001i32 => f.write_str("DEDICATED_ALLOCATION_BUFFER_CREATE_INFO"),
            1000026002i32 => f.write_str("DEDICATED_ALLOCATION_MEMORY_ALLOCATE_INFO"),
            1000028000i32 => f.write_str("PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES"),
            1000028001i32 => f.write_str("PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_PROPERTIES"),
            1000028002i32 => {
                f.write_str("PIPELINE_RASTERIZATION_STATE_STREAM_CREATE_INFO")
            }
            1000029000i32 => f.write_str("CU_MODULE_CREATE_INFO"),
            1000029001i32 => f.write_str("CU_FUNCTION_CREATE_INFO"),
            1000029002i32 => f.write_str("CU_LAUNCH_INFO"),
            1000029004i32 => f.write_str("CU_MODULE_TEXTURING_MODE_CREATE_INFO"),
            1000030000i32 => f.write_str("IMAGE_VIEW_HANDLE_INFO"),
            1000030001i32 => f.write_str("IMAGE_VIEW_ADDRESS_PROPERTIES"),
            1000038000i32 => f.write_str("VIDEO_ENCODE_H264_CAPABILITIES"),
            1000038001i32 => {
                f.write_str("VIDEO_ENCODE_H264_SESSION_PARAMETERS_CREATE_INFO")
            }
            1000038002i32 => f.write_str("VIDEO_ENCODE_H264_SESSION_PARAMETERS_ADD_INFO"),
            1000038003i32 => f.write_str("VIDEO_ENCODE_H264_PICTURE_INFO"),
            1000038004i32 => f.write_str("VIDEO_ENCODE_H264_DPB_SLOT_INFO"),
            1000038005i32 => f.write_str("VIDEO_ENCODE_H264_NALU_SLICE_INFO"),
            1000038006i32 => f.write_str("VIDEO_ENCODE_H264_GOP_REMAINING_FRAME_INFO"),
            1000038007i32 => f.write_str("VIDEO_ENCODE_H264_PROFILE_INFO"),
            1000038008i32 => f.write_str("VIDEO_ENCODE_H264_RATE_CONTROL_INFO"),
            1000038009i32 => f.write_str("VIDEO_ENCODE_H264_RATE_CONTROL_LAYER_INFO"),
            1000038010i32 => f.write_str("VIDEO_ENCODE_H264_SESSION_CREATE_INFO"),
            1000038011i32 => f.write_str("VIDEO_ENCODE_H264_QUALITY_LEVEL_PROPERTIES"),
            1000038012i32 => f.write_str("VIDEO_ENCODE_H264_SESSION_PARAMETERS_GET_INFO"),
            1000038013i32 => {
                f.write_str("VIDEO_ENCODE_H264_SESSION_PARAMETERS_FEEDBACK_INFO")
            }
            1000039000i32 => f.write_str("VIDEO_ENCODE_H265_CAPABILITIES"),
            1000039001i32 => {
                f.write_str("VIDEO_ENCODE_H265_SESSION_PARAMETERS_CREATE_INFO")
            }
            1000039002i32 => f.write_str("VIDEO_ENCODE_H265_SESSION_PARAMETERS_ADD_INFO"),
            1000039003i32 => f.write_str("VIDEO_ENCODE_H265_PICTURE_INFO"),
            1000039004i32 => f.write_str("VIDEO_ENCODE_H265_DPB_SLOT_INFO"),
            1000039005i32 => f.write_str("VIDEO_ENCODE_H265_NALU_SLICE_SEGMENT_INFO"),
            1000039006i32 => f.write_str("VIDEO_ENCODE_H265_GOP_REMAINING_FRAME_INFO"),
            1000039007i32 => f.write_str("VIDEO_ENCODE_H265_PROFILE_INFO"),
            1000039009i32 => f.write_str("VIDEO_ENCODE_H265_RATE_CONTROL_INFO"),
            1000039010i32 => f.write_str("VIDEO_ENCODE_H265_RATE_CONTROL_LAYER_INFO"),
            1000039011i32 => f.write_str("VIDEO_ENCODE_H265_SESSION_CREATE_INFO"),
            1000039012i32 => f.write_str("VIDEO_ENCODE_H265_QUALITY_LEVEL_PROPERTIES"),
            1000039013i32 => f.write_str("VIDEO_ENCODE_H265_SESSION_PARAMETERS_GET_INFO"),
            1000039014i32 => {
                f.write_str("VIDEO_ENCODE_H265_SESSION_PARAMETERS_FEEDBACK_INFO")
            }
            1000040000i32 => f.write_str("VIDEO_DECODE_H264_CAPABILITIES"),
            1000040001i32 => f.write_str("VIDEO_DECODE_H264_PICTURE_INFO"),
            1000040003i32 => f.write_str("VIDEO_DECODE_H264_PROFILE_INFO"),
            1000040004i32 => {
                f.write_str("VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO")
            }
            1000040005i32 => f.write_str("VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO"),
            1000040006i32 => f.write_str("VIDEO_DECODE_H264_DPB_SLOT_INFO"),
            1000041000i32 => f.write_str("TEXTURE_LOD_GATHER_FORMAT_PROPERTIES"),
            1000049000i32 => f.write_str("STREAM_DESCRIPTOR_SURFACE_CREATE_INFO"),
            1000050000i32 => f.write_str("PHYSICAL_DEVICE_CORNER_SAMPLED_IMAGE_FEATURES"),
            1000051000i32 => f.write_str("PRIVATE_VENDOR_INFO_PLACEHOLDER_OFFSET_0"),
            1000057000i32 => f.write_str("IMPORT_MEMORY_WIN32_HANDLE_INFO"),
            1000057001i32 => f.write_str("EXPORT_MEMORY_WIN32_HANDLE_INFO"),
            1000058000i32 => f.write_str("WIN32_KEYED_MUTEX_ACQUIRE_RELEASE_INFO"),
            1000061000i32 => f.write_str("VALIDATION_FLAGS"),
            1000062000i32 => f.write_str("VI_SURFACE_CREATE_INFO"),
            1000067000i32 => f.write_str("IMAGE_VIEW_ASTC_DECODE_MODE"),
            1000067001i32 => f.write_str("PHYSICAL_DEVICE_ASTC_DECODE_FEATURES"),
            1000073002i32 => f.write_str("MEMORY_WIN32_HANDLE_PROPERTIES"),
            1000073003i32 => f.write_str("MEMORY_GET_WIN32_HANDLE_INFO"),
            1000074000i32 => f.write_str("IMPORT_MEMORY_FD_INFO"),
            1000074001i32 => f.write_str("MEMORY_FD_PROPERTIES"),
            1000074002i32 => f.write_str("MEMORY_GET_FD_INFO"),
            1000078000i32 => f.write_str("IMPORT_SEMAPHORE_WIN32_HANDLE_INFO"),
            1000078001i32 => f.write_str("EXPORT_SEMAPHORE_WIN32_HANDLE_INFO"),
            1000078002i32 => f.write_str("D3D12_FENCE_SUBMIT_INFO"),
            1000078003i32 => f.write_str("SEMAPHORE_GET_WIN32_HANDLE_INFO"),
            1000079000i32 => f.write_str("IMPORT_SEMAPHORE_FD_INFO"),
            1000079001i32 => f.write_str("SEMAPHORE_GET_FD_INFO"),
            1000081000i32 => {
                f.write_str("COMMAND_BUFFER_INHERITANCE_CONDITIONAL_RENDERING_INFO")
            }
            1000081001i32 => {
                f.write_str("PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES")
            }
            1000081002i32 => f.write_str("CONDITIONAL_RENDERING_BEGIN_INFO"),
            1000084000i32 => f.write_str("PRESENT_REGIONS"),
            1000087000i32 => f.write_str("PIPELINE_VIEWPORT_W_SCALING_STATE_CREATE_INFO"),
            1000090000i32 => f.write_str("SURFACE_CAPABILITIES_2"),
            1000091000i32 => f.write_str("DISPLAY_POWER_INFO"),
            1000091001i32 => f.write_str("DEVICE_EVENT_INFO"),
            1000091002i32 => f.write_str("DISPLAY_EVENT_INFO"),
            1000091003i32 => f.write_str("SWAPCHAIN_COUNTER_CREATE_INFO"),
            1000092000i32 => f.write_str("PRESENT_TIMES_INFO"),
            1000097000i32 => {
                f.write_str("PHYSICAL_DEVICE_MULTIVIEW_PER_VIEW_ATTRIBUTES_PROPERTIES")
            }
            1000044009i32 => f.write_str("MULTIVIEW_PER_VIEW_ATTRIBUTES_INFO"),
            1000098000i32 => f.write_str("PIPELINE_VIEWPORT_SWIZZLE_STATE_CREATE_INFO"),
            1000099000i32 => f.write_str("PHYSICAL_DEVICE_DISCARD_RECTANGLE_PROPERTIES"),
            1000099001i32 => f.write_str("PIPELINE_DISCARD_RECTANGLE_STATE_CREATE_INFO"),
            1000101000i32 => {
                f.write_str("PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES")
            }
            1000101001i32 => {
                f.write_str("PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO")
            }
            1000102000i32 => f.write_str("PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES"),
            1000102001i32 => {
                f.write_str("PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO")
            }
            1000105000i32 => f.write_str("HDR_METADATA"),
            1000110000i32 => {
                f.write_str("PHYSICAL_DEVICE_RELAXED_LINE_RASTERIZATION_FEATURES")
            }
            1000111000i32 => f.write_str("SHARED_PRESENT_SURFACE_CAPABILITIES"),
            1000114000i32 => f.write_str("IMPORT_FENCE_WIN32_HANDLE_INFO"),
            1000114001i32 => f.write_str("EXPORT_FENCE_WIN32_HANDLE_INFO"),
            1000114002i32 => f.write_str("FENCE_GET_WIN32_HANDLE_INFO"),
            1000115000i32 => f.write_str("IMPORT_FENCE_FD_INFO"),
            1000115001i32 => f.write_str("FENCE_GET_FD_INFO"),
            1000116000i32 => f.write_str("PHYSICAL_DEVICE_PERFORMANCE_QUERY_FEATURES"),
            1000116001i32 => f.write_str("PHYSICAL_DEVICE_PERFORMANCE_QUERY_PROPERTIES"),
            1000116002i32 => f.write_str("QUERY_POOL_PERFORMANCE_CREATE_INFO"),
            1000116003i32 => f.write_str("PERFORMANCE_QUERY_SUBMIT_INFO"),
            1000116004i32 => f.write_str("ACQUIRE_PROFILING_LOCK_INFO"),
            1000116005i32 => f.write_str("PERFORMANCE_COUNTER"),
            1000116006i32 => f.write_str("PERFORMANCE_COUNTER_DESCRIPTION"),
            1000116007i32 => f.write_str("PERFORMANCE_QUERY_RESERVATION_INFO"),
            1000119000i32 => f.write_str("PHYSICAL_DEVICE_SURFACE_INFO_2"),
            1000119002i32 => f.write_str("SURFACE_FORMAT_2"),
            1000121000i32 => f.write_str("DISPLAY_PROPERTIES_2"),
            1000121001i32 => f.write_str("DISPLAY_PLANE_PROPERTIES_2"),
            1000121002i32 => f.write_str("DISPLAY_MODE_PROPERTIES_2"),
            1000121003i32 => f.write_str("DISPLAY_PLANE_INFO_2"),
            1000121004i32 => f.write_str("DISPLAY_PLANE_CAPABILITIES_2"),
            1000122000i32 => f.write_str("IOS_SURFACE_CREATE_INFO"),
            1000123000i32 => f.write_str("MACOS_SURFACE_CREATE_INFO"),
            1000128000i32 => f.write_str("DEBUG_UTILS_OBJECT_NAME_INFO"),
            1000128001i32 => f.write_str("DEBUG_UTILS_OBJECT_TAG_INFO"),
            1000128002i32 => f.write_str("DEBUG_UTILS_LABEL"),
            1000128003i32 => f.write_str("DEBUG_UTILS_MESSENGER_CALLBACK_DATA"),
            1000128004i32 => f.write_str("DEBUG_UTILS_MESSENGER_CREATE_INFO"),
            1000129000i32 => f.write_str("ANDROID_HARDWARE_BUFFER_USAGE"),
            1000129001i32 => f.write_str("ANDROID_HARDWARE_BUFFER_PROPERTIES"),
            1000129002i32 => f.write_str("ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES"),
            1000129003i32 => f.write_str("IMPORT_ANDROID_HARDWARE_BUFFER_INFO"),
            1000129004i32 => f.write_str("MEMORY_GET_ANDROID_HARDWARE_BUFFER_INFO"),
            1000129005i32 => f.write_str("EXTERNAL_FORMAT"),
            1000129006i32 => f.write_str("ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_2"),
            1000134000i32 => f.write_str("PHYSICAL_DEVICE_SHADER_ENQUEUE_FEATURES"),
            1000134001i32 => f.write_str("PHYSICAL_DEVICE_SHADER_ENQUEUE_PROPERTIES"),
            1000134002i32 => f.write_str("EXECUTION_GRAPH_PIPELINE_SCRATCH_SIZE"),
            1000134003i32 => f.write_str("EXECUTION_GRAPH_PIPELINE_CREATE_INFO"),
            1000134004i32 => f.write_str("PIPELINE_SHADER_STAGE_NODE_CREATE_INFO"),
            1000135000i32 => f.write_str("TEXEL_BUFFER_DESCRIPTOR_INFO"),
            1000135001i32 => f.write_str("IMAGE_DESCRIPTOR_INFO"),
            1000135002i32 => f.write_str("RESOURCE_DESCRIPTOR_INFO"),
            1000135003i32 => f.write_str("BIND_HEAP_INFO"),
            1000135004i32 => f.write_str("PUSH_DATA_INFO"),
            1000135005i32 => f.write_str("DESCRIPTOR_SET_AND_BINDING_MAPPING"),
            1000135006i32 => {
                f.write_str("SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO")
            }
            1000135007i32 => f.write_str("OPAQUE_CAPTURE_DATA_CREATE_INFO"),
            1000135008i32 => f.write_str("PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES"),
            1000135009i32 => f.write_str("PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES"),
            1000135010i32 => {
                f.write_str("COMMAND_BUFFER_INHERITANCE_DESCRIPTOR_HEAP_INFO")
            }
            1000135011i32 => f.write_str("SAMPLER_CUSTOM_BORDER_COLOR_INDEX_CREATE_INFO"),
            1000135012i32 => f.write_str("INDIRECT_COMMANDS_LAYOUT_PUSH_DATA_TOKEN"),
            1000135013i32 => f.write_str("SUBSAMPLED_IMAGE_FORMAT_PROPERTIES"),
            1000135014i32 => {
                f.write_str("PHYSICAL_DEVICE_DESCRIPTOR_HEAP_TENSOR_PROPERTIES")
            }
            1000044008i32 => f.write_str("ATTACHMENT_SAMPLE_COUNT_INFO"),
            1000141000i32 => f.write_str("PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES"),
            1000143000i32 => f.write_str("SAMPLE_LOCATIONS_INFO"),
            1000143001i32 => f.write_str("RENDER_PASS_SAMPLE_LOCATIONS_BEGIN_INFO"),
            1000143002i32 => f.write_str("PIPELINE_SAMPLE_LOCATIONS_STATE_CREATE_INFO"),
            1000143003i32 => f.write_str("PHYSICAL_DEVICE_SAMPLE_LOCATIONS_PROPERTIES"),
            1000143004i32 => f.write_str("MULTISAMPLE_PROPERTIES"),
            1000148000i32 => {
                f.write_str("PHYSICAL_DEVICE_BLEND_OPERATION_ADVANCED_FEATURES")
            }
            1000148001i32 => {
                f.write_str("PHYSICAL_DEVICE_BLEND_OPERATION_ADVANCED_PROPERTIES")
            }
            1000148002i32 => {
                f.write_str("PIPELINE_COLOR_BLEND_ADVANCED_STATE_CREATE_INFO")
            }
            1000149000i32 => f.write_str("PIPELINE_COVERAGE_TO_COLOR_STATE_CREATE_INFO"),
            1000150007i32 => f.write_str("WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE"),
            1000150000i32 => f.write_str("ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO"),
            1000150002i32 => f.write_str("ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO"),
            1000150003i32 => f.write_str("ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA"),
            1000150004i32 => {
                f.write_str("ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA")
            }
            1000150005i32 => {
                f.write_str("ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA")
            }
            1000150006i32 => f.write_str("ACCELERATION_STRUCTURE_GEOMETRY"),
            1000150009i32 => f.write_str("ACCELERATION_STRUCTURE_VERSION_INFO"),
            1000150010i32 => f.write_str("COPY_ACCELERATION_STRUCTURE_INFO"),
            1000150011i32 => f.write_str("COPY_ACCELERATION_STRUCTURE_TO_MEMORY_INFO"),
            1000150012i32 => f.write_str("COPY_MEMORY_TO_ACCELERATION_STRUCTURE_INFO"),
            1000150013i32 => {
                f.write_str("PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES")
            }
            1000150014i32 => {
                f.write_str("PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES")
            }
            1000150017i32 => f.write_str("ACCELERATION_STRUCTURE_CREATE_INFO"),
            1000150020i32 => f.write_str("ACCELERATION_STRUCTURE_BUILD_SIZES_INFO"),
            1000347000i32 => f.write_str("PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES"),
            1000347001i32 => {
                f.write_str("PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES")
            }
            1000150015i32 => f.write_str("RAY_TRACING_PIPELINE_CREATE_INFO"),
            1000150016i32 => f.write_str("RAY_TRACING_SHADER_GROUP_CREATE_INFO"),
            1000150018i32 => f.write_str("RAY_TRACING_PIPELINE_INTERFACE_CREATE_INFO"),
            1000348013i32 => f.write_str("PHYSICAL_DEVICE_RAY_QUERY_FEATURES"),
            1000152000i32 => {
                f.write_str("PIPELINE_COVERAGE_MODULATION_STATE_CREATE_INFO")
            }
            1000154000i32 => f.write_str("PHYSICAL_DEVICE_SHADER_SM_BUILTINS_FEATURES"),
            1000154001i32 => f.write_str("PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES"),
            1000158000i32 => f.write_str("DRM_FORMAT_MODIFIER_PROPERTIES_LIST"),
            1000158002i32 => {
                f.write_str("PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO")
            }
            1000158003i32 => f.write_str("IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO"),
            1000158004i32 => {
                f.write_str("IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO")
            }
            1000158005i32 => f.write_str("IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES"),
            1000158006i32 => f.write_str("DRM_FORMAT_MODIFIER_PROPERTIES_LIST_2"),
            1000160000i32 => f.write_str("VALIDATION_CACHE_CREATE_INFO"),
            1000160001i32 => f.write_str("SHADER_MODULE_VALIDATION_CACHE_CREATE_INFO"),
            1000163000i32 => f.write_str("PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES"),
            1000163001i32 => f.write_str("PHYSICAL_DEVICE_PORTABILITY_SUBSET_PROPERTIES"),
            1000164000i32 => {
                f.write_str("PIPELINE_VIEWPORT_SHADING_RATE_IMAGE_STATE_CREATE_INFO")
            }
            1000164001i32 => f.write_str("PHYSICAL_DEVICE_SHADING_RATE_IMAGE_FEATURES"),
            1000164002i32 => f.write_str("PHYSICAL_DEVICE_SHADING_RATE_IMAGE_PROPERTIES"),
            1000164005i32 => {
                f.write_str("PIPELINE_VIEWPORT_COARSE_SAMPLE_ORDER_STATE_CREATE_INFO")
            }
            1000165003i32 => f.write_str("GEOMETRY"),
            1000165004i32 => f.write_str("GEOMETRY_TRIANGLES"),
            1000165005i32 => f.write_str("GEOMETRY_AABB"),
            1000165006i32 => f.write_str("BIND_ACCELERATION_STRUCTURE_MEMORY_INFO"),
            1000165008i32 => {
                f.write_str("ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO")
            }
            1000165009i32 => f.write_str("PHYSICAL_DEVICE_RAY_TRACING_PROPERTIES"),
            1000165012i32 => f.write_str("ACCELERATION_STRUCTURE_INFO"),
            1000166000i32 => {
                f.write_str("PHYSICAL_DEVICE_REPRESENTATIVE_FRAGMENT_TEST_FEATURES")
            }
            1000166001i32 => {
                f.write_str("PIPELINE_REPRESENTATIVE_FRAGMENT_TEST_STATE_CREATE_INFO")
            }
            1000170000i32 => f.write_str("PHYSICAL_DEVICE_IMAGE_VIEW_IMAGE_FORMAT_INFO"),
            1000170001i32 => {
                f.write_str("FILTER_CUBIC_IMAGE_VIEW_IMAGE_FORMAT_PROPERTIES")
            }
            1000172000i32 => {
                f.write_str("PHYSICAL_DEVICE_COOPERATIVE_MATRIX_CONVERSION_FEATURES")
            }
            1000178000i32 => f.write_str("IMPORT_MEMORY_HOST_POINTER_INFO"),
            1000178001i32 => f.write_str("MEMORY_HOST_POINTER_PROPERTIES"),
            1000178002i32 => {
                f.write_str("PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES")
            }
            1000181000i32 => f.write_str("PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES"),
            1000183000i32 => f.write_str("PIPELINE_COMPILER_CONTROL_CREATE_INFO"),
            1000185000i32 => f.write_str("PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES"),
            1000187000i32 => f.write_str("VIDEO_DECODE_H265_CAPABILITIES"),
            1000187001i32 => {
                f.write_str("VIDEO_DECODE_H265_SESSION_PARAMETERS_CREATE_INFO")
            }
            1000187002i32 => f.write_str("VIDEO_DECODE_H265_SESSION_PARAMETERS_ADD_INFO"),
            1000187003i32 => f.write_str("VIDEO_DECODE_H265_PROFILE_INFO"),
            1000187004i32 => f.write_str("VIDEO_DECODE_H265_PICTURE_INFO"),
            1000187005i32 => f.write_str("VIDEO_DECODE_H265_DPB_SLOT_INFO"),
            1000189000i32 => f.write_str("DEVICE_MEMORY_OVERALLOCATION_CREATE_INFO"),
            1000191000i32 => f.write_str("PRESENT_FRAME_TOKEN"),
            1000202000i32 => f.write_str("PHYSICAL_DEVICE_MESH_SHADER_FEATURES"),
            1000202001i32 => f.write_str("PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES"),
            1000204000i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_IMAGE_FOOTPRINT_FEATURES")
            }
            1000205000i32 => {
                f.write_str("PIPELINE_VIEWPORT_EXCLUSIVE_SCISSOR_STATE_CREATE_INFO")
            }
            1000205002i32 => f.write_str("PHYSICAL_DEVICE_EXCLUSIVE_SCISSOR_FEATURES"),
            1000206000i32 => f.write_str("CHECKPOINT_DATA"),
            1000206001i32 => f.write_str("QUEUE_FAMILY_CHECKPOINT_PROPERTIES"),
            1000314008i32 => f.write_str("QUEUE_FAMILY_CHECKPOINT_PROPERTIES_2"),
            1000314009i32 => f.write_str("CHECKPOINT_DATA_2"),
            1000208000i32 => f.write_str("PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES"),
            1000208001i32 => f.write_str("SWAPCHAIN_TIMING_PROPERTIES"),
            1000208002i32 => f.write_str("SWAPCHAIN_TIME_DOMAIN_PROPERTIES"),
            1000208003i32 => f.write_str("PRESENT_TIMINGS_INFO"),
            1000208004i32 => f.write_str("PRESENT_TIMING_INFO"),
            1000208005i32 => f.write_str("PAST_PRESENTATION_TIMING_INFO"),
            1000208006i32 => f.write_str("PAST_PRESENTATION_TIMING_PROPERTIES"),
            1000208007i32 => f.write_str("PAST_PRESENTATION_TIMING"),
            1000208008i32 => f.write_str("PRESENT_TIMING_SURFACE_CAPABILITIES"),
            1000208009i32 => f.write_str("SWAPCHAIN_CALIBRATED_TIMESTAMP_INFO"),
            1000209000i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_INTEGER_FUNCTIONS_2_FEATURES")
            }
            1000210000i32 => f.write_str("QUERY_POOL_PERFORMANCE_QUERY_CREATE_INFO"),
            1000210001i32 => f.write_str("INITIALIZE_PERFORMANCE_API_INFO"),
            1000210002i32 => f.write_str("PERFORMANCE_MARKER_INFO"),
            1000210003i32 => f.write_str("PERFORMANCE_STREAM_MARKER_INFO"),
            1000210004i32 => f.write_str("PERFORMANCE_OVERRIDE_INFO"),
            1000210005i32 => f.write_str("PERFORMANCE_CONFIGURATION_ACQUIRE_INFO"),
            1000212000i32 => f.write_str("PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES"),
            1000213000i32 => f.write_str("DISPLAY_NATIVE_HDR_SURFACE_CAPABILITIES"),
            1000213001i32 => f.write_str("SWAPCHAIN_DISPLAY_NATIVE_HDR_CREATE_INFO"),
            1000214000i32 => f.write_str("IMAGEPIPE_SURFACE_CREATE_INFO"),
            1000217000i32 => f.write_str("METAL_SURFACE_CREATE_INFO"),
            1000218000i32 => f.write_str("PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_FEATURES"),
            1000218001i32 => {
                f.write_str("PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_PROPERTIES")
            }
            1000218002i32 => f.write_str("RENDER_PASS_FRAGMENT_DENSITY_MAP_CREATE_INFO"),
            1000044007i32 => {
                f.write_str("RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT_INFO")
            }
            1000226000i32 => f.write_str("FRAGMENT_SHADING_RATE_ATTACHMENT_INFO"),
            1000226001i32 => {
                f.write_str("PIPELINE_FRAGMENT_SHADING_RATE_STATE_CREATE_INFO")
            }
            1000226002i32 => {
                f.write_str("PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_PROPERTIES")
            }
            1000226003i32 => {
                f.write_str("PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES")
            }
            1000226004i32 => f.write_str("PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE"),
            1000044006i32 => {
                f.write_str("RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO")
            }
            1000227000i32 => f.write_str("PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_2"),
            1000229000i32 => f.write_str("PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES"),
            1000231000i32 => f.write_str("PHYSICAL_DEVICE_SHADER_CONSTANT_DATA_FEATURES"),
            1000233000i32 => f.write_str("PHYSICAL_DEVICE_SHADER_ABORT_FEATURES"),
            1000233001i32 => f.write_str("DEVICE_FAULT_SHADER_ABORT_MESSAGE_INFO"),
            1000233002i32 => f.write_str("PHYSICAL_DEVICE_SHADER_ABORT_PROPERTIES"),
            1000234000i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES")
            }
            1000235000i32 => f.write_str("PHYSICAL_DEVICE_SHADER_QUAD_CONTROL_FEATURES"),
            1000237000i32 => f.write_str("PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES"),
            1000238000i32 => f.write_str("PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES"),
            1000238001i32 => f.write_str("MEMORY_PRIORITY_ALLOCATE_INFO"),
            1000239000i32 => f.write_str("SURFACE_PROTECTED_CAPABILITIES"),
            1000240000i32 => {
                f.write_str(
                    "PHYSICAL_DEVICE_DEDICATED_ALLOCATION_IMAGE_ALIASING_FEATURES",
                )
            }
            1000244002i32 => f.write_str("BUFFER_DEVICE_ADDRESS_CREATE_INFO"),
            1000247000i32 => f.write_str("VALIDATION_FEATURES"),
            1000248000i32 => f.write_str("PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES"),
            1000249000i32 => f.write_str("PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES"),
            1000249001i32 => f.write_str("COOPERATIVE_MATRIX_PROPERTIES"),
            1000249002i32 => f.write_str("PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES"),
            1000250000i32 => {
                f.write_str("PHYSICAL_DEVICE_COVERAGE_REDUCTION_MODE_FEATURES")
            }
            1000250001i32 => f.write_str("PIPELINE_COVERAGE_REDUCTION_STATE_CREATE_INFO"),
            1000250002i32 => f.write_str("FRAMEBUFFER_MIXED_SAMPLES_COMBINATION"),
            1000251000i32 => {
                f.write_str("PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES")
            }
            1000252000i32 => f.write_str("PHYSICAL_DEVICE_YCBCR_IMAGE_ARRAYS_FEATURES"),
            1000254000i32 => f.write_str("PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES"),
            1000254001i32 => {
                f.write_str("PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO")
            }
            1000254002i32 => f.write_str("PHYSICAL_DEVICE_PROVOKING_VERTEX_PROPERTIES"),
            1000255000i32 => f.write_str("SURFACE_FULL_SCREEN_EXCLUSIVE_INFO"),
            1000255002i32 => f.write_str("SURFACE_CAPABILITIES_FULL_SCREEN_EXCLUSIVE"),
            1000255001i32 => f.write_str("SURFACE_FULL_SCREEN_EXCLUSIVE_WIN32_INFO"),
            1000256000i32 => f.write_str("HEADLESS_SURFACE_CREATE_INFO"),
            1000260000i32 => f.write_str("PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES"),
            1000267000i32 => {
                f.write_str("PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES")
            }
            1000269000i32 => {
                f.write_str("PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES")
            }
            1000269001i32 => f.write_str("PIPELINE_INFO"),
            1000269002i32 => f.write_str("PIPELINE_EXECUTABLE_PROPERTIES"),
            1000269003i32 => f.write_str("PIPELINE_EXECUTABLE_INFO"),
            1000269004i32 => f.write_str("PIPELINE_EXECUTABLE_STATISTIC"),
            1000269005i32 => f.write_str("PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION"),
            1000272000i32 => f.write_str("PHYSICAL_DEVICE_MAP_MEMORY_PLACED_FEATURES"),
            1000272001i32 => f.write_str("PHYSICAL_DEVICE_MAP_MEMORY_PLACED_PROPERTIES"),
            1000272002i32 => f.write_str("MEMORY_MAP_PLACED_INFO"),
            1000273000i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES")
            }
            1000277000i32 => {
                f.write_str("PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_PROPERTIES")
            }
            1000277001i32 => f.write_str("GRAPHICS_SHADER_GROUP_CREATE_INFO"),
            1000277002i32 => f.write_str("GRAPHICS_PIPELINE_SHADER_GROUPS_CREATE_INFO"),
            1000277003i32 => f.write_str("INDIRECT_COMMANDS_LAYOUT_TOKEN"),
            1000277004i32 => f.write_str("INDIRECT_COMMANDS_LAYOUT_CREATE_INFO"),
            1000277005i32 => f.write_str("GENERATED_COMMANDS_INFO"),
            1000277006i32 => f.write_str("GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO"),
            1000277007i32 => {
                f.write_str("PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES")
            }
            1000278000i32 => {
                f.write_str("PHYSICAL_DEVICE_INHERITED_VIEWPORT_SCISSOR_FEATURES")
            }
            1000278001i32 => {
                f.write_str("COMMAND_BUFFER_INHERITANCE_VIEWPORT_SCISSOR_INFO")
            }
            1000281000i32 => {
                f.write_str("PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_FEATURES")
            }
            1000282000i32 => {
                f.write_str("COMMAND_BUFFER_INHERITANCE_RENDER_PASS_TRANSFORM_INFO")
            }
            1000282001i32 => f.write_str("RENDER_PASS_TRANSFORM_BEGIN_INFO"),
            1000283000i32 => f.write_str("PHYSICAL_DEVICE_DEPTH_BIAS_CONTROL_FEATURES"),
            1000283001i32 => f.write_str("DEPTH_BIAS_INFO"),
            1000283002i32 => f.write_str("DEPTH_BIAS_REPRESENTATION_INFO"),
            1000284000i32 => f.write_str("PHYSICAL_DEVICE_DEVICE_MEMORY_REPORT_FEATURES"),
            1000284001i32 => f.write_str("DEVICE_DEVICE_MEMORY_REPORT_CREATE_INFO"),
            1000284002i32 => f.write_str("DEVICE_MEMORY_REPORT_CALLBACK_DATA"),
            1000287000i32 => f.write_str("SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO"),
            1000287001i32 => {
                f.write_str("PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_PROPERTIES")
            }
            1000287002i32 => f.write_str("PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES"),
            1000288000i32 => {
                f.write_str("PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_3D_FEATURES")
            }
            1000290000i32 => f.write_str("PIPELINE_LIBRARY_CREATE_INFO"),
            1000292000i32 => f.write_str("PHYSICAL_DEVICE_PRESENT_BARRIER_FEATURES"),
            1000292001i32 => f.write_str("SURFACE_CAPABILITIES_PRESENT_BARRIER"),
            1000292002i32 => f.write_str("SWAPCHAIN_PRESENT_BARRIER_CREATE_INFO"),
            1000294000i32 => f.write_str("PRESENT_ID"),
            1000294001i32 => f.write_str("PHYSICAL_DEVICE_PRESENT_ID_FEATURES"),
            1000299000i32 => f.write_str("VIDEO_ENCODE_INFO"),
            1000299001i32 => f.write_str("VIDEO_ENCODE_RATE_CONTROL_INFO"),
            1000299002i32 => f.write_str("VIDEO_ENCODE_RATE_CONTROL_LAYER_INFO"),
            1000299003i32 => f.write_str("VIDEO_ENCODE_CAPABILITIES"),
            1000299004i32 => f.write_str("VIDEO_ENCODE_USAGE_INFO"),
            1000299005i32 => f.write_str("QUERY_POOL_VIDEO_ENCODE_FEEDBACK_CREATE_INFO"),
            1000299006i32 => {
                f.write_str("PHYSICAL_DEVICE_VIDEO_ENCODE_QUALITY_LEVEL_INFO")
            }
            1000299007i32 => f.write_str("VIDEO_ENCODE_QUALITY_LEVEL_PROPERTIES"),
            1000299008i32 => f.write_str("VIDEO_ENCODE_QUALITY_LEVEL_INFO"),
            1000299009i32 => f.write_str("VIDEO_ENCODE_SESSION_PARAMETERS_GET_INFO"),
            1000299010i32 => f.write_str("VIDEO_ENCODE_SESSION_PARAMETERS_FEEDBACK_INFO"),
            1000300000i32 => f.write_str("PHYSICAL_DEVICE_DIAGNOSTICS_CONFIG_FEATURES"),
            1000300001i32 => f.write_str("DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO"),
            1000307000i32 => f.write_str("CUDA_MODULE_CREATE_INFO"),
            1000307001i32 => f.write_str("CUDA_FUNCTION_CREATE_INFO"),
            1000307002i32 => f.write_str("CUDA_LAUNCH_INFO"),
            1000307003i32 => f.write_str("PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES"),
            1000307004i32 => f.write_str("PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_PROPERTIES"),
            1000308000i32 => f.write_str("REFRESH_OBJECT_LIST"),
            1000309000i32 => f.write_str("PHYSICAL_DEVICE_TILE_SHADING_FEATURES"),
            1000309001i32 => f.write_str("PHYSICAL_DEVICE_TILE_SHADING_PROPERTIES"),
            1000309002i32 => f.write_str("RENDER_PASS_TILE_SHADING_CREATE_INFO"),
            1000309003i32 => f.write_str("PER_TILE_BEGIN_INFO"),
            1000309004i32 => f.write_str("PER_TILE_END_INFO"),
            1000309005i32 => f.write_str("DISPATCH_TILE_INFO"),
            1000310000i32 => f.write_str("QUERY_LOW_LATENCY_SUPPORT"),
            1000311000i32 => f.write_str("EXPORT_METAL_OBJECT_CREATE_INFO"),
            1000311001i32 => f.write_str("EXPORT_METAL_OBJECTS_INFO"),
            1000311002i32 => f.write_str("EXPORT_METAL_DEVICE_INFO"),
            1000311003i32 => f.write_str("EXPORT_METAL_COMMAND_QUEUE_INFO"),
            1000311004i32 => f.write_str("EXPORT_METAL_BUFFER_INFO"),
            1000311005i32 => f.write_str("IMPORT_METAL_BUFFER_INFO"),
            1000311006i32 => f.write_str("EXPORT_METAL_TEXTURE_INFO"),
            1000311007i32 => f.write_str("IMPORT_METAL_TEXTURE_INFO"),
            1000311008i32 => f.write_str("EXPORT_METAL_IO_SURFACE_INFO"),
            1000311009i32 => f.write_str("IMPORT_METAL_IO_SURFACE_INFO"),
            1000311010i32 => f.write_str("EXPORT_METAL_SHARED_EVENT_INFO"),
            1000311011i32 => f.write_str("IMPORT_METAL_SHARED_EVENT_INFO"),
            1000316000i32 => f.write_str("PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES"),
            1000316001i32 => {
                f.write_str("PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_DENSITY_MAP_PROPERTIES")
            }
            1000316002i32 => f.write_str("PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES"),
            1000316003i32 => f.write_str("DESCRIPTOR_ADDRESS_INFO"),
            1000316004i32 => f.write_str("DESCRIPTOR_GET_INFO"),
            1000316005i32 => f.write_str("BUFFER_CAPTURE_DESCRIPTOR_DATA_INFO"),
            1000316006i32 => f.write_str("IMAGE_CAPTURE_DESCRIPTOR_DATA_INFO"),
            1000316007i32 => f.write_str("IMAGE_VIEW_CAPTURE_DESCRIPTOR_DATA_INFO"),
            1000316008i32 => f.write_str("SAMPLER_CAPTURE_DESCRIPTOR_DATA_INFO"),
            1000316010i32 => f.write_str("OPAQUE_CAPTURE_DESCRIPTOR_DATA_CREATE_INFO"),
            1000316011i32 => f.write_str("DESCRIPTOR_BUFFER_BINDING_INFO"),
            1000316012i32 => {
                f.write_str("DESCRIPTOR_BUFFER_BINDING_PUSH_DESCRIPTOR_BUFFER_HANDLE")
            }
            1000316009i32 => {
                f.write_str("ACCELERATION_STRUCTURE_CAPTURE_DESCRIPTOR_DATA_INFO")
            }
            1000318000i32 => f.write_str("DEVICE_MEMORY_COPY"),
            1000318001i32 => f.write_str("COPY_DEVICE_MEMORY_INFO"),
            1000318002i32 => f.write_str("DEVICE_MEMORY_IMAGE_COPY"),
            1000318003i32 => f.write_str("COPY_DEVICE_MEMORY_IMAGE_INFO"),
            1000318004i32 => f.write_str("MEMORY_RANGE_BARRIERS_INFO"),
            1000318005i32 => f.write_str("MEMORY_RANGE_BARRIER"),
            1000318006i32 => {
                f.write_str("PHYSICAL_DEVICE_DEVICE_ADDRESS_COMMANDS_FEATURES")
            }
            1000318007i32 => f.write_str("BIND_INDEX_BUFFER_3_INFO"),
            1000318008i32 => f.write_str("BIND_VERTEX_BUFFER_3_INFO"),
            1000318009i32 => f.write_str("DRAW_INDIRECT_2_INFO"),
            1000318010i32 => f.write_str("DRAW_INDIRECT_COUNT_2_INFO"),
            1000318011i32 => f.write_str("DISPATCH_INDIRECT_2_INFO"),
            1000318012i32 => f.write_str("CONDITIONAL_RENDERING_BEGIN_INFO_2"),
            1000318013i32 => f.write_str("BIND_TRANSFORM_FEEDBACK_BUFFER_2_INFO"),
            1000318014i32 => f.write_str("MEMORY_MARKER_INFO"),
            1000318015i32 => f.write_str("ACCELERATION_STRUCTURE_CREATE_INFO_2"),
            1000320000i32 => {
                f.write_str("PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES")
            }
            1000320001i32 => {
                f.write_str("PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES")
            }
            1000320002i32 => f.write_str("GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO"),
            1000321000i32 => {
                f.write_str(
                    "PHYSICAL_DEVICE_SHADER_EARLY_AND_LATE_FRAGMENT_TESTS_FEATURES",
                )
            }
            1000322000i32 => {
                f.write_str("PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_PROPERTIES")
            }
            1000323000i32 => {
                f.write_str(
                    "PHYSICAL_DEVICE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_FEATURES",
                )
            }
            1000326000i32 => {
                f.write_str("PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_ENUMS_PROPERTIES")
            }
            1000326001i32 => {
                f.write_str("PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_ENUMS_FEATURES")
            }
            1000326002i32 => {
                f.write_str("PIPELINE_FRAGMENT_SHADING_RATE_ENUM_STATE_CREATE_INFO")
            }
            1000327000i32 => {
                f.write_str("ACCELERATION_STRUCTURE_GEOMETRY_MOTION_TRIANGLES_DATA")
            }
            1000327001i32 => {
                f.write_str("PHYSICAL_DEVICE_RAY_TRACING_MOTION_BLUR_FEATURES")
            }
            1000327002i32 => f.write_str("ACCELERATION_STRUCTURE_MOTION_INFO"),
            1000330000i32 => {
                f.write_str("PHYSICAL_DEVICE_YCBCR_2_PLANE_444_FORMATS_FEATURES")
            }
            1000332000i32 => {
                f.write_str("PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_2_FEATURES")
            }
            1000332001i32 => {
                f.write_str("PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_2_PROPERTIES")
            }
            1000333000i32 => f.write_str("COPY_COMMAND_TRANSFORM_INFO"),
            1000336000i32 => {
                f.write_str("PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES")
            }
            1000338000i32 => {
                f.write_str("PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_FEATURES")
            }
            1000338001i32 => f.write_str("IMAGE_COMPRESSION_CONTROL"),
            1000338004i32 => f.write_str("IMAGE_COMPRESSION_PROPERTIES"),
            1000339000i32 => {
                f.write_str("PHYSICAL_DEVICE_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_FEATURES")
            }
            1000340000i32 => f.write_str("PHYSICAL_DEVICE_4444_FORMATS_FEATURES"),
            1000341000i32 => f.write_str("PHYSICAL_DEVICE_FAULT_FEATURES"),
            1000341001i32 => f.write_str("DEVICE_FAULT_COUNTS"),
            1000341002i32 => f.write_str("DEVICE_FAULT_INFO"),
            1000344000i32 => f.write_str("PHYSICAL_DEVICE_RGBA10X6_FORMATS_FEATURES"),
            1000346000i32 => f.write_str("DIRECTFB_SURFACE_CREATE_INFO"),
            1000352000i32 => {
                f.write_str("PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES")
            }
            1000352001i32 => f.write_str("VERTEX_INPUT_BINDING_DESCRIPTION_2"),
            1000352002i32 => f.write_str("VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2"),
            1000353000i32 => f.write_str("PHYSICAL_DEVICE_DRM_PROPERTIES"),
            1000354000i32 => {
                f.write_str("PHYSICAL_DEVICE_ADDRESS_BINDING_REPORT_FEATURES")
            }
            1000354001i32 => f.write_str("DEVICE_ADDRESS_BINDING_CALLBACK_DATA"),
            1000355000i32 => f.write_str("PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES"),
            1000355001i32 => {
                f.write_str("PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO")
            }
            1000356000i32 => {
                f.write_str("PHYSICAL_DEVICE_PRIMITIVE_TOPOLOGY_LIST_RESTART_FEATURES")
            }
            1000364000i32 => f.write_str("IMPORT_MEMORY_ZIRCON_HANDLE_INFO"),
            1000364001i32 => f.write_str("MEMORY_ZIRCON_HANDLE_PROPERTIES"),
            1000364002i32 => f.write_str("MEMORY_GET_ZIRCON_HANDLE_INFO"),
            1000365000i32 => f.write_str("IMPORT_SEMAPHORE_ZIRCON_HANDLE_INFO"),
            1000365001i32 => f.write_str("SEMAPHORE_GET_ZIRCON_HANDLE_INFO"),
            1000366000i32 => f.write_str("BUFFER_COLLECTION_CREATE_INFO"),
            1000366001i32 => f.write_str("IMPORT_MEMORY_BUFFER_COLLECTION"),
            1000366002i32 => f.write_str("BUFFER_COLLECTION_IMAGE_CREATE_INFO"),
            1000366003i32 => f.write_str("BUFFER_COLLECTION_PROPERTIES"),
            1000366004i32 => f.write_str("BUFFER_CONSTRAINTS_INFO"),
            1000366005i32 => f.write_str("BUFFER_COLLECTION_BUFFER_CREATE_INFO"),
            1000366006i32 => f.write_str("IMAGE_CONSTRAINTS_INFO"),
            1000366007i32 => f.write_str("IMAGE_FORMAT_CONSTRAINTS_INFO"),
            1000366008i32 => f.write_str("SYSMEM_COLOR_SPACE"),
            1000366009i32 => f.write_str("BUFFER_COLLECTION_CONSTRAINTS_INFO"),
            1000369000i32 => f.write_str("SUBPASS_SHADING_PIPELINE_CREATE_INFO"),
            1000369001i32 => f.write_str("PHYSICAL_DEVICE_SUBPASS_SHADING_FEATURES"),
            1000369002i32 => f.write_str("PHYSICAL_DEVICE_SUBPASS_SHADING_PROPERTIES"),
            1000370000i32 => f.write_str("PHYSICAL_DEVICE_INVOCATION_MASK_FEATURES"),
            1000371000i32 => f.write_str("MEMORY_GET_REMOTE_ADDRESS_INFO"),
            1000371001i32 => f.write_str("PHYSICAL_DEVICE_EXTERNAL_MEMORY_RDMA_FEATURES"),
            1000372000i32 => f.write_str("PIPELINE_PROPERTIES_IDENTIFIER"),
            1000372001i32 => f.write_str("PHYSICAL_DEVICE_PIPELINE_PROPERTIES_FEATURES"),
            1000373000i32 => f.write_str("IMPORT_FENCE_SCI_SYNC_INFO"),
            1000373001i32 => f.write_str("EXPORT_FENCE_SCI_SYNC_INFO"),
            1000373002i32 => f.write_str("FENCE_GET_SCI_SYNC_INFO"),
            1000373003i32 => f.write_str("SCI_SYNC_ATTRIBUTES_INFO"),
            1000373004i32 => f.write_str("IMPORT_SEMAPHORE_SCI_SYNC_INFO"),
            1000373005i32 => f.write_str("EXPORT_SEMAPHORE_SCI_SYNC_INFO"),
            1000373006i32 => f.write_str("SEMAPHORE_GET_SCI_SYNC_INFO"),
            1000373007i32 => f.write_str("PHYSICAL_DEVICE_EXTERNAL_SCI_SYNC_FEATURES"),
            1000374000i32 => f.write_str("IMPORT_MEMORY_SCI_BUF_INFO"),
            1000374001i32 => f.write_str("EXPORT_MEMORY_SCI_BUF_INFO"),
            1000374002i32 => f.write_str("MEMORY_GET_SCI_BUF_INFO"),
            1000374003i32 => f.write_str("MEMORY_SCI_BUF_PROPERTIES"),
            1000374004i32 => {
                f.write_str("PHYSICAL_DEVICE_EXTERNAL_MEMORY_SCI_BUF_FEATURES")
            }
            1000375000i32 => f.write_str("PHYSICAL_DEVICE_FRAME_BOUNDARY_FEATURES"),
            1000375001i32 => f.write_str("FRAME_BOUNDARY"),
            1000376000i32 => {
                f.write_str(
                    "PHYSICAL_DEVICE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_FEATURES",
                )
            }
            1000376001i32 => f.write_str("SUBPASS_RESOLVE_PERFORMANCE_QUERY"),
            1000376002i32 => f.write_str("MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_INFO"),
            1000377000i32 => {
                f.write_str("PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES")
            }
            1000378000i32 => f.write_str("SCREEN_SURFACE_CREATE_INFO"),
            1000381000i32 => f.write_str("PHYSICAL_DEVICE_COLOR_WRITE_ENABLE_FEATURES"),
            1000381001i32 => f.write_str("PIPELINE_COLOR_WRITE_CREATE_INFO"),
            1000382000i32 => {
                f.write_str("PHYSICAL_DEVICE_PRIMITIVES_GENERATED_QUERY_FEATURES")
            }
            1000386000i32 => {
                f.write_str("PHYSICAL_DEVICE_RAY_TRACING_MAINTENANCE_1_FEATURES")
            }
            1000387000i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES")
            }
            1000390000i32 => {
                f.write_str("PHYSICAL_DEVICE_VIDEO_ENCODE_RGB_CONVERSION_FEATURES")
            }
            1000390001i32 => f.write_str("VIDEO_ENCODE_RGB_CONVERSION_CAPABILITIES"),
            1000390002i32 => f.write_str("VIDEO_ENCODE_PROFILE_RGB_CONVERSION_INFO"),
            1000390003i32 => {
                f.write_str("VIDEO_ENCODE_SESSION_RGB_CONVERSION_CREATE_INFO")
            }
            1000391000i32 => f.write_str("PHYSICAL_DEVICE_IMAGE_VIEW_MIN_LOD_FEATURES"),
            1000391001i32 => f.write_str("IMAGE_VIEW_MIN_LOD_CREATE_INFO"),
            1000392000i32 => f.write_str("PHYSICAL_DEVICE_MULTI_DRAW_FEATURES"),
            1000392001i32 => f.write_str("PHYSICAL_DEVICE_MULTI_DRAW_PROPERTIES"),
            1000393000i32 => f.write_str("PHYSICAL_DEVICE_IMAGE_2D_VIEW_OF_3D_FEATURES"),
            1000395000i32 => f.write_str("PHYSICAL_DEVICE_SHADER_TILE_IMAGE_FEATURES"),
            1000395001i32 => f.write_str("PHYSICAL_DEVICE_SHADER_TILE_IMAGE_PROPERTIES"),
            1000396000i32 => f.write_str("MICROMAP_BUILD_INFO"),
            1000396001i32 => f.write_str("MICROMAP_VERSION_INFO"),
            1000396002i32 => f.write_str("COPY_MICROMAP_INFO"),
            1000396003i32 => f.write_str("COPY_MICROMAP_TO_MEMORY_INFO"),
            1000396004i32 => f.write_str("COPY_MEMORY_TO_MICROMAP_INFO"),
            1000396005i32 => f.write_str("PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES"),
            1000396006i32 => f.write_str("PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES"),
            1000396007i32 => f.write_str("MICROMAP_CREATE_INFO"),
            1000396008i32 => f.write_str("MICROMAP_BUILD_SIZES_INFO"),
            1000396009i32 => {
                f.write_str("ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP")
            }
            1000397000i32 => {
                f.write_str("PHYSICAL_DEVICE_DISPLACEMENT_MICROMAP_FEATURES")
            }
            1000397001i32 => {
                f.write_str("PHYSICAL_DEVICE_DISPLACEMENT_MICROMAP_PROPERTIES")
            }
            1000397002i32 => {
                f.write_str("ACCELERATION_STRUCTURE_TRIANGLES_DISPLACEMENT_MICROMAP")
            }
            1000404000i32 => {
                f.write_str("PHYSICAL_DEVICE_CLUSTER_CULLING_SHADER_FEATURES")
            }
            1000404001i32 => {
                f.write_str("PHYSICAL_DEVICE_CLUSTER_CULLING_SHADER_PROPERTIES")
            }
            1000404002i32 => {
                f.write_str("PHYSICAL_DEVICE_CLUSTER_CULLING_SHADER_VRS_FEATURES")
            }
            1000411000i32 => f.write_str("PHYSICAL_DEVICE_BORDER_COLOR_SWIZZLE_FEATURES"),
            1000411001i32 => {
                f.write_str("SAMPLER_BORDER_COLOR_COMPONENT_MAPPING_CREATE_INFO")
            }
            1000412000i32 => {
                f.write_str("PHYSICAL_DEVICE_PAGEABLE_DEVICE_LOCAL_MEMORY_FEATURES")
            }
            1000417000i32 => f.write_str("DEVICE_QUEUE_SHADER_CORE_CONTROL_CREATE_INFO"),
            1000417001i32 => f.write_str("PHYSICAL_DEVICE_SCHEDULING_CONTROLS_FEATURES"),
            1000417002i32 => {
                f.write_str("PHYSICAL_DEVICE_SCHEDULING_CONTROLS_PROPERTIES")
            }
            1000418000i32 => {
                f.write_str("PHYSICAL_DEVICE_IMAGE_SLICED_VIEW_OF_3D_FEATURES")
            }
            1000418001i32 => f.write_str("IMAGE_VIEW_SLICED_CREATE_INFO"),
            1000420000i32 => {
                f.write_str("PHYSICAL_DEVICE_DESCRIPTOR_SET_HOST_MAPPING_FEATURES")
            }
            1000420001i32 => f.write_str("DESCRIPTOR_SET_BINDING_REFERENCE"),
            1000420002i32 => f.write_str("DESCRIPTOR_SET_LAYOUT_HOST_MAPPING_INFO"),
            1000422000i32 => {
                f.write_str("PHYSICAL_DEVICE_NON_SEAMLESS_CUBE_MAP_FEATURES")
            }
            1000424000i32 => f.write_str("PHYSICAL_DEVICE_RENDER_PASS_STRIPED_FEATURES"),
            1000424001i32 => {
                f.write_str("PHYSICAL_DEVICE_RENDER_PASS_STRIPED_PROPERTIES")
            }
            1000424002i32 => f.write_str("RENDER_PASS_STRIPE_BEGIN_INFO"),
            1000424003i32 => f.write_str("RENDER_PASS_STRIPE_INFO"),
            1000424004i32 => f.write_str("RENDER_PASS_STRIPE_SUBMIT_INFO"),
            1000426000i32 => f.write_str("PHYSICAL_DEVICE_COPY_MEMORY_INDIRECT_FEATURES"),
            1000428000i32 => {
                f.write_str("PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_COMPUTE_FEATURES")
            }
            1000428001i32 => f.write_str("COMPUTE_PIPELINE_INDIRECT_BUFFER_INFO"),
            1000428002i32 => f.write_str("PIPELINE_INDIRECT_DEVICE_ADDRESS_INFO"),
            1000429008i32 => {
                f.write_str("PHYSICAL_DEVICE_RAY_TRACING_LINEAR_SWEPT_SPHERES_FEATURES")
            }
            1000429009i32 => {
                f.write_str("ACCELERATION_STRUCTURE_GEOMETRY_LINEAR_SWEPT_SPHERES_DATA")
            }
            1000429010i32 => f.write_str("ACCELERATION_STRUCTURE_GEOMETRY_SPHERES_DATA"),
            1000430000i32 => {
                f.write_str("PHYSICAL_DEVICE_LINEAR_COLOR_ATTACHMENT_FEATURES")
            }
            1000434000i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES")
            }
            1000435000i32 => f.write_str("APPLICATION_PARAMETERS"),
            1000437000i32 => {
                f.write_str(
                    "PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_SWAPCHAIN_FEATURES",
                )
            }
            1000440000i32 => f.write_str("PHYSICAL_DEVICE_IMAGE_PROCESSING_FEATURES"),
            1000440001i32 => f.write_str("PHYSICAL_DEVICE_IMAGE_PROCESSING_PROPERTIES"),
            1000440002i32 => f.write_str("IMAGE_VIEW_SAMPLE_WEIGHT_CREATE_INFO"),
            1000451000i32 => {
                f.write_str("PHYSICAL_DEVICE_NESTED_COMMAND_BUFFER_FEATURES")
            }
            1000451001i32 => {
                f.write_str("PHYSICAL_DEVICE_NESTED_COMMAND_BUFFER_PROPERTIES")
            }
            1000452000i32 => f.write_str("NATIVE_BUFFER_USAGE"),
            1000452001i32 => f.write_str("NATIVE_BUFFER_PROPERTIES"),
            1000452002i32 => f.write_str("NATIVE_BUFFER_FORMAT_PROPERTIES"),
            1000452003i32 => f.write_str("IMPORT_NATIVE_BUFFER_INFO"),
            1000452004i32 => f.write_str("MEMORY_GET_NATIVE_BUFFER_INFO"),
            1000453000i32 => f.write_str("EXTERNAL_MEMORY_ACQUIRE_UNMODIFIED"),
            1000455000i32 => {
                f.write_str("PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES")
            }
            1000455001i32 => {
                f.write_str("PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_PROPERTIES")
            }
            1000458000i32 => {
                f.write_str("PHYSICAL_DEVICE_SUBPASS_MERGE_FEEDBACK_FEATURES")
            }
            1000458001i32 => f.write_str("RENDER_PASS_CREATION_CONTROL"),
            1000458002i32 => f.write_str("RENDER_PASS_CREATION_FEEDBACK_CREATE_INFO"),
            1000458003i32 => f.write_str("RENDER_PASS_SUBPASS_FEEDBACK_CREATE_INFO"),
            1000459000i32 => f.write_str("DIRECT_DRIVER_LOADING_INFO"),
            1000459001i32 => f.write_str("DIRECT_DRIVER_LOADING_LIST"),
            1000460000i32 => f.write_str("TENSOR_CREATE_INFO"),
            1000460001i32 => f.write_str("TENSOR_VIEW_CREATE_INFO"),
            1000460002i32 => f.write_str("BIND_TENSOR_MEMORY_INFO"),
            1000460003i32 => f.write_str("WRITE_DESCRIPTOR_SET_TENSOR"),
            1000460004i32 => f.write_str("PHYSICAL_DEVICE_TENSOR_PROPERTIES"),
            1000460005i32 => f.write_str("TENSOR_FORMAT_PROPERTIES"),
            1000460006i32 => f.write_str("TENSOR_DESCRIPTION"),
            1000460007i32 => f.write_str("TENSOR_MEMORY_REQUIREMENTS_INFO"),
            1000460008i32 => f.write_str("TENSOR_MEMORY_BARRIER"),
            1000460009i32 => f.write_str("PHYSICAL_DEVICE_TENSOR_FEATURES"),
            1000460010i32 => f.write_str("DEVICE_TENSOR_MEMORY_REQUIREMENTS"),
            1000460011i32 => f.write_str("COPY_TENSOR_INFO"),
            1000460012i32 => f.write_str("TENSOR_COPY"),
            1000460013i32 => f.write_str("TENSOR_DEPENDENCY_INFO"),
            1000460014i32 => f.write_str("MEMORY_DEDICATED_ALLOCATE_INFO_TENSOR"),
            1000460015i32 => f.write_str("PHYSICAL_DEVICE_EXTERNAL_TENSOR_INFO"),
            1000460016i32 => f.write_str("EXTERNAL_TENSOR_PROPERTIES"),
            1000460017i32 => f.write_str("EXTERNAL_MEMORY_TENSOR_CREATE_INFO"),
            1000460018i32 => {
                f.write_str("PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_TENSOR_FEATURES")
            }
            1000460019i32 => {
                f.write_str("PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_TENSOR_PROPERTIES")
            }
            1000460020i32 => f.write_str("DESCRIPTOR_GET_TENSOR_INFO"),
            1000460021i32 => f.write_str("TENSOR_CAPTURE_DESCRIPTOR_DATA_INFO"),
            1000460022i32 => f.write_str("TENSOR_VIEW_CAPTURE_DESCRIPTOR_DATA_INFO"),
            1000460023i32 => f.write_str("FRAME_BOUNDARY_TENSORS"),
            1000462000i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_MODULE_IDENTIFIER_FEATURES")
            }
            1000462001i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_MODULE_IDENTIFIER_PROPERTIES")
            }
            1000462002i32 => {
                f.write_str("PIPELINE_SHADER_STAGE_MODULE_IDENTIFIER_CREATE_INFO")
            }
            1000462003i32 => f.write_str("SHADER_MODULE_IDENTIFIER"),
            1000464000i32 => f.write_str("PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES"),
            1000464001i32 => f.write_str("PHYSICAL_DEVICE_OPTICAL_FLOW_PROPERTIES"),
            1000464002i32 => f.write_str("OPTICAL_FLOW_IMAGE_FORMAT_INFO"),
            1000464003i32 => f.write_str("OPTICAL_FLOW_IMAGE_FORMAT_PROPERTIES"),
            1000464004i32 => f.write_str("OPTICAL_FLOW_SESSION_CREATE_INFO"),
            1000464005i32 => f.write_str("OPTICAL_FLOW_EXECUTE_INFO"),
            1000464010i32 => f.write_str("OPTICAL_FLOW_SESSION_CREATE_PRIVATE_DATA_INFO"),
            1000465000i32 => f.write_str("PHYSICAL_DEVICE_LEGACY_DITHERING_FEATURES"),
            1000468000i32 => {
                f.write_str("PHYSICAL_DEVICE_EXTERNAL_FORMAT_RESOLVE_FEATURES")
            }
            1000468001i32 => {
                f.write_str("PHYSICAL_DEVICE_EXTERNAL_FORMAT_RESOLVE_PROPERTIES")
            }
            1000468002i32 => {
                f.write_str("ANDROID_HARDWARE_BUFFER_FORMAT_RESOLVE_PROPERTIES")
            }
            1000476000i32 => f.write_str("PHYSICAL_DEVICE_ANTI_LAG_FEATURES"),
            1000476001i32 => f.write_str("ANTI_LAG_DATA"),
            1000476002i32 => f.write_str("ANTI_LAG_PRESENTATION_INFO"),
            1000478000i32 => {
                f.write_str("PHYSICAL_DEVICE_DENSE_GEOMETRY_FORMAT_FEATURES")
            }
            1000478001i32 => {
                f.write_str(
                    "ACCELERATION_STRUCTURE_DENSE_GEOMETRY_FORMAT_TRIANGLES_DATA",
                )
            }
            1000479000i32 => f.write_str("SURFACE_CAPABILITIES_PRESENT_ID_2"),
            1000479001i32 => f.write_str("PRESENT_ID_2"),
            1000479002i32 => f.write_str("PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES"),
            1000480000i32 => f.write_str("SURFACE_CAPABILITIES_PRESENT_WAIT_2"),
            1000480001i32 => f.write_str("PHYSICAL_DEVICE_PRESENT_WAIT_2_FEATURES"),
            1000480002i32 => f.write_str("PRESENT_WAIT_2_INFO"),
            1000481000i32 => {
                f.write_str("PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES")
            }
            1000482000i32 => f.write_str("PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES"),
            1000482001i32 => f.write_str("PHYSICAL_DEVICE_SHADER_OBJECT_PROPERTIES"),
            1000482002i32 => f.write_str("SHADER_CREATE_INFO"),
            1000483000i32 => f.write_str("PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES"),
            1000483001i32 => f.write_str("PIPELINE_BINARY_CREATE_INFO"),
            1000483002i32 => f.write_str("PIPELINE_BINARY_INFO"),
            1000483003i32 => f.write_str("PIPELINE_BINARY_KEY"),
            1000483004i32 => f.write_str("PHYSICAL_DEVICE_PIPELINE_BINARY_PROPERTIES"),
            1000483005i32 => f.write_str("RELEASE_CAPTURED_PIPELINE_DATA_INFO"),
            1000483006i32 => f.write_str("PIPELINE_BINARY_DATA_INFO"),
            1000483007i32 => f.write_str("PIPELINE_CREATE_INFO"),
            1000483008i32 => f.write_str("DEVICE_PIPELINE_BINARY_INTERNAL_CACHE_CONTROL"),
            1000483009i32 => f.write_str("PIPELINE_BINARY_HANDLES_INFO"),
            1000484000i32 => f.write_str("PHYSICAL_DEVICE_TILE_PROPERTIES_FEATURES"),
            1000484001i32 => f.write_str("TILE_PROPERTIES"),
            1000485000i32 => f.write_str("PHYSICAL_DEVICE_AMIGO_PROFILING_FEATURES"),
            1000485001i32 => f.write_str("AMIGO_PROFILING_SUBMIT_INFO"),
            1000488000i32 => {
                f.write_str("PHYSICAL_DEVICE_MULTIVIEW_PER_VIEW_VIEWPORTS_FEATURES")
            }
            1000489000i32 => f.write_str("SEMAPHORE_SCI_SYNC_POOL_CREATE_INFO"),
            1000489001i32 => f.write_str("SEMAPHORE_SCI_SYNC_CREATE_INFO"),
            1000489002i32 => f.write_str("PHYSICAL_DEVICE_EXTERNAL_SCI_SYNC_2_FEATURES"),
            1000489003i32 => {
                f.write_str("DEVICE_SEMAPHORE_SCI_SYNC_POOL_RESERVATION_CREATE_INFO")
            }
            1000490000i32 => {
                f.write_str("PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES")
            }
            1000490001i32 => {
                f.write_str("PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES")
            }
            1000491000i32 => f.write_str("PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES"),
            1000491001i32 => f.write_str("PHYSICAL_DEVICE_COOPERATIVE_VECTOR_PROPERTIES"),
            1000491002i32 => f.write_str("COOPERATIVE_VECTOR_PROPERTIES"),
            1000491004i32 => f.write_str("CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO"),
            1000492000i32 => {
                f.write_str("PHYSICAL_DEVICE_EXTENDED_SPARSE_ADDRESS_SPACE_FEATURES")
            }
            1000492001i32 => {
                f.write_str("PHYSICAL_DEVICE_EXTENDED_SPARSE_ADDRESS_SPACE_PROPERTIES")
            }
            1000495000i32 => {
                f.write_str("PHYSICAL_DEVICE_LEGACY_VERTEX_ATTRIBUTES_FEATURES")
            }
            1000495001i32 => {
                f.write_str("PHYSICAL_DEVICE_LEGACY_VERTEX_ATTRIBUTES_PROPERTIES")
            }
            1000496000i32 => f.write_str("LAYER_SETTINGS_CREATE_INFO"),
            1000497000i32 => f.write_str("PHYSICAL_DEVICE_SHADER_CORE_BUILTINS_FEATURES"),
            1000497001i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_CORE_BUILTINS_PROPERTIES")
            }
            1000498000i32 => {
                f.write_str("PHYSICAL_DEVICE_PIPELINE_LIBRARY_GROUP_HANDLES_FEATURES")
            }
            1000499000i32 => {
                f.write_str(
                    "PHYSICAL_DEVICE_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_FEATURES",
                )
            }
            1000504000i32 => {
                f.write_str("PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES")
            }
            1000505000i32 => f.write_str("LATENCY_SLEEP_MODE_INFO"),
            1000505001i32 => f.write_str("LATENCY_SLEEP_INFO"),
            1000505002i32 => f.write_str("SET_LATENCY_MARKER_INFO"),
            1000505003i32 => f.write_str("GET_LATENCY_MARKER_INFO"),
            1000505004i32 => f.write_str("LATENCY_TIMINGS_FRAME_REPORT"),
            1000505005i32 => f.write_str("LATENCY_SUBMISSION_PRESENT_ID"),
            1000505006i32 => f.write_str("OUT_OF_BAND_QUEUE_TYPE_INFO"),
            1000505007i32 => f.write_str("SWAPCHAIN_LATENCY_CREATE_INFO"),
            1000505008i32 => f.write_str("LATENCY_SURFACE_CAPABILITIES"),
            1000507000i32 => f.write_str("DATA_GRAPH_PIPELINE_CREATE_INFO"),
            1000507001i32 => f.write_str("DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO"),
            1000507002i32 => f.write_str("DATA_GRAPH_PIPELINE_RESOURCE_INFO"),
            1000507003i32 => f.write_str("DATA_GRAPH_PIPELINE_CONSTANT"),
            1000507004i32 => {
                f.write_str("DATA_GRAPH_PIPELINE_SESSION_MEMORY_REQUIREMENTS_INFO")
            }
            1000507005i32 => f.write_str("BIND_DATA_GRAPH_PIPELINE_SESSION_MEMORY_INFO"),
            1000507006i32 => f.write_str("PHYSICAL_DEVICE_DATA_GRAPH_FEATURES"),
            1000507007i32 => f.write_str("DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO"),
            1000507008i32 => f.write_str("DATA_GRAPH_PIPELINE_PROPERTY_QUERY_RESULT"),
            1000507009i32 => f.write_str("DATA_GRAPH_PIPELINE_INFO"),
            1000507010i32 => {
                f.write_str("DATA_GRAPH_PIPELINE_COMPILER_CONTROL_CREATE_INFO")
            }
            1000507011i32 => {
                f.write_str("DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENTS_INFO")
            }
            1000507012i32 => {
                f.write_str("DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENT")
            }
            1000507013i32 => f.write_str("DATA_GRAPH_PIPELINE_IDENTIFIER_CREATE_INFO"),
            1000507014i32 => f.write_str("DATA_GRAPH_PIPELINE_DISPATCH_INFO"),
            1000507016i32 => f.write_str("DATA_GRAPH_PROCESSING_ENGINE_CREATE_INFO"),
            1000507017i32 => {
                f.write_str("QUEUE_FAMILY_DATA_GRAPH_PROCESSING_ENGINE_PROPERTIES")
            }
            1000507018i32 => f.write_str("QUEUE_FAMILY_DATA_GRAPH_PROPERTIES"),
            1000507019i32 => {
                f.write_str(
                    "PHYSICAL_DEVICE_QUEUE_FAMILY_DATA_GRAPH_PROCESSING_ENGINE_INFO",
                )
            }
            1000507015i32 => {
                f.write_str(
                    "DATA_GRAPH_PIPELINE_CONSTANT_TENSOR_SEMI_STRUCTURED_SPARSITY_INFO",
                )
            }
            1000510000i32 => {
                f.write_str("PHYSICAL_DEVICE_MULTIVIEW_PER_VIEW_RENDER_AREAS_FEATURES")
            }
            1000510001i32 => {
                f.write_str("MULTIVIEW_PER_VIEW_RENDER_AREAS_RENDER_PASS_BEGIN_INFO")
            }
            1000511000i32 => {
                f.write_str("PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_PROPERTIES")
            }
            1000512000i32 => f.write_str("VIDEO_DECODE_AV1_CAPABILITIES"),
            1000512001i32 => f.write_str("VIDEO_DECODE_AV1_PICTURE_INFO"),
            1000512003i32 => f.write_str("VIDEO_DECODE_AV1_PROFILE_INFO"),
            1000512004i32 => {
                f.write_str("VIDEO_DECODE_AV1_SESSION_PARAMETERS_CREATE_INFO")
            }
            1000512005i32 => f.write_str("VIDEO_DECODE_AV1_DPB_SLOT_INFO"),
            1000513000i32 => f.write_str("VIDEO_ENCODE_AV1_CAPABILITIES"),
            1000513001i32 => {
                f.write_str("VIDEO_ENCODE_AV1_SESSION_PARAMETERS_CREATE_INFO")
            }
            1000513002i32 => f.write_str("VIDEO_ENCODE_AV1_PICTURE_INFO"),
            1000513003i32 => f.write_str("VIDEO_ENCODE_AV1_DPB_SLOT_INFO"),
            1000513004i32 => f.write_str("PHYSICAL_DEVICE_VIDEO_ENCODE_AV1_FEATURES"),
            1000513005i32 => f.write_str("VIDEO_ENCODE_AV1_PROFILE_INFO"),
            1000513006i32 => f.write_str("VIDEO_ENCODE_AV1_RATE_CONTROL_INFO"),
            1000513007i32 => f.write_str("VIDEO_ENCODE_AV1_RATE_CONTROL_LAYER_INFO"),
            1000513008i32 => f.write_str("VIDEO_ENCODE_AV1_QUALITY_LEVEL_PROPERTIES"),
            1000513009i32 => f.write_str("VIDEO_ENCODE_AV1_SESSION_CREATE_INFO"),
            1000513010i32 => f.write_str("VIDEO_ENCODE_AV1_GOP_REMAINING_FRAME_INFO"),
            1000514000i32 => f.write_str("PHYSICAL_DEVICE_VIDEO_DECODE_VP9_FEATURES"),
            1000514001i32 => f.write_str("VIDEO_DECODE_VP9_CAPABILITIES"),
            1000514002i32 => f.write_str("VIDEO_DECODE_VP9_PICTURE_INFO"),
            1000514003i32 => f.write_str("VIDEO_DECODE_VP9_PROFILE_INFO"),
            1000515000i32 => f.write_str("PHYSICAL_DEVICE_VIDEO_MAINTENANCE_1_FEATURES"),
            1000515001i32 => f.write_str("VIDEO_INLINE_QUERY_INFO"),
            1000516000i32 => {
                f.write_str("PHYSICAL_DEVICE_PER_STAGE_DESCRIPTOR_SET_FEATURES")
            }
            1000518000i32 => f.write_str("PHYSICAL_DEVICE_IMAGE_PROCESSING_2_FEATURES"),
            1000518001i32 => f.write_str("PHYSICAL_DEVICE_IMAGE_PROCESSING_2_PROPERTIES"),
            1000518002i32 => f.write_str("SAMPLER_BLOCK_MATCH_WINDOW_CREATE_INFO"),
            1000519000i32 => f.write_str("SAMPLER_CUBIC_WEIGHTS_CREATE_INFO"),
            1000519001i32 => f.write_str("PHYSICAL_DEVICE_CUBIC_WEIGHTS_FEATURES"),
            1000519002i32 => f.write_str("BLIT_IMAGE_CUBIC_WEIGHTS_INFO"),
            1000520000i32 => f.write_str("PHYSICAL_DEVICE_YCBCR_DEGAMMA_FEATURES"),
            1000520001i32 => {
                f.write_str("SAMPLER_YCBCR_CONVERSION_YCBCR_DEGAMMA_CREATE_INFO")
            }
            1000521000i32 => f.write_str("PHYSICAL_DEVICE_CUBIC_CLAMP_FEATURES"),
            1000524000i32 => {
                f.write_str(
                    "PHYSICAL_DEVICE_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_FEATURES",
                )
            }
            1000527000i32 => {
                f.write_str("PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES")
            }
            1000527001i32 => f.write_str("ATTACHMENT_FEEDBACK_LOOP_INFO"),
            1000529000i32 => f.write_str("SCREEN_BUFFER_PROPERTIES"),
            1000529001i32 => f.write_str("SCREEN_BUFFER_FORMAT_PROPERTIES"),
            1000529002i32 => f.write_str("IMPORT_SCREEN_BUFFER_INFO"),
            1000529004i32 => {
                f.write_str("PHYSICAL_DEVICE_EXTERNAL_MEMORY_SCREEN_BUFFER_FEATURES")
            }
            1000530000i32 => f.write_str("PHYSICAL_DEVICE_LAYERED_DRIVER_PROPERTIES"),
            1000545007i32 => f.write_str("SET_DESCRIPTOR_BUFFER_OFFSETS_INFO"),
            1000545008i32 => f.write_str("BIND_DESCRIPTOR_BUFFER_EMBEDDED_SAMPLERS_INFO"),
            1000546000i32 => {
                f.write_str("PHYSICAL_DEVICE_DESCRIPTOR_POOL_OVERALLOCATION_FEATURES")
            }
            1000547000i32 => f.write_str("PHYSICAL_DEVICE_TILE_MEMORY_HEAP_FEATURES"),
            1000547001i32 => f.write_str("PHYSICAL_DEVICE_TILE_MEMORY_HEAP_PROPERTIES"),
            1000547002i32 => f.write_str("TILE_MEMORY_REQUIREMENTS"),
            1000547003i32 => f.write_str("TILE_MEMORY_BIND_INFO"),
            1000547004i32 => f.write_str("TILE_MEMORY_SIZE_INFO"),
            1000549002i32 => f.write_str("COPY_MEMORY_INDIRECT_INFO"),
            1000549003i32 => f.write_str("COPY_MEMORY_TO_IMAGE_INDIRECT_INFO"),
            1000550002i32 => f.write_str("DECOMPRESS_MEMORY_INFO"),
            1000551000i32 => f.write_str("DISPLAY_SURFACE_STEREO_CREATE_INFO"),
            1000551001i32 => f.write_str("DISPLAY_MODE_STEREO_PROPERTIES"),
            1000552000i32 => f.write_str("VIDEO_ENCODE_INTRA_REFRESH_CAPABILITIES"),
            1000552001i32 => {
                f.write_str("VIDEO_ENCODE_SESSION_INTRA_REFRESH_CREATE_INFO")
            }
            1000552002i32 => f.write_str("VIDEO_ENCODE_INTRA_REFRESH_INFO"),
            1000552003i32 => f.write_str("VIDEO_REFERENCE_INTRA_REFRESH_INFO"),
            1000552004i32 => {
                f.write_str("PHYSICAL_DEVICE_VIDEO_ENCODE_INTRA_REFRESH_FEATURES")
            }
            1000553000i32 => f.write_str("VIDEO_ENCODE_QUANTIZATION_MAP_CAPABILITIES"),
            1000553001i32 => f.write_str("VIDEO_FORMAT_QUANTIZATION_MAP_PROPERTIES"),
            1000553002i32 => f.write_str("VIDEO_ENCODE_QUANTIZATION_MAP_INFO"),
            1000553005i32 => {
                f.write_str(
                    "VIDEO_ENCODE_QUANTIZATION_MAP_SESSION_PARAMETERS_CREATE_INFO",
                )
            }
            1000553009i32 => {
                f.write_str("PHYSICAL_DEVICE_VIDEO_ENCODE_QUANTIZATION_MAP_FEATURES")
            }
            1000553003i32 => {
                f.write_str("VIDEO_ENCODE_H264_QUANTIZATION_MAP_CAPABILITIES")
            }
            1000553004i32 => {
                f.write_str("VIDEO_ENCODE_H265_QUANTIZATION_MAP_CAPABILITIES")
            }
            1000553006i32 => f.write_str("VIDEO_FORMAT_H265_QUANTIZATION_MAP_PROPERTIES"),
            1000553007i32 => {
                f.write_str("VIDEO_ENCODE_AV1_QUANTIZATION_MAP_CAPABILITIES")
            }
            1000553008i32 => f.write_str("VIDEO_FORMAT_AV1_QUANTIZATION_MAP_PROPERTIES"),
            1000555000i32 => f.write_str("PHYSICAL_DEVICE_RAW_ACCESS_CHAINS_FEATURES"),
            1000556000i32 => f.write_str("EXTERNAL_COMPUTE_QUEUE_DEVICE_CREATE_INFO"),
            1000556001i32 => f.write_str("EXTERNAL_COMPUTE_QUEUE_CREATE_INFO"),
            1000556002i32 => f.write_str("EXTERNAL_COMPUTE_QUEUE_DATA_PARAMS"),
            1000556003i32 => {
                f.write_str("PHYSICAL_DEVICE_EXTERNAL_COMPUTE_QUEUE_PROPERTIES")
            }
            1000558000i32 => {
                f.write_str(
                    "PHYSICAL_DEVICE_SHADER_RELAXED_EXTENDED_INSTRUCTION_FEATURES",
                )
            }
            1000559000i32 => {
                f.write_str("PHYSICAL_DEVICE_COMMAND_BUFFER_INHERITANCE_FEATURES")
            }
            1000562000i32 => f.write_str("PHYSICAL_DEVICE_MAINTENANCE_7_FEATURES"),
            1000562001i32 => f.write_str("PHYSICAL_DEVICE_MAINTENANCE_7_PROPERTIES"),
            1000562002i32 => f.write_str("PHYSICAL_DEVICE_LAYERED_API_PROPERTIES_LIST"),
            1000562003i32 => f.write_str("PHYSICAL_DEVICE_LAYERED_API_PROPERTIES"),
            1000562004i32 => f.write_str("PHYSICAL_DEVICE_LAYERED_API_VULKAN_PROPERTIES"),
            1000563000i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT16_VECTOR_FEATURES")
            }
            1000564000i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_REPLICATED_COMPOSITES_FEATURES")
            }
            1000567000i32 => f.write_str("PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES"),
            1000568000i32 => {
                f.write_str("PHYSICAL_DEVICE_RAY_TRACING_VALIDATION_FEATURES")
            }
            1000569000i32 => {
                f.write_str("PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES")
            }
            1000569001i32 => {
                f.write_str("PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES")
            }
            1000569002i32 => {
                f.write_str("CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT")
            }
            1000569003i32 => {
                f.write_str("CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT")
            }
            1000569004i32 => {
                f.write_str("CLUSTER_ACCELERATION_STRUCTURE_MOVE_OBJECTS_INPUT")
            }
            1000569005i32 => f.write_str("CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO"),
            1000569006i32 => f.write_str("CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO"),
            1000569007i32 => {
                f.write_str(
                    "RAY_TRACING_PIPELINE_CLUSTER_ACCELERATION_STRUCTURE_CREATE_INFO",
                )
            }
            1000570000i32 => {
                f.write_str(
                    "PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_FEATURES",
                )
            }
            1000570001i32 => {
                f.write_str(
                    "PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_PROPERTIES",
                )
            }
            1000570002i32 => {
                f.write_str("WRITE_DESCRIPTOR_SET_PARTITIONED_ACCELERATION_STRUCTURE")
            }
            1000570003i32 => {
                f.write_str("PARTITIONED_ACCELERATION_STRUCTURE_INSTANCES_INPUT")
            }
            1000570004i32 => f.write_str("BUILD_PARTITIONED_ACCELERATION_STRUCTURE_INFO"),
            1000570005i32 => f.write_str("PARTITIONED_ACCELERATION_STRUCTURE_FLAGS"),
            1000572003i32 => f.write_str("INDIRECT_EXECUTION_SET_CREATE_INFO"),
            1000572008i32 => f.write_str("WRITE_INDIRECT_EXECUTION_SET_PIPELINE"),
            1000572009i32 => f.write_str("WRITE_INDIRECT_EXECUTION_SET_SHADER"),
            1000572010i32 => f.write_str("INDIRECT_EXECUTION_SET_PIPELINE_INFO"),
            1000572011i32 => f.write_str("INDIRECT_EXECUTION_SET_SHADER_INFO"),
            1000572012i32 => f.write_str("INDIRECT_EXECUTION_SET_SHADER_LAYOUT_INFO"),
            1000572013i32 => f.write_str("GENERATED_COMMANDS_PIPELINE_INFO"),
            1000572014i32 => f.write_str("GENERATED_COMMANDS_SHADER_INFO"),
            1000573001i32 => f.write_str("PHYSICAL_DEVICE_FAULT_PROPERTIES"),
            1000573003i32 => f.write_str("DEVICE_FAULT_DEBUG_INFO"),
            1000574000i32 => f.write_str("PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES"),
            1000574002i32 => f.write_str("MEMORY_BARRIER_ACCESS_FLAGS_3"),
            1000575000i32 => {
                f.write_str("PHYSICAL_DEVICE_IMAGE_ALIGNMENT_CONTROL_FEATURES")
            }
            1000575001i32 => {
                f.write_str("PHYSICAL_DEVICE_IMAGE_ALIGNMENT_CONTROL_PROPERTIES")
            }
            1000575002i32 => f.write_str("IMAGE_ALIGNMENT_CONTROL_CREATE_INFO"),
            1000579000i32 => f.write_str("PHYSICAL_DEVICE_SHADER_FMA_FEATURES"),
            1000580000i32 => f.write_str("PUSH_CONSTANT_BANK_INFO"),
            1000580001i32 => f.write_str("PHYSICAL_DEVICE_PUSH_CONSTANT_BANK_FEATURES"),
            1000580002i32 => f.write_str("PHYSICAL_DEVICE_PUSH_CONSTANT_BANK_PROPERTIES"),
            1000582000i32 => f.write_str("PHYSICAL_DEVICE_DEPTH_CLAMP_CONTROL_FEATURES"),
            1000582001i32 => {
                f.write_str("PIPELINE_VIEWPORT_DEPTH_CLAMP_CONTROL_CREATE_INFO")
            }
            1000584000i32 => f.write_str("PHYSICAL_DEVICE_MAINTENANCE_9_FEATURES"),
            1000584001i32 => f.write_str("PHYSICAL_DEVICE_MAINTENANCE_9_PROPERTIES"),
            1000584002i32 => f.write_str("QUEUE_FAMILY_OWNERSHIP_TRANSFER_PROPERTIES"),
            1000586000i32 => f.write_str("PHYSICAL_DEVICE_VIDEO_MAINTENANCE_2_FEATURES"),
            1000586001i32 => {
                f.write_str("VIDEO_DECODE_H264_INLINE_SESSION_PARAMETERS_INFO")
            }
            1000586002i32 => {
                f.write_str("VIDEO_DECODE_H265_INLINE_SESSION_PARAMETERS_INFO")
            }
            1000586003i32 => {
                f.write_str("VIDEO_DECODE_AV1_INLINE_SESSION_PARAMETERS_INFO")
            }
            1000685000i32 => f.write_str("SURFACE_CREATE_INFO"),
            1000590000i32 => f.write_str("PHYSICAL_DEVICE_HDR_VIVID_FEATURES"),
            1000590001i32 => f.write_str("HDR_VIVID_DYNAMIC_METADATA"),
            1000593000i32 => f.write_str("PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES"),
            1000593001i32 => {
                f.write_str("COOPERATIVE_MATRIX_FLEXIBLE_DIMENSIONS_PROPERTIES")
            }
            1000593002i32 => {
                f.write_str("PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_PROPERTIES")
            }
            1000596000i32 => {
                f.write_str("PHYSICAL_DEVICE_PIPELINE_OPACITY_MICROMAP_FEATURES")
            }
            1000602000i32 => f.write_str("IMPORT_MEMORY_METAL_HANDLE_INFO"),
            1000602001i32 => f.write_str("MEMORY_METAL_HANDLE_PROPERTIES"),
            1000602002i32 => f.write_str("MEMORY_GET_METAL_HANDLE_INFO"),
            1000605000i32 => {
                f.write_str("PHYSICAL_DEVICE_PERFORMANCE_COUNTERS_BY_REGION_FEATURES")
            }
            1000605001i32 => {
                f.write_str("PHYSICAL_DEVICE_PERFORMANCE_COUNTERS_BY_REGION_PROPERTIES")
            }
            1000605004i32 => {
                f.write_str("RENDER_PASS_PERFORMANCE_COUNTERS_BY_REGION_BEGIN_INFO")
            }
            1000607000i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_INSTRUMENTATION_FEATURES")
            }
            1000607001i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_INSTRUMENTATION_PROPERTIES")
            }
            1000607002i32 => f.write_str("SHADER_INSTRUMENTATION_CREATE_INFO"),
            1000607003i32 => f.write_str("SHADER_INSTRUMENTATION_METRIC_DESCRIPTION"),
            1000608000i32 => {
                f.write_str("PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_ROBUSTNESS_FEATURES")
            }
            1000609000i32 => f.write_str("PHYSICAL_DEVICE_FORMAT_PACK_FEATURES"),
            1000611000i32 => {
                f.write_str("PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_LAYERED_FEATURES")
            }
            1000611001i32 => {
                f.write_str("PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_LAYERED_PROPERTIES")
            }
            1000611002i32 => {
                f.write_str("PIPELINE_FRAGMENT_DENSITY_MAP_LAYERED_CREATE_INFO")
            }
            1000613000i32 => f.write_str("SET_PRESENT_CONFIG"),
            1000613001i32 => f.write_str("PHYSICAL_DEVICE_PRESENT_METERING_FEATURES"),
            1000425002i32 => {
                f.write_str("RENDER_PASS_FRAGMENT_DENSITY_MAP_OFFSET_END_INFO")
            }
            1000620000i32 => {
                f.write_str("PHYSICAL_DEVICE_ZERO_INITIALIZE_DEVICE_MEMORY_FEATURES")
            }
            1000627000i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_64_BIT_INDEXING_FEATURES")
            }
            1000628000i32 => f.write_str("PHYSICAL_DEVICE_CUSTOM_RESOLVE_FEATURES"),
            1000628001i32 => f.write_str("BEGIN_CUSTOM_RESOLVE_INFO"),
            1000628002i32 => f.write_str("CUSTOM_RESOLVE_CREATE_INFO"),
            1000629000i32 => f.write_str("PHYSICAL_DEVICE_DATA_GRAPH_MODEL_FEATURES"),
            1000629001i32 => f.write_str("DATA_GRAPH_PIPELINE_BUILTIN_MODEL_CREATE_INFO"),
            1000630000i32 => f.write_str("PHYSICAL_DEVICE_MAINTENANCE_10_FEATURES"),
            1000630001i32 => f.write_str("PHYSICAL_DEVICE_MAINTENANCE_10_PROPERTIES"),
            1000630002i32 => f.write_str("RENDERING_ATTACHMENT_FLAGS_INFO"),
            1000630004i32 => f.write_str("RESOLVE_IMAGE_MODE_INFO"),
            1000635000i32 => f.write_str("PHYSICAL_DEVICE_SHADER_LONG_VECTOR_FEATURES"),
            1000635001i32 => f.write_str("PHYSICAL_DEVICE_SHADER_LONG_VECTOR_PROPERTIES"),
            1000637000i32 => {
                f.write_str("PHYSICAL_DEVICE_PIPELINE_CACHE_INCREMENTAL_MODE_FEATURES")
            }
            1000642000i32 => {
                f.write_str(
                    "PHYSICAL_DEVICE_SHADER_UNIFORM_BUFFER_UNSIZED_ARRAY_FEATURES",
                )
            }
            1000645000i32 => f.write_str("COMPUTE_OCCUPANCY_PRIORITY_PARAMETERS"),
            1000645001i32 => {
                f.write_str("PHYSICAL_DEVICE_COMPUTE_OCCUPANCY_PRIORITY_FEATURES")
            }
            1000662000i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_SUBGROUP_PARTITIONED_FEATURES")
            }
            1000664000i32 => f.write_str("UBM_SURFACE_CREATE_INFO"),
            1000673000i32 => {
                f.write_str("PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_PRODUCT_FEATURES")
            }
            other => write!(f, "{}({})", stringify!(StructureType), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDebugReportObjectTypeEXT")]
pub struct DebugReportObjectTypeEXT(i32);
impl DebugReportObjectTypeEXT {
    pub const UNKNOWN: Self = Self(0i32);
    pub const INSTANCE: Self = Self(1i32);
    pub const PHYSICAL_DEVICE: Self = Self(2i32);
    pub const DEVICE: Self = Self(3i32);
    pub const QUEUE: Self = Self(4i32);
    pub const SEMAPHORE: Self = Self(5i32);
    pub const COMMAND_BUFFER: Self = Self(6i32);
    pub const FENCE: Self = Self(7i32);
    pub const DEVICE_MEMORY: Self = Self(8i32);
    pub const BUFFER: Self = Self(9i32);
    pub const IMAGE: Self = Self(10i32);
    pub const EVENT: Self = Self(11i32);
    pub const QUERY_POOL: Self = Self(12i32);
    pub const BUFFER_VIEW: Self = Self(13i32);
    pub const IMAGE_VIEW: Self = Self(14i32);
    pub const SHADER_MODULE: Self = Self(15i32);
    pub const PIPELINE_CACHE: Self = Self(16i32);
    pub const PIPELINE_LAYOUT: Self = Self(17i32);
    pub const RENDER_PASS: Self = Self(18i32);
    pub const PIPELINE: Self = Self(19i32);
    pub const DESCRIPTOR_SET_LAYOUT: Self = Self(20i32);
    pub const SAMPLER: Self = Self(21i32);
    pub const DESCRIPTOR_POOL: Self = Self(22i32);
    pub const DESCRIPTOR_SET: Self = Self(23i32);
    pub const FRAMEBUFFER: Self = Self(24i32);
    pub const COMMAND_POOL: Self = Self(25i32);
    pub const SURFACE_: Self = Self(26i32);
    pub const SWAPCHAIN_: Self = Self(27i32);
    pub const DEBUG_REPORT_CALLBACK_: Self = Self(28i32);
    pub const DEBUG_REPORT: Self = Self::DEBUG_REPORT_CALLBACK_;
    pub const DISPLAY_: Self = Self(29i32);
    pub const DISPLAY_MODE_: Self = Self(30i32);
    pub const VALIDATION_CACHE_: Self = Self(33i32);
    pub const VALIDATION_CACHE: Self = Self::VALIDATION_CACHE_;
    pub const SAMPLER_YCBCR_CONVERSION: Self = Self(1000156000i32);
    pub const DESCRIPTOR_UPDATE_TEMPLATE: Self = Self(1000085000i32);
    pub const CU_MODULE_: Self = Self(1000029000i32);
    pub const CU_FUNCTION_: Self = Self(1000029001i32);
    pub const DESCRIPTOR_UPDATE_TEMPLATE_: Self = Self::DESCRIPTOR_UPDATE_TEMPLATE;
    pub const ACCELERATION_STRUCTURE_: Self = Self(1000150000i32);
    pub const SAMPLER_YCBCR_CONVERSION_: Self = Self::SAMPLER_YCBCR_CONVERSION;
    pub const CUDA_MODULE_: Self = Self(1000307000i32);
    pub const CUDA_FUNCTION_: Self = Self(1000307001i32);
    pub const BUFFER_COLLECTION_: Self = Self(1000366000i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DebugReportObjectTypeEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("UNKNOWN"),
            1i32 => f.write_str("INSTANCE"),
            2i32 => f.write_str("PHYSICAL_DEVICE"),
            3i32 => f.write_str("DEVICE"),
            4i32 => f.write_str("QUEUE"),
            5i32 => f.write_str("SEMAPHORE"),
            6i32 => f.write_str("COMMAND_BUFFER"),
            7i32 => f.write_str("FENCE"),
            8i32 => f.write_str("DEVICE_MEMORY"),
            9i32 => f.write_str("BUFFER"),
            10i32 => f.write_str("IMAGE"),
            11i32 => f.write_str("EVENT"),
            12i32 => f.write_str("QUERY_POOL"),
            13i32 => f.write_str("BUFFER_VIEW"),
            14i32 => f.write_str("IMAGE_VIEW"),
            15i32 => f.write_str("SHADER_MODULE"),
            16i32 => f.write_str("PIPELINE_CACHE"),
            17i32 => f.write_str("PIPELINE_LAYOUT"),
            18i32 => f.write_str("RENDER_PASS"),
            19i32 => f.write_str("PIPELINE"),
            20i32 => f.write_str("DESCRIPTOR_SET_LAYOUT"),
            21i32 => f.write_str("SAMPLER"),
            22i32 => f.write_str("DESCRIPTOR_POOL"),
            23i32 => f.write_str("DESCRIPTOR_SET"),
            24i32 => f.write_str("FRAMEBUFFER"),
            25i32 => f.write_str("COMMAND_POOL"),
            26i32 => f.write_str("SURFACE_"),
            27i32 => f.write_str("SWAPCHAIN_"),
            28i32 => f.write_str("DEBUG_REPORT_CALLBACK_"),
            29i32 => f.write_str("DISPLAY_"),
            30i32 => f.write_str("DISPLAY_MODE_"),
            33i32 => f.write_str("VALIDATION_CACHE_"),
            1000156000i32 => f.write_str("SAMPLER_YCBCR_CONVERSION"),
            1000085000i32 => f.write_str("DESCRIPTOR_UPDATE_TEMPLATE"),
            1000029000i32 => f.write_str("CU_MODULE_"),
            1000029001i32 => f.write_str("CU_FUNCTION_"),
            1000150000i32 => f.write_str("ACCELERATION_STRUCTURE_"),
            1000307000i32 => f.write_str("CUDA_MODULE_"),
            1000307001i32 => f.write_str("CUDA_FUNCTION_"),
            1000366000i32 => f.write_str("BUFFER_COLLECTION_"),
            other => write!(f, "{}({})", stringify!(DebugReportObjectTypeEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkBorderColor")]
pub struct BorderColor(i32);
impl BorderColor {
    pub const FLOAT_TRANSPARENT_BLACK: Self = Self(0i32);
    pub const INT_TRANSPARENT_BLACK: Self = Self(1i32);
    pub const FLOAT_OPAQUE_BLACK: Self = Self(2i32);
    pub const INT_OPAQUE_BLACK: Self = Self(3i32);
    pub const FLOAT_OPAQUE_WHITE: Self = Self(4i32);
    pub const INT_OPAQUE_WHITE: Self = Self(5i32);
    pub const FLOAT_CUSTOM: Self = Self(1000287003i32);
    pub const INT_CUSTOM: Self = Self(1000287004i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for BorderColor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("FLOAT_TRANSPARENT_BLACK"),
            1i32 => f.write_str("INT_TRANSPARENT_BLACK"),
            2i32 => f.write_str("FLOAT_OPAQUE_BLACK"),
            3i32 => f.write_str("INT_OPAQUE_BLACK"),
            4i32 => f.write_str("FLOAT_OPAQUE_WHITE"),
            5i32 => f.write_str("INT_OPAQUE_WHITE"),
            1000287003i32 => f.write_str("FLOAT_CUSTOM"),
            1000287004i32 => f.write_str("INT_CUSTOM"),
            other => write!(f, "{}({})", stringify!(BorderColor), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkSamplerYcbcrModelConversion")]
pub struct SamplerYcbcrModelConversion(i32);
impl SamplerYcbcrModelConversion {
    pub const RGB_IDENTITY: Self = Self(0i32);
    pub const YCBCR_IDENTITY: Self = Self(1i32);
    pub const YCBCR_709: Self = Self(2i32);
    pub const YCBCR_601: Self = Self(3i32);
    pub const YCBCR_2020: Self = Self(4i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for SamplerYcbcrModelConversion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("RGB_IDENTITY"),
            1i32 => f.write_str("YCBCR_IDENTITY"),
            2i32 => f.write_str("YCBCR_709"),
            3i32 => f.write_str("YCBCR_601"),
            4i32 => f.write_str("YCBCR_2020"),
            other => write!(f, "{}({})", stringify!(SamplerYcbcrModelConversion), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkCooperativeVectorMatrixLayoutNV")]
pub struct CooperativeVectorMatrixLayoutNV(i32);
impl CooperativeVectorMatrixLayoutNV {
    pub const ROW_MAJOR: Self = Self(0i32);
    pub const COLUMN_MAJOR: Self = Self(1i32);
    pub const INFERENCING_OPTIMAL: Self = Self(2i32);
    pub const TRAINING_OPTIMAL: Self = Self(3i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for CooperativeVectorMatrixLayoutNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("ROW_MAJOR"),
            1i32 => f.write_str("COLUMN_MAJOR"),
            2i32 => f.write_str("INFERENCING_OPTIMAL"),
            3i32 => f.write_str("TRAINING_OPTIMAL"),
            other => {
                write!(f, "{}({})", stringify!(CooperativeVectorMatrixLayoutNV), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkOpacityMicromapSpecialIndexEXT")]
pub struct OpacityMicromapSpecialIndexEXT(i32);
impl OpacityMicromapSpecialIndexEXT {
    pub const FULLY_TRANSPARENT: Self = Self(-1i32);
    pub const FULLY_OPAQUE: Self = Self(-2i32);
    pub const FULLY_UNKNOWN_TRANSPARENT: Self = Self(-3i32);
    pub const FULLY_UNKNOWN_OPAQUE: Self = Self(-4i32);
    pub const CLUSTER_GEOMETRY_DISABLE_OPACITY_MICROMAP: Self = Self(-5i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for OpacityMicromapSpecialIndexEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            -1i32 => f.write_str("FULLY_TRANSPARENT"),
            -2i32 => f.write_str("FULLY_OPAQUE"),
            -3i32 => f.write_str("FULLY_UNKNOWN_TRANSPARENT"),
            -4i32 => f.write_str("FULLY_UNKNOWN_OPAQUE"),
            -5i32 => f.write_str("CLUSTER_GEOMETRY_DISABLE_OPACITY_MICROMAP"),
            other => {
                write!(f, "{}({})", stringify!(OpacityMicromapSpecialIndexEXT), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDeviceFaultVendorBinaryHeaderVersionKHR")]
pub struct DeviceFaultVendorBinaryHeaderVersionKHR(i32);
impl DeviceFaultVendorBinaryHeaderVersionKHR {
    pub const ONE: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DeviceFaultVendorBinaryHeaderVersionKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            1i32 => f.write_str("ONE"),
            other => {
                write!(
                    f, "{}({})", stringify!(DeviceFaultVendorBinaryHeaderVersionKHR),
                    other
                )
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDirectDriverLoadingModeLUNARG")]
pub struct DirectDriverLoadingModeLUNARG(i32);
impl DirectDriverLoadingModeLUNARG {
    pub const EXCLUSIVE: Self = Self(0i32);
    pub const INCLUSIVE: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DirectDriverLoadingModeLUNARG {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("EXCLUSIVE"),
            1i32 => f.write_str("INCLUSIVE"),
            other => {
                write!(f, "{}({})", stringify!(DirectDriverLoadingModeLUNARG), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkTimeDomainKHR")]
pub struct TimeDomainKHR(i32);
impl TimeDomainKHR {
    pub const DEVICE: Self = Self(0i32);
    pub const CLOCK_MONOTONIC: Self = Self(1i32);
    pub const CLOCK_MONOTONIC_RAW: Self = Self(2i32);
    pub const QUERY_PERFORMANCE_COUNTER: Self = Self(3i32);
    pub const PRESENT_STAGE_LOCAL: Self = Self(1000208000i32);
    pub const SWAPCHAIN_LOCAL: Self = Self(1000208001i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for TimeDomainKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("DEVICE"),
            1i32 => f.write_str("CLOCK_MONOTONIC"),
            2i32 => f.write_str("CLOCK_MONOTONIC_RAW"),
            3i32 => f.write_str("QUERY_PERFORMANCE_COUNTER"),
            1000208000i32 => f.write_str("PRESENT_STAGE_LOCAL"),
            1000208001i32 => f.write_str("SWAPCHAIN_LOCAL"),
            other => write!(f, "{}({})", stringify!(TimeDomainKHR), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkBuildAccelerationStructureModeKHR")]
pub struct BuildAccelerationStructureModeKHR(i32);
impl BuildAccelerationStructureModeKHR {
    pub const BUILD: Self = Self(0i32);
    pub const UPDATE: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for BuildAccelerationStructureModeKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("BUILD"),
            1i32 => f.write_str("UPDATE"),
            other => {
                write!(f, "{}({})", stringify!(BuildAccelerationStructureModeKHR), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkOutOfBandQueueTypeNV")]
pub struct OutOfBandQueueTypeNV(i32);
impl OutOfBandQueueTypeNV {
    pub const RENDER: Self = Self(0i32);
    pub const PRESENT: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for OutOfBandQueueTypeNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("RENDER"),
            1i32 => f.write_str("PRESENT"),
            other => write!(f, "{}({})", stringify!(OutOfBandQueueTypeNV), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkCubicFilterWeightsQCOM")]
pub struct CubicFilterWeightsQCOM(i32);
impl CubicFilterWeightsQCOM {
    pub const CATMULL_ROM: Self = Self(0i32);
    pub const ZERO_TANGENT_CARDINAL: Self = Self(1i32);
    pub const B_SPLINE: Self = Self(2i32);
    pub const MITCHELL_NETRAVALI: Self = Self(3i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for CubicFilterWeightsQCOM {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("CATMULL_ROM"),
            1i32 => f.write_str("ZERO_TANGENT_CARDINAL"),
            2i32 => f.write_str("B_SPLINE"),
            3i32 => f.write_str("MITCHELL_NETRAVALI"),
            other => write!(f, "{}({})", stringify!(CubicFilterWeightsQCOM), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkAccelerationStructureBuildTypeKHR")]
pub struct AccelerationStructureBuildTypeKHR(i32);
impl AccelerationStructureBuildTypeKHR {
    pub const HOST: Self = Self(0i32);
    pub const DEVICE: Self = Self(1i32);
    pub const HOST_OR_DEVICE: Self = Self(2i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for AccelerationStructureBuildTypeKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("HOST"),
            1i32 => f.write_str("DEVICE"),
            2i32 => f.write_str("HOST_OR_DEVICE"),
            other => {
                write!(f, "{}({})", stringify!(AccelerationStructureBuildTypeKHR), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkObjectType")]
pub struct ObjectType(i32);
impl ObjectType {
    pub const UNKNOWN: Self = Self(0i32);
    pub const INSTANCE: Self = Self(1i32);
    pub const PHYSICAL_DEVICE: Self = Self(2i32);
    pub const DEVICE: Self = Self(3i32);
    pub const QUEUE: Self = Self(4i32);
    pub const SEMAPHORE: Self = Self(5i32);
    pub const COMMAND_BUFFER: Self = Self(6i32);
    pub const FENCE: Self = Self(7i32);
    pub const DEVICE_MEMORY: Self = Self(8i32);
    pub const BUFFER: Self = Self(9i32);
    pub const IMAGE: Self = Self(10i32);
    pub const EVENT: Self = Self(11i32);
    pub const QUERY_POOL: Self = Self(12i32);
    pub const BUFFER_VIEW: Self = Self(13i32);
    pub const IMAGE_VIEW: Self = Self(14i32);
    pub const SHADER_MODULE: Self = Self(15i32);
    pub const PIPELINE_CACHE: Self = Self(16i32);
    pub const PIPELINE_LAYOUT: Self = Self(17i32);
    pub const RENDER_PASS: Self = Self(18i32);
    pub const PIPELINE: Self = Self(19i32);
    pub const DESCRIPTOR_SET_LAYOUT: Self = Self(20i32);
    pub const SAMPLER: Self = Self(21i32);
    pub const DESCRIPTOR_POOL: Self = Self(22i32);
    pub const DESCRIPTOR_SET: Self = Self(23i32);
    pub const FRAMEBUFFER: Self = Self(24i32);
    pub const COMMAND_POOL: Self = Self(25i32);
    pub const DESCRIPTOR_UPDATE_TEMPLATE: Self = Self(1000085000i32);
    pub const SAMPLER_YCBCR_CONVERSION: Self = Self(1000156000i32);
    pub const PRIVATE_DATA_SLOT: Self = Self(1000295000i32);
    pub const SURFACE: Self = Self(1000000000i32);
    pub const SWAPCHAIN: Self = Self(1000001000i32);
    pub const DISPLAY: Self = Self(1000002000i32);
    pub const DISPLAY_MODE: Self = Self(1000002001i32);
    pub const DEBUG_REPORT_CALLBACK: Self = Self(1000011000i32);
    pub const VIDEO_SESSION: Self = Self(1000023000i32);
    pub const VIDEO_SESSION_PARAMETERS: Self = Self(1000023001i32);
    pub const CU_MODULE: Self = Self(1000029000i32);
    pub const CU_FUNCTION: Self = Self(1000029001i32);
    pub const DEBUG_UTILS_MESSENGER: Self = Self(1000128000i32);
    pub const ACCELERATION_STRUCTURE: Self = Self(1000150000i32);
    pub const VALIDATION_CACHE: Self = Self(1000160000i32);
    pub const PERFORMANCE_CONFIGURATION: Self = Self(1000210000i32);
    pub const DEFERRED_OPERATION: Self = Self(1000268000i32);
    pub const INDIRECT_COMMANDS_LAYOUT: Self = Self(1000277000i32);
    pub const CUDA_MODULE: Self = Self(1000307000i32);
    pub const CUDA_FUNCTION: Self = Self(1000307001i32);
    pub const BUFFER_COLLECTION: Self = Self(1000366000i32);
    pub const MICROMAP: Self = Self(1000396000i32);
    pub const TENSOR: Self = Self(1000460000i32);
    pub const TENSOR_VIEW: Self = Self(1000460001i32);
    pub const OPTICAL_FLOW_SESSION: Self = Self(1000464000i32);
    pub const SHADER: Self = Self(1000482000i32);
    pub const PIPELINE_BINARY: Self = Self(1000483000i32);
    pub const SEMAPHORE_SCI_SYNC_POOL: Self = Self(1000489000i32);
    pub const DATA_GRAPH_PIPELINE_SESSION: Self = Self(1000507000i32);
    pub const EXTERNAL_COMPUTE_QUEUE: Self = Self(1000556000i32);
    pub const INDIRECT_EXECUTION_SET: Self = Self(1000572001i32);
    pub const SHADER_INSTRUMENTATION: Self = Self(1000607000i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ObjectType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("UNKNOWN"),
            1i32 => f.write_str("INSTANCE"),
            2i32 => f.write_str("PHYSICAL_DEVICE"),
            3i32 => f.write_str("DEVICE"),
            4i32 => f.write_str("QUEUE"),
            5i32 => f.write_str("SEMAPHORE"),
            6i32 => f.write_str("COMMAND_BUFFER"),
            7i32 => f.write_str("FENCE"),
            8i32 => f.write_str("DEVICE_MEMORY"),
            9i32 => f.write_str("BUFFER"),
            10i32 => f.write_str("IMAGE"),
            11i32 => f.write_str("EVENT"),
            12i32 => f.write_str("QUERY_POOL"),
            13i32 => f.write_str("BUFFER_VIEW"),
            14i32 => f.write_str("IMAGE_VIEW"),
            15i32 => f.write_str("SHADER_MODULE"),
            16i32 => f.write_str("PIPELINE_CACHE"),
            17i32 => f.write_str("PIPELINE_LAYOUT"),
            18i32 => f.write_str("RENDER_PASS"),
            19i32 => f.write_str("PIPELINE"),
            20i32 => f.write_str("DESCRIPTOR_SET_LAYOUT"),
            21i32 => f.write_str("SAMPLER"),
            22i32 => f.write_str("DESCRIPTOR_POOL"),
            23i32 => f.write_str("DESCRIPTOR_SET"),
            24i32 => f.write_str("FRAMEBUFFER"),
            25i32 => f.write_str("COMMAND_POOL"),
            1000085000i32 => f.write_str("DESCRIPTOR_UPDATE_TEMPLATE"),
            1000156000i32 => f.write_str("SAMPLER_YCBCR_CONVERSION"),
            1000295000i32 => f.write_str("PRIVATE_DATA_SLOT"),
            1000000000i32 => f.write_str("SURFACE"),
            1000001000i32 => f.write_str("SWAPCHAIN"),
            1000002000i32 => f.write_str("DISPLAY"),
            1000002001i32 => f.write_str("DISPLAY_MODE"),
            1000011000i32 => f.write_str("DEBUG_REPORT_CALLBACK"),
            1000023000i32 => f.write_str("VIDEO_SESSION"),
            1000023001i32 => f.write_str("VIDEO_SESSION_PARAMETERS"),
            1000029000i32 => f.write_str("CU_MODULE"),
            1000029001i32 => f.write_str("CU_FUNCTION"),
            1000128000i32 => f.write_str("DEBUG_UTILS_MESSENGER"),
            1000150000i32 => f.write_str("ACCELERATION_STRUCTURE"),
            1000160000i32 => f.write_str("VALIDATION_CACHE"),
            1000210000i32 => f.write_str("PERFORMANCE_CONFIGURATION"),
            1000268000i32 => f.write_str("DEFERRED_OPERATION"),
            1000277000i32 => f.write_str("INDIRECT_COMMANDS_LAYOUT"),
            1000307000i32 => f.write_str("CUDA_MODULE"),
            1000307001i32 => f.write_str("CUDA_FUNCTION"),
            1000366000i32 => f.write_str("BUFFER_COLLECTION"),
            1000396000i32 => f.write_str("MICROMAP"),
            1000460000i32 => f.write_str("TENSOR"),
            1000460001i32 => f.write_str("TENSOR_VIEW"),
            1000464000i32 => f.write_str("OPTICAL_FLOW_SESSION"),
            1000482000i32 => f.write_str("SHADER"),
            1000483000i32 => f.write_str("PIPELINE_BINARY"),
            1000489000i32 => f.write_str("SEMAPHORE_SCI_SYNC_POOL"),
            1000507000i32 => f.write_str("DATA_GRAPH_PIPELINE_SESSION"),
            1000556000i32 => f.write_str("EXTERNAL_COMPUTE_QUEUE"),
            1000572001i32 => f.write_str("INDIRECT_EXECUTION_SET"),
            1000607000i32 => f.write_str("SHADER_INSTRUMENTATION"),
            other => write!(f, "{}({})", stringify!(ObjectType), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkValidationFeatureDisableEXT")]
pub struct ValidationFeatureDisableEXT(i32);
impl ValidationFeatureDisableEXT {
    pub const ALL: Self = Self(0i32);
    pub const SHADERS: Self = Self(1i32);
    pub const THREAD_SAFETY: Self = Self(2i32);
    pub const API_PARAMETERS: Self = Self(3i32);
    pub const OBJECT_LIFETIMES: Self = Self(4i32);
    pub const CORE_CHECKS: Self = Self(5i32);
    pub const UNIQUE_HANDLES: Self = Self(6i32);
    pub const SHADER_VALIDATION_CACHE: Self = Self(7i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ValidationFeatureDisableEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("ALL"),
            1i32 => f.write_str("SHADERS"),
            2i32 => f.write_str("THREAD_SAFETY"),
            3i32 => f.write_str("API_PARAMETERS"),
            4i32 => f.write_str("OBJECT_LIFETIMES"),
            5i32 => f.write_str("CORE_CHECKS"),
            6i32 => f.write_str("UNIQUE_HANDLES"),
            7i32 => f.write_str("SHADER_VALIDATION_CACHE"),
            other => write!(f, "{}({})", stringify!(ValidationFeatureDisableEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkAccelerationStructureTypeKHR")]
pub struct AccelerationStructureTypeKHR(i32);
impl AccelerationStructureTypeKHR {
    pub const TOP_LEVEL: Self = Self(0i32);
    pub const BOTTOM_LEVEL: Self = Self(1i32);
    pub const GENERIC: Self = Self(2i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for AccelerationStructureTypeKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("TOP_LEVEL"),
            1i32 => f.write_str("BOTTOM_LEVEL"),
            2i32 => f.write_str("GENERIC"),
            other => write!(f, "{}({})", stringify!(AccelerationStructureTypeKHR), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPresentModeKHR")]
pub struct PresentModeKHR(i32);
impl PresentModeKHR {
    pub const IMMEDIATE: Self = Self(0i32);
    pub const MAILBOX: Self = Self(1i32);
    pub const FIFO: Self = Self(2i32);
    pub const FIFO_RELAXED: Self = Self(3i32);
    pub const SHARED_DEMAND_REFRESH: Self = Self(1000111000i32);
    pub const SHARED_CONTINUOUS_REFRESH: Self = Self(1000111001i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PresentModeKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("IMMEDIATE"),
            1i32 => f.write_str("MAILBOX"),
            2i32 => f.write_str("FIFO"),
            3i32 => f.write_str("FIFO_RELAXED"),
            1000111000i32 => f.write_str("SHARED_DEMAND_REFRESH"),
            1000111001i32 => f.write_str("SHARED_CONTINUOUS_REFRESH"),
            other => write!(f, "{}({})", stringify!(PresentModeKHR), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkShaderGroupShaderKHR")]
pub struct ShaderGroupShaderKHR(i32);
impl ShaderGroupShaderKHR {
    pub const GENERAL: Self = Self(0i32);
    pub const CLOSEST_HIT: Self = Self(1i32);
    pub const ANY_HIT: Self = Self(2i32);
    pub const INTERSECTION: Self = Self(3i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ShaderGroupShaderKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("GENERAL"),
            1i32 => f.write_str("CLOSEST_HIT"),
            2i32 => f.write_str("ANY_HIT"),
            3i32 => f.write_str("INTERSECTION"),
            other => write!(f, "{}({})", stringify!(ShaderGroupShaderKHR), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPerformanceConfigurationTypeINTEL")]
pub struct PerformanceConfigurationTypeINTEL(i32);
impl PerformanceConfigurationTypeINTEL {
    pub const COMMAND_QUEUE_METRICS_DISCOVERY_ACTIVATED: Self = Self(0i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PerformanceConfigurationTypeINTEL {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("COMMAND_QUEUE_METRICS_DISCOVERY_ACTIVATED"),
            other => {
                write!(f, "{}({})", stringify!(PerformanceConfigurationTypeINTEL), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPhysicalDeviceType")]
pub struct PhysicalDeviceType(i32);
impl PhysicalDeviceType {
    pub const OTHER: Self = Self(0i32);
    pub const INTEGRATED_GPU: Self = Self(1i32);
    pub const DISCRETE_GPU: Self = Self(2i32);
    pub const VIRTUAL_GPU: Self = Self(3i32);
    pub const CPU: Self = Self(4i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PhysicalDeviceType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("OTHER"),
            1i32 => f.write_str("INTEGRATED_GPU"),
            2i32 => f.write_str("DISCRETE_GPU"),
            3i32 => f.write_str("VIRTUAL_GPU"),
            4i32 => f.write_str("CPU"),
            other => write!(f, "{}({})", stringify!(PhysicalDeviceType), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDeviceAddressBindingTypeEXT")]
pub struct DeviceAddressBindingTypeEXT(i32);
impl DeviceAddressBindingTypeEXT {
    pub const BIND: Self = Self(0i32);
    pub const UNBIND: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DeviceAddressBindingTypeEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("BIND"),
            1i32 => f.write_str("UNBIND"),
            other => write!(f, "{}({})", stringify!(DeviceAddressBindingTypeEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkLayeredDriverUnderlyingApiMSFT")]
pub struct LayeredDriverUnderlyingApiMSFT(i32);
impl LayeredDriverUnderlyingApiMSFT {
    pub const NONE: Self = Self(0i32);
    pub const D3D12: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for LayeredDriverUnderlyingApiMSFT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("NONE"),
            1i32 => f.write_str("D3D12"),
            other => {
                write!(f, "{}({})", stringify!(LayeredDriverUnderlyingApiMSFT), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDriverId")]
pub struct DriverId(i32);
impl DriverId {
    pub const AMD_PROPRIETARY: Self = Self(1i32);
    pub const AMD_OPEN_SOURCE: Self = Self(2i32);
    pub const MESA_RADV: Self = Self(3i32);
    pub const NVIDIA_PROPRIETARY: Self = Self(4i32);
    pub const INTEL_PROPRIETARY_WINDOWS: Self = Self(5i32);
    pub const INTEL_OPEN_SOURCE: Self = Self(6i32);
    pub const IMAGINATION_PROPRIETARY: Self = Self(7i32);
    pub const QUALCOMM_PROPRIETARY: Self = Self(8i32);
    pub const ARM_PROPRIETARY: Self = Self(9i32);
    pub const GOOGLE_SWIFTSHADER: Self = Self(10i32);
    pub const GGP_PROPRIETARY: Self = Self(11i32);
    pub const BROADCOM_PROPRIETARY: Self = Self(12i32);
    pub const MESA_LLVMPIPE: Self = Self(13i32);
    pub const MOLTENVK: Self = Self(14i32);
    pub const COREAVI_PROPRIETARY: Self = Self(15i32);
    pub const JUICE_PROPRIETARY: Self = Self(16i32);
    pub const VERISILICON_PROPRIETARY: Self = Self(17i32);
    pub const MESA_TURNIP: Self = Self(18i32);
    pub const MESA_V3DV: Self = Self(19i32);
    pub const MESA_PANVK: Self = Self(20i32);
    pub const SAMSUNG_PROPRIETARY: Self = Self(21i32);
    pub const MESA_VENUS: Self = Self(22i32);
    pub const MESA_DOZEN: Self = Self(23i32);
    pub const MESA_NVK: Self = Self(24i32);
    pub const IMAGINATION_OPEN_SOURCE: Self = Self(25i32);
    pub const MESA_HONEYKRISP: Self = Self(26i32);
    pub const VULKAN_SC_EMULATION_ON_VULKAN: Self = Self(27i32);
    pub const MESA_KOSMICKRISP: Self = Self(28i32);
    pub const INTEL_OPEN_SOURCE_: Self = Self::INTEL_OPEN_SOURCE;
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DriverId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            1i32 => f.write_str("AMD_PROPRIETARY"),
            2i32 => f.write_str("AMD_OPEN_SOURCE"),
            3i32 => f.write_str("MESA_RADV"),
            4i32 => f.write_str("NVIDIA_PROPRIETARY"),
            5i32 => f.write_str("INTEL_PROPRIETARY_WINDOWS"),
            6i32 => f.write_str("INTEL_OPEN_SOURCE"),
            7i32 => f.write_str("IMAGINATION_PROPRIETARY"),
            8i32 => f.write_str("QUALCOMM_PROPRIETARY"),
            9i32 => f.write_str("ARM_PROPRIETARY"),
            10i32 => f.write_str("GOOGLE_SWIFTSHADER"),
            11i32 => f.write_str("GGP_PROPRIETARY"),
            12i32 => f.write_str("BROADCOM_PROPRIETARY"),
            13i32 => f.write_str("MESA_LLVMPIPE"),
            14i32 => f.write_str("MOLTENVK"),
            15i32 => f.write_str("COREAVI_PROPRIETARY"),
            16i32 => f.write_str("JUICE_PROPRIETARY"),
            17i32 => f.write_str("VERISILICON_PROPRIETARY"),
            18i32 => f.write_str("MESA_TURNIP"),
            19i32 => f.write_str("MESA_V3DV"),
            20i32 => f.write_str("MESA_PANVK"),
            21i32 => f.write_str("SAMSUNG_PROPRIETARY"),
            22i32 => f.write_str("MESA_VENUS"),
            23i32 => f.write_str("MESA_DOZEN"),
            24i32 => f.write_str("MESA_NVK"),
            25i32 => f.write_str("IMAGINATION_OPEN_SOURCE"),
            26i32 => f.write_str("MESA_HONEYKRISP"),
            27i32 => f.write_str("VULKAN_SC_EMULATION_ON_VULKAN"),
            28i32 => f.write_str("MESA_KOSMICKRISP"),
            other => write!(f, "{}({})", stringify!(DriverId), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDefaultVertexAttributeValueKHR")]
pub struct DefaultVertexAttributeValueKHR(i32);
impl DefaultVertexAttributeValueKHR {
    pub const ZERO_ZERO_ZERO_ZERO: Self = Self(0i32);
    pub const ZERO_ZERO_ZERO_ONE: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DefaultVertexAttributeValueKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("ZERO_ZERO_ZERO_ZERO"),
            1i32 => f.write_str("ZERO_ZERO_ZERO_ONE"),
            other => {
                write!(f, "{}({})", stringify!(DefaultVertexAttributeValueKHR), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkViewportCoordinateSwizzleNV")]
pub struct ViewportCoordinateSwizzleNV(i32);
impl ViewportCoordinateSwizzleNV {
    pub const POSITIVE_X: Self = Self(0i32);
    pub const NEGATIVE_X: Self = Self(1i32);
    pub const POSITIVE_Y: Self = Self(2i32);
    pub const NEGATIVE_Y: Self = Self(3i32);
    pub const POSITIVE_Z: Self = Self(4i32);
    pub const NEGATIVE_Z: Self = Self(5i32);
    pub const POSITIVE_W: Self = Self(6i32);
    pub const NEGATIVE_W: Self = Self(7i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ViewportCoordinateSwizzleNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("POSITIVE_X"),
            1i32 => f.write_str("NEGATIVE_X"),
            2i32 => f.write_str("POSITIVE_Y"),
            3i32 => f.write_str("NEGATIVE_Y"),
            4i32 => f.write_str("POSITIVE_Z"),
            5i32 => f.write_str("NEGATIVE_Z"),
            6i32 => f.write_str("POSITIVE_W"),
            7i32 => f.write_str("NEGATIVE_W"),
            other => write!(f, "{}({})", stringify!(ViewportCoordinateSwizzleNV), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDataGraphPipelineSessionBindPointTypeARM")]
pub struct DataGraphPipelineSessionBindPointTypeARM(i32);
impl DataGraphPipelineSessionBindPointTypeARM {
    pub const MEMORY: Self = Self(0i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DataGraphPipelineSessionBindPointTypeARM {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("MEMORY"),
            other => {
                write!(
                    f, "{}({})", stringify!(DataGraphPipelineSessionBindPointTypeARM),
                    other
                )
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkFragmentShadingRateCombinerOpKHR")]
pub struct FragmentShadingRateCombinerOpKHR(i32);
impl FragmentShadingRateCombinerOpKHR {
    pub const KEEP: Self = Self(0i32);
    pub const REPLACE: Self = Self(1i32);
    pub const MIN: Self = Self(2i32);
    pub const MAX: Self = Self(3i32);
    pub const MUL: Self = Self(4i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for FragmentShadingRateCombinerOpKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("KEEP"),
            1i32 => f.write_str("REPLACE"),
            2i32 => f.write_str("MIN"),
            3i32 => f.write_str("MAX"),
            4i32 => f.write_str("MUL"),
            other => {
                write!(f, "{}({})", stringify!(FragmentShadingRateCombinerOpKHR), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkSystemAllocationScope")]
pub struct SystemAllocationScope(i32);
impl SystemAllocationScope {
    pub const COMMAND: Self = Self(0i32);
    pub const OBJECT: Self = Self(1i32);
    pub const CACHE: Self = Self(2i32);
    pub const DEVICE: Self = Self(3i32);
    pub const INSTANCE: Self = Self(4i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for SystemAllocationScope {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("COMMAND"),
            1i32 => f.write_str("OBJECT"),
            2i32 => f.write_str("CACHE"),
            3i32 => f.write_str("DEVICE"),
            4i32 => f.write_str("INSTANCE"),
            other => write!(f, "{}({})", stringify!(SystemAllocationScope), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkCoarseSampleOrderTypeNV")]
pub struct CoarseSampleOrderTypeNV(i32);
impl CoarseSampleOrderTypeNV {
    pub const DEFAULT: Self = Self(0i32);
    pub const CUSTOM: Self = Self(1i32);
    pub const PIXEL_MAJOR: Self = Self(2i32);
    pub const SAMPLE_MAJOR: Self = Self(3i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for CoarseSampleOrderTypeNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("DEFAULT"),
            1i32 => f.write_str("CUSTOM"),
            2i32 => f.write_str("PIXEL_MAJOR"),
            3i32 => f.write_str("SAMPLE_MAJOR"),
            other => write!(f, "{}({})", stringify!(CoarseSampleOrderTypeNV), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPerformanceValueTypeINTEL")]
pub struct PerformanceValueTypeINTEL(i32);
impl PerformanceValueTypeINTEL {
    pub const UINT32: Self = Self(0i32);
    pub const UINT64: Self = Self(1i32);
    pub const FLOAT: Self = Self(2i32);
    pub const BOOL: Self = Self(3i32);
    pub const STRING: Self = Self(4i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PerformanceValueTypeINTEL {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("UINT32"),
            1i32 => f.write_str("UINT64"),
            2i32 => f.write_str("FLOAT"),
            3i32 => f.write_str("BOOL"),
            4i32 => f.write_str("STRING"),
            other => write!(f, "{}({})", stringify!(PerformanceValueTypeINTEL), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDataGraphModelCacheTypeQCOM")]
pub struct DataGraphModelCacheTypeQCOM(i32);
impl DataGraphModelCacheTypeQCOM {
    pub const GENERIC_BINARY: Self = Self(0i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DataGraphModelCacheTypeQCOM {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("GENERIC_BINARY"),
            other => write!(f, "{}({})", stringify!(DataGraphModelCacheTypeQCOM), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkGeometryTypeKHR")]
pub struct GeometryTypeKHR(i32);
impl GeometryTypeKHR {
    pub const TRIANGLES: Self = Self(0i32);
    pub const AABBS: Self = Self(1i32);
    pub const INSTANCES: Self = Self(2i32);
    pub const SPHERES: Self = Self(1000429004i32);
    pub const LINEAR_SWEPT_SPHERES: Self = Self(1000429005i32);
    pub const DENSE_GEOMETRY_FORMAT_TRIANGLES: Self = Self(1000478000i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for GeometryTypeKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("TRIANGLES"),
            1i32 => f.write_str("AABBS"),
            2i32 => f.write_str("INSTANCES"),
            1000429004i32 => f.write_str("SPHERES"),
            1000429005i32 => f.write_str("LINEAR_SWEPT_SPHERES"),
            1000478000i32 => f.write_str("DENSE_GEOMETRY_FORMAT_TRIANGLES"),
            other => write!(f, "{}({})", stringify!(GeometryTypeKHR), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkShaderInfoTypeAMD")]
pub struct ShaderInfoTypeAMD(i32);
impl ShaderInfoTypeAMD {
    pub const STATISTICS: Self = Self(0i32);
    pub const BINARY: Self = Self(1i32);
    pub const DISASSEMBLY: Self = Self(2i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ShaderInfoTypeAMD {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("STATISTICS"),
            1i32 => f.write_str("BINARY"),
            2i32 => f.write_str("DISASSEMBLY"),
            other => write!(f, "{}({})", stringify!(ShaderInfoTypeAMD), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkAttachmentStoreOp")]
pub struct AttachmentStoreOp(i32);
impl AttachmentStoreOp {
    pub const STORE: Self = Self(0i32);
    pub const DONT_CARE: Self = Self(1i32);
    pub const NONE: Self = Self(1000301000i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for AttachmentStoreOp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("STORE"),
            1i32 => f.write_str("DONT_CARE"),
            1000301000i32 => f.write_str("NONE"),
            other => write!(f, "{}({})", stringify!(AttachmentStoreOp), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkLineRasterizationMode")]
pub struct LineRasterizationMode(i32);
impl LineRasterizationMode {
    pub const DEFAULT: Self = Self(0i32);
    pub const RECTANGULAR: Self = Self(1i32);
    pub const BRESENHAM: Self = Self(2i32);
    pub const RECTANGULAR_SMOOTH: Self = Self(3i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for LineRasterizationMode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("DEFAULT"),
            1i32 => f.write_str("RECTANGULAR"),
            2i32 => f.write_str("BRESENHAM"),
            3i32 => f.write_str("RECTANGULAR_SMOOTH"),
            other => write!(f, "{}({})", stringify!(LineRasterizationMode), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPipelineCacheValidationVersion")]
pub struct PipelineCacheValidationVersion(i32);
impl PipelineCacheValidationVersion {
    pub const SAFETY_CRITICAL_ONE: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PipelineCacheValidationVersion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            1i32 => f.write_str("SAFETY_CRITICAL_ONE"),
            other => {
                write!(f, "{}({})", stringify!(PipelineCacheValidationVersion), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkOpticalFlowSessionBindingPointNV")]
pub struct OpticalFlowSessionBindingPointNV(i32);
impl OpticalFlowSessionBindingPointNV {
    pub const UNKNOWN: Self = Self(0i32);
    pub const INPUT: Self = Self(1i32);
    pub const REFERENCE: Self = Self(2i32);
    pub const HINT: Self = Self(3i32);
    pub const FLOW_VECTOR: Self = Self(4i32);
    pub const BACKWARD_FLOW_VECTOR: Self = Self(5i32);
    pub const COST: Self = Self(6i32);
    pub const BACKWARD_COST: Self = Self(7i32);
    pub const GLOBAL_FLOW: Self = Self(8i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for OpticalFlowSessionBindingPointNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("UNKNOWN"),
            1i32 => f.write_str("INPUT"),
            2i32 => f.write_str("REFERENCE"),
            3i32 => f.write_str("HINT"),
            4i32 => f.write_str("FLOW_VECTOR"),
            5i32 => f.write_str("BACKWARD_FLOW_VECTOR"),
            6i32 => f.write_str("COST"),
            7i32 => f.write_str("BACKWARD_COST"),
            8i32 => f.write_str("GLOBAL_FLOW"),
            other => {
                write!(f, "{}({})", stringify!(OpticalFlowSessionBindingPointNV), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkCopyMicromapModeEXT")]
pub struct CopyMicromapModeEXT(i32);
impl CopyMicromapModeEXT {
    pub const CLONE: Self = Self(0i32);
    pub const SERIALIZE: Self = Self(1i32);
    pub const DESERIALIZE: Self = Self(2i32);
    pub const COMPACT: Self = Self(3i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for CopyMicromapModeEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("CLONE"),
            1i32 => f.write_str("SERIALIZE"),
            2i32 => f.write_str("DESERIALIZE"),
            3i32 => f.write_str("COMPACT"),
            other => write!(f, "{}({})", stringify!(CopyMicromapModeEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkOpacityMicromapFormatEXT")]
pub struct OpacityMicromapFormatEXT(i32);
impl OpacityMicromapFormatEXT {
    pub const _2_STATE: Self = Self(1i32);
    pub const _4_STATE: Self = Self(2i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for OpacityMicromapFormatEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            1i32 => f.write_str("_2_STATE"),
            2i32 => f.write_str("_4_STATE"),
            other => write!(f, "{}({})", stringify!(OpacityMicromapFormatEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkComponentTypeKHR")]
pub struct ComponentTypeKHR(i32);
impl ComponentTypeKHR {
    pub const FLOAT16: Self = Self(0i32);
    pub const FLOAT32: Self = Self(1i32);
    pub const FLOAT64: Self = Self(2i32);
    pub const SINT8: Self = Self(3i32);
    pub const SINT16: Self = Self(4i32);
    pub const SINT32: Self = Self(5i32);
    pub const SINT64: Self = Self(6i32);
    pub const UINT8: Self = Self(7i32);
    pub const UINT16: Self = Self(8i32);
    pub const UINT32: Self = Self(9i32);
    pub const UINT64: Self = Self(10i32);
    pub const BFLOAT16: Self = Self(1000141000i32);
    pub const SINT8_PACKED: Self = Self(1000491000i32);
    pub const UINT8_PACKED: Self = Self(1000491001i32);
    pub const FLOAT_E4M3: Self = Self::FLOAT8_E4M3;
    pub const FLOAT_E5M2: Self = Self::FLOAT8_E5M2;
    pub const FLOAT8_E4M3: Self = Self(1000491002i32);
    pub const FLOAT8_E5M2: Self = Self(1000491003i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ComponentTypeKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("FLOAT16"),
            1i32 => f.write_str("FLOAT32"),
            2i32 => f.write_str("FLOAT64"),
            3i32 => f.write_str("SINT8"),
            4i32 => f.write_str("SINT16"),
            5i32 => f.write_str("SINT32"),
            6i32 => f.write_str("SINT64"),
            7i32 => f.write_str("UINT8"),
            8i32 => f.write_str("UINT16"),
            9i32 => f.write_str("UINT32"),
            10i32 => f.write_str("UINT64"),
            1000141000i32 => f.write_str("BFLOAT16"),
            1000491000i32 => f.write_str("SINT8_PACKED"),
            1000491001i32 => f.write_str("UINT8_PACKED"),
            1000491002i32 => f.write_str("FLOAT8_E4M3"),
            1000491003i32 => f.write_str("FLOAT8_E5M2"),
            other => write!(f, "{}({})", stringify!(ComponentTypeKHR), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkImageViewType")]
pub struct ImageViewType(i32);
impl ImageViewType {
    pub const _1D: Self = Self(0i32);
    pub const _2D: Self = Self(1i32);
    pub const _3D: Self = Self(2i32);
    pub const CUBE: Self = Self(3i32);
    pub const _1D_ARRAY: Self = Self(4i32);
    pub const _2D_ARRAY: Self = Self(5i32);
    pub const CUBE_ARRAY: Self = Self(6i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ImageViewType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("_1D"),
            1i32 => f.write_str("_2D"),
            2i32 => f.write_str("_3D"),
            3i32 => f.write_str("CUBE"),
            4i32 => f.write_str("_1D_ARRAY"),
            5i32 => f.write_str("_2D_ARRAY"),
            6i32 => f.write_str("CUBE_ARRAY"),
            other => write!(f, "{}({})", stringify!(ImageViewType), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPipelineRobustnessBufferBehavior")]
pub struct PipelineRobustnessBufferBehavior(i32);
impl PipelineRobustnessBufferBehavior {
    pub const DEVICE_DEFAULT: Self = Self(0i32);
    pub const DISABLED: Self = Self(1i32);
    pub const ROBUST_BUFFER_ACCESS: Self = Self(2i32);
    pub const ROBUST_BUFFER_ACCESS_2: Self = Self(3i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PipelineRobustnessBufferBehavior {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("DEVICE_DEFAULT"),
            1i32 => f.write_str("DISABLED"),
            2i32 => f.write_str("ROBUST_BUFFER_ACCESS"),
            3i32 => f.write_str("ROBUST_BUFFER_ACCESS_2"),
            other => {
                write!(f, "{}({})", stringify!(PipelineRobustnessBufferBehavior), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkQueryPoolSamplingModeINTEL")]
pub struct QueryPoolSamplingModeINTEL(i32);
impl QueryPoolSamplingModeINTEL {
    pub const MANUAL: Self = Self(0i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for QueryPoolSamplingModeINTEL {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("MANUAL"),
            other => write!(f, "{}({})", stringify!(QueryPoolSamplingModeINTEL), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkImageType")]
pub struct ImageType(i32);
impl ImageType {
    pub const _1D: Self = Self(0i32);
    pub const _2D: Self = Self(1i32);
    pub const _3D: Self = Self(2i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ImageType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("_1D"),
            1i32 => f.write_str("_2D"),
            2i32 => f.write_str("_3D"),
            other => write!(f, "{}({})", stringify!(ImageType), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkVideoEncodeAV1RateControlGroupKHR")]
pub struct VideoEncodeAV1RateControlGroupKHR(i32);
impl VideoEncodeAV1RateControlGroupKHR {
    pub const INTRA: Self = Self(0i32);
    pub const PREDICTIVE: Self = Self(1i32);
    pub const BIPREDICTIVE: Self = Self(2i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for VideoEncodeAV1RateControlGroupKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("INTRA"),
            1i32 => f.write_str("PREDICTIVE"),
            2i32 => f.write_str("BIPREDICTIVE"),
            other => {
                write!(f, "{}({})", stringify!(VideoEncodeAV1RateControlGroupKHR), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkImageTiling")]
pub struct ImageTiling(i32);
impl ImageTiling {
    pub const OPTIMAL: Self = Self(0i32);
    pub const LINEAR: Self = Self(1i32);
    pub const DRM_FORMAT_MODIFIER: Self = Self(1000158000i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ImageTiling {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("OPTIMAL"),
            1i32 => f.write_str("LINEAR"),
            1000158000i32 => f.write_str("DRM_FORMAT_MODIFIER"),
            other => write!(f, "{}({})", stringify!(ImageTiling), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkCommandBufferLevel")]
pub struct CommandBufferLevel(i32);
impl CommandBufferLevel {
    pub const PRIMARY: Self = Self(0i32);
    pub const SECONDARY: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for CommandBufferLevel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("PRIMARY"),
            1i32 => f.write_str("SECONDARY"),
            other => write!(f, "{}({})", stringify!(CommandBufferLevel), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkBuildMicromapModeEXT")]
pub struct BuildMicromapModeEXT(i32);
impl BuildMicromapModeEXT {
    pub const BUILD: Self = Self(0i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for BuildMicromapModeEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("BUILD"),
            other => write!(f, "{}({})", stringify!(BuildMicromapModeEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkValidationCacheHeaderVersionEXT")]
pub struct ValidationCacheHeaderVersionEXT(i32);
impl ValidationCacheHeaderVersionEXT {
    pub const ONE: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ValidationCacheHeaderVersionEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            1i32 => f.write_str("ONE"),
            other => {
                write!(f, "{}({})", stringify!(ValidationCacheHeaderVersionEXT), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkCompressedTriangleFormatAMDX")]
pub struct CompressedTriangleFormatAMDX(i32);
impl CompressedTriangleFormatAMDX {
    pub const DGF1: Self = Self(0i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for CompressedTriangleFormatAMDX {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("DGF1"),
            other => write!(f, "{}({})", stringify!(CompressedTriangleFormatAMDX), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkInternalAllocationType")]
pub struct InternalAllocationType(i32);
impl InternalAllocationType {
    pub const EXECUTABLE: Self = Self(0i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for InternalAllocationType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("EXECUTABLE"),
            other => write!(f, "{}({})", stringify!(InternalAllocationType), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkClusterAccelerationStructureOpModeNV")]
pub struct ClusterAccelerationStructureOpModeNV(i32);
impl ClusterAccelerationStructureOpModeNV {
    pub const IMPLICIT_DESTINATIONS: Self = Self(0i32);
    pub const EXPLICIT_DESTINATIONS: Self = Self(1i32);
    pub const COMPUTE_SIZES: Self = Self(2i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ClusterAccelerationStructureOpModeNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("IMPLICIT_DESTINATIONS"),
            1i32 => f.write_str("EXPLICIT_DESTINATIONS"),
            2i32 => f.write_str("COMPUTE_SIZES"),
            other => {
                write!(
                    f, "{}({})", stringify!(ClusterAccelerationStructureOpModeNV), other
                )
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkTessellationDomainOrigin")]
pub struct TessellationDomainOrigin(i32);
impl TessellationDomainOrigin {
    pub const UPPER_LEFT: Self = Self(0i32);
    pub const LOWER_LEFT: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for TessellationDomainOrigin {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("UPPER_LEFT"),
            1i32 => f.write_str("LOWER_LEFT"),
            other => write!(f, "{}({})", stringify!(TessellationDomainOrigin), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPolygonMode")]
pub struct PolygonMode(i32);
impl PolygonMode {
    pub const FILL: Self = Self(0i32);
    pub const LINE: Self = Self(1i32);
    pub const POINT: Self = Self(2i32);
    pub const FILL_RECTANGLE: Self = Self(1000153000i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PolygonMode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("FILL"),
            1i32 => f.write_str("LINE"),
            2i32 => f.write_str("POINT"),
            1000153000i32 => f.write_str("FILL_RECTANGLE"),
            other => write!(f, "{}({})", stringify!(PolygonMode), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDynamicState")]
pub struct DynamicState(i32);
impl DynamicState {
    pub const VIEWPORT: Self = Self(0i32);
    pub const SCISSOR: Self = Self(1i32);
    pub const LINE_WIDTH: Self = Self(2i32);
    pub const DEPTH_BIAS: Self = Self(3i32);
    pub const BLEND_CONSTANTS: Self = Self(4i32);
    pub const DEPTH_BOUNDS: Self = Self(5i32);
    pub const STENCIL_COMPARE_MASK: Self = Self(6i32);
    pub const STENCIL_WRITE_MASK: Self = Self(7i32);
    pub const STENCIL_REFERENCE: Self = Self(8i32);
    pub const CULL_MODE: Self = Self(1000267000i32);
    pub const FRONT_FACE: Self = Self(1000267001i32);
    pub const PRIMITIVE_TOPOLOGY: Self = Self(1000267002i32);
    pub const VIEWPORT_WITH_COUNT: Self = Self(1000267003i32);
    pub const SCISSOR_WITH_COUNT: Self = Self(1000267004i32);
    pub const VERTEX_INPUT_BINDING_STRIDE: Self = Self(1000267005i32);
    pub const DEPTH_TEST_ENABLE: Self = Self(1000267006i32);
    pub const DEPTH_WRITE_ENABLE: Self = Self(1000267007i32);
    pub const DEPTH_COMPARE_OP: Self = Self(1000267008i32);
    pub const DEPTH_BOUNDS_TEST_ENABLE: Self = Self(1000267009i32);
    pub const STENCIL_TEST_ENABLE: Self = Self(1000267010i32);
    pub const STENCIL_OP: Self = Self(1000267011i32);
    pub const RASTERIZER_DISCARD_ENABLE: Self = Self(1000377001i32);
    pub const DEPTH_BIAS_ENABLE: Self = Self(1000377002i32);
    pub const PRIMITIVE_RESTART_ENABLE: Self = Self(1000377004i32);
    pub const LINE_STIPPLE: Self = Self(1000259000i32);
    pub const VIEWPORT_W_SCALING: Self = Self(1000087000i32);
    pub const DISCARD_RECTANGLE: Self = Self(1000099000i32);
    pub const DISCARD_RECTANGLE_ENABLE: Self = Self(1000099001i32);
    pub const DISCARD_RECTANGLE_MODE: Self = Self(1000099002i32);
    pub const SAMPLE_LOCATIONS: Self = Self(1000143000i32);
    pub const RAY_TRACING_PIPELINE_STACK_SIZE: Self = Self(1000347000i32);
    pub const VIEWPORT_SHADING_RATE_PALETTE: Self = Self(1000164004i32);
    pub const VIEWPORT_COARSE_SAMPLE_ORDER: Self = Self(1000164006i32);
    pub const EXCLUSIVE_SCISSOR_ENABLE: Self = Self(1000205000i32);
    pub const EXCLUSIVE_SCISSOR: Self = Self(1000205001i32);
    pub const FRAGMENT_SHADING_RATE: Self = Self(1000226000i32);
    pub const VERTEX_INPUT: Self = Self(1000352000i32);
    pub const PATCH_CONTROL_POINTS: Self = Self(1000377000i32);
    pub const LOGIC_OP: Self = Self(1000377003i32);
    pub const COLOR_WRITE_ENABLE: Self = Self(1000381000i32);
    pub const DEPTH_CLAMP_ENABLE: Self = Self(1000455003i32);
    pub const POLYGON_MODE: Self = Self(1000455004i32);
    pub const RASTERIZATION_SAMPLES: Self = Self(1000455005i32);
    pub const SAMPLE_MASK: Self = Self(1000455006i32);
    pub const ALPHA_TO_COVERAGE_ENABLE: Self = Self(1000455007i32);
    pub const ALPHA_TO_ONE_ENABLE: Self = Self(1000455008i32);
    pub const LOGIC_OP_ENABLE: Self = Self(1000455009i32);
    pub const COLOR_BLEND_ENABLE: Self = Self(1000455010i32);
    pub const COLOR_BLEND_EQUATION: Self = Self(1000455011i32);
    pub const COLOR_WRITE_MASK: Self = Self(1000455012i32);
    pub const TESSELLATION_DOMAIN_ORIGIN: Self = Self(1000455002i32);
    pub const RASTERIZATION_STREAM: Self = Self(1000455013i32);
    pub const CONSERVATIVE_RASTERIZATION_MODE: Self = Self(1000455014i32);
    pub const EXTRA_PRIMITIVE_OVERESTIMATION_SIZE: Self = Self(1000455015i32);
    pub const DEPTH_CLIP_ENABLE: Self = Self(1000455016i32);
    pub const SAMPLE_LOCATIONS_ENABLE: Self = Self(1000455017i32);
    pub const COLOR_BLEND_ADVANCED: Self = Self(1000455018i32);
    pub const PROVOKING_VERTEX_MODE: Self = Self(1000455019i32);
    pub const LINE_RASTERIZATION_MODE: Self = Self(1000455020i32);
    pub const LINE_STIPPLE_ENABLE: Self = Self(1000455021i32);
    pub const DEPTH_CLIP_NEGATIVE_ONE_TO_ONE: Self = Self(1000455022i32);
    pub const VIEWPORT_W_SCALING_ENABLE: Self = Self(1000455023i32);
    pub const VIEWPORT_SWIZZLE: Self = Self(1000455024i32);
    pub const COVERAGE_TO_COLOR_ENABLE: Self = Self(1000455025i32);
    pub const COVERAGE_TO_COLOR_LOCATION: Self = Self(1000455026i32);
    pub const COVERAGE_MODULATION_MODE: Self = Self(1000455027i32);
    pub const COVERAGE_MODULATION_TABLE_ENABLE: Self = Self(1000455028i32);
    pub const COVERAGE_MODULATION_TABLE: Self = Self(1000455029i32);
    pub const SHADING_RATE_IMAGE_ENABLE: Self = Self(1000455030i32);
    pub const REPRESENTATIVE_FRAGMENT_TEST_ENABLE: Self = Self(1000455031i32);
    pub const COVERAGE_REDUCTION_MODE: Self = Self(1000455032i32);
    pub const ATTACHMENT_FEEDBACK_LOOP_ENABLE: Self = Self(1000524000i32);
    pub const DEPTH_CLAMP_RANGE: Self = Self(1000582000i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DynamicState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("VIEWPORT"),
            1i32 => f.write_str("SCISSOR"),
            2i32 => f.write_str("LINE_WIDTH"),
            3i32 => f.write_str("DEPTH_BIAS"),
            4i32 => f.write_str("BLEND_CONSTANTS"),
            5i32 => f.write_str("DEPTH_BOUNDS"),
            6i32 => f.write_str("STENCIL_COMPARE_MASK"),
            7i32 => f.write_str("STENCIL_WRITE_MASK"),
            8i32 => f.write_str("STENCIL_REFERENCE"),
            1000267000i32 => f.write_str("CULL_MODE"),
            1000267001i32 => f.write_str("FRONT_FACE"),
            1000267002i32 => f.write_str("PRIMITIVE_TOPOLOGY"),
            1000267003i32 => f.write_str("VIEWPORT_WITH_COUNT"),
            1000267004i32 => f.write_str("SCISSOR_WITH_COUNT"),
            1000267005i32 => f.write_str("VERTEX_INPUT_BINDING_STRIDE"),
            1000267006i32 => f.write_str("DEPTH_TEST_ENABLE"),
            1000267007i32 => f.write_str("DEPTH_WRITE_ENABLE"),
            1000267008i32 => f.write_str("DEPTH_COMPARE_OP"),
            1000267009i32 => f.write_str("DEPTH_BOUNDS_TEST_ENABLE"),
            1000267010i32 => f.write_str("STENCIL_TEST_ENABLE"),
            1000267011i32 => f.write_str("STENCIL_OP"),
            1000377001i32 => f.write_str("RASTERIZER_DISCARD_ENABLE"),
            1000377002i32 => f.write_str("DEPTH_BIAS_ENABLE"),
            1000377004i32 => f.write_str("PRIMITIVE_RESTART_ENABLE"),
            1000259000i32 => f.write_str("LINE_STIPPLE"),
            1000087000i32 => f.write_str("VIEWPORT_W_SCALING"),
            1000099000i32 => f.write_str("DISCARD_RECTANGLE"),
            1000099001i32 => f.write_str("DISCARD_RECTANGLE_ENABLE"),
            1000099002i32 => f.write_str("DISCARD_RECTANGLE_MODE"),
            1000143000i32 => f.write_str("SAMPLE_LOCATIONS"),
            1000347000i32 => f.write_str("RAY_TRACING_PIPELINE_STACK_SIZE"),
            1000164004i32 => f.write_str("VIEWPORT_SHADING_RATE_PALETTE"),
            1000164006i32 => f.write_str("VIEWPORT_COARSE_SAMPLE_ORDER"),
            1000205000i32 => f.write_str("EXCLUSIVE_SCISSOR_ENABLE"),
            1000205001i32 => f.write_str("EXCLUSIVE_SCISSOR"),
            1000226000i32 => f.write_str("FRAGMENT_SHADING_RATE"),
            1000352000i32 => f.write_str("VERTEX_INPUT"),
            1000377000i32 => f.write_str("PATCH_CONTROL_POINTS"),
            1000377003i32 => f.write_str("LOGIC_OP"),
            1000381000i32 => f.write_str("COLOR_WRITE_ENABLE"),
            1000455003i32 => f.write_str("DEPTH_CLAMP_ENABLE"),
            1000455004i32 => f.write_str("POLYGON_MODE"),
            1000455005i32 => f.write_str("RASTERIZATION_SAMPLES"),
            1000455006i32 => f.write_str("SAMPLE_MASK"),
            1000455007i32 => f.write_str("ALPHA_TO_COVERAGE_ENABLE"),
            1000455008i32 => f.write_str("ALPHA_TO_ONE_ENABLE"),
            1000455009i32 => f.write_str("LOGIC_OP_ENABLE"),
            1000455010i32 => f.write_str("COLOR_BLEND_ENABLE"),
            1000455011i32 => f.write_str("COLOR_BLEND_EQUATION"),
            1000455012i32 => f.write_str("COLOR_WRITE_MASK"),
            1000455002i32 => f.write_str("TESSELLATION_DOMAIN_ORIGIN"),
            1000455013i32 => f.write_str("RASTERIZATION_STREAM"),
            1000455014i32 => f.write_str("CONSERVATIVE_RASTERIZATION_MODE"),
            1000455015i32 => f.write_str("EXTRA_PRIMITIVE_OVERESTIMATION_SIZE"),
            1000455016i32 => f.write_str("DEPTH_CLIP_ENABLE"),
            1000455017i32 => f.write_str("SAMPLE_LOCATIONS_ENABLE"),
            1000455018i32 => f.write_str("COLOR_BLEND_ADVANCED"),
            1000455019i32 => f.write_str("PROVOKING_VERTEX_MODE"),
            1000455020i32 => f.write_str("LINE_RASTERIZATION_MODE"),
            1000455021i32 => f.write_str("LINE_STIPPLE_ENABLE"),
            1000455022i32 => f.write_str("DEPTH_CLIP_NEGATIVE_ONE_TO_ONE"),
            1000455023i32 => f.write_str("VIEWPORT_W_SCALING_ENABLE"),
            1000455024i32 => f.write_str("VIEWPORT_SWIZZLE"),
            1000455025i32 => f.write_str("COVERAGE_TO_COLOR_ENABLE"),
            1000455026i32 => f.write_str("COVERAGE_TO_COLOR_LOCATION"),
            1000455027i32 => f.write_str("COVERAGE_MODULATION_MODE"),
            1000455028i32 => f.write_str("COVERAGE_MODULATION_TABLE_ENABLE"),
            1000455029i32 => f.write_str("COVERAGE_MODULATION_TABLE"),
            1000455030i32 => f.write_str("SHADING_RATE_IMAGE_ENABLE"),
            1000455031i32 => f.write_str("REPRESENTATIVE_FRAGMENT_TEST_ENABLE"),
            1000455032i32 => f.write_str("COVERAGE_REDUCTION_MODE"),
            1000524000i32 => f.write_str("ATTACHMENT_FEEDBACK_LOOP_ENABLE"),
            1000582000i32 => f.write_str("DEPTH_CLAMP_RANGE"),
            other => write!(f, "{}({})", stringify!(DynamicState), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkClusterAccelerationStructureOpTypeNV")]
pub struct ClusterAccelerationStructureOpTypeNV(i32);
impl ClusterAccelerationStructureOpTypeNV {
    pub const MOVE_OBJECTS: Self = Self(0i32);
    pub const BUILD_CLUSTERS_BOTTOM_LEVEL: Self = Self(1i32);
    pub const BUILD_TRIANGLE_CLUSTER: Self = Self(2i32);
    pub const BUILD_TRIANGLE_CLUSTER_TEMPLATE: Self = Self(3i32);
    pub const INSTANTIATE_TRIANGLE_CLUSTER: Self = Self(4i32);
    pub const GET_CLUSTER_TEMPLATE_INDICES: Self = Self(5i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ClusterAccelerationStructureOpTypeNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("MOVE_OBJECTS"),
            1i32 => f.write_str("BUILD_CLUSTERS_BOTTOM_LEVEL"),
            2i32 => f.write_str("BUILD_TRIANGLE_CLUSTER"),
            3i32 => f.write_str("BUILD_TRIANGLE_CLUSTER_TEMPLATE"),
            4i32 => f.write_str("INSTANTIATE_TRIANGLE_CLUSTER"),
            5i32 => f.write_str("GET_CLUSTER_TEMPLATE_INDICES"),
            other => {
                write!(
                    f, "{}({})", stringify!(ClusterAccelerationStructureOpTypeNV), other
                )
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkFaultQueryBehavior")]
pub struct FaultQueryBehavior(i32);
impl FaultQueryBehavior {
    pub const GET_AND_CLEAR_ALL_FAULTS: Self = Self(0i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for FaultQueryBehavior {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("GET_AND_CLEAR_ALL_FAULTS"),
            other => write!(f, "{}({})", stringify!(FaultQueryBehavior), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkImageLayout")]
pub struct ImageLayout(i32);
impl ImageLayout {
    pub const UNDEFINED: Self = Self(0i32);
    pub const GENERAL: Self = Self(1i32);
    pub const COLOR_ATTACHMENT_OPTIMAL: Self = Self(2i32);
    pub const DEPTH_STENCIL_ATTACHMENT_OPTIMAL: Self = Self(3i32);
    pub const DEPTH_STENCIL_READ_ONLY_OPTIMAL: Self = Self(4i32);
    pub const SHADER_READ_ONLY_OPTIMAL: Self = Self(5i32);
    pub const TRANSFER_SRC_OPTIMAL: Self = Self(6i32);
    pub const TRANSFER_DST_OPTIMAL: Self = Self(7i32);
    pub const PREINITIALIZED: Self = Self(8i32);
    pub const DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL: Self = Self(1000117000i32);
    pub const DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL: Self = Self(1000117001i32);
    pub const DEPTH_ATTACHMENT_OPTIMAL: Self = Self(1000241000i32);
    pub const DEPTH_READ_ONLY_OPTIMAL: Self = Self(1000241001i32);
    pub const STENCIL_ATTACHMENT_OPTIMAL: Self = Self(1000241002i32);
    pub const STENCIL_READ_ONLY_OPTIMAL: Self = Self(1000241003i32);
    pub const READ_ONLY_OPTIMAL: Self = Self(1000314000i32);
    pub const ATTACHMENT_OPTIMAL: Self = Self(1000314001i32);
    pub const RENDERING_LOCAL_READ: Self = Self(1000232000i32);
    pub const PRESENT_SRC: Self = Self(1000001002i32);
    pub const VIDEO_DECODE_DST: Self = Self(1000024000i32);
    pub const VIDEO_DECODE_SRC: Self = Self(1000024001i32);
    pub const VIDEO_DECODE_DPB: Self = Self(1000024002i32);
    pub const SHARED_PRESENT: Self = Self(1000111000i32);
    pub const SHADING_RATE_OPTIMAL: Self = Self::FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL;
    pub const FRAGMENT_DENSITY_MAP_OPTIMAL: Self = Self(1000218000i32);
    pub const FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL: Self = Self(1000164003i32);
    pub const VIDEO_ENCODE_DST: Self = Self(1000299000i32);
    pub const VIDEO_ENCODE_SRC: Self = Self(1000299001i32);
    pub const VIDEO_ENCODE_DPB: Self = Self(1000299002i32);
    pub const ATTACHMENT_FEEDBACK_LOOP_OPTIMAL: Self = Self(1000339000i32);
    pub const TENSOR_ALIASING: Self = Self(1000460000i32);
    pub const VIDEO_ENCODE_QUANTIZATION_MAP: Self = Self(1000553000i32);
    pub const ZERO_INITIALIZED: Self = Self(1000620000i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ImageLayout {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("UNDEFINED"),
            1i32 => f.write_str("GENERAL"),
            2i32 => f.write_str("COLOR_ATTACHMENT_OPTIMAL"),
            3i32 => f.write_str("DEPTH_STENCIL_ATTACHMENT_OPTIMAL"),
            4i32 => f.write_str("DEPTH_STENCIL_READ_ONLY_OPTIMAL"),
            5i32 => f.write_str("SHADER_READ_ONLY_OPTIMAL"),
            6i32 => f.write_str("TRANSFER_SRC_OPTIMAL"),
            7i32 => f.write_str("TRANSFER_DST_OPTIMAL"),
            8i32 => f.write_str("PREINITIALIZED"),
            1000117000i32 => f.write_str("DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL"),
            1000117001i32 => f.write_str("DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL"),
            1000241000i32 => f.write_str("DEPTH_ATTACHMENT_OPTIMAL"),
            1000241001i32 => f.write_str("DEPTH_READ_ONLY_OPTIMAL"),
            1000241002i32 => f.write_str("STENCIL_ATTACHMENT_OPTIMAL"),
            1000241003i32 => f.write_str("STENCIL_READ_ONLY_OPTIMAL"),
            1000314000i32 => f.write_str("READ_ONLY_OPTIMAL"),
            1000314001i32 => f.write_str("ATTACHMENT_OPTIMAL"),
            1000232000i32 => f.write_str("RENDERING_LOCAL_READ"),
            1000001002i32 => f.write_str("PRESENT_SRC"),
            1000024000i32 => f.write_str("VIDEO_DECODE_DST"),
            1000024001i32 => f.write_str("VIDEO_DECODE_SRC"),
            1000024002i32 => f.write_str("VIDEO_DECODE_DPB"),
            1000111000i32 => f.write_str("SHARED_PRESENT"),
            1000218000i32 => f.write_str("FRAGMENT_DENSITY_MAP_OPTIMAL"),
            1000164003i32 => f.write_str("FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL"),
            1000299000i32 => f.write_str("VIDEO_ENCODE_DST"),
            1000299001i32 => f.write_str("VIDEO_ENCODE_SRC"),
            1000299002i32 => f.write_str("VIDEO_ENCODE_DPB"),
            1000339000i32 => f.write_str("ATTACHMENT_FEEDBACK_LOOP_OPTIMAL"),
            1000460000i32 => f.write_str("TENSOR_ALIASING"),
            1000553000i32 => f.write_str("VIDEO_ENCODE_QUANTIZATION_MAP"),
            1000620000i32 => f.write_str("ZERO_INITIALIZED"),
            other => write!(f, "{}({})", stringify!(ImageLayout), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkMemoryOverallocationBehaviorAMD")]
pub struct MemoryOverallocationBehaviorAMD(i32);
impl MemoryOverallocationBehaviorAMD {
    pub const DEFAULT: Self = Self(0i32);
    pub const ALLOWED: Self = Self(1i32);
    pub const DISALLOWED: Self = Self(2i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for MemoryOverallocationBehaviorAMD {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("DEFAULT"),
            1i32 => f.write_str("ALLOWED"),
            2i32 => f.write_str("DISALLOWED"),
            other => {
                write!(f, "{}({})", stringify!(MemoryOverallocationBehaviorAMD), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkClusterAccelerationStructureTypeNV")]
pub struct ClusterAccelerationStructureTypeNV(i32);
impl ClusterAccelerationStructureTypeNV {
    pub const CLUSTERS_BOTTOM_LEVEL: Self = Self(0i32);
    pub const TRIANGLE_CLUSTER: Self = Self(1i32);
    pub const TRIANGLE_CLUSTER_TEMPLATE: Self = Self(2i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ClusterAccelerationStructureTypeNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("CLUSTERS_BOTTOM_LEVEL"),
            1i32 => f.write_str("TRIANGLE_CLUSTER"),
            2i32 => f.write_str("TRIANGLE_CLUSTER_TEMPLATE"),
            other => {
                write!(
                    f, "{}({})", stringify!(ClusterAccelerationStructureTypeNV), other
                )
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkFragmentShadingRateTypeNV")]
pub struct FragmentShadingRateTypeNV(i32);
impl FragmentShadingRateTypeNV {
    pub const FRAGMENT_SIZE: Self = Self(0i32);
    pub const ENUMS: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for FragmentShadingRateTypeNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("FRAGMENT_SIZE"),
            1i32 => f.write_str("ENUMS"),
            other => write!(f, "{}({})", stringify!(FragmentShadingRateTypeNV), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkFragmentShadingRateNV")]
pub struct FragmentShadingRateNV(i32);
impl FragmentShadingRateNV {
    pub const _1_INVOCATION_PER_PIXEL: Self = Self(0i32);
    pub const _1_INVOCATION_PER_1X2_PIXELS: Self = Self(1i32);
    pub const _1_INVOCATION_PER_2X1_PIXELS: Self = Self(4i32);
    pub const _1_INVOCATION_PER_2X2_PIXELS: Self = Self(5i32);
    pub const _1_INVOCATION_PER_2X4_PIXELS: Self = Self(6i32);
    pub const _1_INVOCATION_PER_4X2_PIXELS: Self = Self(9i32);
    pub const _1_INVOCATION_PER_4X4_PIXELS: Self = Self(10i32);
    pub const _2_INVOCATIONS_PER_PIXEL: Self = Self(11i32);
    pub const _4_INVOCATIONS_PER_PIXEL: Self = Self(12i32);
    pub const _8_INVOCATIONS_PER_PIXEL: Self = Self(13i32);
    pub const _16_INVOCATIONS_PER_PIXEL: Self = Self(14i32);
    pub const NO_INVOCATIONS: Self = Self(15i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for FragmentShadingRateNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("_1_INVOCATION_PER_PIXEL"),
            1i32 => f.write_str("_1_INVOCATION_PER_1X2_PIXELS"),
            4i32 => f.write_str("_1_INVOCATION_PER_2X1_PIXELS"),
            5i32 => f.write_str("_1_INVOCATION_PER_2X2_PIXELS"),
            6i32 => f.write_str("_1_INVOCATION_PER_2X4_PIXELS"),
            9i32 => f.write_str("_1_INVOCATION_PER_4X2_PIXELS"),
            10i32 => f.write_str("_1_INVOCATION_PER_4X4_PIXELS"),
            11i32 => f.write_str("_2_INVOCATIONS_PER_PIXEL"),
            12i32 => f.write_str("_4_INVOCATIONS_PER_PIXEL"),
            13i32 => f.write_str("_8_INVOCATIONS_PER_PIXEL"),
            14i32 => f.write_str("_16_INVOCATIONS_PER_PIXEL"),
            15i32 => f.write_str("NO_INVOCATIONS"),
            other => write!(f, "{}({})", stringify!(FragmentShadingRateNV), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDisplacementMicromapFormatNV")]
pub struct DisplacementMicromapFormatNV(i32);
impl DisplacementMicromapFormatNV {
    pub const _64_TRIANGLES_64_BYTES: Self = Self(1i32);
    pub const _256_TRIANGLES_128_BYTES: Self = Self(2i32);
    pub const _1024_TRIANGLES_128_BYTES: Self = Self(3i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DisplacementMicromapFormatNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            1i32 => f.write_str("_64_TRIANGLES_64_BYTES"),
            2i32 => f.write_str("_256_TRIANGLES_128_BYTES"),
            3i32 => f.write_str("_1024_TRIANGLES_128_BYTES"),
            other => write!(f, "{}({})", stringify!(DisplacementMicromapFormatNV), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkIndexType")]
pub struct IndexType(i32);
impl IndexType {
    pub const UINT16: Self = Self(0i32);
    pub const UINT32: Self = Self(1i32);
    pub const UINT8: Self = Self(1000265000i32);
    pub const NONE: Self = Self(1000165000i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for IndexType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("UINT16"),
            1i32 => f.write_str("UINT32"),
            1000265000i32 => f.write_str("UINT8"),
            1000165000i32 => f.write_str("NONE"),
            other => write!(f, "{}({})", stringify!(IndexType), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPrimitiveTopology")]
pub struct PrimitiveTopology(i32);
impl PrimitiveTopology {
    pub const POINT_LIST: Self = Self(0i32);
    pub const LINE_LIST: Self = Self(1i32);
    pub const LINE_STRIP: Self = Self(2i32);
    pub const TRIANGLE_LIST: Self = Self(3i32);
    pub const TRIANGLE_STRIP: Self = Self(4i32);
    pub const TRIANGLE_FAN: Self = Self(5i32);
    pub const LINE_LIST_WITH_ADJACENCY: Self = Self(6i32);
    pub const LINE_STRIP_WITH_ADJACENCY: Self = Self(7i32);
    pub const TRIANGLE_LIST_WITH_ADJACENCY: Self = Self(8i32);
    pub const TRIANGLE_STRIP_WITH_ADJACENCY: Self = Self(9i32);
    pub const PATCH_LIST: Self = Self(10i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PrimitiveTopology {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("POINT_LIST"),
            1i32 => f.write_str("LINE_LIST"),
            2i32 => f.write_str("LINE_STRIP"),
            3i32 => f.write_str("TRIANGLE_LIST"),
            4i32 => f.write_str("TRIANGLE_STRIP"),
            5i32 => f.write_str("TRIANGLE_FAN"),
            6i32 => f.write_str("LINE_LIST_WITH_ADJACENCY"),
            7i32 => f.write_str("LINE_STRIP_WITH_ADJACENCY"),
            8i32 => f.write_str("TRIANGLE_LIST_WITH_ADJACENCY"),
            9i32 => f.write_str("TRIANGLE_STRIP_WITH_ADJACENCY"),
            10i32 => f.write_str("PATCH_LIST"),
            other => write!(f, "{}({})", stringify!(PrimitiveTopology), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkRayTracingLssPrimitiveEndCapsModeNV")]
pub struct RayTracingLssPrimitiveEndCapsModeNV(i32);
impl RayTracingLssPrimitiveEndCapsModeNV {
    pub const NONE: Self = Self(0i32);
    pub const CHAINED: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for RayTracingLssPrimitiveEndCapsModeNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("NONE"),
            1i32 => f.write_str("CHAINED"),
            other => {
                write!(
                    f, "{}({})", stringify!(RayTracingLssPrimitiveEndCapsModeNV), other
                )
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkMicromapTypeEXT")]
pub struct MicromapTypeEXT(i32);
impl MicromapTypeEXT {
    pub const OPACITY_MICROMAP: Self = Self(0i32);
    pub const DISPLACEMENT_MICROMAP: Self = Self(1000397000i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for MicromapTypeEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("OPACITY_MICROMAP"),
            1000397000i32 => f.write_str("DISPLACEMENT_MICROMAP"),
            other => write!(f, "{}({})", stringify!(MicromapTypeEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkFormat")]
pub struct Format(i32);
impl Format {
    pub const UNDEFINED: Self = Self(0i32);
    pub const R4G4_UNORM_PACK8: Self = Self(1i32);
    pub const R4G4B4A4_UNORM_PACK16: Self = Self(2i32);
    pub const B4G4R4A4_UNORM_PACK16: Self = Self(3i32);
    pub const R5G6B5_UNORM_PACK16: Self = Self(4i32);
    pub const B5G6R5_UNORM_PACK16: Self = Self(5i32);
    pub const R5G5B5A1_UNORM_PACK16: Self = Self(6i32);
    pub const B5G5R5A1_UNORM_PACK16: Self = Self(7i32);
    pub const A1R5G5B5_UNORM_PACK16: Self = Self(8i32);
    pub const R8_UNORM: Self = Self(9i32);
    pub const R8_SNORM: Self = Self(10i32);
    pub const R8_USCALED: Self = Self(11i32);
    pub const R8_SSCALED: Self = Self(12i32);
    pub const R8_UINT: Self = Self(13i32);
    pub const R8_SINT: Self = Self(14i32);
    pub const R8_SRGB: Self = Self(15i32);
    pub const R8G8_UNORM: Self = Self(16i32);
    pub const R8G8_SNORM: Self = Self(17i32);
    pub const R8G8_USCALED: Self = Self(18i32);
    pub const R8G8_SSCALED: Self = Self(19i32);
    pub const R8G8_UINT: Self = Self(20i32);
    pub const R8G8_SINT: Self = Self(21i32);
    pub const R8G8_SRGB: Self = Self(22i32);
    pub const R8G8B8_UNORM: Self = Self(23i32);
    pub const R8G8B8_SNORM: Self = Self(24i32);
    pub const R8G8B8_USCALED: Self = Self(25i32);
    pub const R8G8B8_SSCALED: Self = Self(26i32);
    pub const R8G8B8_UINT: Self = Self(27i32);
    pub const R8G8B8_SINT: Self = Self(28i32);
    pub const R8G8B8_SRGB: Self = Self(29i32);
    pub const B8G8R8_UNORM: Self = Self(30i32);
    pub const B8G8R8_SNORM: Self = Self(31i32);
    pub const B8G8R8_USCALED: Self = Self(32i32);
    pub const B8G8R8_SSCALED: Self = Self(33i32);
    pub const B8G8R8_UINT: Self = Self(34i32);
    pub const B8G8R8_SINT: Self = Self(35i32);
    pub const B8G8R8_SRGB: Self = Self(36i32);
    pub const R8G8B8A8_UNORM: Self = Self(37i32);
    pub const R8G8B8A8_SNORM: Self = Self(38i32);
    pub const R8G8B8A8_USCALED: Self = Self(39i32);
    pub const R8G8B8A8_SSCALED: Self = Self(40i32);
    pub const R8G8B8A8_UINT: Self = Self(41i32);
    pub const R8G8B8A8_SINT: Self = Self(42i32);
    pub const R8G8B8A8_SRGB: Self = Self(43i32);
    pub const B8G8R8A8_UNORM: Self = Self(44i32);
    pub const B8G8R8A8_SNORM: Self = Self(45i32);
    pub const B8G8R8A8_USCALED: Self = Self(46i32);
    pub const B8G8R8A8_SSCALED: Self = Self(47i32);
    pub const B8G8R8A8_UINT: Self = Self(48i32);
    pub const B8G8R8A8_SINT: Self = Self(49i32);
    pub const B8G8R8A8_SRGB: Self = Self(50i32);
    pub const A8B8G8R8_UNORM_PACK32: Self = Self(51i32);
    pub const A8B8G8R8_SNORM_PACK32: Self = Self(52i32);
    pub const A8B8G8R8_USCALED_PACK32: Self = Self(53i32);
    pub const A8B8G8R8_SSCALED_PACK32: Self = Self(54i32);
    pub const A8B8G8R8_UINT_PACK32: Self = Self(55i32);
    pub const A8B8G8R8_SINT_PACK32: Self = Self(56i32);
    pub const A8B8G8R8_SRGB_PACK32: Self = Self(57i32);
    pub const A2R10G10B10_UNORM_PACK32: Self = Self(58i32);
    pub const A2R10G10B10_SNORM_PACK32: Self = Self(59i32);
    pub const A2R10G10B10_USCALED_PACK32: Self = Self(60i32);
    pub const A2R10G10B10_SSCALED_PACK32: Self = Self(61i32);
    pub const A2R10G10B10_UINT_PACK32: Self = Self(62i32);
    pub const A2R10G10B10_SINT_PACK32: Self = Self(63i32);
    pub const A2B10G10R10_UNORM_PACK32: Self = Self(64i32);
    pub const A2B10G10R10_SNORM_PACK32: Self = Self(65i32);
    pub const A2B10G10R10_USCALED_PACK32: Self = Self(66i32);
    pub const A2B10G10R10_SSCALED_PACK32: Self = Self(67i32);
    pub const A2B10G10R10_UINT_PACK32: Self = Self(68i32);
    pub const A2B10G10R10_SINT_PACK32: Self = Self(69i32);
    pub const R16_UNORM: Self = Self(70i32);
    pub const R16_SNORM: Self = Self(71i32);
    pub const R16_USCALED: Self = Self(72i32);
    pub const R16_SSCALED: Self = Self(73i32);
    pub const R16_UINT: Self = Self(74i32);
    pub const R16_SINT: Self = Self(75i32);
    pub const R16_SFLOAT: Self = Self(76i32);
    pub const R16G16_UNORM: Self = Self(77i32);
    pub const R16G16_SNORM: Self = Self(78i32);
    pub const R16G16_USCALED: Self = Self(79i32);
    pub const R16G16_SSCALED: Self = Self(80i32);
    pub const R16G16_UINT: Self = Self(81i32);
    pub const R16G16_SINT: Self = Self(82i32);
    pub const R16G16_SFLOAT: Self = Self(83i32);
    pub const R16G16B16_UNORM: Self = Self(84i32);
    pub const R16G16B16_SNORM: Self = Self(85i32);
    pub const R16G16B16_USCALED: Self = Self(86i32);
    pub const R16G16B16_SSCALED: Self = Self(87i32);
    pub const R16G16B16_UINT: Self = Self(88i32);
    pub const R16G16B16_SINT: Self = Self(89i32);
    pub const R16G16B16_SFLOAT: Self = Self(90i32);
    pub const R16G16B16A16_UNORM: Self = Self(91i32);
    pub const R16G16B16A16_SNORM: Self = Self(92i32);
    pub const R16G16B16A16_USCALED: Self = Self(93i32);
    pub const R16G16B16A16_SSCALED: Self = Self(94i32);
    pub const R16G16B16A16_UINT: Self = Self(95i32);
    pub const R16G16B16A16_SINT: Self = Self(96i32);
    pub const R16G16B16A16_SFLOAT: Self = Self(97i32);
    pub const R32_UINT: Self = Self(98i32);
    pub const R32_SINT: Self = Self(99i32);
    pub const R32_SFLOAT: Self = Self(100i32);
    pub const R32G32_UINT: Self = Self(101i32);
    pub const R32G32_SINT: Self = Self(102i32);
    pub const R32G32_SFLOAT: Self = Self(103i32);
    pub const R32G32B32_UINT: Self = Self(104i32);
    pub const R32G32B32_SINT: Self = Self(105i32);
    pub const R32G32B32_SFLOAT: Self = Self(106i32);
    pub const R32G32B32A32_UINT: Self = Self(107i32);
    pub const R32G32B32A32_SINT: Self = Self(108i32);
    pub const R32G32B32A32_SFLOAT: Self = Self(109i32);
    pub const R64_UINT: Self = Self(110i32);
    pub const R64_SINT: Self = Self(111i32);
    pub const R64_SFLOAT: Self = Self(112i32);
    pub const R64G64_UINT: Self = Self(113i32);
    pub const R64G64_SINT: Self = Self(114i32);
    pub const R64G64_SFLOAT: Self = Self(115i32);
    pub const R64G64B64_UINT: Self = Self(116i32);
    pub const R64G64B64_SINT: Self = Self(117i32);
    pub const R64G64B64_SFLOAT: Self = Self(118i32);
    pub const R64G64B64A64_UINT: Self = Self(119i32);
    pub const R64G64B64A64_SINT: Self = Self(120i32);
    pub const R64G64B64A64_SFLOAT: Self = Self(121i32);
    pub const B10G11R11_UFLOAT_PACK32: Self = Self(122i32);
    pub const E5B9G9R9_UFLOAT_PACK32: Self = Self(123i32);
    pub const D16_UNORM: Self = Self(124i32);
    pub const X8_D24_UNORM_PACK32: Self = Self(125i32);
    pub const D32_SFLOAT: Self = Self(126i32);
    pub const S8_UINT: Self = Self(127i32);
    pub const D16_UNORM_S8_UINT: Self = Self(128i32);
    pub const D24_UNORM_S8_UINT: Self = Self(129i32);
    pub const D32_SFLOAT_S8_UINT: Self = Self(130i32);
    pub const BC1_RGB_UNORM_BLOCK: Self = Self(131i32);
    pub const BC1_RGB_SRGB_BLOCK: Self = Self(132i32);
    pub const BC1_RGBA_UNORM_BLOCK: Self = Self(133i32);
    pub const BC1_RGBA_SRGB_BLOCK: Self = Self(134i32);
    pub const BC2_UNORM_BLOCK: Self = Self(135i32);
    pub const BC2_SRGB_BLOCK: Self = Self(136i32);
    pub const BC3_UNORM_BLOCK: Self = Self(137i32);
    pub const BC3_SRGB_BLOCK: Self = Self(138i32);
    pub const BC4_UNORM_BLOCK: Self = Self(139i32);
    pub const BC4_SNORM_BLOCK: Self = Self(140i32);
    pub const BC5_UNORM_BLOCK: Self = Self(141i32);
    pub const BC5_SNORM_BLOCK: Self = Self(142i32);
    pub const BC6H_UFLOAT_BLOCK: Self = Self(143i32);
    pub const BC6H_SFLOAT_BLOCK: Self = Self(144i32);
    pub const BC7_UNORM_BLOCK: Self = Self(145i32);
    pub const BC7_SRGB_BLOCK: Self = Self(146i32);
    pub const ETC2_R8G8B8_UNORM_BLOCK: Self = Self(147i32);
    pub const ETC2_R8G8B8_SRGB_BLOCK: Self = Self(148i32);
    pub const ETC2_R8G8B8A1_UNORM_BLOCK: Self = Self(149i32);
    pub const ETC2_R8G8B8A1_SRGB_BLOCK: Self = Self(150i32);
    pub const ETC2_R8G8B8A8_UNORM_BLOCK: Self = Self(151i32);
    pub const ETC2_R8G8B8A8_SRGB_BLOCK: Self = Self(152i32);
    pub const EAC_R11_UNORM_BLOCK: Self = Self(153i32);
    pub const EAC_R11_SNORM_BLOCK: Self = Self(154i32);
    pub const EAC_R11G11_UNORM_BLOCK: Self = Self(155i32);
    pub const EAC_R11G11_SNORM_BLOCK: Self = Self(156i32);
    pub const ASTC_4x4_UNORM_BLOCK: Self = Self(157i32);
    pub const ASTC_4x4_SRGB_BLOCK: Self = Self(158i32);
    pub const ASTC_5x4_UNORM_BLOCK: Self = Self(159i32);
    pub const ASTC_5x4_SRGB_BLOCK: Self = Self(160i32);
    pub const ASTC_5x5_UNORM_BLOCK: Self = Self(161i32);
    pub const ASTC_5x5_SRGB_BLOCK: Self = Self(162i32);
    pub const ASTC_6x5_UNORM_BLOCK: Self = Self(163i32);
    pub const ASTC_6x5_SRGB_BLOCK: Self = Self(164i32);
    pub const ASTC_6x6_UNORM_BLOCK: Self = Self(165i32);
    pub const ASTC_6x6_SRGB_BLOCK: Self = Self(166i32);
    pub const ASTC_8x5_UNORM_BLOCK: Self = Self(167i32);
    pub const ASTC_8x5_SRGB_BLOCK: Self = Self(168i32);
    pub const ASTC_8x6_UNORM_BLOCK: Self = Self(169i32);
    pub const ASTC_8x6_SRGB_BLOCK: Self = Self(170i32);
    pub const ASTC_8x8_UNORM_BLOCK: Self = Self(171i32);
    pub const ASTC_8x8_SRGB_BLOCK: Self = Self(172i32);
    pub const ASTC_10x5_UNORM_BLOCK: Self = Self(173i32);
    pub const ASTC_10x5_SRGB_BLOCK: Self = Self(174i32);
    pub const ASTC_10x6_UNORM_BLOCK: Self = Self(175i32);
    pub const ASTC_10x6_SRGB_BLOCK: Self = Self(176i32);
    pub const ASTC_10x8_UNORM_BLOCK: Self = Self(177i32);
    pub const ASTC_10x8_SRGB_BLOCK: Self = Self(178i32);
    pub const ASTC_10x10_UNORM_BLOCK: Self = Self(179i32);
    pub const ASTC_10x10_SRGB_BLOCK: Self = Self(180i32);
    pub const ASTC_12x10_UNORM_BLOCK: Self = Self(181i32);
    pub const ASTC_12x10_SRGB_BLOCK: Self = Self(182i32);
    pub const ASTC_12x12_UNORM_BLOCK: Self = Self(183i32);
    pub const ASTC_12x12_SRGB_BLOCK: Self = Self(184i32);
    pub const G8B8G8R8_422_UNORM: Self = Self(1000156000i32);
    pub const B8G8R8G8_422_UNORM: Self = Self(1000156001i32);
    pub const G8_B8_R8_3PLANE_420_UNORM: Self = Self(1000156002i32);
    pub const G8_B8R8_2PLANE_420_UNORM: Self = Self(1000156003i32);
    pub const G8_B8_R8_3PLANE_422_UNORM: Self = Self(1000156004i32);
    pub const G8_B8R8_2PLANE_422_UNORM: Self = Self(1000156005i32);
    pub const G8_B8_R8_3PLANE_444_UNORM: Self = Self(1000156006i32);
    pub const R10X6_UNORM_PACK16: Self = Self(1000156007i32);
    pub const R10X6G10X6_UNORM_2PACK16: Self = Self(1000156008i32);
    pub const R10X6G10X6B10X6A10X6_UNORM_4PACK16: Self = Self(1000156009i32);
    pub const G10X6B10X6G10X6R10X6_422_UNORM_4PACK16: Self = Self(1000156010i32);
    pub const B10X6G10X6R10X6G10X6_422_UNORM_4PACK16: Self = Self(1000156011i32);
    pub const G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16: Self = Self(1000156012i32);
    pub const G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16: Self = Self(1000156013i32);
    pub const G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16: Self = Self(1000156014i32);
    pub const G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16: Self = Self(1000156015i32);
    pub const G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16: Self = Self(1000156016i32);
    pub const R12X4_UNORM_PACK16: Self = Self(1000156017i32);
    pub const R12X4G12X4_UNORM_2PACK16: Self = Self(1000156018i32);
    pub const R12X4G12X4B12X4A12X4_UNORM_4PACK16: Self = Self(1000156019i32);
    pub const G12X4B12X4G12X4R12X4_422_UNORM_4PACK16: Self = Self(1000156020i32);
    pub const B12X4G12X4R12X4G12X4_422_UNORM_4PACK16: Self = Self(1000156021i32);
    pub const G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16: Self = Self(1000156022i32);
    pub const G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16: Self = Self(1000156023i32);
    pub const G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16: Self = Self(1000156024i32);
    pub const G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16: Self = Self(1000156025i32);
    pub const G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16: Self = Self(1000156026i32);
    pub const G16B16G16R16_422_UNORM: Self = Self(1000156027i32);
    pub const B16G16R16G16_422_UNORM: Self = Self(1000156028i32);
    pub const G16_B16_R16_3PLANE_420_UNORM: Self = Self(1000156029i32);
    pub const G16_B16R16_2PLANE_420_UNORM: Self = Self(1000156030i32);
    pub const G16_B16_R16_3PLANE_422_UNORM: Self = Self(1000156031i32);
    pub const G16_B16R16_2PLANE_422_UNORM: Self = Self(1000156032i32);
    pub const G16_B16_R16_3PLANE_444_UNORM: Self = Self(1000156033i32);
    pub const G8_B8R8_2PLANE_444_UNORM: Self = Self(1000330000i32);
    pub const G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16: Self = Self(1000330001i32);
    pub const G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16: Self = Self(1000330002i32);
    pub const G16_B16R16_2PLANE_444_UNORM: Self = Self(1000330003i32);
    pub const A4R4G4B4_UNORM_PACK16: Self = Self(1000340000i32);
    pub const A4B4G4R4_UNORM_PACK16: Self = Self(1000340001i32);
    pub const ASTC_4x4_SFLOAT_BLOCK: Self = Self(1000066000i32);
    pub const ASTC_5x4_SFLOAT_BLOCK: Self = Self(1000066001i32);
    pub const ASTC_5x5_SFLOAT_BLOCK: Self = Self(1000066002i32);
    pub const ASTC_6x5_SFLOAT_BLOCK: Self = Self(1000066003i32);
    pub const ASTC_6x6_SFLOAT_BLOCK: Self = Self(1000066004i32);
    pub const ASTC_8x5_SFLOAT_BLOCK: Self = Self(1000066005i32);
    pub const ASTC_8x6_SFLOAT_BLOCK: Self = Self(1000066006i32);
    pub const ASTC_8x8_SFLOAT_BLOCK: Self = Self(1000066007i32);
    pub const ASTC_10x5_SFLOAT_BLOCK: Self = Self(1000066008i32);
    pub const ASTC_10x6_SFLOAT_BLOCK: Self = Self(1000066009i32);
    pub const ASTC_10x8_SFLOAT_BLOCK: Self = Self(1000066010i32);
    pub const ASTC_10x10_SFLOAT_BLOCK: Self = Self(1000066011i32);
    pub const ASTC_12x10_SFLOAT_BLOCK: Self = Self(1000066012i32);
    pub const ASTC_12x12_SFLOAT_BLOCK: Self = Self(1000066013i32);
    pub const A1B5G5R5_UNORM_PACK16: Self = Self(1000470000i32);
    pub const A8_UNORM: Self = Self(1000470001i32);
    pub const PVRTC1_2BPP_UNORM_BLOCK: Self = Self(1000054000i32);
    pub const PVRTC1_4BPP_UNORM_BLOCK: Self = Self(1000054001i32);
    pub const PVRTC2_2BPP_UNORM_BLOCK: Self = Self(1000054002i32);
    pub const PVRTC2_4BPP_UNORM_BLOCK: Self = Self(1000054003i32);
    pub const PVRTC1_2BPP_SRGB_BLOCK: Self = Self(1000054004i32);
    pub const PVRTC1_4BPP_SRGB_BLOCK: Self = Self(1000054005i32);
    pub const PVRTC2_2BPP_SRGB_BLOCK: Self = Self(1000054006i32);
    pub const PVRTC2_4BPP_SRGB_BLOCK: Self = Self(1000054007i32);
    pub const ASTC_3x3x3_UNORM_BLOCK: Self = Self(1000288000i32);
    pub const ASTC_3x3x3_SRGB_BLOCK: Self = Self(1000288001i32);
    pub const ASTC_3x3x3_SFLOAT_BLOCK: Self = Self(1000288002i32);
    pub const ASTC_4x3x3_UNORM_BLOCK: Self = Self(1000288003i32);
    pub const ASTC_4x3x3_SRGB_BLOCK: Self = Self(1000288004i32);
    pub const ASTC_4x3x3_SFLOAT_BLOCK: Self = Self(1000288005i32);
    pub const ASTC_4x4x3_UNORM_BLOCK: Self = Self(1000288006i32);
    pub const ASTC_4x4x3_SRGB_BLOCK: Self = Self(1000288007i32);
    pub const ASTC_4x4x3_SFLOAT_BLOCK: Self = Self(1000288008i32);
    pub const ASTC_4x4x4_UNORM_BLOCK: Self = Self(1000288009i32);
    pub const ASTC_4x4x4_SRGB_BLOCK: Self = Self(1000288010i32);
    pub const ASTC_4x4x4_SFLOAT_BLOCK: Self = Self(1000288011i32);
    pub const ASTC_5x4x4_UNORM_BLOCK: Self = Self(1000288012i32);
    pub const ASTC_5x4x4_SRGB_BLOCK: Self = Self(1000288013i32);
    pub const ASTC_5x4x4_SFLOAT_BLOCK: Self = Self(1000288014i32);
    pub const ASTC_5x5x4_UNORM_BLOCK: Self = Self(1000288015i32);
    pub const ASTC_5x5x4_SRGB_BLOCK: Self = Self(1000288016i32);
    pub const ASTC_5x5x4_SFLOAT_BLOCK: Self = Self(1000288017i32);
    pub const ASTC_5x5x5_UNORM_BLOCK: Self = Self(1000288018i32);
    pub const ASTC_5x5x5_SRGB_BLOCK: Self = Self(1000288019i32);
    pub const ASTC_5x5x5_SFLOAT_BLOCK: Self = Self(1000288020i32);
    pub const ASTC_6x5x5_UNORM_BLOCK: Self = Self(1000288021i32);
    pub const ASTC_6x5x5_SRGB_BLOCK: Self = Self(1000288022i32);
    pub const ASTC_6x5x5_SFLOAT_BLOCK: Self = Self(1000288023i32);
    pub const ASTC_6x6x5_UNORM_BLOCK: Self = Self(1000288024i32);
    pub const ASTC_6x6x5_SRGB_BLOCK: Self = Self(1000288025i32);
    pub const ASTC_6x6x5_SFLOAT_BLOCK: Self = Self(1000288026i32);
    pub const ASTC_6x6x6_UNORM_BLOCK: Self = Self(1000288027i32);
    pub const ASTC_6x6x6_SRGB_BLOCK: Self = Self(1000288028i32);
    pub const ASTC_6x6x6_SFLOAT_BLOCK: Self = Self(1000288029i32);
    pub const R8_BOOL: Self = Self(1000460000i32);
    pub const R16_SFLOAT_FPENCODING_BFLOAT16: Self = Self(1000460001i32);
    pub const R8_SFLOAT_FPENCODING_FLOAT8E4M3: Self = Self(1000460002i32);
    pub const R8_SFLOAT_FPENCODING_FLOAT8E5M2: Self = Self(1000460003i32);
    pub const R16G16_SFIXED5: Self = Self(1000464000i32);
    pub const R16G16_S10_5: Self = Self::R16G16_SFIXED5;
    pub const R10X6_UINT_PACK16: Self = Self(1000609000i32);
    pub const R10X6G10X6_UINT_2PACK16: Self = Self(1000609001i32);
    pub const R10X6G10X6B10X6A10X6_UINT_4PACK16: Self = Self(1000609002i32);
    pub const R12X4_UINT_PACK16: Self = Self(1000609003i32);
    pub const R12X4G12X4_UINT_2PACK16: Self = Self(1000609004i32);
    pub const R12X4G12X4B12X4A12X4_UINT_4PACK16: Self = Self(1000609005i32);
    pub const R14X2_UINT_PACK16: Self = Self(1000609006i32);
    pub const R14X2G14X2_UINT_2PACK16: Self = Self(1000609007i32);
    pub const R14X2G14X2B14X2A14X2_UINT_4PACK16: Self = Self(1000609008i32);
    pub const R14X2_UNORM_PACK16: Self = Self(1000609009i32);
    pub const R14X2G14X2_UNORM_2PACK16: Self = Self(1000609010i32);
    pub const R14X2G14X2B14X2A14X2_UNORM_4PACK16: Self = Self(1000609011i32);
    pub const G14X2_B14X2R14X2_2PLANE_420_UNORM_3PACK16: Self = Self(1000609012i32);
    pub const G14X2_B14X2R14X2_2PLANE_422_UNORM_3PACK16: Self = Self(1000609013i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for Format {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("UNDEFINED"),
            1i32 => f.write_str("R4G4_UNORM_PACK8"),
            2i32 => f.write_str("R4G4B4A4_UNORM_PACK16"),
            3i32 => f.write_str("B4G4R4A4_UNORM_PACK16"),
            4i32 => f.write_str("R5G6B5_UNORM_PACK16"),
            5i32 => f.write_str("B5G6R5_UNORM_PACK16"),
            6i32 => f.write_str("R5G5B5A1_UNORM_PACK16"),
            7i32 => f.write_str("B5G5R5A1_UNORM_PACK16"),
            8i32 => f.write_str("A1R5G5B5_UNORM_PACK16"),
            9i32 => f.write_str("R8_UNORM"),
            10i32 => f.write_str("R8_SNORM"),
            11i32 => f.write_str("R8_USCALED"),
            12i32 => f.write_str("R8_SSCALED"),
            13i32 => f.write_str("R8_UINT"),
            14i32 => f.write_str("R8_SINT"),
            15i32 => f.write_str("R8_SRGB"),
            16i32 => f.write_str("R8G8_UNORM"),
            17i32 => f.write_str("R8G8_SNORM"),
            18i32 => f.write_str("R8G8_USCALED"),
            19i32 => f.write_str("R8G8_SSCALED"),
            20i32 => f.write_str("R8G8_UINT"),
            21i32 => f.write_str("R8G8_SINT"),
            22i32 => f.write_str("R8G8_SRGB"),
            23i32 => f.write_str("R8G8B8_UNORM"),
            24i32 => f.write_str("R8G8B8_SNORM"),
            25i32 => f.write_str("R8G8B8_USCALED"),
            26i32 => f.write_str("R8G8B8_SSCALED"),
            27i32 => f.write_str("R8G8B8_UINT"),
            28i32 => f.write_str("R8G8B8_SINT"),
            29i32 => f.write_str("R8G8B8_SRGB"),
            30i32 => f.write_str("B8G8R8_UNORM"),
            31i32 => f.write_str("B8G8R8_SNORM"),
            32i32 => f.write_str("B8G8R8_USCALED"),
            33i32 => f.write_str("B8G8R8_SSCALED"),
            34i32 => f.write_str("B8G8R8_UINT"),
            35i32 => f.write_str("B8G8R8_SINT"),
            36i32 => f.write_str("B8G8R8_SRGB"),
            37i32 => f.write_str("R8G8B8A8_UNORM"),
            38i32 => f.write_str("R8G8B8A8_SNORM"),
            39i32 => f.write_str("R8G8B8A8_USCALED"),
            40i32 => f.write_str("R8G8B8A8_SSCALED"),
            41i32 => f.write_str("R8G8B8A8_UINT"),
            42i32 => f.write_str("R8G8B8A8_SINT"),
            43i32 => f.write_str("R8G8B8A8_SRGB"),
            44i32 => f.write_str("B8G8R8A8_UNORM"),
            45i32 => f.write_str("B8G8R8A8_SNORM"),
            46i32 => f.write_str("B8G8R8A8_USCALED"),
            47i32 => f.write_str("B8G8R8A8_SSCALED"),
            48i32 => f.write_str("B8G8R8A8_UINT"),
            49i32 => f.write_str("B8G8R8A8_SINT"),
            50i32 => f.write_str("B8G8R8A8_SRGB"),
            51i32 => f.write_str("A8B8G8R8_UNORM_PACK32"),
            52i32 => f.write_str("A8B8G8R8_SNORM_PACK32"),
            53i32 => f.write_str("A8B8G8R8_USCALED_PACK32"),
            54i32 => f.write_str("A8B8G8R8_SSCALED_PACK32"),
            55i32 => f.write_str("A8B8G8R8_UINT_PACK32"),
            56i32 => f.write_str("A8B8G8R8_SINT_PACK32"),
            57i32 => f.write_str("A8B8G8R8_SRGB_PACK32"),
            58i32 => f.write_str("A2R10G10B10_UNORM_PACK32"),
            59i32 => f.write_str("A2R10G10B10_SNORM_PACK32"),
            60i32 => f.write_str("A2R10G10B10_USCALED_PACK32"),
            61i32 => f.write_str("A2R10G10B10_SSCALED_PACK32"),
            62i32 => f.write_str("A2R10G10B10_UINT_PACK32"),
            63i32 => f.write_str("A2R10G10B10_SINT_PACK32"),
            64i32 => f.write_str("A2B10G10R10_UNORM_PACK32"),
            65i32 => f.write_str("A2B10G10R10_SNORM_PACK32"),
            66i32 => f.write_str("A2B10G10R10_USCALED_PACK32"),
            67i32 => f.write_str("A2B10G10R10_SSCALED_PACK32"),
            68i32 => f.write_str("A2B10G10R10_UINT_PACK32"),
            69i32 => f.write_str("A2B10G10R10_SINT_PACK32"),
            70i32 => f.write_str("R16_UNORM"),
            71i32 => f.write_str("R16_SNORM"),
            72i32 => f.write_str("R16_USCALED"),
            73i32 => f.write_str("R16_SSCALED"),
            74i32 => f.write_str("R16_UINT"),
            75i32 => f.write_str("R16_SINT"),
            76i32 => f.write_str("R16_SFLOAT"),
            77i32 => f.write_str("R16G16_UNORM"),
            78i32 => f.write_str("R16G16_SNORM"),
            79i32 => f.write_str("R16G16_USCALED"),
            80i32 => f.write_str("R16G16_SSCALED"),
            81i32 => f.write_str("R16G16_UINT"),
            82i32 => f.write_str("R16G16_SINT"),
            83i32 => f.write_str("R16G16_SFLOAT"),
            84i32 => f.write_str("R16G16B16_UNORM"),
            85i32 => f.write_str("R16G16B16_SNORM"),
            86i32 => f.write_str("R16G16B16_USCALED"),
            87i32 => f.write_str("R16G16B16_SSCALED"),
            88i32 => f.write_str("R16G16B16_UINT"),
            89i32 => f.write_str("R16G16B16_SINT"),
            90i32 => f.write_str("R16G16B16_SFLOAT"),
            91i32 => f.write_str("R16G16B16A16_UNORM"),
            92i32 => f.write_str("R16G16B16A16_SNORM"),
            93i32 => f.write_str("R16G16B16A16_USCALED"),
            94i32 => f.write_str("R16G16B16A16_SSCALED"),
            95i32 => f.write_str("R16G16B16A16_UINT"),
            96i32 => f.write_str("R16G16B16A16_SINT"),
            97i32 => f.write_str("R16G16B16A16_SFLOAT"),
            98i32 => f.write_str("R32_UINT"),
            99i32 => f.write_str("R32_SINT"),
            100i32 => f.write_str("R32_SFLOAT"),
            101i32 => f.write_str("R32G32_UINT"),
            102i32 => f.write_str("R32G32_SINT"),
            103i32 => f.write_str("R32G32_SFLOAT"),
            104i32 => f.write_str("R32G32B32_UINT"),
            105i32 => f.write_str("R32G32B32_SINT"),
            106i32 => f.write_str("R32G32B32_SFLOAT"),
            107i32 => f.write_str("R32G32B32A32_UINT"),
            108i32 => f.write_str("R32G32B32A32_SINT"),
            109i32 => f.write_str("R32G32B32A32_SFLOAT"),
            110i32 => f.write_str("R64_UINT"),
            111i32 => f.write_str("R64_SINT"),
            112i32 => f.write_str("R64_SFLOAT"),
            113i32 => f.write_str("R64G64_UINT"),
            114i32 => f.write_str("R64G64_SINT"),
            115i32 => f.write_str("R64G64_SFLOAT"),
            116i32 => f.write_str("R64G64B64_UINT"),
            117i32 => f.write_str("R64G64B64_SINT"),
            118i32 => f.write_str("R64G64B64_SFLOAT"),
            119i32 => f.write_str("R64G64B64A64_UINT"),
            120i32 => f.write_str("R64G64B64A64_SINT"),
            121i32 => f.write_str("R64G64B64A64_SFLOAT"),
            122i32 => f.write_str("B10G11R11_UFLOAT_PACK32"),
            123i32 => f.write_str("E5B9G9R9_UFLOAT_PACK32"),
            124i32 => f.write_str("D16_UNORM"),
            125i32 => f.write_str("X8_D24_UNORM_PACK32"),
            126i32 => f.write_str("D32_SFLOAT"),
            127i32 => f.write_str("S8_UINT"),
            128i32 => f.write_str("D16_UNORM_S8_UINT"),
            129i32 => f.write_str("D24_UNORM_S8_UINT"),
            130i32 => f.write_str("D32_SFLOAT_S8_UINT"),
            131i32 => f.write_str("BC1_RGB_UNORM_BLOCK"),
            132i32 => f.write_str("BC1_RGB_SRGB_BLOCK"),
            133i32 => f.write_str("BC1_RGBA_UNORM_BLOCK"),
            134i32 => f.write_str("BC1_RGBA_SRGB_BLOCK"),
            135i32 => f.write_str("BC2_UNORM_BLOCK"),
            136i32 => f.write_str("BC2_SRGB_BLOCK"),
            137i32 => f.write_str("BC3_UNORM_BLOCK"),
            138i32 => f.write_str("BC3_SRGB_BLOCK"),
            139i32 => f.write_str("BC4_UNORM_BLOCK"),
            140i32 => f.write_str("BC4_SNORM_BLOCK"),
            141i32 => f.write_str("BC5_UNORM_BLOCK"),
            142i32 => f.write_str("BC5_SNORM_BLOCK"),
            143i32 => f.write_str("BC6H_UFLOAT_BLOCK"),
            144i32 => f.write_str("BC6H_SFLOAT_BLOCK"),
            145i32 => f.write_str("BC7_UNORM_BLOCK"),
            146i32 => f.write_str("BC7_SRGB_BLOCK"),
            147i32 => f.write_str("ETC2_R8G8B8_UNORM_BLOCK"),
            148i32 => f.write_str("ETC2_R8G8B8_SRGB_BLOCK"),
            149i32 => f.write_str("ETC2_R8G8B8A1_UNORM_BLOCK"),
            150i32 => f.write_str("ETC2_R8G8B8A1_SRGB_BLOCK"),
            151i32 => f.write_str("ETC2_R8G8B8A8_UNORM_BLOCK"),
            152i32 => f.write_str("ETC2_R8G8B8A8_SRGB_BLOCK"),
            153i32 => f.write_str("EAC_R11_UNORM_BLOCK"),
            154i32 => f.write_str("EAC_R11_SNORM_BLOCK"),
            155i32 => f.write_str("EAC_R11G11_UNORM_BLOCK"),
            156i32 => f.write_str("EAC_R11G11_SNORM_BLOCK"),
            157i32 => f.write_str("ASTC_4x4_UNORM_BLOCK"),
            158i32 => f.write_str("ASTC_4x4_SRGB_BLOCK"),
            159i32 => f.write_str("ASTC_5x4_UNORM_BLOCK"),
            160i32 => f.write_str("ASTC_5x4_SRGB_BLOCK"),
            161i32 => f.write_str("ASTC_5x5_UNORM_BLOCK"),
            162i32 => f.write_str("ASTC_5x5_SRGB_BLOCK"),
            163i32 => f.write_str("ASTC_6x5_UNORM_BLOCK"),
            164i32 => f.write_str("ASTC_6x5_SRGB_BLOCK"),
            165i32 => f.write_str("ASTC_6x6_UNORM_BLOCK"),
            166i32 => f.write_str("ASTC_6x6_SRGB_BLOCK"),
            167i32 => f.write_str("ASTC_8x5_UNORM_BLOCK"),
            168i32 => f.write_str("ASTC_8x5_SRGB_BLOCK"),
            169i32 => f.write_str("ASTC_8x6_UNORM_BLOCK"),
            170i32 => f.write_str("ASTC_8x6_SRGB_BLOCK"),
            171i32 => f.write_str("ASTC_8x8_UNORM_BLOCK"),
            172i32 => f.write_str("ASTC_8x8_SRGB_BLOCK"),
            173i32 => f.write_str("ASTC_10x5_UNORM_BLOCK"),
            174i32 => f.write_str("ASTC_10x5_SRGB_BLOCK"),
            175i32 => f.write_str("ASTC_10x6_UNORM_BLOCK"),
            176i32 => f.write_str("ASTC_10x6_SRGB_BLOCK"),
            177i32 => f.write_str("ASTC_10x8_UNORM_BLOCK"),
            178i32 => f.write_str("ASTC_10x8_SRGB_BLOCK"),
            179i32 => f.write_str("ASTC_10x10_UNORM_BLOCK"),
            180i32 => f.write_str("ASTC_10x10_SRGB_BLOCK"),
            181i32 => f.write_str("ASTC_12x10_UNORM_BLOCK"),
            182i32 => f.write_str("ASTC_12x10_SRGB_BLOCK"),
            183i32 => f.write_str("ASTC_12x12_UNORM_BLOCK"),
            184i32 => f.write_str("ASTC_12x12_SRGB_BLOCK"),
            1000156000i32 => f.write_str("G8B8G8R8_422_UNORM"),
            1000156001i32 => f.write_str("B8G8R8G8_422_UNORM"),
            1000156002i32 => f.write_str("G8_B8_R8_3PLANE_420_UNORM"),
            1000156003i32 => f.write_str("G8_B8R8_2PLANE_420_UNORM"),
            1000156004i32 => f.write_str("G8_B8_R8_3PLANE_422_UNORM"),
            1000156005i32 => f.write_str("G8_B8R8_2PLANE_422_UNORM"),
            1000156006i32 => f.write_str("G8_B8_R8_3PLANE_444_UNORM"),
            1000156007i32 => f.write_str("R10X6_UNORM_PACK16"),
            1000156008i32 => f.write_str("R10X6G10X6_UNORM_2PACK16"),
            1000156009i32 => f.write_str("R10X6G10X6B10X6A10X6_UNORM_4PACK16"),
            1000156010i32 => f.write_str("G10X6B10X6G10X6R10X6_422_UNORM_4PACK16"),
            1000156011i32 => f.write_str("B10X6G10X6R10X6G10X6_422_UNORM_4PACK16"),
            1000156012i32 => f.write_str("G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16"),
            1000156013i32 => f.write_str("G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16"),
            1000156014i32 => f.write_str("G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16"),
            1000156015i32 => f.write_str("G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16"),
            1000156016i32 => f.write_str("G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16"),
            1000156017i32 => f.write_str("R12X4_UNORM_PACK16"),
            1000156018i32 => f.write_str("R12X4G12X4_UNORM_2PACK16"),
            1000156019i32 => f.write_str("R12X4G12X4B12X4A12X4_UNORM_4PACK16"),
            1000156020i32 => f.write_str("G12X4B12X4G12X4R12X4_422_UNORM_4PACK16"),
            1000156021i32 => f.write_str("B12X4G12X4R12X4G12X4_422_UNORM_4PACK16"),
            1000156022i32 => f.write_str("G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16"),
            1000156023i32 => f.write_str("G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16"),
            1000156024i32 => f.write_str("G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16"),
            1000156025i32 => f.write_str("G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16"),
            1000156026i32 => f.write_str("G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16"),
            1000156027i32 => f.write_str("G16B16G16R16_422_UNORM"),
            1000156028i32 => f.write_str("B16G16R16G16_422_UNORM"),
            1000156029i32 => f.write_str("G16_B16_R16_3PLANE_420_UNORM"),
            1000156030i32 => f.write_str("G16_B16R16_2PLANE_420_UNORM"),
            1000156031i32 => f.write_str("G16_B16_R16_3PLANE_422_UNORM"),
            1000156032i32 => f.write_str("G16_B16R16_2PLANE_422_UNORM"),
            1000156033i32 => f.write_str("G16_B16_R16_3PLANE_444_UNORM"),
            1000330000i32 => f.write_str("G8_B8R8_2PLANE_444_UNORM"),
            1000330001i32 => f.write_str("G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16"),
            1000330002i32 => f.write_str("G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16"),
            1000330003i32 => f.write_str("G16_B16R16_2PLANE_444_UNORM"),
            1000340000i32 => f.write_str("A4R4G4B4_UNORM_PACK16"),
            1000340001i32 => f.write_str("A4B4G4R4_UNORM_PACK16"),
            1000066000i32 => f.write_str("ASTC_4x4_SFLOAT_BLOCK"),
            1000066001i32 => f.write_str("ASTC_5x4_SFLOAT_BLOCK"),
            1000066002i32 => f.write_str("ASTC_5x5_SFLOAT_BLOCK"),
            1000066003i32 => f.write_str("ASTC_6x5_SFLOAT_BLOCK"),
            1000066004i32 => f.write_str("ASTC_6x6_SFLOAT_BLOCK"),
            1000066005i32 => f.write_str("ASTC_8x5_SFLOAT_BLOCK"),
            1000066006i32 => f.write_str("ASTC_8x6_SFLOAT_BLOCK"),
            1000066007i32 => f.write_str("ASTC_8x8_SFLOAT_BLOCK"),
            1000066008i32 => f.write_str("ASTC_10x5_SFLOAT_BLOCK"),
            1000066009i32 => f.write_str("ASTC_10x6_SFLOAT_BLOCK"),
            1000066010i32 => f.write_str("ASTC_10x8_SFLOAT_BLOCK"),
            1000066011i32 => f.write_str("ASTC_10x10_SFLOAT_BLOCK"),
            1000066012i32 => f.write_str("ASTC_12x10_SFLOAT_BLOCK"),
            1000066013i32 => f.write_str("ASTC_12x12_SFLOAT_BLOCK"),
            1000470000i32 => f.write_str("A1B5G5R5_UNORM_PACK16"),
            1000470001i32 => f.write_str("A8_UNORM"),
            1000054000i32 => f.write_str("PVRTC1_2BPP_UNORM_BLOCK"),
            1000054001i32 => f.write_str("PVRTC1_4BPP_UNORM_BLOCK"),
            1000054002i32 => f.write_str("PVRTC2_2BPP_UNORM_BLOCK"),
            1000054003i32 => f.write_str("PVRTC2_4BPP_UNORM_BLOCK"),
            1000054004i32 => f.write_str("PVRTC1_2BPP_SRGB_BLOCK"),
            1000054005i32 => f.write_str("PVRTC1_4BPP_SRGB_BLOCK"),
            1000054006i32 => f.write_str("PVRTC2_2BPP_SRGB_BLOCK"),
            1000054007i32 => f.write_str("PVRTC2_4BPP_SRGB_BLOCK"),
            1000288000i32 => f.write_str("ASTC_3x3x3_UNORM_BLOCK"),
            1000288001i32 => f.write_str("ASTC_3x3x3_SRGB_BLOCK"),
            1000288002i32 => f.write_str("ASTC_3x3x3_SFLOAT_BLOCK"),
            1000288003i32 => f.write_str("ASTC_4x3x3_UNORM_BLOCK"),
            1000288004i32 => f.write_str("ASTC_4x3x3_SRGB_BLOCK"),
            1000288005i32 => f.write_str("ASTC_4x3x3_SFLOAT_BLOCK"),
            1000288006i32 => f.write_str("ASTC_4x4x3_UNORM_BLOCK"),
            1000288007i32 => f.write_str("ASTC_4x4x3_SRGB_BLOCK"),
            1000288008i32 => f.write_str("ASTC_4x4x3_SFLOAT_BLOCK"),
            1000288009i32 => f.write_str("ASTC_4x4x4_UNORM_BLOCK"),
            1000288010i32 => f.write_str("ASTC_4x4x4_SRGB_BLOCK"),
            1000288011i32 => f.write_str("ASTC_4x4x4_SFLOAT_BLOCK"),
            1000288012i32 => f.write_str("ASTC_5x4x4_UNORM_BLOCK"),
            1000288013i32 => f.write_str("ASTC_5x4x4_SRGB_BLOCK"),
            1000288014i32 => f.write_str("ASTC_5x4x4_SFLOAT_BLOCK"),
            1000288015i32 => f.write_str("ASTC_5x5x4_UNORM_BLOCK"),
            1000288016i32 => f.write_str("ASTC_5x5x4_SRGB_BLOCK"),
            1000288017i32 => f.write_str("ASTC_5x5x4_SFLOAT_BLOCK"),
            1000288018i32 => f.write_str("ASTC_5x5x5_UNORM_BLOCK"),
            1000288019i32 => f.write_str("ASTC_5x5x5_SRGB_BLOCK"),
            1000288020i32 => f.write_str("ASTC_5x5x5_SFLOAT_BLOCK"),
            1000288021i32 => f.write_str("ASTC_6x5x5_UNORM_BLOCK"),
            1000288022i32 => f.write_str("ASTC_6x5x5_SRGB_BLOCK"),
            1000288023i32 => f.write_str("ASTC_6x5x5_SFLOAT_BLOCK"),
            1000288024i32 => f.write_str("ASTC_6x6x5_UNORM_BLOCK"),
            1000288025i32 => f.write_str("ASTC_6x6x5_SRGB_BLOCK"),
            1000288026i32 => f.write_str("ASTC_6x6x5_SFLOAT_BLOCK"),
            1000288027i32 => f.write_str("ASTC_6x6x6_UNORM_BLOCK"),
            1000288028i32 => f.write_str("ASTC_6x6x6_SRGB_BLOCK"),
            1000288029i32 => f.write_str("ASTC_6x6x6_SFLOAT_BLOCK"),
            1000460000i32 => f.write_str("R8_BOOL"),
            1000460001i32 => f.write_str("R16_SFLOAT_FPENCODING_BFLOAT16"),
            1000460002i32 => f.write_str("R8_SFLOAT_FPENCODING_FLOAT8E4M3"),
            1000460003i32 => f.write_str("R8_SFLOAT_FPENCODING_FLOAT8E5M2"),
            1000464000i32 => f.write_str("R16G16_SFIXED5"),
            1000609000i32 => f.write_str("R10X6_UINT_PACK16"),
            1000609001i32 => f.write_str("R10X6G10X6_UINT_2PACK16"),
            1000609002i32 => f.write_str("R10X6G10X6B10X6A10X6_UINT_4PACK16"),
            1000609003i32 => f.write_str("R12X4_UINT_PACK16"),
            1000609004i32 => f.write_str("R12X4G12X4_UINT_2PACK16"),
            1000609005i32 => f.write_str("R12X4G12X4B12X4A12X4_UINT_4PACK16"),
            1000609006i32 => f.write_str("R14X2_UINT_PACK16"),
            1000609007i32 => f.write_str("R14X2G14X2_UINT_2PACK16"),
            1000609008i32 => f.write_str("R14X2G14X2B14X2A14X2_UINT_4PACK16"),
            1000609009i32 => f.write_str("R14X2_UNORM_PACK16"),
            1000609010i32 => f.write_str("R14X2G14X2_UNORM_2PACK16"),
            1000609011i32 => f.write_str("R14X2G14X2B14X2A14X2_UNORM_4PACK16"),
            1000609012i32 => f.write_str("G14X2_B14X2R14X2_2PLANE_420_UNORM_3PACK16"),
            1000609013i32 => f.write_str("G14X2_B14X2R14X2_2PLANE_422_UNORM_3PACK16"),
            other => write!(f, "{}({})", stringify!(Format), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkAccelerationStructureCompatibilityKHR")]
pub struct AccelerationStructureCompatibilityKHR(i32);
impl AccelerationStructureCompatibilityKHR {
    pub const COMPATIBLE: Self = Self(0i32);
    pub const INCOMPATIBLE: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for AccelerationStructureCompatibilityKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("COMPATIBLE"),
            1i32 => f.write_str("INCOMPATIBLE"),
            other => {
                write!(
                    f, "{}({})", stringify!(AccelerationStructureCompatibilityKHR), other
                )
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPhysicalDeviceDataGraphProcessingEngineTypeARM")]
pub struct PhysicalDeviceDataGraphProcessingEngineTypeARM(i32);
impl PhysicalDeviceDataGraphProcessingEngineTypeARM {
    pub const DEFAULT: Self = Self(0i32);
    pub const NEURAL: Self = Self(1000629000i32);
    pub const COMPUTE: Self = Self(1000629001i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PhysicalDeviceDataGraphProcessingEngineTypeARM {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("DEFAULT"),
            1000629000i32 => f.write_str("NEURAL"),
            1000629001i32 => f.write_str("COMPUTE"),
            other => {
                write!(
                    f, "{}({})",
                    stringify!(PhysicalDeviceDataGraphProcessingEngineTypeARM), other
                )
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkCopyAccelerationStructureModeKHR")]
pub struct CopyAccelerationStructureModeKHR(i32);
impl CopyAccelerationStructureModeKHR {
    pub const CLONE: Self = Self(0i32);
    pub const COMPACT: Self = Self(1i32);
    pub const SERIALIZE: Self = Self(2i32);
    pub const DESERIALIZE: Self = Self(3i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for CopyAccelerationStructureModeKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("CLONE"),
            1i32 => f.write_str("COMPACT"),
            2i32 => f.write_str("SERIALIZE"),
            3i32 => f.write_str("DESERIALIZE"),
            other => {
                write!(f, "{}({})", stringify!(CopyAccelerationStructureModeKHR), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkChromaLocation")]
pub struct ChromaLocation(i32);
impl ChromaLocation {
    pub const COSITED_EVEN: Self = Self(0i32);
    pub const MIDPOINT: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ChromaLocation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("COSITED_EVEN"),
            1i32 => f.write_str("MIDPOINT"),
            other => write!(f, "{}({})", stringify!(ChromaLocation), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPerformanceCounterUnitKHR")]
pub struct PerformanceCounterUnitKHR(i32);
impl PerformanceCounterUnitKHR {
    pub const GENERIC: Self = Self(0i32);
    pub const PERCENTAGE: Self = Self(1i32);
    pub const NANOSECONDS: Self = Self(2i32);
    pub const BYTES: Self = Self(3i32);
    pub const BYTES_PER_SECOND: Self = Self(4i32);
    pub const KELVIN: Self = Self(5i32);
    pub const WATTS: Self = Self(6i32);
    pub const VOLTS: Self = Self(7i32);
    pub const AMPS: Self = Self(8i32);
    pub const HERTZ: Self = Self(9i32);
    pub const CYCLES: Self = Self(10i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PerformanceCounterUnitKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("GENERIC"),
            1i32 => f.write_str("PERCENTAGE"),
            2i32 => f.write_str("NANOSECONDS"),
            3i32 => f.write_str("BYTES"),
            4i32 => f.write_str("BYTES_PER_SECOND"),
            5i32 => f.write_str("KELVIN"),
            6i32 => f.write_str("WATTS"),
            7i32 => f.write_str("VOLTS"),
            8i32 => f.write_str("AMPS"),
            9i32 => f.write_str("HERTZ"),
            10i32 => f.write_str("CYCLES"),
            other => write!(f, "{}({})", stringify!(PerformanceCounterUnitKHR), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkBlockMatchWindowCompareModeQCOM")]
pub struct BlockMatchWindowCompareModeQCOM(i32);
impl BlockMatchWindowCompareModeQCOM {
    pub const MIN: Self = Self(0i32);
    pub const MAX: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for BlockMatchWindowCompareModeQCOM {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("MIN"),
            1i32 => f.write_str("MAX"),
            other => {
                write!(f, "{}({})", stringify!(BlockMatchWindowCompareModeQCOM), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPhysicalDeviceDataGraphOperationTypeARM")]
pub struct PhysicalDeviceDataGraphOperationTypeARM(i32);
impl PhysicalDeviceDataGraphOperationTypeARM {
    pub const SPIRV_EXTENDED_INSTRUCTION_SET: Self = Self(0i32);
    pub const NEURAL_MODEL: Self = Self(1000629000i32);
    pub const BUILTIN_MODEL: Self = Self(1000629001i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PhysicalDeviceDataGraphOperationTypeARM {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("SPIRV_EXTENDED_INSTRUCTION_SET"),
            1000629000i32 => f.write_str("NEURAL_MODEL"),
            1000629001i32 => f.write_str("BUILTIN_MODEL"),
            other => {
                write!(
                    f, "{}({})", stringify!(PhysicalDeviceDataGraphOperationTypeARM),
                    other
                )
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPointClippingBehavior")]
pub struct PointClippingBehavior(i32);
impl PointClippingBehavior {
    pub const ALL_CLIP_PLANES: Self = Self(0i32);
    pub const USER_CLIP_PLANES_ONLY: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PointClippingBehavior {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("ALL_CLIP_PLANES"),
            1i32 => f.write_str("USER_CLIP_PLANES_ONLY"),
            other => write!(f, "{}({})", stringify!(PointClippingBehavior), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkSubpassMergeStatusEXT")]
pub struct SubpassMergeStatusEXT(i32);
impl SubpassMergeStatusEXT {
    pub const MERGED: Self = Self(0i32);
    pub const DISALLOWED: Self = Self(1i32);
    pub const NOT_MERGED_SIDE_EFFECTS: Self = Self(2i32);
    pub const NOT_MERGED_SAMPLES_MISMATCH: Self = Self(3i32);
    pub const NOT_MERGED_VIEWS_MISMATCH: Self = Self(4i32);
    pub const NOT_MERGED_ALIASING: Self = Self(5i32);
    pub const NOT_MERGED_DEPENDENCIES: Self = Self(6i32);
    pub const NOT_MERGED_INCOMPATIBLE_INPUT_ATTACHMENT: Self = Self(7i32);
    pub const NOT_MERGED_TOO_MANY_ATTACHMENTS: Self = Self(8i32);
    pub const NOT_MERGED_INSUFFICIENT_STORAGE: Self = Self(9i32);
    pub const NOT_MERGED_DEPTH_STENCIL_COUNT: Self = Self(10i32);
    pub const NOT_MERGED_RESOLVE_ATTACHMENT_REUSE: Self = Self(11i32);
    pub const NOT_MERGED_SINGLE_SUBPASS: Self = Self(12i32);
    pub const NOT_MERGED_UNSPECIFIED: Self = Self(13i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for SubpassMergeStatusEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("MERGED"),
            1i32 => f.write_str("DISALLOWED"),
            2i32 => f.write_str("NOT_MERGED_SIDE_EFFECTS"),
            3i32 => f.write_str("NOT_MERGED_SAMPLES_MISMATCH"),
            4i32 => f.write_str("NOT_MERGED_VIEWS_MISMATCH"),
            5i32 => f.write_str("NOT_MERGED_ALIASING"),
            6i32 => f.write_str("NOT_MERGED_DEPENDENCIES"),
            7i32 => f.write_str("NOT_MERGED_INCOMPATIBLE_INPUT_ATTACHMENT"),
            8i32 => f.write_str("NOT_MERGED_TOO_MANY_ATTACHMENTS"),
            9i32 => f.write_str("NOT_MERGED_INSUFFICIENT_STORAGE"),
            10i32 => f.write_str("NOT_MERGED_DEPTH_STENCIL_COUNT"),
            11i32 => f.write_str("NOT_MERGED_RESOLVE_ATTACHMENT_REUSE"),
            12i32 => f.write_str("NOT_MERGED_SINGLE_SUBPASS"),
            13i32 => f.write_str("NOT_MERGED_UNSPECIFIED"),
            other => write!(f, "{}({})", stringify!(SubpassMergeStatusEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkBlendOp")]
pub struct BlendOp(i32);
impl BlendOp {
    pub const ADD: Self = Self(0i32);
    pub const SUBTRACT: Self = Self(1i32);
    pub const REVERSE_SUBTRACT: Self = Self(2i32);
    pub const MIN: Self = Self(3i32);
    pub const MAX: Self = Self(4i32);
    pub const ZERO: Self = Self(1000148000i32);
    pub const SRC: Self = Self(1000148001i32);
    pub const DST: Self = Self(1000148002i32);
    pub const SRC_OVER: Self = Self(1000148003i32);
    pub const DST_OVER: Self = Self(1000148004i32);
    pub const SRC_IN: Self = Self(1000148005i32);
    pub const DST_IN: Self = Self(1000148006i32);
    pub const SRC_OUT: Self = Self(1000148007i32);
    pub const DST_OUT: Self = Self(1000148008i32);
    pub const SRC_ATOP: Self = Self(1000148009i32);
    pub const DST_ATOP: Self = Self(1000148010i32);
    pub const XOR: Self = Self(1000148011i32);
    pub const MULTIPLY: Self = Self(1000148012i32);
    pub const SCREEN: Self = Self(1000148013i32);
    pub const OVERLAY: Self = Self(1000148014i32);
    pub const DARKEN: Self = Self(1000148015i32);
    pub const LIGHTEN: Self = Self(1000148016i32);
    pub const COLORDODGE: Self = Self(1000148017i32);
    pub const COLORBURN: Self = Self(1000148018i32);
    pub const HARDLIGHT: Self = Self(1000148019i32);
    pub const SOFTLIGHT: Self = Self(1000148020i32);
    pub const DIFFERENCE: Self = Self(1000148021i32);
    pub const EXCLUSION: Self = Self(1000148022i32);
    pub const INVERT: Self = Self(1000148023i32);
    pub const INVERT_RGB: Self = Self(1000148024i32);
    pub const LINEARDODGE: Self = Self(1000148025i32);
    pub const LINEARBURN: Self = Self(1000148026i32);
    pub const VIVIDLIGHT: Self = Self(1000148027i32);
    pub const LINEARLIGHT: Self = Self(1000148028i32);
    pub const PINLIGHT: Self = Self(1000148029i32);
    pub const HARDMIX: Self = Self(1000148030i32);
    pub const HSL_HUE: Self = Self(1000148031i32);
    pub const HSL_SATURATION: Self = Self(1000148032i32);
    pub const HSL_COLOR: Self = Self(1000148033i32);
    pub const HSL_LUMINOSITY: Self = Self(1000148034i32);
    pub const PLUS: Self = Self(1000148035i32);
    pub const PLUS_CLAMPED: Self = Self(1000148036i32);
    pub const PLUS_CLAMPED_ALPHA: Self = Self(1000148037i32);
    pub const PLUS_DARKER: Self = Self(1000148038i32);
    pub const MINUS: Self = Self(1000148039i32);
    pub const MINUS_CLAMPED: Self = Self(1000148040i32);
    pub const CONTRAST: Self = Self(1000148041i32);
    pub const INVERT_OVG: Self = Self(1000148042i32);
    pub const RED: Self = Self(1000148043i32);
    pub const GREEN: Self = Self(1000148044i32);
    pub const BLUE: Self = Self(1000148045i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for BlendOp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("ADD"),
            1i32 => f.write_str("SUBTRACT"),
            2i32 => f.write_str("REVERSE_SUBTRACT"),
            3i32 => f.write_str("MIN"),
            4i32 => f.write_str("MAX"),
            1000148000i32 => f.write_str("ZERO"),
            1000148001i32 => f.write_str("SRC"),
            1000148002i32 => f.write_str("DST"),
            1000148003i32 => f.write_str("SRC_OVER"),
            1000148004i32 => f.write_str("DST_OVER"),
            1000148005i32 => f.write_str("SRC_IN"),
            1000148006i32 => f.write_str("DST_IN"),
            1000148007i32 => f.write_str("SRC_OUT"),
            1000148008i32 => f.write_str("DST_OUT"),
            1000148009i32 => f.write_str("SRC_ATOP"),
            1000148010i32 => f.write_str("DST_ATOP"),
            1000148011i32 => f.write_str("XOR"),
            1000148012i32 => f.write_str("MULTIPLY"),
            1000148013i32 => f.write_str("SCREEN"),
            1000148014i32 => f.write_str("OVERLAY"),
            1000148015i32 => f.write_str("DARKEN"),
            1000148016i32 => f.write_str("LIGHTEN"),
            1000148017i32 => f.write_str("COLORDODGE"),
            1000148018i32 => f.write_str("COLORBURN"),
            1000148019i32 => f.write_str("HARDLIGHT"),
            1000148020i32 => f.write_str("SOFTLIGHT"),
            1000148021i32 => f.write_str("DIFFERENCE"),
            1000148022i32 => f.write_str("EXCLUSION"),
            1000148023i32 => f.write_str("INVERT"),
            1000148024i32 => f.write_str("INVERT_RGB"),
            1000148025i32 => f.write_str("LINEARDODGE"),
            1000148026i32 => f.write_str("LINEARBURN"),
            1000148027i32 => f.write_str("VIVIDLIGHT"),
            1000148028i32 => f.write_str("LINEARLIGHT"),
            1000148029i32 => f.write_str("PINLIGHT"),
            1000148030i32 => f.write_str("HARDMIX"),
            1000148031i32 => f.write_str("HSL_HUE"),
            1000148032i32 => f.write_str("HSL_SATURATION"),
            1000148033i32 => f.write_str("HSL_COLOR"),
            1000148034i32 => f.write_str("HSL_LUMINOSITY"),
            1000148035i32 => f.write_str("PLUS"),
            1000148036i32 => f.write_str("PLUS_CLAMPED"),
            1000148037i32 => f.write_str("PLUS_CLAMPED_ALPHA"),
            1000148038i32 => f.write_str("PLUS_DARKER"),
            1000148039i32 => f.write_str("MINUS"),
            1000148040i32 => f.write_str("MINUS_CLAMPED"),
            1000148041i32 => f.write_str("CONTRAST"),
            1000148042i32 => f.write_str("INVERT_OVG"),
            1000148043i32 => f.write_str("RED"),
            1000148044i32 => f.write_str("GREEN"),
            1000148045i32 => f.write_str("BLUE"),
            other => write!(f, "{}({})", stringify!(BlendOp), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkValidationCheckEXT")]
pub struct ValidationCheckEXT(i32);
impl ValidationCheckEXT {
    pub const ALL: Self = Self(0i32);
    pub const SHADERS: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ValidationCheckEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("ALL"),
            1i32 => f.write_str("SHADERS"),
            other => write!(f, "{}({})", stringify!(ValidationCheckEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkVendorId")]
pub struct VendorId(i32);
impl VendorId {
    pub const KHRONOS: Self = Self(65536i32);
    pub const VIV: Self = Self(65537i32);
    pub const VSI: Self = Self(65538i32);
    pub const KAZAN: Self = Self(65539i32);
    pub const CODEPLAY: Self = Self(65540i32);
    pub const POCL: Self = Self(65542i32);
    pub const MOBILEYE: Self = Self(65543i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for VendorId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            65536i32 => f.write_str("KHRONOS"),
            65537i32 => f.write_str("VIV"),
            65538i32 => f.write_str("VSI"),
            65539i32 => f.write_str("KAZAN"),
            65540i32 => f.write_str("CODEPLAY"),
            65542i32 => f.write_str("POCL"),
            65543i32 => f.write_str("MOBILEYE"),
            other => write!(f, "{}({})", stringify!(VendorId), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPipelineExecutableStatisticFormatKHR")]
pub struct PipelineExecutableStatisticFormatKHR(i32);
impl PipelineExecutableStatisticFormatKHR {
    pub const BOOL32: Self = Self(0i32);
    pub const INT64: Self = Self(1i32);
    pub const UINT64: Self = Self(2i32);
    pub const FLOAT64: Self = Self(3i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PipelineExecutableStatisticFormatKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("BOOL32"),
            1i32 => f.write_str("INT64"),
            2i32 => f.write_str("UINT64"),
            3i32 => f.write_str("FLOAT64"),
            other => {
                write!(
                    f, "{}({})", stringify!(PipelineExecutableStatisticFormatKHR), other
                )
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkStencilOp")]
pub struct StencilOp(i32);
impl StencilOp {
    pub const KEEP: Self = Self(0i32);
    pub const ZERO: Self = Self(1i32);
    pub const REPLACE: Self = Self(2i32);
    pub const INCREMENT_AND_CLAMP: Self = Self(3i32);
    pub const DECREMENT_AND_CLAMP: Self = Self(4i32);
    pub const INVERT: Self = Self(5i32);
    pub const INCREMENT_AND_WRAP: Self = Self(6i32);
    pub const DECREMENT_AND_WRAP: Self = Self(7i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for StencilOp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("KEEP"),
            1i32 => f.write_str("ZERO"),
            2i32 => f.write_str("REPLACE"),
            3i32 => f.write_str("INCREMENT_AND_CLAMP"),
            4i32 => f.write_str("DECREMENT_AND_CLAMP"),
            5i32 => f.write_str("INVERT"),
            6i32 => f.write_str("INCREMENT_AND_WRAP"),
            7i32 => f.write_str("DECREMENT_AND_WRAP"),
            other => write!(f, "{}({})", stringify!(StencilOp), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkComponentSwizzle")]
pub struct ComponentSwizzle(i32);
impl ComponentSwizzle {
    pub const IDENTITY: Self = Self(0i32);
    pub const ZERO: Self = Self(1i32);
    pub const ONE: Self = Self(2i32);
    pub const R: Self = Self(3i32);
    pub const G: Self = Self(4i32);
    pub const B: Self = Self(5i32);
    pub const A: Self = Self(6i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ComponentSwizzle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("IDENTITY"),
            1i32 => f.write_str("ZERO"),
            2i32 => f.write_str("ONE"),
            3i32 => f.write_str("R"),
            4i32 => f.write_str("G"),
            5i32 => f.write_str("B"),
            6i32 => f.write_str("A"),
            other => write!(f, "{}({})", stringify!(ComponentSwizzle), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkRayTracingInvocationReorderModeEXT")]
pub struct RayTracingInvocationReorderModeEXT(i32);
impl RayTracingInvocationReorderModeEXT {
    pub const NONE: Self = Self(0i32);
    pub const REORDER: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for RayTracingInvocationReorderModeEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("NONE"),
            1i32 => f.write_str("REORDER"),
            other => {
                write!(
                    f, "{}({})", stringify!(RayTracingInvocationReorderModeEXT), other
                )
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkFrontFace")]
pub struct FrontFace(i32);
impl FrontFace {
    pub const COUNTER_CLOCKWISE: Self = Self(0i32);
    pub const CLOCKWISE: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for FrontFace {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("COUNTER_CLOCKWISE"),
            1i32 => f.write_str("CLOCKWISE"),
            other => write!(f, "{}({})", stringify!(FrontFace), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDepthClampModeEXT")]
pub struct DepthClampModeEXT(i32);
impl DepthClampModeEXT {
    pub const VIEWPORT_RANGE: Self = Self(0i32);
    pub const USER_DEFINED_RANGE: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DepthClampModeEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("VIEWPORT_RANGE"),
            1i32 => f.write_str("USER_DEFINED_RANGE"),
            other => write!(f, "{}({})", stringify!(DepthClampModeEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkSubpassContents")]
pub struct SubpassContents(i32);
impl SubpassContents {
    pub const INLINE: Self = Self(0i32);
    pub const SECONDARY_COMMAND_BUFFERS: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for SubpassContents {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("INLINE"),
            1i32 => f.write_str("SECONDARY_COMMAND_BUFFERS"),
            other => write!(f, "{}({})", stringify!(SubpassContents), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDisplayPowerStateEXT")]
pub struct DisplayPowerStateEXT(i32);
impl DisplayPowerStateEXT {
    pub const OFF: Self = Self(0i32);
    pub const SUSPEND: Self = Self(1i32);
    pub const ON: Self = Self(2i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DisplayPowerStateEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("OFF"),
            1i32 => f.write_str("SUSPEND"),
            2i32 => f.write_str("ON"),
            other => write!(f, "{}({})", stringify!(DisplayPowerStateEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkIndirectCommandsTokenTypeNV")]
pub struct IndirectCommandsTokenTypeNV(i32);
impl IndirectCommandsTokenTypeNV {
    pub const SHADER_GROUP: Self = Self(0i32);
    pub const STATE_FLAGS: Self = Self(1i32);
    pub const INDEX_BUFFER: Self = Self(2i32);
    pub const VERTEX_BUFFER: Self = Self(3i32);
    pub const PUSH_CONSTANT: Self = Self(4i32);
    pub const DRAW_INDEXED: Self = Self(5i32);
    pub const DRAW: Self = Self(6i32);
    pub const DRAW_TASKS: Self = Self(7i32);
    pub const PUSH_DATA: Self = Self(1000135000i32);
    pub const DRAW_MESH_TASKS: Self = Self(1000328000i32);
    pub const PIPELINE: Self = Self(1000428003i32);
    pub const DISPATCH: Self = Self(1000428004i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for IndirectCommandsTokenTypeNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("SHADER_GROUP"),
            1i32 => f.write_str("STATE_FLAGS"),
            2i32 => f.write_str("INDEX_BUFFER"),
            3i32 => f.write_str("VERTEX_BUFFER"),
            4i32 => f.write_str("PUSH_CONSTANT"),
            5i32 => f.write_str("DRAW_INDEXED"),
            6i32 => f.write_str("DRAW"),
            7i32 => f.write_str("DRAW_TASKS"),
            1000135000i32 => f.write_str("PUSH_DATA"),
            1000328000i32 => f.write_str("DRAW_MESH_TASKS"),
            1000428003i32 => f.write_str("PIPELINE"),
            1000428004i32 => f.write_str("DISPATCH"),
            other => write!(f, "{}({})", stringify!(IndirectCommandsTokenTypeNV), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkCompareOp")]
pub struct CompareOp(i32);
impl CompareOp {
    pub const NEVER: Self = Self(0i32);
    pub const LESS: Self = Self(1i32);
    pub const EQUAL: Self = Self(2i32);
    pub const LESS_OR_EQUAL: Self = Self(3i32);
    pub const GREATER: Self = Self(4i32);
    pub const NOT_EQUAL: Self = Self(5i32);
    pub const GREATER_OR_EQUAL: Self = Self(6i32);
    pub const ALWAYS: Self = Self(7i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for CompareOp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("NEVER"),
            1i32 => f.write_str("LESS"),
            2i32 => f.write_str("EQUAL"),
            3i32 => f.write_str("LESS_OR_EQUAL"),
            4i32 => f.write_str("GREATER"),
            5i32 => f.write_str("NOT_EQUAL"),
            6i32 => f.write_str("GREATER_OR_EQUAL"),
            7i32 => f.write_str("ALWAYS"),
            other => write!(f, "{}({})", stringify!(CompareOp), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkFaultLevel")]
pub struct FaultLevel(i32);
impl FaultLevel {
    pub const UNASSIGNED: Self = Self(0i32);
    pub const CRITICAL: Self = Self(1i32);
    pub const RECOVERABLE: Self = Self(2i32);
    pub const WARNING: Self = Self(3i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for FaultLevel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("UNASSIGNED"),
            1i32 => f.write_str("CRITICAL"),
            2i32 => f.write_str("RECOVERABLE"),
            3i32 => f.write_str("WARNING"),
            other => write!(f, "{}({})", stringify!(FaultLevel), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDisplaySurfaceStereoTypeNV")]
pub struct DisplaySurfaceStereoTypeNV(i32);
impl DisplaySurfaceStereoTypeNV {
    pub const NONE: Self = Self(0i32);
    pub const ONBOARD_DIN: Self = Self(1i32);
    pub const HDMI_3D: Self = Self(2i32);
    pub const INBAND_DISPLAYPORT: Self = Self(3i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DisplaySurfaceStereoTypeNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("NONE"),
            1i32 => f.write_str("ONBOARD_DIN"),
            2i32 => f.write_str("HDMI_3D"),
            3i32 => f.write_str("INBAND_DISPLAYPORT"),
            other => write!(f, "{}({})", stringify!(DisplaySurfaceStereoTypeNV), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkSciSyncClientTypeNV")]
pub struct SciSyncClientTypeNV(i32);
impl SciSyncClientTypeNV {
    pub const SIGNALER: Self = Self(0i32);
    pub const WAITER: Self = Self(1i32);
    pub const SIGNALER_WAITER: Self = Self(2i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for SciSyncClientTypeNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("SIGNALER"),
            1i32 => f.write_str("WAITER"),
            2i32 => f.write_str("SIGNALER_WAITER"),
            other => write!(f, "{}({})", stringify!(SciSyncClientTypeNV), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkSharingMode")]
pub struct SharingMode(i32);
impl SharingMode {
    pub const EXCLUSIVE: Self = Self(0i32);
    pub const CONCURRENT: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for SharingMode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("EXCLUSIVE"),
            1i32 => f.write_str("CONCURRENT"),
            other => write!(f, "{}({})", stringify!(SharingMode), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkSamplerMipmapMode")]
pub struct SamplerMipmapMode(i32);
impl SamplerMipmapMode {
    pub const NEAREST: Self = Self(0i32);
    pub const LINEAR: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for SamplerMipmapMode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("NEAREST"),
            1i32 => f.write_str("LINEAR"),
            other => write!(f, "{}({})", stringify!(SamplerMipmapMode), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPipelineCacheHeaderVersion")]
pub struct PipelineCacheHeaderVersion(i32);
impl PipelineCacheHeaderVersion {
    pub const ONE: Self = Self(1i32);
    pub const SAFETY_CRITICAL_ONE: Self = Self(1000298001i32);
    pub const DATA_GRAPH: Self = Self(1000629000i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PipelineCacheHeaderVersion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            1i32 => f.write_str("ONE"),
            1000298001i32 => f.write_str("SAFETY_CRITICAL_ONE"),
            1000629000i32 => f.write_str("DATA_GRAPH"),
            other => write!(f, "{}({})", stringify!(PipelineCacheHeaderVersion), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkSamplerYcbcrRange")]
pub struct SamplerYcbcrRange(i32);
impl SamplerYcbcrRange {
    pub const ITU_FULL: Self = Self(0i32);
    pub const ITU_NARROW: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for SamplerYcbcrRange {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("ITU_FULL"),
            1i32 => f.write_str("ITU_NARROW"),
            other => write!(f, "{}({})", stringify!(SamplerYcbcrRange), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkQueryType")]
pub struct QueryType(i32);
impl QueryType {
    pub const OCCLUSION: Self = Self(0i32);
    pub const PIPELINE_STATISTICS: Self = Self(1i32);
    pub const TIMESTAMP: Self = Self(2i32);
    pub const RESULT_STATUS_ONLY: Self = Self(1000023000i32);
    pub const TRANSFORM_FEEDBACK_STREAM: Self = Self(1000028004i32);
    pub const PERFORMANCE_QUERY: Self = Self(1000116000i32);
    pub const ACCELERATION_STRUCTURE_COMPACTED_SIZE: Self = Self(1000150000i32);
    pub const ACCELERATION_STRUCTURE_SERIALIZATION_SIZE: Self = Self(1000150001i32);
    pub const VIDEO_ENCODE_FEEDBACK: Self = Self(1000299000i32);
    pub const MESH_PRIMITIVES_GENERATED: Self = Self(1000328000i32);
    pub const PRIMITIVES_GENERATED: Self = Self(1000382000i32);
    pub const ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS: Self = Self(
        1000386000i32,
    );
    pub const ACCELERATION_STRUCTURE_SIZE: Self = Self(1000386001i32);
    pub const MICROMAP_SERIALIZATION_SIZE: Self = Self(1000396000i32);
    pub const MICROMAP_COMPACTED_SIZE: Self = Self(1000396001i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for QueryType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("OCCLUSION"),
            1i32 => f.write_str("PIPELINE_STATISTICS"),
            2i32 => f.write_str("TIMESTAMP"),
            1000023000i32 => f.write_str("RESULT_STATUS_ONLY"),
            1000028004i32 => f.write_str("TRANSFORM_FEEDBACK_STREAM"),
            1000116000i32 => f.write_str("PERFORMANCE_QUERY"),
            1000150000i32 => f.write_str("ACCELERATION_STRUCTURE_COMPACTED_SIZE"),
            1000150001i32 => f.write_str("ACCELERATION_STRUCTURE_SERIALIZATION_SIZE"),
            1000299000i32 => f.write_str("VIDEO_ENCODE_FEEDBACK"),
            1000328000i32 => f.write_str("MESH_PRIMITIVES_GENERATED"),
            1000382000i32 => f.write_str("PRIMITIVES_GENERATED"),
            1000386000i32 => {
                f.write_str("ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS")
            }
            1000386001i32 => f.write_str("ACCELERATION_STRUCTURE_SIZE"),
            1000396000i32 => f.write_str("MICROMAP_SERIALIZATION_SIZE"),
            1000396001i32 => f.write_str("MICROMAP_COMPACTED_SIZE"),
            other => write!(f, "{}({})", stringify!(QueryType), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkAttachmentLoadOp")]
pub struct AttachmentLoadOp(i32);
impl AttachmentLoadOp {
    pub const LOAD: Self = Self(0i32);
    pub const CLEAR: Self = Self(1i32);
    pub const DONT_CARE: Self = Self(2i32);
    pub const NONE: Self = Self(1000400000i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for AttachmentLoadOp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("LOAD"),
            1i32 => f.write_str("CLEAR"),
            2i32 => f.write_str("DONT_CARE"),
            1000400000i32 => f.write_str("NONE"),
            other => write!(f, "{}({})", stringify!(AttachmentLoadOp), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDescriptorType")]
pub struct DescriptorType(i32);
impl DescriptorType {
    pub const SAMPLER: Self = Self(0i32);
    pub const COMBINED_IMAGE_SAMPLER: Self = Self(1i32);
    pub const SAMPLED_IMAGE: Self = Self(2i32);
    pub const STORAGE_IMAGE: Self = Self(3i32);
    pub const UNIFORM_TEXEL_BUFFER: Self = Self(4i32);
    pub const STORAGE_TEXEL_BUFFER: Self = Self(5i32);
    pub const UNIFORM_BUFFER: Self = Self(6i32);
    pub const STORAGE_BUFFER: Self = Self(7i32);
    pub const UNIFORM_BUFFER_DYNAMIC: Self = Self(8i32);
    pub const STORAGE_BUFFER_DYNAMIC: Self = Self(9i32);
    pub const INPUT_ATTACHMENT: Self = Self(10i32);
    pub const INLINE_UNIFORM_BLOCK: Self = Self(1000138000i32);
    pub const ACCELERATION_STRUCTURE: Self = Self(1000150000i32);
    pub const SAMPLE_WEIGHT_IMAGE: Self = Self(1000440000i32);
    pub const BLOCK_MATCH_IMAGE: Self = Self(1000440001i32);
    pub const TENSOR: Self = Self(1000460000i32);
    pub const PARTITIONED_ACCELERATION_STRUCTURE: Self = Self(1000570000i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DescriptorType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("SAMPLER"),
            1i32 => f.write_str("COMBINED_IMAGE_SAMPLER"),
            2i32 => f.write_str("SAMPLED_IMAGE"),
            3i32 => f.write_str("STORAGE_IMAGE"),
            4i32 => f.write_str("UNIFORM_TEXEL_BUFFER"),
            5i32 => f.write_str("STORAGE_TEXEL_BUFFER"),
            6i32 => f.write_str("UNIFORM_BUFFER"),
            7i32 => f.write_str("STORAGE_BUFFER"),
            8i32 => f.write_str("UNIFORM_BUFFER_DYNAMIC"),
            9i32 => f.write_str("STORAGE_BUFFER_DYNAMIC"),
            10i32 => f.write_str("INPUT_ATTACHMENT"),
            1000138000i32 => f.write_str("INLINE_UNIFORM_BLOCK"),
            1000150000i32 => f.write_str("ACCELERATION_STRUCTURE"),
            1000440000i32 => f.write_str("SAMPLE_WEIGHT_IMAGE"),
            1000440001i32 => f.write_str("BLOCK_MATCH_IMAGE"),
            1000460000i32 => f.write_str("TENSOR"),
            1000570000i32 => f.write_str("PARTITIONED_ACCELERATION_STRUCTURE"),
            other => write!(f, "{}({})", stringify!(DescriptorType), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkCoverageReductionModeNV")]
pub struct CoverageReductionModeNV(i32);
impl CoverageReductionModeNV {
    pub const MERGE: Self = Self(0i32);
    pub const TRUNCATE: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for CoverageReductionModeNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("MERGE"),
            1i32 => f.write_str("TRUNCATE"),
            other => write!(f, "{}({})", stringify!(CoverageReductionModeNV), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkFullScreenExclusiveEXT")]
pub struct FullScreenExclusiveEXT(i32);
impl FullScreenExclusiveEXT {
    pub const DEFAULT: Self = Self(0i32);
    pub const ALLOWED: Self = Self(1i32);
    pub const DISALLOWED: Self = Self(2i32);
    pub const APPLICATION_CONTROLLED: Self = Self(3i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for FullScreenExclusiveEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("DEFAULT"),
            1i32 => f.write_str("ALLOWED"),
            2i32 => f.write_str("DISALLOWED"),
            3i32 => f.write_str("APPLICATION_CONTROLLED"),
            other => write!(f, "{}({})", stringify!(FullScreenExclusiveEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkSciSyncPrimitiveTypeNV")]
pub struct SciSyncPrimitiveTypeNV(i32);
impl SciSyncPrimitiveTypeNV {
    pub const FENCE: Self = Self(0i32);
    pub const SEMAPHORE: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for SciSyncPrimitiveTypeNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("FENCE"),
            1i32 => f.write_str("SEMAPHORE"),
            other => write!(f, "{}({})", stringify!(SciSyncPrimitiveTypeNV), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkProvokingVertexModeEXT")]
pub struct ProvokingVertexModeEXT(i32);
impl ProvokingVertexModeEXT {
    pub const FIRST_VERTEX: Self = Self(0i32);
    pub const LAST_VERTEX: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ProvokingVertexModeEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("FIRST_VERTEX"),
            1i32 => f.write_str("LAST_VERTEX"),
            other => write!(f, "{}({})", stringify!(ProvokingVertexModeEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDepthBiasRepresentationEXT")]
pub struct DepthBiasRepresentationEXT(i32);
impl DepthBiasRepresentationEXT {
    pub const LEAST_REPRESENTABLE_VALUE_FORMAT: Self = Self(0i32);
    pub const LEAST_REPRESENTABLE_VALUE_FORCE_UNORM: Self = Self(1i32);
    pub const FLOAT: Self = Self(2i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DepthBiasRepresentationEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("LEAST_REPRESENTABLE_VALUE_FORMAT"),
            1i32 => f.write_str("LEAST_REPRESENTABLE_VALUE_FORCE_UNORM"),
            2i32 => f.write_str("FLOAT"),
            other => write!(f, "{}({})", stringify!(DepthBiasRepresentationEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkIndirectExecutionSetInfoTypeEXT")]
pub struct IndirectExecutionSetInfoTypeEXT(i32);
impl IndirectExecutionSetInfoTypeEXT {
    pub const PIPELINES: Self = Self(0i32);
    pub const SHADER_OBJECTS: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for IndirectExecutionSetInfoTypeEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("PIPELINES"),
            1i32 => f.write_str("SHADER_OBJECTS"),
            other => {
                write!(f, "{}({})", stringify!(IndirectExecutionSetInfoTypeEXT), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDeviceEventTypeEXT")]
pub struct DeviceEventTypeEXT(i32);
impl DeviceEventTypeEXT {
    pub const DISPLAY_HOTPLUG: Self = Self(0i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DeviceEventTypeEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("DISPLAY_HOTPLUG"),
            other => write!(f, "{}({})", stringify!(DeviceEventTypeEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkBlendFactor")]
pub struct BlendFactor(i32);
impl BlendFactor {
    pub const ZERO: Self = Self(0i32);
    pub const ONE: Self = Self(1i32);
    pub const SRC_COLOR: Self = Self(2i32);
    pub const ONE_MINUS_SRC_COLOR: Self = Self(3i32);
    pub const DST_COLOR: Self = Self(4i32);
    pub const ONE_MINUS_DST_COLOR: Self = Self(5i32);
    pub const SRC_ALPHA: Self = Self(6i32);
    pub const ONE_MINUS_SRC_ALPHA: Self = Self(7i32);
    pub const DST_ALPHA: Self = Self(8i32);
    pub const ONE_MINUS_DST_ALPHA: Self = Self(9i32);
    pub const CONSTANT_COLOR: Self = Self(10i32);
    pub const ONE_MINUS_CONSTANT_COLOR: Self = Self(11i32);
    pub const CONSTANT_ALPHA: Self = Self(12i32);
    pub const ONE_MINUS_CONSTANT_ALPHA: Self = Self(13i32);
    pub const SRC_ALPHA_SATURATE: Self = Self(14i32);
    pub const SRC1_COLOR: Self = Self(15i32);
    pub const ONE_MINUS_SRC1_COLOR: Self = Self(16i32);
    pub const SRC1_ALPHA: Self = Self(17i32);
    pub const ONE_MINUS_SRC1_ALPHA: Self = Self(18i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for BlendFactor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("ZERO"),
            1i32 => f.write_str("ONE"),
            2i32 => f.write_str("SRC_COLOR"),
            3i32 => f.write_str("ONE_MINUS_SRC_COLOR"),
            4i32 => f.write_str("DST_COLOR"),
            5i32 => f.write_str("ONE_MINUS_DST_COLOR"),
            6i32 => f.write_str("SRC_ALPHA"),
            7i32 => f.write_str("ONE_MINUS_SRC_ALPHA"),
            8i32 => f.write_str("DST_ALPHA"),
            9i32 => f.write_str("ONE_MINUS_DST_ALPHA"),
            10i32 => f.write_str("CONSTANT_COLOR"),
            11i32 => f.write_str("ONE_MINUS_CONSTANT_COLOR"),
            12i32 => f.write_str("CONSTANT_ALPHA"),
            13i32 => f.write_str("ONE_MINUS_CONSTANT_ALPHA"),
            14i32 => f.write_str("SRC_ALPHA_SATURATE"),
            15i32 => f.write_str("SRC1_COLOR"),
            16i32 => f.write_str("ONE_MINUS_SRC1_COLOR"),
            17i32 => f.write_str("SRC1_ALPHA"),
            18i32 => f.write_str("ONE_MINUS_SRC1_ALPHA"),
            other => write!(f, "{}({})", stringify!(BlendFactor), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDataGraphPipelinePropertyARM")]
pub struct DataGraphPipelinePropertyARM(i32);
impl DataGraphPipelinePropertyARM {
    pub const CREATION_LOG: Self = Self(0i32);
    pub const IDENTIFIER: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DataGraphPipelinePropertyARM {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("CREATION_LOG"),
            1i32 => f.write_str("IDENTIFIER"),
            other => write!(f, "{}({})", stringify!(DataGraphPipelinePropertyARM), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkResult")]
pub struct Result(i32);
impl Result {
    pub const SUCCESS: Self = Self(0i32);
    pub const NOT_READY: Self = Self(1i32);
    pub const TIMEOUT: Self = Self(2i32);
    pub const EVENT_SET: Self = Self(3i32);
    pub const EVENT_RESET: Self = Self(4i32);
    pub const INCOMPLETE: Self = Self(5i32);
    pub const ERROR_OUT_OF_HOST_MEMORY: Self = Self(-1i32);
    pub const ERROR_OUT_OF_DEVICE_MEMORY: Self = Self(-2i32);
    pub const ERROR_INITIALIZATION_FAILED: Self = Self(-3i32);
    pub const ERROR_DEVICE_LOST: Self = Self(-4i32);
    pub const ERROR_MEMORY_MAP_FAILED: Self = Self(-5i32);
    pub const ERROR_LAYER_NOT_PRESENT: Self = Self(-6i32);
    pub const ERROR_EXTENSION_NOT_PRESENT: Self = Self(-7i32);
    pub const ERROR_FEATURE_NOT_PRESENT: Self = Self(-8i32);
    pub const ERROR_INCOMPATIBLE_DRIVER: Self = Self(-9i32);
    pub const ERROR_TOO_MANY_OBJECTS: Self = Self(-10i32);
    pub const ERROR_FORMAT_NOT_SUPPORTED: Self = Self(-11i32);
    pub const ERROR_FRAGMENTED_POOL: Self = Self(-12i32);
    pub const ERROR_UNKNOWN: Self = Self(-13i32);
    pub const ERROR_VALIDATION_FAILED: Self = Self(-1000011001i32);
    pub const ERROR_OUT_OF_POOL_MEMORY: Self = Self(-1000069000i32);
    pub const ERROR_INVALID_EXTERNAL_HANDLE: Self = Self(-1000072003i32);
    pub const ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS: Self = Self(-1000257000i32);
    pub const ERROR_FRAGMENTATION: Self = Self(-1000161000i32);
    pub const PIPELINE_COMPILE_REQUIRED: Self = Self(1000297000i32);
    pub const ERROR_NOT_PERMITTED: Self = Self(-1000174001i32);
    pub const ERROR_INVALID_PIPELINE_CACHE_DATA: Self = Self(-1000298000i32);
    pub const ERROR_NO_PIPELINE_MATCH: Self = Self(-1000298001i32);
    pub const ERROR_SURFACE_LOST: Self = Self(-1000000000i32);
    pub const ERROR_NATIVE_WINDOW_IN_USE: Self = Self(-1000000001i32);
    pub const SUBOPTIMAL: Self = Self(1000001003i32);
    pub const ERROR_OUT_OF_DATE: Self = Self(-1000001004i32);
    pub const ERROR_INCOMPATIBLE_DISPLAY: Self = Self(-1000003001i32);
    pub const ERROR_INVALID_SHADER: Self = Self(-1000012000i32);
    pub const ERROR_IMAGE_USAGE_NOT_SUPPORTED: Self = Self(-1000023000i32);
    pub const ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED: Self = Self(-1000023001i32);
    pub const ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED: Self = Self(-1000023002i32);
    pub const ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED: Self = Self(-1000023003i32);
    pub const ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED: Self = Self(-1000023004i32);
    pub const ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED: Self = Self(-1000023005i32);
    pub const ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT: Self = Self(
        -1000158000i32,
    );
    pub const ERROR_PRESENT_TIMING_QUEUE_FULL: Self = Self(-1000208000i32);
    pub const ERROR_INVALID_DEVICE_ADDRESS: Self = Self::ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS;
    pub const ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST: Self = Self(-1000255000i32);
    pub const THREAD_IDLE: Self = Self(1000268000i32);
    pub const THREAD_DONE: Self = Self(1000268001i32);
    pub const OPERATION_DEFERRED: Self = Self(1000268002i32);
    pub const OPERATION_NOT_DEFERRED: Self = Self(1000268003i32);
    pub const ERROR_PIPELINE_COMPILE_REQUIRED: Self = Self::PIPELINE_COMPILE_REQUIRED;
    pub const ERROR_INVALID_VIDEO_STD_PARAMETERS: Self = Self(-1000299000i32);
    pub const ERROR_COMPRESSION_EXHAUSTED: Self = Self(-1000338000i32);
    pub const INCOMPATIBLE_SHADER_BINARY: Self = Self(1000482000i32);
    pub const ERROR_INCOMPATIBLE_SHADER_BINARY: Self = Self::INCOMPATIBLE_SHADER_BINARY;
    pub const PIPELINE_BINARY_MISSING: Self = Self(1000483000i32);
    pub const ERROR_NOT_ENOUGH_SPACE: Self = Self(-1000483000i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for Result {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("SUCCESS"),
            1i32 => f.write_str("NOT_READY"),
            2i32 => f.write_str("TIMEOUT"),
            3i32 => f.write_str("EVENT_SET"),
            4i32 => f.write_str("EVENT_RESET"),
            5i32 => f.write_str("INCOMPLETE"),
            -1i32 => f.write_str("ERROR_OUT_OF_HOST_MEMORY"),
            -2i32 => f.write_str("ERROR_OUT_OF_DEVICE_MEMORY"),
            -3i32 => f.write_str("ERROR_INITIALIZATION_FAILED"),
            -4i32 => f.write_str("ERROR_DEVICE_LOST"),
            -5i32 => f.write_str("ERROR_MEMORY_MAP_FAILED"),
            -6i32 => f.write_str("ERROR_LAYER_NOT_PRESENT"),
            -7i32 => f.write_str("ERROR_EXTENSION_NOT_PRESENT"),
            -8i32 => f.write_str("ERROR_FEATURE_NOT_PRESENT"),
            -9i32 => f.write_str("ERROR_INCOMPATIBLE_DRIVER"),
            -10i32 => f.write_str("ERROR_TOO_MANY_OBJECTS"),
            -11i32 => f.write_str("ERROR_FORMAT_NOT_SUPPORTED"),
            -12i32 => f.write_str("ERROR_FRAGMENTED_POOL"),
            -13i32 => f.write_str("ERROR_UNKNOWN"),
            -1000011001i32 => f.write_str("ERROR_VALIDATION_FAILED"),
            -1000069000i32 => f.write_str("ERROR_OUT_OF_POOL_MEMORY"),
            -1000072003i32 => f.write_str("ERROR_INVALID_EXTERNAL_HANDLE"),
            -1000257000i32 => f.write_str("ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS"),
            -1000161000i32 => f.write_str("ERROR_FRAGMENTATION"),
            1000297000i32 => f.write_str("PIPELINE_COMPILE_REQUIRED"),
            -1000174001i32 => f.write_str("ERROR_NOT_PERMITTED"),
            -1000298000i32 => f.write_str("ERROR_INVALID_PIPELINE_CACHE_DATA"),
            -1000298001i32 => f.write_str("ERROR_NO_PIPELINE_MATCH"),
            -1000000000i32 => f.write_str("ERROR_SURFACE_LOST"),
            -1000000001i32 => f.write_str("ERROR_NATIVE_WINDOW_IN_USE"),
            1000001003i32 => f.write_str("SUBOPTIMAL"),
            -1000001004i32 => f.write_str("ERROR_OUT_OF_DATE"),
            -1000003001i32 => f.write_str("ERROR_INCOMPATIBLE_DISPLAY"),
            -1000012000i32 => f.write_str("ERROR_INVALID_SHADER"),
            -1000023000i32 => f.write_str("ERROR_IMAGE_USAGE_NOT_SUPPORTED"),
            -1000023001i32 => f.write_str("ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED"),
            -1000023002i32 => f.write_str("ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED"),
            -1000023003i32 => f.write_str("ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED"),
            -1000023004i32 => f.write_str("ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED"),
            -1000023005i32 => f.write_str("ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED"),
            -1000158000i32 => {
                f.write_str("ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT")
            }
            -1000208000i32 => f.write_str("ERROR_PRESENT_TIMING_QUEUE_FULL"),
            -1000255000i32 => f.write_str("ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST"),
            1000268000i32 => f.write_str("THREAD_IDLE"),
            1000268001i32 => f.write_str("THREAD_DONE"),
            1000268002i32 => f.write_str("OPERATION_DEFERRED"),
            1000268003i32 => f.write_str("OPERATION_NOT_DEFERRED"),
            -1000299000i32 => f.write_str("ERROR_INVALID_VIDEO_STD_PARAMETERS"),
            -1000338000i32 => f.write_str("ERROR_COMPRESSION_EXHAUSTED"),
            1000482000i32 => f.write_str("INCOMPATIBLE_SHADER_BINARY"),
            1000483000i32 => f.write_str("PIPELINE_BINARY_MISSING"),
            -1000483000i32 => f.write_str("ERROR_NOT_ENOUGH_SPACE"),
            other => write!(f, "{}({})", stringify!(Result), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkVideoEncodeAV1PredictionModeKHR")]
pub struct VideoEncodeAV1PredictionModeKHR(i32);
impl VideoEncodeAV1PredictionModeKHR {
    pub const INTRA_ONLY: Self = Self(0i32);
    pub const SINGLE_REFERENCE: Self = Self(1i32);
    pub const UNIDIRECTIONAL_COMPOUND: Self = Self(2i32);
    pub const BIDIRECTIONAL_COMPOUND: Self = Self(3i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for VideoEncodeAV1PredictionModeKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("INTRA_ONLY"),
            1i32 => f.write_str("SINGLE_REFERENCE"),
            2i32 => f.write_str("UNIDIRECTIONAL_COMPOUND"),
            3i32 => f.write_str("BIDIRECTIONAL_COMPOUND"),
            other => {
                write!(f, "{}({})", stringify!(VideoEncodeAV1PredictionModeKHR), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkIndirectCommandsTokenTypeEXT")]
pub struct IndirectCommandsTokenTypeEXT(i32);
impl IndirectCommandsTokenTypeEXT {
    pub const EXECUTION_SET: Self = Self(0i32);
    pub const PUSH_CONSTANT: Self = Self(1i32);
    pub const SEQUENCE_INDEX: Self = Self(2i32);
    pub const INDEX_BUFFER: Self = Self(3i32);
    pub const VERTEX_BUFFER: Self = Self(4i32);
    pub const DRAW_INDEXED: Self = Self(5i32);
    pub const DRAW: Self = Self(6i32);
    pub const DRAW_INDEXED_COUNT: Self = Self(7i32);
    pub const DRAW_COUNT: Self = Self(8i32);
    pub const DISPATCH: Self = Self(9i32);
    pub const PUSH_DATA: Self = Self(1000135000i32);
    pub const PUSH_DATA_SEQUENCE_INDEX: Self = Self(1000135001i32);
    pub const DRAW_MESH_TASKS_: Self = Self(1000202002i32);
    pub const DRAW_MESH_TASKS_COUNT_: Self = Self(1000202003i32);
    pub const DRAW_MESH_TASKS: Self = Self(1000328000i32);
    pub const DRAW_MESH_TASKS_COUNT: Self = Self(1000328001i32);
    pub const TRACE_RAYS2: Self = Self(1000386004i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for IndirectCommandsTokenTypeEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("EXECUTION_SET"),
            1i32 => f.write_str("PUSH_CONSTANT"),
            2i32 => f.write_str("SEQUENCE_INDEX"),
            3i32 => f.write_str("INDEX_BUFFER"),
            4i32 => f.write_str("VERTEX_BUFFER"),
            5i32 => f.write_str("DRAW_INDEXED"),
            6i32 => f.write_str("DRAW"),
            7i32 => f.write_str("DRAW_INDEXED_COUNT"),
            8i32 => f.write_str("DRAW_COUNT"),
            9i32 => f.write_str("DISPATCH"),
            1000135000i32 => f.write_str("PUSH_DATA"),
            1000135001i32 => f.write_str("PUSH_DATA_SEQUENCE_INDEX"),
            1000202002i32 => f.write_str("DRAW_MESH_TASKS_"),
            1000202003i32 => f.write_str("DRAW_MESH_TASKS_COUNT_"),
            1000328000i32 => f.write_str("DRAW_MESH_TASKS"),
            1000328001i32 => f.write_str("DRAW_MESH_TASKS_COUNT"),
            1000386004i32 => f.write_str("TRACE_RAYS2"),
            other => write!(f, "{}({})", stringify!(IndirectCommandsTokenTypeEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkConservativeRasterizationModeEXT")]
pub struct ConservativeRasterizationModeEXT(i32);
impl ConservativeRasterizationModeEXT {
    pub const DISABLED: Self = Self(0i32);
    pub const OVERESTIMATE: Self = Self(1i32);
    pub const UNDERESTIMATE: Self = Self(2i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ConservativeRasterizationModeEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("DISABLED"),
            1i32 => f.write_str("OVERESTIMATE"),
            2i32 => f.write_str("UNDERESTIMATE"),
            other => {
                write!(f, "{}({})", stringify!(ConservativeRasterizationModeEXT), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkRayTracingLssIndexingModeNV")]
pub struct RayTracingLssIndexingModeNV(i32);
impl RayTracingLssIndexingModeNV {
    pub const LIST: Self = Self(0i32);
    pub const SUCCESSIVE: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for RayTracingLssIndexingModeNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("LIST"),
            1i32 => f.write_str("SUCCESSIVE"),
            other => write!(f, "{}({})", stringify!(RayTracingLssIndexingModeNV), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPartitionedAccelerationStructureOpTypeNV")]
pub struct PartitionedAccelerationStructureOpTypeNV(i32);
impl PartitionedAccelerationStructureOpTypeNV {
    pub const WRITE_INSTANCE: Self = Self(0i32);
    pub const UPDATE_INSTANCE: Self = Self(1i32);
    pub const WRITE_PARTITION_TRANSLATION: Self = Self(2i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PartitionedAccelerationStructureOpTypeNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("WRITE_INSTANCE"),
            1i32 => f.write_str("UPDATE_INSTANCE"),
            2i32 => f.write_str("WRITE_PARTITION_TRANSLATION"),
            other => {
                write!(
                    f, "{}({})", stringify!(PartitionedAccelerationStructureOpTypeNV),
                    other
                )
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDescriptorMappingSourceEXT")]
pub struct DescriptorMappingSourceEXT(i32);
impl DescriptorMappingSourceEXT {
    pub const HEAP_WITH_CONSTANT_OFFSET: Self = Self(0i32);
    pub const HEAP_WITH_PUSH_INDEX: Self = Self(1i32);
    pub const HEAP_WITH_INDIRECT_INDEX: Self = Self(2i32);
    pub const HEAP_WITH_INDIRECT_INDEX_ARRAY: Self = Self(3i32);
    pub const RESOURCE_HEAP_DATA: Self = Self(4i32);
    pub const PUSH_DATA: Self = Self(5i32);
    pub const PUSH_ADDRESS: Self = Self(6i32);
    pub const INDIRECT_ADDRESS: Self = Self(7i32);
    pub const HEAP_WITH_SHADER_RECORD_INDEX: Self = Self(8i32);
    pub const SHADER_RECORD_DATA: Self = Self(9i32);
    pub const SHADER_RECORD_ADDRESS: Self = Self(10i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DescriptorMappingSourceEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("HEAP_WITH_CONSTANT_OFFSET"),
            1i32 => f.write_str("HEAP_WITH_PUSH_INDEX"),
            2i32 => f.write_str("HEAP_WITH_INDIRECT_INDEX"),
            3i32 => f.write_str("HEAP_WITH_INDIRECT_INDEX_ARRAY"),
            4i32 => f.write_str("RESOURCE_HEAP_DATA"),
            5i32 => f.write_str("PUSH_DATA"),
            6i32 => f.write_str("PUSH_ADDRESS"),
            7i32 => f.write_str("INDIRECT_ADDRESS"),
            8i32 => f.write_str("HEAP_WITH_SHADER_RECORD_INDEX"),
            9i32 => f.write_str("SHADER_RECORD_DATA"),
            10i32 => f.write_str("SHADER_RECORD_ADDRESS"),
            other => write!(f, "{}({})", stringify!(DescriptorMappingSourceEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkTensorTilingARM")]
pub struct TensorTilingARM(i32);
impl TensorTilingARM {
    pub const OPTIMAL: Self = Self(0i32);
    pub const LINEAR: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for TensorTilingARM {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("OPTIMAL"),
            1i32 => f.write_str("LINEAR"),
            other => write!(f, "{}({})", stringify!(TensorTilingARM), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkShaderCodeTypeEXT")]
pub struct ShaderCodeTypeEXT(i32);
impl ShaderCodeTypeEXT {
    pub const BINARY: Self = Self(0i32);
    pub const SPIRV: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ShaderCodeTypeEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("BINARY"),
            1i32 => f.write_str("SPIRV"),
            other => write!(f, "{}({})", stringify!(ShaderCodeTypeEXT), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDeviceMemoryReportEventTypeEXT")]
pub struct DeviceMemoryReportEventTypeEXT(i32);
impl DeviceMemoryReportEventTypeEXT {
    pub const ALLOCATE: Self = Self(0i32);
    pub const FREE: Self = Self(1i32);
    pub const IMPORT: Self = Self(2i32);
    pub const UNIMPORT: Self = Self(3i32);
    pub const ALLOCATION_FAILED: Self = Self(4i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DeviceMemoryReportEventTypeEXT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("ALLOCATE"),
            1i32 => f.write_str("FREE"),
            2i32 => f.write_str("IMPORT"),
            3i32 => f.write_str("UNIMPORT"),
            4i32 => f.write_str("ALLOCATION_FAILED"),
            other => {
                write!(f, "{}({})", stringify!(DeviceMemoryReportEventTypeEXT), other)
            }
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkAntiLagModeAMD")]
pub struct AntiLagModeAMD(i32);
impl AntiLagModeAMD {
    pub const DRIVER_CONTROL: Self = Self(0i32);
    pub const ON: Self = Self(1i32);
    pub const OFF: Self = Self(2i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for AntiLagModeAMD {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("DRIVER_CONTROL"),
            1i32 => f.write_str("ON"),
            2i32 => f.write_str("OFF"),
            other => write!(f, "{}({})", stringify!(AntiLagModeAMD), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPipelineBindPoint")]
pub struct PipelineBindPoint(i32);
impl PipelineBindPoint {
    pub const GRAPHICS: Self = Self(0i32);
    pub const COMPUTE: Self = Self(1i32);
    pub const EXECUTION_GRAPH: Self = Self(1000134000i32);
    pub const RAY_TRACING: Self = Self(1000165000i32);
    pub const SUBPASS_SHADING: Self = Self(1000369003i32);
    pub const DATA_GRAPH: Self = Self(1000507000i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PipelineBindPoint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("GRAPHICS"),
            1i32 => f.write_str("COMPUTE"),
            1000134000i32 => f.write_str("EXECUTION_GRAPH"),
            1000165000i32 => f.write_str("RAY_TRACING"),
            1000369003i32 => f.write_str("SUBPASS_SHADING"),
            1000507000i32 => f.write_str("DATA_GRAPH"),
            other => write!(f, "{}({})", stringify!(PipelineBindPoint), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkSamplerReductionMode")]
pub struct SamplerReductionMode(i32);
impl SamplerReductionMode {
    pub const WEIGHTED_AVERAGE: Self = Self(0i32);
    pub const MIN: Self = Self(1i32);
    pub const MAX: Self = Self(2i32);
    pub const WEIGHTED_AVERAGE_RANGECLAMP: Self = Self(1000521000i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for SamplerReductionMode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("WEIGHTED_AVERAGE"),
            1i32 => f.write_str("MIN"),
            2i32 => f.write_str("MAX"),
            1000521000i32 => f.write_str("WEIGHTED_AVERAGE_RANGECLAMP"),
            other => write!(f, "{}({})", stringify!(SamplerReductionMode), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkPhysicalDeviceLayeredApiKHR")]
pub struct PhysicalDeviceLayeredApiKHR(i32);
impl PhysicalDeviceLayeredApiKHR {
    pub const VULKAN: Self = Self(0i32);
    pub const D3D12: Self = Self(1i32);
    pub const METAL: Self = Self(2i32);
    pub const OPENGL: Self = Self(3i32);
    pub const OPENGLES: Self = Self(4i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for PhysicalDeviceLayeredApiKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("VULKAN"),
            1i32 => f.write_str("D3D12"),
            2i32 => f.write_str("METAL"),
            3i32 => f.write_str("OPENGL"),
            4i32 => f.write_str("OPENGLES"),
            other => write!(f, "{}({})", stringify!(PhysicalDeviceLayeredApiKHR), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkShadingRatePaletteEntryNV")]
pub struct ShadingRatePaletteEntryNV(i32);
impl ShadingRatePaletteEntryNV {
    pub const NO_INVOCATIONS: Self = Self(0i32);
    pub const _16_INVOCATIONS_PER_PIXEL: Self = Self(1i32);
    pub const _8_INVOCATIONS_PER_PIXEL: Self = Self(2i32);
    pub const _4_INVOCATIONS_PER_PIXEL: Self = Self(3i32);
    pub const _2_INVOCATIONS_PER_PIXEL: Self = Self(4i32);
    pub const _1_INVOCATION_PER_PIXEL: Self = Self(5i32);
    pub const _1_INVOCATION_PER_2X1_PIXELS: Self = Self(6i32);
    pub const _1_INVOCATION_PER_1X2_PIXELS: Self = Self(7i32);
    pub const _1_INVOCATION_PER_2X2_PIXELS: Self = Self(8i32);
    pub const _1_INVOCATION_PER_4X2_PIXELS: Self = Self(9i32);
    pub const _1_INVOCATION_PER_2X4_PIXELS: Self = Self(10i32);
    pub const _1_INVOCATION_PER_4X4_PIXELS: Self = Self(11i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for ShadingRatePaletteEntryNV {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("NO_INVOCATIONS"),
            1i32 => f.write_str("_16_INVOCATIONS_PER_PIXEL"),
            2i32 => f.write_str("_8_INVOCATIONS_PER_PIXEL"),
            3i32 => f.write_str("_4_INVOCATIONS_PER_PIXEL"),
            4i32 => f.write_str("_2_INVOCATIONS_PER_PIXEL"),
            5i32 => f.write_str("_1_INVOCATION_PER_PIXEL"),
            6i32 => f.write_str("_1_INVOCATION_PER_2X1_PIXELS"),
            7i32 => f.write_str("_1_INVOCATION_PER_1X2_PIXELS"),
            8i32 => f.write_str("_1_INVOCATION_PER_2X2_PIXELS"),
            9i32 => f.write_str("_1_INVOCATION_PER_4X2_PIXELS"),
            10i32 => f.write_str("_1_INVOCATION_PER_2X4_PIXELS"),
            11i32 => f.write_str("_1_INVOCATION_PER_4X4_PIXELS"),
            other => write!(f, "{}({})", stringify!(ShadingRatePaletteEntryNV), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDeviceFaultAddressTypeKHR")]
pub struct DeviceFaultAddressTypeKHR(i32);
impl DeviceFaultAddressTypeKHR {
    pub const NONE: Self = Self(0i32);
    pub const READ_INVALID: Self = Self(1i32);
    pub const WRITE_INVALID: Self = Self(2i32);
    pub const EXECUTE_INVALID: Self = Self(3i32);
    pub const INSTRUCTION_POINTER_UNKNOWN: Self = Self(4i32);
    pub const INSTRUCTION_POINTER_INVALID: Self = Self(5i32);
    pub const INSTRUCTION_POINTER_FAULT: Self = Self(6i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DeviceFaultAddressTypeKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("NONE"),
            1i32 => f.write_str("READ_INVALID"),
            2i32 => f.write_str("WRITE_INVALID"),
            3i32 => f.write_str("EXECUTE_INVALID"),
            4i32 => f.write_str("INSTRUCTION_POINTER_UNKNOWN"),
            5i32 => f.write_str("INSTRUCTION_POINTER_INVALID"),
            6i32 => f.write_str("INSTRUCTION_POINTER_FAULT"),
            other => write!(f, "{}({})", stringify!(DeviceFaultAddressTypeKHR), other),
        }
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[doc(alias = "VkDescriptorUpdateTemplateType")]
pub struct DescriptorUpdateTemplateType(i32);
impl DescriptorUpdateTemplateType {
    pub const DESCRIPTOR_SET: Self = Self(0i32);
    pub const PUSH_DESCRIPTORS: Self = Self(1i32);
    #[inline]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}
impl core::fmt::Debug for DescriptorUpdateTemplateType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0i32 => f.write_str("DESCRIPTOR_SET"),
            1i32 => f.write_str("PUSH_DESCRIPTORS"),
            other => write!(f, "{}({})", stringify!(DescriptorUpdateTemplateType), other),
        }
    }
}
