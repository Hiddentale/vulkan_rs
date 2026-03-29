/// Trait implemented by all Vulkan handle types.
pub trait Handle: Copy + Eq + std::hash::Hash {
    /// The raw representation type (`usize` for dispatchable, `u64` for non-dispatchable).
    type Repr;
    /// Returns the null handle.
    fn null() -> Self;
    /// Constructs a handle from its raw representation.
    fn from_raw(raw: Self::Repr) -> Self;
    /// Returns the raw representation.
    fn as_raw(self) -> Self::Repr;
    /// Returns `true` if this is the null handle.
    fn is_null(self) -> bool;
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkInstance")]
pub struct Instance(usize);
impl Handle for Instance {
    type Repr = usize;
    #[inline]
    fn null() -> Self {
        Self(0usize)
    }
    #[inline]
    fn from_raw(raw: usize) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> usize {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0usize
    }
}
impl Default for Instance {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(Instance), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkPhysicalDevice")]
pub struct PhysicalDevice(usize);
impl Handle for PhysicalDevice {
    type Repr = usize;
    #[inline]
    fn null() -> Self {
        Self(0usize)
    }
    #[inline]
    fn from_raw(raw: usize) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> usize {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0usize
    }
}
impl Default for PhysicalDevice {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for PhysicalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(PhysicalDevice), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkDevice")]
pub struct Device(usize);
impl Handle for Device {
    type Repr = usize;
    #[inline]
    fn null() -> Self {
        Self(0usize)
    }
    #[inline]
    fn from_raw(raw: usize) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> usize {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0usize
    }
}
impl Default for Device {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(Device), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkQueue")]
pub struct Queue(usize);
impl Handle for Queue {
    type Repr = usize;
    #[inline]
    fn null() -> Self {
        Self(0usize)
    }
    #[inline]
    fn from_raw(raw: usize) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> usize {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0usize
    }
}
impl Default for Queue {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for Queue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(Queue), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkCommandBuffer")]
pub struct CommandBuffer(usize);
impl Handle for CommandBuffer {
    type Repr = usize;
    #[inline]
    fn null() -> Self {
        Self(0usize)
    }
    #[inline]
    fn from_raw(raw: usize) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> usize {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0usize
    }
}
impl Default for CommandBuffer {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for CommandBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(CommandBuffer), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkDeviceMemory")]
pub struct DeviceMemory(u64);
impl Handle for DeviceMemory {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for DeviceMemory {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for DeviceMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(DeviceMemory), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkCommandPool")]
pub struct CommandPool(u64);
impl Handle for CommandPool {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for CommandPool {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for CommandPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(CommandPool), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkBuffer")]
pub struct Buffer(u64);
impl Handle for Buffer {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for Buffer {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(Buffer), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkBufferView")]
pub struct BufferView(u64);
impl Handle for BufferView {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for BufferView {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for BufferView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(BufferView), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkImage")]
pub struct Image(u64);
impl Handle for Image {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for Image {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(Image), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkImageView")]
pub struct ImageView(u64);
impl Handle for ImageView {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for ImageView {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for ImageView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(ImageView), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkShaderModule")]
pub struct ShaderModule(u64);
impl Handle for ShaderModule {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for ShaderModule {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for ShaderModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(ShaderModule), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkPipeline")]
pub struct Pipeline(u64);
impl Handle for Pipeline {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for Pipeline {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(Pipeline), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkPipelineLayout")]
pub struct PipelineLayout(u64);
impl Handle for PipelineLayout {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for PipelineLayout {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for PipelineLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(PipelineLayout), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkSampler")]
pub struct Sampler(u64);
impl Handle for Sampler {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for Sampler {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for Sampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(Sampler), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkDescriptorSet")]
pub struct DescriptorSet(u64);
impl Handle for DescriptorSet {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for DescriptorSet {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for DescriptorSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(DescriptorSet), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkDescriptorSetLayout")]
pub struct DescriptorSetLayout(u64);
impl Handle for DescriptorSetLayout {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for DescriptorSetLayout {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for DescriptorSetLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(DescriptorSetLayout), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkDescriptorPool")]
pub struct DescriptorPool(u64);
impl Handle for DescriptorPool {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for DescriptorPool {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for DescriptorPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(DescriptorPool), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkFence")]
pub struct Fence(u64);
impl Handle for Fence {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for Fence {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for Fence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(Fence), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkSemaphore")]
pub struct Semaphore(u64);
impl Handle for Semaphore {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for Semaphore {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for Semaphore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(Semaphore), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkEvent")]
pub struct Event(u64);
impl Handle for Event {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for Event {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(Event), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkQueryPool")]
pub struct QueryPool(u64);
impl Handle for QueryPool {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for QueryPool {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for QueryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(QueryPool), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkFramebuffer")]
pub struct Framebuffer(u64);
impl Handle for Framebuffer {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for Framebuffer {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for Framebuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(Framebuffer), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkRenderPass")]
pub struct RenderPass(u64);
impl Handle for RenderPass {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for RenderPass {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for RenderPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(RenderPass), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkPipelineCache")]
pub struct PipelineCache(u64);
impl Handle for PipelineCache {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for PipelineCache {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for PipelineCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(PipelineCache), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkPipelineBinaryKHR")]
pub struct PipelineBinaryKHR(u64);
impl Handle for PipelineBinaryKHR {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for PipelineBinaryKHR {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for PipelineBinaryKHR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(PipelineBinaryKHR), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkIndirectCommandsLayoutNV")]
pub struct IndirectCommandsLayoutNV(u64);
impl Handle for IndirectCommandsLayoutNV {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for IndirectCommandsLayoutNV {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for IndirectCommandsLayoutNV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(IndirectCommandsLayoutNV), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkIndirectCommandsLayoutEXT")]
pub struct IndirectCommandsLayoutEXT(u64);
impl Handle for IndirectCommandsLayoutEXT {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for IndirectCommandsLayoutEXT {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for IndirectCommandsLayoutEXT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(IndirectCommandsLayoutEXT), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkIndirectExecutionSetEXT")]
pub struct IndirectExecutionSetEXT(u64);
impl Handle for IndirectExecutionSetEXT {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for IndirectExecutionSetEXT {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for IndirectExecutionSetEXT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(IndirectExecutionSetEXT), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkDescriptorUpdateTemplate")]
pub struct DescriptorUpdateTemplate(u64);
impl Handle for DescriptorUpdateTemplate {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for DescriptorUpdateTemplate {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for DescriptorUpdateTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(DescriptorUpdateTemplate), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkSamplerYcbcrConversion")]
pub struct SamplerYcbcrConversion(u64);
impl Handle for SamplerYcbcrConversion {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for SamplerYcbcrConversion {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for SamplerYcbcrConversion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(SamplerYcbcrConversion), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkValidationCacheEXT")]
pub struct ValidationCacheEXT(u64);
impl Handle for ValidationCacheEXT {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for ValidationCacheEXT {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for ValidationCacheEXT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(ValidationCacheEXT), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkAccelerationStructureKHR")]
pub struct AccelerationStructureKHR(u64);
impl Handle for AccelerationStructureKHR {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for AccelerationStructureKHR {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for AccelerationStructureKHR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(AccelerationStructureKHR), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkAccelerationStructureNV")]
pub struct AccelerationStructureNV(u64);
impl Handle for AccelerationStructureNV {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for AccelerationStructureNV {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for AccelerationStructureNV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(AccelerationStructureNV), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkPerformanceConfigurationINTEL")]
pub struct PerformanceConfigurationINTEL(u64);
impl Handle for PerformanceConfigurationINTEL {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for PerformanceConfigurationINTEL {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for PerformanceConfigurationINTEL {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(PerformanceConfigurationINTEL), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkBufferCollectionFUCHSIA")]
pub struct BufferCollectionFUCHSIA(u64);
impl Handle for BufferCollectionFUCHSIA {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for BufferCollectionFUCHSIA {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for BufferCollectionFUCHSIA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(BufferCollectionFUCHSIA), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkDeferredOperationKHR")]
pub struct DeferredOperationKHR(u64);
impl Handle for DeferredOperationKHR {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for DeferredOperationKHR {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for DeferredOperationKHR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(DeferredOperationKHR), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkPrivateDataSlot")]
pub struct PrivateDataSlot(u64);
impl Handle for PrivateDataSlot {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for PrivateDataSlot {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for PrivateDataSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(PrivateDataSlot), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkCuModuleNVX")]
pub struct CuModuleNVX(u64);
impl Handle for CuModuleNVX {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for CuModuleNVX {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for CuModuleNVX {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(CuModuleNVX), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkCuFunctionNVX")]
pub struct CuFunctionNVX(u64);
impl Handle for CuFunctionNVX {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for CuFunctionNVX {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for CuFunctionNVX {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(CuFunctionNVX), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkOpticalFlowSessionNV")]
pub struct OpticalFlowSessionNV(u64);
impl Handle for OpticalFlowSessionNV {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for OpticalFlowSessionNV {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for OpticalFlowSessionNV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(OpticalFlowSessionNV), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkMicromapEXT")]
pub struct MicromapEXT(u64);
impl Handle for MicromapEXT {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for MicromapEXT {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for MicromapEXT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(MicromapEXT), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkShaderEXT")]
pub struct ShaderEXT(u64);
impl Handle for ShaderEXT {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for ShaderEXT {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for ShaderEXT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(ShaderEXT), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkTensorARM")]
pub struct TensorARM(u64);
impl Handle for TensorARM {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for TensorARM {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for TensorARM {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(TensorARM), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkTensorViewARM")]
pub struct TensorViewARM(u64);
impl Handle for TensorViewARM {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for TensorViewARM {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for TensorViewARM {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(TensorViewARM), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkDataGraphPipelineSessionARM")]
pub struct DataGraphPipelineSessionARM(u64);
impl Handle for DataGraphPipelineSessionARM {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for DataGraphPipelineSessionARM {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for DataGraphPipelineSessionARM {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(DataGraphPipelineSessionARM), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkShaderInstrumentationARM")]
pub struct ShaderInstrumentationARM(u64);
impl Handle for ShaderInstrumentationARM {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for ShaderInstrumentationARM {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for ShaderInstrumentationARM {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(ShaderInstrumentationARM), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkDisplayKHR")]
pub struct DisplayKHR(u64);
impl Handle for DisplayKHR {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for DisplayKHR {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for DisplayKHR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(DisplayKHR), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkDisplayModeKHR")]
pub struct DisplayModeKHR(u64);
impl Handle for DisplayModeKHR {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for DisplayModeKHR {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for DisplayModeKHR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(DisplayModeKHR), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkSurfaceKHR")]
pub struct SurfaceKHR(u64);
impl Handle for SurfaceKHR {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for SurfaceKHR {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for SurfaceKHR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(SurfaceKHR), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkSwapchainKHR")]
pub struct SwapchainKHR(u64);
impl Handle for SwapchainKHR {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for SwapchainKHR {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for SwapchainKHR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(SwapchainKHR), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkDebugReportCallbackEXT")]
pub struct DebugReportCallbackEXT(u64);
impl Handle for DebugReportCallbackEXT {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for DebugReportCallbackEXT {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for DebugReportCallbackEXT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(DebugReportCallbackEXT), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkDebugUtilsMessengerEXT")]
pub struct DebugUtilsMessengerEXT(u64);
impl Handle for DebugUtilsMessengerEXT {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for DebugUtilsMessengerEXT {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for DebugUtilsMessengerEXT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(DebugUtilsMessengerEXT), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkVideoSessionKHR")]
pub struct VideoSessionKHR(u64);
impl Handle for VideoSessionKHR {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for VideoSessionKHR {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for VideoSessionKHR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(VideoSessionKHR), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkVideoSessionParametersKHR")]
pub struct VideoSessionParametersKHR(u64);
impl Handle for VideoSessionParametersKHR {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for VideoSessionParametersKHR {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for VideoSessionParametersKHR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(VideoSessionParametersKHR), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkSemaphoreSciSyncPoolNV")]
pub struct SemaphoreSciSyncPoolNV(u64);
impl Handle for SemaphoreSciSyncPoolNV {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for SemaphoreSciSyncPoolNV {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for SemaphoreSciSyncPoolNV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(SemaphoreSciSyncPoolNV), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkCudaModuleNV")]
pub struct CudaModuleNV(u64);
impl Handle for CudaModuleNV {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for CudaModuleNV {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for CudaModuleNV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(CudaModuleNV), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkCudaFunctionNV")]
pub struct CudaFunctionNV(u64);
impl Handle for CudaFunctionNV {
    type Repr = u64;
    #[inline]
    fn null() -> Self {
        Self(0u64)
    }
    #[inline]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> u64 {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0u64
    }
}
impl Default for CudaFunctionNV {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for CudaFunctionNV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(CudaFunctionNV), self.0)
    }
}
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[doc(alias = "VkExternalComputeQueueNV")]
pub struct ExternalComputeQueueNV(usize);
impl Handle for ExternalComputeQueueNV {
    type Repr = usize;
    #[inline]
    fn null() -> Self {
        Self(0usize)
    }
    #[inline]
    fn from_raw(raw: usize) -> Self {
        Self(raw)
    }
    #[inline]
    fn as_raw(self) -> usize {
        self.0
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0 == 0usize
    }
}
impl Default for ExternalComputeQueueNV {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}
impl std::fmt::Debug for ExternalComputeQueueNV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#x})", stringify!(ExternalComputeQueueNV), self.0)
    }
}
