# Usage Notes

Vulkan 1.4 version of `cmd_bind_descriptor_sets` that uses an
extensible `BindDescriptorSetsInfo` struct.

Functionally equivalent to the 1.0 version. The extensible struct
enables future extensions to modify binding behaviour via pNext.

Prefer this when targeting Vulkan 1.4+.
