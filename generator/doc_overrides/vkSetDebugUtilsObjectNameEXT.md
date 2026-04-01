# Usage Notes

Assigns a human-readable name to any Vulkan object. The name
appears in validation layer messages, GPU debuggers (RenderDoc,
Nsight), and crash reports.

`DebugUtilsObjectNameInfoEXT` specifies the object type, handle,
and a null-terminated UTF-8 name string.

Set the name to null to remove a previously assigned name.

This is the most impactful debugging tool in Vulkan, name
every object you create.

Requires `VK_EXT_debug_utils`.
