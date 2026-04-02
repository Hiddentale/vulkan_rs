# Usage Notes

Returns a function pointer for an instance-level or device-level
Vulkan command. This is the root of the Vulkan function pointer
loading chain.

In normal usage you do not need to call this yourself, `Instance`
and `Device` load all function pointers automatically. This is
primarily useful for:

- Loading commands that are not yet exposed as wrapper methods.
- Raw interop with other Vulkan libraries (e.g. OpenXR).
- Implementing custom loaders.

When called with a null instance, returns pointers for global
commands (`vkEnumerateInstanceVersion`,
`vkEnumerateInstanceExtensionProperties`, etc.).

The returned pointer may go through a loader trampoline. For
device-level commands, `get_device_proc_addr` returns a
driver-direct pointer that is slightly faster.
