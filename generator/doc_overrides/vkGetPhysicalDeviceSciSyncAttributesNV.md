# Usage Notes

Fills an NvSciSync attribute list with Vulkan's requirements for
synchronisation. Used to negotiate sync object attributes when
sharing synchronisation primitives between Vulkan and other NvSciSync
consumers (e.g. camera, display, or compute pipelines).

This is a platform-specific extension for NVIDIA's Safety Critical
ecosystem. Not available on desktop or mobile platforms.
