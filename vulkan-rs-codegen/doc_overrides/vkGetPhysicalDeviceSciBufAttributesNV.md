# Usage Notes

Fills an NvSciBuf attribute list with Vulkan's requirements for
a given image or buffer creation info. Used to negotiate buffer
attributes when sharing memory between Vulkan and other NvSciBuf
consumers.

This is a platform-specific extension for NVIDIA's Safety Critical
ecosystem. Not available on desktop or mobile platforms.
