# Usage Notes

Queries the memory requirements for a video session. A video
session may require multiple memory bindings (each with different
memory type requirements) for internal buffers used during
decode/encode.

Each returned `VideoSessionMemoryRequirementsKHR` has a
`memory_bind_index` and a `MemoryRequirements` describing the
size, alignment, and compatible memory types.

Allocate a `DeviceMemory` for each requirement and bind them all
with `bind_video_session_memory_khr` before using the session.
