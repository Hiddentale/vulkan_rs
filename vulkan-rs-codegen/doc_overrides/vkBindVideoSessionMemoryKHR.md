# Usage Notes

Binds device memory to a video session. Each binding corresponds
to a `memory_bind_index` from
`get_video_session_memory_requirements_khr`.

All required memory bindings must be satisfied before the session
can be used in video coding operations. Each
`BindVideoSessionMemoryInfoKHR` specifies the bind index, memory
object, offset, and size.

Memory can only be bound once per index, rebinding is not
allowed.
