# Usage Notes

Fully indirect variant of `cmd_dispatch_graph_amdx`. Both the
dispatch payloads and the count are read from GPU buffers,
enabling fully GPU-driven execution graph dispatch.

Requires `VK_AMDX_shader_enqueue`.
