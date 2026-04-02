# Usage Notes

Initializes the scratch memory buffer for an execution graph
pipeline. Must be called before any `cmd_dispatch_graph_*_amdx`
command that uses this scratch buffer.

Requires `VK_AMDX_shader_enqueue`.
