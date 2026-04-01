# Usage Notes

Queries the scratch memory size required to execute an execution
graph pipeline. Allocate a buffer of at least this size and
initialize it with `cmd_initialize_graph_scratch_memory_amdx`
before dispatching the graph.

Requires `VK_AMDX_shader_enqueue`.
