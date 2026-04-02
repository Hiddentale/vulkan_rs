# Usage Notes

Queries the memory bind point requirements for a data graph
pipeline session. Uses the two-call idiom. Each returned
requirement describes a bind point that must be satisfied with
`bind_data_graph_pipeline_session_memory_arm` before dispatch.

Requires `VK_ARM_data_graph`.
