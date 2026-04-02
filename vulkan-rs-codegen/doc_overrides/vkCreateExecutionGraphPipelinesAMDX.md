# Usage Notes

Creates one or more execution graph pipelines for GPU-driven
shader dispatch. An execution graph is a DAG of shader nodes
where each node can enqueue work to other nodes, enabling complex
GPU-driven workflows without CPU round-trips.

Requires `VK_AMDX_shader_enqueue`.
