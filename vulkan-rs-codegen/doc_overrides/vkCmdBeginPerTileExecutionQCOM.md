# Usage Notes

Begins a per-tile execution region within a render pass on
Qualcomm tile-based GPUs. Commands recorded between this and
`cmd_end_per_tile_execution_qcom` are executed once per tile,
enabling tile-local compute and shading optimisations.

Requires `VK_QCOM_tile_shading`.
