# Usage Notes

Binds tile memory for use within a per-tile execution region on
Qualcomm tile-based GPUs. Pass `None` to unbind. Tile memory
provides fast on-chip scratch storage scoped to each tile.

Requires `VK_QCOM_tile_shading`.
