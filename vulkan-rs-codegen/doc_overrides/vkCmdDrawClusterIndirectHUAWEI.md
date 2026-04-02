# Usage Notes

Indirect variant of `cmd_draw_cluster_huawei`. Reads the cluster
dispatch parameters from a buffer at the given offset, allowing
GPU-driven cluster culling without CPU readback.

Requires `VK_HUAWEI_cluster_culling_shader`.
