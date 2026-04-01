# Usage Notes

Enumerates the cooperative matrix types and configurations
supported by the physical device. Each returned
`CooperativeMatrixPropertiesKHR` describes a supported combination
of matrix dimensions (M, N, K), component types (A, B, C, Result),
and scope (subgroup or workgroup).

Use these results to select valid cooperative matrix parameters
for SPIR-V `OpCooperativeMatrixMulAddKHR` operations.

Requires `VK_KHR_cooperative_matrix`.
