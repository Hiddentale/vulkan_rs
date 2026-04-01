# Usage Notes

GPU-side acceleration structure build with indirect parameters. The
primitive counts and build ranges are read from GPU buffers rather
than specified on the CPU.

This enables fully GPU-driven scene management where a compute
shader determines which geometry to include and writes the build
parameters.

Requires the `acceleration_structure_indirect_build` feature.
