# Usage Notes

Retrieves internal representations (IR) of a pipeline executable.
These are driver-specific intermediate or final shader
representations, for example, SPIR-V, vendor IR, or GPU ISA
disassembly.

Each representation has a name, description, and opaque data
blob. Whether the data is human-readable text or binary depends
on `is_text` in the returned structure.

The pipeline must have been created with
`PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS`. Enabling
this flag may disable optimizations, so only use it for
debugging and shader analysis, not in production.

Identify the executable by index from
`get_pipeline_executable_properties_khr`.
