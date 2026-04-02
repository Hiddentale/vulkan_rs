# Usage Notes

Retrieves compiler statistics for a specific pipeline executable.
Statistics include metrics like register usage, instruction count,
scratch memory, and other driver-specific values.

Identify the executable by index from
`get_pipeline_executable_properties_khr` via
`PipelineExecutableInfoKHR`.

Each statistic has a name, description, format (bool, int, float,
or string), and value. The available statistics are
driver-specific, different vendors report different metrics.

The pipeline must have been created with
`PIPELINE_CREATE_CAPTURE_STATISTICS`. This is a profiling tool
for shader optimization, use it to compare register pressure
or instruction counts across shader variants.
