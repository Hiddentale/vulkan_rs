# Usage Notes

Lists the executable components within a pipeline. A single
pipeline may contain multiple executables, for example, a
graphics pipeline typically has separate vertex and fragment
shader executables.

Each returned `PipelineExecutablePropertiesKHR` contains a name,
description, and shader stage flags identifying the executable.
Use these to index into `get_pipeline_executable_statistics_khr`
and `get_pipeline_executable_internal_representations_khr`.

The pipeline must have been created with
`PIPELINE_CREATE_CAPTURE_STATISTICS` or
`PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS`. This is a
debugging and profiling tool, not intended for shipping builds.
