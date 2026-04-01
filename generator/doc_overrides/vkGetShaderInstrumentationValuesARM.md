# Usage Notes

Retrieves collected shader instrumentation metric values after
instrumented commands have completed execution. Returns metric
blocks whose count is written to `p_metric_block_count`. Ensure
the instrumented submission has finished before querying.

Requires `VK_ARM_shader_instrumentation`.
