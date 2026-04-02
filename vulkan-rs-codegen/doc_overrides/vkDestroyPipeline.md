# Usage Notes

Destroys a graphics, compute, or ray tracing pipeline. The pipeline
must not be bound in any command buffer that is still pending
execution.

Pipelines are expensive to create but cheap to keep around. Only
destroy them when you are certain they will not be needed again
(e.g. during level transitions or application shutdown).

Shader modules used to create the pipeline can be destroyed
independently, the pipeline retains its own copy of the compiled
state.
