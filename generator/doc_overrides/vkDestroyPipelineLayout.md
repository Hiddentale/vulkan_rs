# Usage Notes

Destroys a pipeline layout. All pipelines and descriptor sets that
were created with this layout must no longer be in use.

In practice, pipeline layouts are typically created once and live for
the duration of the application or a major rendering context. There
is little reason to destroy them early.
