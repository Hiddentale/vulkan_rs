# Usage Notes

Ends a dynamic rendering instance started by `cmd_begin_rendering`.
Store operations and any resolve operations specified in the
`RenderingInfo` are executed at this point.

After this call, no draw commands may be recorded until a new
rendering or render pass instance is begun.
