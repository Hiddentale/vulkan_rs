# Usage Notes

Updates push constant values for the bound pipeline layout. Push
constants are a fast path for small, frequently-changing data that
avoids descriptor set updates entirely.

**Size limit**: the total push constant range is at least 128 bytes
(device limit `max_push_constants_size`). Use push constants for
per-draw data like transform matrices, material indices, or time
values.

**Stage flags**: the `stage_flags` parameter must match the stages
declared in the pipeline layout's push constant range. You can
update different stage ranges separately (e.g. update the vertex
shader's range without touching the fragment shader's range).

Push constant data persists across draw/dispatch calls until the
pipeline layout is changed or the values are overwritten.

For Vulkan 1.4+, `cmd_push_constants2` uses an extensible struct.
