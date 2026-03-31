# Usage Notes

Blocks the calling thread until one or all of the given fences are
signaled, or until the timeout expires.

**`wait_all`**: when `true`, the call returns only after *every*
fence in the list is signaled. When `false`, it returns as soon as
*any* one fence is signaled.

**Timeout**: specified in nanoseconds. `u64::MAX` means wait
indefinitely. A timeout of zero performs a non-blocking check
(equivalent to polling `get_fence_status` on each fence).

Returns `VK_TIMEOUT` if the timeout expires before the condition is
met. This is not an error — check the return value and handle it
(e.g. log a warning or retry).

**Typical frame loop**:

```text
wait_for_fences(&[frame_fence], true, u64::MAX)
reset_fences(&[frame_fence])
// record and submit...
```

After `wait_for_fences` returns successfully, all GPU work
associated with those fences is complete and the resources are safe
to reuse.
