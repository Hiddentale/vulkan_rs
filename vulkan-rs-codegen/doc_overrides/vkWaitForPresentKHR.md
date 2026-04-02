# Usage Notes

Blocks the calling thread until a specific present operation
completes on the display. `present_id` identifies which present
to wait for, set it via `PresentIdKHR` chained into
`PresentInfoKHR` during `queue_present_khr`.

`timeout` is in nanoseconds. Returns `TIMEOUT` if the deadline
expires before the present completes, `SUCCESS` if the present
finished. Use `u64::MAX` for an indefinite wait.

Requires `VK_KHR_present_wait` and `VK_KHR_present_id`.

This is useful for frame pacing, wait for the previous frame's
present to complete before starting the next frame's work to
avoid queuing excessive frames.
