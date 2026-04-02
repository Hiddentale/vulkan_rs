# Usage Notes

Extensible version of `wait_for_present_khr`. Takes a
`PresentWait2InfoKHR` struct (with `pNext` support) instead of
separate `present_id` and `timeout` parameters.

Provided by `VK_KHR_present_wait2`. Otherwise identical in
behavior, blocks until the specified present ID completes or
the timeout expires.
