# Usage Notes

Non-blocking check of whether a fence is signaled. Returns
`VK_SUCCESS` if signaled, `VK_NOT_READY` if still pending.

Use this for polling patterns where you want to do other work while
waiting:

```text
loop {
    if get_fence_status(fence) == VK_SUCCESS { break; }
    // do other work...
}
```

For blocking waits, prefer `wait_for_fences` which is more efficient
than a spin loop, it lets the CPU sleep until the driver signals.

This call can also return device-lost errors, so check the result
even in non-error paths.
