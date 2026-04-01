# Usage Notes

Returns the current counter value of a timeline semaphore. Timeline
semaphores (Vulkan 1.2) use a monotonically increasing 64-bit
counter instead of binary signaled/unsignaled state.

Use this for non-blocking progress checks:

```text
let value = get_semaphore_counter_value(semaphore);
if value >= expected_frame {
    // GPU has finished frame N, safe to reuse resources
}
```

For blocking waits, use `wait_semaphores`. For signaling from the
CPU, use `signal_semaphore`.

Only valid for semaphores created with `SEMAPHORE_TYPE_TIMELINE`.
Calling this on a binary semaphore is an error.
