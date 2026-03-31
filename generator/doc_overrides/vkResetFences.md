# Usage Notes

Resets one or more fences to the unsignaled state. Must be called
before reusing a fence in a new `queue_submit` call.

The fence must not be currently waited on by `wait_for_fences` from
another thread. A common safe pattern:

1. `wait_for_fences` — blocks until signaled.
2. `reset_fences` — immediately reset after the wait returns.
3. Submit new work with the fence.

Resetting a fence that is already unsignaled is valid but wasteful.
Resetting a fence that is pending (submitted but not yet signaled)
is an error.
