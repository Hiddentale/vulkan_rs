# Usage Notes

Queries recorded fault data from the device. Part of Vulkan SC
(Safety Critical) for fault reporting in safety-certified
environments. Uses the two-call idiom. The
`fault_query_behavior` controls whether queried faults are
cleared. Also reports the count of unrecorded faults that
overflowed the internal buffer.

Requires Vulkan SC.
