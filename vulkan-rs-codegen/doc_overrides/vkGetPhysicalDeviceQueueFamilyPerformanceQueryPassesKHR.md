# Usage Notes

Returns the number of passes required to collect all the
performance counters specified in
`QueryPoolPerformanceCreateInfoKHR`.

Hardware can typically sample only a limited number of counters
per pass. If this returns N, you must submit the same command
buffer N times (each with a different pass index set via
`PerformanceQuerySubmitInfoKHR` in the pNext of
`SubmitInfo`/`SubmitInfo2`) to collect all results.

Requires `VK_KHR_performance_query`.
