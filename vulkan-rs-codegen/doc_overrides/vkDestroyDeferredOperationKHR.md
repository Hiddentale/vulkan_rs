# Usage Notes

Destroys a deferred operation handle. The operation must have
completed before destruction, either all joining threads returned
`SUCCESS` or `THREAD_DONE_KHR`, or the operation was never
deferred.

Do not destroy while threads are still joined to the operation.
