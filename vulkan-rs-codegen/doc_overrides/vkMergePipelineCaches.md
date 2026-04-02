# Usage Notes

Merges one or more source caches into a destination cache. Useful
when multiple threads each use their own cache during parallel
pipeline creation, merge them into a single cache before
serializing to disk.

The source caches are not modified or destroyed by this call. The
destination cache receives all entries from the sources that it does
not already contain. Duplicate entries are ignored.

After merging, you can destroy the source caches if they are no
longer needed.
