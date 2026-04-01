# Usage Notes

Queries whether a fence can be exported to or imported from an
external handle type (e.g. POSIX file descriptor, Win32 handle,
or sync file).

The returned properties indicate:

- Whether the external handle type is compatible with fences.
- Whether the handle is a copy or a reference to the fence state.
- Which other handle types the fence can be exported to.

Use this before creating a fence intended for cross-process
synchronisation.
