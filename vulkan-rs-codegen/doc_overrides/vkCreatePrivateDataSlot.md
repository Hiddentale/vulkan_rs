# Usage Notes

Creates a private data slot that can be used to attach arbitrary
application-defined `u64` values to any Vulkan object. Private data
is a lightweight alternative to external hash maps for per-object
metadata.

Each slot represents one "key" that can be set on any object via
`set_private_data` and read back via `get_private_data`.

Use cases:

- Tagging objects with debug IDs.
- Associating application-specific state without a separate lookup
  table.
- Layer and tool implementations that need per-object bookkeeping.

Private data slots are cheap. The slot itself is just an index;
the per-object storage is allocated lazily by the driver.
