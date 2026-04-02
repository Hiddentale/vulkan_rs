# Usage Notes

Stores a `u64` value on any Vulkan object for the given private data
slot. Overwrites any previously stored value for this object/slot
pair.

Private data is per-device, the slot must have been created from
the same device that owns the object. Setting data on an object from
a different device is undefined behaviour.

To clear the association, set the value to 0 or destroy the slot.
