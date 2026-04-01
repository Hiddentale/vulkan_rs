# Usage Notes

Returns a function pointer for a device-level command. This is the
device-specific equivalent of `get_instance_proc_addr` and returns
pointers dispatched directly through the device's driver, bypassing
the loader trampoline.

In normal usage you do not need to call this yourself, `Device`
loads all function pointers automatically at creation time. Use this
only if you need a command that is not yet exposed as a wrapper
method, or for raw interop scenarios.

The returned pointer is only valid for the device it was queried
from. Passing a command name that the device does not support
returns a null pointer.
