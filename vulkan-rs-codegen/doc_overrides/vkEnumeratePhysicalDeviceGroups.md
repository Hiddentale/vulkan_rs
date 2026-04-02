# Usage Notes

Enumerates the available device groups, sets of physical devices
that can be used together as a single logical device for multi-GPU
rendering (e.g. SLI/CrossFire).

Each `PhysicalDeviceGroupProperties` lists the physical devices in
the group and whether the group supports memory allocation that
spans all devices (`subset_allocation`).

On single-GPU systems, this returns one group containing one device.

To use a device group, pass `DeviceGroupDeviceCreateInfo` in the
pNext chain of `DeviceCreateInfo` with the desired physical devices.
This is an advanced multi-GPU feature; most applications use a
single physical device.
