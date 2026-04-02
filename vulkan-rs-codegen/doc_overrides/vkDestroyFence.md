# Usage Notes

Destroys a fence object. The fence must not be in use by any
pending `queue_submit` call, wait on it or call `device_wait_idle`
before destroying.

Fences are lightweight objects but are still tracked by the driver.
Destroy them during teardown or when they are no longer part of your
synchronization scheme.
