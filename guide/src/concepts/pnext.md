# The pNext Extension Chain

## Motivation

Vulkan evolves through extensions, and extensions often need to add fields
to existing structs. But Vulkan structs are `#[repr(C)]` with a fixed
layout, you cannot just add fields. The solution is `pNext`: a linked
list pointer in every extensible struct that lets you chain additional
data structures onto it.

This is Vulkan's most powerful extensibility mechanism and one of its
most confusing features for newcomers. Once you understand it, enabling
new Vulkan features and extensions becomes straightforward.

## Intuition

### The envelope analogy

Every Vulkan struct with a `pNext` field is an envelope. The main struct
is the letter inside. The `pNext` chain lets you stuff additional pages
into the same envelope.

The driver opens the envelope, reads the main page, then checks if there
are more pages. Each extra page has a header (`sType`) that identifies
what it is, so the driver knows how to interpret it. Pages it doesn't
recognize are silently skipped.

```text
DeviceCreateInfo (envelope)
├── sType: DEVICE_CREATE_INFO          (header: "this is a device create info")
├── pNext ──────────────────────────┐
├── ... (normal fields)             │
│                                   v
│               PhysicalDeviceVulkan12Features (extra page)
│               ├── sType: PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
│               ├── pNext ──────────────────────────┐
│               ├── ... (Vulkan 1.2 feature flags)  │
│                                                   v
│                           PhysicalDeviceVulkan13Features (another page)
│                           ├── sType: PHYSICAL_DEVICE_VULKAN_1_3_FEATURES
│                           ├── pNext: null (end of chain)
│                           ├── ... (Vulkan 1.3 feature flags)
```

### Under the hood: two pointers

Every extensible Vulkan struct starts with the same two fields:

```rust,ignore
pub struct SomeCreateInfo {
    pub s_type: StructureType,           // identifies the struct type
    pub p_next: *const core::ffi::c_void, // pointer to next struct in chain
    // ... rest of the fields
}
```

The `sType` field is a discriminator, like a tagged union. The driver
reads `sType` to know what struct it's looking at, then casts the
pointer to the correct type. This is the same pattern as COM's
`QueryInterface` or protobuf's `Any`.

## Worked example: enabling Vulkan 1.2 and 1.3 features

The most common use of pNext chains is enabling device features from
newer Vulkan versions or extensions.

### Without vulkan_rs builders (raw C-style)

```rust,ignore
use vulkan_rs::vk;
use vulkan_rs::vk::structs::*;
use vulkan_rs::vk::enums::*;

// You would need to manually link the structs:
let mut features_13 = PhysicalDeviceVulkan13Features {
    s_type: StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
    p_next: core::ptr::null_mut() as *const _,
    dynamic_rendering: 1,   // enable dynamic rendering
    synchronization2: 1,    // enable synchronization2
    ..unsafe { core::mem::zeroed() }
};

let mut features_12 = PhysicalDeviceVulkan12Features {
    s_type: StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
    p_next: &mut features_13 as *mut _ as *const _,  // link to next
    buffer_device_address: 1,
    descriptor_indexing: 1,
    ..unsafe { core::mem::zeroed() }
};

let device_info = DeviceCreateInfo {
    s_type: StructureType::DEVICE_CREATE_INFO,
    p_next: &mut features_12 as *mut _ as *const _,  // link to chain
    // ...
};
```

This is error-prone: wrong `sType`, dangling pointers, forgetting to
link the chain. vulkan_rs builders fix all of these problems.

### With vulkan_rs builders (type-safe)

```rust,ignore
use vulkan_rs::vk;
use vulkan_rs::vk::structs::*;

let mut features_12 = *PhysicalDeviceVulkan12Features::builder()
    .buffer_device_address(1)
    .descriptor_indexing(1);

let mut features_13 = *PhysicalDeviceVulkan13Features::builder()
    .dynamic_rendering(1)
    .synchronization2(1);

let device_info = DeviceCreateInfo::builder()
    .push_next(&mut features_12)
    .push_next(&mut features_13)
    // ... other fields
    ;
```

The builder handles:
- **`sType` is set automatically** by `builder()`.
- **`pNext` linking is handled by `push_next`**, which prepends each
  struct to the chain.
- **Type safety** via marker traits: `push_next` only accepts types
  that the Vulkan spec says are valid extensions for that struct.
  Passing an invalid type is a compile error.

> *Before reading on: what do you think happens if you chain a struct
> that the driver doesn't recognize (e.g., an extension struct the
> driver doesn't support)?*
>
> Answer: The driver skips it. Every struct in the chain has an `sType`
> header. The driver reads each `sType`, processes structs it
> recognizes, and follows the `pNext` pointer past structs it doesn't.
> This is how forward compatibility works: old drivers ignore new
> extension structs.

## How push_next works

The `push_next` method *prepends* to the chain. Each call inserts the
new struct at the front:

```rust,ignore
// push_next implementation (simplified):
pub fn push_next<T: ExtendsDeviceCreateInfo>(mut self, next: &'a mut T) -> Self {
    unsafe {
        let next_ptr = next as *mut T as *mut BaseOutStructure;
        // Point the new struct's pNext to the current chain head.
        (*next_ptr).p_next = self.inner.p_next as *mut _;
        // Make the new struct the chain head.
        self.inner.p_next = next_ptr as *const _;
    }
    self
}
```

After two `push_next` calls:

```text
DeviceCreateInfo.pNext → features_13 → features_12 → null
                         (last pushed    (first pushed
                          is first)       is last)
```

The order in the chain does not matter to the driver. It walks the
entire chain regardless of order.

## The Extends marker traits

For each extensible struct, `vulkan_rs` generates an `unsafe trait`:

```rust,ignore
pub unsafe trait ExtendsDeviceCreateInfo {}
```

Types that the Vulkan spec says can appear in `DeviceCreateInfo`'s
pNext chain implement this trait:

```rust,ignore
unsafe impl ExtendsDeviceCreateInfo for PhysicalDeviceVulkan12Features {}
unsafe impl ExtendsDeviceCreateInfo for PhysicalDeviceVulkan13Features {}
unsafe impl ExtendsDeviceCreateInfo for DevicePrivateDataCreateInfo {}
// ... hundreds more
```

These traits are generated from the `structextends` attribute in
`vk.xml`, so they are always in sync with the Vulkan spec.

If you try to `push_next` a struct that doesn't implement the trait:

```rust,ignore
use vulkan_rs::vk;
use vulkan_rs::vk::structs::*;

// Compile error: PhysicalDeviceMemoryProperties does not implement
// ExtendsDeviceCreateInfo
let info = DeviceCreateInfo::builder()
    .push_next(&mut mem_props);  // ← won't compile
```

## The builder Deref pattern

vulkan_rs builders implement `Deref<Target = InnerStruct>`, so you can
pass a builder anywhere a reference to the inner struct is expected:

```rust,ignore
use vulkan_rs::vk;
use vulkan_rs::vk::structs::*;

let info = DeviceCreateInfo::builder()
    .queue_create_infos(&queue_infos)
    .push_next(&mut features_12);

// No need to call .build(), just pass &info or *info.
let device = unsafe { instance.create_device(physical_device, &info, None)? };
```

The `*info` dereference gives you the inner `DeviceCreateInfo`.
The `&info` auto-derefs to `&DeviceCreateInfo` through `Deref`.

## Lifetime safety

Builders carry a lifetime parameter `'a` to ensure that references
passed to `push_next` (and slice methods like `queue_create_infos`)
live long enough:

```rust,ignore
pub struct DeviceCreateInfoBuilder<'a> {
    inner: DeviceCreateInfo,
    _marker: PhantomData<&'a ()>,
}
```

This means the builder and everything chained into it must live in
the same scope. The compiler enforces this:

```rust,ignore
use vulkan_rs::vk;
use vulkan_rs::vk::structs::*;

let info = {
    let mut features = PhysicalDeviceVulkan12Features::builder();
    DeviceCreateInfo::builder()
        .push_next(&mut features)
    // ← compile error: `features` does not live long enough
};
```

## Common pNext patterns

### Querying supported features

Chain feature structs into `PhysicalDeviceFeatures2` and call
`get_physical_device_features2`:

```rust,ignore
use vulkan_rs::vk;
use vulkan_rs::vk::structs::*;

let mut features_12 = *PhysicalDeviceVulkan12Features::builder();
let mut features_13 = *PhysicalDeviceVulkan13Features::builder();

let mut features2 = PhysicalDeviceFeatures2::builder()
    .push_next(&mut features_12)
    .push_next(&mut features_13);

unsafe {
    instance.get_physical_device_features2(physical_device, &mut *features2);
};

// Now features_12 and features_13 are filled in by the driver.
if features_12.buffer_device_address != 0 {
    println!("Buffer device address is supported");
}
```

### Enabling features at device creation

Pass the same structs (with your desired features set to 1) into
`DeviceCreateInfo` via `push_next`, as shown in the worked example
above.

## Formal reference

### Key types

| Type | Purpose |
|------|---------|
| `BaseInStructure` | Generic pNext chain traversal (const). Fields: `s_type`, `p_next`. |
| `BaseOutStructure` | Generic pNext chain traversal (mutable). Fields: `s_type`, `p_next`. |
| `StructureType` | Enum identifying each struct type. Set automatically by `builder()`. |
| `ExtendsXxx` traits | Marker traits generated from `vk.xml` `structextends` attribute. |

### Rules

1. **Never set `sType` manually.** `builder()` does it for you.
2. **Never manipulate `pNext` directly.** Use `push_next`.
3. **Order in the chain does not matter.** The driver walks the full chain.
4. **Lifetimes must be valid.** All chained structs must outlive the API
   call that consumes them.
5. **Unknown structs are skipped.** Chaining an extension struct the
   driver doesn't support is safe, it will be ignored.

### API reference links

- [`BaseOutStructure`](https://docs.rs/vulkan-rs/latest/vulkan_rs/vk/struct.BaseOutStructure.html)
- [`StructureType`](https://docs.rs/vulkan-rs/latest/vulkan_rs/vk/struct.StructureType.html)
- [Vulkan spec: Extending Structures](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#fundamentals-validusage-pNext)

## Key takeaways

- `pNext` is a linked list that lets extensions add data to existing
  structs without changing their layout.
- `vulkan_rs` builders make pNext chains type-safe: `push_next` only
  accepts types the spec allows, `sType` is set automatically, and
  lifetimes are enforced by the compiler.
- The most common use case is enabling device features from Vulkan 1.2,
  1.3, or extensions at device creation time.
- Chain order does not matter. Unknown structs are silently skipped.
