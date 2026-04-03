# Use Push Constants

> **Task:** Pass small, frequently-changing data (like a model matrix)
> to shaders without descriptor sets or buffer allocations.

## Prerequisites

- [Pipelines](../concepts/pipelines.md) (pipeline layout)
- [Descriptor Sets](../concepts/descriptors.md) (for comparison with uniform buffers)

## What push constants are

Push constants are a small block of data written directly into the
command buffer. Unlike uniform buffers, they require no buffer
allocation, no memory binding, and no descriptor set update. You
declare a range in the pipeline layout, record the data inline during
command recording, and the shader reads it.

The tradeoff is size: the Vulkan spec guarantees at least **128 bytes**
of push constant storage. Most desktop GPUs offer 256 bytes. This is
enough for a 4x4 matrix (64 bytes) plus a handful of scalar parameters,
but not enough for large data sets.

## When to use push constants vs uniform buffers

| Criterion | Push constants | Uniform buffers |
|-----------|---------------|-----------------|
| Size | Up to 128-256 bytes | Unlimited |
| Setup cost | None (inline in command buffer) | Allocate buffer, bind memory, write descriptor |
| Per-draw update | Free (just `cmd_push_constants`) | Requires dynamic offsets or multiple descriptors |
| Best for | Model matrix, time, material index | Large arrays, shared view/projection data |

**Rule of thumb:** if the data changes per draw call and fits in 128
bytes, use push constants. For anything larger or shared across many
draws, use a uniform buffer.

## Step 1: Define the push constant data

Create a `#[repr(C)]` struct that matches the layout the shader expects.

```rust,ignore
#[repr(C)]
#[derive(Clone, Copy)]
struct PushConstants {
    model: [f32; 16],  // 4x4 matrix, 64 bytes
    time: f32,          // 4 bytes
    _padding: [f32; 3], // align to 16 bytes if needed
}
```

> *Before reading on: why does the struct need `#[repr(C)]`? What would
> happen if Rust reordered the fields?*

`#[repr(C)]` guarantees that the fields are laid out in declaration
order with C-compatible alignment. Without it, the Rust compiler may
reorder fields, and the shader would read garbage.

## Step 2: Declare push constant range in the pipeline layout

The push constant range tells Vulkan how many bytes of push constant
data your shaders use and which stages access them.

```rust,ignore
use vulkan_rust::vk;
use vk::*;

let push_constant_range = PushConstantRange {
    stage_flags: ShaderStageFlags::VERTEX,
    offset: 0,
    size: std::mem::size_of::<PushConstants>() as u32,
};

let push_ranges = [push_constant_range];
let layout_info = PipelineLayoutCreateInfo::builder()
    .set_layouts(&descriptor_set_layouts) // can be empty if you have no descriptors
    .push_constant_ranges(&push_ranges);

let pipeline_layout = unsafe {
    device.create_pipeline_layout(&layout_info, None)
}
.expect("Failed to create pipeline layout");
```

If both vertex and fragment shaders read push constants, you have two
options:

- **One range** with `stage_flags: VERTEX | FRAGMENT` if both stages read
  the same bytes.
- **Two ranges** at different offsets if each stage reads different data.

```rust,ignore
use vulkan_rust::vk;
use vk::*;

// Example: vertex reads bytes 0..64, fragment reads bytes 64..80.
let ranges = [
    PushConstantRange {
        stage_flags: ShaderStageFlags::VERTEX,
        offset: 0,
        size: 64,
    },
    PushConstantRange {
        stage_flags: ShaderStageFlags::FRAGMENT,
        offset: 64,
        size: 16,
    },
];
```

## Step 3: Declare push constants in the shader

In GLSL, push constants appear as a `uniform` block with the
`push_constant` layout qualifier.

**Vertex shader:**

```glsl
#version 450

layout(push_constant) uniform PushConstants {
    mat4 model;
    float time;
} pc;

layout(location = 0) in vec3 inPosition;

void main() {
    gl_Position = pc.model * vec4(inPosition, 1.0);
}
```

There can be only one `push_constant` block per shader stage. The block
members must match the byte layout of your Rust struct.

## Step 4: Record push constants during command recording

Use `cmd_push_constants` to write the data into the command buffer. This
is typically called once per draw, right before the draw command.

```rust,ignore
use vulkan_rust::vk;
use vk::*;

let push_data = PushConstants {
    model: compute_model_matrix(entity),
    time: elapsed_seconds,
    _padding: [0.0; 3],
};

unsafe {
    device.cmd_push_constants(
        cmd,
        pipeline_layout,
        ShaderStageFlags::VERTEX,
        0, // offset in bytes
        std::slice::from_raw_parts(
            &push_data as *const PushConstants as *const u8,
            std::mem::size_of::<PushConstants>(),
        ),
    );

    device.cmd_draw(cmd, vertex_count, 1, 0, 0);
}
```

For a scene with many objects, you push new constants before each draw:

```rust,ignore
use vulkan_rust::vk;
use vk::*;

for entity in &scene.entities {
    let push_data = PushConstants {
        model: entity.transform,
        time: elapsed_seconds,
        _padding: [0.0; 3],
    };

    unsafe {
        device.cmd_push_constants(
            cmd, pipeline_layout,
            ShaderStageFlags::VERTEX,
            0,
            std::slice::from_raw_parts(
                &push_data as *const PushConstants as *const u8,
                std::mem::size_of::<PushConstants>(),
            ),
        );

        device.cmd_draw_indexed(
            cmd, entity.index_count, 1, entity.first_index, 0, 0,
        );
    }
}
```

## A helper for safe byte casting

The `std::slice::from_raw_parts` pattern is error-prone. A small
helper makes it clearer:

```rust,ignore
use vulkan_rust::vk;
use vk::*;

/// Reinterpret a reference to a `Copy` type as a `&[u8]` slice
/// suitable for `cmd_push_constants`.
///
/// # Safety
/// The type must be `#[repr(C)]` with no padding that contains
/// uninitialized bytes.
unsafe fn as_push_bytes<T: Copy>(data: &T) -> &[u8] {
    std::slice::from_raw_parts(
        data as *const T as *const u8,
        std::mem::size_of::<T>(),
    )
}

// Usage:
unsafe {
    device.cmd_push_constants(
        cmd, pipeline_layout,
        ShaderStageFlags::VERTEX,
        0,
        as_push_bytes(&push_data),
    );
}
```

## Common mistakes

**Exceeding the size limit.** If your push constant struct is larger
than the device's `max_push_constants_size` (query from
`PhysicalDeviceLimits`), pipeline creation will fail. Check the limit
at startup.

**Mismatched stage flags.** The `stage_flags` in `cmd_push_constants`
must match the flags declared in the push constant range. If your range
says `VERTEX | FRAGMENT` but you push with `VERTEX` only, the
validation layer will warn.

**Incorrect offset.** The `offset` parameter in `cmd_push_constants` is
a byte offset into the push constant block. If you update only part of
the block (e.g. fragment-only data at offset 64), the vertex portion
retains its previously pushed values.

**Forgetting `#[repr(C)]`.** Without it, Rust may reorder struct fields.
The GPU will read bytes at fixed offsets, so reordered fields mean
corrupted data with no obvious error.

## Notes

- **Alignment.** GLSL `push_constant` blocks follow `std430` layout
  rules. A `vec3` takes 12 bytes (not 16) but the next member aligns to
  its own size. Prefer `vec4`/`mat4` to avoid alignment surprises, or
  add explicit padding in your Rust struct.
- **Performance.** Push constants are the fastest way to pass small
  per-draw data. On most architectures they live in GPU registers or a
  small on-chip cache, not in memory.
- **Compatibility.** 128 bytes is the guaranteed minimum. If you need
  more, check `max_push_constants_size` in `PhysicalDeviceLimits`. Most
  desktop drivers report 256 bytes.
- **Combining with descriptors.** Push constants and descriptor sets are
  complementary. A typical setup uses push constants for per-draw data
  (model matrix) and uniform buffers via descriptors for per-frame data
  (view/projection matrices, lighting).
