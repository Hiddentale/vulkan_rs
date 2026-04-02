# Descriptor Sets & Resource Binding

## Motivation

Shaders need access to resources: buffers containing transformation
matrices, images to sample, storage buffers for compute output.
Descriptors are Vulkan's mechanism for connecting shader bindings
(`layout(binding = 0) uniform ...`) to actual GPU resources.

The descriptor system is more complex than OpenGL's `glBindTexture`,
but it exists because binding resources one at a time is a bottleneck.
Vulkan lets you bind *sets* of resources at once, and reuse those sets
across multiple draw calls.

## Intuition

### The surgeon's tray

Think of a descriptor set as a tray of tools laid out for a surgeon:

- The **descriptor set layout** is the diagram showing which tool goes
  in which slot ("slot 0: scalpel, slot 1: forceps, slot 2: sutures").
- The **descriptor pool** is the sterilization room where trays are
  prepared (pre-allocated memory for many trays).
- The **descriptor set** is one prepared tray, with actual tools in
  each slot.
- **Writing** a descriptor set is placing specific tools into the slots.
- **Binding** is sliding the tray under the surgeon's hands during
  the operation.

The flow:

```text
1. Define layout     →  "what slots exist and what types they hold"
2. Create pool       →  "how many trays can we prepare at once"
3. Allocate set      →  "give me an empty tray matching this layout"
4. Write descriptors →  "put this buffer in slot 0, this image in slot 1"
5. Bind set          →  "use this tray for the next draw calls"
```

> *Before reading on: why do you think Vulkan uses descriptor "pools"
> instead of allocating descriptors individually? What performance
> problem does this solve?*
>
> Answer: Same reason as command pools, individual allocations are
> expensive because each one requires driver bookkeeping and possibly
> a kernel call. Pools pre-allocate a block of memory and hand out
> descriptors cheaply from that block.

### Descriptor types

Each slot in a descriptor set has a specific type:

| Type | What it binds | GLSL example |
|------|--------------|--------------|
| `UNIFORM_BUFFER` | Read-only buffer (matrices, parameters) | `layout(binding=0) uniform UBO { mat4 mvp; };` |
| `STORAGE_BUFFER` | Read/write buffer (compute data) | `layout(binding=0) buffer SSBO { float data[]; };` |
| `COMBINED_IMAGE_SAMPLER` | Image + sampler together | `layout(binding=0) uniform sampler2D tex;` |
| `SAMPLED_IMAGE` | Image without sampler | `layout(binding=0) uniform texture2D tex;` |
| `SAMPLER` | Sampler without image | `layout(binding=0) uniform sampler s;` |
| `STORAGE_IMAGE` | Read/write image (compute) | `layout(binding=0, rgba8) uniform image2D img;` |
| `INPUT_ATTACHMENT` | Previous subpass output | `layout(input_attachment_index=0) uniform subpassInput;` |

The most common are `UNIFORM_BUFFER` and `COMBINED_IMAGE_SAMPLER`.

## Worked example: binding a uniform buffer and a texture

### Step 1: Create a descriptor set layout

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;
use vk::enums::*;

// Describe the bindings: slot 0 is a uniform buffer visible to
// the vertex shader, slot 1 is a combined image sampler visible
// to the fragment shader.
let bindings = [
    DescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type: DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 1,
        stage_flags: ShaderStageFlags::VERTEX,
        p_immutable_samplers: core::ptr::null(),
    },
    DescriptorSetLayoutBinding {
        binding: 1,
        descriptor_type: DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 1,
        stage_flags: ShaderStageFlags::FRAGMENT,
        p_immutable_samplers: core::ptr::null(),
    },
];

let layout_info = DescriptorSetLayoutCreateInfo::builder()
    .bindings(&bindings);

let descriptor_layout = unsafe {
    device.create_descriptor_set_layout(&layout_info, None)?
};

// This layout is also passed to create_pipeline_layout, connecting
// the pipeline to the descriptor set structure.
```

### Step 2: Create a descriptor pool

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;
use vk::enums::*;

// The pool must have enough room for the descriptor types we need.
// If we want 10 sets, each with 1 uniform buffer and 1 image sampler:
let pool_sizes = [
    DescriptorPoolSize {
        r#type: DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 10,
    },
    DescriptorPoolSize {
        r#type: DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 10,
    },
];

let pool_info = DescriptorPoolCreateInfo::builder()
    .max_sets(10)
    .pool_sizes(&pool_sizes);

let descriptor_pool = unsafe {
    device.create_descriptor_pool(&pool_info, None)?
};
```

### Step 3: Allocate a descriptor set

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;
use vk::handles::*;

let alloc_info = DescriptorSetAllocateInfo::builder()
    .descriptor_pool(descriptor_pool)
    .set_layouts(&[descriptor_layout]);

let mut descriptor_set = DescriptorSet::null();
unsafe {
    device.allocate_descriptor_sets(&alloc_info, &mut descriptor_set)?;
};
```

### Step 4: Write descriptors (point slots to actual resources)

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;
use vk::enums::*;

// Point binding 0 to our uniform buffer.
let buffer_info = DescriptorBufferInfo {
    buffer: uniform_buffer,
    offset: 0,
    range: std::mem::size_of::<UniformData>() as u64,
};

// Point binding 1 to our texture.
let image_info = DescriptorImageInfo {
    sampler: texture_sampler,
    image_view: texture_image_view,
    image_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
};

let writes = [
    *WriteDescriptorSet::builder()
        .dst_set(descriptor_set)
        .dst_binding(0)
        .descriptor_type(DescriptorType::UNIFORM_BUFFER)
        .buffer_info(&[buffer_info]),
    *WriteDescriptorSet::builder()
        .dst_set(descriptor_set)
        .dst_binding(1)
        .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&[image_info]),
];

// This updates the descriptor set immediately. No command buffer needed.
unsafe { device.update_descriptor_sets(&writes, &[]) };
```

### Step 5: Bind during command recording

```rust,ignore
use vulkan_rs::vk;
use vk::enums::*;

unsafe {
    device.cmd_bind_descriptor_sets(
        command_buffer,
        PipelineBindPoint::GRAPHICS,
        pipeline_layout,
        0,                       // first set index
        &[descriptor_set],       // sets to bind
        &[],                     // dynamic offsets (none)
    );

    // Now draw calls in this command buffer can access the
    // uniform buffer at binding 0 and the texture at binding 1.
    device.cmd_draw(command_buffer, vertex_count, 1, 0, 0);
};
```

## Multiple descriptor sets

You can bind multiple descriptor sets at once. A common pattern:

```text
Set 0: Per-frame data      (camera matrices, lighting, time)
Set 1: Per-material data   (textures, material properties)
Set 2: Per-object data     (model matrix)
```

This lets you update and bind sets at different frequencies. Set 0
changes once per frame, set 1 changes when you switch materials,
set 2 changes per object. You only rebind the sets that changed.

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;
use vk::enums::*;

// In pipeline layout creation:
let layouts = [per_frame_layout, per_material_layout, per_object_layout];
let layout_info = PipelineLayoutCreateInfo::builder()
    .set_layouts(&layouts);

// During rendering:
unsafe {
    // Bind set 0 once per frame.
    device.cmd_bind_descriptor_sets(
        cmd, PipelineBindPoint::GRAPHICS,
        pipeline_layout, 0, &[per_frame_set], &[],
    );

    for material in &materials {
        // Bind set 1 per material.
        device.cmd_bind_descriptor_sets(
            cmd, PipelineBindPoint::GRAPHICS,
            pipeline_layout, 1, &[material.descriptor_set], &[],
        );

        for object in &material.objects {
            // Bind set 2 per object.
            device.cmd_bind_descriptor_sets(
                cmd, PipelineBindPoint::GRAPHICS,
                pipeline_layout, 2, &[object.descriptor_set], &[],
            );
            device.cmd_draw(cmd, object.vertex_count, 1, 0, 0);
        }
    }
};
```

> *Before reading on: in the pattern above, when you bind set 1 for
> a new material, does set 0 (per-frame) stay bound or does it need
> to be rebound?*
>
> Answer: It stays bound. Binding set N only affects set N. Sets at
> other indices remain bound from their previous `cmd_bind_descriptor_sets`
> call, as long as the pipeline layout is compatible.

## Formal reference

### The descriptor set creation flow

```text
DescriptorSetLayoutBinding[]
          │
          v
DescriptorSetLayoutCreateInfo ──> create_descriptor_set_layout ──> DescriptorSetLayout
                                                                          │
                    ┌─────────────────────────────────────────────────────┘
                    v
DescriptorPoolCreateInfo ──> create_descriptor_pool ──> DescriptorPool
                    │                                            │
                    v                                            v
DescriptorSetAllocateInfo ──────> allocate_descriptor_sets ──> DescriptorSet
                                                                    │
                                                                    v
WriteDescriptorSet[] ──────────> update_descriptor_sets   (set is now usable)
                                                            │
                                                            v
cmd_bind_descriptor_sets ──────> (shaders can access resources)
```

### Descriptor types reference

| Type | Read/Write | Typical use |
|------|-----------|-------------|
| `UNIFORM_BUFFER` | Read | Matrices, parameters (small, frequently updated) |
| `UNIFORM_BUFFER_DYNAMIC` | Read | Same, with dynamic offset at bind time |
| `STORAGE_BUFFER` | Read/Write | Large data, compute buffers |
| `STORAGE_BUFFER_DYNAMIC` | Read/Write | Same, with dynamic offset |
| `COMBINED_IMAGE_SAMPLER` | Read | Textures |
| `SAMPLED_IMAGE` | Read | Image without sampler (separate sampler) |
| `SAMPLER` | N/A | Sampler without image |
| `STORAGE_IMAGE` | Read/Write | Compute shader image output |
| `INPUT_ATTACHMENT` | Read | Previous subpass output |
| `INLINE_UNIFORM_BLOCK` | Read | Small uniform data inline in the set |

### Destruction order

1. Destroy pipeline layouts before descriptor set layouts.
2. Destroying a descriptor pool frees all sets allocated from it.
3. Descriptor set layouts can be destroyed after pipeline creation
   (the pipeline bakes a copy of the layout information).

### API reference links

- [`DescriptorSetLayout`](https://docs.rs/vulkan-rs/latest/vulkan_rs/vk/struct.DescriptorSetLayout.html)
- [`DescriptorPool`](https://docs.rs/vulkan-rs/latest/vulkan_rs/vk/struct.DescriptorPool.html)
- [`DescriptorSet`](https://docs.rs/vulkan-rs/latest/vulkan_rs/vk/struct.DescriptorSet.html)
- [`WriteDescriptorSet`](https://docs.rs/vulkan-rs/latest/vulkan_rs/vk/struct.WriteDescriptorSet.html)
- [`DescriptorType`](https://docs.rs/vulkan-rs/latest/vulkan_rs/vk/struct.DescriptorType.html)
- [Vulkan spec: Resource Descriptors](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#descriptorsets)

## Key takeaways

- Descriptors connect shader bindings to GPU resources (buffers, images).
- The flow is: define layout → create pool → allocate set → write → bind.
- Use multiple descriptor sets (per-frame, per-material, per-object) to
  minimize rebinding. Only rebind sets that change.
- Descriptor pools work like command pools: pre-allocate in bulk, hand
  out cheaply.
- `update_descriptor_sets` is a CPU-side operation, not a GPU command.
  You can update sets between submissions without recording commands.
