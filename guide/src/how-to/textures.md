# Load and Sample Textures

> **Task:** Load an image from disk, upload it to GPU memory, and sample
> it in a fragment shader.

## Prerequisites

You should be comfortable with:
- [Memory Management](../concepts/memory.md) (staging buffers, memory types)
- [Command Buffers](../concepts/command-buffers.md) (one-shot transfers)
- [Descriptor Sets](../concepts/descriptors.md) (binding samplers)
- [Synchronization](../concepts/synchronization.md) (image layout transitions)

## Overview

Sampling a texture in Vulkan requires several steps that OpenGL handled
behind the scenes: creating a staging buffer, allocating a device-local
image, transitioning layouts with pipeline barriers, copying data, and
finally binding the image through a descriptor set. This recipe walks
through each step.

## Step 1: Load pixels from disk

Use the `image` crate to decode an image file into raw RGBA pixels.

```rust,ignore
let img = image::open("assets/texture.png")
    .expect("Failed to open image")
    .to_rgba8();

let (width, height) = img.dimensions();
let pixels = img.as_raw();
let image_size = (width * height * 4) as u64; // 4 bytes per RGBA pixel
```

## Step 2: Create a staging buffer

The CPU cannot write directly to device-local memory on most hardware.
Upload the pixels into a host-visible staging buffer first.

```rust,ignore
use vulkan_rust::vk;
use vk::*;

let staging_info = BufferCreateInfo::builder()
    .size(image_size)
    .usage(BufferUsageFlags::TRANSFER_SRC)
    .sharing_mode(SharingMode::EXCLUSIVE);

let staging_buffer = unsafe { device.create_buffer(&staging_info, None) }
    .expect("Failed to create staging buffer");
let staging_reqs = unsafe { device.get_buffer_memory_requirements(staging_buffer) };

let staging_memory = allocate_and_bind_buffer(
    device,
    staging_buffer,
    &staging_reqs,
    &mem_properties,
    MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
);

// Map, copy pixels, unmap.
unsafe {
    let ptr = device.map_memory(
        staging_memory, 0, image_size,
        MemoryMapFlags::empty(),
    )
    .expect("Failed to map memory");
    core::ptr::copy_nonoverlapping(
        pixels.as_ptr(), ptr as *mut u8, image_size as usize,
    );
    device.unmap_memory(staging_memory);
}
```

> See [Memory Management](../concepts/memory.md) for the
> `allocate_and_bind_buffer` helper and the `find_memory_type` algorithm.

## Step 3: Create the device-local image

The image needs `TRANSFER_DST` (we will copy into it) and `SAMPLED`
(the fragment shader will sample it).

```rust,ignore
use vulkan_rust::vk;
use vk::*;

let image_info = ImageCreateInfo::builder()
    .image_type(ImageType::_2D)
    .format(Format::R8G8B8A8_SRGB)
    .extent(Extent3D { width, height, depth: 1 })
    .mip_levels(1)
    .array_layers(1)
    .samples(SampleCountFlagBits::_1)
    .tiling(ImageTiling::OPTIMAL)
    .usage(
        ImageUsageFlags::TRANSFER_DST
        | ImageUsageFlags::SAMPLED
    )
    .sharing_mode(SharingMode::EXCLUSIVE)
    .initial_layout(ImageLayout::UNDEFINED);

let texture_image = unsafe { device.create_image(&image_info, None) }
    .expect("Failed to create image");

// Allocate DEVICE_LOCAL memory and bind it to the image.
let img_reqs = unsafe { device.get_image_memory_requirements(texture_image) };
let texture_memory = allocate_and_bind_image(
    device, texture_image, &img_reqs, &mem_properties,
    MemoryPropertyFlags::DEVICE_LOCAL,
);
```

## Step 4: Transition layout UNDEFINED to TRANSFER_DST_OPTIMAL

Before copying into the image, transition it to a layout the transfer
engine can write to. This requires a pipeline barrier.

> *Before reading on: why can't we just copy into an image that is in
> UNDEFINED layout? What does the layout tell the driver?*

```rust,ignore
use vulkan_rust::vk;
use vk::*;
use vk::constants;

let barrier_to_transfer = ImageMemoryBarrier::builder()
    .old_layout(ImageLayout::UNDEFINED)
    .new_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
    .src_queue_family_index(QUEUE_FAMILY_IGNORED)
    .dst_queue_family_index(QUEUE_FAMILY_IGNORED)
    .image(texture_image)
    .subresource_range(ImageSubresourceRange {
        aspect_mask: ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    })
    // No prior access to wait for (image was UNDEFINED).
    .src_access_mask(AccessFlags::NONE)
    // The transfer write must wait until the transition completes.
    .dst_access_mask(AccessFlags::TRANSFER_WRITE);

unsafe {
    device.cmd_pipeline_barrier(
        cmd,
        PipelineStageFlags::TOP_OF_PIPE,   // src stage: nothing before
        PipelineStageFlags::TRANSFER,       // dst stage: transfer write
        DependencyFlags::empty(),
        &[],             // memory barriers
        &[],             // buffer memory barriers
        &[*barrier_to_transfer],
    );
}
```

> See [Synchronization](../concepts/synchronization.md) for a deeper
> explanation of pipeline barriers and access masks.

## Step 5: Copy staging buffer to image

```rust,ignore
use vulkan_rust::vk;
use vk::*;

let region = BufferImageCopy {
    buffer_offset: 0,
    // 0 means tightly packed (no padding between rows).
    buffer_row_length: 0,
    buffer_image_height: 0,
    image_subresource: ImageSubresourceLayers {
        aspect_mask: ImageAspectFlags::COLOR,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1,
    },
    image_offset: Offset3D { x: 0, y: 0, z: 0 },
    image_extent: Extent3D { width, height, depth: 1 },
};

unsafe {
    device.cmd_copy_buffer_to_image(
        cmd,
        staging_buffer,
        texture_image,
        ImageLayout::TRANSFER_DST_OPTIMAL,
        &[region],
    );
}
```

## Step 6: Transition layout TRANSFER_DST to SHADER_READ_ONLY

After the copy, transition the image to a layout the shader can read.

```rust,ignore
use vulkan_rust::vk;
use vk::*;
use vk::constants;

let barrier_to_shader = ImageMemoryBarrier::builder()
    .old_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
    .new_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
    .src_queue_family_index(QUEUE_FAMILY_IGNORED)
    .dst_queue_family_index(QUEUE_FAMILY_IGNORED)
    .image(texture_image)
    .subresource_range(ImageSubresourceRange {
        aspect_mask: ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    })
    .src_access_mask(AccessFlags::TRANSFER_WRITE)
    .dst_access_mask(AccessFlags::SHADER_READ);

unsafe {
    device.cmd_pipeline_barrier(
        cmd,
        PipelineStageFlags::TRANSFER,
        PipelineStageFlags::FRAGMENT_SHADER,
        DependencyFlags::empty(),
        &[], &[],
        &[*barrier_to_shader],
    );
}
```

## Step 7: Create image view and sampler

The shader does not access images directly. It reads through an
**image view** (which selects format, mip levels, and array layers) and a
**sampler** (which controls filtering and addressing).

```rust,ignore
use vulkan_rust::vk;
use vk::*;

let view_info = ImageViewCreateInfo::builder()
    .image(texture_image)
    .view_type(ImageViewType::_2D)
    .format(Format::R8G8B8A8_SRGB)
    .subresource_range(ImageSubresourceRange {
        aspect_mask: ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    });

let texture_view = unsafe { device.create_image_view(&view_info, None) }
    .expect("Failed to create image view");

let sampler_info = SamplerCreateInfo::builder()
    .mag_filter(Filter::LINEAR)
    .min_filter(Filter::LINEAR)
    .address_mode_u(SamplerAddressMode::REPEAT)
    .address_mode_v(SamplerAddressMode::REPEAT)
    .address_mode_w(SamplerAddressMode::REPEAT)
    // Requires the samplerAnisotropy device feature to be enabled.
    // Set anisotropy_enable(0) if the feature is not available.
    .anisotropy_enable(true)
    .max_anisotropy(16.0)
    .border_color(BorderColor::INT_OPAQUE_BLACK)
    .mipmap_mode(SamplerMipmapMode::LINEAR)
    .min_lod(0.0)
    .max_lod(0.0);

let sampler = unsafe { device.create_sampler(&sampler_info, None) }
    .expect("Failed to create sampler");
```

## Step 8: Bind via descriptor set

Update a descriptor set so the shader can access the combined
image/sampler pair at a binding point.

```rust,ignore
use vulkan_rust::vk;
use vk::*;

let image_descriptor = DescriptorImageInfo {
    sampler,
    image_view: texture_view,
    image_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
};

let write = WriteDescriptorSet::builder()
    .dst_set(descriptor_set)
    .dst_binding(1) // must match the binding in the shader
    .dst_array_element(0)
    .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
    .image_info(&[image_descriptor]);

unsafe { device.update_descriptor_sets(&[*write], &[]) };
```

In the fragment shader (GLSL):

```glsl
layout(set = 0, binding = 1) uniform sampler2D texSampler;

void main() {
    outColor = texture(texSampler, fragTexCoord);
}
```

> See [Descriptor Sets](../concepts/descriptors.md) for descriptor pool
> creation and layout setup.

## Cleanup

Because `vulkan_rust` handles do not implement `Drop`, you must destroy
resources manually when they are no longer needed.

```rust,ignore
// Wait for the GPU to finish using these resources first.
unsafe {
    device.device_wait_idle()
        .expect("Failed to wait for device idle");
    device.destroy_sampler(sampler, None);
    device.destroy_image_view(texture_view, None);
    device.destroy_image(texture_image, None);
    device.free_memory(texture_memory, None);
    // Staging buffer should already be destroyed after the upload.
}
```

## Notes

- **Format choice.** `R8G8B8A8_SRGB` is correct for most color textures.
  Use `R8G8B8A8_UNORM` for data textures (normal maps, roughness) where
  sRGB gamma correction would be wrong.
- **Mipmaps.** This recipe creates a single mip level. For proper texture
  filtering at a distance, generate a full mip chain using
  `cmd_blit_image` in a loop, with a barrier between each level.
- **One-shot command buffer.** Steps 4 through 6 are typically recorded
  into a short-lived command buffer that is submitted and waited on
  immediately. Reuse command buffers from a transient pool for this.
