# Pipelines

> **Threshold concept.** In OpenGL, you set rendering state one call at a
> time, blend mode here, depth test there, and the driver compiles the
> final state lazily. In Vulkan, *all* state is compiled into a pipeline
> object up front. This removes driver guesswork and hitching at the cost
> of more explicit setup.

## Motivation

A GPU is not a general-purpose processor. It is a configurable state
machine with fixed-function stages (vertex input, rasterization, blending)
and programmable stages (vertex shader, fragment shader). A pipeline
object captures the *full configuration* of this machine, every stage,
every setting, so the driver can compile it to hardware instructions
once and reuse it many times.

This is why OpenGL applications sometimes stutter when a new material
appears: the driver has to compile a new internal pipeline on the fly.
In Vulkan, you create all your pipelines at load time and switch between
them during rendering with zero compilation cost.

## Intuition

### The mixing console preset

A pipeline is like a preset on a mixing console. Instead of adjusting
every knob during a live performance (and risking a pop or crackle), you
save the full board state as a preset and recall it instantly. You can
have many presets and switch between them, but you cannot twiddle
individual knobs mid-song.

(Vulkan 1.3 added *dynamic state* to relax this, certain knobs *can*
be adjusted at draw time. But the core idea holds: most state is baked.)

### What goes into a graphics pipeline

A graphics pipeline is the largest create info in the Vulkan API. It
bundles together every stage of the rendering process:

```text
GraphicsPipelineCreateInfo
│
├── Shader stages            (vertex shader, fragment shader, ...)
├── Vertex input state       (what vertex data looks like)
├── Input assembly state     (triangles, lines, points)
├── Viewport state           (viewport + scissor rectangle)
├── Rasterization state      (polygon mode, culling, depth bias)
├── Multisample state        (MSAA settings)
├── Depth/stencil state      (depth test, stencil test)
├── Color blend state        (blending per attachment)
├── Dynamic state            (which of the above can change at draw time)
├── Pipeline layout          (what resources the shaders expect)
└── Render pass + subpass    (which render pass this pipeline is used in)
```

Every one of these must be specified. There are no defaults. This is
verbose, but it means the driver has complete information at creation
time and can optimize aggressively.

> *Before reading on: if you need to render some objects with blending
> and some without, how many pipeline objects do you need?*
>
> Answer: Two. Each pipeline bakes its blend state. You
> `cmd_bind_pipeline` to switch between them during command recording.
> Dynamic state (Vulkan 1.3) can make some of these switches cheaper,
> but you still need separate pipelines for fundamental differences
> like different shaders.

### Pipeline layout: the bridge to resources

A pipeline layout declares what resources the shaders expect:

- **Descriptor set layouts**: "binding 0 is a uniform buffer, binding 1
  is a sampled image" (covered in [Descriptor Sets](descriptors.md))
- **Push constant ranges**: small inline data passed at draw time
  (covered in [Push Constants](../how-to/push-constants.md))

The pipeline layout is shared between pipeline creation and command
recording, ensuring the resources you bind match what the shaders expect.

## Worked example: creating a graphics pipeline

This is a minimal pipeline for rendering colored triangles.

### Step 1: Load shaders

```rust,ignore
use vulkan_rs::vk;
use vulkan_rs::vk::structs::*;

// SPIR-V bytecode, compiled from GLSL with glslc or shaderc.
let vert_code: &[u32] = /* load from file or include_bytes! */;
let frag_code: &[u32] = /* load from file or include_bytes! */;

let vert_info = ShaderModuleCreateInfo::builder()
    .code(vert_code);
let frag_info = ShaderModuleCreateInfo::builder()
    .code(frag_code);

let vert_module = unsafe { device.create_shader_module(&vert_info, None)? };
let frag_module = unsafe { device.create_shader_module(&frag_info, None)? };

// Shader stage descriptions.
let entry_name = c"main";  // GLSL entry point

let stages = [
    *PipelineShaderStageCreateInfo::builder()
        .stage(ShaderStageFlags::VERTEX)
        .module(vert_module)
        .name(entry_name),
    *PipelineShaderStageCreateInfo::builder()
        .stage(ShaderStageFlags::FRAGMENT)
        .module(frag_module)
        .name(entry_name),
];
```

### Step 2: Define vertex input

```rust,ignore
use vulkan_rs::vk;
use vulkan_rs::vk::structs::*;
use vulkan_rs::vk::enums::*;

// Describe how vertex data is laid out in memory.
let binding = VertexInputBindingDescription {
    binding: 0,
    stride: std::mem::size_of::<Vertex>() as u32,
    input_rate: VertexInputRate::VERTEX,
};

let attributes = [
    // position: vec3 at offset 0
    VertexInputAttributeDescription {
        location: 0,
        binding: 0,
        format: Format::R32G32B32_SFLOAT,
        offset: 0,
    },
    // color: vec3 at offset 12
    VertexInputAttributeDescription {
        location: 1,
        binding: 0,
        format: Format::R32G32B32_SFLOAT,
        offset: 12,
    },
];

let vertex_input = PipelineVertexInputStateCreateInfo::builder()
    .vertex_binding_descriptions(&[binding])
    .vertex_attribute_descriptions(&attributes);
```

### Step 3: Configure fixed-function state

```rust,ignore
use vulkan_rs::vk;
use vulkan_rs::vk::structs::*;
use vulkan_rs::vk::enums::*;
use vulkan_rs::vk::bitmasks::*;

let input_assembly = PipelineInputAssemblyStateCreateInfo::builder()
    .topology(PrimitiveTopology::TRIANGLE_LIST);

// Use dynamic viewport and scissor so we don't bake window size
// into the pipeline. Set them at draw time with cmd_set_viewport
// and cmd_set_scissor.
let viewport_state = PipelineViewportStateCreateInfo::builder()
    .viewport_count(1)
    .scissor_count(1);

let rasterizer = PipelineRasterizationStateCreateInfo::builder()
    .polygon_mode(PolygonMode::FILL)
    .cull_mode(CullModeFlags::BACK)
    .front_face(FrontFace::COUNTER_CLOCKWISE)
    .line_width(1.0);

let multisampling = PipelineMultisampleStateCreateInfo::builder()
    .rasterization_samples(SampleCountFlagBits::_1);

let depth_stencil = PipelineDepthStencilStateCreateInfo::builder()
    .depth_test_enable(1)
    .depth_write_enable(1)
    .depth_compare_op(CompareOp::LESS);

// No blending: write color directly.
let blend_attachment = PipelineColorBlendAttachmentState {
    blend_enable: 0,
    color_write_mask: ColorComponentFlags::R
        | ColorComponentFlags::G
        | ColorComponentFlags::B
        | ColorComponentFlags::A,
    ..unsafe { core::mem::zeroed() }
};

let color_blending = PipelineColorBlendStateCreateInfo::builder()
    .attachments(&[blend_attachment]);

// Dynamic state: viewport and scissor are set at draw time.
let dynamic_states = [
    DynamicState::VIEWPORT,
    DynamicState::SCISSOR,
];
let dynamic_state = PipelineDynamicStateCreateInfo::builder()
    .dynamic_states(&dynamic_states);
```

### Step 4: Create pipeline layout and pipeline

```rust,ignore
use vulkan_rs::vk;
use vulkan_rs::vk::structs::*;
use vulkan_rs::vk::handles::*;
use vulkan_rs::vk::Handle;

// Empty layout (no descriptor sets, no push constants).
let layout_info = PipelineLayoutCreateInfo::builder();
let pipeline_layout = unsafe {
    device.create_pipeline_layout(&layout_info, None)?
};

// Assemble everything into one create info.
let pipeline_info = GraphicsPipelineCreateInfo::builder()
    .stages(&stages)
    .vertex_input_state(&vertex_input)
    .input_assembly_state(&input_assembly)
    .viewport_state(&viewport_state)
    .rasterization_state(&rasterizer)
    .multisample_state(&multisampling)
    .depth_stencil_state(&depth_stencil)
    .color_blend_state(&color_blending)
    .dynamic_state(&dynamic_state)
    .layout(pipeline_layout)
    .render_pass(render_pass)
    .subpass(0);

// create_graphics_pipelines can create multiple pipelines at once.
let mut pipeline = Pipeline::null();
unsafe {
    device.create_graphics_pipelines(
        PipelineCache::null(),  // no cache for now
        &[*pipeline_info],
        None,
        &mut pipeline,
    )?;
};

// Shader modules can be destroyed after pipeline creation.
// The compiled code is baked into the pipeline.
unsafe {
    device.destroy_shader_module(vert_module, None);
    device.destroy_shader_module(frag_module, None);
};
```

### Step 5: Use in command recording

```rust,ignore
use vulkan_rs::vk;
use vulkan_rs::vk::enums::*;

unsafe {
    device.cmd_bind_pipeline(
        command_buffer,
        PipelineBindPoint::GRAPHICS,
        pipeline,
    );

    // Set dynamic state.
    device.cmd_set_viewport(command_buffer, 0, &[viewport]);
    device.cmd_set_scissor(command_buffer, 0, &[scissor]);

    // Draw.
    device.cmd_draw(command_buffer, vertex_count, 1, 0, 0);
};
```

## Compute pipelines

Compute pipelines are dramatically simpler: just a shader stage and
a pipeline layout. No vertex input, no rasterization, no blending.

```rust,ignore
use vulkan_rs::vk;
use vulkan_rs::vk::structs::*;
use vulkan_rs::vk::handles::*;
use vulkan_rs::vk::Handle;

let compute_info = ComputePipelineCreateInfo::builder()
    .stage(*PipelineShaderStageCreateInfo::builder()
        .stage(ShaderStageFlags::COMPUTE)
        .module(compute_module)
        .name(c"main"))
    .layout(compute_layout);

let mut compute_pipeline = Pipeline::null();
unsafe {
    device.create_compute_pipelines(
        PipelineCache::null(),
        &[*compute_info],
        None,
        &mut compute_pipeline,
    )?;
};
```

## Pipeline cache

Creating pipelines involves compiling shaders to GPU-specific machine
code. A pipeline cache stores this compiled output so subsequent
creations (in the same run or across runs, if you save/load the cache)
are faster.

```rust,ignore
use vulkan_rs::vk;
use vulkan_rs::vk::structs::*;

// Create a cache (optionally seeded with data from a previous run).
let cache_info = PipelineCacheCreateInfo::builder();
let cache = unsafe { device.create_pipeline_cache(&cache_info, None)? };

// Pass the cache when creating pipelines.
unsafe {
    device.create_graphics_pipelines(cache, &[*pipeline_info], None, &mut pipeline)?;
};

// At shutdown, retrieve cache data and save to disk for next run.
// (use get_pipeline_cache_data)
```

## Dynamic state (Vulkan 1.3)

By default, every setting in the pipeline is baked. Dynamic state lets
you mark specific settings as "set at draw time":

| Dynamic State | What it replaces |
|---------------|-----------------|
| `VIEWPORT` | Viewport in viewport state |
| `SCISSOR` | Scissor in viewport state |
| `LINE_WIDTH` | Line width in rasterization state |
| `DEPTH_TEST_ENABLE` | Depth test enable in depth/stencil state |
| `CULL_MODE` | Cull mode in rasterization state |
| `FRONT_FACE` | Front face in rasterization state |
| `PRIMITIVE_TOPOLOGY` | Topology in input assembly state |

Vulkan 1.3 made `VIEWPORT` and `SCISSOR` dynamic by convention (almost
everyone was using them dynamically anyway). More aggressive dynamic
state lets you consolidate pipelines: instead of separate pipelines for
different cull modes, use one pipeline with `CULL_MODE` dynamic.

## Formal reference

### Graphics pipeline stages (in order)

| Stage | State struct | Required? |
|-------|-------------|-----------|
| Vertex input | `PipelineVertexInputStateCreateInfo` | Yes |
| Input assembly | `PipelineInputAssemblyStateCreateInfo` | Yes |
| Tessellation | `PipelineTessellationStateCreateInfo` | Only with tessellation shaders |
| Viewport | `PipelineViewportStateCreateInfo` | Yes (unless rasterizer discards) |
| Rasterization | `PipelineRasterizationStateCreateInfo` | Yes |
| Multisample | `PipelineMultisampleStateCreateInfo` | Yes |
| Depth/stencil | `PipelineDepthStencilStateCreateInfo` | If render pass has depth attachment |
| Color blend | `PipelineColorBlendStateCreateInfo` | If render pass has color attachments |
| Dynamic | `PipelineDynamicStateCreateInfo` | Optional |

### Destruction order

1. Destroy pipelines before their pipeline layout.
2. Destroy pipeline layouts before their descriptor set layouts.
3. Shader modules can be destroyed immediately after pipeline creation.
4. Pipeline caches can be destroyed at any time (they are independent
   of the pipelines created through them).

### API reference links

- [`Pipeline`](https://docs.rs/vulkan-rs/latest/vulkan_rs/vk/struct.Pipeline.html)
- [`PipelineLayout`](https://docs.rs/vulkan-rs/latest/vulkan_rs/vk/struct.PipelineLayout.html)
- [`GraphicsPipelineCreateInfo`](https://docs.rs/vulkan-rs/latest/vulkan_rs/vk/struct.GraphicsPipelineCreateInfo.html)
- [`DynamicState`](https://docs.rs/vulkan-rs/latest/vulkan_rs/vk/struct.DynamicState.html)
- [Vulkan spec: Pipelines](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#pipelines)

## Key takeaways

- A graphics pipeline bakes all rendering state into one object: shaders,
  vertex layout, rasterization, blending, depth test, everything.
- You create pipelines at load time and switch between them with
  `cmd_bind_pipeline` during rendering. Zero compilation cost at draw time.
- Compute pipelines are much simpler: just a shader + layout.
- Dynamic state lets you defer certain settings to draw time, reducing
  the number of pipeline objects you need.
- Pipeline caches avoid redundant shader compilation across pipeline
  creations and across application runs.
