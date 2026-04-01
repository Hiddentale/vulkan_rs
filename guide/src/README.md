# The vulkan_rs Guide

Vulkan is a powerful graphics and compute API, but its explicitness comes at
a cost: there is a lot to learn before you can put a single triangle on screen.
Most documentation dumps the full specification on you and expects you to swim.
This guide takes a different approach.

Every concept in this guide follows the same progression:

1. **Why it matters**, the problem this concept solves, in plain language.
2. **Intuition**, a mental model, analogy, or diagram that builds the right
   picture *before* you see any code.
3. **Worked example**, annotated code you can read, run, and modify.
4. **Formal reference**, spec terminology, edge cases, and links to the
   rustdoc API reference for when you need the full picture.

This structure is deliberate. Research in cognitive science shows that
understanding develops from concrete to abstract, not the other way around.
We build your intuition first, then formalize it.

## Who this guide is for

You know Rust. You have some idea that GPUs exist and do interesting things.
You may or may not have used OpenGL, DirectX, Metal, or WebGPU before, none
of that is required. This guide assumes zero prior Vulkan knowledge.

If you are coming from another Vulkan crate like `ash`, the
[migration guide](how-to/migrate-from-ash.md) shows the differences
side by side.

## How this guide is organized

This guide follows the [Diataxis](https://diataxis.fr/) documentation
framework, which separates content by *purpose*:

| Section | Purpose | Start here if... |
|---------|---------|------------------|
| **[Getting Started](getting-started/installation.md)** | Step-by-step tutorials | You want to draw something *now* |
| **[Concepts](concepts/how-to-read.md)** | Explanations of how Vulkan works | You want to understand *why* |
| **[How-To Guides](how-to/textures.md)** | Recipes for specific tasks | You know what you need to do |
| **[Architecture](architecture/design.md)** | Design decisions behind vulkan_rs | You want to contribute or evaluate |

### Concept dependency map

The concepts section is ordered so each chapter builds on the ones before it.
Here is the dependency structure:

```text
Object Model
    |
    +---> Memory Management
    |         |
    |         v
    +---> Command Buffers ----+
    |         |               |
    |         v               v
    +---> Synchronization   Render Passes
    |         |               |
    |         v               v
    +---> Pipelines <---------+
    |         |
    |         v
    +---> Descriptor Sets
    |
    +---> pNext Extension Chain  (independent, read any time)
    +---> Validation Layers      (independent, read any time)
```

You can read linearly from top to bottom, or jump to whatever you need.
The dependency map shows you which chapters you should read first if
something doesn't make sense.

## API documentation

This guide is a companion to the [API reference](https://docs.rs/vulkan-rs).
The API docs cover *every* type, method, and constant with spec links,
error codes, safety requirements, and thread safety annotations.
This guide covers the *why* and *how* that API docs cannot.

## Quick taste

Here is the minimum code to initialize Vulkan with `vulkan_rs`:

```rust,no_run
use vulkan_rs::{Entry, LibloadingLoader};

fn main() {
    // Load the Vulkan loader library from the system.
    let loader = unsafe { LibloadingLoader::new() }.expect("Failed to find Vulkan");
    let entry = unsafe { Entry::new(loader) }.expect("Failed to load Vulkan");

    // Query the highest Vulkan version the driver supports.
    let version = entry.version().expect("Failed to query version");
    println!("Vulkan {}.{}.{}", version.major, version.minor, version.patch);
}
```

Ready to go further? Start with [Installation](getting-started/installation.md).
