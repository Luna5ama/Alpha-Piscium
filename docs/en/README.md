# Alpha Piscium Developer Documentation

Language: English | [简体中文](../sc/README.md)

This directory documents Alpha Piscium's own architecture, resource conventions, and development workflow. It
intentionally does not repeat shader-pack interfaces, stage semantics, or property syntax already covered by
the [Iris Shaders documentation](https://github.com/IrisShaders/docs).

## Start here

1. [Development quick start](development/quick-start.md)
2. [Project structure](development/project-structure.md)
3. [Common workflows and configuration](development/workflows.md)
4. [Other scripts](development/scripts.md)

## Rendering pipeline and modules

- [Pipeline overview](modules/pipeline.md)
- [Geometry, G-buffer, and materials](modules/geometry-materials.md)
- [Shadows](modules/shadows.md)
- [Global illumination](modules/global-illumination.md)
- [Atmosphere, sky, and clouds](modules/atmosphere-clouds.md)
- [Water, caustics, and translucency](modules/water-translucency.md)
- [Post-processing and display transform](modules/post-processing.md)

## Documentation boundary

The documents focus on four project-local questions: where code lives, how passes enter the pipeline, which tool owns
each resource, and which script must run after a change. Generated files are called out explicitly and should not be
edited in place instead of their source.
