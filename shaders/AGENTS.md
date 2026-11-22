# Changing shader behavior

## Find the executing pass

Read this branch's [program list](../scripts/programs.main.kts) for the entry, enabled settings, defines and dispatch. Follow its includes into the implementation. The [pipeline topic](../docs/rendering/pipeline.md) explains producer and consumer dependencies. For opaque GI, follow [direction generation](pass/composite/GIReSTIRInitalSampleHiZ.comp.glsl) into the [voxel tracing pass](pass/composite/GIReSTIRInitalSampleVoxelFallback.comp.glsl) before attributing cost or a missed hit.

Put new stage entries in [pass/](pass/) and reusable rendering code in its [technique](techniques/). Put helpers used by multiple modules in [util/](util/). Follow existing Iris wrappers to their entry. Voxy supplies its own [opaque](voxy_opaque.glsl) and [translucent](voxy_translucent.glsl) fragment hooks.

## Preserve the rendering contract

- For a resource change, trace its format, binding, coordinates, packing and lifetime through allocation, clear, writes and every read. For history, include jitter, current and previous transforms, reprojection and reset. Edit the owning [tile layout](shadesmith.json) or [Iris declarations](../scripts/shaders.properties) and regenerate through the [scripts guide](../scripts/AGENTS.md).
- For sampling changes, keep the proposal, PDF, weighting and visibility test consistent. Preserve the distinction between a miss and an exhausted trace where they contribute differently. Use direct computation, explicit selection or shared memory for runtime indexing of local or constant arrays.
- Give Iris `workGroups` and `workGroupsRender` literal components or macros that expand to literals. Select variants with the preprocessor. Entries grouped in one `pass(...)` may execute concurrently. Put a consumer that depends on another entry's writes in a later slot.

## Check the result

An edit to included GLSL can be reloaded directly in Iris. Check enabled and disabled setting branches and use a scene that exposes the change. For temporal work, include motion, disocclusion and history reset. Inspect an affected intermediate alongside the final image when later filters could mask an error.
