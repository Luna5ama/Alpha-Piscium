"""Rebuild AP's Blender 5.0 AgX GPU transforms with OpenColorIO 2.6.0."""
import argparse
from pathlib import Path
import re

import numpy as np
import PyOpenColorIO as ocio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Blender v5.0.0 release/datafiles/colormanagement/config.ocio")
    args = parser.parse_args()
    if ocio.__version__ != "2.6.0":
        raise ValueError(f"Expected OpenColorIO 2.6.0, found {ocio.__version__}")

    config = ocio.Config.CreateFromFile(str(args.config))
    shaders = Path(__file__).resolve().parents[1] / "shaders"
    views = [("sRGB", "AgX", "SDR"), ("Rec.2100-HLG", "AgX - HDR 1000 nits", "HDR")]
    for display, view, tag in views:
        processor = config.getProcessor("ACES2065-1", display, view, ocio.TRANSFORM_DIR_FORWARD)
        desc = ocio.GpuShaderDesc.CreateShaderDesc()
        desc.setLanguage(ocio.GPU_LANGUAGE_GLSL_4_0)
        processor.getDefaultGPUProcessor().extractGpuShaderInfo(desc)
        texture, = desc.get3DTextures()
        if texture.edgeLen != 57 or list(desc.getTextures()):
            raise ValueError(f"Unexpected lookup texture layout for {display}/{view}")
        source = np.asarray(texture.getValues()).reshape(-1, 3)
        rgba = np.ones((source.shape[0], 4), dtype="<f2")
        rgba[:, :3] = source
        (shaders / "textures" / f"agx-{tag.lower()}.bin").write_bytes(rgba.tobytes())

        code = desc.getShaderText().replace(texture.samplerName, "usam_agx").replace("OCIOMain", "blenderAgX")
        # OCIO supplies tetrahedral interpolation. Fetch each corner exactly on Iris.
        code, replacements = re.subn(
            r"texture\(usam_agx, (\w+)\)",
            r"texelFetch(usam_agx, clamp(ivec3(floor(\1 * 57.0)), ivec3(0), ivec3(56)), 0)", code)
        if not replacements:
            raise ValueError(f"No tetrahedral lookup found for {view}")
        header = f"// OpenColorIO {ocio.__version__} generated from Blender 5.0.0, {display} / {view}.\n"
        lines = [line.rstrip() for line in (header + code).splitlines()]
        path = shaders / "techniques" / "displaytransform" / f"AgX{tag}.glsl"
        path.write_bytes(("\n".join(lines).rstrip() + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
