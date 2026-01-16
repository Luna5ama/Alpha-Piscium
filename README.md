# Alpha Piscium
High-quality realistic Minecraft shaderpack featuring global illumination, volumetric clouds, atmospheric scattering, and stunning visual effects.

Official Website link: https://alphapiscium.org/ \
Discord server link: https://discord.gg/E2Uq2MmHgq

<img width="1920" height="1080" alt="Alpha Piscium" src="https://github.com/user-attachments/assets/bab05ffa-9c46-4f9b-a69c-94bc6c82f3d5" />

## Features
-  **Real-time Global Illumination**
    - [*Reservoir-based SpatioTemporal Importance Resampling*](https://research.nvidia.com/sites/default/files/pubs/2020-07_Spatiotemporal-reservoir-resampling/ReSTIR.pdf) (ReSTIR) based Screen-Space Global Illumination
    - [*ReBLUR*](https://doi.org/10.1007/978-1-4842-7185-8_49) based denoising
- Realistic, real-time, and fast **Atmospheric Scattering**
    - [*Unreal Engine Sky Atmosphere Rendering Technique*](https://sebh.github.io/publications/egsr2020.pdf)
    - [*Epipolar Sampling*](https://gdcvault.com/play/1018227/Practical-Implementation-of-Light-Scattering)
- Real-time **Volumetric Clouds**
    - [*Nubis*](https://www.guerrilla-games.com/read/nubis-evolved)
- High-quality adaptive **Shadows**
    - [*Rectilinear Texture Warping*](https://www.cspaul.com/publications/Rosen.2012.I3D.pdf) (RTWSM)

## Installation
1. Download the Alpha Piscium shaders for Minecraft from [Modrinth](https://modrinth.com/shader/alpha-piscium) or [GitHub Release](https://github.com/Luna5ama/Alpha-Piscium/releases).
2. Install [Iris](https://www.irisshaders.dev/download) shader loader.
3. Open the Minecraft launcher and choose the Iris profile you just created with the version you're playing.
4. Launch Minecraft.
5. Go to **Options** → **Video Settings** → **Shader Packs**, then click **Open Shader Pack Folder**.
6. Move the downloaded ZIP file into this folder (`.minecraft/shaderpacks`).
7. In-game, select the newly installed pack from the shaders list. (If you select a shader pack using Iris, its filename will turn yellow.)
8. Click **Done** or **Apply** and the shader will be loaded.

## FAQ
- **No lighting at night**: Please use PBR resource packs. For vanilla style, SPBR is recommended. Otherwise, try Patrix.
- **Can't load shaderpack**: Please use Iris with version 1.7 or higher.
- **Everything goes dark when turning back**: This is due to the limitations of screenspace techniques. There isn't a perfect way to fix this.
- **Intel GPU**: Alpha Piscium uses cutting-edge graphics technology. Unfortunately, Intel doesn't maintain their OpenGL driver well, so Intel GPUs are not supported.
- **AMD GPU**: Check if your GPU driver is newer than 22.7.1. If older, please update your driver.
- **Other loading errors**: Press Ctrl+D in Iris's shaderpack selection screen, then reload the shaderpack. Make an [issue](https://github.com/Luna5ama/Alpha-Piscium/issues) with the error message and the relevant files in `.minecraft/patched_shaders` as shown in the error message.

## License
Files under the `scripts` directory are licensed under the MIT License. All other files are licensed under the GNU General Public License v3.0.
