# Alpha Piscium
High-performance realistic Minecraft shaderpack featuring SSVBIL, volumetric clouds, atmospheric scattering, and stunning visual effects.

Official Website link: https://alphapiscium.org/ \
Discord server link: https://discord.gg/E2Uq2MmHgq

<img width="2560" height="1441" alt="2025-09-26_03 40 46" src="https://github.com/user-attachments/assets/58c281ec-120e-4376-a0e7-8bb1cde38f27" />

## Features
-  **Global Ilumination**
    - [*Screen Space Visibility Bitmask Indirect Lighting*](https://arxiv.org/pdf/2301.11376) (SSVBIL)
    - [*Ground Truth Visibility Bitmask Global Illumination*](https://www.shadertoy.com/view/lfdBWn) (GT-VBGI)
- Realistic, real-time, and fast **Atmospheric Scattering**
    - [*Unreal Engine Sky Atmosphere Rendering Technique*](https://sebh.github.io/publications/egsr2020.pdf)
    - [*Epipolar Sampling*](https://gdcvault.com/play/1018227/Practical-Implementation-of-Light-Scattering)
- High-quality, fast, and adaptive **Shadows**
    - [*Rectilinear Texture Warping*](https://www.cspaul.com/publications/Rosen.2012.I3D.pdf) (RTWSM)
- Real-time **Volumetric Clouds**
    - [*Nubis*](https://drive.google.com/file/d/0B-D275g6LH7LOE1RcVFERGpkS28/view?resourcekey=0-P04mYcVQ1lDPdn7FDunEIw)

## Installation
1. Download the Alpha Piscium shaders for Minecraft from the file section above.
2. Install Iris (You can download Iris on https://www.irisshaders.dev/download)
3. Open the Minecraft launcher, and choose the Iris profile you just created with the version you're playing
4. Launch Minecraft.
5. Go to "Options", then "Video Settings", and select "Shader Packs". Next, click on "Open Shader Pack Folder" to access the shaderpacks folder.
6. Move the downloaded ZIP file into this folder (.minecraft\shaderpacks).
7. In game, choose the newly installed pack from the shaders list. (If you select a shader pack using Iris, its filename will turn yellow.)
8. Click "Done" or "Apply" and all new features have been loaded.

## FAQ
- No lighting in the night: Please use a PBR resource packs. For vanilla style, SPBR is recommended. Otherwise, Patrix.
- Can't load shaderpack: Please use Iris with version 1.7+.
- Everything goes dark when turning back: This is due to the limitation of screenspace techniques, there isn't a perfect way to fix this.
- Intel GPU: Alpha Piscium uses cutting edge graphics technology. Unforunately Intel doesn't maintain their OpenGL driver well. Thus can't support Intel GPU.
- AMD GPU: Check if your GPU driver is newer than 22.7.1, if older please update your driver.
- Other loading error: Press Ctrl+D in Iris's shaderpack selection screen. Reload the shaderpack. Make an [issue](https://github.com/Luna5ama/Alpha-Piscium/issues) with the error message and the relevant files in `.minecraft/patched_shaders` as shown in the error message.

## License
Files under `script` directory are licensed under MIT license. All other files are licensed under GNU General Public License v3.0.
