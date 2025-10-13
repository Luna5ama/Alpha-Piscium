# Alpha Piscium
Experimental Minecraft shaderpack. ~~(aka. graphics tech demo)~~

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

## FAQ
- No lighting in the night: Please use a PBR resource packs. For vanilla style, SPBR is recommended. Otherwise, Patrix.
- Can't load shaderpack: Please use Iris with version 1.7+.
- Everything goes dark when turning back: This is due to the limitation of screenspace techniques, there isn't a perfect way to fix this.
- Intel GPU: Alpha Piscium uses cutting edge graphics technology. Unforunately Intel doesn't maintain their OpenGL driver well. Thus can't support Intel GPU.
- Error while loading on AMD GPU and other error: Press Ctrl+D in Iris's shaderpack selection screen. Reload the shaderpack. Make an [issue](https://github.com/Luna5ama/Alpha-Piscium/issues) with the error message and the relevant files in `.minecraft/patched_shaders` as shown in the error message.

## License
Files under `script` directory are licensed under MIT license. All other files are licensed under GNU General Public License v3.0.
