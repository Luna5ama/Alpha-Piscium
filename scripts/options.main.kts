@file:Import("options.lib.kts")

import java.io.File
import kotlin.io.path.Path
import kotlin.io.path.readLines
import kotlin.math.pow

options(File("shaders.properties"), File("../shaders"), "base/Options.glsl") {
    mainScreen(2) {
        screen("TERRAIN", 2) {
            lang {
                name = "Terrain"
            }
            screen("BLOCKLIGHT", 1) {
                lang {
                    name = "Block Lighting"
                }
                slider("SETTING_FIRE_TEMPERATURE", 1400, 100..5000 step 100) {
                    lang {
                        name = "Fire Temperature"
                        comment =
                            "Controls the color temperature of fire in Kelvin. Default: 1400 K (based on real fire). Higher values produce whiter/bluer light."
                    }
                }
                slider("SETTING_LAVA_TEMPERATURE", 1300, 100..5000 step 100) {
                    lang {
                        name = "Lava Temperature"
                        comment =
                            "Controls the color temperature of lava in Kelvin. Default: 1300 K (based on real lava). Higher values produce whiter/bluer light."
                    }
                }
                slider("SETTING_EMISSIVE_STRENGTH", 4.0, 0.0..8.0 step 0.25) {
                    lang {
                        name = "Emissive Brightness"
                        comment = "Global brightness multiplier for all light-emitting materials and blocks."
                    }
                }
                slider("SETTING_PARTICLE_EMISSIVE_STRENGTH", 0.0, 0.0..1.0 step 0.1) {
                    lang {
                        name = "Particle Emissive Intensity"
                        comment = "Brightness multiplier for glowing particles like torches and fires."
                    }
                }
                slider("SETTING_ENTITY_EMISSIVE_STRENGTH", 0.2, 0.0..1.0 step 0.1) {
                    lang {
                        name = "Entity Emissive Intensity"
                        comment = "Brightness multiplier for glowing entities like blazes and magma cubes."
                    }
                }
                empty()
                slider("SETTING_EMISSIVE_PBR_VALUE_CURVE", 0.9, 0.1..4.0 step 0.05) {
                    lang {
                        name = "PBR Resource Pack Emissive Contrast"
                        comment =
                            "Adjusts contrast of emissive values from PBR resource packs. Higher values create stronger differences between bright and dim areas."
                    }
                }
                slider("SETTING_EMISSIVE_ALBEDO_COLOR_CURVE", 2.0, 0.1..4.0 step 0.05) {
                    lang {
                        name = "Emissive Color Saturation"
                        comment =
                            "Controls color intensity of emissive materials. Higher values produce more vibrant, saturated colors."
                    }
                }
                slider("SETTING_EMISSIVE_ALBEDO_LUM_CURVE", 0.5, 0.0..4.0 step 0.05) {
                    lang {
                        name = "Color Texture-Based Emission Strength"
                        comment =
                            "Controls how much the base texture brightness affects emission. Higher values make brighter textures glow more intensely."
                    }
                }
                empty()
                slider("SETTING_EMISSIVE_ARMOR_GLINT_MULT", -10, -20..0 step 1) {
                    lang {
                        name = "Enchanted Armor Glow"
                        prefix = "2^"
                        comment = "Brightness of the enchanted armor glint effect. The actual multiplier is 2^x."
                    }
                }
                slider("SETTING_EMISSIVE_ARMOR_GLINT_CURVE", 1.3, 0.1..2.0 step 0.1) {
                    lang {
                        name = "Enchanted Armor Glow Contrast"
                        comment =
                            "Adjusts contrast of enchanted armor glint. Higher values make the brightest parts more prominent."
                    }
                }
            }
            screen("NORMAL_MAPPING", 1) {
                lang {
                    name = "Normal Mapping"
                }
                toggle("SETTING_NORMAL_MAPPING", true) {
                    lang {
                        name = "Enable Normal Mapping"
                        comment =
                            "Enables surface detail from normal maps, adding depth and texture to blocks without additional geometry."
                    }
                }
                slider("SETTING_NORMAL_MAPPING_STRENGTH", 0.0, -5.0..5.0 step 0.5) {
                    lang {
                        name = "Normal Mapping Strength"
                        prefix = "2^"
                        comment =
                            "Controls the intensity of surface detail effects. Higher values increase depth perception. The actual strength is 2^x."
                    }
                }
            }
            screen("SPECULAR_MAPPING", 1) {
                lang {
                    name = "Specular Mapping"
                }
                slider("SETTING_MINIMUM_F0", 12, 4..32) {
                    lang {
                        name = "Minimum Reflectivity (F0)"
                        prefix = "2^-"
                        comment =
                            "Sets the baseline reflectivity for all materials. Higher values make surfaces more reflective overall. The actual value is calculated as 2^-x."
                    }
                }
                empty()
                slider("SETTING_SOLID_MINIMUM_ROUGHNESS", 6, 4..16) {
                    lang {
                        name = "Minimum Solid Roughness"
                        prefix = "2^-"
                        comment =
                            "The smoothest (most mirror-like) that solid blocks can appear. Higher values allow sharper reflections. The actual value is calculated as 2^-x."
                    }
                }
                slider("SETTING_SOLID_MAXIMUM_ROUGHNESS", 5, 2..16) {
                    lang {
                        name = "Maximum Solid Roughness"
                        prefix = "1-2^-"
                        comment =
                            "The roughest (most diffuse) that solid blocks can appear. Higher values allow more matte surfaces. The actual value is calculated as 1-2^-x."
                    }
                }
                empty()
                slider("SETTING_WATER_ROUGHNESS", 9.0, 4.0..12.0 step 0.5) {
                    lang {
                        name = "Water Surface Roughness"
                        prefix = "2^-"
                        comment =
                            "Controls how smooth and reflective water appears. Lower values create calmer, more mirror-like water. The actual value is calculated as 2^-x."
                    }
                }
                slider("SETTING_TRANSLUCENT_ROUGHNESS_REDUCTION", 1.0, 0.0..8.0 step 0.5) {
                    lang {
                        name = "Translucent Roughness Reduction"
                        prefix = "2^-"
                        comment =
                            "Makes translucent blocks (such as glass) smoother than their resource pack values. Higher values create more mirror-like appearances. The actual value is calculated as 2^-x."
                    }
                }
                slider("SETTING_TRANSLUCENT_MINIMUM_ROUGHNESS", 10.0, 4.0..16.0 step 0.5) {
                    lang {
                        name = "Translucent Minimum Roughness"
                        prefix = "2^-"
                        comment =
                            "The smoothest that translucent blocks (such as glass) can appear. Higher values allow sharper reflections on translucent. The actual value is calculated as 2^-x."
                    }
                }
                slider("SETTING_TRANSLUCENT_MAXIMUM_ROUGHNESS", 5.0, 1.0..16.0 step 0.5) {
                    lang {
                        name = "Translucent Maximum Roughness"
                        prefix = "2^-"
                        comment =
                            "The roughest that translucent blocks (such as glass) can appear. Higher values allow more frosted glass effects. The actual value is calculated as 2^-x."
                    }
                }
                empty()
                slider("SETTING_MAXIMUM_SPECULAR_LUMINANCE", 65536, powerOfTwoRange(8..24)) {
                    lang {
                        name = "Maximum Specular Luminance"
                        comment =
                            "Limits how bright reflections and highlights can be (in 1000 cd/m²). Prevents overly intense glare from very smooth surfaces."
                    }
                }
            }
            screen("SSS", 1) {
                slider("SETTING_SSS_STRENGTH", 1.2, 0.0..5.0 step 0.1) {
                    lang {
                        name = "Strength"
                        comment =
                            "Overall intensity of light passing through semi-transparent materials like leaves, creating a soft glow effect."
                    }
                }
                slider("SETTING_SSS_HIGHLIGHT", 0.8, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Sheen"
                        comment =
                            "Intensity of the soft sheen highlight on materials with subsurface scattering, like leaves in sunlight."
                    }
                }
                slider("SETTING_SSS_SCTR_FACTOR", 4.0, 0.0..10.0 step 0.1) {
                    lang {
                        name = "Scatter Factor"
                        comment =
                            "How much light scatters inside semi-transparent materials. Lower values create a stronger glow-through effect."
                    }
                }
                empty()
                slider("SETTING_SSS_DIFFUSE_RANGE", 0.3, 0.0..4.0 step 0.1) {
                    lang {
                        name = "Glow Spread"
                        comment =
                            "How far the glow effect spreads across the surface. Higher values create a more diffused, softer appearance."
                    }
                }
                slider("SETTING_SSS_DEPTH_RANGE", 0.6, 0.0..4.0 step 0.1) {
                    lang {
                        name = "Material Thickness"
                        comment =
                            "How deep light penetrates into the material. Higher values simulate thicker, more translucent materials."
                    }
                }
                slider("SETTING_SSS_MAX_DEPTH_RANGE", 0.9, 0.0..4.0 step 0.1) {
                    lang {
                        name = "Maximum Thickness"
                        comment = "Upper limit for how thick materials can appear for light penetration calculations."
                    }
                }
            }
            empty()
            empty()
            screen("SHADOW", 2) {
                lang {
                    name = "Shadows"
                }
                slider("SETTING_SHADOW_MAP_RESOLUTION", 2048, listOf(1024, 2048, 3072, 4096)) {
                    lang {
                        name = "Shadow Map Resolution"
                        comment = "Higher values produce sharper, more detailed shadows but reduce performance."
                    }
                }
                constSlider("shadowDistance", 192.0, listOf(64.0, 128.0, 192.0, 256.0, 384.0, 512.0)) {
                    lang {
                        name = "Shadow Render Distance"
                        comment = "How far from the player shadows map are rendered."
                        64.0 value "4 chunks"
                        128.0 value "8 chunks"
                        192.0 value "12 chunks"
                        256.0 value "16 chunks"
                        384.0 value "24 chunks"
                        512.0 value "32 chunks"
                    }
                }
                empty()
                empty()
                screen("RTWSM", 1) {
                    lang {
                        name = "RTWSM"
                        comment =
                            "Rectilinear Texture Warping Shadow Mapping settings. A advanced techniques that allocate more shadow details adaptively based on scene and view."
                    }
                    slider("SETTING_RTWSM_IMAP_SIZE", 256, listOf(256, 512, 1024)) {
                        lang {
                            name = "Importance Map Resolution"
                            comment =
                                "Resolution for analyzing where shadows need more detail. Higher values improve accuracy but reduce performance."
                        }
                    }
                    empty()
                    toggle("SETTING_RTWSM_F", true) {
                        lang {
                            name = "Forward Importance Analysis"
                        }
                    }
                    slider("SETTING_RTWSM_F_BASE", 1.0, 0.1..10.0 step 0.1) {
                        lang {
                            name = "Forward Base Value"
                        }
                    }
                    slider("SETTING_RTWSM_F_MIN", -20, -20..0) {
                        lang {
                            name = "Forward Min Value"
                            comment =
                                "Minimum importance value for forward importance analysis. The actual minimum value is calculated as 2^x."
                            prefix = "2^"
                        }
                    }
                    slider("SETTING_RTWSM_F_D", 0.5, 0.0..2.0 step 0.05) {
                        lang {
                            name = "Forward Distance Function"
                            comment = "Reduces weight based on distance. Larger setting value means slower decay."
                        }
                    }
                    empty()
                    toggle("SETTING_RTWSM_B", true) {
                        lang {
                            name = "Backward Importance Analysis"
                        }
                    }
                    slider("SETTING_RTWSM_B_BASE", 5.0, 0.1..10.0 step 0.1) {
                        lang {
                            name = "Backward Base Value"
                        }
                    }
                    slider("SETTING_RTWSM_B_MIN", -10, -20..0) {
                        lang {
                            name = "Backward Min Value"
                            comment =
                                "Minimum importance value for backward importance analysis. The actual minimum value is calculated as 2^x."
                            prefix = "2^"
                        }
                    }
                    slider("SETTING_RTWSM_B_D", 0.6, 0.0..2.0 step 0.05) {
                        lang {
                            name = "Backward Distance Function"
                            comment = "Reduces weight based on distance. Larger setting value means slower decay."
                        }
                    }
                    slider("SETTING_RTWSM_B_P", 4.0, 0.0..10.0 step 0.5) {
                        lang {
                            name = "Backward Perpendicular Function"
                            comment = "Adds extra weight to surface perpendicular to light direction."
                        }
                    }
                    slider("SETTING_RTWSM_B_PP", 16, (0..8).map { 1 shl it }) {
                        lang {
                            name = "Backward Perpendicular Function Power"
                        }
                    }
                    slider("SETTING_RTWSM_B_SN", 2.0, 0.0..10.0 step 0.5) {
                        lang {
                            name = "Backward Surface Normal Function"
                            comment = "Adds extra weight to surface directly facing towards camera."
                        }
                    }
                    slider("SETTING_RTWSM_B_SE", 0.0, 0.0..10.0 step 0.5) {
                        lang {
                            name = "Backward Shadow Edge Function"
                            comment = "Adds extra weight for shadow edges."
                        }
                    }
                }
                screen("PCSS", 1) {
                    lang {
                        name = "Soft Shadows"
                        comment = "Realistic soft shadow edges based on distance from the shadow caster using PCSS"
                    }
                    slider("SETTING_PCSS_BLOCKER_SEARCH_COUNT", 2, listOf(1, 2, 4, 8, 16)) {
                        lang {
                            name = "Blocker Search Count"
                            comment =
                                "Number of samples used to determine shadow softness. Higher values improve quality but reduce performance."
                        }
                    }
                    slider("SETTING_PCSS_BLOCKER_SEARCH_LOD", 4, 0..8) {
                        lang {
                            name = "Blocker Search LOD"
                        }
                    }
                    empty()
                    slider("SETTING_PCSS_BPF", 0.0, 0.0..10.0 step 0.5) {
                        lang {
                            name = "Base Penumbra (blur) Factor"
                            comment = "Constant amount of blur applied to all shadows, regardless of distance."
                        }
                    }
                    slider("SETTING_PCSS_VPF", 1.0, 0.0..2.0 step 0.1) {
                        lang {
                            name = "Variable Penumbra Factor"
                            comment =
                                "How much shadows blur based on distance from the caster. Multiplied by sun size - larger sun creates softer shadows."
                        }
                    }
                }
            }
            screen("VBGI", 1) {
                lang {
                    name = "VBGI"
                    comment = "Advanced screen space technique that creates realistic indirect lighting"
                }
                slider("SETTING_VBGI_STEPS", 32, listOf(8, 12, 16, 24, 32, 64, 96, 128)) {
                    lang {
                        name = "Step Samples"
                        comment =
                            "Number of samples for GI sampling. Lower values may cause light leaks, higher values improve quality but reduce performance."
                    }
                }
                slider("SETTING_VBGI_FALLBACK_SAMPLES", 8, powerOfTwoRange(1..5)) {
                    lang {
                        name = "Fallback Samples"
                        comment =
                            "Number of samples used to sample environment and sky probe. Higher value increases quality but also decreases performance."
                    }
                }
                empty()
                slider("SETTING_VBGI_RADIUS", 64, (0..8).map { 1 shl it }) {
                    lang {
                        name = "Sample Radius"
                    }
                }
                slider("SETTING_VBGI_MAX_RADIUS", 128, (0..8).map { 1 shl it }) {
                    lang {
                        name = "Max Sample Radius"
                    }
                }
                slider("SETTING_VBGI_THICKNESS", 0.25, 0.1..1.0 step 0.01) {
                    lang {
                        name = "Thickness"
                        comment =
                            "Assumed thickness of surfaces for shadow calculations. Higher values create stronger shadowing and less light leaking."
                    }
                }
                empty()
                toggle("SETTING_VBGI_PROBE_HQ_OCC", true) {
                    lang {
                        name = "High Quality Probe Lighting Occlusion"
                        comment =
                            "Performs additional shadow checks for environment lighting, improving shadow accuracy but reducing performance slightly."
                    }
                }
                slider("SETTING_VBGI_PROBE_DIR_MATCH_WEIGHT", 1, -10..10) {
                    lang {
                        name = "Environment Probe Direction Matching Strictness"
                        comment =
                            "Higher values reduce incorrect lighting by requiring better direction alignment, preventing light from wrong angles."
                    }
                }
                slider("SETTING_VBGI_PROBE_FADE_START_DIST", 16, 0..32 step 4) {
                    lang {
                        name = "Environment Probe Fade Start"
                        comment = "Distance in blocks where environment probe lighting begins to fade out."
                    }
                }
                slider("SETTING_VBGI_PROBE_FADE_END_DIST", 32, 0..64 step 4) {
                    lang {
                        name = "Environment Probe Fade End"
                        comment = "Distance in blocks where environment probe lighting is completely faded out."
                    }
                }
                empty()
                toggle("SETTING_VBGI_MC_SKYLIGHT_ATTENUATION", true) {
                    lang {
                        name = "Vanilla Skylight Attenuation"
                        comment =
                            "Uses Minecraft's built-in skylight values to reduce sky lighting in enclosed spaces."
                    }
                }
                empty()
                slider("SETTING_VBGI_SKYLIGHT_STRENGTH", 1.0, 0.0..4.0 step 0.05) {
                    lang {
                        name = "Sky Light Intensity"
                        comment = "Brightness of light coming from the sky."
                    }
                }
                slider("SETTING_VGBI_ENV_STRENGTH", 1.0, 0.0..4.0 step 0.05) {
                    lang {
                        name = "Environment Light Intensity"
                        comment = "Brightness of indirect light from the environment probe."
                    }
                }
                slider("SETTING_VGBI_IB_STRENGTH", 1.0, 0.0..4.0 step 0.05) {
                    lang {
                        name = "Indirect Bounce Intensity"
                        comment = "Brightness of light that bounces off surfaces to illuminate other areas."
                    }
                }
                empty()
                slider("SETTING_VBGI_DGI_STRENGTH", 1.0, 0.0..4.0 step 0.05) {
                    lang {
                        name = "Diffuse Bounce Intensity"
                        comment = "Intensity of indirect lighting on matte (non-reflective) surfaces."
                    }
                }
                slider("SETTING_VBGI_SGI_STRENGTH", 1.0, 0.0..4.0 step 0.05) {
                    lang {
                        name = "Reflective Bounce Intensity"
                        comment = "Intensity of indirect lighting in reflections and specular highlights."
                    }
                }
                slider("SETTING_VBGI_GI_MB", 1.0, 0.0..2.0 step 0.01) {
                    lang {
                        name = "Multi-Bounce Multiplier"
                        comment =
                            "Simulates light bouncing multiple times. Higher values brighten scenes by allowing more light bounces."
                    }
                }
            }
            screen("DENOISER", 1) {
                lang {
                    name = "Denoiser"
                }
                toggle("SETTING_DENOISER", true) {
                    lang {
                        name = "Denoiser"
                        comment =
                            "Smooths out grainy artifacts in lighting, especially noticeable in dark areas or with global illumination."
                    }
                }
                slider("SETTING_DENOISER_REPROJ_NORMAL_EDGE_WEIGHT", 1.0, 0.0..16.0 step 0.1) {
                    lang {
                        name = "Reprojection Normal Edge Weight"
                        comment =
                            "How strictly to preserve detail at surface angle changes. Higher values keep sharper edges but may show more noise."
                    }
                }
                slider("SETTING_DENOISER_REPROJ_GEOMETRY_EDGE_WEIGHT", 9.0, 0.0..16.0 step 0.1) {
                    lang {
                        name = "Reprojection Geometry Edge Weight"
                        comment =
                            "How strictly to preserve detail at object edges. Higher values keep sharper edges but may show more noise."
                    }
                }
                empty()
                slider("SETTING_DENOISER_MAX_ACCUM", 256, (2..10).map { 1 shl it }) {
                    lang {
                        name = "Max Accumulation"
                        comment =
                            "Maximum frames to accumulate. Higher values create smoother results but may cause ghosting during movement."
                    }
                }
                slider("SETTING_DENOISER_ACCUM_DECAY", 1.0, 0.5..2.0 step 0.01) {
                    lang {
                        name = "Accumulation Decay"
                        comment =
                            "Current mix rate decay factor for temporal accumulation. Larger value means faster decay."
                    }
                }
                empty()
                slider("SETTING_DENOISER_MAX_FAST_ACCUM", 16, 1..32 step 1) {
                    lang {
                        name = "Max Fast Accumulation"
                    }
                }
                slider("SETTING_DENOISER_FAST_HISTORY_CLAMPING_THRESHOLD", 2.0, 1.0..4.0 step 0.1) {
                    lang {
                        name = "Fast History Clamping Threshold"
                        comment =
                            "Prevents ghosting during movement and light update. Higher values reduce trails but may show more flickering."
                    }
                }
                empty()
                slider("SETTING_DENOISER_VARIANCE_BOOST_ADD_FACTOR", 10, 0..64) {
                    lang {
                        name = "Initial Noise Smoothing"
                        comment =
                            "Extra smoothing applied when first entering an area. Higher values smooth faster. (Actual value: 2^-x)"
                    }
                }
                slider("SETTING_DENOISER_VARIANCE_BOOST_MULTIPLY", 2.5, 1.0..4.0 step 0.1) {
                    lang {
                        name = "Initial Smoothing Multiplier"
                        comment =
                            "Multiplier for extra smoothing when first entering an area. Higher values smooth more aggressively."
                    }
                }
                slider("SETTING_DENOISER_VARIANCE_BOOST_FRAMES", 16, (0..6).map { 1 shl it }) {
                    lang {
                        name = "Initial Smoothing Duration"
                        comment = "How many frames to apply extra smoothing when first entering an area."
                    }
                }
                slider("SETTING_DENOISER_VARIANCE_BOOST_DECAY", 2, 1..16 step 1) {
                    lang {
                        name = "Initial Smoothing Fade Speed"
                        comment =
                            "How quickly the initial extra smoothing fades away. Higher values fade faster."
                    }
                }
                empty()
                slider("SETTING_DENOISER_MIN_VARIANCE_FACTOR", 25, 0..64) {
                    lang {
                        name = "Minimum Variance Factor"
                        comment =
                            "Minimum amount of smoothing always applied. Lower values create smoother but potentially blurrier results. (Used as: max(variance, 2^-x))"
                    }
                }
                empty()
                slider("SETTING_DENOISER_FILTER_NORMAL_WEIGHT", 128, (0..10).map { 1 shl it }) {
                    lang {
                        name = "Filter Normal Weight"
                        comment =
                            "How much to preserve detail at surface angle changes. Higher values keep sharper edges between different surfaces."
                    }
                }
                slider("SETTING_DENOISER_FILTER_DEPTH_WEIGHT", 64, (0..10).map { 1 shl it }) {
                    lang {
                        name = "Filter Depth Weight"
                        comment =
                            "How much to preserve detail at depth changes. Higher values keep sharper edges between near and far objects."
                    }
                }
                slider("SETTING_DENOISER_FILTER_COLOR_WEIGHT", 56, 0..128) {
                    lang {
                        name = "Filter Color Weight"
                        comment =
                            "How much to preserve detail at color changes. Lower values smooth more but may blur color transitions."
                    }
                }
            }
        }
        screen("VOLUMETRICS", 2) {
            lang {
                name = "Volumetrics"
            }
            slider("SETTING_ATM_ALT_SCALE", 1000, listOf(1, 10, 100).flatMap { 1 * it..10 * it step it } + 1000) {
                lang {
                    name = "Altitude Scale"
                    comment = "Value of 1 means 1 block = 1 km, value of 10 means 10 blocks = 1 km, and so on."
                }
            }
            slider("SETTING_ATM_D_SCALE", 1000, listOf(1, 10, 100).flatMap { 1 * it..10 * it step it } + 1000) {
                lang {
                    name = "Distance Scale"
                    comment = "Value of 1 means 1 block = 1 km, value of 10 means 10 blocks = 1 km, and so on."
                }
            }
            empty()
            empty()
            slider("SETTING_EPIPOLAR_SLICES", 1024, listOf(256, 512, 1024, 2048)) {
                lang {
                    name = "Epipolar Slices"
                    comment =
                        "Number of epipolar slices used in volumetric lighting. Higher value increases quality but also decreases performance."
                }
            }
            slider("SETTING_SLICE_SAMPLES", 512, listOf(128, 256, 512, 1024)) {
                lang {
                    name = "Slice Samples"
                    comment =
                        "Number of samples per slice used in volumetric lighting. Higher value increases quality but also decreases performance."
                }
            }
            empty()
            empty()
            screen("AIR", 2) {
                lang {
                    name = "Air"
                }
                screen("MIE_COEFF", 1) {
                    lang {
                        name = "Mie Coefficients"
                        comment = "Controls propeties of Haze & Fog"
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY", 2.0, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Mie Turbidity"
                            prefix = "2^"
                            comment =
                                "Overall haziness/cloudiness of the atmosphere. Higher values create mistier, more atmospheric scenes. (Actual value: 2^x)"
                        }
                    }
                    toggle("SETTING_ATM_MIE_TIME", true) {
                        lang {
                            name = "Time of Day Mie Turbidity"
                            comment =
                                "Automatically adjusts atmospheric haze throughout the day (more haze at sunrise/sunset)."
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_EARLY_MORNING", 4.5, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Early Morning Turbidity"
                            prefix = "2^"
                            comment = "Atmospheric haze during early morning hours (before sunrise)."
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_SUNRISE", 5.25, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Sunrise Turbidity"
                            prefix = "2^"
                            comment =
                                "Atmospheric haze during sunrise, creating vivid colors and dramatic skies."
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_MORNING", 4.0, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Morning Turbidity"
                            prefix = "2^"
                            comment = "Atmospheric haze during morning hours (after sunrise)."
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_NOON", 2.0, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Noon Turbidity"
                            prefix = "2^"
                            comment = "Atmospheric haze at noon, typically clearest time of day."
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_AFTERNOON", 1.0, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Afternoon Turbidity"
                            prefix = "2^"
                            comment = "Atmospheric haze during afternoon hours."
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_SUNSET", 3.0, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Sunset Turbidity"
                            prefix = "2^"
                            comment =
                                "Atmospheric haze during sunset, creating vivid colors and dramatic skies."
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_NIGHT", 3.5, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Night Turbidity"
                            prefix = "2^"
                            comment = "Atmospheric haze during early night hours."
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_MIDNIGHT", 4.0, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Midnight Turbidity"
                            prefix = "2^"
                            comment = "Atmospheric haze at midnight."
                        }
                    }
                    empty()
                    slider("SETTING_ATM_MIE_SCT_MUL", 1.0, 0.0..5.0 step 0.05) {
                        lang {
                            name = "Mie Scattering Multiplier"
                            comment =
                                "How much light scatters in the haze. Higher values create brighter, more visible atmospheric haze."
                        }
                    }
                    slider("SETTING_ATM_MIE_ABS_MUL", 0.1, 0.0..2.0 step 0.01) {
                        lang {
                            name = "Mie Absorption Multiplier"
                            comment =
                                "How much light is absorbed by haze. Higher values create darker, more atmospheric fog."
                        }
                    }
                }
                screen("RAY_COEFF", 1) {
                    lang {
                        name = "Rayleigh Coefficients"
                    }
                    slider("SETTING_ATM_RAY_SCT_MUL", 1.0, 0.0..5.0 step 0.05) {
                        lang {
                            name = "Rayleigh Scattering Multiplier"
                            comment =
                                "Controls the intensity of blue sky color. Higher values create deeper, more saturated blue skies."
                        }
                    }
                    slider("SETTING_ATM_OZO_ABS_MUL", 1.0, 0.0..5.0 step 0.05) {
                        lang {
                            name = "Ozone Absorption Multiplier"
                            comment =
                                "Simulates ozone layer effects on sky color. Higher values enhance orange/red colors at sunrise and sunset."
                        }
                    }
                }
                slider("SETTING_ATM_GROUND_ALBEDO_R", 45, 0..255) {
                    lang {
                        name = "Ground Color - Red"
                        comment =
                            "Red component of light reflected from the ground into the atmosphere. Affects horizon and overall color."
                    }
                }
                slider("SETTING_ATM_GROUND_ALBEDO_G", 89, 0..255) {
                    lang {
                        name = "Ground Color - Green"
                        comment =
                            "Green component of light reflected from the ground into the atmosphere. Affects horizon and overall color."
                    }
                }
                slider("SETTING_ATM_GROUND_ALBEDO_B", 82, 0..255) {
                    lang {
                        name = "Ground Color - Blue"
                        comment =
                            "Blue component of light reflected from the ground into the atmosphere. Affects horizon and overall color."
                    }
                }
                empty()
                empty()
                empty()
                slider("SETTING_SKYVIEW_RES", 256, powerOfTwoRange(7..10)) {
                    lang {
                        name = "Sky View Resolution"
                        comment =
                            "Resolution of sky calculations. Higher values improve sky color accuracy but reduce performance."
                    }
                }
                toggle("SETTING_DEPTH_BREAK_CORRECTION", true) {
                    lang {
                        name = "Depth Break Correction"
                    }
                }
                empty()
                empty()
                slider("SETTING_SKY_SAMPLES", 32, 16..64 step 8) {
                    lang {
                        name = "Sky Samples"
                    }
                }
                slider("SETTING_LIGHT_SHAFT_SAMPLES", 12, 4..32 step 4) {
                    lang {
                        name = "Light Shaft Samples"
                        comment =
                            "Samples for volumetric light shafts (god rays). Higher values create smoother, more detailed rays but reduce performance."
                    }
                }
                slider("SETTING_LIGHT_SHAFT_SHADOW_SAMPLES", 8, 1..16 step 1) {
                    lang {
                        name = "Light Shaft Shadow Samples"
                        comment =
                            "Shadow samples in god rays. Higher values improve shadow accuracy in light shafts but reduce performance."
                    }
                }
                slider("SETTING_LIGHT_SHAFT_DEPTH_BREAK_CORRECTION_SAMPLES", 32, 8..64 step 8) {
                    lang {
                        name = "Light Shaft Depth Break Correction Samples"
                        comment =
                            "Shadow samples used in depth break correction. Higher values improve shadow accuracy in light shafts but reduce performance."
                    }
                }
                slider("SETTING_LIGHT_SHAFT_SOFTNESS", 5, 0..10 step 1) {
                    lang {
                        name = "Light Shaft Softness"
                        comment =
                            "How soft and diffused the light shafts appear. Higher values create more diffused, atmospheric rays."
                    }
                }
            }
            screen("CLOUDS_LIGHTING", 1) {
                lang {
                    name = "Cloud Lighting"
                }
                slider("SETTING_CLOUDS_MS_ORDER", 4, 1..10) {
                    lang {
                        name = "Multi-Scattering Order"
                    }
                }
                slider("SETTING_CLOUDS_MS_FALLOFF_SCTTERING", 0.55, 0.0..1.0 step 0.05) {
                    lang {
                        name = "Multi-Scattering Scattering Falloff"
                    }
                }
                slider("SETTING_CLOUDS_MS_FALLOFF_EXTINCTION", 0.6, 0.0..1.0 step 0.05) {
                    lang {
                        name = "Multi-Scattering Extinction Falloff"
                    }
                }
                slider("SETTING_CLOUDS_MS_FALLOFF_PHASE", 0.6, 0.0..1.0 step 0.05) {
                    lang {
                        name = "Multi-Scattering Phase Falloff"
                    }
                }
                slider("SETTING_CLOUDS_MS_FALLOFF_AMB", 0.1, 0.0..1.0 step 0.05) {
                    lang {
                        name = "Multi-Scattering Ambient Irradiance Falloff"
                    }
                }
                empty()
                slider("SETTING_CLOUDS_AMB_UNI_PHASE_RATIO", 0.5, 0.0..1.0 step 0.05) {
                    lang {
                        name = "Ambient Irradiance Uniform Phase Ratio"
                        comment =
                            "How evenly ambient light spreads in clouds. Higher values create more uniform, diffused ambient lighting."
                    }
                }
            }
            screen("LOW_CLOUDS", 1) {
                lang {
                    name = "Low Altitude Clouds"
                }
                toggle("SETTING_CLOUDS_CU", true) {
                    lang {
                        name = "Enable Cumulus Clouds"
                        comment = "Toggles puffy, volumetric clouds at lower altitudes."
                    }
                }
                toggle("SETTING_CLOUDS_LOW_UPSCALE_FACTOR", 4, 0..6) {
                    lang {
                        name = "Upscale Factor"
                        comment =
                            "Renders clouds at lower resolution then upscales. Higher values improve performance but may reduce detail."
                        0 value "1.0 x"
                        1 value "1.5 x"
                        2 value "2.0 x"
                        3 value "2.5 x"
                        4 value "3.0 x"
                        5 value "3.5 x"
                        6 value "4.0 x"
                    }
                }
                slider("SETTING_CLOUDS_LOW_MAX_ACCUM", 32, powerOfTwoRangeAndHalf(2..7)) {
                    lang {
                        name = "Max Accumulation"
                        comment =
                            "Frames blended for smooth clouds. Higher values create smoother clouds but may cause ghosting during fast movement."
                    }
                }
                slider("SETTING_CLOUDS_LOW_CONFIDENCE_CURVE", 4.0, 1.0..8.0 step 0.5) {
                    lang {
                        name = "Confidence Curve"
                        comment =
                            "How quickly clouds sharpen over time. Higher values sharpen faster but may show more noise initially."
                    }
                }
                slider("SETTING_CLOUDS_LOW_VARIANCE_CLIPPING", 0.25, 0.0..1.0 step 0.05) {
                    lang {
                        name = "Variance Clipping"
                        comment =
                            "Prevents cloud trails during movement. Higher values reduce ghosting but may increase flickering."
                    }
                }
                empty()
                slider("SETTING_CLOUDS_LOW_STEP_MIN", 24, 16..128 step 8) {
                    lang {
                        name = "Ray Marching Min Step"
                        comment =
                            "Minimum samples through clouds. This value is typically used in clouds directly on top. Higher values improve detail but reduce performance."
                    }
                }
                slider("SETTING_CLOUDS_LOW_STEP_MAX", 72, 32..256 step 8) {
                    lang {
                        name = "Ray Marching Max Step"
                        comment =
                            "Maximum samples through thick clouds.  This value is typically used in clouds near horizon. Higher values improve quality of dense clouds but reduce performance."
                    }
                }
                empty()
                slider("SETTING_CLOUDS_CU_WEIGHT", 0.75, 0.0..1.0 step 0.05) {
                    lang {
                        name = "Cumulus Weight"
                    }
                }
                slider("SETTING_CLOUDS_CU_HEIGHT", 2.0, 0.0..8.0 step 0.1) {
                    lang {
                        name = "Cloud Altitude"
                        suffix = " km"
                        comment = "Altitude where cumulus clouds begin to form."
                    }
                }
                slider("SETTING_CLOUDS_CU_THICKNESS", 2.0, 0.0..4.0 step 0.1) {
                    lang {
                        name = "Cloud Layer Thickness"
                        suffix = " km"
                        comment =
                            "Vertical thickness of the cloud layer. Thicker clouds are more dramatic and puffy."
                    }
                }
                slider("SETTING_CLOUDS_CU_DENSITY", 1.0, 0.0..4.0 step 0.05) {
                    lang {
                        name = "Cloud Density"
                        suffix = " x"
                        comment =
                            "How thick and opaque clouds appear. Higher values create denser, more solid-looking clouds."
                    }
                }
                slider("SETTING_CLOUDS_CU_COVERAGE", 0.3, 0.0..1.0 step 0.05) {
                    lang {
                        name = "Sky Coverage"
                        comment =
                            "How much of the sky is covered by clouds. 0 = clear sky, 1 = completely overcast."
                    }
                }
                slider("SETTING_CLOUDS_CU_PHASE_RATIO", 0.9, 0.0..1.0 step 0.05) {
                    lang {
                        name = "Cumulus Phase Ratio"
                    }
                }
                empty()
                toggle("SETTING_CLOUDS_CU_WIND", true) {
                    lang {
                        name = "Enable Cloud Movement"
                        comment = "Allows clouds to drift across the sky over time."
                    }
                }
                slider("SETTING_CLOUDS_CU_WIND_SPEED", 0.0, -4.0..4.0 step 0.25) {
                    lang {
                        name = "Wind Speed"
                        comment =
                            "Speed of cloud movement. Negative values move clouds in the opposite direction."
                    }
                }
            }
            screen("HIGH_CLOUDS", 1) {
                lang {
                    name = "High Altitude Clouds"
                }
                toggle("SETTING_CLOUDS_CI", true) {
                    lang {
                        name = "Enable Cirrus Clouds"
                        comment =
                            "Toggles wispy, high-altitude ice crystal clouds that add atmosphere to the sky."
                    }
                }
                slider("SETTING_CLOUDS_CI_HEIGHT", 9.0, 6.0..14.0 step 0.1) {
                    lang {
                        name = "Cloud Altitude"
                        suffix = " km"
                        comment =
                            "Altitude of cirrus clouds. Higher altitudes create thinner, more delicate wisps."
                    }
                }
                slider("SETTING_CLOUDS_CI_DENSITY", 1.0, 0.0..4.0 step 0.05) {
                    lang {
                        name = "Cloud Density"
                        suffix = " x"
                        comment =
                            "How visible and opaque the cirrus clouds are. Higher values create more prominent wisps."
                    }
                }
                slider("SETTING_CLOUDS_CI_COVERAGE", 0.4, 0.0..1.0 step 0.05) {
                    lang {
                        name = "Sky Coverage"
                        comment =
                            "How much of the high sky is covered by cirrus clouds. 0 = clear, 1 = fully covered."
                    }
                }
                slider("SETTING_CLOUDS_CI_PHASE_RATIO", 0.6, 0.0..1.0 step 0.05) {
                    lang {
                        name = "Cirrus Phase Ratio"
                    }
                }
            }
            empty()
            empty()
            screen("WATER_SURFACE", 1) {
                lang {
                    name = "Water Surface"
                }
                toggle("SETTING_WATER_REFRACT_APPROX", true) {
                    lang {
                        name = "Approximate Refraction"
                        comment =
                            "Approximated refraction direction that works better with screen space refraction."
                    }
                }
                toggle("SETTING_WATER_CAUSTICS", false) {
                    lang {
                        name = "Enable Water Caustics"
                        comment =
                            "Shows light patterns on surfaces beneath water, like you see at the bottom of pools."
                    }
                }
                empty()
                slider("SETTING_WATER_NORMAL_SCALE", 1.0, 0.0..4.0 step 0.5) {
                    lang {
                        name = "Water Normal Scale"
                        comment =
                            "Intensity of water surface waves and ripples. Higher values create choppier water."
                    }
                }
                empty()
                toggle("SETTING_WATER_PARALLAX", true) {
                    lang {
                        name = "Enable Water Parallax"
                        comment =
                            "Creates realistic depth in water waves, making them appear 3D instead of flat."
                    }
                }
                slider("SETTING_WATER_PARALLAX_STRENGTH", 1.0, 0.0..4.0 step 0.5) {
                    lang {
                        name = "Water Parallax Strength"
                        comment =
                            "How deep and three-dimensional water waves appear. Higher values create more pronounced depth."
                    }
                }
                slider("SETTING_WATER_PARALLAX_LINEAR_STEPS", 8, powerOfTwoRangeAndHalf(2..5)) {
                    lang {
                        name = "Water Parallax Linear Sample Steps"
                        comment =
                            "Samples for wave depth effect. Higher values reduce visual breaks between waves but reduce performance."
                    }
                }
                slider("SETTING_WATER_PARALLAX_SECANT_STEPS", 2, 1..8) {
                    lang {
                        name = "Water Parallax Secant Sample Steps"
                        comment =
                            "Additional refinement passes for wave depth. Higher values create smoother waves but reduce performance."
                    }
                }
            }
            screen("WATER_VOLUME", 1) {
                lang {
                    name = "Water Volume"
                }
                slider("SETTING_WATER_SCATTERING_REFRACTION_APPROX", true) {
                    lang {
                        name = "Approximate Refraction Light Shafts"
                        comment = "Approximate under water light shafts causes by water waves"
                    }
                }
                slider("SETTING_WATER_SCATTERING_REFRACTION_APPROX_CONTRAST", 5, 0..12) {
                    lang {
                        name = "Refraction Light Shaft Contrast"
                        comment = "Sharpness of underwater light rays created by surface waves. "
                    }
                }
                empty()
                slider("SETTING_WATER_SCATTERING_R", 14, 0..100) {
                    lang {
                        name = "Scattering Coefficient - Red"
                        suffix = " %"
                        comment =
                            "How much red light bounces in water. Lower values create more blue-tinted water."
                    }
                }
                slider("SETTING_WATER_SCATTERING_G", 22, 0..100) {
                    lang {
                        name = "Scattering Coefficient - Green"
                        suffix = " %"
                        comment = "How much green light bounces in water. Affects overall water color tone."
                    }
                }
                slider("SETTING_WATER_SCATTERING_B", 38, 0..100) {
                    lang {
                        name = "Scattering Coefficient - Blue"
                        suffix = " %"
                        comment = "How much blue light bounces in water. Higher values create bluer water."
                    }
                }
                slider("SETTING_WATER_SCATTERING_MULTIPLIER", -8.75, -15.0..-5.0 step 0.25) {
                    lang {
                        name = "Scattering Coefficient Multiplier"
                        prefix = "2^"
                        comment =
                            "Global multiplier for how much light bounces in water. Higher values brighten underwater scenes. (Multiplier: 2^x)"
                    }
                }
                empty()
                slider("SETTING_WATER_ABSORPTION_R", 100, 0..100) {
                    lang {
                        name = "Absorption Coefficient - Red"
                        suffix = " %"
                        comment =
                            "How quickly red light fades underwater. Higher values remove red faster, creating bluer water."
                    }
                }
                slider("SETTING_WATER_ABSORPTION_G", 40, 0..100) {
                    lang {
                        name = "Absorption Coefficient - Green"
                        suffix = " %"
                        comment =
                            "How quickly green light fades underwater. Affects visibility distance and water color."
                    }
                }
                slider("SETTING_WATER_ABSORPTION_B", 24, 0..100) {
                    lang {
                        name = "Absorption Coefficient - Blue"
                        suffix = " %"
                        comment =
                            "How quickly blue light fades underwater. Lower values maintain blue color in deeper water."
                    }
                }
                slider("SETTING_WATER_ABSORPTION_MULTIPLIER", -9.25, -15.0..-5.0 step 0.25) {
                    lang {
                        name = "Absorption Coefficient Multiplier"
                        prefix = "2^"
                        comment =
                            "Global multiplier for water absorption. Higher values create murkier water with less visibility. (Multiplier: 2^x)"
                    }
                }
                empty()
                slider("SETTING_WATER_SHADOW_SAMPLE", 64, powerOfTwoRangeAndHalf(4..8)) {
                    lang {
                        name = "Shadow Samples"
                        comment =
                            "Samples for shadows visible underwater. Higher values improve shadow smoothness but reduce performance."
                    }
                }
                slider("SETTING_WATER_SHADOW_SAMPLE_POOL_SIZE", 8, 2..16 step 2) {
                    lang {
                        name = "Shadow Sample Pool Size"
                        comment = "Higher values increase shadowing quality but also decrease performance."
                    }
                }
            }
        }
        screen("OUTER_SPACE", 2) {
            lang {
                name = "Outer Space"
            }
            screen("SUN_MOON", 1) {
                lang {
                    name = "Sun & Moon"
                }
                slider("SETTING_SUN_RADIUS", 1.0, (-7..10).map { 2.0.pow(it) }) {
                    lang {
                        name = "Sun Size"
                        comment =
                            "Size of the sun in the sky. 1.0 = realistic size. Larger suns create softer, wider shadows."
                        suffix = " R"
                    }
                }
                slider("SETTING_SUN_DISTANCE", 1.0, (-7..10).map { 2.0.pow(it) }) {
                    lang {
                        name = "Sun Distance"
                        comment =
                            "Distance of sun in AU (astronomical units), which is relative to real sun distance of 149.6 million km."
                        suffix = " AU"
                    }
                }
                constSlider("sunPathRotation", -20.0, -90.0..90.0 step 1.0) {
                    lang {
                        name = "Sun Angle in Sky"
                        comment =
                            "Adjusts the sun's path across the sky. Changes the angle of sunlight and shadow direction."
                        suffix = " °"
                    }
                }
                toggle("SETTING_REAL_SUN_TEMPERATURE", true) {
                    lang {
                        name = "Realistic Sun Color"
                        comment =
                            "Uses the real sun's color temperature (5772 K) for accurate warm yellow-white sunlight."
                    }
                }
                slider("SETTING_SUN_TEMPERATURE", 5700, (1000..10000 step 100) + (11000..50000 step 1000)) {
                    lang {
                        name = "Sun Color Temperature"
                        comment =
                            "Color of sunlight in Kelvin. Lower = warmer/redder (sunset), Higher = cooler/bluer (noon). Default: 5700 K."
                        suffix = " K"
                    }
                }
                empty()
                slider("SETTING_MOON_RADIUS", 1.0, (-7..10).map { 2.0.pow(it) }) {
                    lang {
                        name = "Moon Size"
                        comment =
                            "Size of the moon in the sky. 1.0 = realistic size (about 1/4 of Earth's diameter)."
                        suffix = " R"
                    }
                }
                slider("SETTING_MOON_DISTANCE", 1.0, (-7..10).map { 2.0.pow(it) }) {
                    lang {
                        name = "Moon Distance"
                        comment =
                            "Distance to the moon relative to reality (384,399 km). Affects moon's apparent size in the sky."
                        suffix = " D"
                    }
                }
                slider("SETTING_MOON_ALBEDO", 0.12, 0.01..1.0 step 0.01) {
                    lang {
                        name = "Moon Brightness"
                        comment =
                            "How reflective the moon surface is. 0.12 = realistic (moon reflects 12% of sunlight). Higher = brighter nights."
                    }
                }
                slider("SETTING_MOON_COLOR_R", 0.8, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Moon Tint - Red"
                        comment =
                            "Red component of moon color. Adjust to create warmer or cooler moonlight."
                    }
                }
                slider("SETTING_MOON_COLOR_G", 0.9, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Moon Tint - Green"
                        comment = "Green component of moon color. Affects the overall tone of moonlight."
                    }
                }
                slider("SETTING_MOON_COLOR_B", 1.0, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Moon Tint - Blue"
                        comment =
                            "Blue component of moon color. Higher values create cooler, bluer moonlight."
                    }
                }
            }
            screen("STARS", 1) {
                lang {
                    name = "Stars"
                }
                slider("SETTING_STARMAP_INTENSITY", 6, 0..16) {
                    lang {
                        name = "Star Brightness"
                        comment =
                            "Overall brightness of stars in the night sky. Higher values make stars more visible."
                    }
                }
                slider("SETTING_STARMAP_BRIGHT_STAR_BOOST", 4, 0..8) {
                    lang {
                        name = "Bright Star Enhancement"
                        comment =
                            "Extra brightness for the most prominent stars. Creates more realistic star size variation."
                    }
                }
                slider("SETTING_STARMAP_GAMMA", 0.8, 0.1..2.0 step 0.1) {
                    lang {
                        name = "Star Contrast"
                        comment =
                            "Adjusts contrast between bright and dim stars. Lower values make faint stars more visible."
                    }
                }
            }
        }
        screen("POSTFX", 2) {
            lang {
                name = "Post Processing"
            }
            screen("DOF", 1) {
                lang {
                    name = "Depth of Field"
                }
                toggle("SETTING_DOF", false) {
                    lang {
                        name = "Enable Depth of Field"
                        comment =
                            "Blurs distant or nearby objects like a camera lens, focusing attention on what you're looking at."
                    }
                }
                empty()
                slider("SETTING_DOF_FOCAL_LENGTH", 50.0, listOf(18.0, 24.0, 35.0, 50.0, 75.0, 100.0)) {
                    lang {
                        name = "Focal Length"
                        suffix = " mm"
                    }
                }
                slider("SETTING_DOF_F_STOP", 1.4, listOf(1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0)) {
                    lang {
                        name = "F-Stop"
                        prefix = "f/"
                    }
                }
                toggle("SETTING_APERTURE_SHAPE", 1, 0..1) {
                    lang {
                        name = "Bokeh Shape"
                        comment = "Shape of blurred light points (bokeh)."
                        0 value "Circle"
                        1 value "Hexagon"
                    }
                }
                empty()
                slider("SETTING_DOF_QUALITY", 3, 1..5) {
                    lang {
                        name = "Blur Quality"
                        comment =
                            "Quality of depth of field blur. Higher values create smoother blur but reduce performance."
                    }
                }
                slider("SETTING_DOF_MAX_SAMPLE_RADIUS", 8, listOf(2, 4, 8, 12, 16, 20, 24)) {
                    lang {
                        name = "Maximum Blur Radius"
                        comment =
                            "Maximum blur distance in pixels. Should match your aperture setting - too low cuts off blur, too high causes artifacts."
                    }
                }
                slider("SETTING_DOF_MASKING_HEURISTIC", 8, 0..32) {
                    lang {
                        name = "Masking Heuristic"
                        comment =
                            "How strictly to separate foreground from background blur. Higher values prevent blur bleeding between objects."
                    }
                }
                empty()
                toggle("SETTING_DOF_MANUAL_FOCUS", false) {
                    lang {
                        name = "Manual Focus"
                        comment =
                            "Set focus distance manually instead of automatically focusing on what you're looking at."
                    }
                }
                slider("SETTING_DOF_FOCUS_DISTANCE_COARSE", 10, 1..100) {
                    lang {
                        name = "Focus Distance (Coarse)"
                        suffix = " m"
                        comment =
                            "Rough focus distance adjustment in meters. Only works with Manual Focus enabled."
                    }
                }
                slider("SETTING_DOF_FOCUS_DISTANCE_FINE", 0.0, -1.0..1.0 step 0.01) {
                    lang {
                        name = "Focus Distance (Fine-Tune)"
                        suffix = " m"
                        comment = "Precise focus distance adjustment. Adds/subtracts from coarse setting."
                    }
                }
                slider("SETTING_DOF_FOCUS_TIME", 2.0, 0.0..10.0 step 0.25) {
                    lang {
                        name = "Focus Speed"
                        comment =
                            "How quickly focus adjusts when looking at different distances. Lower = faster, higher = more cinematic."
                    }
                }
                toggle("SETTING_DOF_SHOW_FOCUS_PLANE", false) {
                    lang {
                        name = "Show Focus Plane"
                        comment =
                            "Displays the exact distance that's in focus, helpful for adjusting manual focus."
                    }
                }
            }
            screen("BLOOM", 1) {
                lang {
                    name = "Bloom"
                }
                toggle("SETTING_BLOOM", true) {
                    lang {
                        name = "Enable Bloom"
                        comment =
                            "Makes bright areas glow and bleed into surrounding pixels, like light overexposing a camera."
                    }
                }
                slider("SETTING_BLOOM_INTENSITY", 1.0, 0.5..10.0 step 0.5) {
                    lang {
                        name = "Glow Strength"
                        comment =
                            "How bright the bloom glow effect is. Higher values create more intense, dramatic glowing."
                    }
                }
                slider("SETTING_BLOOM_RADIUS", 1.0, 1.0..5.0 step 0.5) {
                    lang {
                        name = "Glow Spread"
                        comment =
                            "How far the bloom glow spreads. Higher values create wider halos but may make the whole screen hazy."
                    }
                }
                slider("SETTING_BLOOM_PASS", 8, 1..10) {
                    lang {
                        name = "Blur Passes"
                        comment =
                            "Processing passes for bloom. Higher values increase glow reach and smoothness but reduce performance."
                    }
                }
                empty()
                slider("SETTING_BLOOM_UNDERWATER_BOOST", 10, 1..20 step 1) {
                    lang {
                        name = "Underwater Glow Boost"
                        comment =
                            "Extra bloom intensity when underwater, creating a dreamy, diffused underwater atmosphere."
                    }
                }
            }
            screen("PURKINJE_EFFECT", 1) {
                lang {
                    name = "Purkinje Effect (Night Vision)"
                }
                toggle("SETTING_PURKINJE_EFFECT", true) {
                    lang {
                        name = "Enable Purkinje Effect"
                        comment =
                            "Simulates how human eyes lose color vision in darkness, creating a more realistic night experience."
                    }
                }
                slider("SETTING_PURKINJE_EFFECT_MIN_LUM", -8.0, -10.0..1.0 step 0.1) {
                    lang {
                        name = "Minimum Luminance"
                        prefix = "10^"
                        suffix = " cd/m²"
                        comment =
                            "Below this brightness, colors fade to monochrome. Lower = colors disappear in dimmer light."
                    }
                }
                slider("SETTING_PURKINJE_EFFECT_MAX_LUM", 0.0, -10.0..1.0 step 0.1) {
                    lang {
                        name = "Maximum Luminance"
                        prefix = "10^"
                        suffix = " cd/m²"
                        comment =
                            "Above this brightness, colors appear fully. Higher = need brighter light to see full colors."
                    }
                }
                empty()
                slider("SETTING_PURKINJE_EFFECT_CR", 0.9, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Night Vision Tint - Red"
                        comment =
                            "Red tint of monochrome night vision. Default creates bluish night vision like real eyes."
                    }
                }
                slider("SETTING_PURKINJE_EFFECT_CG", 0.95, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Night Vision Tint - Green"
                        comment = "Green tint of monochrome night vision."
                    }
                }
                slider("SETTING_PURKINJE_EFFECT_CB", 1.0, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Night Vision Tint - Blue"
                        comment =
                            "Blue tint of monochrome night vision. Higher values create bluer night scenes."
                    }
                }
            }
            screen("EXPOSURE", 1) {
                lang {
                    name = "Exposure"
                }
                toggle("SETTING_EXPOSURE_MANUAL", false) {
                    lang {
                        name = "Manual Exposure"
                        comment =
                            "Lock exposure to a fixed value instead of automatically adjusting to scene brightness."
                    }
                }
                slider("SETTING_EXPOSURE_MANUAL_EV_COARSE", 3, -32..32) {
                    lang {
                        name = "Exposure EV (Coarse)"
                        comment =
                            "Rough brightness adjustment in EV stops. Negative = darker, positive = brighter."
                    }
                }
                slider("SETTING_EXPOSURE_MANUAL_EV_FINE", 0.0, -1.0..1.0 step 0.01) {
                    lang {
                        name = "Exposure EV (Fine-Tune)"
                        comment = "Precise brightness adjustment. Adds/subtracts from coarse setting."
                    }
                }
                empty()
                slider("SETTING_EXPOSURE_MIN_EV", -3.0, -32.0..32.0 step 0.5) {
                    lang {
                        name = "Auto Exposure Min EV"
                    }
                }
                slider("SETTING_EXPOSURE_MAX_EV", 12.0, -32.0..32.0 step 0.5) {
                    lang {
                        name = "Auto Exposure Max EV"
                    }
                }
                slider("SETTING_EXPOSURE_EMISSIVE_WEIGHTING", 0.1, 0.0..4.0 step 0.1) {
                    lang {
                        name = "Emissive Weighting"
                        comment = "Extra weighting for emissive block pixels."
                    }
                }
                slider("SETTING_EXPOSURE_CENTER_WEIGHTING", 4.0, 0.0..8.0 step 0.1) {
                    lang {
                        name = "Center Focus Priority"
                        comment =
                            "How much the center of the screen influences exposure. Higher = adjusts more to what you're looking at directly."
                    }
                }
                slider("SETTING_EXPOSURE_CENTER_WEIGHTING_CURVE", 3.0, 1.0..8.0 step 0.1) {
                    lang {
                        name = "Center Focus Sharpness"
                        comment =
                            "How sharply center weighting focuses on the middle. Higher = tighter focus on screen center, ignoring edges more."
                    }
                }
                empty()
                slider("SETTING_EXPOSURE_AVG_LUM_MIX", 0.25, 0.0..1.0 step 0.05) {
                    lang {
                        name = "Overall Brightness Method Weight"
                        comment =
                            "Influence of overall scene brightness on exposure. Higher = adjusts more to keep average brightness consistent."
                    }
                }
                slider("SETTING_EXPOSURE_AVG_LUM_TIME", 4.0, 0.0..10.0 step 0.25) {
                    lang {
                        name = "Overall Brightness Adapt Speed"
                        comment =
                            "How quickly exposure adapts based on overall brightness. Lower = faster adjustment."
                    }
                }
                slider("SETTING_EXPOSURE_AVG_LUM_MIN_TARGET", 30, 1..255) {
                    lang {
                        name = "Dark Scene Target Brightness"
                        comment =
                            "Target brightness for dark environments (caves, night). Higher values make dark scenes brighter."
                    }
                }
                slider("SETTING_EXPOSURE_AVG_LUM_MAX_TARGET", 60, 1..255) {
                    lang {
                        name = "Bright Scene Target Brightness"
                        comment =
                            "Target brightness for bright environments (daylight outdoors). Higher values make bright scenes brighter."
                    }
                }
                slider(
                    "SETTING_EXPOSURE_AVG_LUM_TARGET_CURVE",
                    0.5,
                    (0.01..1.0 step 0.01) + (1.1..4.0 step 0.1)
                ) {
                    lang {
                        name = "Medium Brightness Response"
                        comment =
                            "Affects medium-brightness scenes (sunset/sunrise). Lower values darken these transitional lighting conditions."
                    }
                }
                empty()
                slider("SETTING_EXPOSURE_HS_MIX", 1.0, 0.0..1.0 step 0.05) {
                    lang {
                        name = "Highlight/Shadow Areas Method Weight"
                        comment =
                            "Influence of brightest and darkest areas on exposure. Higher = prevents over/underexposure of extremes."
                    }
                }
                slider("SETTING_EXPOSURE_HS_TIME", 2.0, 0.0..10.0 step 0.25) {
                    lang {
                        name = "Highlight/Shadow Areas Adapt Speed"
                        comment =
                            "How quickly exposure adapts to bright and dark regions. Lower = faster adjustment."
                    }
                }
                slider("SETTING_EXPOSURE_H_LUM", 197, 1..255) {
                    lang {
                        name = "Highlight Area Threshold"
                        comment =
                            "Brightness level considered 'highlight'. Exposure adjusts to prevent these areas from being too bright."
                    }
                }
                slider("SETTING_EXPOSURE_H_PERCENT", 5.0, 0.5..10.0 step 0.5) {
                    lang {
                        name = "Highlight Area Protection"
                        comment =
                            "Keeps this percentage of bright pixels from overexposing. Higher values darken overall to preserve bright details."
                    }
                }
                slider("SETTING_EXPOSURE_S_LUM", 16, 0..255) {
                    lang {
                        name = "Shadow Area Threshold"
                        comment =
                            "Brightness level considered 'shadow'. Exposure adjusts to keep these areas visible."
                    }
                }
                slider("SETTING_EXPOSURE_S_PERCENT", 3.0, 0.5..10.0 step 0.5) {
                    lang {
                        name = "Shadow Area Visibility"
                        comment =
                            "Keeps this percentage of dark pixels from becoming pure black. Higher values brighten overall to reveal shadow detail."
                    }
                }
            }
            screen("TONE_MAPPING", 1) {
                lang {
                    name = "Tone Mapping & Color Grading"
                }
                slider("SETTING_TONE_MAPPING_DYNAMIC_RANGE", 13.5, 4.0..32.0 step 0.5) {
                    lang {
                        name = "Dynamic Range"
                        comment =
                            "Range of brightness levels preserved from dark to bright. Higher values maintain more detail in extremes but may look flat."
                    }
                }
                empty()
                toggle("SETTING_TONE_MAPPING_LOOK", 3, 0..3) {
                    lang {
                        name = "AgX Preset"
                        comment =
                            "Pre-configured color grading styles. Choose Custom to manually adjust colors below."
                        0 value "Default"
                        1 value "Golden"
                        2 value "Punchy"
                        3 value "Custom"
                    }
                }
                empty()
                slider("SETTING_TONE_MAPPING_OFFSET_R", 0.0, -1.0..1.0 step 0.01) {
                    lang {
                        name = "Red Lift"
                        comment =
                            "Adds or removes red from all brightness levels. Negative = less red, positive = more red."
                    }
                }
                slider("SETTING_TONE_MAPPING_OFFSET_G", 0.0, -1.0..1.0 step 0.01) {
                    lang {
                        name = "Green Lift"
                        comment =
                            "Adds or removes green from all brightness levels. Negative = less green, positive = more green."
                    }
                }
                slider("SETTING_TONE_MAPPING_OFFSET_B", 0.0, -1.0..1.0 step 0.01) {
                    lang {
                        name = "Blue Lift"
                        comment =
                            "Adds or removes blue from all brightness levels. Negative = less blue, positive = more blue."
                    }
                }
                empty()
                slider("SETTING_TONE_MAPPING_SLOPE_R", 1.0, 0.1..2.0 step 0.01) {
                    lang {
                        name = "Red Gain"
                        comment =
                            "Multiplies red channel intensity. Below 1.0 reduces red, above 1.0 increases red in mid-tones."
                    }
                }
                slider("SETTING_TONE_MAPPING_SLOPE_G", 1.0, 0.1..2.0 step 0.01) {
                    lang {
                        name = "Green Gain"
                        comment =
                            "Multiplies green channel intensity. Below 1.0 reduces green, above 1.0 increases green in mid-tones."
                    }
                }
                slider("SETTING_TONE_MAPPING_SLOPE_B", 1.0, 0.1..2.0 step 0.01) {
                    lang {
                        name = "Blue Gain"
                        comment =
                            "Multiplies blue channel intensity. Below 1.0 reduces blue, above 1.0 increases blue in mid-tones."
                    }
                }
                empty()
                slider("SETTING_TONE_MAPPING_POWER_R", 1.0, 0.1..2.0 step 0.01) {
                    lang {
                        name = "Red Contrast"
                        comment =
                            "Adjusts contrast in red channel. Higher values increase red contrast, making reds more dramatic."
                    }
                }
                slider("SETTING_TONE_MAPPING_POWER_G", 1.0, 0.1..2.0 step 0.01) {
                    lang {
                        name = "Green Contrast"
                        comment =
                            "Adjusts contrast in green channel. Higher values increase green contrast."
                    }
                }
                slider("SETTING_TONE_MAPPING_POWER_B", 1.0, 0.1..2.0 step 0.01) {
                    lang {
                        name = "Blue Contrast"
                        comment =
                            "Adjusts contrast in blue channel. Higher values increase blue contrast, making blues more dramatic."
                    }
                }
                empty()
                slider("SETTING_TONE_MAPPING_SATURATION", 1.1, 0.0..2.0 step 0.01) {
                    lang {
                        name = "Color Saturation"
                        comment =
                            "Overall color intensity. 0 = black & white, 1 = normal, 2 = hyper-saturated."
                    }
                }
            }
            screen("AA", 1) {
                lang {
                    name = "Anti-Aliasing"
                }
                toggle("SETTING_TAA", true) {
                    lang {
                        name = "Enable Temporal Anti-Aliasing"
                        comment =
                            "Smooths jagged edges by blending multiple frames. Highly recommended for clean image quality."
                    }
                }
                toggle("SETTING_TAA_JITTER", true) {
                    lang {
                        name = "Enable Sub-Pixel Jittering"
                        comment =
                            "Slightly shifts the camera each frame for better TAA quality. Required for TAA to work effectively."
                    }
                }
                slider("SETTING_TAA_CAS_SHARPNESS", 1.5, 0.0..5.0 step 0.25) {
                    lang {
                        name = "Sharpening Strength"
                        comment =
                            "Restores sharpness lost from anti-aliasing using AMD FidelityFX CAS. Higher values create crisper images."
                    }
                }
            }
        }
        screen("COLOR_MANAGEMENT", 1) {
            lang {
                name = "Color Management"
                comment = "Advanced color space settings. Only change if you know what you're doing!"
            }
            toggle("SETTING_MATERIAL_COLOR_SPACE", 1, 0..7) {
                lang {
                    name = "Texture Color Space"
                    comment = "Color space of your resource pack textures. sRGB is standard for most packs."
                    0 value "CIE XYZ"
                    1 value "sRGB"
                    2 value "Rec. 709"
                    3 value "Rec. 2020"
                    4 value "DCI-P3"
                    5 value "Adobe RGB (1998)"
                    6 value "ACES2065-1"
                    7 value "ACEScg"
                }
            }
            toggle("SETTING_MATERIAL_TRANSFER_FUNC", 3, 0..7) {
                lang {
                    name = "Texture Gamma Curve"
                    comment =
                        "Gamma/transfer function of textures. sRGB is standard for most resource packs."
                    0 value "Linear"
                    1 value "Rec. 601"
                    2 value "Rec. 709"
                    3 value "sRGB"
                    4 value "Exponential 2.2"
                    5 value "Exponential 2.4"
                    6 value "ST 2084 (PQ)"
                    7 value "HLG"
                }
            }
            empty()
            toggle("SETTING_WORKING_COLOR_SPACE", 7, 0..7) {
                lang {
                    name = "Internal Processing Color Space"
                    comment =
                        "Color space used for lighting calculations. ACEScg is recommended for wide-gamut rendering."
                    0 value "CIE XYZ"
                    1 value "sRGB"
                    2 value "Rec. 709"
                    3 value "Rec. 2020"
                    4 value "DCI-P3"
                    5 value "Adobe RGB (1998)"
                    6 value "ACES2065-1"
                    7 value "ACEScg"
                }
            }
            empty()
            toggle("SETTING_DRT_WORKING_COLOR_SPACE", 3, 0..7) {
                lang {
                    name = "Tone Mapping Color Space"
                    comment =
                        "Color space for tone mapping operations. Rec. 2020 works better with AgX tone mapping."
                    0 value "CIE XYZ"
                    1 value "sRGB"
                    2 value "Rec. 709"
                    3 value "Rec. 2020"
                    4 value "DCI-P3"
                    5 value "Adobe RGB (1998)"
                    6 value "ACES2065-1"
                    7 value "ACEScg"
                }
            }
            empty()
            toggle("SETTING_OUTPUT_COLOR_SPACE", 1, 0..7) {
                lang {
                    name = "Monitor Color Space"
                    comment =
                        "Color space of your monitor. sRGB for standard monitors, Rec. 2020 or DCI-P3 for wide-gamut displays."
                    0 value "CIE XYZ"
                    1 value "sRGB"
                    2 value "Rec. 709"
                    3 value "Rec. 2020"
                    4 value "DCI-P3"
                    5 value "Adobe RGB (1998)"
                    6 value "ACES2065-1"
                    7 value "ACEScg"
                }
            }
            toggle("SETTING_OUTPUT_TRANSFER_FUNC", 3, 0..7) {
                lang {
                    name = "Monitor Gamma Curve"
                    comment =
                        "Gamma/transfer function of your monitor. sRGB for most monitors, ST 2084 (PQ) for HDR displays."
                    0 value "Linear"
                    1 value "Rec. 601"
                    2 value "Rec. 709"
                    3 value "sRGB"
                    4 value "Exponential 2.2"
                    5 value "Exponential 2.4"
                    6 value "ST 2084 (PQ)"
                    7 value "HLG"
                }
            }
        }
        screen("MISC", 2) {
            lang {
                name = "Miscellaneous"
            }
            toggle("SETTING_SCREENSHOT_MODE", false) {
                lang {
                    name = "Screenshot Mode"
                    comment =
                        "Disables animations and temporal clamping for cleaner, higher-quality screenshots."
                }
            }
            slider("SETTING_SCREENSHOT_MODE_SKIP_INITIAL", 60, 10..200 step 10) {
                lang {
                    name = "Screenshot Mode Warmup Frames"
                    comment =
                        "Frames to wait before taking screenshot, allowing lighting and effects to stabilize for best quality."
                }
            }
            toggle("SETTING_CONSTELLATIONS", false) {
                lang {
                    name = "Show Star Constellations"
                    comment = "Displays constellation lines connecting stars in the night sky."
                }
            }
            slider("SETTING_TIME_SPEED_HISTORY_RESET_THRESHOLD", 32, powerOfTwoRangeAndHalf(2..10)) {
                lang {
                    name = "Time Change Sensitivity"
                    comment =
                        "How sensitive effects are to time changes (/time set). Higher values prevent flickering when rapidly changing time."
                }
            }
        }
        empty()
        empty()
        screen("SPONSORS", 4) {
            lang {
                name = "Sponsors"
            }
            toggle("SPONSOR_TITLE1", 0, 0..0) {
                lang {
                    name = "Special"
                    0 value ""
                }
            }
            toggle("SPONSOR_TITLE2", 0, 0..0) {
                lang {
                    name = "Thanks"
                    0 value ""
                }
            }
            toggle("SPONSOR_TITLE3", 0, 0..0) {
                lang {
                    name = "To"
                    0 value ""
                }
            }
            empty()
            empty()
            empty()
            empty()
            empty()
            Path("sponsors.txt").readLines().forEachIndexed { i, sname ->
                toggle("SPONSOR_$i", 0, 0..0) {
                    lang {
                        name = sname
                        0 value ""
                    }
                }
            }
        }
        screen("DEBUG", 3) {
            lang {
                name = "Debug"
            }
            toggle("SETTING_DEBUG_WHITE_WORLD", false) {
                lang {
                    name = "White World"
                }
            }
            toggle("SETTING_DEBUG_OUTPUT", 0, 0..3) {
                lang {
                    name = "Debug Output"
                    0 value "Off"
                    1 value "Tone Mapping"
                    2 value "TAA"
                    3 value "Final"
                }
            }
            slider("SETTING_DEBUG_SCALE", 1.0, 0.5..2.0 step 0.1) {
                lang {
                    name = "Debug Scale"
                }
            }
            toggle("SETTING_DEBUG_GAMMA_CORRECT", true) {
                lang {
                    name = "Gamma Correct"
                }
            }
            slider("SETTING_DEBUG_EXP", 0.0, -10.0..10.0 step 0.1) {
                lang {
                    name = "Exposure"
                }
            }
            toggle("SETTING_DEBUG_NEGATE", false) {
                lang {
                    name = "Negate"
                }
            }
            toggle("SETTING_DEBUG_ALPHA", false) {
                lang {
                    name = "Alpha"
                }
            }
            toggle("SETTING_DEBUG_DEDICATED", false) {
                lang {
                    name = "Dedicated Debug"
                }
            }
            empty()
            empty()
            empty()
            empty()
            toggle("SETTING_DEBUG_TEMP_TEX", 0, 0..6) {
                lang {
                    name = "Temp Tex"
                    0 value "Off"
                    1 value "temp1"
                    2 value "temp2"
                    3 value "temp3"
                    4 value "temp4"
                    5 value "temp5"
                    6 value "temp6"
                }
            }
            toggle("SETTING_DEBUG_GBUFFER_DATA", 0, 0..12) {
                lang {
                    name = "GBuffer Data"
                    0 value "Off"
                    1 value "View Z"
                    2 value "Albedo"
                    3 value "Normal"
                    4 value "Geometry Normal"
                    5 value "Roughness"
                    6 value "F0"
                    7 value "Porosity"
                    8 value "SSS"
                    9 value "Emissive"
                    10 value "Light Map Block"
                    11 value "Light Map Sky"
                    12 value "isHand"
                }
            }
            toggle("SETTING_DEBUG_NORMAL_MODE", 0, 0..1) {
                lang {
                    name = "Normal Mode"
                    0 value "World"
                    1 value "View"
                }
            }
            slider("SETTING_DEBUG_NORMAL_X_RANGE", 1.0, 0.0..1.0 step 0.1) {
                lang {
                    name = "Normal X Range"
                }
            }
            slider("SETTING_DEBUG_NORMAL_Y_RANGE", 1.0, 0.0..1.0 step 0.1) {
                lang {
                    name = "Normal Y Range"
                }
            }
            slider("SETTING_DEBUG_NORMAL_Z_RANGE", 1.0, 0.0..1.0 step 0.1) {
                lang {
                    name = "Normal Z Range"
                }
            }
            empty()
            empty()
            empty()
            toggle("SETTING_DEBUG_DENOISER", 0, 0..6) {
                lang {
                    name = "Denoiser"
                    0 value "Off"
                    1 value "Color"
                    2 value "Fast Color"
                    3 value "HLen"
                    4 value "Moment"
                    5 value "Moment²"
                    6 value "Variance"
                }
            }
            toggle("SETTING_DEBUG_GI_INPUTS", 0, 0..6) {
                lang {
                    name = "GI Inputs"
                    0 value "Off"
                    1 value "Radiance"
                    2 value "Light Map Sky"
                    3 value "Emissive"
                    4 value "Normal"
                    5 value "View Z"
                    6 value "Geometry Normal"
                }
            }
            toggle("SETTING_DEBUG_ENV_PROBE", false) {
                lang {
                    name = "Env Probe"
                }
            }
            toggle("SETTING_DEBUG_RTWSM", false) {
                lang {
                    name = "RTWSM"
                }
            }
            empty()
            empty()
            empty()
            empty()
            empty()
            toggle("SETTING_DEBUG_ATMOSPHERE", false) {
                lang {
                    name = "Atmosphere"
                }
            }
            toggle("SETTING_DEBUG_SKY_VIEW_LUT", false) {
                lang {
                    name = "Sky View LUT"
                }
            }
            toggle("SETTING_DEBUG_EPIPOLAR_LINES", false) {
                lang {
                    name = "Epipolar Lines"
                }
            }
            toggle("SETTING_DEBUG_CLOUDS_AMBLUT", false) {
                lang {
                    name = "Clouds Amb. LUT"
                }
            }
            toggle("SETTING_DEBUG_CLOUDS_SS", 0, 0..4) {
                lang {
                    name = "Clouds Upscaling"
                    0 value "Off"
                    1 value "Scattering"
                    2 value "Transmittance"
                    3 value "View Z"
                    4 value "HLen"
                }
            }
            empty()
            empty()
            empty()
            empty()
            toggle("SETTING_DEBUG_STARMAP", false) {
                lang {
                    name = "Star Map"
                }
            }
            toggle("SETTING_DEBUG_AE", false) {
                lang {
                    name = "Auto Exposure"
                }
            }
            toggle("SETTING_DEBUG_NOISE_GEN", false) {
                lang {
                    name = "Noise Generation"
                }
            }
        }
    }
}