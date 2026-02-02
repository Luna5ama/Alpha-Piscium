@file:Import("options.lib.kts")

import java.io.File
import java.util.*
import kotlin.io.path.Path
import kotlin.io.path.listDirectoryEntries
import kotlin.io.path.nameWithoutExtension
import kotlin.io.path.readLines
import kotlin.math.pow

options(File("shaders.properties"), File("../shaders"), "base/Options.glsl", "base/TextOptions.glsl") {
    mainScreen(2) {
        row {
            data class Version(val major: Int, val minor: Int, val patch: Int, val beta: Int) : Comparable<Version> {
                override fun compareTo(other: Version): Int {
                    var cmp = major.compareTo(other.major)
                    if (cmp != 0) return cmp
                    cmp = minor.compareTo(other.minor)
                    if (cmp != 0) return cmp
                    cmp = patch.compareTo(other.patch)
                    if (cmp != 0) return cmp
                    return beta.compareTo(other.beta)
                }

                override fun toString(): String {
                    return if (beta == Int.MAX_VALUE) {
                        "$major.$minor.$patch"
                    } else {
                        "$major.$minor.$patch-Beta$beta"
                    }
                }
            }
            val delimiter = """\d+\.\d+\.\d+((?:-beta\d+)?)""".toRegex(RegexOption.IGNORE_CASE)
            fun parseVersion(str: String): Version {
                val splitStr = str.split('.', '-')
                val major = splitStr[0].toInt()
                val minor = splitStr[1].toInt()
                val patch = splitStr[2].toInt()
                val beta = if (splitStr.size > 3) splitStr[3].lowercase().removePrefix("beta").toInt() else Int.MAX_VALUE
                return Version(major, minor, patch, beta)
            }
            val changelogPath = Path("../changelogs")
            val lastestVersion = changelogPath.listDirectoryEntries("*.md").asSequence()
                .map { it.nameWithoutExtension }
                .map { parseVersion(it) }
                .max()
                .toString()
            toggle("TITLE_VERSION", Profile.Ultra.ordinal, Profile.entries.indices) {
                Profile.entries.forEach {
                    it preset it.ordinal
                }
                lang {
                    name = "Alpha Piscium by Luna"
                    Profile.entries.forEach {
                        it.ordinal value "§${it.color.code}$lastestVersion"
                    }
                }
            }
            profile {
                lang {
                    comment = """
                        §c§lLow§r: Potato mode, §c§lnot recommented§r.
                        §e§lMedium§r: Baseline performance mode.
                        §a§lHigh§r: Balanced quality and performance.
                        §6§lUltra§r: The §6§ltrue§r Alpha Piscium experience as it should be.
                        §d§lExtreme§r: Even better experience at §osome§r cost.
                        §5§lInsane§r: "Can it run Crysis?" §kxyz69420§r.
                    """.trimIndent()
                }
                lang(Locale.SIMPLIFIED_CHINESE) {
                    comment = """
                        §c§lLow§r: 丐中丐画质, §c§l不推荐§r。
                        §e§lMedium§r: 基础性能模式。
                        §a§lHigh§r: 画质与性能的平衡。
                        §6§lUltra§r: §6§l真正的§r外屏七体验，就是它了。
                        §d§lExtreme§r: 更好的体验，只需要§o亿点点§r代价.
                        §5§lInsane§r: 《显 卡 危 机》 §kxyz69420§r.
                    """.trimIndent()
                }
            }
        }
        emptyRow()
        screen(2) {
            lang {
                name = "Terrain"
            }
            lang(Locale.SIMPLIFIED_CHINESE) {
                name = "地形"
            }
            row {
                screen(1) {
                    lang {
                        name = "Block Lighting"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "方块光照"
                    }
                    slider("SETTING_FIRE_TEMPERATURE", 1400, 100..5000 step 100) {
                        lang {
                            name = "Fire Temperature"
                            comment =
                                "Controls the color temperature of fire in Kelvin. Default: 1400 K (based on real fire). Higher values produce whiter/bluer light."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "火焰温度"
                            comment = "控制火焰的色温（开尔文）。默认值：1400 K（基于真实火焰）。数值越高，光线越白/越蓝。"
                        }
                    }
                    slider("SETTING_LAVA_TEMPERATURE", 1300, 100..5000 step 100) {
                        lang {
                            name = "Lava Temperature"
                            comment =
                                "Controls the color temperature of lava in Kelvin. Default: 1300 K (based on real lava). Higher values produce whiter/bluer light."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "岩浆温度"
                            comment = "控制岩浆的色温（开尔文）。默认值：1300 K（基于真实岩浆）。数值越高，光线越白/越蓝。"
                        }
                    }
                    slider("SETTING_EMISSIVE_STRENGTH", 4.0, 0.0..16.0 step 0.5) {
                        lang {
                            name = "Emissive Brightness"
                            comment = "Global brightness multiplier for all light-emitting materials and blocks."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "自发光亮度"
                            comment = "所有发光材质和方块的全局亮度倍数。"
                        }
                    }
                    slider("SETTING_PARTICLE_EMISSIVE_STRENGTH", 0.0, 0.0..1.0 step 0.1) {
                        lang {
                            name = "Particle Emissive Intensity"
                            comment = "Brightness multiplier for glowing particles like torches and fires."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "粒子自发光强度"
                            comment = "发光粒子（如火把和火焰）的亮度倍数。"
                        }
                    }
                    slider("SETTING_ENTITY_EMISSIVE_STRENGTH", 0.2, 0.0..1.0 step 0.1) {
                        lang {
                            name = "Entity Emissive Intensity"
                            comment = "Brightness multiplier for glowing entities."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "实体自发光强度"
                            comment = "实体发光的亮度倍数。"
                        }
                    }
                    empty()
                    slider("SETTING_EMISSIVE_PBR_VALUE_CURVE", 0.9, 0.1..4.0 step 0.05) {
                        lang {
                            name = "PBR Resource Pack Emissive Contrast"
                            comment =
                                "Adjusts contrast of emissive values from PBR resource packs. Higher values create stronger differences between bright and dim areas."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "PBR资源包自发光对比度"
                            comment = "调整PBR资源包自发光值的对比度。数值越高，明暗区域的差异越大。"
                        }
                    }
                    slider("SETTING_EMISSIVE_ALBEDO_COLOR_CURVE", 2.0, 0.1..4.0 step 0.05) {
                        lang {
                            name = "Emissive Color Saturation"
                            comment =
                                "Controls color intensity of emissive materials. Higher values produce more vibrant, saturated colors."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "自发光颜色饱和度"
                            comment = "控制自发光材质的颜色强度。数值越高，颜色越鲜艳饱和。"
                        }
                    }
                    slider("SETTING_EMISSIVE_ALBEDO_LUM_CURVE", 0.0, -4.0..4.0 step 0.1) {
                        lang {
                            name = "Color Texture-Based Emission Strength"
                            comment =
                                "Controls how much the base texture brightness affects emission. Higher values make brighter textures glow more intensely. This is recommand for packs that have \"binary\" emissive values."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "基于颜色纹理的发光强度"
                            comment = "控制基础纹理亮度对发光的影响程度。数值越高，较亮的纹理发光越强烈。"
                        }
                    }
                    empty()
                    slider("SETTING_EMISSIVE_ARMOR_GLINT_MULT", -10, -20..0 step 1) {
                        lang {
                            name = "Enchanted Armor Glow"
                            prefix = "2^"
                            comment = "Brightness of the enchanted armor glint effect. The actual multiplier is 2^x."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "附魔盔甲光效"
                            prefix = "2^"
                            comment = "附魔盔甲光效的亮度。实际倍数为 2^x。"
                        }
                    }
                    slider("SETTING_EMISSIVE_ARMOR_GLINT_CURVE", 1.3, 0.1..2.0 step 0.1) {
                        lang {
                            name = "Enchanted Armor Glow Contrast"
                            comment =
                                "Adjusts contrast of enchanted armor glint. Higher values make the brightest parts more prominent."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "附魔盔甲光效对比度"
                            comment = "调整附魔盔甲光效的对比度。数值越高，最亮的部分越突出。"
                        }
                    }
                }
                screen(1) {
                    lang {
                        name = "Normal Mapping"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "法线贴图"
                    }
                    toggle("SETTING_NORMAL_MAPPING", true) {
                        lang {
                            name = "Normal Mapping"
                            comment =
                                "Enables surface detail from normal maps, adding depth and texture to blocks without additional geometry."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "启用法线贴图"
                            comment = "启用法线贴图的表面细节，在不增加几何体的情况下为方块增加深度和纹理。"
                        }
                    }
                    slider("SETTING_NORMAL_MAPPING_STRENGTH", 0.0, -5.0..5.0 step 0.5) {
                        lang {
                            name = "Normal Mapping Strength"
                            prefix = "2^"
                            comment =
                                "Controls the intensity of surface detail effects. Higher values increase depth perception. The actual strength is 2^x."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "法线贴图强度"
                            prefix = "2^"
                            comment = "控制表面细节效果的强度。数值越高，深度感越强。实际强度为 2^x。"
                        }
                    }
                }
                screen(1) {
                    lang {
                        name = "Specular Mapping"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "高光贴图"
                    }
                    slider("SETTING_MINIMUM_F0", 12, 4..32) {
                        lang {
                            name = "Minimum Reflectivity (F0)"
                            prefix = "2^-"
                            comment =
                                "Sets the reflectivity (F0) lower bound for all materials. Higher values make surfaces more reflective overall. The actual value is calculated as 2^-x."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "最小反射率 (F0)"
                            prefix = "2^-"
                            comment = "设置所有材质的反射率（F0）下限。数值越高，表面整体反射性越强。实际值计算为 2^-x。"
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
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "固体最小粗糙度"
                            prefix = "2^-"
                            comment = "固体方块可呈现的最光滑（最镜面）程度。数值越高，允许更锐利的反射。实际值计算为 2^-x。"
                        }
                    }
                    slider("SETTING_SOLID_MAXIMUM_ROUGHNESS", 5, 2..16) {
                        lang {
                            name = "Maximum Solid Roughness"
                            prefix = "1-2^-"
                            comment =
                                "The roughest (most diffuse) that solid blocks can appear. Higher values allow more matte surfaces. The actual value is calculated as 1-2^-x."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "固体最大粗糙度"
                            prefix = "1-2^-"
                            comment =
                                "固体方块可呈现的最粗糙（最漫反射）程度。数值越高，允许更哑光的表面。实际值计算为 1-2^-x。"
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
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "水面粗糙度"
                            prefix = "2^-"
                            comment = "控制水面的光滑和反射程度。数值越小，水面越平静、越像镜面。实际值计算为 2^-x。"
                        }
                    }
                    slider("SETTING_TRANSLUCENT_ROUGHNESS_REDUCTION", 1.0, 0.0..8.0 step 0.5) {
                        lang {
                            name = "Translucent Roughness Reduction"
                            prefix = "2^-"
                            comment =
                                "Makes translucent blocks (such as glass) smoother than their resource pack values. Higher values create more mirror-like appearances. The actual value is calculated as 2^-x."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "半透明方块粗糙度降低"
                            prefix = "2^-"
                            comment = "使半透明方块（如玻璃）比资源包设定值更光滑。数值越高，越像镜面。实际值计算为 2^-x。"
                        }
                    }
                    slider("SETTING_TRANSLUCENT_MINIMUM_ROUGHNESS", 10.0, 4.0..16.0 step 0.5) {
                        lang {
                            name = "Translucent Minimum Roughness"
                            prefix = "2^-"
                            comment =
                                "The smoothest that translucent blocks (such as glass) can appear. Higher values allow sharper reflections on translucent. The actual value is calculated as 2^-x."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "半透明方块最小粗糙度"
                            prefix = "2^-"
                            comment =
                                "半透明方块（如玻璃）可呈现的最光滑程度。数值越高，允许更锐利的反射。实际值计算为 2^-x。"
                        }
                    }
                    slider("SETTING_TRANSLUCENT_MAXIMUM_ROUGHNESS", 5.0, 1.0..16.0 step 0.5) {
                        lang {
                            name = "Translucent Maximum Roughness"
                            prefix = "2^-"
                            comment =
                                "The roughest that translucent blocks (such as glass) can appear. Higher values allow more frosted glass effects. The actual value is calculated as 2^-x."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "半透明方块最大粗糙度"
                            prefix = "2^-"
                            comment =
                                "半透明方块（如玻璃）可呈现的最粗糙程度。数值越高，允许更磨砂的玻璃效果。实际值计算为 2^-x。"
                        }
                    }
                    empty()
                    slider("SETTING_MAXIMUM_SPECULAR_LUMINANCE", 65536, powerOfTwoRange(8..24)) {
                        lang {
                            name = "Maximum Specular Luminance"
                            comment =
                                "Limits how bright reflections and highlights can be (in 1000 cd/m²). Prevents overly intense glare from very smooth surfaces."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "最大高光亮度"
                            comment = "限制反射和高光的最大亮度（单位：1000 cd/m²）。防止非常光滑表面产生过强的眩光。"
                        }
                    }
                }
                screen(1) {
                    lang {
                        name = "Subsurface Scattering"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "次表面散射"
                    }
                    slider("SETTING_SSS_DIFFUSE_RANGE", 0.8, 0.0..4.0 step 0.1) {
                        lang {
                            name = "Diffuse Range"
                            comment =
                                "Higher values create a more diffused, softer appearance."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "扩散范围"
                            comment = "数值越高，外观越扩散、越柔和。"
                        }
                    }
                    slider("SETTING_SSS_DEPTH_RANGE", 0.3, 0.0..4.0 step 0.1) {
                        lang {
                            name = "Depth Range"
                            comment =
                                "How deep light penetrates into the material. Higher values simulate thicker, more translucent materials."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "深度范围"
                            comment = "光线渗透材质的深度。数值越高，模拟更厚、更半透明的材质。"
                        }
                    }
                }
                screen(1) {
                    lang {
                        name = "Other Material Settings"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "其他材质设置"
                    }
                    slider("SETTING_TRANSLUCENT_ABSORPTION_SATURATION", 1.0, 0.0..4.0 step 0.5) {
                        lang {
                            name = "Translucent Absorption Saturation"
                            comment =
                                "Controls translucent block color absorption saturation. Higher values create more vivid color tinting."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "半透明吸收饱和度"
                            comment = "控制半透明方块颜色吸收饱和度。数值越高，染色颜色越鲜艳。"
                        }
                    }
                }
            }
            row {
                empty()
            }
            screen(2) {
                lang {
                    name = "Shadows"
                }
                lang(Locale.SIMPLIFIED_CHINESE) {
                    name = "阴影"
                }
                slider("SETTING_SHADOW_MAP_RESOLUTION", 2048, listOf(1024, 2048, 3072, 4096)) {
                    Profile.Low preset 1024
                    Profile.Medium preset 2048
                    Profile.High preset 2048
                    Profile.Ultra preset 2048
                    Profile.Extreme preset 3072
                    Profile.Insane preset 4096

                    lang {
                        name = "Shadow Map Resolution"
                        comment = "Higher values produce sharper, more detailed shadows but reduce performance."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "阴影贴图分辨率"
                        comment = "数值越高，阴影越锐利、越细致，但会降低性能。"
                    }
                }
                constSlider("shadowDistance", 512.0, listOf(64.0, 128.0, 192.0, 256.0, 384.0, 512.0)) {
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
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        Profile.Low preset 192.0
                        Profile.Medium preset 256.0
                        Profile.High preset 384.0
                        Profile.Ultra preset 512.0
                        Profile.Extreme preset 512.0
                        Profile.Insane preset 512.0

                        name = "阴影渲染距离"
                        comment = "距离玩家多远的阴影贴图会被渲染。"
                        64.0 value "4 区块"
                        128.0 value "8 区块"
                        192.0 value "12 区块"
                        256.0 value "16 区块"
                        384.0 value "24 区块"
                        512.0 value "32 区块"
                    }
                }
                empty()
                empty()
                screen(1) {
                    lang {
                        name = "RTWSM"
                        comment =
                            "Rectilinear Texture Warping Shadow Mapping settings. A advanced techniques that allocate more shadow details adaptively based on scene and view."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "RTWSM"
                        comment = "直线纹理扭曲阴影贴图（RTWSM）设置。一种根据场景和视角自适应分配更多阴影细节的高级技术。"
                    }
                    toggle("SETTING_RTWSM_F", true) {
                        lang {
                            name = "Forward Importance Analysis"
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "前向重要性分析"
                        }
                    }
                    slider("SETTING_RTWSM_F_BASE", 1.0, 0.1..10.0 step 0.1) {
                        lang {
                            name = "Forward Base Value"
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "前向分析基础值"
                        }
                    }
                    slider("SETTING_RTWSM_F_MIN", -20, -20..0) {
                        lang {
                            name = "Forward Min Value"
                            comment =
                                "Minimum importance value for forward importance analysis. The actual minimum value is calculated as 2^x."
                            prefix = "2^"
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "前向分析最小值"
                            comment = "前向重要性分析的最小重要性值。实际最小值计算为 2^x。"
                            prefix = "2^"
                        }
                    }
                    slider("SETTING_RTWSM_F_D", 0.5, 0.0..2.0 step 0.05) {
                        lang {
                            name = "Forward Distance Function"
                            comment = "Reduces weight based on distance. Larger setting value means slower decay."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "前向分析距离函数"
                            comment = "根据距离降低权重。设置值越大，衰减越慢。"
                        }
                    }
                    empty()
                    toggle("SETTING_RTWSM_B", true) {
                        lang {
                            name = "Backward Importance Analysis"
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "反向重要性分析"
                        }
                    }
                    slider("SETTING_RTWSM_B_BASE", 5.0, 0.1..10.0 step 0.1) {
                        lang {
                            name = "Backward Base Value"
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "反向分析基础值"
                        }
                    }
                    slider("SETTING_RTWSM_B_MIN", -10, -20..0) {
                        lang {
                            name = "Backward Min Value"
                            comment =
                                "Minimum importance value for backward importance analysis. The actual minimum value is calculated as 2^x."
                            prefix = "2^"
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "反向分析最小值"
                            comment = "反向重要性分析的最小重要性值。实际最小值计算为 2^x。"
                            prefix = "2^"
                        }
                    }
                    slider("SETTING_RTWSM_B_D", 0.6, 0.0..2.0 step 0.05) {
                        lang {
                            name = "Backward Distance Function"
                            comment = "Reduces weight based on distance. Larger setting value means slower decay."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "反向分析距离函数"
                            comment = "根据距离降低权重。设置值越大，衰减越慢。"
                        }
                    }
                    slider("SETTING_RTWSM_B_P", 1.0, 0.0..4.0 step 0.25) {
                        lang {
                            name = "Backward Perpendicular Function"
                            comment = "Adds extra weight to surface perpendicular to light direction."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "反向分析垂直函数"
                            comment = "为垂直于光线方向的表面增加额外权重。"
                        }
                    }
                    slider("SETTING_RTWSM_B_SN", 2.0, 0.0..10.0 step 0.5) {
                        lang {
                            name = "Backward Surface Normal Function"
                            comment = "Adds extra weight to surface directly facing towards camera."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "反向分析表面法线函数"
                            comment = "为直接面向相机的表面增加额外权重。"
                        }
                    }
                    slider("SETTING_RTWSM_B_SE", 0.0, 0.0..10.0 step 0.5) {
                        lang {
                            name = "Backward Shadow Edge Function"
                            comment = "Adds extra weight for shadow edges."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "反向分析阴影边缘函数"
                            comment = "为阴影边缘增加额外权重。"
                        }
                    }
                }
                screen(1) {
                    lang {
                        name = "Soft Shadows"
                        comment = "Realistic soft shadow edges based on distance from the shadow caster using PCSS"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "软阴影"
                        comment = "使用PCSS根据与阴影投射者的距离产生真实的柔和阴影边缘"
                    }
                    slider("SETTING_PCSS_BLOCKER_SEARCH_COUNT", 6, powerOfTwoAndHalfRange(2..4)) {
                        Profile.Low preset 4
                        Profile.Medium preset 4
                        Profile.High preset 4
                        Profile.Ultra preset 6
                        Profile.Extreme preset 8
                        Profile.Insane preset 16

                        lang {
                            name = "Blocker Search Count"
                            comment =
                                "Number of samples used to determine shadow softness. Higher values improve blur radius accuracy but reduce performance."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "遮挡物搜索采样数"
                            comment = "用于确定阴影柔和度的采样数。数值越高，质量越好，但会降低性能。"
                        }
                    }
                    slider("SETTING_PCSS_SAMPLE_COUNT", 12, powerOfTwoAndHalfRange(0..6)) {
                        Profile.Low preset 4
                        Profile.Medium preset 6
                        Profile.High preset 8
                        Profile.Ultra preset 12
                        Profile.Extreme preset 16
                        Profile.Insane preset 32

                        lang {
                            name = "PCSS Sample Count"
                            comment =
                                "Number of samples used for PCSS shadow filtering. Higher values improve quality but reduce performance."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "PCSS采样数"
                            comment = "用于PCSS阴影过滤的采样数。数值越高，质量越好，但会降低性能。"
                        }
                    }
                    empty()
                    slider("SETTING_PCSS_BPF", 0.0, 0.0..10.0 step 0.5) {
                        lang {
                            name = "Base Penumbra Factor"
                            comment = "Constant amount of blur applied to all shadows, regardless of distance."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "基础半影倍率"
                            comment = "应用于所有阴影的恒定模糊量，不考虑距离。"
                        }
                    }
                    slider("SETTING_PCSS_VPF", 1.0, 0.0..2.0 step 0.1) {
                        lang {
                            name = "Variable Penumbra Factor"
                            comment =
                                "How much shadows blur based on distance from the caster. Multiplied by sun size - larger sun creates softer shadows."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "可变半影倍率"
                            comment = "阴影基于与投射者距离的模糊程度。乘以太阳大小 - 太阳越大，阴影越柔和。"
                        }
                    }
                }

            }
            screen(1) {
                lang {
                    name = "ReSTIR SSGI"
                }
                slider("SETTING_GI_INITIAL_SST_STEPS", 128, powerOfTwoAndHalfRange(4..8)) {
                    Profile.Low preset 48
                    Profile.Medium preset 64
                    Profile.High preset 96
                    Profile.Ultra preset 128
                    Profile.Extreme preset 192
                    Profile.Insane preset 256

                    lang {
                        name = "Initial Sampling Screen Space Tracing Steps"
                        comment = "Higher values improve quality but reduce performance."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "初始采样屏幕空间追踪步数"
                        comment = "数值越高，质量越好，但会降低性能。"
                    }
                }
                slider("SETTING_GI_VALIDATE_SST_STEPS", 64, powerOfTwoAndHalfRange(2..8)) {
                    Profile.Low preset 24
                    Profile.Medium preset 32
                    Profile.High preset 48
                    Profile.Ultra preset 64
                    Profile.Extreme preset 96
                    Profile.Insane preset 128

                    lang {
                        name = "Validation Sampling Screen Space Tracing Steps"
                        comment = "Higher values improve quality but reduce performance. These rays are less important thus it is fine to keep this at a lower value."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "验证采样屏幕空间追踪步数"
                        comment = "数值越高，质量越好，但会降低性能。这些光线不太重要，因此保持较低数值是可以的。"
                    }
                }
                slider("SETTING_GI_SST_THICKNESS", 0.1, 0.01..0.5 step 0.01) {
                    lang {
                        name = "Screen Space Tracing Thickness"
                        comment = "Assumed thickness for screen space tracing."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "屏幕空间追踪厚度"
                        comment = "屏幕空间追踪的假定厚度。"
                    }
                }
                slider("SETTING_GI_PROBE_FADE_START", 4, powerOfTwoRange(2..10)) {
                    lang {
                        name = "Environment Probe Fade Start"
                        comment = "Distance in blocks where environment probe lighting begins to fade out."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "环境探针淡出开始距离"
                        comment = "环境探针照明开始淡出的距离（方块数）。"
                    }
                }
                slider("SETTING_GI_PROBE_FADE_END", 32, powerOfTwoRange(2..10)) {
                    lang {
                        name = "Environment Probe Fade End"
                        comment = "Distance in blocks where environment probe lighting is completely faded out."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "环境探针淡出结束距离"
                        comment = "环境探针照明完全淡出的距离（方块数）。"
                    }
                }
                toggle("SETTING_GI_MC_SKYLIGHT_ATTENUATION", true) {
                lang {
                    name = "Vanilla Skylight Attenuation"
                    comment =
                        "Uses Minecraft's built-in skylight values to reduce sky lighting in enclosed spaces."
                }
                lang(Locale.SIMPLIFIED_CHINESE) {
                    name = "原版天空光衰减"
                    comment = "使用Minecraft内置的天空光值来减少封闭空间中的天空照明。"
                }
            }
                empty()
                toggle("SETTING_GI_SPATIAL_REUSE", true) {
                    lang {
                        name = "Spatial Reuse"
                        comment = "Reuses GI samples from nearby pixels to improve performance."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "空间重用"
                        comment = "重用来自附近像素的GI样本以提高性能。"
                    }
                }
                slider("SETTING_GI_SPATIAL_REUSE_COUNT", 6, 1..16) {
                    Profile.Low preset 4
                    Profile.Medium preset 5
                    Profile.High preset 6
                    Profile.Ultra preset 6
                    Profile.Extreme preset 6
                    Profile.Insane preset 8

                    lang {
                        name = "Spatial Reuse Sample Count"
                        comment = "Number of nearby pixels to reuse GI samples from."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "空间重用采样数"
                        comment = "重用GI样本的附近像素数量。"
                    }
                }
                toggle("SETTING_GI_SPATIAL_REUSE_COUNT_DYNAMIC", false) {
                    lang {
                        name = "Dynamic Spatial Reuse Sample Count"
                        comment = "Decreases spatial reuse sample count to reduce biases for accumulated result."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "动态空间重用采样数"
                        comment = "减少空间重用采样数以降低累积结果的偏差。"
                    }
                }
                slider("SETTING_GI_SPATIAL_REUSE_RADIUS", 64, powerOfTwoAndHalfRange(4..8)) {
                    Profile.Low preset 24
                    Profile.Medium preset 32
                    Profile.High preset 48
                    Profile.Ultra preset 64
                    Profile.Extreme preset 64
                    Profile.Insane preset 64
                    lang {
                        name = "Spatial Reuse Radius"
                        comment = "Radius to search for nearby GI samples to reuse."
                        suffix = " pixels"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "空间重用半径"
                        comment = "搜索以重用附近GI样本的半径。"
                        suffix = " 像素"
                    }
                }
                slider("SETTING_GI_SPATIAL_REUSE_FEEDBACK", 16, listOf(0) + powerOfTwoAndHalfRange(0..6)) {
                    lang {
                        name = "Spatial Reuse Feedback Threshold"
                        comment =
                            "Reuse previous frame's spatially reused samples when the number of samples is below this threshold."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "空间重用反馈阈值"
                        comment = "当样本数量低于此阈值时，重用上一帧的空间重用样本。"
                    }
                }
            }
            screen(1) {
                lang {
                    name = "Denoiser"
                }
                lang(Locale.SIMPLIFIED_CHINESE) {
                    name = "降噪器"
                }
                toggle("SETTING_DENOISER_SPATIAL", true) {
                    lang {
                        name = "Denoiser Spatial Filter"
                        comment = "Applies a spatial denoising filter to the GI results to reduce noise."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "空间降噪"
                        comment = "对GI结果应用空间降噪滤镜以减少噪点。"
                    }
                }
                slider("SETTING_DENOISER_SPATIAL_SAMPLES", 8, 1..16) {
                    Profile.Low preset 4
                    Profile.Medium preset 6
                    Profile.High preset 8
                    lang {
                        name = "Spatial Denoiser Sample Count"
                        comment = "Number of samples used in main spatial denoising pass."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "空间降噪采样数"
                        comment = "第一轮空间降噪的采样数。"
                    }
                }
                slider("SETTING_DENOISER_SPATIAL_SAMPLES_POST", 8, 1..16) {
                    Profile.Low preset 4
                    Profile.Medium preset 6
                    Profile.High preset 8
                    lang {
                        name = "Post Spatial Denoiser Sample Count"
                        comment = "Number of samples used for post spatial denoising pass."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "后空间降噪采样数"
                        comment = "用于第二轮空间降噪的采样数。"
                    }
                }
                empty()
                toggle("SETTING_DENOISER_ACCUM", true) {
                    lang {
                        name = "Temporal Accumulation"
                        comment = "Accumulates GI results over multiple frames to improve quality."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "时间累积"
                        comment = "在多帧中累积GI结果以提高质量。"
                    }
                }
                slider("SETTING_DENOISER_HISTORY_LENGTH", 256, powerOfTwoAndHalfRange(2..8)) {
                    lang {
                        name = "Temporal History Length"
                        comment = "Number of frames to accumulate for temporal denoising."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "累积历史长度"
                        comment = "用于时间降噪的累积帧数。"
                    }
                }
                empty()
                toggle("SETTING_DENOISER_FAST_HISTORY_CLAMPING", true) {
                    lang {
                        name = "Fast History Clamping"
                        comment = "Clamps to fast history to reduce ghosting artifacts."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "快速历史夹紧"
                        comment = "夹紧到快速历史以减少重影伪影。"
                    }
                }
                slider("SETTING_DENOISER_FAST_HISTORY_LENGTH", 32, powerOfTwoAndHalfRange(2..8)) {
                    Profile.Low preset 64
                    Profile.Medium preset 48
                    Profile.High preset 32
                    lang {
                        name = "Temporal Fast History Length"
                        comment =
                            "Number of frames to accumulate for the fast history that is used to keep the results responsive to changes."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "快速累积历史长度"
                        comment = "用于累积快速历史的帧数，用于快速响应变化。"
                    }
                }
                empty()
                slider("SETTING_DENOISER_FIREFLY_SUPPRESSION", 5, 0..10) {
                    lang {
                        name = "Firefly Suppression Strength"
                        comment =
                            "Reduces sudden bright spots in the GI results. Higher values increase suppression but can introduce lighting lags."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "亮点抑制强度"
                        comment = "减少GI结果中的突然出现的亮点。数值越高，抑制效果越强，但可能会增加光照延迟。"
                    }
                }
                toggle("SETTING_DENOISER_HISTORY_FIX", true) {
                    lang {
                        name = "Disocclusion Fix"
                        comment = "Fix heavy noise in disoccluded areas when the camera moves."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "遮挡消失修正"
                        comment = "修正摄像机移动时遮挡消失区域的严重噪点。"
                    }
                }
                slider("SETTING_DENOISER_HISTORY_FIX_NORMAL_WEIGHT", 5, 0..10) {
                    lang {
                        name = "Disocclusion Fix Normal Weight"
                        comment =
                            "Weight of normal similarity when performing disocclusion fix. Higher values make the fix more sensitive to normal changes and reduces excessive blur."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "遮挡消失修正法线权重"
                        comment = "修正遮挡消失时法线相似度的权重。数值越高，修正对法线变化越敏感，并减少过度模糊。"
                    }
                }
                slider("SETTING_DENOISER_HISTORY_FIX_DEPTH_WEIGHT", 5, 0..10) {
                    lang {
                        name = "Disocclusion Fix Depth Weight"
                        comment =
                            "Weight of depth similarity when performing disocclusion fix. Higher values make the fix more sensitive to depth changes and reduces excessive blur."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "遮挡消失修正深度权重"
                        comment = "修正遮挡消失时深度相似度的权重。数值越高，修正对深度变化越敏感，并减少过度模糊。"
                    }
                }
                empty()
                slider("SETTING_DENOISER_STABILIZATION_MAX_ACCUM", 64, powerOfTwoAndHalfRange(2..8)) {
                    lang {
                        name = "Stabilization Maximum Accumulated Frames"
                        comment =
                            "Maximum accumulated frames that is used for calculating blend weight. Smaller values increase responsiveness but may introduce flickering."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "降噪稳定最大累积帧数"
                        comment = "用于计算混合权重的最大累积帧数。数值越小，响应性越强，但可能会引入闪烁。"
                    }
                }
            }
        }
        screen(2) {
            lang {
                name = "Volumetrics"
            }
            lang(Locale.SIMPLIFIED_CHINESE) {
                name = "体积效果"
            }

            row {
                text("GLOBAL_SCALING", "Global Settings") {
                    valueLang(Locale.SIMPLIFIED_CHINESE, "全局设置")
                }
            }
            row {
                slider("SETTING_ATM_ALT_SCALE", 1000, listOf(1, 10, 100).flatMap { 1 * it..10 * it step it } + 1000) {
                    lang {
                        name = "Altitude Scale"
                        comment = "Value of 1 means 1 block = 1 km, value of 10 means 10 blocks = 1 km, and so on."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "大气高度比例"
                        comment = "值为 1 表示 1 个方块=1 千米，值为 10 表示 10 个方块=1 千米，依此类推。"
                    }
                }
                slider("SETTING_ATM_D_SCALE", 1000, listOf(1, 10, 100).flatMap { 1 * it..10 * it step it } + 1000) {
                    lang {
                        name = "Distance Scale"
                        comment = "Value of 1 means 1 block = 1 km, value of 10 means 10 blocks = 1 km, and so on."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "大气距离比例"
                        comment = "值为 1 表示 1 个方块=1 千米，值为 10 表示 10 个方块=1 千米，依此类推。"
                    }
                }
                screen(1) {
                    lang {
                        name = "More Global Settings"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "更多全局设置"
                    }
                    slider("SETTING_ATM_GROUND_ALBEDO_R", 45, 0..255) {
                        lang {
                            name = "Ground Color - Red"
                            comment =
                                "Red component of light reflected from the ground into the atmosphere. Affects horizon and overall color."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "地面颜色 - 红"
                            comment = "从地面反射到大气中的光线的红色分量。影响地平线和整体颜色。"
                        }
                    }
                    slider("SETTING_ATM_GROUND_ALBEDO_G", 89, 0..255) {
                        lang {
                            name = "Ground Color - Green"
                            comment =
                                "Green component of light reflected from the ground into the atmosphere. Affects horizon and overall color."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "地面颜色 - 绿"
                            comment = "从地面反射到大气中的光线的绿色分量。影响地平线和整体颜色。"
                        }
                    }
                    slider("SETTING_ATM_GROUND_ALBEDO_B", 82, 0..255) {
                        lang {
                            name = "Ground Color - Blue"
                            comment =
                                "Blue component of light reflected from the ground into the atmosphere. Affects horizon and overall color."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "地面颜色 - 蓝"
                            comment = "从地面反射到大气中的光线的蓝色分量。影响地平线和整体颜色。"
                        }
                    }
                }
            }
            row {
                empty()
            }

            row {
                text("AIR", "Air, Water") {
                    valueLang(Locale.SIMPLIFIED_CHINESE, "空气、水")
                }
            }
            row {
                slider("SETTING_EPIPOLAR_SLICES", 1024, listOf(256, 512, 1024, 2048)) {
                    Profile.Low preset 256
                    Profile.Medium preset 512
                    Profile.High preset 1024
                    Profile.Ultra preset 1024
                    Profile.Extreme preset 2048
                    Profile.Insane preset 2048

                    lang {
                        name = "Epipolar Slices"
                        comment =
                            "Number of epipolar slices used in volumetric lighting. Higher value increases quality but also decreases performance."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "对极（Epipolar）切片数"
                        comment = "体积光照中使用的对极切（Epipolar）片数。数值越高，质量越好，但也会降低性能。"
                    }
                }
                slider("SETTING_SLICE_SAMPLES", 512, listOf(128, 256, 512, 1024)) {
                    Profile.Low preset 128
                    Profile.Medium preset 256
                    Profile.High preset 512
                    Profile.Ultra preset 512
                    Profile.Extreme preset 1024
                    Profile.Insane preset 1024

                    lang {
                        name = "Slice Samples"
                        comment =
                            "Number of samples per epipolar slice used in volumetric lighting. Higher value increases quality but also decreases performance."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "对极（Epipolar）切片采样数"
                        comment = "体积光照中每个对极（Epipolar）切片使用的采样数。数值越高，质量越好，但也会降低性能。"
                    }
                }
            }
            row {
                screen(1) {
                    lang {
                        name = "Mie Coefficients"
                        comment = "Controls propeties of Haze & Fog"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "米氏系数"
                        comment = "控制雾霾和雾的属性"
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY", 2.0, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Mie Turbidity"
                            prefix = "2^"
                            comment =
                                "Overall haziness/cloudiness of the atmosphere. Higher values create mistier, more atmospheric scenes. (Actual value: 2^x)"
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "米氏浊度"
                            prefix = "2^"
                            comment = "大气的整体雾度/云量。数值越高，创建更朦胧、更有大气感的场景。（实际值：2^x）"
                        }
                    }
                    toggle("SETTING_ATM_MIE_TIME", true) {
                        lang {
                            name = "Time of Day Mie Turbidity"
                            comment =
                                "Automatically adjusts atmospheric haze throughout the day (more haze at sunrise/sunset)."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "时间段米氏浊度"
                            comment = "自动调整全天的大气雾度（日出/日落时雾度更大）。"
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_EARLY_MORNING", 4.5, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Early Morning Turbidity"
                            prefix = "2^"
                            comment = "Atmospheric haze during early morning hours (before sunrise)."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "清晨浊度"
                            prefix = "2^"
                            comment = "清晨时段（日出前）的大气雾度。"
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_SUNRISE", 5.25, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Sunrise Turbidity"
                            prefix = "2^"
                            comment =
                                "Atmospheric haze during sunrise, creating vivid colors and dramatic skies."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "日出浊度"
                            prefix = "2^"
                            comment = "日出时段的大气雾度，创造鲜艳的色彩和戏剧性的天空。"
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_MORNING", 4.0, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Morning Turbidity"
                            prefix = "2^"
                            comment = "Atmospheric haze during morning hours (after sunrise)."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "早晨浊度"
                            prefix = "2^"
                            comment = "早晨时段（日出后）的大气雾度。"
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_NOON", 2.0, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Noon Turbidity"
                            prefix = "2^"
                            comment = "Atmospheric haze at noon, typically clearest time of day."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "正午浊度"
                            prefix = "2^"
                            comment = "正午时段的大气雾度，通常是一天中最清晰的时间。"
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_AFTERNOON", 1.0, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Afternoon Turbidity"
                            prefix = "2^"
                            comment = "Atmospheric haze during afternoon hours."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "下午浊度"
                            prefix = "2^"
                            comment = "下午时段的大气雾度。"
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_SUNSET", 3.0, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Sunset Turbidity"
                            prefix = "2^"
                            comment =
                                "Atmospheric haze during sunset, creating vivid colors and dramatic skies."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "日落浊度"
                            prefix = "2^"
                            comment = "日落时段的大气雾度，创造鲜艳的色彩和戏剧性的天空。"
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_NIGHT", 3.5, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Night Turbidity"
                            prefix = "2^"
                            comment = "Atmospheric haze during early night hours."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "夜晚浊度"
                            prefix = "2^"
                            comment = "夜晚时段的大气雾度。"
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_MIDNIGHT", 4.0, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Midnight Turbidity"
                            prefix = "2^"
                            comment = "Atmospheric haze at midnight."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "午夜浊度"
                            prefix = "2^"
                            comment = "午夜时段的大气雾度。"
                        }
                    }
                    empty()
                    slider("SETTING_ATM_MIE_SCT_MUL", 1.0, 0.0..5.0 step 0.05) {
                        lang {
                            name = "Mie Scattering Multiplier"
                            comment =
                                "How much light scatters in the haze. Higher values create brighter, more visible atmospheric haze."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "米氏散射乘数"
                            comment = "光线在雾霾中散射的程度。数值越高，大气雾霾越明亮、越可见。"
                        }
                    }
                    slider("SETTING_ATM_MIE_ABS_MUL", 0.1, 0.0..2.0 step 0.01) {
                        lang {
                            name = "Mie Absorption Multiplier"
                            comment =
                                "How much light is absorbed by haze. Higher values create darker, more atmospheric fog."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "米氏吸收乘数"
                            comment = "雾霾吸收光线的程度。数值越高，雾气越暗、越有大气感。"
                        }
                    }
                }
                screen(1) {
                    lang {
                        name = "Rayleigh Coefficients"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "瑞利系数"
                    }
                    slider("SETTING_ATM_RAY_SCT_MUL", 1.0, 0.0..5.0 step 0.05) {
                        lang {
                            name = "Rayleigh Scattering Multiplier"
                            comment =
                                "Controls the intensity of blue sky color. Higher values create deeper, more saturated blue skies."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "瑞利散射乘数"
                            comment = "控制蓝天颜色的强度。数值越高，创造更深、更饱和的蓝天。"
                        }
                    }
                    slider("SETTING_ATM_OZO_ABS_MUL", 1.0, 0.0..5.0 step 0.05) {
                        lang {
                            name = "Ozone Absorption Multiplier"
                            comment =
                                "Simulates ozone layer effects on sky color. Higher values enhance blue colors at sky dome during sunrise and sunset."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "臭氧吸收乘数"
                            comment = "模拟臭氧层对天空颜色的影响。数值越高，日出和日落时天顶的深蓝色越明显。"
                        }
                    }
                }
                screen(1) {
                    lang {
                        name = "More Air Settings"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "更多空气设置"
                    }
                    slider("SETTING_SKYVIEW_RES", 256, powerOfTwoRange(7..10)) {
                        lang {
                            name = "Sky View Resolution"
                            comment =
                                "Resolution of sky calculations. Higher values improve sky color accuracy but reduce performance."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "天空视图分辨率"
                            comment = "天空计算的分辨率。数值越高，天空颜色精度越高，但会降低性能。"
                        }
                    }
                    toggle("SETTING_DEPTH_BREAK_CORRECTION", true) {
                        lang {
                            name = "Depth Break Correction"
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "深度断裂校正"
                        }
                    }
                    empty()
                    slider("SETTING_SKY_SAMPLES", 32, 16..64 step 8) {
                        lang {
                            name = "Sky Samples"
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "天空采样步进数"
                        }
                    }
                    slider("SETTING_LIGHT_SHAFT_SAMPLES", 12, 4..32 step 4) {
                        lang {
                            name = "Light Shaft Samples"
                            comment =
                                "Samples for volumetric light shafts (god rays). Higher values create smoother, more detailed rays but reduce performance."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "体积光采样步进数"
                            comment = "体积光束（丁达尔效应）的采样数。数值越高，光线越平滑、越细致，但会降低性能。"
                        }
                    }
                    slider("SETTING_LIGHT_SHAFT_SHADOW_SAMPLES", 8, 1..16 step 1) {
                        lang {
                            name = "Light Shaft Shadow Samples"
                            comment =
                                "Shadow samples in god rays. Higher values improve shadow accuracy in light shafts but reduce performance."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "体积光阴影采样数"
                            comment = "体积光渲染中的阴影采样数。数值越高，光束中的阴影精度越高，但会降低性能。"
                        }
                    }
                    slider("SETTING_LIGHT_SHAFT_DEPTH_BREAK_CORRECTION_SAMPLES", 32, 8..64 step 8) {
                        lang {
                            name = "Light Shaft Depth Break Correction Samples"
                            comment =
                                "Shadow samples used in depth break correction. Higher values improve shadow accuracy in light shafts but reduce performance."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "体积光深度断裂校正采样数"
                            comment = "深度断裂校正中使用的阴影采样数。数值越高，光束中的阴影精度越高，但会降低性能。"
                        }
                    }
                    slider("SETTING_LIGHT_SHAFT_SOFTNESS", 5, 0..10 step 1) {
                        lang {
                            name = "Light Shaft Softness"
                            comment =
                                "How soft and diffused the light shafts appear. Higher values create more diffused, rays."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "体积光柔和度"
                            comment = "光束的柔和和扩散程度。数值越高，创造更扩散的光线。"
                        }
                    }
                }
            }
            row {
                screen(1) {
                    lang {
                        name = "Water Surface"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "水面"
                    }
                    toggle("SETTING_WATER_REFRACT_APPROX", true) {
                        lang {
                            name = "Approximate Refraction"
                            comment =
                                "Approximated refraction direction that works better with screen space refraction."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "折射近似"
                            comment = "使用近似的折射方向，更适合屏幕空间折射。"
                        }
                    }
                    toggle("SETTING_WATER_CAUSTICS", true) {
                        lang {
                            name = "Water Caustics"
                            comment =
                                "Shows light patterns on surfaces beneath water, like you see at the bottom of pools."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "启用水焦散"
                            comment = "显示水下表面上的光斑图案，就像您在泳池底部看到的那样。"
                        }
                    }
                    empty()
                    slider("SETTING_WATER_NORMAL_SCALE", 1.5, 0.0..5.0 step 0.5) {
                        lang {
                            name = "Water Normal Intensity"
                            comment =
                                "Intensity of water surface waves and ripples. Higher values create choppier water."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "水法线强度"
                            comment = "水面波浪和涟漪的强度。数值越高，水越波涛汹涌。"
                        }
                    }
                    empty()
                    toggle("SETTING_WATER_PARALLAX", true) {
                        lang {
                            name = "Water Parallax"
                            comment =
                                "Creates realistic depth in water waves, making them appear 3D instead of flat."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "启用水视差"
                            comment = "给水波添加立体感，使它们看起来是3D而不是平面的。"
                        }
                    }
                    slider("SETTING_WATER_PARALLAX_STRENGTH", 1.5, 0.0..5.0 step 0.5) {
                        lang {
                            name = "Water Parallax Strength"
                            comment =
                                "How deep and three-dimensional water waves appear. Higher values create more pronounced depth."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "水视差强度"
                            comment = "水波的深度和三维感。数值越高，深度越明显。"
                        }
                    }
                    slider("SETTING_WATER_PARALLAX_LINEAR_STEPS", 8, powerOfTwoAndHalfRange(2..5)) {
                        lang {
                            name = "Water Parallax Linear Sample Steps"
                            comment =
                                "Samples for wave depth effect. Higher values reduce visual breaks between waves but reduce performance."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "水视差线性采样步数"
                            comment = "波浪深度效果的采样数。数值越高，波浪间的视觉断裂越少，但会降低性能。"
                        }
                    }
                    slider("SETTING_WATER_PARALLAX_SECANT_STEPS", 2, 1..8) {
                        lang {
                            name = "Water Parallax Secant Sample Steps"
                            comment =
                                "Additional refinement passes for wave depth. Higher values create smoother waves but reduce performance."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "水视差割线采样步数"
                            comment = "波浪深度的额外细化。数值越高，波浪越平滑，但会降低性能。"
                        }
                    }
                }
                screen(1) {
                    lang {
                        name = "Water Volume"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "水体积渲染"
                    }
                    slider("SETTING_WATER_SCATTERING_REFRACTION_APPROX", true) {
                        lang {
                            name = "Approximate Refraction Light Shafts"
                            comment = "Approximate under water light shafts causes by water waves"
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "模拟折射光束"
                            comment = "模拟由水面波动引起的水下光束"
                        }
                    }
                    slider("SETTING_WATER_SCATTERING_REFRACTION_APPROX_CONTRAST", 5, 0..12) {
                        lang {
                            name = "Refraction Light Shaft Contrast"
                            comment = "Sharpness of underwater light rays created by surface waves. "
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "折射光束对比度"
                            comment = "由水面波浪创造的水下光线的锐度。"
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
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "散射系数 - 红"
                            suffix = " %"
                            comment = "红光在水中反弹的程度。数值越低，水越偏蓝。"
                        }
                    }
                    slider("SETTING_WATER_SCATTERING_G", 22, 0..100) {
                        lang {
                            name = "Scattering Coefficient - Green"
                            suffix = " %"
                            comment = "How much green light bounces in water. Affects overall water color tone."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "散射系数 - 绿"
                            suffix = " %"
                            comment = "绿光在水中反弹的程度。影响整体水色色调。"
                        }
                    }
                    slider("SETTING_WATER_SCATTERING_B", 38, 0..100) {
                        lang {
                            name = "Scattering Coefficient - Blue"
                            suffix = " %"
                            comment = "How much blue light bounces in water. Higher values create bluer water."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "散射系数 - 蓝"
                            suffix = " %"
                            comment = "蓝光在水中反弹的程度。数值越高，水越蓝。"
                        }
                    }
                    slider("SETTING_WATER_SCATTERING_MULTIPLIER", -8.75, -15.0..-5.0 step 0.25) {
                        lang {
                            name = "Scattering Coefficient Multiplier"
                            prefix = "2^"
                            comment =
                                "Global multiplier for how much light bounces in water. Higher values brighten underwater scenes. (Multiplier: 2^x)"
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "散射系数倍数"
                            prefix = "2^"
                            comment = "光在水中反弹程度的全局倍数。数值越高，水下场景越明亮。（倍数：2^x）"
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
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "吸收系数 - 红"
                            suffix = " %"
                            comment = "红光在水下消失的速度。数值越高，红色消失越快，水越蓝。"
                        }
                    }
                    slider("SETTING_WATER_ABSORPTION_G", 40, 0..100) {
                        lang {
                            name = "Absorption Coefficient - Green"
                            suffix = " %"
                            comment =
                                "How quickly green light fades underwater. Affects visibility distance and water color."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "吸收系数 - 绿"
                            suffix = " %"
                            comment = "绿光在水下消失的速度。影响可见距离和水色。"
                        }
                    }
                    slider("SETTING_WATER_ABSORPTION_B", 24, 0..100) {
                        lang {
                            name = "Absorption Coefficient - Blue"
                            suffix = " %"
                            comment =
                                "How quickly blue light fades underwater. Lower values maintain blue color in deeper water."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "吸收系数 - 蓝"
                            suffix = " %"
                            comment = "蓝光在水下消失的速度。数值越低，更深的水中保持蓝色。"
                        }
                    }
                    slider("SETTING_WATER_ABSORPTION_MULTIPLIER", -9.25, -15.0..-5.0 step 0.25) {
                        lang {
                            name = "Absorption Coefficient Multiplier"
                            prefix = "2^"
                            comment =
                                "Global multiplier for water absorption. Higher values create murkier water with less visibility. (Multiplier: 2^x)"
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "吸收系数倍数"
                            prefix = "2^"
                            comment = "水吸收的全局倍数。数值越高，水越浑浊，可见度越低。（倍数：2^x）"
                        }
                    }
                    empty()
                    slider("SETTING_WATER_LIGHT_SHAFT_SOFTNESS", 7, 0..10 step 1) {
                        lang {
                            name = "Light Shaft Softness"
                            comment =
                                "How soft and diffused the light shafts appear. Higher values create more diffused, rays."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "体积光柔和度"
                            comment = "光束的柔和和扩散程度。数值越高，创造更扩散的光线。"
                        }
                    }
                    slider("SETTING_WATER_SHADOW_SAMPLE", 64, powerOfTwoAndHalfRange(4..8)) {
                        lang {
                            name = "Shadow Samples"
                            comment =
                                "Samples for shadows visible underwater. Higher values improve shadow smoothness but reduce performance."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "阴影采样数"
                            comment = "水下可见阴影的采样数。数值越高，阴影越平滑，但会降低性能。"
                        }
                    }
                    slider("SETTING_WATER_SHADOW_SAMPLE_POOL_SIZE", 8, 2..16 step 2) {
                        lang {
                            name = "Shadow Sample Pool Size"
                            comment = "Higher values increase shadowing quality but also decrease performance."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "阴影采样池大小"
                            comment = "数值越高，阴影质量越高，但也会降低性能。"
                        }
                    }
                }
            }
            row {
                empty()
            }

            row {
                text("CLOUDS", "Clouds") {
                    valueLang(Locale.SIMPLIFIED_CHINESE, "云")
                }
            }
            row {
                screen(1) {
                    lang {
                        name = "Cloud Lighting"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "云照明"
                    }
                    slider("SETTING_CLOUDS_MS_RADIUS", -2.5, -5.0..0.0 step 0.25) {
                        lang {
                            name = "Multi-Scattering Radius"
                            prefix = "2^"
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "多重散射半径"
                        }
                    }
                    slider("SETTING_CLOUDS_AMB_BACKSCATTER_FACTOR", 0.5, 0.0..1.0 step 0.05) {
                        lang {
                            name = "Ambient Backscatter Phase Factor"
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "环境后向散射相位因子"
                        }
                    }
                }
            }
            row {
                text("LOW_CLOUDS", "Low Clouds") {
                    valueLang(Locale.SIMPLIFIED_CHINESE, "低云")
                }
            }
            row {
                toggle("SETTING_CLOUDS_CU", true) {
                    lang {
                        name = "Enable Low Clouds"
                        comment = "Toggles puffy, volumetric clouds at lower altitudes."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "启用低云"
                        comment = "开关低空的蓬松体积云。"
                    }
                }
                screen(1) {
                    lang {
                        name = "Rendering Settings"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "渲染设置"
                    }
                    toggle("SETTING_CLOUDS_LOW_UPSCALE_FACTOR", 2, 0..6) {
                        Profile.Low preset 6
                        Profile.Medium preset 4
                        Profile.High preset 3
                        Profile.Ultra preset 2
                        Profile.Extreme preset 1
                        Profile.Insane preset 0

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
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "升采样因子"
                            comment = "以较低分辨率渲染云然后升采样。数值越高，性能越好，但可能减少细节。"
                            0 value "1.0 倍"
                            1 value "1.5 倍"
                            2 value "2.0 倍"
                            3 value "2.5 倍"
                            4 value "3.0 倍"
                            5 value "3.5 倍"
                            6 value "4.0 倍"
                        }
                    }
                    slider("SETTING_CLOUDS_LOW_MAX_ACCUM", 16, powerOfTwoAndHalfRange(2..7)) {
                        Profile.Low preset 48
                        Profile.Medium preset 32
                        Profile.High preset 24
                        Profile.Ultra preset 16
                        Profile.Extreme preset 12
                        Profile.Insane preset 8
                        lang {
                            name = "Max Accumulation"
                            comment =
                                "Frames blended for smooth clouds. Higher values create smoother clouds but may cause ghosting during fast movement."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "最大累积帧数"
                            comment = "混合以获得平滑云的帧数。数值越高，云越平滑，但在快速移动时可能导致重影。"
                        }
                    }
                    slider("SETTING_CLOUDS_LOW_CONFIDENCE_CURVE", 4.0, 1.0..8.0 step 0.5) {
                        lang {
                            name = "Confidence Curve"
                            comment =
                                "How quickly clouds sharpen over time. Higher values sharpen faster but may show more noise initially."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "置信度曲线"
                            comment = "云随时间变锐利的速度。数值越高，变锐利越快，但初始时可能显示更多噪点。"
                        }
                    }
                    slider("SETTING_CLOUDS_LOW_VARIANCE_CLIPPING", 0.25, 0.0..1.0 step 0.05) {
                        lang {
                            name = "Variance Clipping"
                            comment =
                                "Prevents cloud trails during movement. Higher values reduce ghosting but may increase flickering."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "方差裁剪"
                            comment = "防止移动时的云拖尾。数值越高，重影越少，但可能增加闪烁。"
                        }
                    }
                    empty()
                    slider("SETTING_CLOUDS_LOW_STEP_MIN", 48, 16..128 step 8) {
                        Profile.Low preset 24
                        Profile.Medium preset 24
                        Profile.High preset 32
                        Profile.Ultra preset 48
                        Profile.Extreme preset 64
                        Profile.Insane preset 128
                        lang {
                            name = "Ray Marching Min Step"
                            comment =
                                "Minimum samples through clouds. This value is typically used in clouds directly on top. Higher values improve detail but reduce performance."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "光线步进最小步数"
                            comment = "穿过云的最小采样数。此值通常用于正上方的云。数值越高，细节越好，但会降低性能。"
                        }
                    }
                    slider("SETTING_CLOUDS_LOW_STEP_MAX", 128, 32..256 step 8) {
                        Profile.Low preset 64
                        Profile.Medium preset 64
                        Profile.High preset 96
                        Profile.Ultra preset 128
                        Profile.Extreme preset 192
                        Profile.Insane preset 256
                        lang {
                            name = "Ray Marching Max Step"
                            comment =
                                "Maximum samples through thick clouds.  This value is typically used in clouds near horizon. Higher values improve quality of dense clouds but reduce performance."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "光线步进最大步数"
                            comment =
                                "穿过厚云的最大采样数。此值通常用于地平线附近的云。数值越高，密集云的质量越好，但会降低性能。"
                        }
                    }
                }
                screen(1) {
                    lang {
                        name = "Modeling Settings"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "建模设置"
                    }
                    slider("SETTING_CLOUDS_CU_HEIGHT", 1.0, 0.0..8.0 step 0.1) {
                        lang {
                            name = "Cloud Altitude"
                            suffix = " km"
                            comment = "Altitude where cumulus clouds begin to form."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "云高度"
                            suffix = " 千米"
                            comment = "积云开始形成的高度。"
                        }
                    }
                    slider("SETTING_CLOUDS_CU_THICKNESS", 2.0, 0.0..4.0 step 0.1) {
                        lang {
                            name = "Cloud Layer Thickness"
                            suffix = " km"
                            comment =
                                "Vertical thickness of the cloud layer. Thicker clouds are more dramatic and puffy."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "云层厚度"
                            suffix = " 千米"
                            comment = "云层的垂直厚度。更厚的云更具戏剧性和蓬松感。"
                        }
                    }
                    slider("SETTING_CLOUDS_CU_DENSITY", 1.0, 0.0..4.0 step 0.05) {
                        lang {
                            name = "Cloud Density"
                            suffix = " x"
                            comment =
                                "How thick and opaque clouds appear. Higher values create denser, more solid-looking clouds."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "云浓度"
                            suffix = " 倍"
                            comment = "云的厚度和不透明度。数值越高，云越浓密、看起来越坚实。"
                        }
                    }
                    slider("SETTING_CLOUDS_CU_COVERAGE", 0.5, 0.0..1.0 step 0.05) {
                        lang {
                            name = "Coverage"
                            comment =
                                "How much of the sky is covered by clouds. 0 = clear sky, 1 = completely overcast."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "云层覆盖率"
                            comment = "云覆盖天空的程度。0 = 晴空，1 = 完全阴天。"
                        }
                    }
                    slider("SETTING_CLOUDS_CU_PHASE_RATIO", 0.9, 0.0..1.0 step 0.05) {
                        lang {
                            name = "Cumulus Phase Ratio"
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "积云相位函数比例"
                        }
                    }
                    empty()
                    toggle("SETTING_CLOUDS_CU_WIND", true) {
                        lang {
                            name = "Cloud Movement"
                            comment = "Allows clouds to drift across the sky over time."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "启用云运动"
                            comment = "允许云随时间飘过天空。"
                        }
                    }
                    slider("SETTING_CLOUDS_CU_WIND_SPEED", 0.0, -4.0..4.0 step 0.25) {
                        lang {
                            name = "Wind Speed"
                            comment =
                                "Speed of cloud movement. Negative values move clouds in the opposite direction."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "风速"
                            comment = "云移动的速度。负值使云向相反方向移动。"
                        }
                    }
                }
                screen(1) {
                    lang {
                        name = "Modeling Advanced Settings"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "建模高级设置"
                    }
                    slider("SETTING_CLOUDS_LOW_CONE_FACTOR", 0.5, 0.0..1.0 step 0.05) {
                        lang {
                            name = "Cone Factor"
                            comment =
                                "Controls the how pointed the cloud structures are. Higher values create sharper clouds."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "锥体因子"
                            comment = "控制云顶尖锐程度。数值越高，云越尖锐。"
                        }
                    }
                    slider("SETTING_CLOUDS_LOW_TOP_CURVE_FACTOR", 48, powerOfTwoAndHalfRange(4..10)) {
                        lang {
                            name = "Top Curve Factor"
                            comment =
                                "Controls the shape curve of cloud tops. Higher values creates more angular top."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "云顶曲线因子"
                            comment = "控制云顶的形状曲线。数值越高，云顶越有棱角。"
                        }
                    }
                    slider("SETTING_CLOUDS_LOW_BOTTOM_CURVE_FACTOR", 128, powerOfTwoAndHalfRange(4..10)) {
                        lang {
                            name = "Bottom Curve Factor"
                            comment =
                                "Controls the shape curve of cloud bottoms. Higher values creates more angular bottoms."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "云底曲线因子"
                            comment = "控制云底的形状曲线。数值越高，云底越有棱角。"
                        }
                    }
                    empty()
                    slider("SETTING_CLOUDS_LOW_BASE_FREQ", 0.0, -4.0..4.0 step 0.1) {
                        lang {
                            name = "Base Noise Frequency"
                            comment = "Controls the scale of base cloud size. Higher values create smaller clouds."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "基础噪声频率"
                            comment = "控制云的基础大小比例。数值越高，云越小。"
                        }
                    }
                    slider("SETTING_CLOUDS_LOW_CURL_FREQ", 0.0, -4.0..4.0 step 0.1) {
                        lang {
                            name = "Curl Noise Frequency"
                            comment =
                                "Controls the scale of turbulent curl patterns in clouds. Higher values create finer curls."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "卷曲噪声频率"
                            comment = "控制云中湍流卷曲图案的比例。数值越高，卷曲越细。"
                        }
                    }
                    slider("SETTING_CLOUDS_LOW_BILLOWY_FREQ", 0.0, -4.0..4.0 step 0.1) {
                        lang {
                            name = "Base Billowy Noise Frequency"
                            comment =
                                "Control the scale of billowy formations in clouds. Higher values create smaller billows."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "基础蓬松噪声频率"
                            comment = "控制云中蓬松结构的比例。数值越高，蓬松结构越小。"
                        }
                    }
                    slider("SETTING_CLOUDS_LOW_BILLOWY_CURL_STR", 0.0, -4.0..4.0 step 0.1) {
                        lang {
                            name = "Base Billowy Curl Strength"
                            comment =
                                "Modulates billowy formations with curl noise. Higher values create more turbulent billows."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "基础蓬松卷曲强度"
                            comment = "用卷曲噪声调制蓬松结构。数值越高，蓬松结构越多湍流。"
                        }
                    }
                    slider("SETTING_CLOUDS_HIGH_BILLOWY_FREQ", 0.0, -4.0..4.0 step 0.1) {
                        lang {
                            name = "Detail Billowy Noise Frequency"
                            comment =
                                "Control the scale of billowy formations in clouds. Higher values create smaller billows."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "细节蓬松噪声频率"
                            comment = "控制云中蓬松结构的比例。数值越高，蓬松结构越小。"
                        }
                    }
                    slider("SETTING_CLOUDS_HIGH_BILLOWY_CURL_STR", 0.0, -4.0..4.0 step 0.1) {
                        lang {
                            name = "Detail Billowy Curl Strength"
                            comment =
                                "Modulates billowy formations with curl noise. Higher values create more turbulent billows."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "细节蓬松卷曲强度"
                            comment = "用卷曲噪声调制蓬松结构。数值越高，蓬松结构越多湍流。"
                        }
                    }
                    slider("SETTING_CLOUDS_LOW_WISPS_FREQ", 0.0, -4.0..4.0 step 0.1) {
                        lang {
                            name = "Wisps Noise Frequency"
                            comment =
                                "Controls the scale of wispy details in clouds. Higher values create more and finer wisps."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "丝缕噪声频率"
                            comment = "控制云中丝缕细节的比例。数值越高，丝缕越细越密集。"
                        }
                    }
                    slider("SETTING_CLOUDS_LOW_WISPS_CURL_STR", 0.0, -4.0..4.0 step 0.1) {
                        lang {
                            name = "Wisps Curl Strength"
                            comment =
                                "Modulates wispy details with curl noise. Higher values create more dynamic wisps."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "丝缕卷曲强度"
                            comment = "用卷曲噪声调制丝缕细节。数值越高，丝缕越动态。"
                        }
                    }
                }
            }

            row {
                text("HIGH_CLOUDS", "High Clouds") {
                    valueLang(Locale.SIMPLIFIED_CHINESE, "高云")
                }
            }
            row {
                toggle("SETTING_CLOUDS_CI", true) {
                    lang {
                        name = "Enable High Clouds"
                        comment =
                            "Toggles wispy, high-altitude ice crystal clouds that add atmosphere to the sky."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "启用高云"
                        comment = "开关丝缕状的高空冰晶云，为天空增添大气感。"
                    }
                }
                screen(1) {
                    lang {
                        name = "Settings"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "设置"
                    }
                    slider("SETTING_CLOUDS_CI_HEIGHT", 9.0, 6.0..14.0 step 0.1) {
                        lang {
                            name = "Cloud Altitude"
                            suffix = " km"
                            comment =
                                "Altitude of cirrus clouds. Higher altitudes create thinner, more delicate wisps."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "云高度"
                            suffix = " 千米"
                            comment = "卷云的高度。高度越高，创造更薄、更精致的丝缕。"
                        }
                    }
                    slider("SETTING_CLOUDS_CI_DENSITY", 1.0, 0.0..4.0 step 0.05) {
                        lang {
                            name = "Cloud Density"
                            suffix = " x"
                            comment =
                                "How visible and opaque the cirrus clouds are. Higher values create more prominent wisps."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "云浓度"
                            suffix = " 倍"
                            comment = "卷云的可见度和不透明度。数值越高，丝缕越显著。"
                        }
                    }
                    slider("SETTING_CLOUDS_CI_COVERAGE", 0.4, 0.0..1.0 step 0.05) {
                        lang {
                            name = "Coverage"
                            comment =
                                "How much of the high sky is covered by cirrus clouds. 0 = clear, 1 = fully covered."
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "云层覆盖率"
                            comment = "卷云覆盖高空的程度。0 = 晴朗，1 = 完全覆盖。"
                        }
                    }
                    slider("SETTING_CLOUDS_CI_PHASE_RATIO", 0.6, 0.0..1.0 step 0.05) {
                        lang {
                            name = "Cirrus Phase Ratio"
                        }
                        lang(Locale.SIMPLIFIED_CHINESE) {
                            name = "卷云相位函数比例"
                        }
                    }
                }
            }
        }
        screen(2) {
            lang {
                name = "Outer Space"
            }
            lang(Locale.SIMPLIFIED_CHINESE) {
                name = "外太空"
            }
            screen(1) {
                lang {
                    name = "Sun & Moon"
                }
                lang(Locale.SIMPLIFIED_CHINESE) {
                    name = "太阳和月亮"
                }
                slider("SETTING_SUN_RADIUS", 1.0, (-7..10).map { 2.0.pow(it) }) {
                    lang {
                        name = "Sun Size"
                        comment =
                            "Size of the sun in the sky. 1.0 = realistic size. Larger suns create softer, wider shadows."
                        suffix = " R"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "太阳大小"
                        comment = "天空中太阳的大小。1.0 = 真实大小。更大的太阳创造更柔和、更宽的阴影。"
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
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "太阳距离"
                        comment = "以天文单位（AU）为单位的太阳距离，相对于真实太阳距离1.496亿千米。"
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
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "太阳路径旋转"
                        comment = "调整太阳轨迹的角度。改变阳光和阴影方向的角度。"
                        suffix = " °"
                    }
                }
                toggle("SETTING_REAL_SUN_TEMPERATURE", true) {
                    lang {
                        name = "Realistic Sun Color"
                        comment =
                            "Uses the real sun's color temperature (5772 K) for accurate warm yellow-white sunlight."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "真实太阳温度"
                        comment = "使用真实太阳的色温（5772 K）以获得准确的温暖黄白色阳光。"
                    }
                }
                slider("SETTING_SUN_TEMPERATURE", 5700, (1000..10000 step 100) + (11000..50000 step 1000)) {
                    lang {
                        name = "Sun Color Temperature"
                        comment =
                            "Color of sunlight in Kelvin. Lower = warmer/redder (sunset), Higher = cooler/bluer (noon). Default: 5700 K."
                        suffix = " K"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "太阳温度"
                        comment =
                            "阳光的色温（开尔文）。数值越低 = 越暖/越红（日落），数值越高 = 越冷/越蓝（正午）。默认：5700 K。"
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
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "月亮大小"
                        comment = "天空中月亮的大小。1.0 = 真实大小（约为地球直径的1/4）。"
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
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "月亮距离"
                        comment = "相对于真实的月球距离（384,399千米）。影响月亮在天空中的视觉大小。"
                        suffix = " D"
                    }
                }
                slider("SETTING_MOON_ALBEDO", 0.12, 0.01..1.0 step 0.01) {
                    lang {
                        name = "Moon Brightness"
                        comment =
                            "How reflective the moon surface is. 0.12 = realistic (moon reflects 12% of sunlight). Higher = brighter nights."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "月亮亮度"
                        comment = "月球表面的反射率。0.12 = 真实值（月球反射12%的阳光）。数值越高 = 夜晚越亮。"
                    }
                }
                slider("SETTING_MOON_COLOR_R", 0.8, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Moon Tint - Red"
                        comment =
                            "Red component of moon color. Adjust to create warmer or cooler moonlight."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "月亮色调 - 红"
                        comment = "月亮颜色的红色分量。调整以创造更暖或更冷的月光。"
                    }
                }
                slider("SETTING_MOON_COLOR_G", 0.9, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Moon Tint - Green"
                        comment = "Green component of moon color. Affects the overall tone of moonlight."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "月亮色调 - 绿"
                        comment = "月亮颜色的绿色分量。影响月光的整体色调。"
                    }
                }
                slider("SETTING_MOON_COLOR_B", 1.0, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Moon Tint - Blue"
                        comment =
                            "Blue component of moon color. Higher values create cooler, bluer moonlight."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "月亮色调 - 蓝"
                        comment = "月亮颜色的蓝色分量。数值越高，创造更冷、更蓝的月光。"
                    }
                }
            }
            screen(1) {
                lang {
                    name = "Stars"
                }
                lang(Locale.SIMPLIFIED_CHINESE) {
                    name = "星星"
                }
                slider("SETTING_STARMAP_INTENSITY", 6, 0..16) {
                    lang {
                        name = "Star Brightness"
                        comment =
                            "Overall brightness of stars in the night sky. Higher values make stars more visible."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "星星亮度"
                        comment = "夜空中星星的整体亮度。数值越高，星星越可见。"
                    }
                }
                slider("SETTING_STARMAP_BRIGHT_STAR_BOOST", 4, 0..8) {
                    lang {
                        name = "Bright Star Enhancement"
                        comment =
                            "Extra brightness for the most prominent stars. Creates more realistic star size variation."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "亮星增强"
                        comment = "为最突出的星星增加亮度。创造更真实的星星大小变化。"
                    }
                }
                slider("SETTING_STARMAP_GAMMA", 0.8, 0.1..2.0 step 0.1) {
                    lang {
                        name = "Star Contrast"
                        comment =
                            "Adjusts contrast between bright and dim stars. Lower values make faint stars more visible."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "星星对比度"
                        comment = "调整明亮和暗淡星星之间的对比度。数值越低，暗淡的星星越可见。"
                    }
                }
            }
        }
        screen(2) {
            lang {
                name = "Post Processing"
            }
            lang(Locale.SIMPLIFIED_CHINESE) {
                name = "后处理"
            }
            screen(1) {
                lang {
                    name = "Depth of Field"
                }
                lang(Locale.SIMPLIFIED_CHINESE) {
                    name = "景深"
                }
                toggle("SETTING_DOF", false) {
                    lang {
                        name = "Depth of Field"
                        comment =
                            "Blurs distant or nearby objects like a camera lens, focusing attention on what you're looking at."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "启用景深"
                        comment = "像相机镜头一样模糊远处或近处的物体，将注意力集中在您正在看的东西上。"
                    }
                }
                empty()
                slider("SETTING_DOF_FOCAL_LENGTH", 50.0, listOf(18.0, 24.0, 35.0, 50.0, 75.0, 100.0)) {
                    lang {
                        name = "Focal Length"
                        suffix = " mm"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "焦距"
                        suffix = " 毫米"
                    }
                }
                slider("SETTING_DOF_F_STOP", 1.4, listOf(1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0)) {
                    lang {
                        name = "F-Stop"
                        prefix = "f/"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "光圈"
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
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "散景形状"
                        0 value "圆形"
                        1 value "六边形"
                    }
                }
                empty()
                slider("SETTING_DOF_QUALITY", 3, 1..5) {
                    lang {
                        name = "Blur Quality"
                        comment =
                            "Quality of depth of field blur. Higher values create smoother blur but reduce performance."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "模糊质量"
                        comment = "景深模糊的质量。数值越高，模糊越平滑，但会降低性能。"
                    }
                }
                slider("SETTING_DOF_MAX_SAMPLE_RADIUS", 8, listOf(2, 4, 8, 12, 16, 20, 24)) {
                    lang {
                        name = "Maximum Blur Radius"
                        comment =
                            "Maximum blur distance in pixels. Should match your aperture setting - too low cuts off blur, too high causes artifacts."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "最大模糊半径"
                        comment = "以像素为单位的最大模糊距离。应与您的光圈设置匹配 - 太低会截断模糊，太高会导致伪影。"
                    }
                }
                slider("SETTING_DOF_MASKING_HEURISTIC", 8, 0..32) {
                    lang {
                        name = "Masking Heuristic"
                        comment =
                            "How strictly to separate foreground from background blur. Higher values prevent blur bleeding between objects."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "遮罩阈值"
                        comment = "严格分离前景和背景模糊的程度。数值越高，防止物体间的模糊渗透。"
                    }
                }
                empty()
                toggle("SETTING_DOF_MANUAL_FOCUS", false) {
                    lang {
                        name = "Manual Focus"
                        comment =
                            "Set focus distance manually instead of automatically focusing on what you're looking at."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "手动对焦"
                        comment = "手动设置对焦距离，而不是自动对焦到您正在看的东西。"
                    }
                }
                slider("SETTING_DOF_FOCUS_DISTANCE_COARSE_COARSE", 0, 0..10000 step 100) {
                    lang {
                        name = "Focus Distance (Coarse x100)"
                        suffix = " m"
                        comment =
                            "Rough focus distance adjustment in meters. Only works with Manual Focus enabled."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "对焦距离（粗调 x100）"
                        suffix = " 米"
                        comment = "以米为单位的粗略对焦距离调整。仅在启用手动对焦时有效。"
                    }
                }
                slider("SETTING_DOF_FOCUS_DISTANCE_COARSE", 10, 1..100) {
                    lang {
                        name = "Focus Distance (Coarse)"
                        suffix = " m"
                        comment =
                            "Rough focus distance adjustment in meters. Only works with Manual Focus enabled."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "对焦距离（粗调）"
                        suffix = " 米"
                        comment = "以米为单位的粗略对焦距离调整。仅在启用手动对焦时有效。"
                    }
                }
                slider("SETTING_DOF_FOCUS_DISTANCE_FINE", 0.0, -1.0..1.0 step 0.01) {
                    lang {
                        name = "Focus Distance (Fine-Tune)"
                        suffix = " m"
                        comment = "Precise focus distance adjustment. Adds/subtracts from coarse setting."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "对焦距离（微调）"
                        suffix = " 米"
                        comment = "精确的对焦距离调整。从粗调设置加/减。"
                    }
                }
                slider("SETTING_DOF_FOCUS_TIME", 2.0, 0.0..10.0 step 0.25) {
                    lang {
                        name = "Focus Speed"
                        comment =
                            "How quickly focus adjusts when looking at different distances. Lower = faster, higher = more cinematic."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "对焦速度"
                        comment = "看向不同距离时对焦调整的速度。数值越低 = 越快，数值越高 = 更有电影感。"
                    }
                }
                toggle("SETTING_DOF_SHOW_FOCUS_PLANE", false) {
                    lang {
                        name = "Show Focus Plane"
                        comment =
                            "Displays the exact distance that's in focus, helpful for adjusting manual focus."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "显示对焦平面"
                        comment = "显示对焦的确切距离，有助于调整手动对焦。"
                    }
                }
            }
            screen(1) {
                lang {
                    name = "Bloom"
                }
                lang(Locale.SIMPLIFIED_CHINESE) {
                    name = "泛光（Bloom）"
                }
                toggle("SETTING_BLOOM", true) {
                    lang {
                        name = "Bloom"
                        comment =
                            "Makes bright areas glow and bleed into surrounding pixels, like light overexposing a camera."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "启用泛光"
                        comment = "使明亮区域发光并渗透到周围像素，就像光线使相机过曝一样。"
                    }
                }
                slider("SETTING_BLOOM_INTENSITY", 0.0, -8.0..8.0 step 0.25) {
                    lang {
                        name = "Bloom Intensity"
                        comment =
                            "How bright the bloom glow effect is. Higher values create more intense, dramatic glowing."
                        prefix = "2^"
                        suffix = " x"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "泛光强度"
                        comment = "泛光发光效果的亮度。数值越高，发光越强烈、越戏剧化。"
                    }
                }
                slider("SETTING_BLOOM_RADIUS", 1.0, 1.0..5.0 step 0.5) {
                    lang {
                        name = "Bloom Spread"
                        comment =
                            "How far the bloom glow spreads. Higher values create wider halos but may make the whole screen hazy."
                        suffix = " x"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "泛光半径"
                        comment = "泛光发光扩散的距离。数值越高，光晕越宽，但可能使整个屏幕模糊。"
                    }
                }
                slider("SETTING_BLOOM_PASS", 8, 1..10) {
                    lang {
                        name = "Bloom Passes"
                        comment =
                            "Processing passes for bloom. Higher values increase glow reach and smoothness but reduce performance."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "泛光层级"
                        comment = "泛光的处理次数。数值越高，发光范围和平滑度越大，但会降低性能。"
                    }
                }
                empty()
                slider("SETTING_BLOOM_UNDERWATER_BOOST", 2.0, 0.0..8.0 step 0.25) {
                    lang {
                        name = "Underwater Glow Boost"
                        comment =
                            "Extra bloom intensity when underwater, creating a dreamy, diffused underwater atmosphere."
                        prefix = "2^"
                        suffix = " x"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "水下泛光增强"
                        comment = "水下时的额外泛光强度，创造梦幻般扩散的水下氛围。"
                    }
                }
            }
            screen(1) {
                lang {
                    name = "Purkinje Effect (Night Vision)"
                }
                lang(Locale.SIMPLIFIED_CHINESE) {
                    name = "浦肯野效应（夜视）"
                }
                toggle("SETTING_PURKINJE_EFFECT", true) {
                    lang {
                        name = "Purkinje Effect"
                        comment =
                            "Simulates how human eyes lose color vision in darkness, creating a more realistic night experience."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "启用浦肯野效应"
                        comment = "模拟人眼在黑暗中失去色觉的方式，创造更真实的夜晚体验。"
                    }
                }
                slider("SETTING_PURKINJE_EFFECT_MIN_LUM", -10.0, -10.0..1.0 step 0.5) {
                    lang {
                        name = "Minimum Luminance"
                        prefix = "10^"
                        suffix = " cd/m²"
                        comment =
                            "Below this brightness, colors fade to monochrome. Lower = colors disappear in dimmer light."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "最小亮度"
                        prefix = "10^"
                        suffix = " cd/m²"
                        comment = "低于此亮度，颜色褪为单色。数值越低 = 颜色在更暗的光线中消失。"
                    }
                }
                slider("SETTING_PURKINJE_EFFECT_MAX_LUM", -2.0, -10.0..1.0 step 0.5) {
                    lang {
                        name = "Maximum Luminance"
                        prefix = "10^"
                        suffix = " cd/m²"
                        comment =
                            "Above this brightness, colors appear fully. Higher = need brighter light to see full colors."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "最大亮度"
                        prefix = "10^"
                        suffix = " cd/m²"
                        comment = "高于此亮度，颜色完全显现。数值越高 = 需要更亮的光线才能看到完整的颜色。"
                    }
                }
                empty()
                slider("SETTING_PURKINJE_EFFECT_CR", 0.9, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Night Vision Tint - Red"
                        comment =
                            "Red tint of monochrome night vision. Default creates bluish night vision like real eyes."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "夜视色调 - 红"
                        comment = "单色夜视的红色色调。默认值创造类似真实眼睛的偏蓝夜视。"
                    }
                }
                slider("SETTING_PURKINJE_EFFECT_CG", 0.95, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Night Vision Tint - Green"
                        comment = "Green tint of monochrome night vision."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "夜视色调 - 绿"
                        comment = "单色夜视的绿色色调。"
                    }
                }
                slider("SETTING_PURKINJE_EFFECT_CB", 1.0, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Night Vision Tint - Blue"
                        comment =
                            "Blue tint of monochrome night vision. Higher values create bluer night scenes."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "夜视色调 - 蓝"
                        comment = "单色夜视的蓝色色调。数值越高，夜晚场景越蓝。"
                    }
                }
            }
            screen(1) {
                lang {
                    name = "Exposure"
                }
                lang(Locale.SIMPLIFIED_CHINESE) {
                    name = "曝光"
                }
                toggle("SETTING_EXPOSURE_MANUAL", false) {
                    lang {
                        name = "Manual Exposure"
                        comment =
                            "Lock exposure to a fixed value instead of automatically adjusting to scene brightness."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "手动曝光"
                        comment = "锁定曝光到固定值，而不是自动调整到场景亮度。"
                    }
                }
                slider("SETTING_EXPOSURE_MANUAL_EV_COARSE", 3, -32..32) {
                    lang {
                        name = "Exposure EV (Coarse)"
                        comment =
                            "Rough brightness adjustment in EV stops. Negative = darker, positive = brighter."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "曝光EV（粗调）"
                        comment = "以EV档位为单位的粗略亮度调整。负值 = 更暗，正值 = 更亮。"
                    }
                }
                slider("SETTING_EXPOSURE_MANUAL_EV_FINE", 0.0, -1.0..1.0 step 0.01) {
                    lang {
                        name = "Exposure EV (Fine-Tune)"
                        comment = "Precise brightness adjustment. Adds/subtracts from coarse setting."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "曝光EV（微调）"
                        comment = "精确的亮度调整。从粗调设置加/减。"
                    }
                }
                empty()
                slider("SETTING_EXPOSURE_MIN_EV", -3.0, -32.0..32.0 step 0.5) {
                    lang {
                        name = "Auto Exposure Min EV"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "自动曝光最小值"
                    }
                }
                slider("SETTING_EXPOSURE_MAX_EV", 10.0, -32.0..32.0 step 0.5) {
                    lang {
                        name = "Auto Exposure Max EV"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "自动曝光最大值"
                    }
                }
                slider("SETTING_EXPOSURE_EMISSIVE_WEIGHTING", -3.0, -5.0..5.0 step 0.5) {
                    lang {
                        name = "Emissive Weighting"
                        comment =
                            "Weighting multiplier for emissive block pixels. Lower value = less influence. Higher value = more influence."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "自发光权重"
                        comment = "自发光方块像素的权重乘数。数值越低 = 影响越小。"
                    }
                }
                slider("SETTING_EXPOSURE_DISTANCE_WEIGHTING", 0.5, 0.0..5.0 step 0.5) {
                    lang {
                        name = "Distance Weighting"
                        comment =
                            "How much distance from the player influences exposure. Higher = adjusts more to close scene."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "距离权重"
                        comment = "玩家距离对曝光的影响程度。数值越高 = 更多地调整到近处的物体。"
                    }
                }
                slider("SETTING_EXPOSURE_CENTER_WEIGHTING", 4.0, 0.0..8.0 step 0.1) {
                    lang {
                        name = "Center Focus Priority"
                        comment =
                            "How much the center of the screen influences exposure. Higher = adjusts more to what you're looking at directly."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "中心焦点优先级"
                        comment = "屏幕中心对曝光的影响程度。数值越高 = 更多地调整到您直接看的东西。"
                    }
                }
                slider("SETTING_EXPOSURE_CENTER_WEIGHTING_CURVE", 3.0, 1.0..8.0 step 0.1) {
                    lang {
                        name = "Center Focus Sharpness"
                        comment =
                            "How sharply center weighting focuses on the middle. Higher = tighter focus on screen center, ignoring edges more."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "中心焦点锐度"
                        comment = "中心权重聚焦于中间的锐利程度。数值越高 = 更紧密聚焦于屏幕中心，更多地忽略边缘。"
                    }
                }
                empty()
                slider("SETTING_EXPOSURE_AVG_LUM_MIX", 0.25, 0.0..1.0 step 0.05) {
                    lang {
                        name = "Overall Brightness Method Weight"
                        comment =
                            "Influence of overall scene brightness on exposure. Higher = adjusts more to keep average brightness consistent."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "平均亮度权重"
                        comment = "整体场景亮度对曝光的影响。数值越高 = 更多地调整以保持平均亮度一致。"
                    }
                }
                slider("SETTING_EXPOSURE_AVG_LUM_TIME", 3.0, 0.0..10.0 step 0.25) {
                    lang {
                        name = "Overall Brightness Adapt Speed"
                        comment =
                            "How quickly exposure adapts based on overall brightness. Lower = faster adjustment."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "平均亮度自动曝光时间"
                        comment = "基于整体亮度的曝光适应速度。数值越低 = 调整越快。"
                    }
                }
                slider("SETTING_EXPOSURE_AVG_LUM_MIN_TARGET", 40, 1..255) {
                    lang {
                        name = "Dark Scene Target Brightness"
                        comment =
                            "Target brightness for dark environments (caves, night). Higher values make dark scenes brighter."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "暗场景目标亮度"
                        comment = "黑暗环境（洞穴、夜晚）的目标亮度。数值越高，暗场景越亮。"
                    }
                }
                slider("SETTING_EXPOSURE_AVG_LUM_MAX_TARGET", 140, 1..255) {
                    lang {
                        name = "Bright Scene Target Brightness"
                        comment =
                            "Target brightness for bright environments (daylight outdoors). Higher values make bright scenes brighter."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "亮场景目标亮度"
                        comment = "明亮环境（日光户外）的目标亮度。数值越高，亮场景越亮。"
                    }
                }
                slider(
                    "SETTING_EXPOSURE_AVG_LUM_TARGET_CURVE",
                    -2.0,
                    -4.0..4.0 step 0.1
                ) {
                    lang {
                        name = "Medium Brightness Curve"
                        comment =
                            "Affects medium-brightness scenes (sunset/sunrise). Lower values darken these transitional lighting conditions."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "中等亮度曲线"
                        comment = "影响中等亮度场景（日落/日出）。数值越低，这些过渡光照条件越暗。"
                    }
                }
                empty()
                slider("SETTING_EXPOSURE_HS_MIX", 1.0, 0.0..1.0 step 0.05) {
                    lang {
                        name = "Highlight/Shadow Areas Method Weight"
                        comment =
                            "Influence of brightest and darkest areas on exposure. Higher = prevents over/underexposure of extremes."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "高光/阴影区域权重"
                        comment = "最亮和最暗区域对曝光的影响。数值越高 = 防止极端的过曝/欠曝。"
                    }
                }
                slider("SETTING_EXPOSURE_HS_TIME", 2.0, 0.0..10.0 step 0.25) {
                    lang {
                        name = "Highlight/Shadow Areas Adapt Speed"
                        comment =
                            "How quickly exposure adapts to bright and dark regions. Lower = faster adjustment."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "高光/阴影区域适应速度"
                        comment = "曝光适应明亮和黑暗区域的速度。数值越低 = 调整越快。"
                    }
                }
                slider("SETTING_EXPOSURE_HS_MIN_EV_DELTA", -1.5, -4.0..0.0 step 0.1) {
                    lang {
                        name = "Highlight/Shadow Areas EV Delta Min"
                        comment = "How much stops can exposure adjust downwards for highlights/shadows."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "高光/阴影区域EV最低变化"
                        comment = "曝光可以为高光/阴影向下调整多少档位。"
                    }
                }
                slider("SETTING_EXPOSURE_HS_MAX_EV_DELTA", 1.5, 0.0..4.0 step 0.1) {
                    lang {
                        name = "Highlight/Shadow Areas EV Delta Max"
                        comment = "How much stops can exposure adjust upwards for highlights/shadows."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "高光/阴影区域EV最高变化"
                        comment = "曝光可以为高光/阴影向上调整多少档位。"
                    }
                }
                slider("SETTING_EXPOSURE_H_LUM", 225, 1..255) {
                    lang {
                        name = "Highlight Area Threshold"
                        comment =
                            "Brightness level considered 'highlight'. Exposure adjusts to prevent these areas from being too bright."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "高光区域阈值"
                        comment = "被视为“高光“的亮度级别。曝光调整以防止这些区域过亮。"
                    }
                }
                slider("SETTING_EXPOSURE_H_PERCENT", 5.0, 0.5..10.0 step 0.5) {
                    lang {
                        name = "Highlight Area %"
                        comment =
                            "Keeps this percentage of bright pixels from overexposing. Higher values darken overall to preserve bright details."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "高光区域占比"
                        comment = "保持此百分比的明亮像素不过曝。数值越高，整体变暗以保留明亮细节。"
                    }
                }
                slider("SETTING_EXPOSURE_S_LUM", 33, 0..255) {
                    lang {
                        name = "Shadow Area Threshold"
                        comment =
                            "Brightness level considered 'shadow'. Exposure adjusts to keep these areas visible."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "阴影区域阈值"
                        comment = "被视为“阴影”的亮度级别。曝光调整以保持这些区域可见。"
                    }
                }
                slider("SETTING_EXPOSURE_S_PERCENT", 3.0, 0.5..10.0 step 0.5) {
                    lang {
                        name = "Shadow Area %"
                        comment =
                            "Keeps this percentage of dark pixels from becoming pure black. Higher values brighten overall to reveal shadow detail."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "阴影区域占比"
                        comment = "保持此百分比的暗像素不变成纯黑。数值越高，整体变亮以显示阴影细节。"
                    }
                }
            }
            screen(1) {
                lang {
                    name = "Tone Mapping & Color Grading"
                }
                lang(Locale.SIMPLIFIED_CHINESE) {
                    name = "色调映射和调色"
                }
                slider("SETTING_TONE_MAPPING_DYNAMIC_RANGE", 15.0, 4.0..32.0 step 0.5) {
                    lang {
                        name = "Dynamic Range"
                        comment =
                            "Range of brightness levels preserved from dark to bright. Higher values maintain more detail in extremes but may look flat."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "动态范围"
                        comment = "从暗到亮保留的亮度级别范围。数值越高，在极端情况下保留更多细节，但可能看起来平淡。"
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
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "风格"
                        comment = "预配置的调色风格。选择自定义以手动调整下面的颜色。"
                        0 value "默认"
                        1 value "金色"
                        2 value "鲜明"
                        3 value "自定义"
                    }
                }
                empty()
                slider("SETTING_TONE_MAPPING_OFFSET_R", 0.0, -1.0..1.0 step 0.01) {
                    lang {
                        name = "Red Lift"
                        comment =
                            "Adds or removes red from all brightness levels. Negative = less red, positive = more red."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "红色偏移"
                        comment = "在所有亮度级别添加或移除红色。负值 = 更少红色，正值 = 更多红色。"
                    }
                }
                slider("SETTING_TONE_MAPPING_OFFSET_G", 0.0, -1.0..1.0 step 0.01) {
                    lang {
                        name = "Green Lift"
                        comment =
                            "Adds or removes green from all brightness levels. Negative = less green, positive = more green."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "绿色偏移"
                        comment = "在所有亮度级别添加或移除绿色。负值 = 更少绿色，正值 = 更多绿色。"
                    }
                }
                slider("SETTING_TONE_MAPPING_OFFSET_B", 0.0, -1.0..1.0 step 0.01) {
                    lang {
                        name = "Blue Lift"
                        comment =
                            "Adds or removes blue from all brightness levels. Negative = less blue, positive = more blue."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "蓝色偏移"
                        comment = "在所有亮度级别添加或移除蓝色。负值 = 更少蓝色，正值 = 更多蓝色。"
                    }
                }
                empty()
                slider("SETTING_TONE_MAPPING_SLOPE_R", 1.0, 0.1..2.0 step 0.01) {
                    lang {
                        name = "Red Gain"
                        comment =
                            "Multiplies red channel intensity. Below 1.0 reduces red, above 1.0 increases red in mid-tones."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "红色斜率"
                        comment = "乘以红色通道强度。低于1.0减少红色，高于1.0增加中间色调的红色。"
                    }
                }
                slider("SETTING_TONE_MAPPING_SLOPE_G", 1.0, 0.1..2.0 step 0.01) {
                    lang {
                        name = "Green Gain"
                        comment =
                            "Multiplies green channel intensity. Below 1.0 reduces green, above 1.0 increases green in mid-tones."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "绿色斜率"
                        comment = "乘以绿色通道强度。低于1.0减少绿色，高于1.0增加中间色调的绿色。"
                    }
                }
                slider("SETTING_TONE_MAPPING_SLOPE_B", 1.0, 0.1..2.0 step 0.01) {
                    lang {
                        name = "Blue Gain"
                        comment =
                            "Multiplies blue channel intensity. Below 1.0 reduces blue, above 1.0 increases blue in mid-tones."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "蓝色斜率"
                        comment = "乘以蓝色通道强度。低于1.0减少蓝色，高于1.0增加中间色调的蓝色。"
                    }
                }
                empty()
                slider("SETTING_TONE_MAPPING_POWER_R", 1.05, 0.1..2.0 step 0.01) {
                    lang {
                        name = "Red Contrast"
                        comment =
                            "Adjusts contrast in red channel. Higher values increase red contrast, making reds more dramatic."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "红色幂值"
                        comment = "调整红色通道的对比度。数值越高，红色对比度越大，使红色更具戏剧性。"
                    }
                }
                slider("SETTING_TONE_MAPPING_POWER_G", 1.05, 0.1..2.0 step 0.01) {
                    lang {
                        name = "Green Contrast"
                        comment =
                            "Adjusts contrast in green channel. Higher values increase green contrast."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "绿色幂值"
                        comment = "调整绿色通道的对比度。数值越高，绿色对比度越大。"
                    }
                }
                slider("SETTING_TONE_MAPPING_POWER_B", 1.05, 0.1..2.0 step 0.01) {
                    lang {
                        name = "Blue Contrast"
                        comment =
                            "Adjusts contrast in blue channel. Higher values increase blue contrast, making blues more dramatic."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "蓝色幂值"
                        comment = "调整蓝色通道的对比度。数值越高，蓝色对比度越大，使蓝色更具戏剧性。"
                    }
                }
                empty()
                slider("SETTING_TONE_MAPPING_SATURATION", 1.25, 0.0..2.0 step 0.01) {
                    lang {
                        name = "Color Saturation"
                        comment =
                            "Overall color intensity. 0 = black & white, 1 = normal, 2 = hyper-saturated."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "饱和度"
                        comment = "整体颜色强度。0 = 黑白，1 = 正常，2 = 超饱和。"
                    }
                }
            }
            screen(1) {
                lang {
                    name = "Anti-Aliasing"
                }
                lang(Locale.SIMPLIFIED_CHINESE) {
                    name = "抗锯齿"
                }
                toggle("SETTING_TAA", true) {
                    lang {
                        name = "Temporal Anti-Aliasing (TAA)"
                        comment =
                            "Smooths jagged edges by blending multiple frames. Highly recommended for clean image quality."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "时间抗锯齿（TAA）"
                        comment = "通过混合多个帧来平滑锯齿边缘。强烈推荐以获得干净的图像质量。"
                    }
                }
                toggle("SETTING_TAA_JITTER", true) {
                    lang {
                        name = "Sub-Pixel Jittering"
                        comment =
                            "Slightly shifts the camera each frame for better TAA quality. Required for TAA to work effectively."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "时间抖动"
                        comment = "每帧稍微移动相机以获得更好的TAA质量。TAA有效工作所必需的。"
                    }
                }
                toggle("SETTING_TAA_CURR_FILTER", 2, 0..2) {
                    lang {
                        name = "TAA Current Frame Filter"
                        comment =
                            """Type of filter used to filter current frame image for TAA.
B-Spline: Good at smoothing out aliasing but can be a bit blurry.
Catmull-Rom: Sharper but may causes ringing or halo.
Lanczos2: Sharp as Catmull-Rom but less ringing or halo. Slightly more performance intensive."""
                        0 value "B-Spline"
                        1 value "Catmull-Rom"
                        2 value "Lanczos2"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "TAA当前帧滤镜"
                        comment =
                            """用于过滤TAA当前帧图像的滤镜类型。
B-Spline：适合平滑锯齿，但可能有点模糊。
Catmull-Rom：更锐利，但可能会引起振铃或光晕。
Lanczos2：与Catmull-Rom一样锐利，但振铃或光晕较少。性能开销略高。"""
                        0 value "B-Spline"
                        1 value "Catmull-Rom"
                        2 value "Lanczos2"
                    }
                }
                toggle("SETTING_TAA_HISTORY_FILTER", 4, 1..4) {
                    lang {
                        name = "TAA History Frame Filter"
                        comment =
                            """Type of filter used to filter TAA history frame image.
Bilinear: Fast but blurry.
Catmull-Rom 5 Tap: Balanced sharpness and performance. May cause minor ringing.
Catmull-Rom 9 Tap: Sharper but more performance intensive. Less ringing than 5 Tap.
Catmull-Rom Full: Same quality as 9 Tap but more performance intensive.
Lanczos2: Sharp as Catmull-Rom but less ringing or halo. Most performance intensive."""
                        0 value "Bilinear"
                        1 value "Catmull-Rom 5 Tap"
                        2 value "Catmull-Rom 9 Tap"
                        3 value "Catmull-Rom Full"
                        4 value "Lanczos2"
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "TAA历史帧滤镜"
                        comment =
                            """用于过滤TAA历史帧图像的滤镜类型。
双线性：快速但模糊。
Catmull-Rom 5采样：平衡的清晰度和性能。可能会引起轻微的振铃。
Catmull-Rom 9采样：更清晰但性能开销更大。比5采样振铃更少。
Catmull-Rom 全采样：与9采样相同的质量，但性能开销更大。
Lanczos2：与Catmull-Rom一样清晰，但振铃或光晕较少。性能开销最大。"""
                        0 value "双线性"
                        1 value "Catmull-Rom 5采样"
                        2 value "Catmull-Rom 9采样"
                        3 value "Catmull-Rom 全采样"
                        4 value "Lanczos2"
                    }
                }
                slider("SETTING_TAA_CAS_SHARPNESS", 0.5, 0.0..1.0 step 0.05) {
                    lang {
                        name = "Sharpening Strength"
                        comment =
                            "Restores sharpness lost from anti-aliasing using AMD FidelityFX CAS. Higher values create crisper images."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "锐化强度"
                        comment = "使用AMD FidelityFX CAS恢复因抗锯齿而失去的锐度。数值越高，图像越清晰。"
                    }
                }
            }
        }
        screen(1) {
            lang {
                name = "Color Management"
                comment = "Advanced color space settings. Only change if you know what you're doing!"
            }
            lang(Locale.SIMPLIFIED_CHINESE) {
                name = "色彩管理"
                comment = "高级色彩空间设置。仅在您知道自己在做什么时更改！"
            }
            toggle("SETTING_MATERIAL_COLOR_SPACE", 1, 0..8) {
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
                    8 value "Color McSpaceFace"
                }
                lang(Locale.SIMPLIFIED_CHINESE) {
                    name = "纹理色彩空间"
                    comment = "资源包纹理的色彩空间。sRGB是大多数资源包的标准。"
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
                lang(Locale.SIMPLIFIED_CHINESE) {
                    name = "纹理伽马曲线"
                    comment = "纹理的伽马/传递函数。sRGB是大多数资源包的标准。"
                    0 value "线性"
                    4 value "指数 2.2"
                    5 value "指数 2.4"
                }
            }
            empty()
            toggle("SETTING_WORKING_COLOR_SPACE", 8, 0..8) {
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
                    8 value "Color McSpaceFace"
                }
                lang(Locale.SIMPLIFIED_CHINESE) {
                    name = "内部处理色彩空间"
                    comment = "用于光照计算的色彩空间。推荐使用ACEScg进行广色域渲染。"
                }
            }
            empty()
            toggle("SETTING_DRT_WORKING_COLOR_SPACE", 1, 0..8) {
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
                    8 value "Color McSpaceFace"
                }
                lang(Locale.SIMPLIFIED_CHINESE) {
                    name = "色调映射色彩空间"
                    comment = "用于色调映射操作的色彩空间。Rec. 2020与AgX色调映射配合更好。"
                }
            }
            empty()
            toggle("SETTING_OUTPUT_COLOR_SPACE", 1, 0..8) {
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
                    8 value "Color McSpaceFace"
                }
                lang(Locale.SIMPLIFIED_CHINESE) {
                    name = "显示器色彩空间"
                    comment = "显示器的色彩空间。标准显示器使用sRGB，广色域显示器使用Rec. 2020或DCI-P3。"
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
                lang(Locale.SIMPLIFIED_CHINESE) {
                    name = "显示器伽马曲线"
                    comment = "显示器的伽马/传递函数。大多数显示器使用sRGB，HDR显示器使用ST 2084 (PQ)。"
                    0 value "线性"
                    4 value "指数 2.2"
                    5 value "指数 2.4"
                }
            }
        }
        screen(2) {
            lang {
                name = "Miscellaneous"
            }
            lang(Locale.SIMPLIFIED_CHINESE) {
                name = "杂项"
            }
            row {
                toggle("SETTING_SCREENSHOT_MODE", false) {
                    lang {
                        name = "Screenshot Mode"
                        comment =
                            "Disables animations and temporal clamping for cleaner, higher-quality screenshots."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "截图模式"
                        comment = "禁用动画和时间钳制以获得更干净、更高质量的截图。"
                    }
                }
                slider("SETTING_SCREENSHOT_MODE_SKIP_INITIAL", 60, 10..200 step 10) {
                    lang {
                        name = "Screenshot Mode Warmup Frames"
                        comment =
                            "Frames to wait before taking screenshot, allowing lighting and effects to stabilize for best quality."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "截图模式预热帧数"
                        comment = "在拍摄截图之前等待的帧数，让光照和效果稳定以获得最佳质量。"
                    }
                }
                toggle("SETTING_CONSTELLATIONS", false) {
                    lang {
                        name = "Show Star Constellations"
                        comment = "Displays constellation lines connecting stars in the night sky."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "显示星座"
                        comment = "显示连接夜空中星星的星座线。"
                    }
                }
                slider("SETTING_TIME_CHANGE_SENSITIVITY", -5, -10..0) {
                    lang {
                        name = "Time Change Sensitivity"
                        comment =
                            "How sensitive effects are to time changes (/time set). Higher values make temporal effects more sensitive to time changes, reducing lighting lags."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "时间变化敏感度"
                        comment =
                            "效果对时间变化（/time set）的敏感程度。数值越高，时间变化对时间变化的敏感度越高，减少光照延迟。"
                    }
                }
            }
            row {
                empty()
            }
            row {
                toggle("SETTING_ASSUME_NVIDIA_GPU", false) {
                    lang {
                        name = "Assume NVIDIA GPU"
                        comment = "Forces enable NVIDIA-specific optimizations on non-NVIDIA hardware or workaround on weird driver."
                    }
                    lang(Locale.SIMPLIFIED_CHINESE) {
                        name = "假设NVIDIA GPU"
                        comment = "强制在非NVIDIA硬件上启用NVIDIA特定的优化，或在奇怪的驱动程序上进行变通。"
                    }
                }
            }
            repeat(69) {
                row {
                    empty()
                }
            }
            row {
                text("COOKIE", "Free", "Cookie")
                text("MILK", "Free", "Milk")
            }
        }
        empty()
        empty()
        screen(4) {
            lang {
                name = "Sponsors"
            }
            row {
                text("SPONSOR_TITLE1", "Special")
                text("SPONSOR_TITLE2", "Thanks")
                text("SPONSOR_TITLE3", "To")
            }
            row {
                empty()
            }
            Path("sponsors.txt").readLines().forEachIndexed { i, sname ->
                text("SPONSOR_$i", sname)
            }
        }
        screen(3) {
            lang {
                name = "Debug"
            }
            toggle("SETTING_DEBUG_WHITE_WORLD", false) {
                lang {
                    name = "White World"
                }
            }
            toggle("SETTING_DEBUG_OUTPUT", 0, 0..4) {
                lang {
                    name = "Debug Output"
                    0 value "Off"
                    1 value "TAA"
                    2 value "PostFX"
                    3 value "Tone Mapping"
                    4 value "Final"
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
            slider("SETTING_DEBUG_EV_COARSE", 0, -16..16) {
                lang {
                    name = "EV Coarse"
                }
            }
            slider("SETTING_DEBUG_EV_FINE", 0.0, -1.0..1.0 step 0.01) {
                lang {
                    name = "EV Fine"
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
            toggle("SETTING_DEBUG_GI_TEXT", false) {
                lang {
                    name = "GI Text"
                }
            }
            toggle("SETTING_DEBUG_SST", false) {
                lang {
                    name = "SST"
                }
            }
            toggle("SETTING_DEBUG_SST_STEPS", false) {
                lang {
                    name = "SST Steps"
                }
            }
            toggle("SETTING_DEBUG_TAA", false) {
                lang {
                    name = "TAA"
                }
            }
            toggle("SETTING_GI_USE_REFERENCE", false) {
                lang {
                    name = "Monte Carlo Reference"
                }
            }
        }
    }
}