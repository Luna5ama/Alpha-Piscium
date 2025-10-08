import java.io.File
import java.math.BigDecimal
import java.util.*
import kotlin.io.path.Path
import kotlin.io.path.readLines
import kotlin.math.pow

class FloatProgression(val start: Double, val endInclusive: Double, val step: Double) : Iterable<Double> {
    override fun iterator(): Iterator<Double> = object : Iterator<Double> {
        private val startBig = BigDecimal.valueOf(start)
        private val endInclusiveBig = BigDecimal.valueOf(endInclusive)
        private val stepBig = BigDecimal.valueOf(step)
        private var index = BigDecimal.ZERO

        override fun hasNext(): Boolean = startBig + stepBig * index <= endInclusiveBig

        override fun next(): Double {
            val value = startBig + stepBig * index
            index += BigDecimal.ONE
            return value.toDouble()
        }
    }
}

infix fun ClosedFloatingPointRange<Double>.step(step: Double): FloatProgression =
    FloatProgression(start, endInclusive, step)

class ScreenItem(val name: String) {
    override fun toString(): String = name

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is ScreenItem) return false

        if (name != other.name) return false

        return true
    }

    override fun hashCode(): Int {
        return name.hashCode()
    }

    companion object {
        val EMPTY = ScreenItem("<empty>")
        val WILDCARD = ScreenItem("*")
    }
}

abstract class OptionFactory {
    abstract val scope: Scope

    fun constToggle(name: String, value: Boolean, block: OptionBuilder<Boolean>.() -> Unit = {}): ScreenItem {
        val screenItem = ScreenItem(name)
        scope._addOption(OptionBuilder(name, value, true, emptyList()).apply(block))
        handleOption(screenItem)
        return screenItem
    }

    fun constToggle(
        name: String,
        value: Int,
        range: Iterable<Int>,
        block: OptionBuilder<Int>.() -> Unit = {}
    ): ScreenItem {
        val screenItem = ScreenItem(name)
        scope._addOption(OptionBuilder(name, value, true, range).apply(block))
        handleOption(screenItem)
        return screenItem
    }

    fun constToggle(
        name: String,
        value: Double,
        range: Iterable<Double>,
        block: OptionBuilder<Double>.() -> Unit = {}
    ): ScreenItem {
        val screenItem = ScreenItem(name)
        scope._addOption(OptionBuilder(name, value, true, range).apply(block))
        handleOption(screenItem)
        return screenItem
    }

    fun constSlider(name: String, value: Boolean, block: OptionBuilder<Boolean>.() -> Unit = {}): ScreenItem {
        scope._addSlider(name)
        return constToggle(name, value, block)
    }

    fun constSlider(
        name: String,
        value: Int,
        range: Iterable<Int>,
        block: OptionBuilder<Int>.() -> Unit = {}
    ): ScreenItem {
        scope._addSlider(name)
        return constToggle(name, value, range, block)
    }

    fun constSlider(
        name: String,
        value: Double,
        range: Iterable<Double>,
        block: OptionBuilder<Double>.() -> Unit = {}
    ): ScreenItem {
        scope._addSlider(name)
        return constToggle(name, value, range, block)
    }

    fun toggle(name: String, value: Boolean, block: OptionBuilder<Boolean>.() -> Unit = {}): ScreenItem {
        val screenItem = ScreenItem(name)
        scope._addOption(OptionBuilder(name, value, false, emptyList()).apply(block))
        handleOption(screenItem)
        return screenItem
    }

    fun toggle(name: String, value: Int, range: Iterable<Int>, block: OptionBuilder<Int>.() -> Unit = {}): ScreenItem {
        val screenItem = ScreenItem(name)
        scope._addOption(OptionBuilder(name, value, false, range).apply(block))
        handleOption(screenItem)
        return screenItem
    }

    fun toggle(
        name: String,
        value: Double,
        range: Iterable<Double>,
        block: OptionBuilder<Double>.() -> Unit = {}
    ): ScreenItem {
        val screenItem = ScreenItem(name)
        scope._addOption(OptionBuilder(name, value, false, range).apply(block))
        handleOption(screenItem)
        return screenItem
    }

    fun slider(name: String, value: Boolean, block: OptionBuilder<Boolean>.() -> Unit = {}): ScreenItem {
        scope._addSlider(name)
        return toggle(name, value, block)
    }

    fun slider(
        name: String,
        value: Int,
        range: Iterable<Int>,
        block: OptionBuilder<Int>.() -> Unit = {}
    ): ScreenItem {
        scope._addSlider(name)
        return toggle(name, value, range, block)
    }

    fun slider(
        name: String,
        value: Double,
        range: Iterable<Double>,
        block: OptionBuilder<Double>.() -> Unit = {}
    ): ScreenItem {
        scope._addSlider(name)
        return toggle(name, value, range, block)
    }

    protected open fun handleOption(item: ScreenItem) {}
}


class OptionBuilder<T>(
    val name: String,
    private val value: T,
    private val const: Boolean,
    private val range: Iterable<T>
) {
    private val langBuilder = LangBuilder<T>(name)

    fun lang(block: LangBuilder<T>.() -> Unit) {
        langBuilder.block()
    }

    class LangBuilder<T>(private val optionName: String, private val locale: Locale = Locale.US) {
        var name = ""
            set(value) {
                check(value.isNotEmpty()) { "Name cannot be empty" }; field = value
            }
        var comment = ""
            set(value) {
                check(value.isNotEmpty()) { "Comment cannot be empty" }; field = value
            }

        var prefix = ""
            set(value) {
                check(value.isNotEmpty()) { "Prefix cannot be empty" }; field = value
            }

        var suffix = ""
            set(value) {
                check(value.isNotEmpty()) { "Suffix cannot be empty" }; field = value
            }

        private val valueLabel = mutableMapOf<T, String>()

        infix fun T.value(label: String) {
            valueLabel[this] = label
        }

        fun build(output: Scope.Output) {
            output.writeLang(locale) {
                if (name.isNotEmpty()) appendLine("option.$optionName=$name")
                if (comment.isNotEmpty()) appendLine("option.$optionName.comment=$comment")
                if (prefix.isNotEmpty()) {
                    append("prefix.$optionName=")
                    if (prefix.startsWith(" ")) {
                        append('\\')
                    }
                    appendLine(prefix)
                }
                if (suffix.isNotEmpty()) {
                    append("suffix.$optionName=")
                    if (suffix.startsWith(" ")) {
                        append('\\')
                    }
                    appendLine(suffix)
                }
                valueLabel.forEach { (value, label) ->
                    appendLine("value.$optionName.$value=$label")
                }
            }
        }
    }

    fun build(output: Scope.Output) {
        output.writeOption {
            if (value is Boolean) {
                if (const) {
                    appendLine("const bool $name = $value;")
                } else {
                    if (value) {
                        appendLine("#define $name")
                    } else {
                        appendLine("//#define $name")
                    }
                    appendLine("#ifdef $name")
                    appendLine("#endif")
                }
            } else {
                if (const) {
                    when (value) {
                        is Int -> append("const int $name = $value;")
                        is Double -> append("const float $name = $value;")
                        else -> error("Unsupported type")
                    }
                } else {
                    append("#define $name $value")
                }
                range.joinTo(this, " ", " //[", "]")
                appendLine()
            }
        }
        langBuilder.build(output)
    }
}

class Scope : OptionFactory() {
    private lateinit var _mainScreen: ScreenBuilder
    private val _screens = mutableSetOf<ScreenBuilder>()
    private val _sliders = mutableSetOf<String>()
    private val _options = mutableSetOf<OptionBuilder<*>>()

    override val scope: Scope
        get() = this

    internal fun _addScreen(screen: ScreenBuilder) {
        check(_screens.add(screen)) { "Screen ${screen._name} already exists" }
    }

    internal fun _addOption(option: OptionBuilder<*>) {
        check(_options.add(option)) { "Option ${option.name} already exists" }
    }

    internal fun _addSlider(name: String) {
        check(_sliders.add(name)) { "Slider $name already exists" }
    }

    fun mainScreen(columns: Int, block: ScreenBuilder.() -> Unit) {
        check(!::_mainScreen.isInitialized) { "Main screen already exists" }
        _mainScreen = ScreenBuilder(this, "", columns)
        _mainScreen.apply(block)
    }

    fun build(baseShadersProperties: File): Output {
        val output = Output(baseShadersProperties)
        output.writeShadersProperties {
            _sliders.joinTo(this, " ", "sliders=")
            appendLine()
        }
        _mainScreen.build(output)
        _screens.forEach { screen ->
            screen.build(output)
        }
        _options.forEach { option ->
            option.build(output)
        }
        return output
    }

    class ScreenBuilder(override val scope: Scope, val _name: String, private val columns: Int) : OptionFactory() {
        init {
            check(!_name.contains(' ')) { "Screen name cannot contain space" }
        }

        private val options = mutableSetOf<OptionBuilder<*>>()
        private val ref = if (_name.isEmpty()) "" else ".${this@ScreenBuilder._name}"
        private val items = mutableListOf<ScreenItem>()
        private val langBuilder = LangBuilder(ref)

        fun lang(block: LangBuilder.() -> Unit) {
            langBuilder.block()
        }

        fun build(output: Output) {
            langBuilder.build(output)
            output.writeShadersProperties {
                appendLine("screen$ref.columns=$columns")
                append("screen$ref=")
                val items = if (_name.isEmpty()) (items + ScreenItem.WILDCARD) else items
                items.joinTo(this, " ")
                appendLine()
            }
        }

        fun item(item: ScreenItem) {
            items.add(item)
        }

        fun screen(name: String, columns: Int, block: ScreenBuilder.() -> Unit) {
            val screen = ScreenBuilder(scope, name, columns)
            scope._addScreen(screen)
            screen.apply(block)
            val screenItem = ScreenItem("[$name]")
            items.add(screenItem)
        }

        fun empty() {
            items.add(ScreenItem.EMPTY)
        }

        override fun handleOption(item: ScreenItem) {
            items.add(item)
        }

        class LangBuilder(private val ref: String, private val locale: Locale = Locale.US) {
            var name = ""
                set(value) {
                    check(value.isNotEmpty()) { "Name cannot be empty" }; field = value
                }

            var comment = ""
                set(value) {
                    check(value.isNotEmpty()) { "Comment cannot be empty" }; field = value
                }

            fun build(output: Output) {
                output.writeLang(locale) {
                    if (name.isNotEmpty()) appendLine("screen$ref=$name")
                    if (comment.isNotEmpty()) appendLine("screen$ref.comment=$comment")
                }
            }
        }

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is ScreenBuilder) return false

            if (_name != other._name) return false

            return true
        }

        override fun hashCode(): Int {
            return _name.hashCode()
        }
    }

    class Output(baseShadersProperties: File) {
        private val _options = StringBuilder()
        private val _lang = mutableMapOf<Locale, StringBuilder>()
        private val _shadersProperties = StringBuilder()

        init {
            _options.appendLine("// $NOTICE")
            _shadersProperties.appendLine("# $NOTICE")
            _shadersProperties.appendLine(baseShadersProperties.readText())
            _shadersProperties.appendLine()
            _shadersProperties.appendLine("# --- Generated Stuff ---")
        }

        fun writeOption(block: Appendable.() -> Unit) {
            _options.block()
        }

        fun writeLang(locale: Locale, block: Appendable.() -> Unit) {
            _lang.getOrPut(locale) {
                StringBuilder().apply {
                    appendLine("# $NOTICE")
                }
            }.block()
        }

        fun writeShadersProperties(block: Appendable.() -> Unit) {
            _shadersProperties.block()
        }

        fun writeOutput(optionGlslFile: File, shaderRoot: File) {
            val langDir = File(shaderRoot, "lang")
            langDir.mkdirs()
            optionGlslFile.writeText(_options.toString())
            _lang.forEach { (language, content) ->
                File(langDir, "${language}.lang").writeText(content.toString())
            }
            File(shaderRoot, "shaders.properties").bufferedWriter().use {
                it.append(_shadersProperties)
            }
        }

        companion object {
            const val NOTICE = "This file is generated by options.main.kts. Do not edit this file manually."
        }
    }
}

fun options(baseShadersProperties: File, shaderRootDir: File, optionGlslPath: String, block: Scope.() -> Unit) {
    val absoluteFile = shaderRootDir.absoluteFile
    Scope().apply(block).build(baseShadersProperties).writeOutput(File(absoluteFile, optionGlslPath), absoluteFile)
}

fun powerOfTwoRange(range: IntRange): List<Int> {
    return range.map { 1 shl it }
}

fun powerOfTwoRangeAndHalf(range: IntRange): List<Int> {
    return range.flatMap {
        if (it <= 1) {
            listOf(1 shl it)
        } else {
            listOf((1 shl (it - 1)) + (1 shl (it - 2)), 1 shl it)
        }
    }
}

options(File("shaders.properties"), File("../shaders"), "base/Options.glsl") {
    mainScreen(2) {
        screen("TERRAIN", 2) {
            lang {
                name = "Terrain"
            }
            screen("BLOCKLIGHT", 1) {
                lang {
                    name = "Blocklight"
                }
                slider("SETTING_FIRE_TEMPERATURE", 1400, 100..5000 step 100) {
                    lang {
                        name = "Fire Temperature"
                        comment =
                            "Temperature of fire in K (kelvin). The default value 1400 K is based on real life average."
                    }
                }
                slider("SETTING_LAVA_TEMPERATURE", 1300, 100..5000 step 100) {
                    lang {
                        name = "Lava Temperature"
                        comment =
                            "Temperature of lava in K (kelvin). The default value 1300 K is based on real life average."
                    }
                }
                slider("SETTING_EMISSIVE_STRENGTH", 4.0, 0.0..8.0 step 0.25) {
                    lang {
                        name = "Emissive Strength"
                    }
                }
                empty()
                slider("SETTING_EMISSIVE_PBR_VALUE_CURVE", 0.9, 0.1..4.0 step 0.05) {
                    lang {
                        name = "Emissive PBR Value Curve"
                    }
                }
                slider("SETTING_EMISSIVE_ALBEDO_COLOR_CURVE", 2.0, 0.1..4.0 step 0.05) {
                    lang {
                        name = "Emissive Albedo Color Curve"
                    }
                }
                slider("SETTING_EMISSIVE_ALBEDO_LUM_CURVE", 0.5, 0.0..4.0 step 0.05) {
                    lang {
                        name = "Emissive Albedo Luminance Curve"
                    }
                }
                empty()
                slider("SETTING_EMISSIVE_ARMOR_GLINT_MULT", -10, -20..0 step 1) {
                    lang {
                        name = "Emissive Armor Glint Multiplier"
                        prefix = "2^"
                    }
                }
                slider("SETTING_EMISSIVE_ARMOR_GLINT_CURVE", 1.3, 0.1..2.0 step 0.1) {
                    lang {
                        name = "Emissive Armor Glint Curve"
                    }
                }
            }
            screen("NORMAL_MAPPING", 1) {
                lang {
                    name = "Normal Mapping"
                }
                toggle("SETTING_NORMAL_MAPPING", true) {
                    lang {
                        name = "Normal Mapping"
                    }
                }
                slider("SETTING_NORMAL_MAPPING_STRENGTH", -1.0, -5.0..5.0 step 0.5) {
                    lang {
                        name = "Normal Mapping Strength"
                    }
                }
            }
            screen("SPECULAR_MAPPING", 1) {
                lang {
                    name = "Specular Mapping"
                }
                slider("SETTING_MINIMUM_F0", 12, 4..32) {
                    lang {
                        name = "Minimum F0 Factor"
                    }
                }
                empty()
                slider("SETTING_SOLID_MINIMUM_ROUGHNESS", 6, 4..16) {
                    lang {
                        name = "Minimum Solid Roughness"
                        prefix = "2^"
                    }
                }
                slider("SETTING_SOLID_MAXIMUM_ROUGHNESS", 5, 2..16) {
                    lang {
                        name = "Maximum Solid Roughness"
                        prefix = "2^"
                    }
                }
                empty()
                slider("SETTING_WATER_ROUGHNESS", 9.0, 4.0..12.0 step 0.5) {
                    lang {
                        name = "Water Roughness"
                        prefix = "2^-"
                    }
                }
                slider("SETTING_TRANSLUCENT_ROUGHNESS_REDUCTION", 1.0, 0.0..8.0 step 0.5) {
                    lang {
                        name = "Translucent Roughness Reduction"
                        prefix = "2^-"
                    }
                }
                slider("SETTING_TRANSLUCENT_MINIMUM_ROUGHNESS", 10.0, 4.0..16.0 step 0.5) {
                    lang {
                        name = "Translucent Minimum Roughness"
                        prefix = "2^-"
                    }
                }
                slider("SETTING_TRANSLUCENT_MAXIMUM_ROUGHNESS", 5.0, 1.0..16.0 step 0.5) {
                    lang {
                        name = "Translucent Maximum Roughness"
                        prefix = "2^-"
                    }
                }
                empty()
                slider("SETTING_MAXIMUM_SPECULAR_LUMINANCE", 65536, powerOfTwoRange(8..24)) {
                    lang {
                        name = "Maximum Specular Luminance"
                        comment = "Maximum luminance of specular highlights in unit of 1000 cd/m²."
                    }
                }
            }
            screen("SSS", 1) {
                slider("SETTING_SSS_STRENGTH", 1.2, 0.0..5.0 step 0.1) {
                    lang {
                        name = "SSS Strength"
                    }
                }
                slider("SETTING_SSS_HIGHLIGHT", 0.8, 0.0..1.0 step 0.01) {
                    lang {
                        name = "SSS Highlight"
                    }
                }
                slider("SETTING_SSS_SCTR_FACTOR", 4.0, 0.0..10.0 step 0.1) {
                    lang {
                        name = "SSS Scatter Factor"
                    }
                }
                empty()
                slider("SETTING_SSS_DIFFUSE_RANGE", 0.3, 0.0..4.0 step 0.1) {
                    lang {
                        name = "SSS Diffuse Range"
                    }
                }
                slider("SETTING_SSS_DEPTH_RANGE", 0.6, 0.0..4.0 step 0.1) {
                    lang {
                        name = "SSS Depth Range"
                    }
                }
                slider("SETTING_SSS_MAX_DEPTH_RANGE", 0.9, 0.0..4.0 step 0.1) {
                    lang {
                        name = "SSS Max Depth Range"
                    }
                }
            }
            empty()
            empty()
            screen("SHADOW", 2) {
                lang {
                    name = "Shadow"
                }
                constSlider("shadowMapResolution", 2048, listOf(1024, 2048, 3072, 4096)) {
                    lang {
                        name = "Shadow Map Resolution"
                    }
                }
                constSlider("shadowDistance", 192.0, listOf(64.0, 128.0, 192.0, 256.0, 384.0, 512.0)) {
                    lang {
                        name = "Shadow Render Distance"
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
                        comment = "Rectilinear Texture Warping Shadow Mapping settings"
                    }
                    slider("SETTING_RTWSM_IMAP_SIZE", 256, listOf(256, 512, 1024)) {
                        lang {
                            name = "Importance Map Resolution"
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
                        comment = "Soft Shadows settings"
                    }
                    slider("SETTING_PCSS_BLOCKER_SEARCH_COUNT", 2, listOf(1, 2, 4, 8, 16)) {
                        lang {
                            name = "Blocker Search Count"
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
                            name = "Base Penumbra Factor"
                        }
                    }
                    slider("SETTING_PCSS_VPF", 1.0, 0.0..2.0 step 0.1) {
                        lang {
                            name = "Variable Penumbra Factor"
                            comment =
                                "The penumbra factor is multiplied by the sun angular radius to determine the penumbra size. Noted that the sun angular radius is affected by the sun radius and distance settings."
                        }
                    }
                }
            }
            screen("VBGI", 1) {
                lang {
                    name = "VBGI"
                    comment = "Visibility Bitmask Global Illumination settings"
                }
                slider("SETTING_VBGI_STEPS", 32, listOf(8, 12, 16, 24, 32, 64, 96, 128)) {
                    lang {
                        name = "Step Samples"
                    }
                }
                slider("SETTING_VBGI_FALLBACK_SAMPLES", 8, powerOfTwoRange(1..5)) {
                    lang {
                        name = "Fallback Samples"
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
                    }
                }
                empty()
                toggle("SETTING_VBGI_PROBE_HQ_OCC", true) {
                    lang {
                        name = "High Quality Probe Lighting Occlusion"
                    }
                }
                slider("SETTING_VBGI_PROBE_DIR_MATCH_WEIGHT", 1, -10..10) {
                    lang {
                        name = "Probe Direction Match Weight"
                    }
                }
                slider("SETTING_VBGI_PROBE_FADE_START_DIST", 16, 0..32 step 4) {
                    lang {
                        name = "Probe Direction Match Distance Fade Start Distance"
                    }
                }
                slider("SETTING_VBGI_PROBE_FADE_END_DIST", 32, 0..64 step 4) {
                    lang {
                        name = "Probe Direction Match Distance Fade End Distance"
                    }
                }
                empty()
                toggle("SETTING_VBGI_MC_SKYLIGHT_ATTENUATION", true) {
                    lang {
                        name = "Vanilla Skylight Attenuation"
                    }
                }
                empty()
                slider("SETTING_VBGI_SKYLIGHT_STRENGTH", 1.0, 0.0..4.0 step 0.05) {
                    lang {
                        name = "Skylight Strength"
                    }
                }
                slider("SETTING_VGBI_ENV_STRENGTH", 1.0, 0.0..4.0 step 0.05) {
                    lang {
                        name = "Enviroment Probe Strength"
                    }
                }
                slider("SETTING_VGBI_IB_STRENGTH", 1.0, 0.0..4.0 step 0.05) {
                    lang {
                        name = "Indirect Bounce Strength"
                    }
                }
                empty()
                slider("SETTING_VBGI_DGI_STRENGTH", 1.0, 0.0..4.0 step 0.05) {
                    lang {
                        name = "Diffuse GI Strength"
                    }
                }
                slider("SETTING_VBGI_SGI_STRENGTH", 1.0, 0.0..4.0 step 0.05) {
                    lang {
                        name = "Specular GI Strength"
                    }
                }
                slider("SETTING_VBGI_GI_MB", 1.0, 0.0..2.0 step 0.01) {
                    lang {
                        name = "GI Multi Bounce"
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
                    }
                }
                slider("SETTING_DENOISER_REPROJ_NORMAL_EDGE_WEIGHT", 1.0, 0.0..16.0 step 0.1) {
                    lang {
                        name = "Reprojection Normal Edge Weight"
                    }
                }
                slider("SETTING_DENOISER_REPROJ_GEOMETRY_EDGE_WEIGHT", 9.0, 0.0..16.0 step 0.1) {
                    lang {
                        name = "Reprojection Geometry Edge Weight"
                    }
                }
                empty()
                slider("SETTING_DENOISER_MAX_ACCUM", 256, (2..10).map { 1 shl it }) {
                    lang {
                        name = "Max Accumulation"
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
                    }
                }
                empty()
                slider("SETTING_DENOISER_VARIANCE_BOOST_ADD_FACTOR", 10, 0..64) {
                    lang {
                        name = "Variance Boost Add"
                        comment = "Boost variance for the first few frames. Actual value is calculated as 2^-x."
                    }
                }
                slider("SETTING_DENOISER_VARIANCE_BOOST_MULTIPLY", 2.5, 1.0..4.0 step 0.1) {
                    lang {
                        name = "Variance Boost Multiply"
                        comment = "Boost variance for the first few frames."
                    }
                }
                slider("SETTING_DENOISER_VARIANCE_BOOST_FRAMES", 16, (0..6).map { 1 shl it }) {
                    lang {
                        name = "Variance Boost Frames"
                        comment = "Number of frames to boost variance."
                    }
                }
                slider("SETTING_DENOISER_VARIANCE_BOOST_DECAY", 2, 1..16 step 1) {
                    lang {
                        name = "Variance Boost Decay"
                    }
                }
                empty()
                slider("SETTING_DENOISER_MIN_VARIANCE_FACTOR",25, 0..64) {
                    lang {
                        name = "Minimum Variance Factor"
                        comment =
                            "Minimum variance factor for the filter. Smaller value generally leads to less noisy but more blurry result. This value is used to calculate center variance in Atrous filter as \"max(variance, 2^-x)\""
                    }
                }
                empty()
                slider("SETTING_DENOISER_FILTER_NORMAL_WEIGHT", 128, (0..10).map { 1 shl it }) {
                    lang {
                        name = "Filter Normal Weight"
                    }
                }
                slider("SETTING_DENOISER_FILTER_DEPTH_WEIGHT", 64, (0..10).map { 1 shl it }) {
                    lang {
                        name = "Filter Depth Weight"
                    }
                }
                slider("SETTING_DENOISER_FILTER_COLOR_WEIGHT", 56, 0..128) {
                    lang {
                        name = "Filter Color Weight"
                        comment = "Smaller value generally leads to less noisy but more blurry result."
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
            screen("AIR", 2) {
                lang {
                    name = "Air"
                }
                screen("MIE_COEFF", 1) {
                    lang {
                        name = "Mie Coefficients"
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY", 2.0, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Mie Turbidity"
                            prefix = "2^"
                        }
                    }
                    toggle("SETTING_ATM_MIE_TIME", true) {
                        lang {
                            name = "Time of Day Mie Turbidity"
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_EARLY_MORNING", 4.5, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Early Morning Turbidity"
                            prefix = "2^"
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_SUNRISE", 5.25, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Sunrise Turbidity"
                            prefix = "2^"
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_MORNING", 4.0, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Morning Turbidity"
                            prefix = "2^"
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_NOON", 2.0, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Noon Turbidity"
                            prefix = "2^"
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_AFTERNOON", 1.0, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Afternoon Turbidity"
                            prefix = "2^"
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_SUNSET", 3.0, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Sunset Turbidity"
                            prefix = "2^"
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_NIGHT", 3.5, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Night Turbidity"
                            prefix = "2^"
                        }
                    }
                    slider("SETTING_ATM_MIE_TURBIDITY_MIDNIGHT", 4.0, 0.0..8.0 step 0.25) {
                        lang {
                            name = "Midnight Turbidity"
                            prefix = "2^"
                        }
                    }
                    empty()
                    slider("SETTING_ATM_MIE_SCT_MUL", 1.0, 0.0..5.0 step 0.05) {
                        lang {
                            name = "Mie Scattering Multiplier"
                        }
                    }
                    slider("SETTING_ATM_MIE_ABS_MUL", 0.1, 0.0..2.0 step 0.01) {
                        lang {
                            name = "Mie Absorption Multiplier"
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
                        }
                    }
                    slider("SETTING_ATM_OZO_ABS_MUL", 1.0, 0.0..5.0 step 0.05) {
                        lang {
                            name = "Ozone Absorption Multiplier"
                        }
                    }
                }
                empty()
                empty()
                slider("SETTING_SKYVIEW_RES", 256, powerOfTwoRange(7..10)) {
                    lang {
                        name = "Sky View Resolution"
                    }
                }
                slider("SETTING_EPIPOLAR_SLICES", 1024, listOf(256, 512, 1024, 2048)) {
                    lang {
                        name = "Epipolar Slices"
                    }
                }
                slider("SETTING_SLICE_SAMPLES", 512, listOf(128, 256, 512, 1024)) {
                    lang {
                        name = "Slice Samples"
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
                    }
                }
                slider("SETTING_LIGHT_SHAFT_SHADOW_SAMPLES", 8, 1..16 step 1) {
                    lang {
                        name = "Light Shaft Shadow Samples"
                    }
                }
                slider("SETTING_LIGHT_SHAFT_DEPTH_BREAK_CORRECTION_SAMPLES", 32, 8..64 step 8) {
                    lang {
                        name = "Light Shaft Depth Break Correction Samples"
                    }
                }
                slider("SETTING_LIGHT_SHAFT_SOFTNESS", 5, 0..10 step 1) {
                    lang {
                        name = "Light Shaft Softness"
                    }
                }
            }
            screen("CLOUDS_LIGHTING", 1) {
                lang {
                    name = "Clouds Lighting"
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
                    }
                }
            }
            screen("LOW_CLOUDS", 1) {
                lang {
                    name = "Low Clouds"
                }
                toggle("SETTING_CLOUDS_CU", true) {
                    lang {
                        name = "Cumulus Clouds"
                    }
                }
                toggle("SETTING_CLOUDS_LOW_UPSCALE_FACTOR", 4, 0..6) {
                    lang {
                        name = "Upscale Factor"
                        0 value "1.0 x"
                        1 value "1.5 x"
                        2 value "2.0 x"
                        3 value "2.5 x"
                        4 value "3.0 x"
                        5 value "3.5 x"
                        6 value "4.0 x"
                    }
                }
                slider("SETTING_CLOUDS_LOW_MAX_ACCUM", 64, powerOfTwoRange(2..8)) {
                    lang {
                        name = "Max Accumulation"
                    }
                }
                slider("SETTING_CLOUDS_LOW_SHARPENING", 0.25, 0.0..1.0 step 0.05) {
                    lang {
                        name = "Sharpening"
                    }
                }
                slider("SETTING_CLOUDS_LOW_VARIANCE_CLIPPING", 0.5, 0.0..1.0 step 0.05) {
                    lang {
                        name = "Variance Clipping"
                    }
                }
                empty()
                slider("SETTING_CLOUDS_LOW_STEP_MIN", 24, 4..64 step 4) {
                    lang {
                        name = "Ray Marching Min Step"
                    }
                }
                slider("SETTING_CLOUDS_LOW_STEP_MAX", 64, 16..128 step 4) {
                    lang {
                        name = "Ray Marching Max Step"
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
                        name = "Cumulus Height"
                        suffix = " km"
                    }
                }
                slider("SETTING_CLOUDS_CU_THICKNESS", 1.5, 0.0..4.0 step 0.1) {
                    lang {
                        name = "Cumulus Thickness"
                        suffix = " km"
                    }
                }
                slider("SETTING_CLOUDS_CU_DENSITY", 1.0, 0.0..4.0 step 0.05) {
                    lang {
                        name = "Cumulus Density"
                        suffix = " x"
                    }
                }
                slider("SETTING_CLOUDS_CU_COVERAGE", 0.25, 0.0..1.0 step 0.05) {
                    lang {
                        name = "Cumulus Coverage"
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
                        name = "Cumulus Wind"
                    }
                }
                slider("SETTING_CLOUDS_CU_WIND_SPEED", 0.0, -4.0..4.0 step 0.25) {
                    lang {
                        name = "Cumulus Wind Speed"
                    }
                }
            }
            screen("HIGH_CLOUDS", 1) {
                lang {
                    name = "High Clouds"
                }
                toggle("SETTING_CLOUDS_CI", true) {
                    lang {
                        name = "Cirrus Clouds"
                    }
                }
                slider("SETTING_CLOUDS_CI_HEIGHT", 9.0, 6.0..14.0 step 0.1) {
                    lang {
                        name = "Cirrus Height"
                        suffix = " km"
                    }
                }
                slider("SETTING_CLOUDS_CI_DENSITY", 1.0, 0.0..4.0 step 0.05) {
                    lang {
                        name = "Cirrus Density"
                        suffix = " x"
                    }
                }
                slider("SETTING_CLOUDS_CI_COVERAGE", 0.4, 0.0..1.0 step 0.05) {
                    lang {
                        name = "Cirrus Coverage"
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
            screen("WATER", 1) {
                lang {
                    name = "Water"
                }
                toggle("SETTING_WATER_CAUSTICS", false) {
                    lang {
                        name = "Water Caustics"
                    }
                }
                empty()
                toggle("SETTING_WATER_PARALLEX", true) {
                    lang {
                        name = "Water Parallax"
                    }
                }
                slider("SETTING_WATER_PARALLEX_STRENGTH", 1.0, 0.0..2.0 step 0.05) {
                    lang {
                        name = "Water Parallax Strength"
                    }
                }
                slider("SETTING_WATER_PARALLEX_STEPS", 16, powerOfTwoRangeAndHalf(2..6)) {
                    lang {
                        name = "Water Parallax Steps"
                    }
                }
                empty()
                slider("SETTING_WATER_NORMAL_SCALE", 1.0, 0.0..2.0 step 0.05) {
                    lang {
                        name = "Water Normal Scale"
                    }
                }
                empty()
                slider("SETTING_WATER_SCATTERING_R", 14, 0..100) {
                    lang {
                        name = "Scattering Coefficient Red"
                        suffix = " %"
                    }
                }
                slider("SETTING_WATER_SCATTERING_G", 22, 0..100) {
                    lang {
                        name = "Scattering Coefficient Green"
                        suffix = " %"
                    }
                }
                slider("SETTING_WATER_SCATTERING_B", 38, 0..100) {
                    lang {
                        name = "Scattering Coefficient Blue"
                        suffix = " %"
                    }
                }
                slider("SETTING_WATER_SCATTERING_MULTIPLIER", -8.75, -15.0..-5.0 step 0.25) {
                    lang {
                        name = "Scattering Coefficient Multiplier"
                        prefix = "2^"
                    }
                }
                empty()
                slider("SETTING_WATER_ABSORPTION_R", 100, 0..100) {
                    lang {
                        name = "Absorption Coefficient Red"
                        suffix = " %"
                    }
                }
                slider("SETTING_WATER_ABSORPTION_G", 40, 0..100) {
                    lang {
                        name = "Absorption Coefficient Green"
                        suffix = " %"
                    }
                }
                slider("SETTING_WATER_ABSORPTION_B", 24, 0..100) {
                    lang {
                        name = "Absorption Coefficient Blue"
                        suffix = " %"
                    }
                }
                slider("SETTING_WATER_ABSORPTION_MULTIPLIER", -9.25, -15.0..-5.0 step 0.25) {
                    lang {
                        name = "Absorption Coefficient Multiplier"
                        prefix = "2^"
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
                        name = "Sun Radius"
                        comment = "Radius of sun relative to real sun radius of 696342 km."
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
                        name = "Sun Path Rotation"
                        comment = "Rotation of sun path in degrees."
                        suffix = " °"
                    }
                }
                toggle("SETTING_REAL_SUN_TEMPERATURE", true) {
                    lang {
                        name = "Use Real Sun Temperature"
                        comment = "Use real sun temperature of 5772 K."
                    }
                }
                slider("SETTING_SUN_TEMPERATURE", 5700, (1000..10000 step 100) + (11000..50000 step 1000)) {
                    lang {
                        name = "Sun Temperature"
                        comment = "Temperature of sun in K (kelvin). Affects the color and intensity of sunlight."
                        suffix = " K"
                    }
                }
                empty()
                slider("SETTING_MOON_RADIUS", 1.0, (-7..10).map { 2.0.pow(it) }) {
                    lang {
                        name = "Moon Radius"
                        comment = "Radius of moon relative to real moon radius of 1737.4 km."
                        suffix = " R"
                    }
                }
                slider("SETTING_MOON_DISTANCE", 1.0, (-7..10).map { 2.0.pow(it) }) {
                    lang {
                        name = "Moon Distance"
                        comment = "Distance relative to real moon distance of 384399."
                        suffix = " D"
                    }
                }
                slider("SETTING_MOON_ALBEDO", 0.12, 0.01..1.0 step 0.01) {
                    lang {
                        name = "Moon Albedo"
                    }
                }
                slider("SETTING_MOON_COLOR_R", 0.8, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Moon Color Red"
                    }
                }
                slider("SETTING_MOON_COLOR_G", 0.9, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Moon Color Green"
                    }
                }
                slider("SETTING_MOON_COLOR_B", 1.0, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Moon Color Blue"
                    }
                }
            }
            screen("STARS", 1) {
                lang {
                    name = "Stars"
                }
                slider("SETTING_STARMAP_INTENSITY", 6, 0..16) {
                    lang {
                        name = "Starmap Intensity"
                    }
                }
                slider("SETTING_STARMAP_BRIGHT_STAR_BOOST", 4, 0..8) {
                    lang {
                        name = "Starmap Bright Star Boost"
                    }
                }
                slider("SETTING_STARMAP_GAMMA", 0.8, 0.1..2.0 step 0.1) {
                    lang {
                        name = "Starmap Gamma"
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
                        name = "Depth of Field Enabled"
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
                        name = "Aperture Shape"
                        0 value "Circle"
                        1 value "Hexagon"
                    }
                }
                empty()
                slider("SETTING_DOF_QUALITY", 3, 1..5) {
                    lang {
                        name = "Quality"
                    }
                }
                slider("SETTING_DOF_MAX_SAMPLE_RADIUS", 8, listOf(2, 4, 8, 12, 16, 20, 24)) {
                    lang {
                        name = "Max Sample Radius"
                    }
                }
                slider("SETTING_DOF_MASKING_HEURISTIC", 8, 0..32) {
                    lang {
                        name = "Masking Heuristic"
                        comment = "Larger value means more aggressive masking."
                    }
                }
                empty()
                toggle("SETTING_DOF_MANUAL_FOCUS", false) {
                    lang {
                        name = "Manual Focus"
                    }
                }
                slider("SETTING_DOF_FOCUS_DISTANCE_COARSE", 10, 1..100) {
                    lang {
                        name = "Focus Distance Coarse"
                        suffix = " m"
                    }
                }
                slider("SETTING_DOF_FOCUS_DISTANCE_FINE", 0.0, -1.0..1.0 step 0.01) {
                    lang {
                        name = "Focus Distance Fine"
                        suffix = " m"
                    }
                }
                slider("SETTING_DOF_FOCUS_TIME", 2.0, 0.0..10.0 step 0.25) {
                    lang {
                        name = "Focus Transition Time"
                    }
                }
                toggle("SETTING_DOF_SHOW_FOCUS_PLANE", false) {
                    lang {
                        name = "Show Focus Plane"
                    }
                }
            }
            screen("BLOOM", 1) {
                lang {
                    name = "Bloom"
                }
                toggle("SETTING_BLOOM", true) {
                    lang {
                        name = "Bloom Enabled"
                    }
                }
                slider("SETTING_BLOOM_INTENSITY", 1.0, 0.1..5.0 step 0.1) {
                    lang {
                        name = "Bloom Intensity"
                    }
                }
                slider("SETTING_BLOOM_RADIUS", 1.0, 1.0..5.0 step 0.5) {
                    lang {
                        name = "Bloom Radius"
                    }
                }
                slider("SETTING_BLOOM_PASS", 8, 1..10) {
                    lang {
                        name = "Bloom Pass Count"
                    }
                }
            }
            screen("PURKINJE_EFFECT", 1) {
                lang {
                    name = "Purkinje Effect"
                }
                toggle("SETTING_PURKINJE_EFFECT", true) {
                    lang {
                        name = "Purkinje Effect Enabled"
                    }
                }
                slider("SETTING_PURKINJE_EFFECT_MIN_LUM", -8.0, -10.0..1.0 step 0.1) {
                    lang {
                        name = "Minimum Luminance"
                        prefix = "10^"
                        suffix = " cd/m²"
                        comment = "Luminance below this value will become colorless"
                    }
                }
                slider("SETTING_PURKINJE_EFFECT_MAX_LUM", 0.0, -10.0..1.0 step 0.1) {
                    lang {
                        name = "Maximum Luminance"
                        prefix = "10^"
                        suffix = " cd/m²"
                        comment = "Luminance above this value will become fully colored"
                    }
                }
                empty()
                slider("SETTING_PURKINJE_EFFECT_CR", 0.9, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Scoptic Color Red"
                    }
                }
                slider("SETTING_PURKINJE_EFFECT_CG", 0.95, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Scoptic Color Green"
                    }
                }
                slider("SETTING_PURKINJE_EFFECT_CB", 1.0, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Scoptic Color Blue"
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
                    }
                }
                slider("SETTING_EXPOSURE_MANUAL_EV_COARSE", 3, -32..32) {
                    lang {
                        name = "Manual Exposure EV Coarse"
                    }
                }
                slider("SETTING_EXPOSURE_MANUAL_EV_FINE", 0.0, -1.0..1.0 step 0.01) {
                    lang {
                        name = "Manual Exposure EV Fine"
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
                        name = "Center Weighting"
                        comment = "Weight of center pixels in the exposure calculation. " +
                            "This value is the extra weight added to the very center pixel."
                    }
                }
                slider("SETTING_EXPOSURE_CENTER_WEIGHTING_CURVE", 3.0, 1.0..8.0 step 0.1) {
                    lang {
                        name = "Center Weighting Curve"
                        comment = "Curve for center weighting. " +
                            "Larger value will cause it to focus more on the center pixels."
                    }
                }
                empty()
                slider("SETTING_EXPOSURE_AVG_LUM_MIX", 0.25, 0.0..1.0 step 0.05) {
                    lang {
                        name = "Average Luminance Weight"
                        comment = "Weight of average luminance AE in the final exposure value."
                    }
                }
                slider("SETTING_EXPOSURE_AVG_LUM_TIME", 4.0, 0.0..10.0 step 0.25) {
                    lang {
                        name = "Average Luminance AE Time"
                    }
                }
                slider("SETTING_EXPOSURE_AVG_LUM_MIN_TARGET", 40, 1..255) {
                    lang {
                        name = "Average Luminance Minimum Target"
                        comment = "Target average luminance value for dark scene such as caves, indoors, and nighttime."
                    }
                }
                slider("SETTING_EXPOSURE_AVG_LUM_MAX_TARGET", 70, 1..255) {
                    lang {
                        name = "Average Luminance Maximum Target"
                        comment = "Target average luminance value for bright scene such as daytime outdoors."
                    }
                }
                slider("SETTING_EXPOSURE_AVG_LUM_TARGET_CURVE", 0.5, (0.01..1.0 step 0.01) + (1.1..4.0 step 0.1)) {
                    lang {
                        name = "Average Luminance Target Curve"
                        comment = "Curve for average luminance target. " +
                            "Usually affects scene with medium brightness such as sunset/sunrise. " +
                            "Smaller value will make those scenes darker."
                    }
                }
                empty()
                slider("SETTING_EXPOSURE_HS_MIX", 1.0, 0.0..1.0 step 0.05) {
                    lang {
                        name = "Highlight/Shadow Weight"
                        comment = "Weight of highlight/shadow based AE in the final exposure value."
                    }
                }
                slider("SETTING_EXPOSURE_HS_TIME", 2.0, 0.0..10.0 step 0.25) {
                    lang {
                        name = "Highlight/Shadow AE Time"
                    }
                }
                slider("SETTING_EXPOSURE_H_LUM", 197, 1..255) {
                    lang {
                        name = "Highlight Luminance"
                        comment = "Luminance threshold for highlight."
                    }
                }
                slider("SETTING_EXPOSURE_H_PERCENT", 5.0, 0.5..10.0 step 0.5) {
                    lang {
                        name = "Highlight %"
                        comment = "Adjusting exposure to keep the specified percentage of pixels in the highlight part of histogram."
                    }
                }
                slider("SETTING_EXPOSURE_S_LUM", 16, 0..255) {
                    lang {
                        name = "Shadow Luminance"
                        comment = "Luminance threshold for shadow."
                    }
                }
                slider("SETTING_EXPOSURE_S_PERCENT", 3.0, 0.5..10.0 step 0.5) {
                    lang {
                        name = "Shadow %"
                        comment = "Adjusting exposure to keep the specified percentage of pixels in the shadow part of histogram."
                    }
                }
            }
            screen("TONE_MAPPING", 1) {
                lang {
                    name = "Tone Mapping"
                }
                slider("SETTING_TONE_MAPPING_DYNAMIC_RANGE", 13.5, 4.0..32.0 step 0.5) {
                    lang {
                        name = "Dynamic Range"
                    }
                }
                empty()
                toggle("SETTING_TONE_MAPPING_LOOK", 3, 0..3) {
                    lang {
                        name = "Look"
                        0 value "Default"
                        1 value "Golden"
                        2 value "Punchy"
                        3 value "Custom"
                    }
                }
                empty()
                slider("SETTING_TONE_MAPPING_OFFSET_R", 0.0, -1.0..1.0 step 0.01) {
                    lang {
                        name = "Offset Red"
                    }
                }
                slider("SETTING_TONE_MAPPING_OFFSET_G", 0.0, -1.0..1.0 step 0.01) {
                    lang {
                        name = "Offset Green"
                    }
                }
                slider("SETTING_TONE_MAPPING_OFFSET_B", 0.0, -1.0..1.0 step 0.01) {
                    lang {
                        name = "Offset Blue"
                    }
                }
                empty()
                slider("SETTING_TONE_MAPPING_SLOPE_R", 1.0, 0.1..2.0 step 0.01) {
                    lang {
                        name = "Slope Red"
                    }
                }
                slider("SETTING_TONE_MAPPING_SLOPE_G", 1.0, 0.1..2.0 step 0.01) {
                    lang {
                        name = "Slope Green"
                    }
                }
                slider("SETTING_TONE_MAPPING_SLOPE_B", 1.0, 0.1..2.0 step 0.01) {
                    lang {
                        name = "Slope Blue"
                    }
                }
                empty()
                slider("SETTING_TONE_MAPPING_POWER_R", 1.0, 0.1..2.0 step 0.01) {
                    lang {
                        name = "Power Red"
                    }
                }
                slider("SETTING_TONE_MAPPING_POWER_G", 1.0, 0.1..2.0 step 0.01) {
                    lang {
                        name = "Power Green"
                    }
                }
                slider("SETTING_TONE_MAPPING_POWER_B", 1.0, 0.1..2.0 step 0.01) {
                    lang {
                        name = "Power Blue"
                    }
                }
                empty()
                slider("SETTING_TONE_MAPPING_SATURATION", 1.1, 0.0..2.0 step 0.01) {
                    lang {
                        name = "Saturation"
                    }
                }
            }
            screen("AA", 1) {
                lang {
                    name = "Anti Aliasing"
                }
                toggle("SETTING_TAA", true) {
                    lang {
                        name = "Temporal Anti Aliasing"
                    }
                }
                toggle("SETTING_TAA_JITTER", true) {
                    lang {
                        name = "Temporal Jitter"
                    }
                }
                slider("SETTING_TAA_CAS_SHARPNESS", 1.5, 0.0..5.0 step 0.25) {
                    lang {
                        name = "AMD FidelityFX CAS Sharpness"
                    }
                }
            }
        }
        screen("COLOR_MANAGEMENT", 1) {
            lang {
                name = "Color Management"
            }
            toggle("SETTING_MATERIAL_COLOR_SPACE", 1, 0..7) {
                lang {
                    name = "Input Material Color Space"
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
                    name = "Input Material Transfer Function"
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
                    name = "Rendering Working Color Space"
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
                    name = "DRT Working Color Space"
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
                    name = "Output Color Space"
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
                    name = "Output Transfer Function"
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
                name = "Misc"
            }
            toggle("SETTING_SCREENSHOT_MODE", false) {
                lang {
                    name = "Screenshot Mode"
                }
            }
            toggle("SETTING_SCREENSHOT_MODE_SKIP_INITIAL", false) {
                lang {
                    name = "Screenshot Mode Skip Initial Frames"
                }
            }
            toggle("SETTING_CONSTELLATIONS", false) {
                lang {
                    name = "Show Constellations"
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
            Path("sponsors.txt").readLines().forEachIndexed { i,sname ->
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