import java.io.File
import java.math.BigDecimal
import java.util.*
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
            check(label.isNotEmpty()) { "Label cannot be empty" }
            valueLabel[this] = label
        }

        fun build(output: Scope.Output) {
            output.writeLang(locale) {
                if (name.isNotEmpty()) appendLine("option.$optionName=$name")
                if (comment.isNotEmpty()) appendLine("option.$optionName.comment=$comment")
                if (prefix.isNotEmpty()) appendLine("option.$optionName.prefix=$prefix")
                if (suffix.isNotEmpty()) appendLine("option.$optionName.suffix=$suffix")
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
                range.joinTo(this, " ", " // [", "]")
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

options(File("shaders.properties"), File("../shaders"), "base/Options.glsl") {
    mainScreen(2) {
        screen("LIGHTING", 2) {
            lang {
                name = "Lighting"
            }
            screen("SUNLIGHT", 1) {
                lang {
                    name = "Sunlight"
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
                constSlider("sunPathRotation", -30.0, -90.0..90.0 step 1.0) {
                    lang {
                        name = "Sun Path Rotation"
                        comment = "Rotation of sun path in degrees."
                        suffix = " °"
                    }
                }
                empty()
                toggle("SETTING_REAL_SUN_TEMPERATURE", true) {
                    lang {
                        name = "Use Real Sun Temperature"
                        comment = "Use real sun temperature of 5772 K."
                    }
                }
                slider("SETTING_SUN_TEMPERATURE", 5700, 1000..20000 step 100) {
                    lang {
                        name = "Sun Temperature"
                        comment = "Temperature of sun in K (kelvin). Affects the color and intensity of sunlight."
                        suffix = " K"
                    }
                }
            }
            screen("BLOCKLIGHT", 1) {
                lang {
                    name = "Blocklight"
                }
                slider("SETTING_FIRE_TEMPERATURE", 1600, 1000..20000 step 100) {
                    lang {
                        name = "Fire Temperature"
                        comment =
                            "Temperature of fire in K (kelvin). The default value 1600 K is based on real life average."
                    }
                }
                slider("SETTING_LAVA_TEMPERATURE", 1400, 1000..20000 step 100) {
                    lang {
                        name = "Lava Temperature"
                        comment =
                            "Temperature of lava in K (kelvin). The default value 1400 K is based on real life average."
                    }
                }
                slider("SETTING_EMISSIVE_STRENGTH", 0.25, 0.0..2.0 step 0.01) {
                    lang {
                        name = "Emissive Strength"
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
                    slider("SETTING_RTWSM_IMAP_SIZE", 512, listOf(256, 512, 1024)) {
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
                    slider("SETTING_RTWSM_F_D", 1024, listOf(0) + (0..16).map { 1 shl it }) {
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
                    slider("SETTING_RTWSM_B_D", 128, listOf(0) + (0..10).map { 2 shl it }) {
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
                    slider("SETTING_PCSS_SAMPLE_COUNT", 8, listOf(1, 2, 4, 8, 16, 32, 64)) {
                        lang {
                            name = "Sample Count"
                        }
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
                    slider("SETTING_PCSS_BPF", 2.0, 0.0..10.0 step 0.5) {
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
            screen("SSS", 1) {
                lang {
                    name = "Subsurface Scattering"
                    comment = "Subsurface Scattering settings"
                }
                slider("SETTING_SSS_STRENGTH", 1.0, 0.0..5.0 step 0.1) {
                    lang {
                        name = "Strength"
                    }
                }
                slider("SETTING_SSS_HIGHLIGHT", 0.5, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Highlight"
                    }
                }
                slider("SETTING_SSS_SCTR_FACTOR", 5.0, 0.0..10.0 step 0.1) {
                    lang {
                        name = "Scatter Factor"
                    }
                }
                empty()
                slider("SETTING_SSS_DIFFUSE_RANGE", 0.5, 0.0..4.0 step 0.1) {
                    lang {
                        name = "Diffuse Range"
                    }
                }
                slider("SETTING_SSS_DEPTH_RANGE", 0.5, 0.0..4.0 step 0.1) {
                    lang {
                        name = "Depth Range"
                    }
                }
                slider("SETTING_SSS_MAX_DEPTH_RANGE", 0.9, 0.0..4.0 step 0.1) {
                    lang {
                        name = "Max Depth Range"
                    }
                }
            }
            screen("VBGI", 1) {
                lang {
                    name = "VBGI"
                    comment = "Visibility Bitmask Global Illumination settings"
                }
                slider("SETTING_VBGI_STEPS", 32, listOf(8, 12, 16, 24, 32, 64)) {
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
                slider("SETTING_VBGI_THICKNESS", 1.0, 0.1..10.0 step 0.1) {
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
                slider("SETTING_VBGI_PROBE_DIR_MATCH_WEIGHT", 16, (0..10).map { 1 shl it }) {
                    lang {
                        name = "Probe Direction Match Weight"
                    }
                }
                slider("SETTING_VBGI_PROBE_DIR_MATCH_DIST_THRESHOLD", 1024, (0..10).map { 1 shl it }) {
                    lang {
                        name = "Probe Direction Match Distance Threshold"
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
                empty()
                slider("SETTING_DENOISER_REPROJ_NORMAL_STRICTNESS", 8, (0..10).map { 1 shl it }) {
                    lang {
                        name = "Reprojection Normal Strictness"
                    }
                }
                slider("SETTING_DENOISER_MAX_ACCUM", 64, (2..8).map { 1 shl it }) {
                    lang {
                        name = "Max Accumulation"
                    }
                }
                slider("SETTING_DENOISER_ACCUM_DECAY", 1.0, 0.5..3.0 step 0.01) {
                    lang {
                        name = "Accumulation Decay"
                        comment =
                            "Current mix rate decay factor for temporal accumulation. Larger value means faster decay."
                    }
                }
                empty()
                slider("SETTING_DENOISER_FILTER_NORMAL_STRICTNESS", 256, (0..10).map { 1 shl it }) {
                    lang {
                        name = "Filter Normal Strictness"
                    }
                }
                slider("SETTING_DENOISER_FILTER_DEPTH_STRICTNESS", 64, (0..10).map { 1 shl it }) {
                    lang {
                        name = "Filter Depth Strictness"
                    }
                }
                slider("SETTING_DENOISER_FILTER_COLOR_STRICTNESS", 4, 0..32) {
                    lang {
                        name = "Filter Color Strictness"
                    }
                }
            }
        }
        screen("Material", 1) {
            lang {
                name = "Material"
                comment = "Material settings"
            }
            toggle("SETTING_NORMAL_MAPPING", true) {
                lang {
                    name = "Normal Mapping"
                }
            }
            slider("SETTING_NORMAL_MAPPING_STRENGTH", 0.5, 0.0..1.0 step 0.01) {
                lang {
                    name = "Normal Mapping Strength"
                }
            }
        }
        screen("ATMOSPHERE", 1) {
            lang {
                name = "Atmosphere"
            }
            slider("SETTING_ATM_ALT_SCALE", 100, listOf(1, 10, 100).flatMap { 1 * it..10 * it step it } + 1000) {
                lang {
                    name = "Atmosphere Altitude Scale"
                    comment = "Value of 1 means 1 block = 1 km, value of 10 means 10 blocks = 1 km, and so on."
                }
            }
            slider("SETTING_ATM_D_SCALE", 100, listOf(1, 10, 100).flatMap { 1 * it..10 * it step it } + 1000) {
                lang {
                    name = "Atmosphere Distance Scale"
                    comment = "Value of 1 means 1 block = 1 km, value of 10 means 10 blocks = 1 km, and so on."
                }
            }
            empty()
            slider("SETTING_ATM_MIE_MUL", 1.0, 0.0..5.0 step 0.01) {
                lang {
                    name = "Mie Scattering Multiplier"
                }
            }
            slider("SETTING_ATM_MIE_ABS", 1.1, 0.0..5.0 step 0.01) {
                lang {
                    name = "Mie Absorption"
                }
            }
            slider("SETTING_ATM_RAY_MUL", 1.0, 0.0..5.0 step 0.01) {
                lang {
                    name = "Rayleigh Scattering Multiplier"
                }
            }
            empty()
            slider("SETTING_EPIPOLAR_SLICES", 512, listOf(256, 512, 1024, 2048)) {
                lang {
                    name = "Epipolar Slices"
                }
            }
            slider("SETTING_SLICE_SAMPLES", 256, listOf(128, 256, 512, 1024)) {
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
            slider("SETTING_SKY_SAMPLES", 64, 16..128) {
                lang {
                    name = "Sky Samples"
                }
            }
            slider("SETTING_LIGHT_SHAFT_SAMPLES", 32, 8..64) {
                lang {
                    name = "Light Shaft Samples"
                }
            }
        }
        screen("POSTFX", 2) {
            lang {
                name = "Post Processing"
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
                slider("SETTING_EXPOSURE_MANUAL_VALUE", 0.0, -10.0..10.0 step 0.1) {
                    lang {
                        name = "Manual Exposure Value"
                    }
                }
                empty()
                slider("SETTING_EXPOSURE_MIN_EXP", -4.0, -10.0..10.0 step 0.1) {
                    lang {
                        name = "Auto Exposure Min"
                    }
                }
                slider("SETTING_EXPOSURE_MAX_EXP", 2.0, -10.0..10.0 step 0.1) {
                    lang {
                        name = "Auto Exposure Max"
                    }
                }
                empty()
                slider("SETTING_EXPOSURE_AVG_LUM_MIX", 0.5, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Average Luminance Weight"
                        comment = "Weight of average luminance AE in the final exposure value."
                    }
                }
                slider("SETTING_EXPOSURE_AVG_LUM_TIME", 3.0, 0.0..10.0 step 0.5) {
                    lang {
                        name = "Average Luminance AE Time"
                    }
                }
                slider("SETTING_EXPOSURE_AVG_LUM_TARGET", 0.2, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Average Luminance Target"
                        comment = "Target average luminance value for average luminance EXPOSURE."
                    }
                }
                empty()
                slider("SETTING_EXPOSURE_TOP_BIN_MIX", 1.0, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Top Bin Weight"
                        comment = "Weight of top bin AE in the final exposure value."
                    }
                }
                slider("SETTING_EXPOSURE_TOP_BIN_TIME", 1.5, 0.0..10.0 step 0.5) {
                    lang {
                        name = "Top Bin AE Time"
                    }
                }
                slider("SETTING_EXPOSURE_TOP_BIN_LUM", 0.3, 0.0..1.0 step 0.01) {
                    lang {
                        name = "Top Bin Luminance"
                        comment = "Luminance threshold for top bin."
                    }
                }
                slider("SETTING_EXPOSURE_TOP_BIN_PERCENT", 5.0, 0.1..10.0 step 0.1) {
                    lang {
                        name = "Top Bin %"
                        comment =
                            "Adjusting exposure to keep the specified percentage of pixels in the top bin of histogram."
                    }
                }
            }
            screen("TONE_MAPPING", 1) {
                lang {
                    name = "Tone Mapping"
                }
                slider("SETTING_TONE_MAPPING_OUTPUT_GAMMA", 2.2, 0.1..4.0 step 0.01) {
                    lang {
                        name = "Output Gamma"
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
                slider("SETTING_TONE_MAPPING_POWER_R", 1.3, 0.1..2.0 step 0.01) {
                    lang {
                        name = "Power Red"
                    }
                }
                slider("SETTING_TONE_MAPPING_POWER_G", 1.3, 0.1..2.0 step 0.01) {
                    lang {
                        name = "Power Green"
                    }
                }
                slider("SETTING_TONE_MAPPING_POWER_B", 1.3, 0.1..2.0 step 0.01) {
                    lang {
                        name = "Power Blue"
                    }
                }
                empty()
                slider("SETTING_TONE_MAPPING_SATURATION", 1.15, 0.0..2.0 step 0.01) {
                    lang {
                        name = "Saturation"
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
        }
        screen("MISC", 2) {
            lang {
                name = "Misc"
            }
            screen("DEBUG", 1) {
                lang {
                    name = "Debug"
                }
                toggle("SETTING_DEBUG_WHITE_WORLD", false) {
                    lang {
                        name = "White World"
                    }
                }
                empty()
                toggle("SETTING_DEBUG_OUTPUT", 0, 0..3) {
                    lang {
                        name = "Debug Output"
                        0 value "Off"
                        1 value "Tone Mapping"
                        2 value "TAA"
                        3 value "Final"
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
                empty()
                toggle("SETTING_DEBUG_TEMP_TEX", 0, 0..7) {
                    lang {
                        name = "Temp Tex"
                        0 value "Off"
                        1 value "temp1"
                        2 value "temp2"
                        3 value "temp3"
                        4 value "temp4"
                        5 value "temp5"
                        6 value "temp6"
                        7 value "temp7"
                    }
                }
                toggle("SETTING_DEBUG_SVGF", 0, 0..4) {
                    lang {
                        name = "SVGF"
                        0 value "Off"
                        1 value "Color"
                        2 value "HLen"
                        3 value "Moment"
                        4 value "Moment^2"
                    }
                }
                toggle("SETTING_DEBUG_packedNZ", 0, 0..2) {
                    lang {
                        name = "Prev NZ"
                        0 value "Off"
                        1 value "Normal"
                        2 value "Z"
                    }
                }
                empty()
                toggle("SETTING_DEBUG_GBUFFER_DATA", 0, 0..10) {
                    lang {
                        name = "GBuffer Data"
                        0 value "Off"
                        1 value "Albedo"
                        2 value "Normal"
                        3 value "Geometry Normal"
                        4 value "Roughness"
                        5 value "F0"
                        6 value "Porosity"
                        7 value "SSS"
                        8 value "Light Map Block"
                        9 value "Light Map Sky"
                        10 value "isHand"
                    }
                }
                toggle("SETTING_DEBUG_NORMAL_MODE", 0, 0..1) {
                    lang {
                        name = "Normal Mode"
                        0 value "World"
                        1 value "View"
                    }
                }
                empty()
                toggle("SETTING_DEBUG_ENV_PROBE", false) {
                    lang {
                        name = "Environment Probe"
                    }
                }
                toggle("SETTING_DEBUG_RTWSM", false) {
                    lang {
                        name = "RTWSM"
                    }
                }
                toggle("SETTING_DEBUG_ATMOSPHERE", false) {
                    lang {
                        name = "Atmosphere"
                    }
                }
                toggle("SETTING_DEBUG_EPIPOLAR", false) {
                    lang {
                        name = "Epipolar"
                    }
                }
            }
            toggle("SETTING_SCREENSHOT_MODE", false) {
                lang {
                    name = "Screenshot Mode"
                }
            }
        }
    }
}