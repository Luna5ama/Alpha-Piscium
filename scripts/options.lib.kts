import java.io.File
import java.math.BigDecimal
import java.util.*
import kotlin.text.appendLine

enum class ColorCode(val code: String) {
    Black("0"),
    DarkBlue("1"),
    DarkGreen("2"),
    DarkAqua("3"),
    DarkRed("4"),
    DarkPurple("5"),
    Gold("6"),
    Gray("7"),
    DarkGray("8"),
    Blue("9"),
    Green("a"),
    Aqua("b"),
    Red("c"),
    LightPurple("d"),
    Yellow("e"),
    White("f"),
}

enum class Profile(val color: ColorCode) {
    Low(ColorCode.Red),
    Medium(ColorCode.Yellow),
    High(ColorCode.Green),
    Ultra(ColorCode.Gold),
    Extreme(ColorCode.LightPurple),
    Insane(ColorCode.DarkPurple),
}

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
        val PROFILE = ScreenItem("<profile>")
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

    fun text(
        name: String,
        key: String,
        value: String = "",
        comment: String = "",
        block: TextOptionBuilder.() -> Unit = {}
    ): ScreenItem {
        val screenItem = ScreenItem(name)
        scope._addOption(TextOptionBuilder(name, key, value, comment).apply(block))
        handleOption(screenItem)
        return screenItem
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


open class OptionBuilder<T>(
    val name: String,
    private val value: T,
    private val const: Boolean,
    private val range: Iterable<T>
) {
    private val profileValues = EnumMap<Profile, T>(Profile::class.java)
    private val langBuilders = mutableMapOf<Locale, LangBuilder<T>>()

    protected open fun newLangBuilder(locale: Locale): LangBuilder<T> {
        return LangBuilder(name, locale)
    }

    infix fun Profile.preset(value: T) {
        profileValues[this] = value
    }

    fun lang(locale: Locale = Locale.US, block: LangBuilder<T>.() -> Unit) {
        langBuilders.getOrPut(locale) { newLangBuilder(locale) }.block()
    }

    open class LangBuilder<T>(val optionName: String, val locale: Locale) {
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

        val valueLabel = mutableMapOf<T, String>()

        infix fun T.value(label: String) {
            valueLabel[this] = label
        }

        open fun build(output: Scope.Output) {
            output.writeLang(locale) {
                if (name.isNotEmpty()) appendLine("option.$optionName=$name")
                if (comment.isNotEmpty()) appendLine("option.$optionName.comment=${comment.replace("\n", "\\n")}")
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

    open fun Scope.Output.writeOptionImpl(block: Appendable.() -> Unit) {
        this.writeOption(block)
    }

    fun build(output: Scope.Output) {
        output.writeOptionImpl {
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
                range.joinTo(this, " ", "//[", "]")
                appendLine()
            }
        }
        langBuilders.forEach { (_, builder) ->
            builder.build(output)
        }
        profileValues.forEach { (profile, value) ->
            if (value is Boolean) {
                val prefix = if (value) "" else "!"
                output.writeProfile(profile, "${prefix}${name}")
            } else {
                output.writeProfile(profile, "${name}=$value")
            }
        }
    }
}

class ProfileBuilder : OptionBuilder<Profile>("", Profile.Low, false, Profile.entries) {
    override fun newLangBuilder(locale: Locale): OptionBuilder.LangBuilder<Profile> {
        println("OK")
        return LangBuilder(locale)
    }

    override fun Scope.Output.writeOptionImpl(block: Appendable.() -> Unit) {
        // No-op
    }

    class LangBuilder(locale: Locale) : OptionBuilder.LangBuilder<Profile>("", locale) {
        override fun build(output: Scope.Output) {
            output.writeLang(locale) {
                if (comment.isNotEmpty()) appendLine("profile.comment=${comment.replace("\n", "\\n")}")
                valueLabel.forEach { (value, label) ->
                    appendLine("profile.$value=$label")
                }
            }
        }
    }
}

class TextOptionBuilder(name: String, key: String, value: String, comment: String) :
    OptionBuilder<Int>(name, 0, false, 0..0) {
    fun valueLang(locale: Locale, key: String, value: String = "", comment: String = "") {
        lang(locale) {
            name = key
            0 value value
            if (comment.isNotEmpty()) {
                this.comment = comment
            }
        }
    }

    override fun Scope.Output.writeOptionImpl(block: Appendable.() -> Unit) {
        this.writeTextOption(block)
    }

    init {
        valueLang(Locale.US, key, value, comment)
    }
}

class Scope : OptionFactory() {
    var screenDepth = 0

    private lateinit var _mainScreen: ScreenBuilder
    private val _screens = mutableSetOf<ScreenBuilder>()
    private val _sliders = mutableSetOf<String>()
    private val _options = mutableSetOf<OptionBuilder<*>>()

    override val scope: Scope
        get() = this

    internal fun _addScreen(screen: ScreenBuilder) {
        screen._name = "SCREEN_${_screens.size}"
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
        _mainScreen = ScreenBuilder(this, columns, 0)
        _mainScreen._name = ""
        screenDepth++
        _mainScreen.apply(block)
        screenDepth--
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
        val maxDepth = _screens.maxOf { it.depth }
        println("Max screen depth: $maxDepth")
        _options.forEach { option ->
            option.build(output)
        }
        return output
    }

    class ScreenBuilder(override val scope: Scope, private val columns: Int, val depth: Int) :
        OptionFactory() {

        lateinit var _name: String

        private val langBuilders = mutableMapOf<Locale, LangBuilder>()
        private val options = mutableSetOf<OptionBuilder<*>>()
        private val ref get() = if (_name.isEmpty()) "" else ".${this@ScreenBuilder._name}"
        private val items = mutableListOf<ScreenItem>()

        var displayName: String? = null
            get() {
                if (_name == "") {
                    return "Main Screen"
                }
                return field
            }

        fun lang(locale: Locale = Locale.US, block: LangBuilder.() -> Unit) {
            check(_name.isNotEmpty()) { "Main screen cannot have lang" }
            val langBuilder = langBuilders.getOrPut(locale) { LangBuilder(ref, locale) }
            langBuilder.block()
            if (locale == Locale.US) {
                displayName = langBuilder.name
            }
        }

        fun build(output: Output) {
            println("${displayName}: depth=$depth, columns=$columns, items=${items.size}")
            langBuilders.forEach { (_, builder) ->
                builder.build(output)
            }
            output.writeShadersProperties {
                appendLine("screen$ref.columns=$columns")
                append("screen$ref=")
                val items = items
                items.joinTo(this, " ")
                appendLine()
            }
        }

        fun item(item: ScreenItem) {
            items.add(item)
        }

        fun screen(columns: Int, block: ScreenBuilder.() -> Unit) {
            val screen = ScreenBuilder(scope, columns, scope.screenDepth)
            scope._addScreen(screen)
            scope.screenDepth++
            screen.apply(block)
            scope.screenDepth--
            val screenItem = ScreenItem("[${screen._name}]")
            items.add(screenItem)
        }

        fun emptyRow() {
            row {
                empty()
            }
        }

        fun row(block: () -> Unit) {
            block()
            val roundedUp = ((items.size + columns - 1) / columns) * columns
            while (items.size < roundedUp) {
                empty()
            }
        }

        fun empty() {
            items.add(ScreenItem.EMPTY)
        }

        fun profile(block: ProfileBuilder.() -> Unit) {
            scope._addOption(ProfileBuilder().apply(block))
            items.add(ScreenItem.PROFILE)
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
        private val _textOptions = StringBuilder()
        private val _lang = mutableMapOf<Locale, StringBuilder>()
        private val _shadersProperties = StringBuilder()
        private val _profiles = EnumMap<Profile, MutableList<String>>(Profile::class.java)

        init {
            val usLang = _lang.getOrPut(Locale.US) { StringBuilder()}
            Profile.entries.forEach {
                val list = mutableListOf<String>()
                if (it.ordinal > 0) {
                    list.add("profile.${Profile.entries[it.ordinal - 1]}")
                }
                _profiles[it] = list

                usLang.appendLine("profile.${it.name}=§${it.color.code}${it.name}")
            }

            _options.appendLine("// $NOTICE")
            _shadersProperties.appendLine("# $NOTICE")
            _shadersProperties.appendLine(baseShadersProperties.readText())
            _shadersProperties.appendLine()
            _shadersProperties.appendLine("# --- Generated Stuff ---")
        }

        fun writeOption(block: Appendable.() -> Unit) {
            _options.block()
        }

        fun writeTextOption(block: Appendable.() -> Unit) {
            _textOptions.block()
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

        fun writeProfile(profile: Profile, value: String) {
            _profiles[profile]!!.add(value)
        }

        fun writeOutput(optionGlslFile: File, textOptionGlslFile: File, shaderRoot: File) {
            writeShadersProperties {
                appendLine()
                _profiles.forEach { (profile, options) ->
                    append("profile.${profile.name}=")
                    options.joinTo(this, " ")
                    appendLine()
                }
            }
            val langDir = File(shaderRoot, "lang")
            langDir.mkdirs()
            optionGlslFile.writeText(_options.toString())
            textOptionGlslFile.writeText(_textOptions.toString())
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

fun options(
    baseShadersProperties: File,
    shaderRootDir: File,
    optionGlslPath: String,
    textOptionGlslPath: String,
    block: Scope.() -> Unit
) {
    val absoluteFile = shaderRootDir.absoluteFile
    Scope().apply(block).build(baseShadersProperties)
        .writeOutput(File(absoluteFile, optionGlslPath), File(absoluteFile, textOptionGlslPath), absoluteFile)
}

fun powerOfTwoRange(range: IntRange): List<Int> {
    return range.map { 1 shl it }
}

fun powerOfTwoAndHalfRange(range: IntRange): List<Int> {
    return range.flatMapIndexed { index, it ->
        if (it <= 1 || index == 0) {
            listOf(1 shl it)
        } else {
            listOf((1 shl (it - 1)) + (1 shl (it - 2)), 1 shl it)
        }
    }
}