import java.io.File
import kotlin.math.*

data class V3(val x: Float, val y: Float, val z: Float) {
    operator fun plus(other: V3) = V3(x + other.x, y + other.y, z + other.z)
    operator fun minus(other: V3) = V3(x - other.x, y - other.y, z - other.z)
    operator fun times(scale: Float) = V3(x * scale, y * scale, z * scale)
    operator fun div(scale: Float) = this * (1.0f / scale)
    fun components() = listOf(x, y, z)
}

val agxMatrix = arrayOf(
    V3(0.842479062253094f, 0.0423282422610123f, 0.0423756549057051f),
    V3(0.0784335999999992f, 0.878468636469772f, 0.0784336f),
    V3(0.0792237451477643f, 0.0791661274605434f, 0.879142973793104f)
)
val agxMatrixInverse = arrayOf(
    V3(1.19687900512017f, -0.0528968517574562f, -0.0529716355144438f),
    V3(-0.0980208811401368f, 1.15190312990417f, -0.0980434501171241f),
    V3(-0.0990297440797205f, -0.0989611768448433f, 1.15107367264116f)
)

fun applyMatrix(columns: Array<V3>, color: V3) =
    columns[0] * color.x + columns[1] * color.y + columns[2] * color.z

fun map(color: V3, transform: (Float) -> Float) =
    V3(transform(color.x), transform(color.y), transform(color.z))

fun log2(value: Float) = (ln(value.toDouble()) / ln(2.0)).toFloat()
fun exp2(value: Float) = 2.0.pow(value.toDouble()).toFloat()

val evRange = 33.0f
val evRangeHalf = evRange * 0.5f

fun encode(color: V3) = map(color) { (max(log2(it), -evRangeHalf) + evRangeHalf) / evRange }
fun decode(color: V3) = map(color) { exp2(it * evRange - evRangeHalf) }
fun forward(color: V3) = encode(applyMatrix(agxMatrix, map(color) { max(it, 0.0f) }))
fun inverse(color: V3) = map(applyMatrix(agxMatrixInverse, decode(map(color) { max(it, 0.0f) }))) { max(it, 0.0f) }

val levels = listOf(0.001f, 0.18f, 1.0f, 100.0f, 1024.0f, 4096.0f, 65504.0f)
val directions = listOf(
    "gray" to V3(1.0f, 1.0f, 1.0f),
    "red" to V3(1.0f, 0.0f, 0.0f),
    "green" to V3(0.0f, 1.0f, 0.0f),
    "blue" to V3(0.0f, 0.0f, 1.0f),
    "orange" to V3(1.0f, 0.12f, 0.01f),
    "cyan" to V3(0.01f, 1.0f, 0.72f),
    "violet" to V3(0.58f, 0.01f, 1.0f)
)
val roundtripAbsoluteTolerance = 5.0e-7f
val roundtripRelativeTolerance = 4.0e-6f

var maxRoundtripError = 0.0f
var maxPureChannelLeakage = 0.0f
for (level in levels) {
    val tolerance = max(roundtripAbsoluteTolerance, level * roundtripRelativeTolerance)
    for ((name, direction) in directions) {
        val expected = direction * level
        val encoded = forward(expected)
        check(encoded.components().all { it.isFinite() && it > 0.0f && it < 1.0f }) {
            "$name at $level falls outside the documented reversible working range: $encoded"
        }
        val actual = inverse(encoded)
        expected.components().zip(actual.components()).forEachIndexed { channel, (reference, result) ->
            check(result.isFinite()) { "$name at $level produced non-finite channel $channel" }
            val error = abs(result - reference)
            maxRoundtripError = max(maxRoundtripError, error)
            check(error <= tolerance) {
                "$name at $level channel $channel roundtrip error $error exceeds $tolerance: $actual"
            }
            if (reference == 0.0f) maxPureChannelLeakage = max(maxPureChannelLeakage, result)
        }
    }
}
check(maxPureChannelLeakage <= levels.last() * roundtripRelativeTolerance)

fun compressHighlight(value: Float, strength: Int): Float {
    if (strength == 0 || value <= 1.0f) return value
    return when (strength) {
        1 -> 1.0f + ln(value)
        2 -> 2.0f - exp(-(value - 1.0f))
        3 -> 2.0f - 1.0f / value
        4 -> 1.0f
        else -> error("invalid strength")
    }
}

listOf(0.0f, 0.18f, 1.0f, 16.0f, 65504.0f).forEach { check(compressHighlight(it, 0) == it) }
for (strength in 1..4) {
    listOf(0.0f, 0.18f, 1.0f).forEach { check(compressHighlight(it, strength) == it) }
    val input = 16.0f
    val output = compressHighlight(input, strength)
    check(output.isFinite() && output in 1.0f..<input)
}
val compressed = (1..4).map { compressHighlight(16.0f, it) }
check(compressed.zipWithNext().all { (weaker, stronger) -> weaker > stronger })

val saturated = V3(16.0f, 4.0f, 1.0f)
fun compressRgb(color: V3, strength: Int) = map(color) { compressHighlight(it, strength) }
fun luma(color: V3) = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z
fun compressLuma(color: V3, strength: Int): V3 {
    val value = luma(color)
    if (value <= 1.0f) return color
    return color * (compressHighlight(value, strength) / value)
}

val rgbCompressed = compressRgb(saturated, 3)
check(abs(rgbCompressed.x / rgbCompressed.y - saturated.x / saturated.y) > 0.1f)
val lumaCompressed = compressLuma(saturated, 3)
check(abs(lumaCompressed.x / lumaCompressed.y - saturated.x / saturated.y) <= 1.0e-6f)
check(abs(lumaCompressed.y / lumaCompressed.z - saturated.y / saturated.z) <= 1.0e-6f)
check(compressLuma(V3(0.0f, 0.0f, 0.0f), 3) == V3(0.0f, 0.0f, 0.0f))

fun saturate(value: Float) = value.coerceIn(0.0f, 1.0f)
fun rcasLuma(color: V3) = color.z * 0.5f + color.x * 0.5f + color.y
fun rcasFilter(center: V3, ring: List<V3>, sharpness: Float): V3 {
    if (sharpness == 0.0f) return center
    check(ring.size == 4)

    val ringLuma = ring.map(::rcasLuma)
    val centerLuma = rcasLuma(center)
    val lumaRange = (ringLuma + centerLuma).let { it.max() - it.min() }
    var noise = if (lumaRange == 0.0f) 0.0f else
        saturate(abs(ringLuma.average().toFloat() - centerLuma) / lumaRange)
    noise = 1.0f - 0.5f * noise

    val minimum = V3(
        ring.minOf { it.x },
        ring.minOf { it.y },
        ring.minOf { it.z }
    )
    val maximum = V3(
        ring.maxOf { it.x },
        ring.maxOf { it.y },
        ring.maxOf { it.z }
    )
    val minimumRingLuma = ringLuma.min()
    val lowerLimiter = if (minimumRingLuma == 0.0f) 1.0f else saturate(centerLuma / minimumRingLuma)
    fun channelLobe(mn: Float, mx: Float): Float {
        val hitMin = if (mx == 0.0f) 0.0f else mn / (4.0f * mx) * lowerLimiter
        val hitMax = (1.0f - mx) / (4.0f * mn - 4.0f)
        return max(-hitMin, hitMax)
    }
    val limiter = max(
        channelLobe(minimum.x, maximum.x),
        max(channelLobe(minimum.y, maximum.y), channelLobe(minimum.z, maximum.z))
    )
    val sharpnessConfig = exp2(-(2.0f - 2.0f * sharpness))
    val lobe = max(-0.1875f, min(limiter, 0.0f)) * sharpnessConfig * noise
    return (ring.reduce(V3::plus) * lobe + center) / (4.0f * lobe + 1.0f)
}

var maxZeroSharpnessError = 0.0f
for (level in levels) {
    val tolerance = max(roundtripAbsoluteTolerance, level * roundtripRelativeTolerance)
    for ((name, direction) in directions) {
        val input = direction * level
        val encoded = forward(input)
        val result = inverse(rcasFilter(encoded, List(4) { encoded }, 0.0f))
        input.components().zip(result.components()).forEachIndexed { channel, (reference, actual) ->
            val error = abs(actual - reference)
            maxZeroSharpnessError = max(maxZeroSharpnessError, error)
            check(error <= tolerance) {
                "$name at $level zero-sharpness channel $channel error $error exceeds $tolerance"
            }
        }
    }
}

var maxRcasPureChannelLeakage = 0.0f
for (level in levels.filter { it >= 1.0f }) {
    val dimLevel = max(0.001f, level * 0.25f)
    for ((name, direction) in directions.filter { (_, color) -> color.components().count { it == 0.0f } == 2 }) {
        val center = forward(direction * level)
        val ring = List(4) { forward(direction * dimLevel) }
        val result = inverse(rcasFilter(center, ring, 0.5f))
        val tolerance = max(roundtripAbsoluteTolerance, result.components().max() * roundtripRelativeTolerance)
        direction.components().zip(result.components()).forEachIndexed { channel, (source, actual) ->
            check(actual.isFinite()) { "$name at $level RCAS produced non-finite channel $channel" }
            if (source == 0.0f) {
                maxRcasPureChannelLeakage = max(maxRcasPureChannelLeakage, actual)
                check(actual <= tolerance) {
                    "$name at $level RCAS channel $channel leakage $actual exceeds $tolerance: $result"
                }
            }
        }
    }
}

val agxSource = File("../shaders/util/AgxInvertible.glsl").readText()
val bloomSource = File("../shaders/techniques/Bloom.comp.glsl").readText()
val programsSource = File("programs.main.kts").readText()
val prepareSource = File("../shaders/pass/composite/TAAPrepare.comp.glsl").readText()
val resolveSource = File("../shaders/pass/composite/TAAResolve.comp.glsl").readText()
val rcasSource = File("../shaders/pass/composite/RCAS.comp.glsl").readText()
val rcasIntegrationSource = File("../shaders/techniques/ffx/fsr1/RCAS.glsl").readText()
check("const float EV_RANGE = 33.0;" in agxSource)
check("SETTING_BLOOM_HIGHLIGHT_COMPRESSION" !in agxSource)
check("#if BLOOM_PASS == 1 && SETTING_BLOOM_HIGHLIGHT_COMPRESSION != 0" in bloomSource)
check("inputValue.rgb = bloom_compressHighlights" in bloomSource)
check(Regex("bloom_compressHighlights\\(").findAll(bloomSource).count() == 2)
check("#if SETTING_AA_MODE != 2" in prepareSource && "agxInvertible_forward" in prepareSource)
check("agxInvertible_inverse" in resolveSource)
check("#if SETTING_AA_MODE == 2" in rcasSource && "agxInvertible_forward" in rcasSource && "agxInvertible_inverse" in rcasSource)
check("#define SETTING_RCAS_SHARPNESS SETTING_AA_SHARPNESS" in rcasSource)
check("SETTING_TAA_CAS_SHARPNESS" !in rcasIntegrationSource && "SETTING_FSR3_SHARPNESS" !in rcasIntegrationSource)
check("if (SETTING_RCAS_SHARPNESS == 0.0) return rcas_loadInput(outputTexelPos, true);" in rcasIntegrationSource)
check("float sharpness = SETTING_RCAS_SHARPNESS;" in rcasIntegrationSource)
check("cond(\"SETTING_AA_MODE != 0\")" in programsSource)

println("AgX matrix/log roundtrip checks passed: max error=$maxRoundtripError, max pure-channel leakage=$maxPureChannelLeakage")
println("Bloom highlight compression checks passed")
println("RCAS checks passed: max zero-sharpness error=$maxZeroSharpnessError, max pure-channel leakage=$maxRcasPureChannelLeakage")
