import java.io.File
import kotlin.math.*

data class V3(val x: Double, val y: Double, val z: Double) {
    operator fun plus(other: V3) = V3(x + other.x, y + other.y, z + other.z)
    operator fun minus(other: V3) = V3(x - other.x, y - other.y, z - other.z)
    operator fun times(scale: Double) = V3(x * scale, y * scale, z * scale)
    operator fun div(scale: Double) = this * (1.0 / scale)
}

data class V2(val x: Double, val y: Double)

val primaries = arrayOf(
    V3(0.4124, 0.2126, 0.0193),
    V3(0.3576, 0.7152, 0.1192),
    V3(0.1805, 0.0722, 0.9505)
)
val white = primaries.reduce(V3::plus)

fun xyzToXy(v: V3): V2 {
    val sum = v.x + v.y + v.z
    return V2(v.x / sum, v.y / sum)
}

fun xyToXyz(v: V2, y: Double) = V3(v.x * y / v.y, y, (1.0 - v.x - v.y) * y / v.y)

fun rotatePrimary(primary: V3, hue: Double, saturation: Double): V3 {
    val p = xyzToXy(primary)
    val w = xyzToXy(white)
    val angle = Math.toRadians(hue * 0.25)
    val scale = 1.0 + saturation * 0.01
    val x = (p.x - w.x) * scale
    val y = (p.y - w.y) * scale
    return xyToXyz(V2(w.x + x * cos(angle) - y * sin(angle), w.y + x * sin(angle) + y * cos(angle)), primary.y)
}

fun calibration(hue: List<Double>, saturation: List<Double>): Array<V3> {
    val columns = Array(3) { rotatePrimary(primaries[it], hue[it], saturation[it]) }
    val correction = (white - columns.reduce(V3::plus)) / 3.0
    return Array(3) { columns[it] + correction }
}

fun apply(columns: Array<V3>, color: V3) =
    columns[0] * color.x + columns[1] * color.y + columns[2] * color.z

fun assertNear(actual: V3, expected: V3, epsilon: Double = 1.0e-10) {
    check(abs(actual.x - expected.x) <= epsilon)
    check(abs(actual.y - expected.y) <= epsilon)
    check(abs(actual.z - expected.z) <= epsilon)
}

fun rgbToHsl(c: V3): V3 {
    val maxC = max(c.x, max(c.y, c.z))
    val minC = min(c.x, min(c.y, c.z))
    val delta = maxC - minC
    val l = (maxC + minC) * 0.5
    val s = if (delta < 1.0e-6) 0.0 else delta / (1.0 - abs(2.0 * l - 1.0))
    var h = when {
        delta < 1.0e-6 -> 0.0
        maxC == c.x -> ((c.y - c.z) / delta).mod(6.0)
        maxC == c.y -> (c.z - c.x) / delta + 2.0
        else -> (c.x - c.y) / delta + 4.0
    } * 60.0
    if (h < 0.0) h += 360.0
    return V3(h, s, l)
}

fun hueToRgb(p: Double, q: Double, input: Double): Double {
    var t = input
    if (t < 0.0) t += 1.0
    if (t > 1.0) t -= 1.0
    return when {
        t < 1.0 / 6.0 -> p + (q - p) * 6.0 * t
        t < 0.5 -> q
        t < 2.0 / 3.0 -> p + (q - p) * (2.0 / 3.0 - t) * 6.0
        else -> p
    }
}

fun hslToRgb(hsl: V3): V3 {
    if (hsl.y < 1.0e-6) return V3(hsl.z, hsl.z, hsl.z)
    val h = hsl.x / 360.0
    val q = if (hsl.z < 0.5) hsl.z * (1.0 + hsl.y) else hsl.z + hsl.y - hsl.z * hsl.y
    val p = 2.0 * hsl.z - q
    return V3(hueToRgb(p, q, h + 1.0 / 3.0), hueToRgb(p, q, h), hueToRgb(p, q, h - 1.0 / 3.0))
}

listOf(
    listOf(0.0, 0.0, 0.0) to listOf(0.0, 0.0, 0.0),
    listOf(100.0, -100.0, 100.0) to listOf(100.0, 100.0, 100.0),
    listOf(-100.0, 75.0, 20.0) to listOf(-100.0, 50.0, 100.0)
).forEach { (hue, saturation) ->
    assertNear(apply(calibration(hue, saturation), V3(1.0, 1.0, 1.0)), white)
}

val sample = V3(0.2, 0.5, 0.9)
assertNear(apply(calibration(List(3) { 0.0 }, List(3) { 0.0 }), sample), apply(primaries, sample))
val grayscale = primaries[0].y * sample.x + primaries[1].y * sample.y + primaries[2].y * sample.z
assertNear(apply(calibration(List(3) { 0.0 }, List(3) { -100.0 }), sample), white * grayscale)

listOf(V3(1.0, 0.0, 0.0), V3(0.2, 0.5, 0.9), V3(0.8, 0.3, 0.6)).forEach {
    assertNear(hslToRgb(rgbToHsl(it)), it)
}
val gray = V3(0.4, 0.4, 0.4)
check(rgbToHsl(gray).y == 0.0)

val drt = File("../shaders/techniques/displaytransform/DRT.glsl").readText()
val mixer = File("../shaders/techniques/displaytransform/HSLMixer.glsl").readText()
check("whiteCorrection" in drt && "inverse(primariesToXYZ)" !in drt)
check("if (hsl.y < 1.0e-6) return color;" in mixer)
check("+ 1.0e-6)" !in mixer)

println("Color grading math checks passed")
