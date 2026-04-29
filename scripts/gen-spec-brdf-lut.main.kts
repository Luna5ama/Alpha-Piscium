import java.io.File
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.StandardOpenOption
import java.util.stream.IntStream
import javax.imageio.ImageIO
import kotlin.io.path.Path
import kotlin.math.*

fun fresnelAdobeScalar(f0: Double, f82: Double, cosTheta: Double): Double {
    val oneMinusF0 = 1.0 - f0
    val b = Math.fma(oneMinusF0, 0.462664878484, f0) * Math.fma(f82, -17.6513843536, 17.6513843536)
    return (Math.fma(
        Math.fma(Math.fma(cosTheta, cosTheta, -cosTheta), b, oneMinusF0),
        (1.0 - cosTheta).pow(5.0),
        f0
    )).coerceIn(0.0, 1.0)
}

// --- 蒙特卡洛积分辅助函数 ---

fun radicalInverseVdc(bits: Int): Double {
    var b = bits.toLong()
    b = (b shl 16) or (b shr 16)
    b = ((b and 0x55555555) shl 1) or ((b and 0xAAAAAAAA) shr 1)
    b = ((b and 0x33333333) shl 2) or ((b and 0xCCCCCCCC) shr 2)
    b = ((b and 0x0F0F0F0F) shl 4) or ((b and 0xF0F0F0F0) shr 4)
    b = ((b and 0x00FF00FF) shl 8) or ((b and 0xFF00FF00) shr 8)
    return b.toDouble() * 2.3283064365386963e-10
}

fun hammersley(i: Int, n: Int): Pair<Double, Double> {
    return Pair(i.toDouble() / n.toDouble(), radicalInverseVdc(i))
}

fun importanceSampleGGX(xi: Pair<Double, Double>, roughness: Double): DoubleArray {
    val a = roughness * roughness
    val phi = 2.0 * PI * xi.first
    val cosTheta = sqrt((1.0 - xi.second) / (1.0 + (a * a - 1.0) * xi.second))
    val sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta))

    // 返回切线空间的 H 向量
    return doubleArrayOf(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta)
}

fun geometrySchlickGGX(nDotV: Double, roughness: Double): Double {
    val k = (roughness * roughness) / 2.0
    return nDotV / (nDotV * (1.0 - k) + k)
}

fun geometrySmith(nDotV: Double, nDotL: Double, roughness: Double): Double {
    return geometrySchlickGGX(nDotV, roughness) * geometrySchlickGGX(nDotL, roughness)
}

val lutSize = 256
val numSamples = 4096
val tempBuffer = Array(lutSize) { Array(lutSize) { DoubleArray(3) } }

IntStream.range(0, lutSize * lutSize).parallel().forEach { index ->
    val x = index % lutSize
    val y = index / lutSize
    val nDotV = ((x.toDouble() + 0.5) / lutSize)
    val roughness = 1.0 - ((y.toDouble() + 0.5) / lutSize)

    val v = doubleArrayOf(sqrt(1.0 - nDotV * nDotV), 0.0, nDotV)

    var weightF0 = 0.0
    var weightF82 = 0.0
    var weightBias = 0.0

    for (i in 0 until numSamples) {
        val xi = hammersley(i, numSamples)
        val h = importanceSampleGGX(xi, roughness)

        // 计算 L = reflect(-V, H)
        val vDotH = v[0] * h[0] + v[1] * h[1] + v[2] * h[2]
        val l = doubleArrayOf(
            2.0 * vDotH * h[0] - v[0],
            2.0 * vDotH * h[1] - v[1],
            2.0 * vDotH * h[2] - v[2]
        )

        val nDotL = l[2].coerceIn(0.0, 1.0)
        val nDotH = h[2].coerceIn(0.0, 1.0)
        val vDotHClamped = vDotH.coerceIn(0.0, 1.0)

        if (nDotL > 0.0f) {
            val g = geometrySmith(nDotV, nDotL, roughness)
            val gVis = (g * vDotHClamped) / (nDotH * nDotV)

            val fBias = fresnelAdobeScalar(0.0, 0.0, vDotHClamped)
            val fF0 = fresnelAdobeScalar(1.0, 0.0, vDotHClamped) - fBias
            val fF82 = fresnelAdobeScalar(0.0, 1.0, vDotHClamped) - fBias

            weightBias += fBias * gVis
            weightF0 += fF0 * gVis
            weightF82 += fF82 * gVis
        }
    }

    val r = (weightF0 / numSamples).coerceIn(0.0, 1.0)
    val g = (weightF82 / numSamples).coerceIn(0.0, 1.0)
    val b = (weightBias / numSamples).coerceIn(0.0, 1.0)

    val rgb = tempBuffer[y][x]
    rgb[0] = r
    rgb[1] = g
    rgb[2] = b
}

val previewPng = false

if (previewPng) {
    val image = java.awt.image.BufferedImage(lutSize, lutSize, java.awt.image.BufferedImage.TYPE_INT_RGB)
    for (y in 0 until lutSize) {
        for (x in 0 until lutSize) {
            val rgb = tempBuffer[y][x]
            val r = (rgb[0] * 255.0).toInt().coerceIn(0, 255)
            val g = (rgb[1] * 255.0).toInt().coerceIn(0, 255)
            val b = (rgb[2] * 255.0).toInt().coerceIn(0, 255)
            val color = (r shl 16) or (g shl 8) or b
            image.setRGB(x, y, color)
        }
    }
    ImageIO.write(image, "png", File("specular_lut_${lutSize}_$numSamples.png"))
} else {
    val outputPath = Path("../shaders/textures/specular_brdf_lut.bin")
    val outputSize = lutSize * lutSize * 2 * 3
    FileChannel.open(
        outputPath,
        StandardOpenOption.CREATE,
        StandardOpenOption.READ,
        StandardOpenOption.WRITE,
        StandardOpenOption.TRUNCATE_EXISTING
    ).use { outputChannel ->
        val mapped = outputChannel.map(FileChannel.MapMode.READ_WRITE, 0L, outputSize.toLong())
            .order(ByteOrder.nativeOrder())
        for (y in 0 until lutSize) {
            for (x in 0 until lutSize) {
                for (c in tempBuffer[y][x]) {
                    mapped.putShort((c * 65535.0).toInt().coerceIn(0, 65535).toShort())
                }
            }
        }
    }
}
