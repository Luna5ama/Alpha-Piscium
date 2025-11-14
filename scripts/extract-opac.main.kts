import java.awt.image.BufferedImage
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import javax.imageio.ImageIO
import kotlin.io.path.*
import kotlin.math.*

val dataDir = Path("../data")
val opacDataDir = dataDir.resolve("opac_raw")
val outputDir = dataDir.resolve("opac")
outputDir.createDirectories()
val spacesRegex = """\s+""".toRegex()
val scientificRegex = """E([+-\\d]+)""".toRegex()

/**
 * Performs Catmull-Rom spline interpolation.
 *
 * @param Xs Array of x-coordinates of the data points
 * @param Ys Array of y-coordinates of the data points
 * @param xPos The x value to interpolate at
 * @return The interpolated y value
 */
fun catmullRomInterpolate(Xs: DoubleArray, Ys: DoubleArray, xPos: Double): Double {
    // Validate input
    if (Xs.size < 4 || Xs.size != Ys.size) {
        throw IllegalArgumentException("Arrays must contain at least 4 points and be of equal length")
    }

    // Handle out of bounds cases
    if (xPos <= Xs[0]) return Ys[0]
    if (xPos >= Xs[Xs.size - 1]) return Ys[Ys.size - 1]

    // Find the segment containing xPos
    var i = 1
    while (i < Xs.size - 2 && Xs[i + 1] < xPos) {
        i++
    }

    // Get the four control points needed
    val i0 = maxOf(0, i - 1)
    val i1 = i
    val i2 = minOf(i + 1, Xs.size - 1)
    val i3 = minOf(i + 2, Xs.size - 1)

    val x0 = Xs[i0]
    val x1 = Xs[i1]
    val x2 = Xs[i2]
    val x3 = Xs[i3]

    val y0 = Ys[i0]
    val y1 = Ys[i1]
    val y2 = Ys[i2]
    val y3 = Ys[i3]

    // Calculate the t parameter in [0,1]
    val t = if (x2 == x1) 0.0 else (xPos - x1) / (x2 - x1)

    // Catmull-Rom interpolation
    val t2 = t * t
    val t3 = t2 * t

    val h1 = 2.0 * t3 - 3.0 * t2 + 1.0
    val h2 = -2.0 * t3 + 3.0 * t2
    val h3 = t3 - 2.0 * t2 + t
    val h4 = t3 - t2

    val m1 = 0.5 * (y2 - y0) / (x2 - x0)
    val m2 = 0.5 * (y3 - y1) / (x3 - x1)

    return h1 * y1 + h2 * y2 + h3 * m1 * (x2 - x1) + h4 * m2 * (x2 - x1)
}

fun transpose(matrix: List<DoubleArray>): List<DoubleArray> {
    if (matrix.isEmpty()) return emptyList()
    val numRows = matrix.size
    val numCols = matrix[0].size
    return List(numCols) { colIndex ->
        DoubleArray(numRows) { rowIndex -> matrix[rowIndex][colIndex] }
    }
}

fun readCSVRows(filePath: Path): List<List<String>> {
    return filePath.useLines { lines ->
        lines.map { line ->
            line.split(",").map { it.trim() }
        }.toList()
    }
}

val cieCMF = transpose(
    readCSVRows(dataDir.resolve("CIE_1931_CMF_2deg.csv"))
        .drop(1) // Skip header
        .map { it.map(String::toDouble).toDoubleArray() }
)

val cn = (cieCMF[2] zip cieCMF[4]).sumOf {
    it.first * it.second
}

val XYZ2AP0R = doubleArrayOf(1.0498110175, 0.0, -9.74845e-05,)
val XYZ2AP0G = doubleArrayOf(-0.4959030231, 1.3733130458, 0.0982400361,)
val XYZ2AP0B = doubleArrayOf(0.0, 0.0, 0.9912520182)
val XYZ2AP0 = listOf(XYZ2AP0R, XYZ2AP0G, XYZ2AP0B)

fun doColorMatching(
    Xs: DoubleArray,
    Ys: DoubleArray,
): DoubleArray {
    val Cxyz = (0..<3).map { xyzIndex ->
        cieCMF[0].indices.sumOf { waveLengthIndex ->
            val x = cieCMF[0][waveLengthIndex]
            val c = cieCMF[xyzIndex + 1][waveLengthIndex]
            val i = cieCMF[4][waveLengthIndex]
            val f = catmullRomInterpolate(Xs, Ys, x / 1000.0)
            f * c * i
        } / cn
    }.toDoubleArray()

    val Csrgb = XYZ2AP0.map { row ->
        (Cxyz zip row).sumOf { it.first * it.second }
    }.toDoubleArray()

    return Csrgb
}

val angAndPhaseCols = mutableListOf<Pair<DoubleArray, List<DoubleArray>>>()

opacDataDir.useDirectoryEntries { entries ->
    entries
        .filter { it.isRegularFile() && it.extension == "txt" }
        .forEach { filepath ->
            println(filepath.nameWithoutExtension)
            val textLines = filepath.readLines()
            val opticalParameterIndex = textLines.indexOf("# optical parameters:")

            val name = filepath.nameWithoutExtension

            val opticalParamRows = textLines.asSequence()
                .drop(opticalParameterIndex + 6)
                .takeWhile { it.length > 1 }
                .map { it.substring(3) }
                .map {
                    it.splitToSequence(spacesRegex)
                        .map(String::toDouble)
                        .toList()
                        .toDoubleArray()
                }
                .toList()

            val cols = transpose(opticalParamRows)
            val X = cols[0]
            val sctr = doColorMatching(X, cols[2])
            val exti = doColorMatching(X, (cols[2] zip cols[3]).map { it.first + it.second }.toDoubleArray())
            val asym = doColorMatching(X, cols[5])
            println("scattering: ${sctr.contentToString()}")
            println("extinction: ${exti.contentToString()}")
            println("asymmetry: ${asym.contentToString()}")

            val vpfIndex = textLines.indexOf("# volume phase function [1/km]:")
            val Xs = textLines[vpfIndex + 5]
                .substring(13)
                .splitToSequence(spacesRegex)
                .map(String::toDouble)
                .toList()
                .toDoubleArray()

            fun integratePhaseFunction(angles: DoubleArray, phaseValues: DoubleArray): Double {
                var sum = 0.0

                // Integrate using the trapezoidal rule with sin(θ) weighting
                for (i in 0 until angles.size - 1) {
                    val theta1 = Math.toRadians(angles[i])
                    val theta2 = Math.toRadians(angles[i + 1])

                    val sinTheta1 = sin(theta1)
                    val sinTheta2 = sin(theta2)

                    val y1 = phaseValues[i] * sinTheta1
                    val y2 = phaseValues[i + 1] * sinTheta2

                    // Trapezoidal area
                    sum += (theta2 - theta1) * (y1 + y2) / 2.0
                }

                // Multiply by 2π for the azimuthal integration
                return sum * 2.0 * Math.PI
            }

            val rows = textLines.asSequence()
                .drop(vpfIndex + 7)
                .map {
                    val list = it.substring(2)
                        .splitToSequence(spacesRegex)
                        .map(String::toDouble)
                        .toList()
                    list.first() to (list.drop(1).toDoubleArray() zip cols[2]).map { (phase, sca) -> phase / sca }
                        .toDoubleArray()
                }
                .toList()

            val angleCol = rows.map { it.first }.toDoubleArray()

            fun luma(rgb: DoubleArray): Double {
                return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            }

            val phaseRGBRows = rows.asSequence()
                .map { (angle, Ys) -> doColorMatching(Xs, Ys) }
                .toList()

//            val maxV = phaseRGBRows
//                .drop(angleCol.indexOfFirst { it == 1.0 })
//                .maxOf { luma(it) } * 256.0
//
//            phaseRGBRows.forEach { rgbValue ->
//                val lumaIt = luma(rgbValue)
//                var mul = min(1.0, maxV / lumaIt)
//                repeat(rgbValue.size) {
//                    rgbValue[it] *= mul
//                }
//            }

            val phaseRGBCols = transpose(phaseRGBRows)

            val phaseSums = (0..<3).map { rgbIndex ->
                integratePhaseFunction(angleCol, phaseRGBCols[rgbIndex])
            }

            repeat(3) { rgbIndex ->
                phaseRGBCols[rgbIndex].indices.forEach { angleIndex ->
                    phaseRGBCols[rgbIndex][angleIndex] /= phaseSums[rgbIndex]
                }
            }

            outputDir.resolve("${name}_phase.csv").bufferedWriter().use { writer ->
                (angleCol zip phaseRGBRows).forEach { (angle, rgb) ->
                    writer.append(angle.toString())
                    repeat(3) {
                        writer.append(',')
                        writer.append(rgb[it].toString())
                    }
                    writer.appendLine()
                }
            }

            angAndPhaseCols.add(angleCol to transpose(phaseRGBRows))

            println()
        }
}

/**
 * Converts SRGB colors to LogLuv format.
 * @param vRGB Input RGB color as FloatArray of size 3
 * @return LogLuv encoded color as FloatArray of size 4
 */
fun sRGBToLogLuv(vRGB: DoubleArray): DoubleArray {
    // Check if all RGB values are less than or equal to 0
    if (vRGB[0] <= 0.0 && vRGB[1] <= 0.0 && vRGB[2] <= 0.0) {
        return doubleArrayOf(0.0, 0.0, 0.0, 0.0)
    }

    val M = arrayOf(
        doubleArrayOf(0.2209, 0.1138, 0.0102),
        doubleArrayOf(0.3390, 0.6780, 0.1130),
        doubleArrayOf(0.4184, 0.7319, 0.2969)
    )

    // Calculate Xp_Y_XYZp = M * vRGB
    val Xp_Y_XYZp = DoubleArray(3)
    for (i in 0 until 3) {
        Xp_Y_XYZp[i] = M[i][0] * vRGB[0] + M[i][1] * vRGB[1] + M[i][2] * vRGB[2]
        Xp_Y_XYZp[i] = maxOf(Xp_Y_XYZp[i], 1e-6)
    }

    val vResult = DoubleArray(4)
    vResult[0] = Xp_Y_XYZp[0] / Xp_Y_XYZp[2]
    vResult[1] = Xp_Y_XYZp[1] / Xp_Y_XYZp[2]

    val Le = 2f * log2(Xp_Y_XYZp[1]) + 127.0
    vResult[3] = Le - floor(Le)
    vResult[2] = (Le - (floor(vResult[3] * 255.0)) / 255.0) / 255.0

    return vResult
}

/**
 * Converts LogLuv32 format to sRGB colors.
 *
 * @param vLogLuv Input LogLuv color as DoubleArray of size 4
 * @return sRGB color as DoubleArray of size 3
 */
fun logLuv32TosRGB(vLogLuv: DoubleArray): DoubleArray {
    // Check if all LogLuv values are less than or equal to 0
    if (vLogLuv[0] <= 0.0 && vLogLuv[1] <= 0.0 && vLogLuv[2] <= 0.0 && vLogLuv[3] <= 0.0) {
        return doubleArrayOf(0.0, 0.0, 0.0)
    }

    // [ERI07] Inverse M matrix, for decoding
    val inverseM = arrayOf(
        doubleArrayOf(6.0014, -1.3320, 0.3008),
        doubleArrayOf(-2.7008, 3.1029, -1.0882),
        doubleArrayOf(-1.7996, -5.7721, 5.6268)
    )

    val Le = vLogLuv[2] * 255 + vLogLuv[3]
    val Xp_Y_XYZp = DoubleArray(3)

    Xp_Y_XYZp[1] = 2.0.pow((Le - 127) / 2)
    Xp_Y_XYZp[2] = Xp_Y_XYZp[1] / vLogLuv[1]
    Xp_Y_XYZp[0] = vLogLuv[0] * Xp_Y_XYZp[2]

    val vRGB = DoubleArray(3)
    for (i in 0 until 3) {
        vRGB[i] = inverseM[i][0] * Xp_Y_XYZp[0] +
            inverseM[i][1] * Xp_Y_XYZp[1] +
            inverseM[i][2] * Xp_Y_XYZp[2]
        vRGB[i] = maxOf(vRGB[i], 0.0)
    }

    return vRGB
}

/**
 * Performs cubic B-spline interpolation.
 *
 * @param Xs Array of x-coordinates of the data points
 * @param Ys Array of y-coordinates of the data points
 * @param xPos The x value to interpolate at
 * @return The interpolated y value
 */
fun cubicBSplineInterpolate(Xs: DoubleArray, Ys: DoubleArray, xPos: Double): Double {
    // Validate input
    if (Xs.size < 4 || Xs.size != Ys.size) {
        throw IllegalArgumentException("Arrays must contain at least 4 points and be of equal length")
    }

    // Handle out of bounds cases
    if (xPos <= Xs[0]) return Ys[0]
    if (xPos >= Xs[Xs.size - 1]) return Ys[Ys.size - 1]

    // Find the segment containing xPos
    var i = 0
    while (i < Xs.size - 1 && Xs[i + 1] < xPos) {
        i++
    }

    // Get the four control points needed
    val i0 = maxOf(0, i - 1)
    val i1 = i
    val i2 = minOf(i + 1, Xs.size - 1)
    val i3 = minOf(i + 2, Xs.size - 1)

    val x0 = Xs[i0]
    val x1 = Xs[i1]
    val x2 = Xs[i2]
    val x3 = Xs[i3]

    val y0 = Ys[i0]
    val y1 = Ys[i1]
    val y2 = Ys[i2]
    val y3 = Ys[i3]

    // Calculate the t parameter in [0,1]
    val t = if (x2 == x1) 0.0 else (xPos - x1) / (x2 - x1)

    // B-spline basis functions
    val t2 = t * t
    val t3 = t2 * t

    val b0 = (1 - t) * (1 - t) * (1 - t) / 6.0
    val b1 = (3 * t3 - 6 * t2 + 4) / 6.0
    val b2 = (-3 * t3 + 3 * t2 + 3 * t + 1) / 6.0
    val b3 = t3 / 6.0

    // Calculate the interpolated value
    return y0 * b0 + y1 * b1 + y2 * b2 + y3 * b3
}

fun inversePolynomial(
    y: Double,
    initialGuess: Double = 0.0,
    maxIterations: Int = 32,
    epsilon: Double = 1e-10
): Double {
    // Constants from the formula
    val a0 = 0.672617934627
    val a1 = -0.0713555761181
    val a2 = 0.0299320735609
    val b = 0.264767018876

    var x = initialGuess

    // Newton-Raphson method
    repeat(maxIterations) {
        // Calculate polynomial part
        val poly = a0 + a1 * x + a2 * x * x

        // Calculate f(x) = (poly) * x^b
        val fx = poly * x.pow(b)

        // Calculate f'(x) = b * poly * x^(b-1) + (a1 + 2*a2*x) * x^b
        val dPoly = a1 + 2 * a2 * x
        val dfx = b * poly * x.pow(b - 1) + dPoly * x.pow(b)

        // Newton's update
        val xNew = x - (fx - y) / max(dfx, 1e-10)

        // Check for convergence
        if (abs(xNew - x) < epsilon) {
            return xNew
        }

        x = xNew
    }

    return x // Return best estimate after maxIterations
}

var maxError = Double.MIN_VALUE

fun Double.toHalf() = java.lang.Float.floatToFloat16(this.toFloat())
fun Short.halfToDouble() = java.lang.Float.float16ToFloat(this).toDouble()

fun RGBToRGBM(rgb: DoubleArray, m: Double): ShortArray {
    val maxV = rgb.max()
    val d = maxV / m
    return shortArrayOf(
        (rgb[0] / d).toHalf(),
        (rgb[1] / d).toHalf(),
        (rgb[2] / d).toHalf(),
        (d).toHalf()
    )
}

fun RGBMToRGB(rgbm: ShortArray): DoubleArray {
    val rgbmDouble = doubleArrayOf(
        rgbm[0].halfToDouble(),
        rgbm[1].halfToDouble(),
        rgbm[2].halfToDouble(),
        rgbm[3].halfToDouble()
    )

    val d = rgbmDouble[3]
    return doubleArrayOf(
        rgbmDouble[0] * d,
        rgbmDouble[1] * d,
        rgbmDouble[2] * d
    )
}



val phaseLUTWidth = 256
val phaseLUTHeight = angAndPhaseCols.size
val outputSize = phaseLUTWidth.toLong() * phaseLUTHeight.toLong() * 2L * 4L // RGBA16F
val outputImagePath = Path("../shaders/textures/opac_cloud_phases.bin")
println("$phaseLUTWidth x $phaseLUTHeight -> ${outputSize / (1024 * 1024)} MB")

FileChannel.open(
    outputImagePath,
    StandardOpenOption.CREATE,
    StandardOpenOption.READ,
    StandardOpenOption.WRITE,
    StandardOpenOption.TRUNCATE_EXISTING
).use { outputChannel ->
    val mapped = outputChannel.map(FileChannel.MapMode.READ_WRITE, 0L, outputSize)
        .order(ByteOrder.nativeOrder())
    angAndPhaseCols.forEachIndexed { index, anglesAndPhase ->
        val (angles, phaseRGB) = anglesAndPhase
        val maxVal = phaseRGB.maxOf { it.max() }
        val mDiv = sqrt(maxVal)
        repeat(phaseLUTWidth) { px ->
            val px01 = px.toDouble() / (phaseLUTWidth - 1)
            var theta = when(px01) {
                0.0 -> 0.0
                1.0 -> PI
                else -> inversePolynomial(px01, 1e-10)
            }
            val thetaDeg = Math.toDegrees(theta)
            val rgbValue = (0..<3).map { rgbIndex ->
                cubicBSplineInterpolate(angles, phaseRGB[rgbIndex], thetaDeg)
            }.toDoubleArray()
            val logluv32Value = RGBToRGBM(rgbValue, mDiv)
            val rgbValueBack = RGBMToRGB(logluv32Value)
            maxError = max(maxError, (rgbValue zip rgbValueBack).maxOf {
                abs(it.first - it.second) / it.first
            })
            logluv32Value.forEach {
                mapped.putShort(it)
            }
        }
    }
}

println("Max error: $maxError")