import java.nio.file.Path
import kotlin.io.path.*
import kotlin.math.sin

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
    // Find the segment containing xPos
    var i = 1
    while (i < Xs.size - 2 && Xs[i + 1] < xPos) {
        i++
    }

    // Get the four points needed for the interpolation
    val x0 = Xs[i - 1]
    val x1 = Xs[i]
    val x2 = Xs[i + 1]
    val x3 = Xs[i + 2]

    val y0 = Ys[i - 1]
    val y1 = Ys[i]
    val y2 = Ys[i + 1]
    val y3 = Ys[i + 2]

    // Calculate the t parameter in [0,1]
    val t = (xPos - x1) / (x2 - x1)

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

val XYZ2SRGBR = doubleArrayOf(3.2406255, -1.537208, -0.4986286)
val XYZ2SRGBG = doubleArrayOf(-0.9689307, 1.8757561, 0.0415175)
val XYZ2SRGBB = doubleArrayOf(0.0557101, -0.2040211, 1.0569959)
val XYZ2SRGB = listOf(XYZ2SRGBR, XYZ2SRGBG, XYZ2SRGBB)

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

    val Csrgb = XYZ2SRGB.map { row ->
        (Cxyz zip row).sumOf { it.first * it.second }
    }.toDoubleArray()

    return Csrgb
}

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
            val ext = cols[1]
            val sca = cols[2]
            val asym = cols[5]
            println("extinction: ${doColorMatching(X, ext).contentToString()}")
            println("scattering: ${doColorMatching(X, sca).contentToString()}")
            println("asymmetry: ${doColorMatching(X, asym).contentToString()}")

            val vpfIndex = textLines.indexOf("# volume phase function [1/km]:")
            outputDir.resolve("${name}_phase.csv").bufferedWriter().use { writer ->
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
                        list.first() to (list.drop(1).toDoubleArray() zip sca).map { (phase, sca) -> phase / sca }
                            .toDoubleArray()
                    }
                    .toList()

                val angles = rows.map { it.first }.toDoubleArray()

                val phaseRGB = rows.asSequence()
                    .map { (angle, Ys) -> doColorMatching(Xs, Ys) }
                    .toList()

                val phaseSums = (0..<3).map { rgbIndex ->
                    integratePhaseFunction(angles, phaseRGB.map { it[rgbIndex] }.toDoubleArray())
                }

                phaseRGB.forEach { anglePhase ->
                    repeat(3) { rgbIndex ->
                        anglePhase[rgbIndex] /= phaseSums[rgbIndex]
                    }
                }

                (angles zip phaseRGB).forEach { (angle, rgb) ->
                    writer.append(angle.toString())
                    repeat(3) {
                        writer.append(',')
                        writer.append(rgb[it].toString())
                    }
                    writer.appendLine()
                }
            }

            println()
        }
}
