import kotlin.math.*

fun agxCurveScalar(x: Double): Double {
    val threshold = 0.6060606060606061

    // Select constants based on the threshold
    val a: Double
    val b: Double
    val c: Double

    if (x < threshold) {
        a = 59.507875
        b = 3.0
        c = -0.3333333333333333 // Replaced -0.33333333 with higher precision
    } else {
        a = 69.86278913545539
        b = 3.25                // Equivalent to 13.0 / 4.0
        c = -0.3076923076923077 // Equivalent to -4.0 / 13.0
    }

    // Main curve calculation
    // Equation: 0.5 + (2.0 * (v - threshold)) * (1.0 + a * |v - threshold|^b)^c
    val diff = x - threshold
    return 0.5 + (2.0 * diff) * (1.0 + a * abs(diff).pow(b)).pow(c)
}

fun agxInverse(targetY: Double): Double {
    val newtonRaphsonIterations = 1000
    val initEps = 0.1

    var x = targetY
    var epsilon = initEps

    repeat(newtonRaphsonIterations) {
        val y = agxCurveScalar(x)
        val dy = (agxCurveScalar(x + epsilon) - y) / epsilon
        x -= (y - targetY) / dy
        epsilon *= 0.99
    }

    return x
}

fun agxCurveScalarF(x: Float): Float {
    val threshold = 0.6060606060606061f

    // Select constants based on the threshold
    val a: Float
    val b: Float
    val c: Float

    if (x < threshold) {
        a = 59.507875f
        b = 3.0f
        c = -0.3333333333333333f // Replaced -0.33333333 with higher precision
    } else {
        a = 69.86278913545539f
        b = 3.25f                // Equivalent to 13.0 / 4.0
        c = -0.3076923076923077f // Equivalent to -4.0 / 13.0
    }

    // Main curve calculation
    // Equation: 0.5 + (2.0 * (v - threshold)) * (1.0 + a * |v - threshold|^b)^c
    val diff = x - threshold
    return 0.5f + (2.0f * diff) * (1.0f + a * abs(diff).pow(b)).pow(c)
}

fun agxInverse3(targetY: Float): Float {
    val newtonRaphsonIterations = 3
    val initEps = 0.25f

    var x = targetY
    var epsilon = initEps

    repeat(newtonRaphsonIterations) {
        val y = agxCurveScalarF(x)
        val dy = (agxCurveScalarF(x + epsilon) - y) / epsilon
        x -= (y - targetY) / dy
        epsilon *= 0.2f
    }

    return x
}

fun agxInverse4(targetY: Float): Float {
    val newtonRaphsonIterations = 4
    val initEps = 0.25f

    var x = targetY
    var epsilon = initEps

    repeat(newtonRaphsonIterations) {
        val y = agxCurveScalarF(x)
        val dy = (agxCurveScalarF(x + epsilon) - y) / epsilon
        x -= (y - targetY) / dy
        epsilon *= 0.15f
    }

    return x
}

fun agxInverse5(targetY: Float): Float {
    val newtonRaphsonIterations = 5
    val initEps = 0.25f

    var x = targetY
    var epsilon = initEps

    repeat(newtonRaphsonIterations) {
        val y = agxCurveScalarF(x)
        val dy = (agxCurveScalarF(x + epsilon) - y) / epsilon
        x -= (y - targetY) / dy
        epsilon *= 0.125f
    }

    return x
}

val n = 1024
val yError = (1..n).map { i ->
    val y = i.toDouble() / (n).toDouble()
    val inv = agxInverse4(y.toFloat()).toDouble()
    val newY = agxCurveScalar(inv)
    abs(newY - y)
}

//val xError = (1..n).map { i ->
//    val x = i.toDouble() / (n).toDouble()
//    val y = agxCurveScalar(x)
//    val inv = agxInverse4(y.toFloat()).toDouble()
//    abs(inv - x)
//}

println("8 bit error %.10E".format(1.0 / 255.0))
println("10 bit error %.10E".format(1.0 / 1023.0))
println("12 bit error %.10E".format(1.0 / 4095.0))
println("16 bit error %.10E".format(1.0 / 65535.0))
println("\nY error:")
println("Max error: %.10E".format(yError.maxOrNull() ?: 0.0))
println("Avg error: %.10E".format(yError.average()))
//println("\nX error:")
//println("Max error: %.10E".format(xError.maxOrNull() ?: 0.0))
//println("Avg error: %.10E".format(xError.average()))
