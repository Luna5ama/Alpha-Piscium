import kotlin.math.*

val Gs = doubleArrayOf(0.8615159687912013, 0.8732937077048064, 0.9375708300315341)
var Es = DoubleArray(Gs.size) { 1.0 }

repeat(10000) {
    repeat(Gs.size) { i ->
        val g = Gs[i]
        var e = Es[i]
        val gFromE = 1.0 / e - 2.0 / ln(2.0 * e + 1.0) + 1.0
        val deriv = 4.0 / ((2.0 * e + 1.0) * (ln(2.0 * e + 1.0)).pow(2)) - 1.0 / (e).pow(2)
        e = e - (gFromE - g) / deriv
        Es[i] = e
    }
}

println("Gs: ${Gs.contentToString()}")
println("Es: ${Es.contentToString()}")