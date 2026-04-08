import kotlin.io.path.Path
import kotlin.io.path.outputStream
import kotlin.io.path.readLines

val f82Table = Path("adobe-fresnel.tsv").readLines().asSequence()
    .drop(1)
    .map { line ->
        val (name, f0, f82) = line.split('\t')
        name to f82.removeSurrounding("(", ")").splitToSequence(", ").map { it.toDouble() }.toList()
    }
    .toMap()

// 230: Iron     (Fe)
// 231: Gold     (Au)
// 232: Aluminum (Al)
// 233: Chrome   (Cr)
// 234: Copper   (Cu)
// 235: Lead     (Pb)
// 236: Platinum (Pt)
// 237: Silver   (Ag)
val labpbrTable = listOf(
    "Fe",
    "Au",
    "Al",
    "Cr",
    "Cu",
    "Pb",
    "Pt",
    "Ag",
)

Path("../shaders/textures/f82.bin").outputStream().buffered().use { out ->
    for (i in 0 .. 229) {
        repeat(3) { out.write(255) }
        out.write(0)
    }
    labpbrTable.asSequence()
        .map {
            if (it == "Pb") {
                listOf(0.612066, 0.629024, 0.663700)
            } else {
                f82Table[it]!!
            }
        }
        .forEach { f82 ->
            val minVal = f82.min()
            val minValQuantized = (minVal * 255.0).toInt().coerceIn(0, 255)
            val minValBack = minValQuantized.toDouble() / 255.0
            f82.forEach {
                val normalized = (it - minValBack) / (1.0 - minValBack)
                val quantized = (normalized * 255.0).toInt().coerceIn(0, 255)
                out.write(quantized)
            }
            out.write(minValQuantized)
        }
    for (i in 238 .. 255) {
        repeat(3) { out.write(255) }
        out.write(0)
    }
}