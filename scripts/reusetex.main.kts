/*
    References:
        [LKW26] Lin, Daqi, et al. "ReSTIR PT Enhanced: Algorithmic Advances for Faster and More Robust ReSTIR Path Tracing".
            Proceedings of the ACM on Computer Graphics and Interactive Techniques. 9, 1, Article 13 (2026).
            https://doi.org/10.1145/3804494

        You can find full license texts in /licenses
*/
@file:DependsOn("org.apache.commons:commons-rng-simple:1.6")

import org.apache.commons.rng.UniformRandomProvider
import org.apache.commons.rng.simple.RandomSource
import kotlin.io.path.Path
import kotlin.math.pow
import kotlin.math.sqrt

val size = 256
val sigma = 16.0

fun IntArray.shuffle(random: UniformRandomProvider): Unit {
    for (i in lastIndex downTo 1) {
        val j = random.nextInt(i + 1)
        val copy = this[i]
        this[i] = this[j]
        this[j] = copy
    }
}

fun main(baseRandom: UniformRandomProvider): List<List<Int>> {
    val quads = Array(size) { IntArray(size) }
    var i = 0
    val groupSizeX = 2
    val groupSizeY = 2
    for (y in 0..<size step groupSizeX) {
        for (x in 0..<size step groupSizeY) {
            val groupID = i++
            for (dy in 0..<groupSizeY) {
                for (dx in 0..<groupSizeX) {
                    quads[y + dy][x + dx] = groupID
                }
            }
        }
    }

    val randoms = Array(size / 2) { Array(size / 2) { RandomSource.XO_SHI_RO_256_PP.create(baseRandom.nextLong()) } }

    fun sigmaToShuffleCount(sigma: Double): Int {
        return (0.5 * sigma.pow(2) + 1.46 * sigma.pow(-1) + 1.76 * sigma.pow(-2) + 0.656 * sigma.pow(-3) + 0.5).toInt()
    }

    fun shuffleGrid(offsetX: Int, offsetY: Int) {
        for (y in 0..<size / 2) {
            val dstY = y * 2 + offsetY
            for (x in 0..<size / 2) {
                val dstX = x * 2 + offsetX
                val permuteTemp = IntArray(4)
                var i = 0
                for (dy in 0..<2) {
                    for (dx in 0..<2) {
                        permuteTemp[i++] = quads[(dstY + dy) % size][(dstX + dx) % size]
                    }
                }
                permuteTemp.shuffle(randoms[y][x])
                i = 0
                for (dy in 0..<2) {
                    for (dx in 0..<2) {
                        quads[(dstY + dy) % size][(dstX + dx) % size] = permuteTemp[i++]
                    }
                }
            }
        }
    }

    repeat(sigmaToShuffleCount(sigma)) {
        shuffleGrid(it, it)
    }

    val quadPos = Array(size * size / 4) { IntArray(9) }
    for (y in 0..<size) {
        for (x in 0..<size) {
            val quadId = quads[y][x]
            val arr = quadPos[quadId]
            val idx = (arr[0]++) * 2
            arr[idx + 1] = x
            arr[idx + 2] = y
        }
    }
    require(quadPos.all { it[0] == 4 }) { "Generated reuse texture contains a malformed quad" }

    val temp = quadPos.map { it.slice(1..<9) }
    val lookup = temp.asSequence()
        .withIndex()
        .flatMap { (i, quad) ->
            quad.chunked(2).map { (it[0] to it[1]) to i }
        }
        .toMap(mutableMapOf())

    val final = mutableListOf<List<Int>>()
    for (y in 0..<size) {
        for (x in 0..<size) {
            val myQuad = x to y
            lookup.remove(myQuad)?.let { quadId ->
                val element = temp[quadId]
                val coords = element.chunked(2).map { it[0] to it[1] }.toMutableList()
                require(coords.remove(myQuad)) { "Quad lookup lost its anchor coordinate" }
                for (coord in coords) {
                    require(lookup.remove(coord) == quadId) { "Quad lookup contains inconsistent coordinates" }
                }
                val orderedCoords = listOf(myQuad) + coords
                final.add(orderedCoords.flatMap { listOf(it.first, it.second) })
            }
        }
    }

    require(final.size == size * size / 4) { "Generated ${final.size} quads, expected ${size * size / 4}" }
    return final
}

val baseRandom = RandomSource.XO_SHI_RO_256_PP.create(1145141919810L)
val basePath = Path("../shaders/textures")
val dists = mutableListOf<Double>()

fun packCoords(x0: Int, y0: Int, x1: Int, y1: Int): Int {
    return (x0 and 0xff) or
        ((y0 and 0xff) shl 8) or
        ((x1 and 0xff) shl 16) or
        ((y1 and 0xff) shl 24)
}

fun ByteArray.writeIntLE(offset: Int, value: Int) {
    this[offset] = (value and 0xff).toByte()
    this[offset + 1] = ((value ushr 8) and 0xff).toByte()
    this[offset + 2] = ((value ushr 16) and 0xff).toByte()
    this[offset + 3] = ((value ushr 24) and 0xff).toByte()
}

repeat(8) {
    val data = main(baseRandom)

    for (quad in data) {
        for (a in 0..<4) {
            for (b in a + 1..<4) {
                val x1 = quad[a * 2]
                val y1 = quad[a * 2 + 1]
                val x2 = quad[b * 2]
                val y2 = quad[b * 2 + 1]
                var dx = x2 - x1
                if (dx > size / 2) dx -= size else if (dx < -size / 2) dx += size
                var dy = y2 - y1
                if (dy > size / 2) dy -= size else if (dy < -size / 2) dy += size
                val distSq = dx * dx + dy * dy
                dists += sqrt(distSq.toDouble())
            }
        }
    }

    val outputPath = basePath.resolve("restir_reusetex${it}.bin")
    val outputData = ByteArray(data.size * 8)
    for (i in data.indices) {
        val quadData = data[i]
        val outputBase = i * 8
        outputData.writeIntLE(outputBase, packCoords(quadData[0], quadData[1], quadData[2], quadData[3]))
        outputData.writeIntLE(outputBase + 4, packCoords(quadData[4], quadData[5], quadData[6], quadData[7]))
    }
    outputPath.toFile().writeBytes(outputData)
}

val mean = dists.average()
val stddev = sqrt(dists.map { (it - mean).pow(2) }.average())
println("Mean: $mean")
println("Stddev: $stddev")

val bins = IntArray(1024)
dists.forEach {
    bins[it.toInt()]++
}
val histo = bins.slice(0..bins.indexOfLast { it != 0 })
println("Histogram: $histo")
