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
    val groups = Array(size) { IntArray(size) }
    var i = 0
    val groupSizeX = 4
    val groupSizeY = 2
    for (y in 0..<size step groupSizeY) {
        for (x in 0..<size step groupSizeX) {
            val groupID = i++
            for (dy in 0..<groupSizeY) {
                for (dx in 0..<groupSizeX) {
                    groups[y + dy][x + dx] = groupID
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
                        permuteTemp[i++] = groups[(dstY + dy) % size][(dstX + dx) % size]
                    }
                }
                permuteTemp.shuffle(randoms[y][x])
                i = 0
                for (dy in 0..<2) {
                    for (dx in 0..<2) {
                        groups[(dstY + dy) % size][(dstX + dx) % size] = permuteTemp[i++]
                    }
                }
            }
        }
    }

    repeat(sigmaToShuffleCount(sigma)) {
        shuffleGrid(it, it)
    }

    val groupPos = Array(size * size / 8) { IntArray(17) }
    for (y in 0..<size) {
        for (x in 0..<size) {
            val groupId = groups[y][x]
            val arr = groupPos[groupId]
            val idx = (arr[0]++) * 2
            arr[idx + 1] = x
            arr[idx + 2] = y
        }
    }
    require(groupPos.all { it[0] == 8 }) { "Generated reuse texture contains a malformed group" }

    val temp = groupPos.map { it.slice(1..<17) }
    val lookup = temp.asSequence()
        .withIndex()
        .flatMap { (i, group) ->
            group.chunked(2).map { (it[0] to it[1]) to i }
        }
        .toMap(mutableMapOf())

    val final = mutableListOf<List<Int>>()
    for (y in 0..<size) {
        for (x in 0..<size) {
            val myCoord = x to y
            lookup.remove(myCoord)?.let { groupId ->
                val element = temp[groupId]
                val coords = element.chunked(2).map { it[0] to it[1] }.toMutableList()
                require(coords.remove(myCoord)) { "Reuse group lookup lost its anchor coordinate" }
                for (coord in coords) {
                    require(lookup.remove(coord) == groupId) { "Reuse group lookup contains inconsistent coordinates" }
                }
                val orderedCoords = listOf(myCoord) + coords
                final.add(orderedCoords.flatMap { listOf(it.first, it.second) })
            }
        }
    }

    require(final.size == size * size / 8) { "Generated ${final.size} groups, expected ${size * size / 8}" }
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

    for (group in data) {
        for (a in 0..<8) {
            for (b in a + 1..<8) {
                val x1 = group[a * 2]
                val y1 = group[a * 2 + 1]
                val x2 = group[b * 2]
                val y2 = group[b * 2 + 1]
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
    val outputData = ByteArray(data.size * 16)
    for (i in data.indices) {
        val groupData = data[i]
        val outputBase = i * 16
        outputData.writeIntLE(outputBase, packCoords(groupData[0], groupData[1], groupData[2], groupData[3]))
        outputData.writeIntLE(outputBase + 4, packCoords(groupData[4], groupData[5], groupData[6], groupData[7]))
        outputData.writeIntLE(outputBase + 8, packCoords(groupData[8], groupData[9], groupData[10], groupData[11]))
        outputData.writeIntLE(outputBase + 12, packCoords(groupData[12], groupData[13], groupData[14], groupData[15]))
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
