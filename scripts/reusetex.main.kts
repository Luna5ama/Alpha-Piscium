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
    val pairs = Array(size) { IntArray(size) }
    var i = 0
    for (y in 0..<size) {
        for (x in 0..<size) {
            pairs[y][x] = (i++) / 2
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
                        permuteTemp[i++] = pairs[(dstY + dy) % size][(dstX + dx) % size]
                    }
                }
                permuteTemp.shuffle(randoms[y][x])
                i = 0
                for (dy in 0..<2) {
                    for (dx in 0..<2) {
                        pairs[(dstY + dy) % size][(dstX + dx) % size] = permuteTemp[i++]
                    }
                }
            }
        }
    }

    repeat(sigmaToShuffleCount(sigma)) {
        shuffleGrid(it, it)
    }

    val pairPos = Array(size * size / 2) { IntArray(5) }
    for (y in 0..<size) {
        for (x in 0..<size) {
            val pairId = pairs[y][x]
            val arr = pairPos[pairId]
            val idx = (arr[0]++) * 2
            arr[idx + 1] = x
            arr[idx + 2] = y
        }
    }

    val temp = pairPos.map { it.slice(1..<5) }
    val lookup = temp.asSequence()
        .withIndex()
        .flatMap { (i, pair) ->
            pair.chunked(2).map { (it[0] to it[1]) to i }
        }
        .toMap(mutableMapOf())

    val final = mutableListOf<List<Int>>()
    for (y in 0..<size) {
        for (x in 0..<size) {
            val myPair = x to y
            lookup.remove(myPair)?.let { pairId ->
                val element = temp[pairId]
                var otherPair = element[0] to element[1]
                if (otherPair == myPair) {
                    otherPair = element[2] to element[3]
                }
                lookup.remove(otherPair)
                final.add(listOf(myPair.first, myPair.second, otherPair.first, otherPair.second))
            }
        }
    }

    return final
}

val baseRandom = RandomSource.XO_SHI_RO_256_PP.create(1145141919810L)
val basePath = Path("../shaders/textures")
val dists = mutableListOf<Double>()
repeat(8) {
    val data = main(baseRandom)

    for (pairs in data) {
        val x1 = pairs[0]
        val y1 = pairs[1]
        val x2 = pairs[2]
        val y2 = pairs[3]
        var dx = x2 - x1
        if (dx > size / 2) dx -= size else if (dx < -size / 2) dx += size
        var dy = y2 - y1
        if (dy > size / 2) dy -= size else if (dy < -size / 2) dy += size
        val distSq = dx * dx + dy * dy
        dists += sqrt(distSq.toDouble())
    }

    val outputPath = basePath.resolve("restir_reusetex${it}.bin")
    val outputData = ByteArray(data.size * 4)
    for (i in data.indices) {
        val pairData = data[i]
        val outputBase = i * 4
        outputData[outputBase] = (pairData[0] and 0xff).toByte()
        outputData[outputBase + 1] = (pairData[1] and 0xff).toByte()
        outputData[outputBase + 2] = (pairData[2] and 0xff).toByte()
        outputData[outputBase + 3] = (pairData[3] and 0xff).toByte()
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