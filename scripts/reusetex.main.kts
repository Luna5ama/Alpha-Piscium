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
import java.util.stream.IntStream
import kotlin.io.path.Path
import kotlin.math.hypot
import kotlin.math.pow
import kotlin.math.sqrt

val size = 256
val threadGroupSize = 256
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
    val groupIDGrid = Array(size) { IntArray(size) }
    var i = 0
    val groupSizeX = 4
    val groupSizeY = 2
    for (y in 0..<size step groupSizeY) {
        for (x in 0..<size step groupSizeX) {
            val groupID = i++
            for (dy in 0..<groupSizeY) {
                for (dx in 0..<groupSizeX) {
                    groupIDGrid[y + dy][x + dx] = groupID
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
                        permuteTemp[i++] = groupIDGrid[(dstY + dy) % size][(dstX + dx) % size]
                    }
                }
                permuteTemp.shuffle(randoms[y][x])
                i = 0
                for (dy in 0..<2) {
                    for (dx in 0..<2) {
                        groupIDGrid[(dstY + dy) % size][(dstX + dx) % size] = permuteTemp[i++]
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
            val groupId = groupIDGrid[y][x]
            val arr = groupPos[groupId]
            val idx = (arr[0]++) * 2
            arr[idx + 1] = x
            arr[idx + 2] = y
        }
    }
    require(groupPos.all { it[0] == 8 }) { "Generated reuse texture contains a malformed group" }

    val groups = groupPos
        .map { it.slice(1..<17).chunked(2).map { it[0] to it[1] } }

    val centroids = MutableList(size * size / threadGroupSize) {
        baseRandom.nextDouble(0.0, size.toDouble()) to baseRandom.nextDouble(0.0, size.toDouble())
    }

    val groupAssignments = IntArray(groups.size)

    repeat(1024) {
        IntStream.range(0, groups.size).parallel().forEach { groupID ->
            val group = groups[groupID]
            groupAssignments[groupID] = centroids.withIndex().minBy { (index, centroid) ->
                val (cx, cy) = centroid
                group.sumOf { (x, y) ->
                    hypot((x + 0.5) - cx, (y + 0.5) - cy)
                }
            }.index
        }

        groupAssignments.withIndex().groupBy { it.value }
            .entries
            .parallelStream()
            .forEach { (centroidID, assignedGroups) ->
                val allPoints = assignedGroups.flatMap { groups[it.index] }
                val avgX = allPoints.sumOf { it.first + 0.5 } / allPoints.size
                val avgY = allPoints.sumOf { it.second + 0.5 } / allPoints.size
                centroids[centroidID] = avgX to avgY
            }
    }

    fun morton2D(localX: Int, localY: Int): Int {
        fun spread9(v: Int): Int {
            var x = v and 0x1FF
            x = (x or (x shl 8)) and 0x00FF00FF
            x = (x or (x shl 4)) and 0x0F0F0F0F
            x = (x or (x shl 2)) and 0x33333333
            x = (x or (x shl 1)) and 0x55555555
            return x
        }

        val px = localX + 128
        val py = localY + 128

        require(px in 0..511)
        require(py in 0..511)

        return spread9(px) or (spread9(py) shl 1)
    }

    val comparator = compareBy<Pair<Int, Int>> { morton2D(it.first, it.second) }
    val listComp = compareBy<List<Pair<Int, Int>>> { morton2D(it[0].first, it[0].second) }
    val centroidComp = compareBy<Pair<Int, *>> {
        val centroid = centroids[it.first]
        morton2D(centroid.first.toInt(), centroid.second.toInt())
    }

    data class GroupAssignment(val groupID: Int, val centroidID: Int)

    return groupAssignments.withIndex()
        .map { GroupAssignment(it.index, it.value) }
        .groupBy { it.centroidID }
        .toList()
        .sortedWith(centroidComp)
        .flatMap { (_, assignedGroups) ->
            assignedGroups.map {
                groups[it.groupID].sortedWith(comparator)
            }.sortedWith(listComp).map {
                it.flatMap { listOf(it.first, it.second) }
            }
        }
}

val baseRandom = RandomSource.XO_SHI_RO_256_PP.create(69691145141919810L)
val basePath = Path("../shaders/textures")
val dists = mutableListOf<Double>()

repeat(4) { texIndex ->
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

    val outputPath = basePath.resolve("restir_reusetex${texIndex}.bin")
    val outputData = ByteArray(data.size * 16)
    for (i in data.indices) {
        val groupData = data[i]
        val outputBase = i * 16
        repeat(16) { outputData[outputBase + it] = groupData[it].toByte() }
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
