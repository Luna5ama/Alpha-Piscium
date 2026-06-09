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
import kotlin.math.hypot
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

val size = 256
val threadGroupSize = 128
val groupSizeX = 4
val groupSizeY = 2
val groupPixelCount = groupSizeX * groupSizeY
val groupCount = size * size / groupPixelCount
val clusterPixelCapacity = threadGroupSize
val clusterGroupCapacity = clusterPixelCapacity / groupPixelCount
val centroidCount = groupCount / clusterGroupCapacity
val maxIterations = 16
val reuseCostScale = 1000.0
val sigma = 16.0

require(groupCount * groupPixelCount == size * size)
require(clusterGroupCapacity * groupPixelCount == clusterPixelCapacity)
require(centroidCount * clusterGroupCapacity == groupCount)

fun IntArray.shuffle(random: UniformRandomProvider): Unit {
    for (i in lastIndex downTo 1) {
        val j = random.nextInt(i + 1)
        val copy = this[i]
        this[i] = this[j]
        this[j] = copy
    }
}

class MinHeap(private val distances: LongArray) {
    private val nodes = IntArray(distances.size)
    private val positions = IntArray(distances.size) { -1 }
    var size = 0

    fun clear() {
        for (index in 0..<size) {
            positions[nodes[index]] = -1
        }
        size = 0
    }

    fun addOrDecrease(node: Int) {
        var index = positions[node]
        if (index == -1) {
            index = size++
            nodes[index] = node
            positions[node] = index
        }
        while (index > 0) {
            val parent = (index - 1) / 2
            if (!less(node, nodes[parent])) break
            nodes[index] = nodes[parent]
            positions[nodes[index]] = index
            index = parent
        }
        nodes[index] = node
        positions[node] = index
    }

    fun removeFirst(): Int {
        val first = nodes[0]
        positions[first] = -1
        size--
        if (size == 0) return first

        val last = nodes[size]
        var index = 0
        while (true) {
            val left = index * 2 + 1
            if (left >= size) break
            val right = left + 1
            var child = left
            if (right < size && less(nodes[right], nodes[left])) child = right
            if (!less(nodes[child], last)) break
            nodes[index] = nodes[child]
            positions[nodes[index]] = index
            index = child
        }
        nodes[index] = last
        positions[last] = index
        return first
    }

    private fun less(a: Int, b: Int): Boolean {
        return distances[a] < distances[b] || distances[a] == distances[b] && a < b
    }
}

class MinCostFlowGraph(nodeCount: Int, edgeCapacity: Int) {
    val firstEdge = IntArray(nodeCount) { -1 }
    val to = IntArray(edgeCapacity)
    val next = IntArray(edgeCapacity)
    val capacity = IntArray(edgeCapacity)
    val cost = IntArray(edgeCapacity)
    var edgeCount = 0

    fun addEdge(from: Int, to: Int, capacity: Int, cost: Int): Int {
        val edge = edgeCount
        addDirectedEdge(from, to, capacity, cost)
        addDirectedEdge(to, from, 0, -cost)
        return edge
    }

    private fun addDirectedEdge(from: Int, to: Int, capacity: Int, cost: Int) {
        this.to[edgeCount] = to
        this.capacity[edgeCount] = capacity
        this.cost[edgeCount] = cost
        next[edgeCount] = firstEdge[from]
        firstEdge[from] = edgeCount
        edgeCount++
    }
}

data class AssignmentResult(
    val assignments: IntArray,
    val totalCost: Long
)

fun addMember(members: IntArray, count: Int, groupID: Int) {
    var position = count
    while (position > 0 && members[position - 1] > groupID) {
        members[position] = members[position - 1]
        position--
    }
    members[position] = groupID
}

fun removeMember(members: IntArray, count: Int, groupID: Int) {
    var position = members.binarySearch(groupID, 0, count)
    require(position >= 0)
    while (position + 1 < count) {
        members[position] = members[position + 1]
        position++
    }
}

fun assignGroupsWithMinCostFlow(
    groups: List<List<Pair<Int, Int>>>,
    centroids: List<Pair<Double, Double>>
): AssignmentResult {
    val source = 0
    val groupNodeOffset = 1
    val centroidNodeOffset = groupNodeOffset + groupCount
    val sink = centroidNodeOffset + centroidCount
    val nodeCount = sink + 1
    val forwardEdgeCount = groupCount + groupCount * centroidCount + centroidCount
    val graph = MinCostFlowGraph(nodeCount, forwardEdgeCount * 2)
    val groupCentroidEdgeOffset = groupCount * 2
    val centroidSinkEdgeOffset = groupCentroidEdgeOffset + groupCount * centroidCount * 2

    for (groupID in 0..<groupCount) {
        graph.addEdge(source, groupNodeOffset + groupID, 1, 0)
    }
    for (groupID in 0..<groupCount) {
        val group = groups[groupID]
        for (centroidID in 0..<centroidCount) {
            val (cx, cy) = centroids[centroidID]
            var costDouble = 0.0
            for ((x, y) in group) {
                costDouble += hypot((x + 0.5) - cx, (y + 0.5) - cy)
            }
            graph.addEdge(
                groupNodeOffset + groupID,
                centroidNodeOffset + centroidID,
                1,
                (costDouble * reuseCostScale).roundToInt()
            )
        }
    }
    for (centroidID in 0..<centroidCount) {
        graph.addEdge(centroidNodeOffset + centroidID, sink, clusterGroupCapacity, 0)
    }
    require(graph.edgeCount == forwardEdgeCount * 2)

    val potentials = LongArray(nodeCount)
    val distances = LongArray(nodeCount)
    val parentEdges = IntArray(nodeCount)
    val seen = IntArray(nodeCount)
    val settled = IntArray(nodeCount)
    val settledNodes = IntArray(nodeCount)
    val counts = IntArray(centroidCount)
    val members = Array(centroidCount) { IntArray(clusterGroupCapacity) }
    val heap = MinHeap(distances)
    var search = 0
    var flow = 0

    fun relax(from: Int, edge: Int) {
        val to = graph.to[edge]
        if (settled[to] == search) return
        val reducedCost = graph.cost[edge].toLong() + potentials[from] - potentials[to]
        require(reducedCost >= 0) {
            "Negative reduced cost $reducedCost from $from to $to"
        }
        val nextDistance = distances[from] + reducedCost
        if (seen[to] != search || nextDistance < distances[to]) {
            seen[to] = search
            distances[to] = nextDistance
            parentEdges[to] = edge
            heap.addOrDecrease(to)
        } else if (nextDistance == distances[to] && edge < parentEdges[to]) {
            parentEdges[to] = edge
        }
    }

    // Activate source-to-group unit edges in group order; residual paths can reassign earlier groups.
    for (groupID in 0..<groupCount) {
        val sourceEdge = groupID * 2
        require(graph.to[sourceEdge] == groupNodeOffset + groupID && graph.capacity[sourceEdge] == 1)
        graph.capacity[sourceEdge] = 0
        graph.capacity[sourceEdge xor 1] = 1

        search++
        heap.clear()
        var settledCount = 0
        val start = groupNodeOffset + groupID
        var startPotential = Long.MIN_VALUE
        var edge = graph.firstEdge[start]
        while (edge != -1) {
            if (graph.capacity[edge] > 0 && graph.to[edge] != source) {
                startPotential = maxOf(startPotential, potentials[graph.to[edge]] - graph.cost[edge])
            }
            edge = graph.next[edge]
        }
        potentials[start] = startPotential
        seen[start] = search
        distances[start] = 0
        parentEdges[start] = -1
        heap.addOrDecrease(start)

        while (heap.size > 0) {
            val node = heap.removeFirst()
            settled[node] = search
            settledNodes[settledCount++] = node
            if (node == sink) break

            if (node < centroidNodeOffset) {
                edge = graph.firstEdge[node]
                while (edge != -1) {
                    if (graph.capacity[edge] > 0 && graph.to[edge] != source) {
                        relax(node, edge)
                    }
                    edge = graph.next[edge]
                }
            } else {
                val centroidID = node - centroidNodeOffset
                edge = centroidSinkEdgeOffset + centroidID * 2
                if (graph.capacity[edge] > 0) {
                    relax(node, edge)
                }
                for (memberIndex in 0..<counts[centroidID]) {
                    val member = members[centroidID][memberIndex]
                    edge = groupCentroidEdgeOffset + (member * centroidCount + centroidID) * 2 + 1
                    require(graph.capacity[edge] == 1)
                    relax(node, edge)
                }
            }
        }
        require(settled[sink] == search) {
            "Min-cost flow stopped at $flow / $groupCount groups"
        }

        val sinkDistance = distances[sink]
        for (index in 0..<settledCount) {
            val node = settledNodes[index]
            if (distances[node] < sinkDistance) {
                potentials[node] += distances[node] - sinkDistance
            }
        }

        var node = sink
        while (node != start) {
            edge = parentEdges[node]
            if (edge >= groupCentroidEdgeOffset && edge < centroidSinkEdgeOffset) {
                val forwardEdge = edge and -2
                val assignment = (forwardEdge - groupCentroidEdgeOffset) / 2
                val assignedGroup = assignment / centroidCount
                val centroidID = assignment % centroidCount
                if (edge and 1 == 0) {
                    addMember(members[centroidID], counts[centroidID], assignedGroup)
                    counts[centroidID]++
                } else {
                    removeMember(members[centroidID], counts[centroidID], assignedGroup)
                    counts[centroidID]--
                }
            }
            graph.capacity[edge]--
            graph.capacity[edge xor 1]++
            node = graph.to[edge xor 1]
        }
        flow++
    }

    require(flow == groupCount)
    require(counts.all { it == clusterGroupCapacity })
    counts.fill(0)
    val assignments = IntArray(groupCount) { -1 }
    var totalCost = 0L
    for (groupID in 0..<groupCount) {
        var assignedCount = 0
        for (centroidID in 0..<centroidCount) {
            val edge = groupCentroidEdgeOffset + (groupID * centroidCount + centroidID) * 2
            if (graph.capacity[edge xor 1] == 0) continue
            assignments[groupID] = centroidID
            counts[centroidID]++
            totalCost += graph.cost[edge]
            assignedCount++
        }
        require(assignedCount == 1) { "Group $groupID has $assignedCount centroid assignments" }
    }
    require(counts.all { it == clusterGroupCapacity }) {
        "Expected $clusterGroupCapacity groups per centroid, found ${counts.min()}..${counts.max()}"
    }
    return AssignmentResult(assignments, totalCost)
}

fun updateCentroidsFromAssignments(
    groups: List<List<Pair<Int, Int>>>,
    assignments: IntArray,
    centroids: MutableList<Pair<Double, Double>>
) {
    val sumX = DoubleArray(centroidCount)
    val sumY = DoubleArray(centroidCount)
    val pointCounts = IntArray(centroidCount)
    for (groupID in 0..<groupCount) {
        val centroidID = assignments[groupID]
        for ((x, y) in groups[groupID]) {
            sumX[centroidID] += x + 0.5
            sumY[centroidID] += y + 0.5
            pointCounts[centroidID]++
        }
    }
    require(pointCounts.all { it == clusterPixelCapacity })
    for (centroidID in 0..<centroidCount) {
        centroids[centroidID] = sumX[centroidID] / pointCounts[centroidID] to
            sumY[centroidID] / pointCounts[centroidID]
    }
}

fun main(baseRandom: UniformRandomProvider, textureIndex: Int): List<List<Int>> {
    val groupIDGrid = Array(size) { IntArray(size) }
    var i = 0
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

    val groupPos = Array(groupCount) { IntArray(groupPixelCount * 2 + 1) }
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
        .map { it.slice(1..<groupPixelCount * 2 + 1).chunked(2).map { it[0] to it[1] } }

    val centroids = MutableList(centroidCount) {
        baseRandom.nextDouble(0.0, size.toDouble()) to baseRandom.nextDouble(0.0, size.toDouble())
    }

    val groupAssignments = IntArray(groupCount) { -1 }
    var finalCost = 0L
    var iterationCount = 0
    while (iterationCount < maxIterations) {
        val result = try {
            assignGroupsWithMinCostFlow(groups, centroids)
        } catch (error: RuntimeException) {
            throw IllegalStateException(
                "MCF assignment failed: texture=$textureIndex iteration=${iterationCount + 1} " +
                    "requiredFlow=$groupCount centroidCount=$centroidCount clusterGroupCapacity=$clusterGroupCapacity",
                error
            )
        }
        var changedCount = 0
        for (groupID in 0..<groupCount) {
            if (groupAssignments[groupID] != result.assignments[groupID]) changedCount++
        }
        result.assignments.copyInto(groupAssignments)
        finalCost = result.totalCost
        iterationCount++
        println("texture=$textureIndex iteration=$iterationCount changed=$changedCount cost=$finalCost")
        if (changedCount == 0) break
        updateCentroidsFromAssignments(groups, groupAssignments, centroids)
    }

    val clusterSizes = IntArray(centroidCount)
    for (centroidID in groupAssignments) {
        clusterSizes[centroidID]++
    }
    require(clusterSizes.all { it == clusterGroupCapacity })
    val clusterMean = clusterSizes.average()
    val clusterStddev = sqrt(clusterSizes.sumOf { (it - clusterMean).pow(2) } / clusterSizes.size)
    println(
        "texture=$textureIndex iterations=$iterationCount finalCost=$finalCost " +
            "groups min=${clusterSizes.min()} max=${clusterSizes.max()} mean=$clusterMean stddev=$clusterStddev " +
            "pixels min=${clusterSizes.min() * groupPixelCount} max=${clusterSizes.max() * groupPixelCount}"
    )

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
    val data = main(baseRandom, texIndex)
    require(data.size == groupCount)
    require(data.all { it.size == groupPixelCount * 2 })

    for (group in data) {
        for (a in 0..<groupPixelCount) {
            for (b in a + 1..<groupPixelCount) {
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
    val outputData = ByteArray(data.size * groupPixelCount * 2)
    for (i in data.indices) {
        val groupData = data[i]
        val outputBase = i * groupPixelCount * 2
        repeat(groupPixelCount * 2) { outputData[outputBase + it] = groupData[it].toByte() }
    }
    require(outputData.size == 131072)
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
