import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.util.concurrent.Callable
import java.util.concurrent.ExecutorCompletionService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

val WIDTH = 128
val HEIGHT = 128
val DEPTH = 64
val CHANNELS = 3
val SLICE_SAMPLES = WIDTH * HEIGHT
val TOTAL_SAMPLES = SLICE_SAMPLES * DEPTH
val UNBALANCED_MAX_ITERATIONS = 12
val BALANCED_MAX_ITERATIONS = 64
val DEFAULT_QUANTIZATION = 1_000_000_000L
val AUCTION_EPSILON_SCALE = 5L
val CENTER_EPSILON = 1e-15
val CAP_CONTAINMENT_TOLERANCE = 2e-12
val OBJECTIVE_RELATIVE_TOLERANCE = 1e-10
val RAD_TO_DEG = 180.0 / PI
val ID_BITS = 14
val ID_MASK = (1L shl ID_BITS) - 1L
val MULTISTART_CHECKPOINT_MAGIC = 0x5354424E
val MULTISTART_CHECKPOINT_VERSION = 1
val MULTISTART_SEED_BASE = 0x5342544E4D554C54L

data class InputStats(
    val encodedLengthMin: Double,
    val encodedLengthMean: Double,
    val encodedLengthMax: Double,
    val encodedLengthMaxAbsDeviation: Double,
    val decodedZMin: Double,
    val decodedZMax: Double
)

data class DecodedTexture(
    val directions: DoubleArray,
    val originalIds: IntArray,
    val stats: InputStats
)

data class AlgorithmCounts(
    val hilbertKeyEvaluations: Long = 0,
    val initializationDotEvaluations: Long = 0,
    val assignmentDotEvaluations: Long = 0,
    val lloydIterations: Long = 0,
    val centerUpdates: Long = 0,
    val emptyClusterRepairs: Long = 0
) {
    operator fun plus(other: AlgorithmCounts) = AlgorithmCounts(
        hilbertKeyEvaluations + other.hilbertKeyEvaluations,
        initializationDotEvaluations + other.initializationDotEvaluations,
        assignmentDotEvaluations + other.assignmentDotEvaluations,
        lloydIterations + other.lloydIterations,
        centerUpdates + other.centerUpdates,
        emptyClusterRepairs + other.emptyClusterRepairs
    )
}

data class SliceClustering(
    val labels: IntArray,
    val destinationToSource: IntArray,
    val counts: AlgorithmCounts
)

data class ClusterGeometry(
    val counts: IntArray,
    val centers: DoubleArray,
    val explicitSquaredChord: Double,
    val eliminatedSquaredChord: Double,
    val pairwiseSquaredChord: Double
)

data class SphericalCap(
    val x: Double,
    val y: Double,
    val z: Double,
    val cosRadius: Double,
    val supportSize: Int
)

data class CapResult(
    val radiusDeg: Double,
    val supportSize: Int,
    val containmentTests: Long,
    val maxCosViolation: Double,
    val center: DoubleArray,
    val cosRadius: Double
)

data class AngularCenterResult(
    val center: DoubleArray,
    val objective: Double,
    val iterations: Int,
    val status: String,
    val converged: Boolean
)

data class AngularGeometry(
    val counts: IntArray,
    val centers: DoubleArray,
    val explicitSquaredAngle: Double,
    val centerIterations: Int,
    val centerStatusCounts: Map<String, Int>,
    val converged: Boolean
)

data class AngularLloydResult(
    val labels: IntArray,
    val geometry: AngularGeometry,
    val status: String,
    val converged: Boolean,
    val iterations: Int,
    val history: List<Map<String, Any?>>,
    val transport: TransportStats
)

data class RadiusLloydResult(
    val labels: IntArray,
    val targetRadiusDeg: Double,
    val maximumRadiusDeg: Double,
    val chordObjective: Double,
    val status: String,
    val feasible: Boolean,
    val iterations: Int,
    val history: List<Map<String, Any?>>,
    val transport: TransportStats
)

data class ObjectiveScores(
    val chord: Double,
    val angular2: Double,
    val p99AngleDeg: Double,
    val maximumCapRadiusDeg: Double
)

data class ExportedCandidate(
    val clusterCount: Int,
    val capacity: Int,
    val objective: Double,
    val manifestRelativePath: String,
    val manifestSha256: String,
    val labelsRelativePath: String,
    val labelsSha256: String,
    val indirectionRelativePath: String,
    val indirectionSha256: String,
    val indirectionBytes: ByteArray
)

data class TimedCase(
    val id: String,
    val wallTimeNanos: Long
)

data class CenterInitialization(
    val centers: DoubleArray,
    val dotEvaluations: Long
)

data class SearchInitialization(
    val family: String,
    val seed: Long,
    val centers: DoubleArray,
    val balancedLabels: IntArray?,
    val dotEvaluations: Long
)

data class StartSpec(
    val ordinal: Int,
    val family: String,
    val familyIndex: Int,
    val seed: Long
)

data class StartWork(
    val slice: Int,
    val spec: StartSpec
)

data class StartPhaseResult(
    val outcomesBySlice: List<List<MultistartOutcome>>,
    val complete: Boolean,
    val loadedCheckpoints: Int,
    val computedStarts: Int
)

data class TransportStats(
    val dotCostEvaluations: Long,
    val priceCandidateEvaluations: Long,
    val bids: Long,
    val evictions: Long,
    val scalingPhases: Long,
    val certificateCostEvaluations: Long
) {
    operator fun plus(other: TransportStats) = TransportStats(
        dotCostEvaluations + other.dotCostEvaluations,
        priceCandidateEvaluations + other.priceCandidateEvaluations,
        bids + other.bids,
        evictions + other.evictions,
        scalingPhases + other.scalingPhases,
        certificateCostEvaluations + other.certificateCostEvaluations
    )
}

data class TransportResult(
    val status: String,
    val labels: IntArray,
    val quantizedCost: Long,
    val proofScale: Long,
    val scaledPrimalCost: Long?,
    val scaledDualLowerBound: Long?,
    val certificateGap: Long?,
    val stats: TransportStats
)

data class LloydIteration(
    val index: Int,
    val labelsSha256: String,
    val transportStatus: String,
    val quantizedAssignmentCost: Long,
    val transportProofScale: Long,
    val transportCertificateGap: Long,
    val fixedCenterFloat64Loss: Double,
    val previousFixedCenterFloat64Loss: Double?,
    val fixedCenterQuantizationBound: Double?,
    val updatedCenterFloat64Loss: Double,
    val deltaFromAccepted: Double?,
    val accepted: Boolean,
    val transport: TransportStats
)

data class BalancedLloydResult(
    val labels: IntArray,
    val objective: Double,
    val status: String,
    val converged: Boolean,
    val iterations: List<LloydIteration>,
    val finalCheckQuantizedCost: Long,
    val finalCheckCertificateGap: Long,
    val incumbentQuantizedCost: Long,
    val finalCheckLabelsMatch: Boolean,
    val finalCheckFloat64Gap: Double,
    val quantizationBound: Double,
    val initializationDotEvaluations: Long,
    val firstAssignmentPartitionSha256: String,
    val transport: TransportStats
)

data class MultistartOutcome(
    val clusterCount: Int,
    val slice: Int,
    val ordinal: Int,
    val family: String,
    val familyIndex: Int,
    val seed: Long,
    val labels: IntArray,
    val partitionSha256: String,
    val objective: Double,
    val status: String,
    val converged: Boolean,
    val iterationCount: Int,
    val acceptedIterations: Int,
    val initializerPartitionObjective: Double?,
    val firstAssignmentObjective: Double,
    val initializationPartitionSha256: String?,
    val firstAssignmentPartitionSha256: String,
    val finalCheckQuantizedCost: Long,
    val finalCheckCertificateGap: Long,
    val incumbentQuantizedCost: Long,
    val finalCheckLabelsMatch: Boolean,
    val finalCheckFloat64Gap: Double,
    val quantizationBound: Double,
    val initializationDotEvaluations: Long,
    val transport: TransportStats
)

data class SwapScanResult(
    val firstSample: Int,
    val secondSample: Int,
    val delta: Double,
    val pairsEvaluated: Long
)

data class LocalRefinementResult(
    val labels: IntArray,
    val initialObjective: Double,
    val finalObjective: Double,
    val screenedSwapMoves: Int,
    val screenedSwapGain: Double,
    val pairMoves: Int,
    val pairGain: Double,
    val pairCandidates: Long,
    val reconvergenceStatus: String,
    val reconvergenceAccepted: Boolean,
    val reconvergenceGain: Double,
    val certificateSwapMoves: Int,
    val certificateGain: Double,
    val fullSweepCount: Int,
    val fullSweepPairs: Long,
    val certificateTolerance: Double,
    val certified: Boolean,
    val history: List<Map<String, Any?>>
)

data class PooledNeighborhoodResult(
    val labels: IntArray,
    val objective: Double,
    val status: String,
    val converged: Boolean,
    val starts: List<Map<String, Any?>>
)

data class NeighborhoodStageResult(
    val labels: IntArray,
    val initialObjective: Double,
    val finalObjective: Double,
    val neighborhoodSize: Int,
    val attemptBudget: Int,
    val acceptedAttempts: Int,
    val attempts: List<Map<String, Any?>>,
    val measurements: List<Map<String, Any?>>,
    val acceptedCheckpoints: List<IntArray>,
    val reconvergenceStatus: String,
    val reconvergenceAccepted: Boolean
)

data class ExhaustiveQuantizedResult(
    val labels: IntArray,
    val cost: Long
)

data class ExhaustiveFloat64Result(
    val labels: IntArray,
    val loss: Double
)

class PeakHeapSampler : AutoCloseable {
    private val running = AtomicBoolean(true)
    private val peak = AtomicLong(0)
    private val thread = Thread {
        while (running.get()) {
            observe()
            Thread.sleep(5)
        }
        observe()
    }.apply {
        isDaemon = true
        name = "stbn-clustering-heap-sampler"
        start()
    }

    private fun observe() {
        val runtime = Runtime.getRuntime()
        val used = runtime.totalMemory() - runtime.freeMemory()
        while (true) {
            val previous = peak.get()
            if (used <= previous || peak.compareAndSet(previous, used)) return
        }
    }

    fun peakBytes(): Long {
        observe()
        return peak.get()
    }

    override fun close() {
        running.set(false)
        thread.join()
    }
}

fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
    .digest(bytes)
    .joinToString("") { "%02X".format(it.toInt() and 0xFF) }

fun normalizePath(path: Path): String = path.toString().replace('\\', '/')

fun decodeRgb8(
    bytes: ByteArray,
    width: Int,
    height: Int,
    depth: Int,
    requireUpperHemisphere: Boolean = true
): DecodedTexture {
    require(width > 0 && height > 0 && depth > 0) { "Texture dimensions must be positive" }
    val sampleCountLong = width.toLong() * height * depth
    val expectedBytesLong = sampleCountLong * CHANNELS
    require(expectedBytesLong <= Int.MAX_VALUE) { "Texture dimensions exceed JVM array limits" }
    require(bytes.size == expectedBytesLong.toInt()) {
        "Expected ${expectedBytesLong} RGB8 bytes for ${width}x${height}x${depth}, found ${bytes.size}"
    }

    val sampleCount = sampleCountLong.toInt()
    val sliceSamples = width * height
    val directions = DoubleArray(sampleCount * CHANNELS)
    val originalIds = IntArray(sampleCount) { it % sliceSamples }
    var encodedLengthMin = Double.POSITIVE_INFINITY
    var encodedLengthMax = Double.NEGATIVE_INFINITY
    var encodedLengthSum = 0.0
    var decodedZMin = Double.POSITIVE_INFINITY
    var decodedZMax = Double.NEGATIVE_INFINITY

    for (sample in 0 until sampleCount) {
        val offset = sample * CHANNELS
        val x = ((bytes[offset].toInt() and 0xFF) * (2.0 / 255.0)) - 1.0
        val y = ((bytes[offset + 1].toInt() and 0xFF) * (2.0 / 255.0)) - 1.0
        val z = ((bytes[offset + 2].toInt() and 0xFF) * (2.0 / 255.0)) - 1.0
        val length = sqrt(x * x + y * y + z * z)
        require(length.isFinite() && length > CENTER_EPSILON) { "Sample $sample decodes to a non-finite or zero vector" }
        val nx = x / length
        val ny = y / length
        val nz = z / length
        require(nx.isFinite() && ny.isFinite() && nz.isFinite()) { "Sample $sample normalization is non-finite" }
        if (requireUpperHemisphere) require(nz >= 0.0) { "Sample $sample decodes below the cosine hemisphere: z=$nz" }
        directions[offset] = nx
        directions[offset + 1] = ny
        directions[offset + 2] = nz
        encodedLengthMin = min(encodedLengthMin, length)
        encodedLengthMax = max(encodedLengthMax, length)
        encodedLengthSum += length
        decodedZMin = min(decodedZMin, nz)
        decodedZMax = max(decodedZMax, nz)
    }

    return DecodedTexture(
        directions,
        originalIds,
        InputStats(
            encodedLengthMin,
            encodedLengthSum / sampleCount,
            encodedLengthMax,
            max(abs(encodedLengthMin - 1.0), abs(encodedLengthMax - 1.0)),
            decodedZMin,
            decodedZMax
        )
    )
}

fun validatePermutation(permutation: IntArray, size: Int) {
    require(permutation.size == size) { "Expected permutation size $size, found ${permutation.size}" }
    val seen = BooleanArray(size)
    for ((destination, source) in permutation.withIndex()) {
        require(source in 0 until size) { "Destination $destination references out-of-range source $source" }
        require(!seen[source]) { "Destination $destination duplicates source $source" }
        seen[source] = true
    }
}

fun mortonEncode8(x: Int, y: Int): Int {
    var sx = x and 15
    var sy = y and 15
    sx = (sx or (sx shl 2)) and 0x33
    sy = (sy or (sy shl 2)) and 0x33
    sx = (sx or (sx shl 1)) and 0x55
    sy = (sy or (sy shl 1)) and 0x55
    return sx or (sy shl 1)
}

fun clusterForDestination(destination: Int, clusterCount: Int): Int {
    require(clusterCount == 128 || clusterCount == 256)
    val x = destination % WIDTH
    val y = destination / WIDTH
    val tile = (y shr 4) * (WIDTH shr 4) + (x shr 4)
    val lane = mortonEncode8(x and 15, y and 15)
    val capacity = SLICE_SAMPLES / clusterCount
    return tile * (256 / capacity) + lane / capacity
}

fun destinationsByCluster(clusterCount: Int): IntArray {
    val keys = LongArray(SLICE_SAMPLES)
    for (destination in 0 until SLICE_SAMPLES) {
        val x = destination % WIDTH
        val y = destination / WIDTH
        val lane = mortonEncode8(x and 15, y and 15)
        val cluster = clusterForDestination(destination, clusterCount)
        keys[destination] = (cluster.toLong() shl 22) or (lane.toLong() shl ID_BITS) or destination.toLong()
    }
    keys.sort()
    return IntArray(SLICE_SAMPLES) { (keys[it] and ID_MASK).toInt() }
}

fun hilbertIndex(xInput: Int, yInput: Int, bits: Int): Long {
    require(bits in 1..16)
    val limit = 1 shl bits
    require(xInput in 0 until limit && yInput in 0 until limit)
    var x = xInput
    var y = yInput
    var result = 0L
    var scale = limit shr 1
    while (scale > 0) {
        val rx = if ((x and scale) != 0) 1 else 0
        val ry = if ((y and scale) != 0) 1 else 0
        result += scale.toLong() * scale * ((3 * rx) xor ry)
        if (ry == 0) {
            if (rx == 1) {
                x = scale - 1 - x
                y = scale - 1 - y
            }
            val swap = x
            x = y
            y = swap
        }
        scale = scale shr 1
    }
    return result
}

fun hilbertIndexForVector(x: Double, y: Double, z: Double): Long {
    val denominator = abs(x) + abs(y) + abs(z)
    require(denominator > CENTER_EPSILON)
    var ox = x / denominator
    var oy = y / denominator
    if (z < 0.0) {
        val previousX = ox
        ox = (1.0 - abs(oy)) * if (previousX < 0.0) -1.0 else 1.0
        oy = (1.0 - abs(previousX)) * if (oy < 0.0) -1.0 else 1.0
    }
    val qx = ((ox * 0.5 + 0.5) * 65535.0).roundToInt().coerceIn(0, 65535)
    val qy = ((oy * 0.5 + 0.5) * 65535.0).roundToInt().coerceIn(0, 65535)
    return hilbertIndex(qx, qy, 16)
}

fun directionKey(directions: DoubleArray, vectorOffset: Int, originalId: Int): Long {
    val index = hilbertIndexForVector(
        directions[vectorOffset],
        directions[vectorOffset + 1],
        directions[vectorOffset + 2]
    )
    return (index shl ID_BITS) or originalId.toLong()
}

fun identityClustering(clusterCount: Int): SliceClustering {
    val labels = IntArray(SLICE_SAMPLES) { clusterForDestination(it, clusterCount) }
    val permutation = IntArray(SLICE_SAMPLES) { it }
    return SliceClustering(labels, permutation, AlgorithmCounts())
}

fun hilbertClustering(directions: DoubleArray, slice: Int, clusterCount: Int): SliceClustering {
    val vectorBase = slice * SLICE_SAMPLES * CHANNELS
    val sourceKeys = LongArray(SLICE_SAMPLES)
    for (source in 0 until SLICE_SAMPLES) {
        sourceKeys[source] = directionKey(directions, vectorBase + source * CHANNELS, source)
    }
    sourceKeys.sort()
    val destinations = destinationsByCluster(clusterCount)
    val capacity = SLICE_SAMPLES / clusterCount
    val labels = IntArray(SLICE_SAMPLES)
    val permutation = IntArray(SLICE_SAMPLES)
    for (order in 0 until SLICE_SAMPLES) {
        val source = (sourceKeys[order] and ID_MASK).toInt()
        val destination = destinations[order]
        val cluster = order / capacity
        labels[source] = cluster
        permutation[destination] = source
    }
    validatePermutation(permutation, SLICE_SAMPLES)
    return SliceClustering(
        labels,
        permutation,
        AlgorithmCounts(hilbertKeyEvaluations = SLICE_SAMPLES.toLong())
    )
}

fun nearestCenter(
    x: Double,
    y: Double,
    z: Double,
    centers: DoubleArray,
    centerCount: Int
): Pair<Int, Double> {
    var bestCenter = 0
    var bestDot = x * centers[0] + y * centers[1] + z * centers[2]
    for (center in 1 until centerCount) {
        val offset = center * CHANNELS
        val dot = x * centers[offset] + y * centers[offset + 1] + z * centers[offset + 2]
        if (dot > bestDot) {
            bestCenter = center
            bestDot = dot
        }
    }
    return bestCenter to bestDot
}

fun farthestFirstCenters(
    directions: DoubleArray,
    vectorBase: Int,
    sampleCount: Int,
    clusterCount: Int
): CenterInitialization {
    require(clusterCount in 1..sampleCount)
    val centers = DoubleArray(clusterCount * CHANNELS)
    val nearestDots = DoubleArray(sampleCount)
    var first = 0
    var firstZ = Double.NEGATIVE_INFINITY
    for (sample in 0 until sampleCount) {
        val z = directions[vectorBase + sample * CHANNELS + 2]
        if (z > firstZ) {
            first = sample
            firstZ = z
        }
    }
    directions.copyInto(centers, 0, vectorBase + first * CHANNELS, vectorBase + first * CHANNELS + CHANNELS)
    var dotEvaluations = 0L
    for (sample in 0 until sampleCount) {
        val offset = vectorBase + sample * CHANNELS
        nearestDots[sample] = directions[offset] * centers[0] +
            directions[offset + 1] * centers[1] +
            directions[offset + 2] * centers[2]
        dotEvaluations++
    }
    for (center in 1 until clusterCount) {
        var farthest = 0
        var farthestDot = nearestDots[0]
        for (sample in 1 until sampleCount) {
            val dot = nearestDots[sample]
            if (dot < farthestDot) {
                farthest = sample
                farthestDot = dot
            }
        }
        val centerOffset = center * CHANNELS
        val sourceOffset = vectorBase + farthest * CHANNELS
        centers[centerOffset] = directions[sourceOffset]
        centers[centerOffset + 1] = directions[sourceOffset + 1]
        centers[centerOffset + 2] = directions[sourceOffset + 2]
        for (sample in 0 until sampleCount) {
            val offset = vectorBase + sample * CHANNELS
            val dot = directions[offset] * centers[centerOffset] +
                directions[offset + 1] * centers[centerOffset + 1] +
                directions[offset + 2] * centers[centerOffset + 2]
            if (dot > nearestDots[sample]) nearestDots[sample] = dot
            dotEvaluations++
        }
    }
    return CenterInitialization(centers, dotEvaluations)
}

fun unbalancedSphericalClustering(
    directions: DoubleArray,
    slice: Int,
    clusterCount: Int
): SliceClustering {
    val vectorBase = slice * SLICE_SAMPLES * CHANNELS
    val initialization = farthestFirstCenters(directions, vectorBase, SLICE_SAMPLES, clusterCount)
    val centers = initialization.centers

    var labels = IntArray(SLICE_SAMPLES) { -1 }
    var assignmentDotEvaluations = 0L
    var iterations = 0
    var centerUpdates = 0L
    var emptyRepairs = 0L
    while (iterations < UNBALANCED_MAX_ITERATIONS) {
        iterations++
        val nextLabels = IntArray(SLICE_SAMPLES)
        val assignedDots = DoubleArray(SLICE_SAMPLES)
        val counts = IntArray(clusterCount)
        for (sample in 0 until SLICE_SAMPLES) {
            val offset = vectorBase + sample * CHANNELS
            val nearest = nearestCenter(
                directions[offset],
                directions[offset + 1],
                directions[offset + 2],
                centers,
                clusterCount
            )
            nextLabels[sample] = nearest.first
            assignedDots[sample] = nearest.second
            counts[nearest.first]++
            assignmentDotEvaluations += clusterCount
        }

        for (empty in 0 until clusterCount) {
            if (counts[empty] != 0) continue
            var donorSample = -1
            var donorLoss = Double.NEGATIVE_INFINITY
            for (sample in 0 until SLICE_SAMPLES) {
                val donor = nextLabels[sample]
                if (counts[donor] <= 1) continue
                val loss = 1.0 - assignedDots[sample]
                if (loss > donorLoss) {
                    donorSample = sample
                    donorLoss = loss
                }
            }
            require(donorSample >= 0) { "Unable to repair empty cluster $empty" }
            counts[nextLabels[donorSample]]--
            nextLabels[donorSample] = empty
            counts[empty]++
            emptyRepairs++
        }

        var changed = 0
        for (sample in 0 until SLICE_SAMPLES) if (labels[sample] != nextLabels[sample]) changed++
        labels = nextLabels
        val sums = DoubleArray(clusterCount * CHANNELS)
        for (sample in 0 until SLICE_SAMPLES) {
            val centerOffset = labels[sample] * CHANNELS
            val offset = vectorBase + sample * CHANNELS
            sums[centerOffset] += directions[offset]
            sums[centerOffset + 1] += directions[offset + 1]
            sums[centerOffset + 2] += directions[offset + 2]
        }
        for (center in 0 until clusterCount) {
            val offset = center * CHANNELS
            val length = sqrt(
                sums[offset] * sums[offset] +
                    sums[offset + 1] * sums[offset + 1] +
                    sums[offset + 2] * sums[offset + 2]
            )
            require(length > CENTER_EPSILON) { "Cluster $center has a degenerate center sum" }
            centers[offset] = sums[offset] / length
            centers[offset + 1] = sums[offset + 1] / length
            centers[offset + 2] = sums[offset + 2] / length
            centerUpdates++
        }
        if (changed == 0) break
    }

    val clusterKeys = LongArray(clusterCount)
    for (cluster in 0 until clusterCount) {
        val offset = cluster * CHANNELS
        val key = hilbertIndexForVector(centers[offset], centers[offset + 1], centers[offset + 2])
        clusterKeys[cluster] = (key shl 9) or cluster.toLong()
    }
    clusterKeys.sort()
    val clusterRank = IntArray(clusterCount)
    for (rank in 0 until clusterCount) clusterRank[(clusterKeys[rank] and 0x1FFL).toInt()] = rank
    val sourceKeys = LongArray(SLICE_SAMPLES)
    for (source in 0 until SLICE_SAMPLES) {
        val offset = vectorBase + source * CHANNELS
        val directionOrder = hilbertIndexForVector(directions[offset], directions[offset + 1], directions[offset + 2])
        sourceKeys[source] = (clusterRank[labels[source]].toLong() shl 46) or
            (directionOrder shl ID_BITS) or source.toLong()
    }
    sourceKeys.sort()
    val permutation = IntArray(SLICE_SAMPLES) { (sourceKeys[it] and ID_MASK).toInt() }
    validatePermutation(permutation, SLICE_SAMPLES)
    return SliceClustering(
        labels,
        permutation,
        AlgorithmCounts(
            hilbertKeyEvaluations = (SLICE_SAMPLES + clusterCount).toLong(),
            initializationDotEvaluations = initialization.dotEvaluations,
            assignmentDotEvaluations = assignmentDotEvaluations,
            lloydIterations = iterations.toLong(),
            centerUpdates = centerUpdates,
            emptyClusterRepairs = emptyRepairs
        )
    )
}

class SlotPriceHeap(
    private val prices: LongArray,
    firstObject: Int,
    capacity: Int
) {
    private val objects = IntArray(capacity) { firstObject + it }

    private fun less(first: Int, second: Int): Boolean =
        prices[first] < prices[second] || (prices[first] == prices[second] && first < second)

    fun first(): Int = objects[0]

    fun second(): Int {
        if (objects.size == 1) return -1
        if (objects.size == 2) return objects[1]
        return if (less(objects[1], objects[2])) objects[1] else objects[2]
    }

    fun increaseFirst(objectId: Int, price: Long) {
        require(objects[0] == objectId)
        require(price > prices[objectId])
        prices[objectId] = price
        var position = 0
        while (true) {
            val left = position * 2 + 1
            if (left >= objects.size) break
            val right = left + 1
            var next = left
            if (right < objects.size && less(objects[right], objects[left])) next = right
            if (!less(objects[next], objects[position])) break
            val swap = objects[position]
            objects[position] = objects[next]
            objects[next] = swap
            position = next
        }
    }
}

fun validateQuantization(sampleCount: Int, quantization: Long) {
    require(sampleCount > 0)
    require(quantization > 0) { "Quantization Q must be positive" }
    val exactDoubleLimit = 1L shl 50
    require(quantization <= exactDoubleLimit) {
        "Quantization Q=$quantization exceeds the exact float64 rounding limit $exactDoubleLimit"
    }
    val safe = try {
        val proofScale = sampleCount.toLong() + 1L
        val maximumScaledEdge = Math.multiplyExact(Math.multiplyExact(4L, quantization), proofScale)
        Math.multiplyExact(maximumScaledEdge, sampleCount.toLong())
        true
    } catch (_: ArithmeticException) {
        false
    }
    require(safe) { "Quantization Q=$quantization is unsafe for the exact $sampleCount-sample certificate" }
}

fun quantizedSquaredChord(dotInput: Double, quantization: Long): Long {
    val dot = dotInput.coerceIn(-1.0, 1.0)
    val scaled = 2.0 * (1.0 - dot) * quantization.toDouble()
    require(scaled.isFinite() && scaled >= 0.0 && scaled <= Long.MAX_VALUE.toDouble())
    return round(scaled).toLong()
}

fun quantizedSquaredAngle(dotInput: Double, quantization: Long): Long {
    val angle = acos(dotInput.coerceIn(-1.0, 1.0))
    val scaled = angle * angle * quantization.toDouble()
    require(scaled.isFinite() && scaled >= 0.0 && scaled <= Long.MAX_VALUE.toDouble())
    return round(scaled).toLong()
}

fun sortedEdgeCostKeys(edgeCosts: LongArray, sampleCount: Int, clusterCount: Int): LongArray {
    require(clusterCount <= 256)
    val result = LongArray(edgeCosts.size)
    for (sample in 0 until sampleCount) {
        val base = sample * clusterCount
        for (cluster in 0 until clusterCount) {
            val cost = edgeCosts[base + cluster]
            require(cost >= 0L && cost <= (Long.MAX_VALUE ushr 8))
            result[base + cluster] = (cost shl 8) or cluster.toLong()
        }
        java.util.Arrays.sort(result, base, base + clusterCount)
    }
    return result
}

fun balancedTransport(
    directions: DoubleArray,
    vectorBase: Int,
    sampleCount: Int,
    centers: DoubleArray,
    clusterCount: Int,
    capacity: Int,
    quantization: Long,
    operationLimit: Long = Long.MAX_VALUE,
    costFunction: ((Double) -> Long)? = null,
    initialLabels: IntArray? = null
): TransportResult {
    require(clusterCount > 0 && capacity > 0)
    require(clusterCount * capacity == sampleCount) {
        "Balanced transport requires clusterCount * capacity == sampleCount"
    }
    require(centers.size == clusterCount * CHANNELS)
    require(vectorBase >= 0 && vectorBase + sampleCount * CHANNELS <= directions.size)
    require(operationLimit >= 0)
    validateQuantization(sampleCount, quantization)
    for (cluster in 0 until clusterCount) {
        val offset = cluster * CHANNELS
        val length = sqrt(
            centers[offset] * centers[offset] +
                centers[offset + 1] * centers[offset + 1] +
                centers[offset + 2] * centers[offset + 2]
        )
        require(length.isFinite() && abs(length - 1.0) <= 1e-10) {
            "Center $cluster is not a finite unit vector: length=$length"
        }
    }

    val proofScale = sampleCount.toLong() + 1L
    val edgeCosts = LongArray(sampleCount * clusterCount)
    var maximumCost = 0L
    if (costFunction == null) {
        for (sample in 0 until sampleCount) {
            val sampleOffset = vectorBase + sample * CHANNELS
            for (cluster in 0 until clusterCount) {
                val centerOffset = cluster * CHANNELS
                val dot = directions[sampleOffset] * centers[centerOffset] +
                    directions[sampleOffset + 1] * centers[centerOffset + 1] +
                    directions[sampleOffset + 2] * centers[centerOffset + 2]
                val cost = Math.multiplyExact(quantizedSquaredChord(dot, quantization), proofScale)
                edgeCosts[sample * clusterCount + cluster] = cost
                maximumCost = max(maximumCost, cost)
            }
        }
    } else {
        for (sample in 0 until sampleCount) {
            val sampleOffset = vectorBase + sample * CHANNELS
            for (cluster in 0 until clusterCount) {
                val centerOffset = cluster * CHANNELS
                val dot = directions[sampleOffset] * centers[centerOffset] +
                    directions[sampleOffset + 1] * centers[centerOffset + 1] +
                    directions[sampleOffset + 2] * centers[centerOffset + 2]
                val unscaledCost = costFunction(dot)
                require(unscaledCost >= 0L) { "Transport edge costs must be nonnegative" }
                val cost = Math.multiplyExact(unscaledCost, proofScale)
                edgeCosts[sample * clusterCount + cluster] = cost
                maximumCost = max(maximumCost, cost)
            }
        }
    }

    val maximumScaledCost = maximumCost
    Math.multiplyExact(maximumScaledCost, sampleCount.toLong())
    val sortedCosts = if (clusterCount <= 256) {
        sortedEdgeCostKeys(edgeCosts, sampleCount, clusterCount)
    } else null
    val prices = LongArray(sampleCount)
    val owners = IntArray(sampleCount) { -1 }
    val sampleObjects = IntArray(sampleCount) { -1 }
    val heaps = Array(clusterCount) { cluster -> SlotPriceHeap(prices, cluster * capacity, capacity) }
    val minimumObjects = IntArray(clusterCount) { it * capacity }
    val minimumPrices = LongArray(clusterCount)
    if (initialLabels != null) {
        require(initialLabels.size == sampleCount)
        val nextObject = IntArray(clusterCount)
        for (sample in 0 until sampleCount) {
            val cluster = initialLabels[sample]
            require(cluster in 0 until clusterCount)
            require(nextObject[cluster] < capacity)
            val objectId = cluster * capacity + nextObject[cluster]++
            require(owners[objectId] < 0)
            owners[objectId] = sample
            sampleObjects[sample] = objectId
        }
        require(nextObject.all { it == capacity })
    }
    var epsilon = 1L
    while (epsilon < maximumScaledCost) epsilon = Math.multiplyExact(epsilon, AUCTION_EPSILON_SCALE)
    var bids = 0L
    var evictions = 0L
    var phases = 0L
    var priceCandidateEvaluations = 0L
    var certificateCostEvaluations = 0L

    fun stats() = TransportStats(
        edgeCosts.size.toLong(),
        priceCandidateEvaluations,
        bids,
        evictions,
        phases,
        certificateCostEvaluations
    )

    while (true) {
        phases++
        val queue = IntArray(sampleCount)
        var queueHead = 0
        var queueTail = 0
        var queueSize = 0
        val maximumPrice = prices.maxOrNull() ?: 0L
        require(maximumPrice <= Long.MAX_VALUE - maximumScaledCost)
        if (phases == 1L && initialLabels == null) {
            owners.fill(-1)
            sampleObjects.fill(-1)
            for (sample in 0 until sampleCount) queue[sample] = sample
            queueSize = sampleCount
        } else {
            // Prices only increase during a phase, so the previous feasible
            // assignment is a valid warm start. Requeue only samples whose
            // assigned object is no longer a best response.
            for (sample in 0 until sampleCount) {
                val currentObject = sampleObjects[sample]
                require(currentObject >= 0)
                val currentCluster = currentObject / capacity
                val row = sample * clusterCount
                if (phases == 1L && initialLabels != null && sortedCosts != null) {
                    val currentCost = edgeCosts[row + currentCluster]
                    val minimumCost = sortedCosts[row] ushr 8
                    priceCandidateEvaluations++
                    if (minimumCost < currentCost) {
                        require(owners[currentObject] == sample)
                        owners[currentObject] = -1
                        sampleObjects[sample] = -1
                        queue[queueTail] = sample
                        queueTail = if (queueTail + 1 == sampleCount) 0 else queueTail + 1
                        queueSize++
                    }
                    continue
                }
                var bestValue = Long.MIN_VALUE
                if (sortedCosts == null) {
                    for (cluster in 0 until clusterCount) {
                        val value = -edgeCosts[row + cluster] - minimumPrices[cluster]
                        if (value > bestValue) bestValue = value
                        priceCandidateEvaluations++
                    }
                } else {
                    var position = 0
                    while (position < clusterCount) {
                        val packed = sortedCosts[row + position]
                        val cluster = (packed and 0xFFL).toInt()
                        val value = -(packed ushr 8) - minimumPrices[cluster]
                        if (value > bestValue) bestValue = value
                        priceCandidateEvaluations++
                        if (position + 1 < clusterCount &&
                            -(sortedCosts[row + position + 1] ushr 8) < bestValue) break
                        position++
                    }
                }
                val currentValue = -edgeCosts[row + currentCluster] - prices[currentObject]
                if (bestValue > currentValue) {
                    require(owners[currentObject] == sample)
                    owners[currentObject] = -1
                    sampleObjects[sample] = -1
                    queue[queueTail] = sample
                    queueTail = if (queueTail + 1 == sampleCount) 0 else queueTail + 1
                    queueSize++
                }
            }
        }
        while (queueSize > 0) {
            if (bids == operationLimit) {
                return TransportResult(
                    "NONTERMINAL_LIMIT",
                    IntArray(sampleCount) { sample ->
                        val objectId = sampleObjects[sample]
                        if (objectId < 0) -1 else objectId / capacity
                    },
                    0L,
                    proofScale,
                    null,
                    null,
                    null,
                    stats()
                )
            }
            val sample = queue[queueHead]
            queueHead = if (queueHead + 1 == sampleCount) 0 else queueHead + 1
            queueSize--
            var bestCluster = -1
            var bestValue = Long.MIN_VALUE
            var secondValue = Long.MIN_VALUE

            val row = sample * clusterCount
            var scannedCandidates = 0
            if (sortedCosts == null) {
                var cluster = 0
                while (cluster < clusterCount) {
                    val value = -edgeCosts[row + cluster] - minimumPrices[cluster]
                    if (value > bestValue || (value == bestValue && cluster < bestCluster)) {
                        secondValue = if (bestValue > secondValue) bestValue else secondValue
                        bestCluster = cluster
                        bestValue = value
                    } else {
                        secondValue = if (value > secondValue) value else secondValue
                    }
                    scannedCandidates++
                    cluster++
                }
            } else {
                var position = 0
                while (position < clusterCount) {
                    val packed = sortedCosts[row + position]
                    val cluster = (packed and 0xFFL).toInt()
                    val value = -(packed ushr 8) - minimumPrices[cluster]
                    if (value > bestValue || (value == bestValue && cluster < bestCluster)) {
                        secondValue = if (bestValue > secondValue) bestValue else secondValue
                        bestCluster = cluster
                        bestValue = value
                    } else {
                        secondValue = if (value > secondValue) value else secondValue
                    }
                    scannedCandidates++
                    if (position + 1 < clusterCount &&
                        -(sortedCosts[row + position + 1] ushr 8) < secondValue) break
                    position++
                }
            }

            priceCandidateEvaluations += scannedCandidates
            require(bestCluster >= 0)
            val bestObject = minimumObjects[bestCluster]
            // The global runner-up is another cluster's minimum or the winner's second slot.
            val secondInBest = heaps[bestCluster].second()
            if (secondInBest >= 0) {
                val value = -edgeCosts[row + bestCluster] - prices[secondInBest]
                priceCandidateEvaluations++
                secondValue = if (value > secondValue) value else secondValue
            }
            val increment = if (sampleCount == 1) epsilon else {
                Math.addExact(Math.subtractExact(bestValue, secondValue), epsilon)
            }
            val bidPrice = Math.addExact(prices[bestObject], increment)
            val previousOwner = owners[bestObject]
            owners[bestObject] = sample
            sampleObjects[sample] = bestObject
            heaps[bestCluster].increaseFirst(bestObject, bidPrice)
            minimumObjects[bestCluster] = heaps[bestCluster].first()
            minimumPrices[bestCluster] = prices[minimumObjects[bestCluster]]
            bids++
            if (previousOwner >= 0) {
                sampleObjects[previousOwner] = -1
                queue[queueTail] = previousOwner
                queueTail = if (queueTail + 1 == sampleCount) 0 else queueTail + 1
                queueSize++
                evictions++
            }
        }

        var quantizedCost = 0L
        var scaledPrimalCost = 0L
        for (sample in 0 until sampleCount) {
            val objectId = sampleObjects[sample]
            require(objectId >= 0)
            val cost = edgeCosts[sample * clusterCount + objectId / capacity]
            quantizedCost = Math.addExact(quantizedCost, cost / proofScale)
            scaledPrimalCost = Math.addExact(scaledPrimalCost, cost)
        }
        // Heap minima are stable for the complete certificate scan. Snapshot
        // them once so the sample x cluster hot loop stays on primitive arrays.
        val certificateMinimumPrices = LongArray(clusterCount) { cluster ->
            prices[heaps[cluster].first()]
        }
        val certificateMinima = LongArray(sampleCount)
        val certificateScans = LongArray(sampleCount)
        val minimumCertificatePrice = certificateMinimumPrices.minOrNull() ?: 0L
        java.util.stream.IntStream.range(0, sampleCount).parallel().forEach { sample ->
            val row = sample * clusterCount
            var minimum = Long.MAX_VALUE
            var scanned = 0L
            if (sortedCosts == null) {
                for (cluster in 0 until clusterCount) {
                    val scaledCost = edgeCosts[row + cluster]
                    val candidate = scaledCost + certificateMinimumPrices[cluster]
                    minimum = if (candidate < minimum) candidate else minimum
                    scanned++
                }
            } else {
                var position = 0
                while (position < clusterCount) {
                    val packed = sortedCosts[row + position]
                    val cluster = (packed and 0xFFL).toInt()
                    val candidate = (packed ushr 8) + certificateMinimumPrices[cluster]
                    minimum = if (candidate < minimum) candidate else minimum
                    scanned++
                    position++
                    if (position < clusterCount) {
                        val nextCost = sortedCosts[row + position] ushr 8
                        if (nextCost >= minimum - minimumCertificatePrice) break
                    }
                }
            }
            certificateMinima[sample] = minimum
            certificateScans[sample] = scanned
        }
        var scaledDualLowerBound = 0L
        for (sample in 0 until sampleCount) {
            scaledDualLowerBound = Math.addExact(scaledDualLowerBound, certificateMinima[sample])
        }
        certificateCostEvaluations += certificateScans.sum()
        for (price in prices) scaledDualLowerBound = Math.subtractExact(scaledDualLowerBound, price)
        val certificateGap = Math.subtractExact(scaledPrimalCost, scaledDualLowerBound)
        require(certificateGap >= 0) { "Dual lower bound exceeds primal cost by ${-certificateGap}" }
        require(certificateGap <= Math.multiplyExact(sampleCount.toLong(), epsilon)) {
            "Auction epsilon certificate failed: gap=$certificateGap N*epsilon=${sampleCount.toLong() * epsilon}"
        }
        if (certificateGap < proofScale) {
            return TransportResult(
                "OPTIMAL",
                IntArray(sampleCount) { sampleObjects[it] / capacity },
                quantizedCost,
                proofScale,
                scaledPrimalCost,
                scaledDualLowerBound,
                certificateGap,
                stats()
            )
        }
        if (epsilon == 1L) {
            return TransportResult(
                "NONTERMINAL_CERTIFICATE",
                IntArray(sampleCount) { sampleObjects[it] / capacity },
                quantizedCost,
                proofScale,
                scaledPrimalCost,
                scaledDualLowerBound,
                certificateGap,
                stats()
            )
        }
        val minimumPrice = prices.minOrNull() ?: 0L
        if (minimumPrice != 0L) for (objectId in prices.indices) prices[objectId] -= minimumPrice
        for (cluster in 0 until clusterCount) minimumPrices[cluster] = prices[minimumObjects[cluster]]
        epsilon = max(1L, epsilon / AUCTION_EPSILON_SCALE)
    }
}

fun requireOptimalTransport(result: TransportResult, sampleCount: Int, clusterCount: Int, capacity: Int) {
    require(result.status == "OPTIMAL") { "Balanced transport ended with ${result.status}" }
    require(result.proofScale == sampleCount.toLong() + 1L)
    require(result.scaledPrimalCost == Math.multiplyExact(result.quantizedCost, result.proofScale))
    require(result.scaledDualLowerBound != null && result.certificateGap != null)
    require(result.certificateGap >= 0L && result.certificateGap < result.proofScale) {
        "Transport lacks an exact integer dual certificate: gap=${result.certificateGap} scale=${result.proofScale}"
    }
    require(result.labels.size == sampleCount)
    val counts = IntArray(clusterCount)
    for ((sample, cluster) in result.labels.withIndex()) {
        require(cluster in 0 until clusterCount) { "Sample $sample has nonintegral/unassigned cluster $cluster" }
        counts[cluster]++
    }
    require(counts.all { it == capacity }) {
        "Balanced transport capacity mismatch: ${counts.minOrNull()}..${counts.maxOrNull()}, expected $capacity"
    }
}

fun fixedCenterFloat64Loss(
    directions: DoubleArray,
    vectorBase: Int,
    labels: IntArray,
    centers: DoubleArray
): Double {
    var result = 0.0
    for (sample in labels.indices) {
        val sampleOffset = vectorBase + sample * CHANNELS
        val centerOffset = labels[sample] * CHANNELS
        val dot = (directions[sampleOffset] * centers[centerOffset] +
            directions[sampleOffset + 1] * centers[centerOffset + 1] +
            directions[sampleOffset + 2] * centers[centerOffset + 2]).coerceIn(-1.0, 1.0)
        result += 2.0 * (1.0 - dot)
    }
    return result
}

fun quantizedCostForLabels(
    directions: DoubleArray,
    vectorBase: Int,
    labels: IntArray,
    centers: DoubleArray,
    quantization: Long
): Long {
    var result = 0L
    for (sample in labels.indices) {
        val sampleOffset = vectorBase + sample * CHANNELS
        val centerOffset = labels[sample] * CHANNELS
        val dot = directions[sampleOffset] * centers[centerOffset] +
            directions[sampleOffset + 1] * centers[centerOffset + 1] +
            directions[sampleOffset + 2] * centers[centerOffset + 2]
        result = Math.addExact(result, quantizedSquaredChord(dot, quantization))
    }
    return result
}

fun labelsBytes(labels: IntArray): ByteArray {
    require(labels.all { it in 0..255 }) { "One-byte labels require values in 0..255" }
    return ByteArray(labels.size) { labels[it].toByte() }
}

fun labelsSha256(labels: IntArray): String = sha256(labelsBytes(labels))

fun forEachBalancedAssignment(
    sampleCount: Int,
    clusterCount: Int,
    capacity: Int,
    action: (IntArray) -> Unit
) {
    require(clusterCount * capacity == sampleCount)
    val labels = IntArray(sampleCount)
    val counts = IntArray(clusterCount)

    fun visit(sample: Int) {
        if (sample == sampleCount) {
            action(labels)
            return
        }
        for (cluster in 0 until clusterCount) {
            if (counts[cluster] == capacity) continue
            labels[sample] = cluster
            counts[cluster]++
            visit(sample + 1)
            counts[cluster]--
        }
    }

    visit(0)
}

fun exhaustiveBalancedQuantized(
    edgeCosts: LongArray,
    sampleCount: Int,
    clusterCount: Int,
    capacity: Int
): ExhaustiveQuantizedResult {
    require(edgeCosts.size == sampleCount * clusterCount)
    var bestCost = Long.MAX_VALUE
    var bestLabels: IntArray? = null
    forEachBalancedAssignment(sampleCount, clusterCount, capacity) { labels ->
        var cost = 0L
        for (sample in 0 until sampleCount) {
            cost = Math.addExact(cost, edgeCosts[sample * clusterCount + labels[sample]])
        }
        if (cost < bestCost) {
            bestCost = cost
            bestLabels = labels.copyOf()
        }
    }
    return ExhaustiveQuantizedResult(bestLabels ?: error("No balanced assignment"), bestCost)
}

fun fixedCenterEdgeCosts(
    directions: DoubleArray,
    vectorBase: Int,
    sampleCount: Int,
    centers: DoubleArray,
    clusterCount: Int,
    quantization: Long
): LongArray {
    val result = LongArray(sampleCount * clusterCount)
    for (sample in 0 until sampleCount) {
        val sampleOffset = vectorBase + sample * CHANNELS
        for (cluster in 0 until clusterCount) {
            val centerOffset = cluster * CHANNELS
            val dot = directions[sampleOffset] * centers[centerOffset] +
                directions[sampleOffset + 1] * centers[centerOffset + 1] +
                directions[sampleOffset + 2] * centers[centerOffset + 2]
            result[sample * clusterCount + cluster] = quantizedSquaredChord(dot, quantization)
        }
    }
    return result
}

fun exhaustiveBalancedFixedFloat64(
    directions: DoubleArray,
    vectorBase: Int,
    sampleCount: Int,
    centers: DoubleArray,
    clusterCount: Int,
    capacity: Int
): ExhaustiveFloat64Result {
    var bestLoss = Double.POSITIVE_INFINITY
    var bestLabels: IntArray? = null
    forEachBalancedAssignment(sampleCount, clusterCount, capacity) { labels ->
        val loss = fixedCenterFloat64Loss(directions, vectorBase, labels, centers)
        if (loss < bestLoss) {
            bestLoss = loss
            bestLabels = labels.copyOf()
        }
    }
    return ExhaustiveFloat64Result(bestLabels ?: error("No balanced assignment"), bestLoss)
}

fun exhaustiveBalancedJointFloat64(
    directions: DoubleArray,
    vectorBase: Int,
    sampleCount: Int,
    clusterCount: Int,
    capacity: Int
): ExhaustiveFloat64Result {
    var bestLoss = Double.POSITIVE_INFINITY
    var bestLabels: IntArray? = null
    forEachBalancedAssignment(sampleCount, clusterCount, capacity) { labels ->
        val loss = computeClusterGeometry(directions, vectorBase, labels, clusterCount).explicitSquaredChord
        if (loss < bestLoss) {
            bestLoss = loss
            bestLabels = labels.copyOf()
        }
    }
    return ExhaustiveFloat64Result(bestLabels ?: error("No balanced partition"), bestLoss)
}

fun computeClusterGeometry(
    directions: DoubleArray,
    vectorBase: Int,
    labels: IntArray,
    clusterCount: Int
): ClusterGeometry {
    val counts = IntArray(clusterCount)
    val sums = DoubleArray(clusterCount * CHANNELS)
    for (sample in labels.indices) {
        val cluster = labels[sample]
        require(cluster in 0 until clusterCount) { "Sample $sample has invalid cluster $cluster" }
        val centerOffset = cluster * CHANNELS
        val offset = vectorBase + sample * CHANNELS
        counts[cluster]++
        sums[centerOffset] += directions[offset]
        sums[centerOffset + 1] += directions[offset + 1]
        sums[centerOffset + 2] += directions[offset + 2]
    }
    val centers = DoubleArray(clusterCount * CHANNELS)
    var eliminated = 0.0
    var pairwise = 0.0
    for (cluster in 0 until clusterCount) {
        require(counts[cluster] > 0) { "Cluster $cluster is empty" }
        val offset = cluster * CHANNELS
        val lengthSquared = sums[offset] * sums[offset] +
            sums[offset + 1] * sums[offset + 1] +
            sums[offset + 2] * sums[offset + 2]
        val length = sqrt(lengthSquared)
        require(length > CENTER_EPSILON) { "Cluster $cluster has a degenerate center sum" }
        centers[offset] = sums[offset] / length
        centers[offset + 1] = sums[offset + 1] / length
        centers[offset + 2] = sums[offset + 2] / length
        eliminated += 2.0 * (counts[cluster] - length)
        pairwise += counts[cluster].toDouble() * counts[cluster] - lengthSquared
    }
    var explicit = 0.0
    for (sample in labels.indices) {
        val offset = vectorBase + sample * CHANNELS
        val centerOffset = labels[sample] * CHANNELS
        val dot = (directions[offset] * centers[centerOffset] +
            directions[offset + 1] * centers[centerOffset + 1] +
            directions[offset + 2] * centers[centerOffset + 2]).coerceIn(-1.0, 1.0)
        explicit += 2.0 * (1.0 - dot)
    }
    return ClusterGeometry(counts, centers, explicit, eliminated, pairwise)
}

fun mix64(value: Long): Long {
    var mixed = value
    mixed = (mixed xor (mixed ushr 30)) * -4658895280553007687L
    mixed = (mixed xor (mixed ushr 27)) * -7723592293110705685L
    return mixed xor (mixed ushr 31)
}

class DeterministicRandom(seed: Long) {
    private var state = seed

    fun nextLong(): Long {
        state += -7046029254386353131L
        return mix64(state)
    }

    fun nextDouble(): Double = (nextLong() ushr 11).toDouble() / 9007199254740992.0

    fun nextInt(bound: Int): Int {
        require(bound > 0)
        return ((nextLong() ushr 1) % bound.toLong()).toInt()
    }
}

fun multistartSeed(family: String, familyIndex: Int): Long {
    require(familyIndex >= 0)
    var familyHash = 0xcbf29ce484222325UL.toLong()
    for (character in family) {
        familyHash = (familyHash xor character.code.toLong()) * 0x100000001b3L
    }
    return mix64(MULTISTART_SEED_BASE xor familyHash xor familyIndex.toLong())
}

fun unsignedHex(value: Long): String = "0x" + java.lang.Long.toUnsignedString(value, 16).uppercase()

fun multistartSpecs(startCount: Int): List<StartSpec> {
    require(startCount >= 3) { "Multistart requires at least one start from each initializer family" }
    val families = arrayOf(
        "squared-chord-d2",
        "recursive-balanced-tangent-pca",
        "structured-hilbert"
    )
    val familyCounts = IntArray(families.size)
    return List(startCount) { ordinal ->
        val familyOrdinal = ordinal % families.size
        val family = families[familyOrdinal]
        val familyIndex = familyCounts[familyOrdinal]++
        StartSpec(ordinal, family, familyIndex, multistartSeed(family, familyIndex))
    }
}

fun sortSamplesByScore(samples: IntArray, scores: DoubleArray): IntArray {
    require(samples.all { it in scores.indices && scores[it].isFinite() })
    var source = samples.copyOf()
    var target = IntArray(samples.size)
    var width = 1
    while (width < source.size) {
        var begin = 0
        while (begin < source.size) {
            val middle = min(begin + width, source.size)
            val end = min(begin + width * 2, source.size)
            var left = begin
            var right = middle
            var output = begin
            while (left < middle || right < end) {
                val takeLeft = when {
                    right == end -> true
                    left == middle -> false
                    scores[source[left]] < scores[source[right]] -> true
                    scores[source[left]] > scores[source[right]] -> false
                    else -> source[left] < source[right]
                }
                target[output++] = if (takeLeft) source[left++] else source[right++]
            }
            begin = end
        }
        val swap = source
        source = target
        target = swap
        width *= 2
    }
    return source
}

fun canonicalPartitionLabels(labels: IntArray, clusterCount: Int): IntArray {
    require(clusterCount in 1..256)
    val counts = IntArray(clusterCount)
    for ((sample, cluster) in labels.withIndex()) {
        require(cluster in 0 until clusterCount) { "Sample $sample has invalid cluster $cluster" }
        counts[cluster]++
    }
    require(counts.all { it > 0 }) { "Canonical partitions require nonempty clusters" }
    val members = buildMembers(labels, counts)
    val order = Array(clusterCount) { it }
    order.sortWith { first, second ->
        val firstMembers = members[first]
        val secondMembers = members[second]
        var comparison = firstMembers.size.compareTo(secondMembers.size)
        var index = 0
        while (comparison == 0 && index < firstMembers.size) {
            comparison = firstMembers[index].compareTo(secondMembers[index])
            index++
        }
        comparison
    }
    val rank = IntArray(clusterCount)
    for (index in order.indices) rank[order[index]] = index
    return IntArray(labels.size) { rank[labels[it]] }
}

fun partitionSha256(labels: IntArray, clusterCount: Int): String =
    labelsSha256(canonicalPartitionLabels(labels, clusterCount))

fun normalizedCentersForLabels(
    directions: DoubleArray,
    vectorBase: Int,
    labels: IntArray,
    clusterCount: Int,
    capacity: Int
): DoubleArray {
    val geometry = computeClusterGeometry(directions, vectorBase, labels, clusterCount)
    require(geometry.counts.all { it == capacity }) {
        "Initializer capacities are ${geometry.counts.minOrNull()}..${geometry.counts.maxOrNull()}, expected $capacity"
    }
    return geometry.centers
}

fun squaredChordD2Initialization(
    directions: DoubleArray,
    vectorBase: Int,
    sampleCount: Int,
    clusterCount: Int,
    seed: Long
): SearchInitialization {
    require(clusterCount in 1..sampleCount)
    require(vectorBase >= 0 && vectorBase + sampleCount * CHANNELS <= directions.size)
    val random = DeterministicRandom(seed)
    val centers = DoubleArray(clusterCount * CHANNELS)
    val selected = BooleanArray(sampleCount)
    val nearestLosses = DoubleArray(sampleCount) { Double.POSITIVE_INFINITY }
    var selectedSample = random.nextInt(sampleCount)
    var dotEvaluations = 0L
    for (center in 0 until clusterCount) {
        selected[selectedSample] = true
        val centerOffset = center * CHANNELS
        val sampleOffset = vectorBase + selectedSample * CHANNELS
        directions.copyInto(centers, centerOffset, sampleOffset, sampleOffset + CHANNELS)
        for (sample in 0 until sampleCount) {
            val offset = vectorBase + sample * CHANNELS
            val dot = directions[offset] * centers[centerOffset] +
                directions[offset + 1] * centers[centerOffset + 1] +
                directions[offset + 2] * centers[centerOffset + 2]
            nearestLosses[sample] = min(nearestLosses[sample], 2.0 * (1.0 - dot.coerceIn(-1.0, 1.0)))
            dotEvaluations++
        }
        if (center + 1 == clusterCount) break
        var total = 0.0
        var fallback = -1
        for (sample in 0 until sampleCount) {
            if (selected[sample]) continue
            if (fallback < 0) fallback = sample
            total += nearestLosses[sample]
        }
        require(fallback >= 0)
        if (!(total > 0.0) || !total.isFinite()) {
            selectedSample = fallback
            continue
        }
        val threshold = random.nextDouble() * total
        var cumulative = 0.0
        var weighted = -1
        for (sample in 0 until sampleCount) {
            if (selected[sample]) continue
            cumulative += nearestLosses[sample]
            if (weighted < 0 && cumulative > threshold) weighted = sample
        }
        selectedSample = if (weighted >= 0) weighted else (sampleCount - 1 downTo 0).first { !selected[it] }
    }
    return SearchInitialization("squared-chord-d2", seed, centers, null, dotEvaluations)
}

fun tangentPcaSplit(
    directions: DoubleArray,
    vectorBase: Int,
    members: IntArray,
    seed: Long,
    node: Int,
    firstProjection: DoubleArray,
    secondProjection: DoubleArray,
    scores: DoubleArray
): Pair<IntArray, IntArray> {
    require(members.size >= 2 && members.size % 2 == 0)
    var sumX = 0.0
    var sumY = 0.0
    var sumZ = 0.0
    for (sample in members) {
        val offset = vectorBase + sample * CHANNELS
        sumX += directions[offset]
        sumY += directions[offset + 1]
        sumZ += directions[offset + 2]
    }
    val sumLength = sqrt(sumX * sumX + sumY * sumY + sumZ * sumZ)
    require(sumLength > CENTER_EPSILON) { "Recursive PCA node $node has a degenerate spherical mean" }
    val centerX = sumX / sumLength
    val centerY = sumY / sumLength
    val centerZ = sumZ / sumLength
    val firstAxis = deterministicPerpendicular(centerX, centerY, centerZ)
    val secondAxis = doubleArrayOf(
        centerY * firstAxis[2] - centerZ * firstAxis[1],
        centerZ * firstAxis[0] - centerX * firstAxis[2],
        centerX * firstAxis[1] - centerY * firstAxis[0]
    )
    var meanFirst = 0.0
    var meanSecond = 0.0
    for (sample in members) {
        val offset = vectorBase + sample * CHANNELS
        val first = directions[offset] * firstAxis[0] + directions[offset + 1] * firstAxis[1] +
            directions[offset + 2] * firstAxis[2]
        val second = directions[offset] * secondAxis[0] + directions[offset + 1] * secondAxis[1] +
            directions[offset + 2] * secondAxis[2]
        firstProjection[sample] = first
        secondProjection[sample] = second
        meanFirst += first
        meanSecond += second
    }
    meanFirst /= members.size
    meanSecond /= members.size
    var covarianceFirst = 0.0
    var covarianceCross = 0.0
    var covarianceSecond = 0.0
    for (sample in members) {
        val first = firstProjection[sample] - meanFirst
        val second = secondProjection[sample] - meanSecond
        covarianceFirst += first * first
        covarianceCross += first * second
        covarianceSecond += second * second
    }
    var angle = 0.5 * atan2(2.0 * covarianceCross, covarianceFirst - covarianceSecond)
    if (seed != 0L) {
        val jitterBits = mix64(seed xor node.toLong()) ushr 11
        val jitter = jitterBits.toDouble() / 9007199254740992.0 - 0.5
        angle += jitter * (PI / 4.0)
    }
    val axisFirst = cos(angle)
    val axisSecond = sin(angle)
    for (sample in members) {
        scores[sample] = firstProjection[sample] * axisFirst + secondProjection[sample] * axisSecond
    }
    val sorted = sortSamplesByScore(members, scores)
    val middle = sorted.size / 2
    return sorted.copyOfRange(0, middle) to sorted.copyOfRange(middle, sorted.size)
}

fun recursiveTangentPcaInitialization(
    directions: DoubleArray,
    vectorBase: Int,
    sampleCount: Int,
    clusterCount: Int,
    seed: Long
): SearchInitialization {
    require(clusterCount > 0 && clusterCount and (clusterCount - 1) == 0) {
        "Recursive tangent/PCA requires a power-of-two cluster count"
    }
    require(sampleCount % clusterCount == 0)
    val firstProjection = DoubleArray(sampleCount)
    val secondProjection = DoubleArray(sampleCount)
    val scores = DoubleArray(sampleCount)
    var groups = listOf(IntArray(sampleCount) { it })
    var node = 1
    while (groups.size < clusterCount) {
        val next = ArrayList<IntArray>(groups.size * 2)
        for (members in groups) {
            val split = tangentPcaSplit(
                directions,
                vectorBase,
                members,
                seed,
                node++,
                firstProjection,
                secondProjection,
                scores
            )
            next += split.first
            next += split.second
        }
        groups = next
    }
    val labels = IntArray(sampleCount)
    for ((cluster, members) in groups.withIndex()) for (sample in members) labels[sample] = cluster
    val capacity = sampleCount / clusterCount
    val centers = normalizedCentersForLabels(directions, vectorBase, labels, clusterCount, capacity)
    val dotEvaluations = 2L * sampleCount * Integer.numberOfTrailingZeros(clusterCount)
    return SearchInitialization("recursive-balanced-tangent-pca", seed, centers, labels, dotEvaluations)
}

fun structuredHilbertInitialization(
    directions: DoubleArray,
    vectorBase: Int,
    sampleCount: Int,
    clusterCount: Int,
    seed: Long
): SearchInitialization {
    require(sampleCount <= ID_MASK + 1L)
    require(sampleCount % clusterCount == 0)
    val random = DeterministicRandom(seed)
    val angle = random.nextDouble() * 2.0 * PI
    val cosine = cos(angle)
    val sine = sin(angle)
    val reflect = random.nextInt(2) != 0
    val keys = LongArray(sampleCount)
    for (sample in 0 until sampleCount) {
        val offset = vectorBase + sample * CHANNELS
        val x = directions[offset]
        val y = directions[offset + 1]
        val rotatedX = cosine * x - sine * y
        var rotatedY = sine * x + cosine * y
        if (reflect) rotatedY = -rotatedY
        val key = hilbertIndexForVector(rotatedX, rotatedY, directions[offset + 2])
        keys[sample] = (key shl ID_BITS) or sample.toLong()
    }
    keys.sort()
    val offset = random.nextInt(sampleCount)
    val reverse = random.nextInt(2) != 0
    val capacity = sampleCount / clusterCount
    val labels = IntArray(sampleCount)
    for (order in 0 until sampleCount) {
        val orderedIndex = if (reverse) {
            Math.floorMod(offset - order, sampleCount)
        } else {
            (offset + order) % sampleCount
        }
        val sample = (keys[orderedIndex] and ID_MASK).toInt()
        labels[sample] = order / capacity
    }
    val centers = normalizedCentersForLabels(directions, vectorBase, labels, clusterCount, capacity)
    return SearchInitialization("structured-hilbert", seed, centers, labels, 0L)
}

fun crossKSplitInitialization(
    directions: DoubleArray,
    vectorBase: Int,
    parentLabels: IntArray,
    parentClusterCount: Int,
    seed: Long
): SearchInitialization {
    val sampleCount = parentLabels.size
    val parentGeometry = computeClusterGeometry(directions, vectorBase, parentLabels, parentClusterCount)
    require(parentGeometry.counts.distinct().size == 1 && parentGeometry.counts[0] % 2 == 0) {
        "Cross-K split requires equal even parent capacities"
    }
    val parentMembers = buildMembers(parentLabels, parentGeometry.counts)
    val firstProjection = DoubleArray(sampleCount)
    val secondProjection = DoubleArray(sampleCount)
    val scores = DoubleArray(sampleCount)
    val labels = IntArray(sampleCount)
    for (parent in 0 until parentClusterCount) {
        val split = tangentPcaSplit(
            directions,
            vectorBase,
            parentMembers[parent],
            seed,
            parent + 1,
            firstProjection,
            secondProjection,
            scores
        )
        for (sample in split.first) labels[sample] = parent * 2
        for (sample in split.second) labels[sample] = parent * 2 + 1
    }
    val clusterCount = parentClusterCount * 2
    val capacity = sampleCount / clusterCount
    val geometry = computeClusterGeometry(directions, vectorBase, labels, clusterCount)
    require(geometry.counts.all { it == capacity })
    val tolerance = OBJECTIVE_RELATIVE_TOLERANCE * max(1.0, parentGeometry.explicitSquaredChord)
    require(geometry.explicitSquaredChord <= parentGeometry.explicitSquaredChord + tolerance) {
        "Cross-K split increased objective from ${parentGeometry.explicitSquaredChord} to ${geometry.explicitSquaredChord}"
    }
    return SearchInitialization(
        "cross-k128-pca-split",
        seed,
        geometry.centers,
        labels,
        2L * sampleCount
    )
}

fun initializationForSpec(
    directions: DoubleArray,
    vectorBase: Int,
    sampleCount: Int,
    clusterCount: Int,
    spec: StartSpec
): SearchInitialization = when (spec.family) {
    "squared-chord-d2" -> squaredChordD2Initialization(
        directions,
        vectorBase,
        sampleCount,
        clusterCount,
        spec.seed
    )
    "recursive-balanced-tangent-pca" -> recursiveTangentPcaInitialization(
        directions,
        vectorBase,
        sampleCount,
        clusterCount,
        spec.seed
    )
    "structured-hilbert" -> structuredHilbertInitialization(
        directions,
        vectorBase,
        sampleCount,
        clusterCount,
        spec.seed
    )
    else -> error("Unknown initializer family ${spec.family}")
}

fun balancedPermutation(
    directions: DoubleArray,
    vectorBase: Int,
    labels: IntArray,
    clusterCount: Int
): IntArray {
    require(labels.size == SLICE_SAMPLES)
    val geometry = computeClusterGeometry(directions, vectorBase, labels, clusterCount)
    val capacity = SLICE_SAMPLES / clusterCount
    require(geometry.counts.all { it == capacity })
    val clusterKeys = LongArray(clusterCount)
    for (cluster in 0 until clusterCount) {
        val offset = cluster * CHANNELS
        val key = hilbertIndexForVector(
            geometry.centers[offset],
            geometry.centers[offset + 1],
            geometry.centers[offset + 2]
        )
        clusterKeys[cluster] = (key shl 9) or cluster.toLong()
    }
    clusterKeys.sort()
    val clusterRank = IntArray(clusterCount)
    for (rank in 0 until clusterCount) clusterRank[(clusterKeys[rank] and 0x1FFL).toInt()] = rank
    val sourceKeys = LongArray(SLICE_SAMPLES)
    for (source in labels.indices) {
        val directionOrder = hilbertIndexForVector(
            directions[vectorBase + source * CHANNELS],
            directions[vectorBase + source * CHANNELS + 1],
            directions[vectorBase + source * CHANNELS + 2]
        )
        sourceKeys[source] = (clusterRank[labels[source]].toLong() shl 46) or
            (directionOrder shl ID_BITS) or source.toLong()
    }
    sourceKeys.sort()
    val destinations = destinationsByCluster(clusterCount)
    val permutation = IntArray(SLICE_SAMPLES)
    for (order in sourceKeys.indices) {
        permutation[destinations[order]] = (sourceKeys[order] and ID_MASK).toInt()
    }
    validatePermutation(permutation, SLICE_SAMPLES)
    return permutation
}

fun balancedLloyd(
    directions: DoubleArray,
    vectorBase: Int,
    sampleCount: Int,
    clusterCount: Int,
    capacity: Int,
    quantization: Long,
    maxIterations: Int,
    initialCenters: CenterInitialization? = null
): BalancedLloydResult {
    require(clusterCount * capacity == sampleCount)
    require(maxIterations > 0)
    val initialization = initialCenters ?: farthestFirstCenters(directions, vectorBase, sampleCount, clusterCount)
    require(initialization.centers.size == clusterCount * CHANNELS)
    var centers = initialization.centers
    var acceptedLabels: IntArray? = null
    var acceptedObjective = Double.POSITIVE_INFINITY
    val seenLabels = linkedMapOf<String, IntArray>()
    val iterations = ArrayList<LloydIteration>()
    var transportTotal = TransportStats(0, 0, 0, 0, 0, 0)
    var status = "MAX_ITERATIONS"
    var converged = false
    var firstAssignmentPartitionSha256: String? = null
    val quantizationBound = sampleCount / quantization.toDouble()

    for (iteration in 1..maxIterations) {
        val transport = balancedTransport(
            directions,
            vectorBase,
            sampleCount,
            centers,
            clusterCount,
            capacity,
            quantization,
            initialLabels = acceptedLabels
        )
        requireOptimalTransport(transport, sampleCount, clusterCount, capacity)
        if (firstAssignmentPartitionSha256 == null) {
            firstAssignmentPartitionSha256 = partitionSha256(transport.labels, clusterCount)
        }
        transportTotal += transport.stats
        val fixedCenterLoss = fixedCenterFloat64Loss(directions, vectorBase, transport.labels, centers)
        val previousFixedCenterLoss = acceptedLabels?.let {
            fixedCenterFloat64Loss(directions, vectorBase, it, centers)
        }
        if (previousFixedCenterLoss != null) {
            val tolerance = 1e-12 * max(1.0, previousFixedCenterLoss)
            require(fixedCenterLoss <= previousFixedCenterLoss + quantizationBound + tolerance) {
                "Fixed-center Q bound failed: candidate=$fixedCenterLoss previous=$previousFixedCenterLoss " +
                    "bound=$quantizationBound"
            }
        }
        val geometry = computeClusterGeometry(directions, vectorBase, transport.labels, clusterCount)
        require(geometry.explicitSquaredChord <= fixedCenterLoss + 1e-10 * max(1.0, fixedCenterLoss)) {
            "Center update increased fixed-assignment loss"
        }
        val candidateObjective = geometry.explicitSquaredChord
        val delta = if (acceptedLabels == null) null else candidateObjective - acceptedObjective
        val digest = labelsSha256(transport.labels)
        val labelStable = acceptedLabels?.contentEquals(transport.labels) == true
        val earlier = seenLabels[digest]
        if (earlier != null && !earlier.contentEquals(transport.labels)) {
            error("SHA-256 collision while detecting balanced-Lloyd cycles")
        }
        val repeated = earlier != null
        val monotonic = acceptedLabels == null || candidateObjective <= acceptedObjective
        val accepted = !labelStable && !repeated && monotonic
        iterations += LloydIteration(
            iteration,
            digest,
            transport.status,
            transport.quantizedCost,
            transport.proofScale,
            transport.certificateGap ?: error("Optimal transport lacks a certificate gap"),
            fixedCenterLoss,
            previousFixedCenterLoss,
            if (previousFixedCenterLoss == null) null else quantizationBound,
            candidateObjective,
            delta,
            accepted,
            transport.stats
        )

        if (labelStable) {
            require(abs(candidateObjective - acceptedObjective) <= 1e-12 * max(1.0, acceptedObjective)) {
                "Stable labels changed original objective from $acceptedObjective to $candidateObjective"
            }
            status = "CONVERGED_LABELS_AND_OBJECTIVE"
            converged = true
            break
        }
        if (repeated) {
            status = "CYCLE"
            break
        }
        if (!monotonic) {
            status = "ORIGINAL_OBJECTIVE_INCREASE"
            break
        }
        acceptedLabels = transport.labels.copyOf()
        acceptedObjective = candidateObjective
        centers = geometry.centers
        seenLabels[digest] = acceptedLabels.copyOf()
    }
    val incumbent = acceptedLabels ?: error("Balanced Lloyd produced no accepted assignment")
    val finalGeometry = computeClusterGeometry(directions, vectorBase, incumbent, clusterCount)
    val finalCheck = balancedTransport(
        directions,
        vectorBase,
        sampleCount,
        finalGeometry.centers,
        clusterCount,
        capacity,
        quantization,
        initialLabels = incumbent
    )
    requireOptimalTransport(finalCheck, sampleCount, clusterCount, capacity)
    transportTotal += finalCheck.stats
    val incumbentQuantizedCost = quantizedCostForLabels(
        directions,
        vectorBase,
        incumbent,
        finalGeometry.centers,
        quantization
    )
    val incumbentFixedLoss = fixedCenterFloat64Loss(directions, vectorBase, incumbent, finalGeometry.centers)
    val checkFixedLoss = fixedCenterFloat64Loss(directions, vectorBase, finalCheck.labels, finalGeometry.centers)
    require(finalCheck.quantizedCost <= incumbentQuantizedCost) {
        "Final fixed-center optimum ${finalCheck.quantizedCost} is worse than incumbent $incumbentQuantizedCost"
    }
    require(checkFixedLoss <= incumbentFixedLoss + quantizationBound + 1e-12 * max(1.0, incumbentFixedLoss)) {
        "Final fixed-center float64 gap exceeds N/Q: check=$checkFixedLoss incumbent=$incumbentFixedLoss " +
            "bound=$quantizationBound"
    }
    if (converged) {
        require(finalCheck.quantizedCost == incumbentQuantizedCost) {
            "Converged result failed final fixed-center quantized optimum check"
        }
        require(finalCheck.labels.contentEquals(incumbent)) {
            "Converged result failed deterministic final label check"
        }
        require(abs(checkFixedLoss - incumbentFixedLoss) <= quantizationBound + 1e-12 * max(1.0, incumbentFixedLoss))
    }
    return BalancedLloydResult(
        incumbent,
        finalGeometry.explicitSquaredChord,
        status,
        converged,
        iterations,
        finalCheck.quantizedCost,
        finalCheck.certificateGap ?: error("Optimal final transport lacks a certificate gap"),
        incumbentQuantizedCost,
        finalCheck.labels.contentEquals(incumbent),
        checkFixedLoss - incumbentFixedLoss,
        quantizationBound,
        initialization.dotEvaluations,
        firstAssignmentPartitionSha256 ?: error("Balanced Lloyd produced no first assignment"),
        transportTotal
    )
}

fun dotPointCap(directions: DoubleArray, vectorBase: Int, sample: Int, cap: SphericalCap): Double {
    val offset = vectorBase + sample * CHANNELS
    return directions[offset] * cap.x + directions[offset + 1] * cap.y + directions[offset + 2] * cap.z
}

fun capFromOne(directions: DoubleArray, vectorBase: Int, sample: Int): SphericalCap {
    val offset = vectorBase + sample * CHANNELS
    return SphericalCap(directions[offset], directions[offset + 1], directions[offset + 2], 1.0, 1)
}

fun deterministicPerpendicular(x: Double, y: Double, z: Double): DoubleArray {
    val ax = abs(x)
    val ay = abs(y)
    val az = abs(z)
    val bx: Double
    val by: Double
    val bz: Double
    if (ax <= ay && ax <= az) {
        bx = 1.0
        by = 0.0
        bz = 0.0
    } else if (ay <= az) {
        bx = 0.0
        by = 1.0
        bz = 0.0
    } else {
        bx = 0.0
        by = 0.0
        bz = 1.0
    }
    val cx = y * bz - z * by
    val cy = z * bx - x * bz
    val cz = x * by - y * bx
    val length = sqrt(cx * cx + cy * cy + cz * cz)
    return doubleArrayOf(cx / length, cy / length, cz / length)
}

fun capFromTwo(directions: DoubleArray, vectorBase: Int, first: Int, second: Int): SphericalCap {
    val a = vectorBase + first * CHANNELS
    val b = vectorBase + second * CHANNELS
    var x = directions[a] + directions[b]
    var y = directions[a + 1] + directions[b + 1]
    var z = directions[a + 2] + directions[b + 2]
    var length = sqrt(x * x + y * y + z * z)
    if (length <= CENTER_EPSILON) {
        val perpendicular = deterministicPerpendicular(directions[a], directions[a + 1], directions[a + 2])
        x = perpendicular[0]
        y = perpendicular[1]
        z = perpendicular[2]
        length = 1.0
    }
    x /= length
    y /= length
    z /= length
    val cosRadius = min(
        directions[a] * x + directions[a + 1] * y + directions[a + 2] * z,
        directions[b] * x + directions[b + 1] * y + directions[b + 2] * z
    )
    return SphericalCap(x, y, z, cosRadius.coerceIn(-1.0, 1.0), 2)
}

fun capContains(directions: DoubleArray, vectorBase: Int, sample: Int, cap: SphericalCap): Boolean =
    dotPointCap(directions, vectorBase, sample, cap) >= cap.cosRadius - CAP_CONTAINMENT_TOLERANCE

fun capFromThree(
    directions: DoubleArray,
    vectorBase: Int,
    first: Int,
    second: Int,
    third: Int
): SphericalCap {
    val a = vectorBase + first * CHANNELS
    val b = vectorBase + second * CHANNELS
    val c = vectorBase + third * CHANNELS
    val ux = directions[b] - directions[a]
    val uy = directions[b + 1] - directions[a + 1]
    val uz = directions[b + 2] - directions[a + 2]
    val vx = directions[c] - directions[a]
    val vy = directions[c + 1] - directions[a + 1]
    val vz = directions[c + 2] - directions[a + 2]
    val nx = uy * vz - uz * vy
    val ny = uz * vx - ux * vz
    val nz = ux * vy - uy * vx
    val length = sqrt(nx * nx + ny * ny + nz * nz)
    if (length > CENTER_EPSILON) {
        val x = nx / length
        val y = ny / length
        val z = nz / length
        val dot = directions[a] * x + directions[a + 1] * y + directions[a + 2] * z
        val sign = if (dot >= 0.0) 1.0 else -1.0
        val sx = x * sign
        val sy = y * sign
        val sz = z * sign
        val cosRadius = minOf(
            directions[a] * sx + directions[a + 1] * sy + directions[a + 2] * sz,
            directions[b] * sx + directions[b + 1] * sy + directions[b + 2] * sz,
            directions[c] * sx + directions[c + 1] * sy + directions[c + 2] * sz
        )
        return SphericalCap(sx, sy, sz, cosRadius.coerceIn(-1.0, 1.0), 3)
    }

    val candidates = arrayOf(
        capFromOne(directions, vectorBase, first),
        capFromOne(directions, vectorBase, second),
        capFromOne(directions, vectorBase, third),
        capFromTwo(directions, vectorBase, first, second),
        capFromTwo(directions, vectorBase, first, third),
        capFromTwo(directions, vectorBase, second, third)
    )
    return candidates
        .filter { candidate ->
            capContains(directions, vectorBase, first, candidate) &&
                capContains(directions, vectorBase, second, candidate) &&
                capContains(directions, vectorBase, third, candidate)
        }
        .maxByOrNull { it.cosRadius }
        ?: error("Degenerate three-point cap has no containing support candidate")
}

fun mix32(value: Int): Long {
    var x = value.toLong() and 0xFFFFFFFFL
    x = ((x xor (x ushr 16)) * 0x7FEB352DL) and 0xFFFFFFFFL
    x = ((x xor (x ushr 15)) * 0x846CA68BL) and 0xFFFFFFFFL
    return (x xor (x ushr 16)) and 0xFFFFFFFFL
}

fun minimumSphericalCap(
    directions: DoubleArray,
    vectorBase: Int,
    members: IntArray
): CapResult {
    require(members.isNotEmpty())
    val keys = LongArray(members.size)
    for (index in members.indices) keys[index] = (mix32(members[index]) shl ID_BITS) or members[index].toLong()
    keys.sort()
    val order = IntArray(keys.size) { (keys[it] and ID_MASK).toInt() }
    var cap = capFromOne(directions, vectorBase, order[0])
    var tests = 0L
    for (i in 1 until order.size) {
        tests++
        if (capContains(directions, vectorBase, order[i], cap)) continue
        cap = capFromOne(directions, vectorBase, order[i])
        for (j in 0 until i) {
            tests++
            if (capContains(directions, vectorBase, order[j], cap)) continue
            cap = capFromTwo(directions, vectorBase, order[i], order[j])
            for (k in 0 until j) {
                tests++
                if (capContains(directions, vectorBase, order[k], cap)) continue
                cap = capFromThree(directions, vectorBase, order[i], order[j], order[k])
            }
        }
    }
    var maxViolation = 0.0
    for (sample in members) {
        tests++
        maxViolation = max(maxViolation, cap.cosRadius - dotPointCap(directions, vectorBase, sample, cap))
    }
    require(maxViolation <= 1e-9) { "Minimum-cap solver containment violation $maxViolation" }
    return CapResult(
        acos(cap.cosRadius.coerceIn(-1.0, 1.0)) * RAD_TO_DEG,
        cap.supportSize,
        tests,
        maxViolation,
        doubleArrayOf(cap.x, cap.y, cap.z),
        cap.cosRadius
    )
}

fun quantileFromSorted(sorted: DoubleArray, q: Double): Double {
    require(sorted.isNotEmpty())
    require(q in 0.0..1.0)
    val position = q * (sorted.size - 1)
    val lower = floor(position).toInt()
    val upper = min(lower + 1, sorted.lastIndex)
    val fraction = position - lower
    return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
}

fun distribution(values: DoubleArray): Map<String, Any?> {
    require(values.isNotEmpty())
    var sum = 0.0
    var squaredSum = 0.0
    var minimum = Double.POSITIVE_INFINITY
    var maximum = Double.NEGATIVE_INFINITY
    for (value in values) {
        require(value.isFinite())
        sum += value
        squaredSum += value * value
        minimum = min(minimum, value)
        maximum = max(maximum, value)
    }
    val sorted = values.copyOf().apply { sort() }
    val mean = sum / values.size
    var variance = 0.0
    for (value in values) {
        val delta = value - mean
        variance += delta * delta
    }
    return linkedMapOf(
        "count" to values.size,
        "min" to minimum,
        "mean" to mean,
        "rms" to sqrt(squaredSum / values.size),
        "stddevPopulation" to sqrt(variance / values.size),
        "p50" to quantileFromSorted(sorted, 0.50),
        "p95" to quantileFromSorted(sorted, 0.95),
        "p99" to quantileFromSorted(sorted, 0.99),
        "max" to maximum
    )
}

fun buildMembers(labels: IntArray, counts: IntArray): Array<IntArray> {
    val members = Array(counts.size) { IntArray(counts[it]) }
    val cursors = IntArray(counts.size)
    for (sample in labels.indices) {
        val cluster = labels[sample]
        members[cluster][cursors[cluster]++] = sample
    }
    return members
}

fun caseMetrics(
    texture: DecodedTexture,
    caseId: String,
    algorithm: String,
    placement: String,
    clusterCount: Int,
    slices: List<SliceClustering>,
    sliceIds: IntArray = IntArray(DEPTH) { it }
): Map<String, Any?> {
    require(slices.size == sliceIds.size)
    val sampleTotal = sliceIds.size * SLICE_SAMPLES
    val allAngles = DoubleArray(sampleTotal)
    val allSquaredChord = DoubleArray(sampleTotal)
    val allClusterSizes = DoubleArray(sliceIds.size * clusterCount)
    val allMeanCenterRadii = DoubleArray(sliceIds.size * clusterCount)
    val allMinimumCapRadii = DoubleArray(sliceIds.size * clusterCount)
    val horizonBounds = doubleArrayOf(0.1, 0.25, 0.5)
    val horizonNames = arrayOf("[0,0.1)", "[0.1,0.25)", "[0.25,0.5)", "[0.5,1]")
    val horizonCounts = IntArray(4)
    for (sample in 0 until sampleTotal) {
        val originalSample = sliceIds[sample / SLICE_SAMPLES] * SLICE_SAMPLES + sample % SLICE_SAMPLES
        val z = texture.directions[originalSample * CHANNELS + 2]
        val stratum = when {
            z < horizonBounds[0] -> 0
            z < horizonBounds[1] -> 1
            z < horizonBounds[2] -> 2
            else -> 3
        }
        horizonCounts[stratum]++
    }
    val horizonAngles = Array(4) { DoubleArray(horizonCounts[it]) }
    val horizonChord = Array(4) { DoubleArray(horizonCounts[it]) }
    val horizonCursors = IntArray(4)
    val perSlice = ArrayList<Map<String, Any?>>(sliceIds.size)
    var explicitTotal = 0.0
    var eliminatedTotal = 0.0
    var pairwiseTotal = 0.0
    var adjacentDotSum = 0.0
    var adjacentCount = 0L
    var capContainmentTests = 0L
    var maxCapViolation = 0.0
    var supportOne = 0L
    var supportTwo = 0L
    var supportThree = 0L
    var aggregateCounts = AlgorithmCounts()

    for ((sliceIndex, slice) in sliceIds.withIndex()) {
        val clustering = slices[sliceIndex]
        require(clustering.labels.size == SLICE_SAMPLES)
        validatePermutation(clustering.destinationToSource, SLICE_SAMPLES)
        aggregateCounts += clustering.counts
        val vectorBase = slice * SLICE_SAMPLES * CHANNELS
        val geometry = computeClusterGeometry(texture.directions, vectorBase, clustering.labels, clusterCount)
        val identityDifference = abs(geometry.explicitSquaredChord - geometry.eliminatedSquaredChord)
        val identityTolerance = OBJECTIVE_RELATIVE_TOLERANCE * max(1.0, geometry.explicitSquaredChord)
        require(identityDifference <= identityTolerance) {
            "Slice $slice objective identity differs by $identityDifference, tolerance $identityTolerance"
        }
        explicitTotal += geometry.explicitSquaredChord
        eliminatedTotal += geometry.eliminatedSquaredChord
        pairwiseTotal += geometry.pairwiseSquaredChord
        val sliceAngles = DoubleArray(SLICE_SAMPLES)
        val sliceChord = DoubleArray(SLICE_SAMPLES)
        val meanCenterRadii = DoubleArray(clusterCount)
        for (source in 0 until SLICE_SAMPLES) {
            require(texture.originalIds[slice * SLICE_SAMPLES + source] == source)
            val offset = vectorBase + source * CHANNELS
            val centerOffset = clustering.labels[source] * CHANNELS
            val dot = (texture.directions[offset] * geometry.centers[centerOffset] +
                texture.directions[offset + 1] * geometry.centers[centerOffset + 1] +
                texture.directions[offset + 2] * geometry.centers[centerOffset + 2]).coerceIn(-1.0, 1.0)
            val angle = acos(dot) * RAD_TO_DEG
            val squaredChord = 2.0 * (1.0 - dot)
            val globalSample = sliceIndex * SLICE_SAMPLES + source
            sliceAngles[source] = angle
            sliceChord[source] = squaredChord
            allAngles[globalSample] = angle
            allSquaredChord[globalSample] = squaredChord
            meanCenterRadii[clustering.labels[source]] = max(meanCenterRadii[clustering.labels[source]], angle)
            val z = texture.directions[offset + 2]
            val stratum = when {
                z < horizonBounds[0] -> 0
                z < horizonBounds[1] -> 1
                z < horizonBounds[2] -> 2
                else -> 3
            }
            val stratumIndex = horizonCursors[stratum]++
            horizonAngles[stratum][stratumIndex] = angle
            horizonChord[stratum][stratumIndex] = squaredChord
        }
        val members = buildMembers(clustering.labels, geometry.counts)
        val minimumCapRadii = DoubleArray(clusterCount)
        for (cluster in 0 until clusterCount) {
            val cap = minimumSphericalCap(texture.directions, vectorBase, members[cluster])
            minimumCapRadii[cluster] = cap.radiusDeg
            capContainmentTests += cap.containmentTests
            maxCapViolation = max(maxCapViolation, cap.maxCosViolation)
            when (cap.supportSize) {
                1 -> supportOne++
                2 -> supportTwo++
                3 -> supportThree++
            }
            val aggregateIndex = sliceIndex * clusterCount + cluster
            allClusterSizes[aggregateIndex] = geometry.counts[cluster].toDouble()
            allMeanCenterRadii[aggregateIndex] = meanCenterRadii[cluster]
            allMinimumCapRadii[aggregateIndex] = cap.radiusDeg
        }
        val permutation = clustering.destinationToSource
        for (y in 0 until HEIGHT) {
            for (x in 0 until WIDTH) {
                val destination = y * WIDTH + x
                val source = permutation[destination]
                val sourceOffset = vectorBase + source * CHANNELS
                if (x + 1 < WIDTH) {
                    val rightOffset = vectorBase + permutation[destination + 1] * CHANNELS
                    adjacentDotSum += texture.directions[sourceOffset] * texture.directions[rightOffset] +
                        texture.directions[sourceOffset + 1] * texture.directions[rightOffset + 1] +
                        texture.directions[sourceOffset + 2] * texture.directions[rightOffset + 2]
                    adjacentCount++
                }
                if (y + 1 < HEIGHT) {
                    val downOffset = vectorBase + permutation[destination + WIDTH] * CHANNELS
                    adjacentDotSum += texture.directions[sourceOffset] * texture.directions[downOffset] +
                        texture.directions[sourceOffset + 1] * texture.directions[downOffset + 1] +
                        texture.directions[sourceOffset + 2] * texture.directions[downOffset + 2]
                    adjacentCount++
                }
            }
        }
        perSlice += linkedMapOf(
            "slice" to slice,
            "explicitSquaredChordTotal" to geometry.explicitSquaredChord,
            "eliminatedSquaredChordTotal" to geometry.eliminatedSquaredChord,
            "objectiveIdentityAbsDifference" to identityDifference,
            "pairwiseSquaredChordTotal" to geometry.pairwiseSquaredChord,
            "angularErrorDeg" to distribution(sliceAngles),
            "squaredChord" to distribution(sliceChord),
            "clusterSize" to distribution(DoubleArray(clusterCount) { geometry.counts[it].toDouble() }),
            "meanCenterRadiusDeg" to distribution(meanCenterRadii),
            "minimumCapRadiusDeg" to distribution(minimumCapRadii)
        )
    }
    require(horizonCursors.contentEquals(horizonCounts))
    val aggregateDifference = abs(explicitTotal - eliminatedTotal)
    val aggregateTolerance = OBJECTIVE_RELATIVE_TOLERANCE * max(1.0, explicitTotal)
    require(aggregateDifference <= aggregateTolerance) {
        "Aggregate objective identity differs by $aggregateDifference, tolerance $aggregateTolerance"
    }
    val horizonMetrics = ArrayList<Map<String, Any?>>(4)
    for (stratum in 0..3) {
        horizonMetrics += linkedMapOf(
            "z" to horizonNames[stratum],
            "squaredChord" to distribution(horizonChord[stratum]),
            "angularErrorDeg" to distribution(horizonAngles[stratum])
        )
    }
    return linkedMapOf(
        "id" to caseId,
        "algorithm" to algorithm,
        "placement" to placement,
        "clusterCount" to clusterCount,
        "capacityTarget" to if (algorithm == "unbalanced-spherical-lloyd") null else SLICE_SAMPLES / clusterCount,
        "sampleCount" to sampleTotal,
        "algorithmCounts" to linkedMapOf(
            "hilbertKeyEvaluations" to aggregateCounts.hilbertKeyEvaluations,
            "initializationDotEvaluations" to aggregateCounts.initializationDotEvaluations,
            "assignmentDotEvaluations" to aggregateCounts.assignmentDotEvaluations,
            "lloydIterations" to aggregateCounts.lloydIterations,
            "centerUpdates" to aggregateCounts.centerUpdates,
            "emptyClusterRepairs" to aggregateCounts.emptyClusterRepairs,
            "metricSampleEvaluations" to sampleTotal,
            "minimumCapContainmentTests" to capContainmentTests
        ),
        "objective" to linkedMapOf(
            "explicitSquaredChordTotal" to explicitTotal,
            "eliminatedSquaredChordTotal" to eliminatedTotal,
            "identityAbsDifference" to aggregateDifference,
            "identityTolerance" to aggregateTolerance,
            "squaredChordMean" to explicitTotal / sampleTotal,
            "halfSquaredChordMean" to explicitTotal / (2.0 * sampleTotal),
            "pairwiseSquaredChordTotal" to pairwiseTotal,
            "pairwiseSquaredChordPerSample" to pairwiseTotal / sampleTotal
        ),
        "squaredChord" to distribution(allSquaredChord),
        "angularErrorDeg" to distribution(allAngles),
        "clusterSize" to distribution(allClusterSizes),
        "meanCenterRadiusDeg" to distribution(allMeanCenterRadii),
        "minimumCapRadiusDeg" to distribution(allMinimumCapRadii),
        "minimumCapSolver" to linkedMapOf(
            "method" to "deterministically shuffled incremental support set",
            "containmentToleranceCos" to CAP_CONTAINMENT_TOLERANCE,
            "maxContainmentViolationCos" to maxCapViolation,
            "supportCounts" to linkedMapOf(
                "one" to supportOne,
                "two" to supportTwo,
                "three" to supportThree
            )
        ),
        "destinationAdjacentMeanDot" to adjacentDotSum / adjacentCount,
        "horizonStrata" to horizonMetrics,
        "perSlice" to perSlice
    )
}

fun <T, R> parallelMap(workers: Int, items: List<T>, action: (T) -> R): List<R> {
    require(workers > 0)
    val executor = Executors.newFixedThreadPool(min(workers, max(1, items.size)))
    try {
        val futures = items.map { item -> executor.submit(Callable { action(item) }) }
        return futures.map { it.get() }
    } finally {
        executor.shutdownNow()
    }
}

fun defaultWorkers(): Int = max(1, Runtime.getRuntime().availableProcessors())

fun appendJson(builder: StringBuilder, value: Any?) {
    when (value) {
        null -> builder.append("null")
        is String -> {
            builder.append('"')
            for (character in value) {
                when (character) {
                    '"' -> builder.append("\\\"")
                    '\\' -> builder.append("\\\\")
                    '\b' -> builder.append("\\b")
                    '\u000C' -> builder.append("\\f")
                    '\n' -> builder.append("\\n")
                    '\r' -> builder.append("\\r")
                    '\t' -> builder.append("\\t")
                    else -> if (character.code < 0x20) {
                        builder.append("\\u%04x".format(character.code))
                    } else {
                        builder.append(character)
                    }
                }
            }
            builder.append('"')
        }
        is Boolean -> builder.append(if (value) "true" else "false")
        is Byte, is Short, is Int, is Long -> builder.append(value.toString())
        is Float -> {
            require(value.isFinite()) { "JSON cannot encode $value" }
            builder.append(if (value == 0.0f) "0.0" else value.toString())
        }
        is Double -> {
            require(value.isFinite()) { "JSON cannot encode $value" }
            builder.append(if (value == 0.0) "0.0" else value.toString())
        }
        is Map<*, *> -> {
            val entries = value.entries.map {
                require(it.key is String) { "JSON object keys must be strings" }
                (it.key as String) to it.value
            }.sortedBy { it.first }
            builder.append('{')
            for ((index, entry) in entries.withIndex()) {
                if (index != 0) builder.append(',')
                appendJson(builder, entry.first)
                builder.append(':')
                appendJson(builder, entry.second)
            }
            builder.append('}')
        }
        is Iterable<*> -> {
            builder.append('[')
            value.forEachIndexed { index, element ->
                if (index != 0) builder.append(',')
                appendJson(builder, element)
            }
            builder.append(']')
        }
        is Array<*> -> appendJson(builder, value.asList())
        is IntArray -> appendJson(builder, value.asList())
        is LongArray -> appendJson(builder, value.asList())
        is DoubleArray -> appendJson(builder, value.asList())
        else -> error("Unsupported JSON value ${value::class.qualifiedName}")
    }
}

fun canonicalJson(value: Any?): ByteArray {
    val builder = StringBuilder()
    appendJson(builder, value)
    builder.append('\n')
    return builder.toString().toByteArray(StandardCharsets.UTF_8)
}

fun writeBytes(path: Path, bytes: ByteArray) {
    Files.createDirectories(path.parent)
    Files.write(path, bytes)
}

fun unsignedShortsLittleEndian(values: List<IntArray>): ByteArray {
    val elementCount = values.sumOf { it.size }
    val result = ByteArray(Math.multiplyExact(elementCount, 2))
    var offset = 0
    for (array in values) for (value in array) {
        require(value in 0..0xFFFF)
        result[offset++] = value.toByte()
        result[offset++] = (value ushr 8).toByte()
    }
    return result
}

fun doublesLittleEndian(values: DoubleArray): ByteArray {
    val result = ByteArray(Math.multiplyExact(values.size, 8))
    var offset = 0
    for (value in values) {
        require(value.isFinite())
        val bits = java.lang.Double.doubleToRawLongBits(value)
        for (byte in 0 until 8) result[offset++] = (bits ushr (byte * 8)).toByte()
    }
    return result
}

fun indirectionRg8(permutations: List<IntArray>): ByteArray {
    require(permutations.size == DEPTH)
    val result = ByteArray(TOTAL_SAMPLES * 2)
    for (slice in 0 until DEPTH) {
        val permutation = permutations[slice]
        validatePermutation(permutation, SLICE_SAMPLES)
        for (destination in 0 until SLICE_SAMPLES) {
            val source = permutation[destination]
            val offset = (slice * SLICE_SAMPLES + destination) * 2
            result[offset] = (source % WIDTH).toByte()
            result[offset + 1] = (source / WIDTH).toByte()
        }
    }
    return result
}

fun decodeIndirectionRg8(bytes: ByteArray): List<IntArray> {
    require(bytes.size == TOTAL_SAMPLES * 2) {
        "Expected ${TOTAL_SAMPLES * 2} RG8 indirection bytes, found ${bytes.size}"
    }
    return List(DEPTH) { slice ->
        val permutation = IntArray(SLICE_SAMPLES)
        for (destination in 0 until SLICE_SAMPLES) {
            val offset = (slice * SLICE_SAMPLES + destination) * 2
            val x = bytes[offset].toInt() and 0xFF
            val y = bytes[offset + 1].toInt() and 0xFF
            require(x < WIDTH && y < HEIGHT) {
                "Slice $slice destination $destination has out-of-range source coordinate ($x,$y)"
            }
            permutation[destination] = y * WIDTH + x
        }
        validatePermutation(permutation, SLICE_SAMPLES)
        permutation
    }
}

fun requireLaneGrouping(labels: IntArray, permutation: IntArray, clusterCount: Int) {
    val groupToLabel = IntArray(clusterCount) { -1 }
    val labelToGroup = IntArray(clusterCount) { -1 }
    for (destination in 0 until SLICE_SAMPLES) {
        val group = clusterForDestination(destination, clusterCount)
        val label = labels[permutation[destination]]
        require(groupToLabel[group] == -1 || groupToLabel[group] == label) {
            "Destination lane group $group mixes labels ${groupToLabel[group]} and $label"
        }
        require(labelToGroup[label] == -1 || labelToGroup[label] == group) {
            "Label $label spans destination lane groups ${labelToGroup[label]} and $group"
        }
        groupToLabel[group] = label
        labelToGroup[label] = group
    }
    require(groupToLabel.all { it >= 0 } && labelToGroup.all { it >= 0 })
}

fun compensatedFixedCenterLoss(
    directions: DoubleArray,
    vectorBase: Int,
    labels: IntArray,
    centers: DoubleArray
): Double {
    var sum = 0.0
    var correction = 0.0
    for (sample in labels.indices) {
        val offset = vectorBase + sample * CHANNELS
        val centerOffset = labels[sample] * CHANNELS
        val dot = (directions[offset] * centers[centerOffset] +
            directions[offset + 1] * centers[centerOffset + 1] +
            directions[offset + 2] * centers[centerOffset + 2]).coerceIn(-1.0, 1.0)
        val value = 2.0 * (1.0 - dot) - correction
        val next = sum + value
        correction = (next - sum) - value
        sum = next
    }
    return sum
}

fun artifactJson(path: Path): Map<String, Any?> {
    val bytes = Files.readAllBytes(path)
    return linkedMapOf("bytes" to bytes.size, "sha256" to sha256(bytes))
}

fun writeBytesAtomic(path: Path, bytes: ByteArray) {
    Files.createDirectories(path.parent)
    val temporary = path.resolveSibling(path.fileName.toString() + ".tmp")
    Files.write(temporary, bytes)
    Files.move(temporary, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
}

fun requireOrWriteExact(path: Path, expected: ByteArray) {
    if (Files.exists(path)) {
        require(Files.readAllBytes(path).contentEquals(expected)) {
            "Existing ${normalizePath(path)} does not match this search manifest"
        }
    } else {
        writeBytesAtomic(path, expected)
    }
}

fun parseOptions(
    arguments: List<String>,
    allowed: Set<String>,
    commaListOptions: Set<String> = emptySet()
): Map<String, String> {
    val result = linkedMapOf<String, String>()
    var index = 0
    while (index < arguments.size) {
        val option = arguments[index]
        require(option in allowed) { "Unknown option $option" }
        require(option !in result) { "Duplicate option $option" }
        index++
        require(index < arguments.size && !arguments[index].startsWith("--")) { "Option $option requires a value" }
        if (option in commaListOptions) {
            val values = ArrayList<String>()
            while (index < arguments.size && !arguments[index].startsWith("--")) values += arguments[index++]
            result[option] = values.joinToString(",")
        } else {
            result[option] = arguments[index++]
        }
    }
    return result
}

fun expectFailure(name: String, action: () -> Unit) {
    var failed = false
    try {
        action()
    } catch (_: IllegalArgumentException) {
        failed = true
    } catch (_: IllegalStateException) {
        failed = true
    }
    require(failed) { "Expected $name to fail" }
}

fun assertClose(name: String, actual: Double, expected: Double, tolerance: Double) {
    require(abs(actual - expected) <= tolerance) {
        "$name mismatch: actual=$actual expected=$expected tolerance=$tolerance"
    }
}

fun vectorLength(values: DoubleArray, offset: Int): Double = sqrt(
    values[offset] * values[offset] +
        values[offset + 1] * values[offset + 1] +
        values[offset + 2] * values[offset + 2]
)

fun normalizedVector(x: Double, y: Double, z: Double): DoubleArray {
    val length = sqrt(x * x + y * y + z * z)
    require(length > CENTER_EPSILON)
    return doubleArrayOf(x / length, y / length, z / length)
}

fun clusterVectorSums(
    directions: DoubleArray,
    vectorBase: Int,
    labels: IntArray,
    clusterCount: Int
): DoubleArray {
    val sums = DoubleArray(clusterCount * CHANNELS)
    for (sample in labels.indices) {
        val clusterOffset = labels[sample] * CHANNELS
        val sampleOffset = vectorBase + sample * CHANNELS
        sums[clusterOffset] += directions[sampleOffset]
        sums[clusterOffset + 1] += directions[sampleOffset + 1]
        sums[clusterOffset + 2] += directions[sampleOffset + 2]
    }
    return sums
}

fun exactSwapDelta(
    directions: DoubleArray,
    vectorBase: Int,
    labels: IntArray,
    sums: DoubleArray,
    firstSample: Int,
    secondSample: Int,
    clusterLengths: DoubleArray? = null
): Double {
    val firstCluster = labels[firstSample]
    val secondCluster = labels[secondSample]
    if (firstCluster == secondCluster) return 0.0
    val firstClusterOffset = firstCluster * CHANNELS
    val secondClusterOffset = secondCluster * CHANNELS
    val firstOffset = vectorBase + firstSample * CHANNELS
    val secondOffset = vectorBase + secondSample * CHANNELS
    val dx = directions[secondOffset] - directions[firstOffset]
    val dy = directions[secondOffset + 1] - directions[firstOffset + 1]
    val dz = directions[secondOffset + 2] - directions[firstOffset + 2]
    val oldLengths = if (clusterLengths == null) {
        vectorLength(sums, firstClusterOffset) + vectorLength(sums, secondClusterOffset)
    } else {
        clusterLengths[firstCluster] + clusterLengths[secondCluster]
    }
    val firstLength = sqrt(
        (sums[firstClusterOffset] + dx) * (sums[firstClusterOffset] + dx) +
            (sums[firstClusterOffset + 1] + dy) * (sums[firstClusterOffset + 1] + dy) +
            (sums[firstClusterOffset + 2] + dz) * (sums[firstClusterOffset + 2] + dz)
    )
    val secondLength = sqrt(
        (sums[secondClusterOffset] - dx) * (sums[secondClusterOffset] - dx) +
            (sums[secondClusterOffset + 1] - dy) * (sums[secondClusterOffset + 1] - dy) +
            (sums[secondClusterOffset + 2] - dz) * (sums[secondClusterOffset + 2] - dz)
    )
    return -2.0 * (firstLength + secondLength - oldLengths)
}

fun betterSwap(delta: Double, first: Int, second: Int, best: SwapScanResult): Boolean =
    delta < best.delta || delta == best.delta &&
        (first < best.firstSample || first == best.firstSample && second < best.secondSample)

fun completeSwapScan(
    directions: DoubleArray,
    vectorBase: Int,
    labels: IntArray,
    clusterCount: Int
): SwapScanResult {
    val geometry = computeClusterGeometry(directions, vectorBase, labels, clusterCount)
    val members = buildMembers(labels, geometry.counts)
    val sums = clusterVectorSums(directions, vectorBase, labels, clusterCount)
    val lengths = DoubleArray(clusterCount) { vectorLength(sums, it * CHANNELS) }
    var best = SwapScanResult(-1, -1, Double.POSITIVE_INFINITY, 0L)
    var evaluated = 0L
    for (firstCluster in 0 until clusterCount) {
        for (secondCluster in firstCluster + 1 until clusterCount) {
            for (firstSample in members[firstCluster]) {
                for (secondSample in members[secondCluster]) {
                    val delta = exactSwapDelta(
                        directions, vectorBase, labels, sums, firstSample, secondSample, lengths
                    )
                    evaluated++
                    if (betterSwap(delta, firstSample, secondSample, best)) {
                        best = SwapScanResult(firstSample, secondSample, delta, evaluated)
                    }
                }
            }
        }
    }
    return best.copy(pairsEvaluated = evaluated)
}

fun neighboringClusterPairs(centers: DoubleArray, clusterCount: Int, neighbors: Int): List<Pair<Int, Int>> {
    val encoded = sortedSetOf<Long>()
    for (first in 0 until clusterCount) {
        val candidates = (0 until clusterCount).filter { it != first }.sortedWith { a, b ->
            val firstOffset = first * CHANNELS
            val aOffset = a * CHANNELS
            val bOffset = b * CHANNELS
            val aDot = centers[firstOffset] * centers[aOffset] +
                centers[firstOffset + 1] * centers[aOffset + 1] + centers[firstOffset + 2] * centers[aOffset + 2]
            val bDot = centers[firstOffset] * centers[bOffset] +
                centers[firstOffset + 1] * centers[bOffset + 1] + centers[firstOffset + 2] * centers[bOffset + 2]
            val comparison = bDot.compareTo(aDot)
            if (comparison != 0) comparison else a.compareTo(b)
        }
        for (second in candidates.take(neighbors)) {
            val low = min(first, second)
            val high = max(first, second)
            encoded += low.toLong() * clusterCount + high
        }
    }
    return encoded.map { (it / clusterCount).toInt() to (it % clusterCount).toInt() }
}

fun boundaryMembers(
    directions: DoubleArray,
    vectorBase: Int,
    labels: IntArray,
    geometry: ClusterGeometry,
    count: Int
): Array<IntArray> {
    val members = buildMembers(labels, geometry.counts)
    val scores = DoubleArray(labels.size)
    for (sample in labels.indices) {
        val sampleOffset = vectorBase + sample * CHANNELS
        val centerOffset = labels[sample] * CHANNELS
        scores[sample] = directions[sampleOffset] * geometry.centers[centerOffset] +
            directions[sampleOffset + 1] * geometry.centers[centerOffset + 1] +
            directions[sampleOffset + 2] * geometry.centers[centerOffset + 2]
    }
    return Array(geometry.counts.size) { cluster ->
        sortSamplesByScore(members[cluster], scores).take(count).toIntArray()
    }
}

fun applyCheckedSwap(
    directions: DoubleArray,
    vectorBase: Int,
    labels: IntArray,
    clusterCount: Int,
    firstSample: Int,
    secondSample: Int,
    tolerance: Double
): Double {
    val before = computeClusterGeometry(directions, vectorBase, labels, clusterCount).explicitSquaredChord
    val firstCluster = labels[firstSample]
    labels[firstSample] = labels[secondSample]
    labels[secondSample] = firstCluster
    val after = computeClusterGeometry(directions, vectorBase, labels, clusterCount).explicitSquaredChord
    require(after < before - tolerance) {
        "Revalidated swap $firstSample/$secondSample did not improve: before=$before after=$after"
    }
    return before - after
}

fun runScreenedSwaps(
    directions: DoubleArray,
    vectorBase: Int,
    labels: IntArray,
    clusterCount: Int,
    tolerance: Double,
    maximumMoves: Int = 32
): Pair<Int, Double> {
    var moves = 0
    var gain = 0.0
    while (moves < maximumMoves) {
        val geometry = computeClusterGeometry(directions, vectorBase, labels, clusterCount)
        val sums = clusterVectorSums(directions, vectorBase, labels, clusterCount)
        val lengths = DoubleArray(clusterCount) { vectorLength(sums, it * CHANNELS) }
        val boundary = boundaryMembers(directions, vectorBase, labels, geometry, 8)
        var best = SwapScanResult(-1, -1, Double.POSITIVE_INFINITY, 0L)
        for ((firstCluster, secondCluster) in neighboringClusterPairs(geometry.centers, clusterCount, 6)) {
            for (firstSample in boundary[firstCluster]) for (secondSample in boundary[secondCluster]) {
                val delta = exactSwapDelta(
                    directions, vectorBase, labels, sums, firstSample, secondSample, lengths
                )
                if (betterSwap(delta, firstSample, secondSample, best)) {
                    best = SwapScanResult(firstSample, secondSample, delta, 0L)
                }
            }
        }
        if (best.firstSample < 0 || best.delta >= -tolerance) break
        gain += applyCheckedSwap(
            directions, vectorBase, labels, clusterCount, best.firstSample, best.secondSample, tolerance
        )
        moves++
    }
    return moves to gain
}

fun fixedCenterTwoWayLabels(
    directions: DoubleArray,
    vectorBase: Int,
    samples: IntArray,
    centers: DoubleArray,
    capacity: Int,
    quantization: Long? = null
): IntArray {
    require(samples.size == capacity * 2 && centers.size == CHANNELS * 2)
    val scores = DoubleArray(samples.size)
    for (index in samples.indices) {
        val offset = vectorBase + samples[index] * CHANNELS
        val firstDot = directions[offset] * centers[0] + directions[offset + 1] * centers[1] +
            directions[offset + 2] * centers[2]
        val secondDot = directions[offset] * centers[3] + directions[offset + 1] * centers[4] +
            directions[offset + 2] * centers[5]
        scores[index] = if (quantization == null) {
            2.0 * (secondDot - firstDot)
        } else {
            (quantizedSquaredChord(firstDot, quantization) -
                quantizedSquaredChord(secondDot, quantization)).toDouble()
        }
    }
    val order = sortSamplesByScore(IntArray(samples.size) { it }, scores)
    return IntArray(samples.size).also { result ->
        for (index in capacity until order.size) result[order[index]] = 1
    }
}

fun centersForTwoWayLabels(
    directions: DoubleArray,
    vectorBase: Int,
    samples: IntArray,
    labels: IntArray
): DoubleArray {
    val sums = DoubleArray(CHANNELS * 2)
    for (index in samples.indices) {
        val centerOffset = labels[index] * CHANNELS
        val sampleOffset = vectorBase + samples[index] * CHANNELS
        sums[centerOffset] += directions[sampleOffset]
        sums[centerOffset + 1] += directions[sampleOffset + 1]
        sums[centerOffset + 2] += directions[sampleOffset + 2]
    }
    for (cluster in 0..1) {
        val offset = cluster * CHANNELS
        val normalized = normalizedVector(sums[offset], sums[offset + 1], sums[offset + 2])
        normalized.copyInto(sums, offset)
    }
    return sums
}

fun twoWayObjective(
    directions: DoubleArray,
    vectorBase: Int,
    samples: IntArray,
    labels: IntArray,
    capacity: Int
): Double {
    val sums = DoubleArray(CHANNELS * 2)
    for (index in samples.indices) {
        val centerOffset = labels[index] * CHANNELS
        val sampleOffset = vectorBase + samples[index] * CHANNELS
        sums[centerOffset] += directions[sampleOffset]
        sums[centerOffset + 1] += directions[sampleOffset + 1]
        sums[centerOffset + 2] += directions[sampleOffset + 2]
    }
    return 2.0 * (capacity * 2 - vectorLength(sums, 0) - vectorLength(sums, CHANNELS))
}

fun lexicographicallyLess(first: IntArray, second: IntArray): Boolean {
    for (index in first.indices) {
        if (first[index] != second[index]) return first[index] < second[index]
    }
    return false
}

fun twoClusterRepartition(
    directions: DoubleArray,
    vectorBase: Int,
    labels: IntArray,
    geometry: ClusterGeometry,
    firstCluster: Int,
    secondCluster: Int,
    capacity: Int
): Pair<IntArray, Double> {
    val members = buildMembers(labels, geometry.counts)
    val samples = (members[firstCluster] + members[secondCluster]).sortedArray()
    val firstOffset = firstCluster * CHANNELS
    val secondOffset = secondCluster * CHANNELS
    val midpoint = normalizedVector(
        geometry.centers[firstOffset] + geometry.centers[secondOffset],
        geometry.centers[firstOffset + 1] + geometry.centers[secondOffset + 1],
        geometry.centers[firstOffset + 2] + geometry.centers[secondOffset + 2]
    )
    val difference = doubleArrayOf(
        geometry.centers[firstOffset] - geometry.centers[secondOffset],
        geometry.centers[firstOffset + 1] - geometry.centers[secondOffset + 1],
        geometry.centers[firstOffset + 2] - geometry.centers[secondOffset + 2]
    )
    val projection = difference[0] * midpoint[0] + difference[1] * midpoint[1] + difference[2] * midpoint[2]
    val tangentLength = sqrt(
        (difference[0] - projection * midpoint[0]) * (difference[0] - projection * midpoint[0]) +
            (difference[1] - projection * midpoint[1]) * (difference[1] - projection * midpoint[1]) +
            (difference[2] - projection * midpoint[2]) * (difference[2] - projection * midpoint[2])
    )
    val primaryAxis = if (tangentLength > CENTER_EPSILON) normalizedVector(
        difference[0] - projection * midpoint[0],
        difference[1] - projection * midpoint[1],
        difference[2] - projection * midpoint[2]
    ) else deterministicPerpendicular(midpoint[0], midpoint[1], midpoint[2])
    val perpendicular = deterministicPerpendicular(midpoint[0], midpoint[1], midpoint[2])
    val crossAxis = normalizedVector(
        midpoint[1] * perpendicular[2] - midpoint[2] * perpendicular[1],
        midpoint[2] * perpendicular[0] - midpoint[0] * perpendicular[2],
        midpoint[0] * perpendicular[1] - midpoint[1] * perpendicular[0]
    )
    val axes = listOf(primaryAxis, perpendicular, crossAxis)
    val currentPairLabels = IntArray(samples.size) { if (labels[samples[it]] == firstCluster) 0 else 1 }
    var bestLabels = currentPairLabels
    val sums = clusterVectorSums(directions, vectorBase, labels, geometry.counts.size)
    val oldPairObjective = 2.0 * (capacity * 2 - vectorLength(sums, firstOffset) - vectorLength(sums, secondOffset))
    var bestPairObjective = oldPairObjective
    for (axis in axes) {
        val firstCenter = normalizedVector(
            midpoint[0] + 0.35 * axis[0], midpoint[1] + 0.35 * axis[1], midpoint[2] + 0.35 * axis[2]
        )
        val secondCenter = normalizedVector(
            midpoint[0] - 0.35 * axis[0], midpoint[1] - 0.35 * axis[1], midpoint[2] - 0.35 * axis[2]
        )
        var centers = firstCenter + secondCenter
        var candidate = currentPairLabels
        for (iteration in 0 until 12) {
            val next = fixedCenterTwoWayLabels(directions, vectorBase, samples, centers, capacity)
            if (next.contentEquals(candidate)) break
            candidate = next
            centers = centersForTwoWayLabels(directions, vectorBase, samples, candidate)
        }
        val objective = twoWayObjective(directions, vectorBase, samples, candidate, capacity)
        if (objective < bestPairObjective || objective == bestPairObjective &&
            lexicographicallyLess(candidate, bestLabels)) {
            bestPairObjective = objective
            bestLabels = candidate
        }
    }
    return bestLabels to (bestPairObjective - oldPairObjective)
}

fun runPairRepartition(
    directions: DoubleArray,
    vectorBase: Int,
    labels: IntArray,
    clusterCount: Int,
    capacity: Int,
    tolerance: Double,
    maximumPasses: Int = 4
): Triple<Int, Double, Long> {
    var moves = 0
    var gain = 0.0
    var candidates = 0L
    repeat(maximumPasses) {
        val geometry = computeClusterGeometry(directions, vectorBase, labels, clusterCount)
        var bestFirst = -1
        var bestSecond = -1
        var bestPairLabels = IntArray(0)
        var bestDelta = 0.0
        for ((first, second) in neighboringClusterPairs(geometry.centers, clusterCount, 8)) {
            val candidate = twoClusterRepartition(
                directions, vectorBase, labels, geometry, first, second, capacity
            )
            candidates++
            if (candidate.second < bestDelta || candidate.second == bestDelta &&
                (first < bestFirst || first == bestFirst && second < bestSecond)) {
                bestFirst = first
                bestSecond = second
                bestPairLabels = candidate.first
                bestDelta = candidate.second
            }
        }
        if (bestFirst < 0 || bestDelta >= -tolerance) return Triple(moves, gain, candidates)
        val before = geometry.explicitSquaredChord
        val members = buildMembers(labels, geometry.counts)
        val samples = (members[bestFirst] + members[bestSecond]).sortedArray()
        for (index in samples.indices) {
            labels[samples[index]] = if (bestPairLabels[index] == 0) bestFirst else bestSecond
        }
        val after = computeClusterGeometry(directions, vectorBase, labels, clusterCount).explicitSquaredChord
        require(after < before - tolerance) {
            "Revalidated pair repartition $bestFirst/$bestSecond did not improve: before=$before after=$after"
        }
        gain += before - after
        moves++
    }
    return Triple(moves, gain, candidates)
}

fun refineLabels(
    directions: DoubleArray,
    vectorBase: Int,
    initialLabels: IntArray,
    clusterCount: Int,
    capacity: Int,
    quantization: Long,
    methods: Set<String>,
    fullSwapCertificate: Boolean,
    tolerance: Double = 1e-12
): LocalRefinementResult {
    require(methods.all { it == "swap" || it == "pair" })
    val labels = initialLabels.copyOf()
    val initialGeometry = computeClusterGeometry(directions, vectorBase, labels, clusterCount)
    require(initialGeometry.counts.all { it == capacity })
    val history = arrayListOf<Map<String, Any?>>()
    var objective = initialGeometry.explicitSquaredChord
    var screenedSwapMoves = 0
    var screenedSwapGain = 0.0
    var pairMoves = 0
    var pairGain = 0.0
    var pairCandidates = 0L

    if ("swap" in methods) {
        val before = objective
        val result = runScreenedSwaps(directions, vectorBase, labels, clusterCount, tolerance)
        objective = computeClusterGeometry(directions, vectorBase, labels, clusterCount).explicitSquaredChord
        screenedSwapMoves = result.first
        screenedSwapGain = before - objective
        assertClose("screened swap gain", screenedSwapGain, result.second, 1e-10)
        history += linkedMapOf(
            "phase" to "screened-swaps",
            "objectiveBefore" to before,
            "objectiveAfter" to objective,
            "acceptedMoves" to screenedSwapMoves,
            "gain" to screenedSwapGain,
            "screen" to "six nearest center neighbors and eight lowest own-center-dot members"
        )
    }
    if ("pair" in methods) {
        val before = objective
        val result = runPairRepartition(
            directions, vectorBase, labels, clusterCount, capacity, tolerance
        )
        objective = computeClusterGeometry(directions, vectorBase, labels, clusterCount).explicitSquaredChord
        pairMoves = result.first
        pairGain = before - objective
        pairCandidates = result.third
        assertClose("pair repartition gain", pairGain, result.second, 1e-10)
        history += linkedMapOf(
            "phase" to "two-cluster-repartition",
            "objectiveBefore" to before,
            "objectiveAfter" to objective,
            "acceptedMoves" to pairMoves,
            "gain" to pairGain,
            "candidateClusterPairs" to pairCandidates,
            "freshAxesPerPair" to 3,
            "maximumFixedCenterIterations" to 12
        )
    }

    var reconvergenceStatus = "NOT_RUN_NO_LOCAL_CHANGE"
    var reconvergenceAccepted = false
    var reconvergenceGain = 0.0
    if (screenedSwapMoves + pairMoves > 0) {
        val before = objective
        val labelsBefore = labels.copyOf()
        val centers = normalizedCentersForLabels(directions, vectorBase, labels, clusterCount, capacity)
        val result = balancedLloyd(
            directions, vectorBase, labels.size, clusterCount, capacity, quantization,
            BALANCED_MAX_ITERATIONS, CenterInitialization(centers, 0)
        )
        reconvergenceStatus = result.status
        if (result.objective <= before) {
            result.labels.copyInto(labels)
            objective = result.objective
            reconvergenceAccepted = !result.labels.contentEquals(labelsBefore) || result.objective < before
            reconvergenceGain = before - objective
        }
        history += linkedMapOf(
            "phase" to "full-edge-balanced-lloyd-reconvergence",
            "objectiveBefore" to before,
            "candidateObjective" to result.objective,
            "objectiveAfter" to objective,
            "status" to result.status,
            "converged" to result.converged,
            "accepted" to reconvergenceAccepted,
            "gain" to reconvergenceGain,
            "iterations" to result.iterations.size
        )
    }

    var certificateSwapMoves = 0
    var certificateGain = 0.0
    var fullSweepCount = 0
    var fullSweepPairs = 0L
    var certified = false
    if (fullSwapCertificate) {
        while (true) {
            val scan = completeSwapScan(directions, vectorBase, labels, clusterCount)
            fullSweepCount++
            fullSweepPairs += scan.pairsEvaluated
            if (scan.firstSample < 0 || scan.delta >= -tolerance) {
                history += linkedMapOf(
                    "phase" to "complete-one-swap-certificate",
                    "sweep" to fullSweepCount,
                    "pairsEvaluated" to scan.pairsEvaluated,
                    "bestDelta" to scan.delta,
                    "tolerance" to tolerance,
                    "accepted" to false
                )
                certified = true
                break
            }
            val before = objective
            certificateGain += applyCheckedSwap(
                directions, vectorBase, labels, clusterCount,
                scan.firstSample, scan.secondSample, tolerance
            )
            objective = computeClusterGeometry(directions, vectorBase, labels, clusterCount).explicitSquaredChord
            certificateSwapMoves++
            history += linkedMapOf(
                "phase" to "complete-sweep-swap",
                "sweep" to fullSweepCount,
                "pairsEvaluated" to scan.pairsEvaluated,
                "predictedDelta" to scan.delta,
                "objectiveBefore" to before,
                "objectiveAfter" to objective,
                "firstSample" to scan.firstSample,
                "secondSample" to scan.secondSample,
                "accepted" to true
            )
            val centers = normalizedCentersForLabels(directions, vectorBase, labels, clusterCount, capacity)
            val reconverged = balancedLloyd(
                directions, vectorBase, labels.size, clusterCount, capacity, quantization,
                BALANCED_MAX_ITERATIONS, CenterInitialization(centers, 0)
            )
            if (reconverged.objective <= objective) {
                certificateGain += objective - reconverged.objective
                reconverged.labels.copyInto(labels)
                objective = reconverged.objective
            }
        }
    }
    val finalGeometry = computeClusterGeometry(directions, vectorBase, labels, clusterCount)
    require(finalGeometry.counts.all { it == capacity })
    require(finalGeometry.explicitSquaredChord <= initialGeometry.explicitSquaredChord)
    assertClose("final refinement objective", objective, finalGeometry.explicitSquaredChord, 1e-10)
    return LocalRefinementResult(
        labels,
        initialGeometry.explicitSquaredChord,
        objective,
        screenedSwapMoves,
        screenedSwapGain,
        pairMoves,
        pairGain,
        pairCandidates,
        reconvergenceStatus,
        reconvergenceAccepted,
        reconvergenceGain,
        certificateSwapMoves,
        certificateGain,
        fullSweepCount,
        fullSweepPairs,
        tolerance,
        certified,
        history
    )
}

fun localRefinementJson(result: LocalRefinementResult): Map<String, Any?> = linkedMapOf(
    "initialObjective" to result.initialObjective,
    "finalObjective" to result.finalObjective,
    "objectiveGain" to result.initialObjective - result.finalObjective,
    "labelsSha256" to labelsSha256(result.labels),
    "partitionSha256" to partitionSha256(result.labels, result.labels.maxOrNull()!! + 1),
    "screenedSwaps" to linkedMapOf(
        "acceptedMoves" to result.screenedSwapMoves,
        "gain" to result.screenedSwapGain
    ),
    "pairRepartition" to linkedMapOf(
        "acceptedMoves" to result.pairMoves,
        "gain" to result.pairGain,
        "candidateClusterPairs" to result.pairCandidates
    ),
    "reconvergence" to linkedMapOf(
        "status" to result.reconvergenceStatus,
        "accepted" to result.reconvergenceAccepted,
        "gain" to result.reconvergenceGain
    ),
    "completeSwapCertificate" to linkedMapOf(
        "certified" to result.certified,
        "acceptedMovesBeforeCertificate" to result.certificateSwapMoves,
        "gain" to result.certificateGain,
        "fullSweepCount" to result.fullSweepCount,
        "pairsEvaluated" to result.fullSweepPairs,
        "tolerance" to result.certificateTolerance
    ),
    "history" to result.history
)

fun randomUnitDirections(random: DeterministicRandom, sampleCount: Int): DoubleArray {
    val directions = DoubleArray(sampleCount * CHANNELS)
    for (sample in 0 until sampleCount) {
        val z = random.nextDouble()
        val angle = 2.0 * PI * random.nextDouble()
        val radius = sqrt(max(0.0, 1.0 - z * z))
        directions[sample * CHANNELS] = radius * cos(angle)
        directions[sample * CHANNELS + 1] = radius * sin(angle)
        directions[sample * CHANNELS + 2] = z
    }
    return directions
}

fun runLocalRefinementSelfTests() {
    var checks = 0
    fun checked(action: () -> Unit) {
        action()
        checks++
    }
    val random = DeterministicRandom(0x4C4F43414C524546L)
    repeat(48) {
        val clusterCount = 4
        val capacity = 3
        val directions = randomUnitDirections(random, clusterCount * capacity)
        val labels = IntArray(clusterCount * capacity) { it % clusterCount }
        val sums = clusterVectorSums(directions, 0, labels, clusterCount)
        for (first in labels.indices) for (second in first + 1 until labels.size) {
            if (labels[first] == labels[second]) continue
            checked {
                val predicted = exactSwapDelta(directions, 0, labels, sums, first, second)
                val before = computeClusterGeometry(directions, 0, labels, clusterCount).explicitSquaredChord
                val swapped = labels.copyOf()
                val cluster = swapped[first]
                swapped[first] = swapped[second]
                swapped[second] = cluster
                val actual = computeClusterGeometry(
                    directions, 0, swapped, clusterCount
                ).explicitSquaredChord - before
                assertClose("exact swap delta", predicted, actual, 2e-13)
            }
        }
    }
    repeat(40) {
        val sampleCount = 8
        val capacity = 4
        val directions = randomUnitDirections(random, sampleCount)
        val centers = randomUnitDirections(random, 2)
        val samples = IntArray(sampleCount) { it }
        val sorted = fixedCenterTwoWayLabels(
            directions, 0, samples, centers, capacity, DEFAULT_QUANTIZATION
        )
        val transport = balancedTransport(
            directions, 0, sampleCount, centers, 2, capacity, DEFAULT_QUANTIZATION
        )
        requireOptimalTransport(transport, sampleCount, 2, capacity)
        checked {
            val sortedCost = quantizedCostForLabels(
                directions, 0, sorted, centers, DEFAULT_QUANTIZATION
            )
            require(sortedCost == transport.quantizedCost) {
                "Two-way sorting cost $sortedCost != transport oracle ${transport.quantizedCost}"
            }
            require(sorted.count { it == 0 } == capacity && sorted.count { it == 1 } == capacity)
        }
    }
    val microDirections = randomUnitDirections(random, 16)
    val microLabels = IntArray(16) { it % 4 }
    val first = refineLabels(
        microDirections, 0, microLabels, 4, 4, DEFAULT_QUANTIZATION,
        setOf("swap", "pair"), true
    )
    val second = refineLabels(
        microDirections, 0, microLabels, 4, 4, DEFAULT_QUANTIZATION,
        setOf("swap", "pair"), true
    )
    checked {
        require(first.labels.contentEquals(second.labels))
        require(canonicalJson(localRefinementJson(first)).contentEquals(
            canonicalJson(localRefinementJson(second))))
        require(first.certified)
        require(first.finalObjective <= first.initialObjective)
        val counts = computeClusterGeometry(microDirections, 0, first.labels, 4).counts
        require(counts.all { it == 4 })
    }
    val receipt = linkedMapOf<String, Any?>(
        "schema" to "alpha-piscium/stbn-clustering-local-refinement-self-test/v1",
        "checks" to checks,
        "swapMicrocases" to 48,
        "swapDelta" to "exact post-update vector-sum center loss versus full objective recomputation",
        "twoClusterMicrocases" to 40,
        "twoClusterOracle" to "quantized fixed-center M-of-2M sorting cost equals generic full-edge transport",
        "deterministicEndToEnd" to true,
        "capacityAndMonotonicity" to true
    )
    val bytes = canonicalJson(receipt)
    println("local-refinement-self-test.json|${bytes.size}|${sha256(bytes)}")
    print(bytes.toString(StandardCharsets.UTF_8))
    println("PASS local refinement self-test: $checks checks")
}

fun usedJvmHeapBytes(): Long {
    val runtime = Runtime.getRuntime()
    return runtime.totalMemory() - runtime.freeMemory()
}

fun pooledDirections(
    directions: DoubleArray,
    vectorBase: Int,
    samples: IntArray
): DoubleArray {
    val pooled = DoubleArray(samples.size * CHANNELS)
    for (index in samples.indices) {
        directions.copyInto(
            pooled,
            index * CHANNELS,
            vectorBase + samples[index] * CHANNELS,
            vectorBase + (samples[index] + 1) * CHANNELS
        )
    }
    return pooled
}

fun pooledLabels(
    labels: IntArray,
    samples: IntArray,
    clusters: IntArray
): IntArray {
    val localCluster = IntArray(clusters.maxOrNull()!! + 1) { -1 }
    for (index in clusters.indices) localCluster[clusters[index]] = index
    return IntArray(samples.size) { index ->
        localCluster[labels[samples[index]]].also { require(it >= 0) }
    }
}

fun pooledNeighborhoodRefinement(
    directions: DoubleArray,
    vectorBase: Int,
    labels: IntArray,
    clusters: IntArray,
    capacity: Int,
    quantization: Long,
    seed: Long
): PooledNeighborhoodResult {
    val geometry = computeClusterGeometry(directions, vectorBase, labels, labels.maxOrNull()!! + 1)
    val members = buildMembers(labels, geometry.counts)
    val samples = clusters.flatMap { members[it].asIterable() }.sorted().toIntArray()
    val pooled = pooledDirections(directions, vectorBase, samples)
    val currentLabels = pooledLabels(labels, samples, clusters)
    val neighborhoodSize = clusters.size
    val sampleCount = samples.size
    require(sampleCount == capacity * neighborhoodSize)
    val incumbentObjective = computeClusterGeometry(
        pooled, 0, currentLabels, neighborhoodSize
    ).explicitSquaredChord
    var bestLabels = currentLabels
    var bestObjective = incumbentObjective
    var bestStatus = "INCUMBENT"
    var bestConverged = true
    val starts = arrayListOf<Map<String, Any?>>()
    val currentCenters = normalizedCentersForLabels(
        pooled, 0, currentLabels, neighborhoodSize, capacity
    )
    val pcaSeed0 = mix64(seed xor 0x50434130L)
    val pcaSeed1 = mix64(seed xor 0x50434131L)
    val d2Seed = mix64(seed xor 0x44325345L)
    val hilbertSeed = mix64(seed xor 0x48494C42L)
    val initializations = listOf(
        Triple("current-centers", CenterInitialization(currentCenters, 0), 0L),
        Triple("fresh-recursive-pca-0", recursiveTangentPcaInitialization(
            pooled, 0, sampleCount, neighborhoodSize, pcaSeed0
        ).let { CenterInitialization(it.centers, it.dotEvaluations) }, pcaSeed0),
        Triple("fresh-recursive-pca-1", recursiveTangentPcaInitialization(
            pooled, 0, sampleCount, neighborhoodSize, pcaSeed1
        ).let { CenterInitialization(it.centers, it.dotEvaluations) }, pcaSeed1),
        Triple("fresh-squared-chord-d2", squaredChordD2Initialization(
            pooled, 0, sampleCount, neighborhoodSize, d2Seed
        ).let { CenterInitialization(it.centers, it.dotEvaluations) }, d2Seed),
        Triple("fresh-structured-hilbert", structuredHilbertInitialization(
            pooled, 0, sampleCount, neighborhoodSize, hilbertSeed
        ).let { CenterInitialization(it.centers, it.dotEvaluations) }, hilbertSeed)
    )
    for ((id, initialization, initializationSeed) in initializations) {
        val result = balancedLloyd(
            pooled,
            0,
            sampleCount,
            neighborhoodSize,
            capacity,
            quantization,
            32,
            initialization
        )
        val partition = partitionSha256(result.labels, neighborhoodSize)
        starts += linkedMapOf(
            "id" to id,
            "seed" to unsignedHex(initializationSeed),
            "objective" to result.objective,
            "status" to result.status,
            "converged" to result.converged,
            "iterations" to result.iterations.size,
            "partitionSha256" to partition,
            "transport" to transportStatsJson(result.transport)
        )
        val bestPartition = partitionSha256(bestLabels, neighborhoodSize)
        if (result.objective < bestObjective || result.objective == bestObjective && partition < bestPartition) {
            bestLabels = result.labels
            bestObjective = result.objective
            bestStatus = result.status
            bestConverged = result.converged
        }
    }
    require(bestObjective <= incumbentObjective)
    return PooledNeighborhoodResult(bestLabels, bestObjective, bestStatus, bestConverged, starts)
}

fun clusterQualityScores(
    directions: DoubleArray,
    vectorBase: Int,
    labels: IntArray,
    geometry: ClusterGeometry
): DoubleArray {
    val loss = DoubleArray(geometry.counts.size)
    val elongation = DoubleArray(geometry.counts.size)
    val horizon = DoubleArray(geometry.counts.size)
    for (sample in labels.indices) {
        val cluster = labels[sample]
        val sampleOffset = vectorBase + sample * CHANNELS
        val centerOffset = cluster * CHANNELS
        val dot = (directions[sampleOffset] * geometry.centers[centerOffset] +
            directions[sampleOffset + 1] * geometry.centers[centerOffset + 1] +
            directions[sampleOffset + 2] * geometry.centers[centerOffset + 2]).coerceIn(-1.0, 1.0)
        val sampleLoss = 2.0 * (1.0 - dot)
        loss[cluster] += sampleLoss
        elongation[cluster] = max(elongation[cluster], sampleLoss)
        if (directions[sampleOffset + 2] < 0.1) horizon[cluster]++
    }
    fun normalized(values: DoubleArray): DoubleArray {
        val maximum = max(values.maxOrNull()!!, 1e-300)
        return DoubleArray(values.size) { values[it] / maximum }
    }
    val normalizedLoss = normalized(loss)
    val normalizedElongation = normalized(elongation)
    val normalizedHorizon = normalized(horizon)
    return DoubleArray(geometry.counts.size) {
        normalizedLoss[it] + normalizedElongation[it] + normalizedHorizon[it]
    }
}

fun eliteDisagreementScores(
    labels: IntArray,
    clusterCount: Int,
    eliteLabels: List<IntArray>
): DoubleArray {
    val geometryCounts = IntArray(clusterCount)
    for (cluster in labels) geometryCounts[cluster]++
    val members = buildMembers(labels, geometryCounts)
    val scores = DoubleArray(clusterCount)
    for (elite in eliteLabels) {
        require(elite.size == labels.size)
        for (cluster in 0 until clusterCount) {
            val overlap = IntArray(clusterCount)
            for (sample in members[cluster]) overlap[elite[sample]]++
            scores[cluster] += 1.0 - overlap.maxOrNull()!!.toDouble() / members[cluster].size
        }
    }
    return scores
}

fun rankedClusters(scores: DoubleArray): IntArray =
    scores.indices.sortedWith { first, second ->
        val comparison = scores[second].compareTo(scores[first])
        if (comparison != 0) comparison else first.compareTo(second)
    }.toIntArray()

fun adjacentNeighborhood(seedCluster: Int, size: Int, centers: DoubleArray): IntArray {
    val clusterCount = centers.size / CHANNELS
    val selected = arrayListOf(seedCluster)
    val used = BooleanArray(clusterCount)
    used[seedCluster] = true
    while (selected.size < size) {
        var best = -1
        var bestDot = Double.NEGATIVE_INFINITY
        for (candidate in 0 until clusterCount) {
            if (used[candidate]) continue
            val candidateOffset = candidate * CHANNELS
            var adjacency = Double.NEGATIVE_INFINITY
            for (cluster in selected) {
                val offset = cluster * CHANNELS
                adjacency = max(
                    adjacency,
                    centers[offset] * centers[candidateOffset] +
                        centers[offset + 1] * centers[candidateOffset + 1] +
                        centers[offset + 2] * centers[candidateOffset + 2]
                )
            }
            if (adjacency > bestDot || adjacency == bestDot && candidate < best) {
                best = candidate
                bestDot = adjacency
            }
        }
        require(best >= 0)
        used[best] = true
        selected += best
    }
    return selected.sorted().toIntArray()
}

fun randomNeighborhood(clusterCount: Int, size: Int, seed: Long): IntArray {
    val order = IntArray(clusterCount) { it }
    val random = DeterministicRandom(seed)
    for (index in order.lastIndex downTo 1) {
        val other = random.nextInt(index + 1)
        val temporary = order[index]
        order[index] = order[other]
        order[other] = temporary
    }
    return order.copyOf(size).sortedArray()
}

fun selectNeighborhood(
    directions: DoubleArray,
    vectorBase: Int,
    labels: IntArray,
    clusterCount: Int,
    size: Int,
    attempt: Int,
    eliteLabels: List<IntArray>,
    scheduleSeed: Long
): Pair<String, IntArray> {
    val geometry = computeClusterGeometry(directions, vectorBase, labels, clusterCount)
    val quality = rankedClusters(clusterQualityScores(directions, vectorBase, labels, geometry))
    val disagreement = rankedClusters(eliteDisagreementScores(labels, clusterCount, eliteLabels))
    return when (attempt % 4) {
        0 -> "center-adjacency" to adjacentNeighborhood(attempt % clusterCount, size, geometry.centers)
        1 -> "high-loss-elongated-horizon" to
            adjacentNeighborhood(quality[(attempt / 4) % clusterCount], size, geometry.centers)
        2 -> "elite-disagreement" to
            adjacentNeighborhood(disagreement[(attempt / 4) % clusterCount], size, geometry.centers)
        else -> "recorded-random" to randomNeighborhood(
            clusterCount, size, mix64(scheduleSeed xor attempt.toLong())
        )
    }
}

fun reconvergeNeighborhoodStage(
    directions: DoubleArray,
    vectorBase: Int,
    labels: IntArray,
    clusterCount: Int,
    capacity: Int,
    quantization: Long
): Triple<String, Boolean, Double> {
    val before = computeClusterGeometry(directions, vectorBase, labels, clusterCount).explicitSquaredChord
    val centers = normalizedCentersForLabels(directions, vectorBase, labels, clusterCount, capacity)
    val result = balancedLloyd(
        directions, vectorBase, labels.size, clusterCount, capacity, quantization,
        BALANCED_MAX_ITERATIONS, CenterInitialization(centers, 0)
    )
    val accepted = result.objective <= before
    if (accepted) result.labels.copyInto(labels)
    return Triple(result.status, accepted, if (accepted) result.objective else before)
}

fun runMultiClusterStage(
    directions: DoubleArray,
    vectorBase: Int,
    initialLabels: IntArray,
    clusterCount: Int,
    capacity: Int,
    neighborhoodSize: Int,
    attemptBudget: Int,
    eliteLabels: List<IntArray>,
    quantization: Long,
    scheduleSeed: Long,
    tolerance: Double = 1e-12
): NeighborhoodStageResult {
    require(neighborhoodSize == 4 || neighborhoodSize == 8)
    require(attemptBudget > 0)
    val labels = initialLabels.copyOf()
    val initialObjective = computeClusterGeometry(
        directions, vectorBase, labels, clusterCount
    ).explicitSquaredChord
    var objective = initialObjective
    var acceptedAttempts = 0
    val attempts = arrayListOf<Map<String, Any?>>()
    val measurements = arrayListOf<Map<String, Any?>>()
    val acceptedCheckpoints = arrayListOf<IntArray>()
    for (attempt in 0 until attemptBudget) {
        val selection = selectNeighborhood(
            directions, vectorBase, labels, clusterCount, neighborhoodSize,
            attempt, eliteLabels, scheduleSeed
        )
        val clusters = selection.second
        val geometry = computeClusterGeometry(directions, vectorBase, labels, clusterCount)
        val members = buildMembers(labels, geometry.counts)
        val samples = clusters.flatMap { members[it].asIterable() }.sorted().toIntArray()
        val currentLocalLabels = pooledLabels(labels, samples, clusters)
        val localBefore = computeClusterGeometry(
            pooledDirections(directions, vectorBase, samples), 0, currentLocalLabels, neighborhoodSize
        ).explicitSquaredChord
        val attemptSeed = mix64(
            scheduleSeed xor neighborhoodSize.toLong().shl(48) xor attempt.toLong()
        )
        val heapBefore = usedJvmHeapBytes()
        val start = System.nanoTime()
        val candidate = pooledNeighborhoodRefinement(
            directions, vectorBase, labels, clusters, capacity, quantization, attemptSeed
        )
        val wallTimeNanos = System.nanoTime() - start
        val heapAfter = usedJvmHeapBytes()
        val candidateLabels = labels.copyOf()
        for (index in samples.indices) candidateLabels[samples[index]] = clusters[candidate.labels[index]]
        val candidateGeometry = computeClusterGeometry(
            directions, vectorBase, candidateLabels, clusterCount
        )
        require(candidateGeometry.counts.all { it == capacity })
        val objectiveBefore = objective
        val accepted = candidateGeometry.explicitSquaredChord < objectiveBefore - tolerance
        if (accepted) {
            candidateLabels.copyInto(labels)
            objective = candidateGeometry.explicitSquaredChord
            acceptedAttempts++
            acceptedCheckpoints += labels.copyOf()
        }
        attempts += linkedMapOf(
            "index" to attempt,
            "strategy" to selection.first,
            "clusters" to clusters,
            "seed" to unsignedHex(attemptSeed),
            "objectiveBefore" to objectiveBefore,
            "pooledObjectiveBefore" to localBefore,
            "pooledCandidateObjective" to candidate.objective,
            "candidateObjective" to candidateGeometry.explicitSquaredChord,
            "objectiveAfter" to objective,
            "acceptedDelta" to if (accepted) objective - objectiveBefore else 0.0,
            "accepted" to accepted,
            "solverStatus" to candidate.status,
            "solverConverged" to candidate.converged,
            "starts" to candidate.starts
        )
        measurements += linkedMapOf(
            "index" to attempt,
            "neighborhoodSize" to neighborhoodSize,
            "wallTimeNanos" to wallTimeNanos,
            "heapBytesBefore" to heapBefore,
            "heapBytesAfter" to heapAfter,
            "heapEndpointMaximumBytes" to max(heapBefore, heapAfter)
        )
    }
    var reconvergenceStatus = "NOT_RUN_NO_ACCEPTED_NEIGHBORHOOD"
    var reconvergenceAccepted = false
    if (acceptedAttempts > 0) {
        val before = objective
        val reconvergence = reconvergeNeighborhoodStage(
            directions, vectorBase, labels, clusterCount, capacity, quantization
        )
        reconvergenceStatus = reconvergence.first
        reconvergenceAccepted = reconvergence.second
        objective = reconvergence.third
        require(objective <= before)
        if (acceptedCheckpoints.lastOrNull()?.contentEquals(labels) != true) {
            acceptedCheckpoints += labels.copyOf()
        }
    }
    val finalGeometry = computeClusterGeometry(directions, vectorBase, labels, clusterCount)
    require(finalGeometry.counts.all { it == capacity })
    require(finalGeometry.explicitSquaredChord <= initialObjective)
    assertClose("multi-cluster stage objective", objective, finalGeometry.explicitSquaredChord, 1e-10)
    return NeighborhoodStageResult(
        labels,
        initialObjective,
        objective,
        neighborhoodSize,
        attemptBudget,
        acceptedAttempts,
        attempts,
        measurements,
        acceptedCheckpoints,
        reconvergenceStatus,
        reconvergenceAccepted
    )
}

fun runPairStage(
    directions: DoubleArray,
    vectorBase: Int,
    initialLabels: IntArray,
    clusterCount: Int,
    capacity: Int,
    attemptBudget: Int,
    quantization: Long,
    tolerance: Double = 1e-12
): NeighborhoodStageResult {
    val labels = initialLabels.copyOf()
    val initialObjective = computeClusterGeometry(
        directions, vectorBase, labels, clusterCount
    ).explicitSquaredChord
    val heapBefore = usedJvmHeapBytes()
    val start = System.nanoTime()
    val result = runPairRepartition(
        directions, vectorBase, labels, clusterCount, capacity, tolerance, attemptBudget
    )
    val wallTimeNanos = System.nanoTime() - start
    val heapAfter = usedJvmHeapBytes()
    var objective = computeClusterGeometry(directions, vectorBase, labels, clusterCount).explicitSquaredChord
    val acceptedCheckpoints = arrayListOf<IntArray>()
    if (result.first > 0) acceptedCheckpoints += labels.copyOf()
    var reconvergenceStatus = "NOT_RUN_NO_ACCEPTED_NEIGHBORHOOD"
    var reconvergenceAccepted = false
    if (result.first > 0) {
        val before = objective
        val reconvergence = reconvergeNeighborhoodStage(
            directions, vectorBase, labels, clusterCount, capacity, quantization
        )
        reconvergenceStatus = reconvergence.first
        reconvergenceAccepted = reconvergence.second
        objective = reconvergence.third
        require(objective <= before)
        if (acceptedCheckpoints.lastOrNull()?.contentEquals(labels) != true) {
            acceptedCheckpoints += labels.copyOf()
        }
    }
    return NeighborhoodStageResult(
        labels,
        initialObjective,
        objective,
        2,
        attemptBudget,
        result.first,
        listOf(linkedMapOf(
            "strategy" to "T05-neighbor-pair-repartition",
            "candidateClusterPairs" to result.third,
            "acceptedMoves" to result.first,
            "objectiveBefore" to initialObjective,
            "objectiveAfter" to objective,
            "acceptedDelta" to objective - initialObjective,
            "solverStatus" to "EXACT_FIXED_CENTER_M_OF_2M_SORT"
        )),
        listOf(linkedMapOf(
            "index" to 0,
            "neighborhoodSize" to 2,
            "wallTimeNanos" to wallTimeNanos,
            "heapBytesBefore" to heapBefore,
            "heapBytesAfter" to heapAfter,
            "heapEndpointMaximumBytes" to max(heapBefore, heapAfter)
        )),
        acceptedCheckpoints,
        reconvergenceStatus,
        reconvergenceAccepted
    )
}

fun neighborhoodStageJson(result: NeighborhoodStageResult): Map<String, Any?> = linkedMapOf(
    "neighborhoodSize" to result.neighborhoodSize,
    "attemptBudget" to result.attemptBudget,
    "acceptedAttempts" to result.acceptedAttempts,
    "initialObjective" to result.initialObjective,
    "finalObjective" to result.finalObjective,
    "gain" to result.initialObjective - result.finalObjective,
    "labelsSha256" to labelsSha256(result.labels),
    "reconvergenceStatus" to result.reconvergenceStatus,
    "reconvergenceAccepted" to result.reconvergenceAccepted,
    "attempts" to result.attempts
)

fun neighborhoodMeasurementsJson(result: NeighborhoodStageResult): Map<String, Any?> = linkedMapOf(
    "neighborhoodSize" to result.neighborhoodSize,
    "attemptBudget" to result.attemptBudget,
    "attempts" to result.measurements
)

fun noisyPlantedDirections(
    random: DeterministicRandom,
    clusterCount: Int,
    capacity: Int,
    noise: Double
): DoubleArray {
    val centers = randomUnitDirections(random, clusterCount)
    val directions = DoubleArray(clusterCount * capacity * CHANNELS)
    for (cluster in 0 until clusterCount) for (member in 0 until capacity) {
        val centerOffset = cluster * CHANNELS
        val sampleOffset = (cluster * capacity + member) * CHANNELS
        val direction = normalizedVector(
            centers[centerOffset] + noise * (2.0 * random.nextDouble() - 1.0),
            centers[centerOffset + 1] + noise * (2.0 * random.nextDouble() - 1.0),
            max(1e-4, centers[centerOffset + 2] + noise * (2.0 * random.nextDouble() - 1.0))
        )
        direction.copyInto(directions, sampleOffset)
    }
    return directions
}

fun shuffledBalancedLabels(random: DeterministicRandom, clusterCount: Int, capacity: Int): IntArray {
    val sampleCount = clusterCount * capacity
    val samples = IntArray(sampleCount) { it }
    for (index in samples.lastIndex downTo 1) {
        val other = random.nextInt(index + 1)
        val temporary = samples[index]
        samples[index] = samples[other]
        samples[other] = temporary
    }
    val labels = IntArray(sampleCount)
    for (order in samples.indices) labels[samples[order]] = order / capacity
    return labels
}

fun runNeighborhoodSelfTests() {
    var checks = 0
    fun checked(action: () -> Unit) {
        action()
        checks++
    }
    val random = DeterministicRandom(0x4E45494748424F52L)
    val scheduleDirections = noisyPlantedDirections(random, 8, 4, 0.08)
    val scheduleLabels = IntArray(32) { it / 4 }
    val scheduleFirst = (0 until 8).map { attempt ->
        selectNeighborhood(
            scheduleDirections, 0, scheduleLabels, 8, 4, attempt,
            listOf(scheduleLabels), 0x5343484544554C45L
        )
    }
    val scheduleSecond = (0 until 8).map { attempt ->
        selectNeighborhood(
            scheduleDirections, 0, scheduleLabels, 8, 4, attempt,
            listOf(scheduleLabels), 0x5343484544554C45L
        )
    }
    checked {
        require(scheduleFirst.zip(scheduleSecond).all { (first, second) ->
            first.first == second.first && first.second.contentEquals(second.second)
        })
        require(scheduleFirst.map { it.first }.toSet() == setOf(
            "center-adjacency",
            "high-loss-elongated-horizon",
            "elite-disagreement",
            "recorded-random"
        ))
    }

    val noImprovementDirections = DoubleArray(16 * CHANNELS)
    val exactCenters = arrayOf(
        normalizedVector(0.8, 0.0, 0.6),
        normalizedVector(-0.8, 0.0, 0.6),
        normalizedVector(0.0, 0.8, 0.6),
        normalizedVector(0.0, -0.8, 0.6)
    )
    val noImprovementLabels = IntArray(16) { it / 4 }
    for (sample in noImprovementLabels.indices) {
        exactCenters[noImprovementLabels[sample]].copyInto(noImprovementDirections, sample * CHANNELS)
    }
    val noImprovement = runMultiClusterStage(
        noImprovementDirections,
        0,
        noImprovementLabels,
        4,
        4,
        4,
        4,
        listOf(noImprovementLabels),
        DEFAULT_QUANTIZATION,
        0x4E4F4348414E4745L
    )
    checked {
        require(noImprovement.labels.contentEquals(noImprovementLabels))
        require(noImprovement.acceptedAttempts == 0)
        require(noImprovement.finalObjective == 0.0)
    }

    var plantedPairObjective = Double.NaN
    var plantedFourObjective = Double.NaN
    var plantedTrial = -1
    for (trial in 0 until 96) {
        val directions = noisyPlantedDirections(random, 4, 8, 0.22)
        val labels = shuffledBalancedLabels(random, 4, 8)
        val pairOnly = refineLabels(
            directions, 0, labels, 4, 8, DEFAULT_QUANTIZATION, setOf("pair"), false
        )
        val four = runMultiClusterStage(
            directions,
            0,
            pairOnly.labels,
            4,
            8,
            4,
            4,
            listOf(labels, pairOnly.labels),
            DEFAULT_QUANTIZATION,
            mix64(0x504C414E54454434L xor trial.toLong())
        )
        if (four.finalObjective < pairOnly.finalObjective - 1e-9) {
            plantedPairObjective = pairOnly.finalObjective
            plantedFourObjective = four.finalObjective
            plantedTrial = trial
            break
        }
    }
    checked {
        require(plantedTrial >= 0) { "No deterministic planted case escaped pair-only refinement" }
        require(plantedFourObjective < plantedPairObjective)
    }

    val eightDirections = noisyPlantedDirections(random, 8, 4, 0.16)
    val eightLabels = shuffledBalancedLabels(random, 8, 4)
    val firstEight = runMultiClusterStage(
        eightDirections, 0, eightLabels, 8, 4, 8, 4,
        listOf(eightLabels), DEFAULT_QUANTIZATION, 0x45494748545F5638L
    )
    val secondEight = runMultiClusterStage(
        eightDirections, 0, eightLabels, 8, 4, 8, 4,
        listOf(eightLabels), DEFAULT_QUANTIZATION, 0x45494748545F5638L
    )
    checked {
        require(firstEight.labels.contentEquals(secondEight.labels))
        require(canonicalJson(neighborhoodStageJson(firstEight)).contentEquals(
            canonicalJson(neighborhoodStageJson(secondEight))))
        require(firstEight.finalObjective <= firstEight.initialObjective)
        require(computeClusterGeometry(eightDirections, 0, firstEight.labels, 8).counts.all { it == 4 })
        require(firstEight.attempts.all { (it["starts"] as List<*>).size == 5 })
    }

    val receipt = linkedMapOf<String, Any?>(
        "schema" to "alpha-piscium/stbn-clustering-neighborhood-self-test/v1",
        "checks" to checks,
        "scheduleStrategies" to scheduleFirst.map { it.first }.distinct(),
        "noImprovementByteIdentical" to true,
        "plantedPairOnlyObjective" to plantedPairObjective,
        "plantedFourClusterObjective" to plantedFourObjective,
        "plantedTrial" to plantedTrial,
        "eightClusterDeterministic" to true,
        "freshStartsPerNeighborhood" to 4,
        "currentCenterStartsPerNeighborhood" to 1,
        "fullEdgeTransport" to true,
        "capacityAndIncumbentOnlyAcceptance" to true
    )
    val bytes = canonicalJson(receipt)
    println("neighborhood-self-test.json|${bytes.size}|${sha256(bytes)}")
    print(bytes.toString(StandardCharsets.UTF_8))
    println("PASS neighborhood self-test: $checks checks")
}

fun squaredAngularLoss(
    directions: DoubleArray,
    vectorBase: Int,
    members: IntArray,
    center: DoubleArray
): Double {
    var objective = 0.0
    for (sample in members) {
        val offset = vectorBase + sample * CHANNELS
        val dot = (directions[offset] * center[0] + directions[offset + 1] * center[1] +
            directions[offset + 2] * center[2]).coerceIn(-1.0, 1.0)
        val angle = acos(dot)
        objective += angle * angle
    }
    return objective
}

fun karcherCenter(
    directions: DoubleArray,
    vectorBase: Int,
    members: IntArray,
    maxIterations: Int = 96,
    gradientTolerance: Double = 1e-10
): AngularCenterResult {
    require(members.isNotEmpty() && maxIterations > 0 && gradientTolerance > 0.0)
    var sumX = 0.0
    var sumY = 0.0
    var sumZ = 0.0
    for (sample in members) {
        val offset = vectorBase + sample * CHANNELS
        sumX += directions[offset]
        sumY += directions[offset + 1]
        sumZ += directions[offset + 2]
    }
    var center = normalizedVector(sumX, sumY, sumZ)
    var objective = squaredAngularLoss(directions, vectorBase, members, center)
    for (iteration in 0 until maxIterations) {
        var tangentX = 0.0
        var tangentY = 0.0
        var tangentZ = 0.0
        for (sample in members) {
            val offset = vectorBase + sample * CHANNELS
            val dot = (directions[offset] * center[0] + directions[offset + 1] * center[1] +
                directions[offset + 2] * center[2]).coerceIn(-1.0, 1.0)
            val angle = acos(dot)
            val scale = if (angle < 1e-10) 1.0 else angle / max(sin(angle), 1e-300)
            tangentX += scale * (directions[offset] - dot * center[0])
            tangentY += scale * (directions[offset + 1] - dot * center[1])
            tangentZ += scale * (directions[offset + 2] - dot * center[2])
        }
        tangentX /= members.size
        tangentY /= members.size
        tangentZ /= members.size
        val radial = tangentX * center[0] + tangentY * center[1] + tangentZ * center[2]
        tangentX -= radial * center[0]
        tangentY -= radial * center[1]
        tangentZ -= radial * center[2]
        val gradientNorm = sqrt(tangentX * tangentX + tangentY * tangentY + tangentZ * tangentZ)
        if (gradientNorm <= gradientTolerance) {
            return AngularCenterResult(center, objective, iteration, "CONVERGED_GRADIENT", true)
        }
        var step = 1.0
        var acceptedCenter: DoubleArray? = null
        var acceptedObjective = objective
        while (step >= 9.313225746154785e-10) {
            val distance = step * gradientNorm
            val tangentScale = sin(distance) / gradientNorm
            val candidate = normalizedVector(
                cos(distance) * center[0] + tangentScale * tangentX,
                cos(distance) * center[1] + tangentScale * tangentY,
                cos(distance) * center[2] + tangentScale * tangentZ
            )
            val candidateObjective = squaredAngularLoss(directions, vectorBase, members, candidate)
            if (candidateObjective < objective) {
                acceptedCenter = candidate
                acceptedObjective = candidateObjective
                break
            }
            step *= 0.5
        }
        if (acceptedCenter == null) {
            val converged = gradientNorm <= 1e-8
            return AngularCenterResult(
                center,
                objective,
                iteration,
                if (converged) "CONVERGED_NUMERIC_STEP" else "LINE_SEARCH_STALLED",
                converged
            )
        }
        center = acceptedCenter
        objective = acceptedObjective
    }
    return AngularCenterResult(center, objective, maxIterations, "MAX_ITERATIONS", false)
}

fun angularGeometry(
    directions: DoubleArray,
    vectorBase: Int,
    labels: IntArray,
    clusterCount: Int
): AngularGeometry {
    val counts = IntArray(clusterCount)
    for (cluster in labels) {
        require(cluster in 0 until clusterCount)
        counts[cluster]++
    }
    require(counts.all { it > 0 })
    val members = buildMembers(labels, counts)
    val centers = DoubleArray(clusterCount * CHANNELS)
    var objective = 0.0
    var iterations = 0
    var converged = true
    val statuses = linkedMapOf<String, Int>()
    for (cluster in 0 until clusterCount) {
        val result = karcherCenter(directions, vectorBase, members[cluster])
        result.center.copyInto(centers, cluster * CHANNELS)
        objective += result.objective
        iterations += result.iterations
        statuses[result.status] = (statuses[result.status] ?: 0) + 1
        converged = converged && result.converged
    }
    return AngularGeometry(counts, centers, objective, iterations, statuses, converged)
}

fun angularFixedLoss(
    directions: DoubleArray,
    vectorBase: Int,
    labels: IntArray,
    centers: DoubleArray
): Double {
    var objective = 0.0
    for (sample in labels.indices) {
        val offset = vectorBase + sample * CHANNELS
        val centerOffset = labels[sample] * CHANNELS
        val dot = (directions[offset] * centers[centerOffset] +
            directions[offset + 1] * centers[centerOffset + 1] +
            directions[offset + 2] * centers[centerOffset + 2]).coerceIn(-1.0, 1.0)
        val angle = acos(dot)
        objective += angle * angle
    }
    return objective
}

fun angularBalancedLloyd(
    directions: DoubleArray,
    vectorBase: Int,
    initialLabels: IntArray,
    clusterCount: Int,
    capacity: Int,
    quantization: Long,
    maxIterations: Int
): AngularLloydResult {
    require(clusterCount * capacity == initialLabels.size && maxIterations > 0)
    var labels = initialLabels.copyOf()
    var geometry = angularGeometry(directions, vectorBase, labels, clusterCount)
    require(geometry.counts.all { it == capacity } && geometry.converged) {
        "Initial angular centers failed: counts=${geometry.counts.minOrNull()}..${geometry.counts.maxOrNull()} " +
            "statuses=${geometry.centerStatusCounts}"
    }
    var bestLabels = labels.copyOf()
    var bestGeometry = geometry
    var status = "MAX_ITERATIONS"
    var converged = false
    var completedIterations = 0
    val seen = linkedSetOf(partitionSha256(labels, clusterCount))
    val history = arrayListOf<Map<String, Any?>>()
    var transportTotal = TransportStats(0, 0, 0, 0, 0, 0)
    for (iteration in 1..maxIterations) {
        val centers = geometry.centers
        val transport = balancedTransport(
            directions,
            vectorBase,
            labels.size,
            centers,
            clusterCount,
            capacity,
            quantization,
            costFunction = { dot -> quantizedSquaredAngle(dot, quantization) }
        )
        requireOptimalTransport(transport, labels.size, clusterCount, capacity)
        transportTotal += transport.stats
        val fixedObjective = angularFixedLoss(directions, vectorBase, transport.labels, centers)
        val candidateGeometry = angularGeometry(directions, vectorBase, transport.labels, clusterCount)
        require(candidateGeometry.converged) { "Karcher center update did not converge" }
        require(candidateGeometry.explicitSquaredAngle <= fixedObjective + 1e-10 * max(1.0, fixedObjective))
        val stable = labels.contentEquals(transport.labels)
        val partition = partitionSha256(transport.labels, clusterCount)
        val repeated = partition in seen
        val tolerance = 1e-12 * max(1.0, geometry.explicitSquaredAngle)
        val accepted = !stable && !repeated &&
            candidateGeometry.explicitSquaredAngle < geometry.explicitSquaredAngle - tolerance
        history += linkedMapOf(
            "iteration" to iteration,
            "fixedCenterObjective" to fixedObjective,
            "updatedCenterObjective" to candidateGeometry.explicitSquaredAngle,
            "objectiveBefore" to geometry.explicitSquaredAngle,
            "accepted" to accepted,
            "labelStable" to stable,
            "repeatedPartition" to repeated,
            "partitionSha256" to partition,
            "centerIterations" to candidateGeometry.centerIterations,
            "centerStatusCounts" to candidateGeometry.centerStatusCounts,
            "transport" to transportStatsJson(transport.stats)
        )
        completedIterations = iteration
        if (stable) {
            status = "CONVERGED_LABELS_AND_ANGULAR_OBJECTIVE"
            converged = true
            break
        }
        if (repeated) {
            status = "CYCLE"
            break
        }
        if (!accepted) {
            status = if (candidateGeometry.explicitSquaredAngle <= geometry.explicitSquaredAngle + tolerance)
                "NO_SIGNIFICANT_ANGULAR_IMPROVEMENT"
            else "ORIGINAL_ANGULAR_OBJECTIVE_INCREASE"
            break
        }
        labels = transport.labels
        geometry = candidateGeometry
        seen += partition
        if (geometry.explicitSquaredAngle < bestGeometry.explicitSquaredAngle ||
            geometry.explicitSquaredAngle == bestGeometry.explicitSquaredAngle &&
            partition < partitionSha256(bestLabels, clusterCount)) {
            bestLabels = labels.copyOf()
            bestGeometry = geometry
        }
    }
    return AngularLloydResult(
        bestLabels,
        bestGeometry,
        status,
        converged,
        completedIterations,
        history,
        transportTotal
    )
}

fun capsForLabels(
    directions: DoubleArray,
    vectorBase: Int,
    labels: IntArray,
    clusterCount: Int
): List<CapResult> {
    val counts = IntArray(clusterCount)
    for (cluster in labels) counts[cluster]++
    require(counts.all { it > 0 })
    val members = buildMembers(labels, counts)
    return List(clusterCount) { minimumSphericalCap(directions, vectorBase, members[it]) }
}

fun radiusConstrainedLloyd(
    directions: DoubleArray,
    vectorBase: Int,
    initialLabels: IntArray,
    clusterCount: Int,
    capacity: Int,
    targetRadiusDeg: Double,
    maxIterations: Int,
    quantization: Long = 10_000L
): RadiusLloydResult {
    require(clusterCount * capacity == initialLabels.size && targetRadiusDeg in 0.0..180.0)
    val targetCos = cos(targetRadiusDeg / RAD_TO_DEG)
    val forbiddenPenalty = Math.addExact(
        Math.multiplyExact(Math.multiplyExact(4L, quantization), initialLabels.size.toLong()),
        1L
    )
    var labels = initialLabels.copyOf()
    var caps = capsForLabels(directions, vectorBase, labels, clusterCount)
    var bestLabels: IntArray? = null
    var bestRadius = Double.POSITIVE_INFINITY
    var bestChord = Double.POSITIVE_INFINITY
    if (caps.maxOf { it.radiusDeg } <= targetRadiusDeg + 1e-8) {
        bestLabels = labels.copyOf()
        bestRadius = caps.maxOf { it.radiusDeg }
        bestChord = computeClusterGeometry(directions, vectorBase, labels, clusterCount).explicitSquaredChord
    }
    val history = arrayListOf<Map<String, Any?>>()
    val seen = linkedSetOf(partitionSha256(labels, clusterCount))
    var transportTotal = TransportStats(0, 0, 0, 0, 0, 0)
    var status = "MAX_ITERATIONS"
    var completedIterations = 0
    for (iteration in 1..maxIterations) {
        val centers = DoubleArray(clusterCount * CHANNELS)
        for (cluster in 0 until clusterCount) caps[cluster].center.copyInto(centers, cluster * CHANNELS)
        val transport = balancedTransport(
            directions,
            vectorBase,
            labels.size,
            centers,
            clusterCount,
            capacity,
            quantization,
            costFunction = { dot ->
                quantizedSquaredChord(dot, quantization) +
                    if (dot < targetCos - CAP_CONTAINMENT_TOLERANCE) forbiddenPenalty else 0L
            }
        )
        requireOptimalTransport(transport, labels.size, clusterCount, capacity)
        transportTotal += transport.stats
        var violations = 0
        for (sample in transport.labels.indices) {
            val offset = vectorBase + sample * CHANNELS
            val centerOffset = transport.labels[sample] * CHANNELS
            val dot = directions[offset] * centers[centerOffset] +
                directions[offset + 1] * centers[centerOffset + 1] +
                directions[offset + 2] * centers[centerOffset + 2]
            if (dot < targetCos - CAP_CONTAINMENT_TOLERANCE) violations++
        }
        completedIterations = iteration
        if (violations > 0) {
            history += linkedMapOf(
                "iteration" to iteration,
                "status" to "NOT_FOUND_FOR_CURRENT_CENTERS",
                "forbiddenAssignments" to violations,
                "transport" to transportStatsJson(transport.stats)
            )
            status = "NOT_FOUND_FOR_CURRENT_CENTERS"
            break
        }
        val candidateCaps = capsForLabels(directions, vectorBase, transport.labels, clusterCount)
        val candidateRadius = candidateCaps.maxOf { it.radiusDeg }
        require(candidateRadius <= targetRadiusDeg + 1e-7) {
            "Radius center update exceeds target: $candidateRadius > $targetRadiusDeg"
        }
        val candidateChord = computeClusterGeometry(
            directions, vectorBase, transport.labels, clusterCount
        ).explicitSquaredChord
        val partition = partitionSha256(transport.labels, clusterCount)
        val stable = labels.contentEquals(transport.labels)
        val repeated = partition in seen
        val accepted = candidateRadius < bestRadius - 1e-12 ||
            abs(candidateRadius - bestRadius) <= 1e-12 && candidateChord < bestChord
        if (accepted) {
            bestLabels = transport.labels.copyOf()
            bestRadius = candidateRadius
            bestChord = candidateChord
        }
        history += linkedMapOf(
            "iteration" to iteration,
            "status" to "FEASIBLE",
            "targetRadiusDeg" to targetRadiusDeg,
            "maximumRadiusDeg" to candidateRadius,
            "chordObjective" to candidateChord,
            "acceptedIncumbent" to accepted,
            "labelStable" to stable,
            "repeatedPartition" to repeated,
            "partitionSha256" to partition,
            "transport" to transportStatsJson(transport.stats)
        )
        if (stable) {
            status = "CONVERGED_LABELS_AND_CAP_CENTERS"
            break
        }
        if (repeated) {
            status = "CYCLE"
            break
        }
        labels = transport.labels
        caps = candidateCaps
        seen += partition
    }
    return RadiusLloydResult(
        bestLabels ?: initialLabels.copyOf(),
        targetRadiusDeg,
        if (bestLabels == null) caps.maxOf { it.radiusDeg } else bestRadius,
        if (bestLabels == null) computeClusterGeometry(
            directions, vectorBase, initialLabels, clusterCount
        ).explicitSquaredChord else bestChord,
        status,
        bestLabels != null,
        completedIterations,
        history,
        transportTotal
    )
}

fun runAngularTailSelfTests() {
    var checks = 0
    fun checked(action: () -> Unit) {
        action()
        checks++
    }
    fun meridianDirections(angles: DoubleArray): DoubleArray {
        val directions = DoubleArray(angles.size * CHANNELS)
        for (index in angles.indices) {
            directions[index * CHANNELS] = sin(angles[index])
            directions[index * CHANNELS + 1] = 0.0
            directions[index * CHANNELS + 2] = cos(angles[index])
        }
        return directions
    }

    val pairAngle = PI / 3.0
    val pair = meridianDirections(doubleArrayOf(0.0, pairAngle))
    val pairCenter = karcherCenter(pair, 0, intArrayOf(0, 1))
    checked {
        val expectedCenter = doubleArrayOf(sin(pairAngle / 2.0), 0.0, cos(pairAngle / 2.0))
        val expectedObjective = 2.0 * (pairAngle / 2.0) * (pairAngle / 2.0)
        require(pairCenter.converged)
        assertClose("analytic two-point Karcher x", pairCenter.center[0], expectedCenter[0], 1e-13)
        assertClose("analytic two-point Karcher z", pairCenter.center[2], expectedCenter[2], 1e-13)
        assertClose("analytic two-point angular objective", pairCenter.objective, expectedObjective, 1e-13)
    }

    val planarAngles = doubleArrayOf(0.08, 0.19, 0.31, 0.52, 0.74)
    val planar = meridianDirections(planarAngles)
    val planarCenter = karcherCenter(planar, 0, planarAngles.indices.toList().toIntArray())
    val referenceAngle = planarAngles.average()
    val referenceObjective = planarAngles.sumOf { (it - referenceAngle) * (it - referenceAngle) }
    checked {
        require(planarCenter.converged)
        assertClose("analytic meridian Karcher x", planarCenter.center[0], sin(referenceAngle), 2e-13)
        assertClose("analytic meridian Karcher z", planarCenter.center[2], cos(referenceAngle), 2e-13)
        assertClose("analytic meridian angular objective", planarCenter.objective, referenceObjective, 2e-13)
    }

    val transportDirections = meridianDirections(doubleArrayOf(0.02, 0.09, 0.18, 0.72, 0.81, 0.93))
    val transportCenters = meridianDirections(doubleArrayOf(0.15, 0.82))
    val angularTransport = balancedTransport(
        transportDirections, 0, 6, transportCenters, 2, 3, DEFAULT_QUANTIZATION,
        costFunction = { dot -> quantizedSquaredAngle(dot, DEFAULT_QUANTIZATION) }
    )
    val angularEdges = LongArray(12)
    for (sample in 0 until 6) for (cluster in 0 until 2) {
        val dot = transportDirections[sample * CHANNELS] * transportCenters[cluster * CHANNELS] +
            transportDirections[sample * CHANNELS + 1] * transportCenters[cluster * CHANNELS + 1] +
            transportDirections[sample * CHANNELS + 2] * transportCenters[cluster * CHANNELS + 2]
        angularEdges[sample * 2 + cluster] = quantizedSquaredAngle(dot, DEFAULT_QUANTIZATION)
    }
    val angularOracle = exhaustiveBalancedQuantized(angularEdges, 6, 2, 3)
    checked {
        requireOptimalTransport(angularTransport, 6, 2, 3)
        require(angularTransport.quantizedCost == angularOracle.cost)
    }

    val planted = meridianDirections(doubleArrayOf(
        -0.03, 0.0, 0.04,
        0.46, 0.50, 0.55
    ))
    val initialLabels = intArrayOf(0, 1, 0, 1, 0, 1)
    val angular = angularBalancedLloyd(
        planted, 0, initialLabels, 2, 3, DEFAULT_QUANTIZATION, 8
    )
    checked {
        require(angular.geometry.converged)
        require(angular.geometry.counts.all { it == 3 })
        require(angular.geometry.explicitSquaredAngle <=
            angularGeometry(planted, 0, initialLabels, 2).explicitSquaredAngle)
    }

    val separatedLabels = intArrayOf(0, 0, 0, 1, 1, 1)
    val caps = capsForLabels(planted, 0, separatedLabels, 2)
    val radius = caps.maxOf { it.radiusDeg }
    val feasible = radiusConstrainedLloyd(planted, 0, separatedLabels, 2, 3, radius, 6)
    checked {
        require(feasible.feasible)
        require(feasible.maximumRadiusDeg <= radius + 1e-7)
        require(computeClusterGeometry(planted, 0, feasible.labels, 2).counts.all { it == 3 })
        val members = buildMembers(feasible.labels, intArrayOf(3, 3))
        for (cluster in 0 until 2) {
            val cap = minimumSphericalCap(planted, 0, members[cluster])
            for (sample in members[cluster]) {
                require(dotPointCap(
                    planted,
                    0,
                    sample,
                    SphericalCap(cap.center[0], cap.center[1], cap.center[2], cap.cosRadius, cap.supportSize)
                ) >= cap.cosRadius - CAP_CONTAINMENT_TOLERANCE)
            }
        }
    }
    val notFound = radiusConstrainedLloyd(planted, 0, separatedLabels, 2, 3, 0.0, 2)
    checked {
        require(!notFound.feasible)
        require(notFound.status == "NOT_FOUND_FOR_CURRENT_CENTERS")
    }

    val receipt = linkedMapOf<String, Any?>(
        "schema" to "alpha-piscium/stbn-clustering-angular-tail-self-test/v1",
        "checks" to checks,
        "karcherReferences" to listOf("analytic two-point midpoint", "analytic single-meridian arithmetic angle"),
        "trueSquaredAngularAssignmentMatchesExhaustive" to true,
        "angularLloydCapacityAndMonotonicity" to true,
        "minimumCapsContainEveryMember" to true,
        "radiusFeasibleAndNotFoundStatuses" to true,
        "encodedRadius" to null
    )
    val bytes = canonicalJson(receipt)
    println("angular-tail-self-test.json|${bytes.size}|${sha256(bytes)}")
    print(bytes.toString(StandardCharsets.UTF_8))
    println("PASS angular-tail self-test: $checks checks")
}

fun rotateCyclic(directions: DoubleArray): DoubleArray {
    val result = DoubleArray(directions.size)
    for (sample in 0 until directions.size / CHANNELS) {
        val offset = sample * CHANNELS
        result[offset] = directions[offset + 2]
        result[offset + 1] = directions[offset]
        result[offset + 2] = directions[offset + 1]
    }
    return result
}

fun coreMetricSignature(directions: DoubleArray, labels: IntArray, clusterCount: Int): DoubleArray {
    val geometry = computeClusterGeometry(directions, 0, labels, clusterCount)
    val members = buildMembers(labels, geometry.counts)
    val angles = DoubleArray(labels.size)
    for (sample in labels.indices) {
        val offset = sample * CHANNELS
        val centerOffset = labels[sample] * CHANNELS
        val dot = (directions[offset] * geometry.centers[centerOffset] +
            directions[offset + 1] * geometry.centers[centerOffset + 1] +
            directions[offset + 2] * geometry.centers[centerOffset + 2]).coerceIn(-1.0, 1.0)
        angles[sample] = acos(dot)
    }
    angles.sort()
    val caps = DoubleArray(clusterCount) {
        minimumSphericalCap(directions, 0, members[it]).radiusDeg
    }.apply { sort() }
    return doubleArrayOf(
        geometry.explicitSquaredChord,
        geometry.eliminatedSquaredChord,
        geometry.pairwiseSquaredChord,
        angles.sum(),
        angles.last(),
        caps.sum(),
        caps.last()
    )
}

fun runCoreSelfTests() {
    var checks = 0
    fun checked(action: () -> Unit) {
        action()
        checks++
    }

    checked {
        expectFailure("invalid dimensions") { decodeRgb8(ByteArray(3), 0, 1, 1) }
        expectFailure("invalid byte length") { decodeRgb8(ByteArray(5), 1, 1, 1) }
    }
    checked {
        val decoded = decodeRgb8(byteArrayOf(128.toByte(), 128.toByte(), 255.toByte()), 1, 1, 1)
        val length = sqrt(decoded.directions.sumOf { it * it })
        assertClose("normalization", length, 1.0, 1e-15)
        require(decoded.directions[2] > 0.999)
    }
    checked {
        val decoded = decodeRgb8(byteArrayOf(255.toByte(), 128.toByte(), 128.toByte()), 1, 1, 1)
        require(decoded.directions[2] in 0.0..0.01)
        expectFailure("negative hemisphere") {
            decodeRgb8(byteArrayOf(128.toByte(), 128.toByte(), 127.toByte()), 1, 1, 1)
        }
    }
    checked {
        validatePermutation(intArrayOf(2, 0, 1), 3)
        expectFailure("duplicate source ID") { validatePermutation(intArrayOf(0, 1, 1), 3) }
        expectFailure("out-of-range source ID") { validatePermutation(intArrayOf(0, 1, 3), 3) }
    }
    checked {
        val sorted = doubleArrayOf(1.0, 1.0, 2.0, 2.0)
        assertClose("tie quantile p50", quantileFromSorted(sorted, 0.5), 1.5, 0.0)
        assertClose("tie quantile p95", quantileFromSorted(sorted, 0.95), 2.0, 0.0)
    }
    checked {
        val seen = BooleanArray(16)
        val coordinates = Array(16) { IntArray(2) }
        for (y in 0..3) for (x in 0..3) {
            val index = hilbertIndex(x, y, 2).toInt()
            require(!seen[index])
            seen[index] = true
            coordinates[index][0] = x
            coordinates[index][1] = y
        }
        require(seen.all { it })
        for (index in 1 until coordinates.size) {
            val distance = abs(coordinates[index][0] - coordinates[index - 1][0]) +
                abs(coordinates[index][1] - coordinates[index - 1][1])
            require(distance == 1) { "Hilbert step $index is not adjacent" }
        }
    }
    val s = 1.0 / sqrt(2.0)
    val directions = doubleArrayOf(
        0.0, 0.0, 1.0,
        s, 0.0, s,
        0.0, s, s,
        -s, 0.0, s,
        0.0, -s, s,
        0.2, 0.0, sqrt(0.96)
    )
    val labels = intArrayOf(0, 0, 0, 1, 1, 1)
    checked {
        val geometry = computeClusterGeometry(directions, 0, labels, 2)
        assertClose(
            "explicit/eliminated squared chord identity",
            geometry.explicitSquaredChord,
            geometry.eliminatedSquaredChord,
            1e-12
        )
        val directPairwise = (0 until labels.size).sumOf { first ->
            ((first + 1) until labels.size).sumOf { second ->
                if (labels[first] != labels[second]) 0.0 else {
                    val a = first * CHANNELS
                    val b = second * CHANNELS
                    val dx = directions[a] - directions[b]
                    val dy = directions[a + 1] - directions[b + 1]
                    val dz = directions[a + 2] - directions[b + 2]
                    dx * dx + dy * dy + dz * dz
                }
            }
        }
        assertClose("pairwise identity", geometry.pairwiseSquaredChord, directPairwise, 1e-12)
    }
    checked {
        val original = coreMetricSignature(directions, labels, 2)
        val rotated = coreMetricSignature(rotateCyclic(directions), labels, 2)
        for (index in original.indices) assertClose("rotation invariant metric $index", rotated[index], original[index], 1e-10)
    }
    checked {
        val relabeled = IntArray(labels.size) { 1 - labels[it] }
        val original = coreMetricSignature(directions, labels, 2)
        val swapped = coreMetricSignature(directions, relabeled, 2)
        for (index in original.indices) assertClose("label invariant metric $index", swapped[index], original[index], 1e-12)
    }
    checked {
        val tiedCenters = doubleArrayOf(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        require(nearestCenter(1.0, 0.0, 0.0, tiedCenters, 2).first == 0)
    }
    checked {
        val opposite = doubleArrayOf(1.0, 0.0, 0.0, -1.0, 0.0, 0.0)
        expectFailure("degenerate center sum") { computeClusterGeometry(opposite, 0, intArrayOf(0, 0), 1) }
    }
    checked {
        val angle = 0.2
        val pair = doubleArrayOf(sin(angle), 0.0, cos(angle), -sin(angle), 0.0, cos(angle))
        val cap = minimumSphericalCap(pair, 0, intArrayOf(0, 1))
        assertClose("two-point minimum cap", cap.radiusDeg, angle * RAD_TO_DEG, 1e-10)
    }
    checked {
        val angle = 0.3
        val radial = sin(angle)
        val z = cos(angle)
        val triple = doubleArrayOf(
            radial, 0.0, z,
            radial * cos(2.0 * PI / 3.0), radial * sin(2.0 * PI / 3.0), z,
            radial * cos(4.0 * PI / 3.0), radial * sin(4.0 * PI / 3.0), z
        )
        val cap = minimumSphericalCap(triple, 0, intArrayOf(0, 1, 2))
        assertClose("three-point minimum cap", cap.radiusDeg, angle * RAD_TO_DEG, 1e-10)
        require(cap.supportSize == 3)
    }
    checked {
        val first = linkedMapOf<String, Any?>("b" to 2, "a" to listOf(1.0, -0.0))
        val second = linkedMapOf<String, Any?>("a" to listOf(1.0, 0.0), "b" to 2)
        require(canonicalJson(first).contentEquals(canonicalJson(second)))
    }
    checked {
        val decoded = decodeRgb8(
            byteArrayOf(
                128.toByte(), 128.toByte(), 255.toByte(),
                255.toByte(), 128.toByte(), 128.toByte()
            ),
            2,
            1,
            1
        )
        require(decoded.originalIds.contentEquals(intArrayOf(0, 1)))
    }
    println("PASS core self-test: $checks checks")
}

fun unitVector(x: Double, y: Double, z: Double): DoubleArray {
    val length = sqrt(x * x + y * y + z * z)
    require(length > CENTER_EPSILON && length.isFinite())
    return doubleArrayOf(x / length, y / length, z / length)
}

fun unitVectors(vararg vectors: DoubleArray): DoubleArray {
    val result = DoubleArray(vectors.size * CHANNELS)
    for ((index, vector) in vectors.withIndex()) {
        require(vector.size == CHANNELS)
        vector.copyInto(result, index * CHANNELS)
    }
    return result
}

fun transportStatsJson(stats: TransportStats): Map<String, Any?> = linkedMapOf(
    "dotCostEvaluations" to stats.dotCostEvaluations,
    "priceCandidateEvaluations" to stats.priceCandidateEvaluations,
    "bids" to stats.bids,
    "evictions" to stats.evictions,
    "scalingPhases" to stats.scalingPhases,
    "certificateCostEvaluations" to stats.certificateCostEvaluations
)

fun runTransportSelfTests() {
    var checks = 0
    fun checked(action: () -> Unit) {
        action()
        checks++
    }

    val directions = unitVectors(
        unitVector(0.05, 0.02, 1.0),
        unitVector(0.15, -0.03, 1.0),
        unitVector(-0.10, 0.12, 1.0),
        unitVector(0.80, 0.10, 0.60),
        unitVector(-0.70, 0.20, 0.68),
        unitVector(0.05, -0.85, 0.52)
    )
    val centers2 = unitVectors(
        unitVector(0.35, -0.10, 1.0),
        unitVector(-0.30, 0.25, 1.0)
    )
    val centers3 = unitVectors(
        unitVector(0.0, 0.0, 1.0),
        unitVector(0.8, 0.1, 0.6),
        unitVector(-0.7, 0.2, 0.68)
    )
    val fixedReceipts = ArrayList<Map<String, Any?>>()

    fun verifyFixed(
        id: String,
        fixedCenters: DoubleArray,
        clusterCount: Int,
        capacity: Int,
        quantization: Long
    ): Map<String, Any?> {
        val sampleCount = directions.size / CHANNELS
        val first = balancedTransport(
            directions,
            0,
            sampleCount,
            fixedCenters,
            clusterCount,
            capacity,
            quantization
        )
        val second = balancedTransport(
            directions,
            0,
            sampleCount,
            fixedCenters,
            clusterCount,
            capacity,
            quantization
        )
        requireOptimalTransport(first, sampleCount, clusterCount, capacity)
        require(first.quantizedCost == second.quantizedCost && first.labels.contentEquals(second.labels)) {
            "$id is not deterministic"
        }
        val exhaustiveQuantized = exhaustiveBalancedQuantized(
            fixedCenterEdgeCosts(directions, 0, sampleCount, fixedCenters, clusterCount, quantization),
            sampleCount,
            clusterCount,
            capacity
        )
        require(first.quantizedCost == exhaustiveQuantized.cost) {
            "$id transport cost ${first.quantizedCost} != exhaustive ${exhaustiveQuantized.cost}"
        }
        require(first.quantizedCost == quantizedCostForLabels(directions, 0, first.labels, fixedCenters, quantization))
        val exhaustiveFloat64 = exhaustiveBalancedFixedFloat64(
            directions,
            0,
            sampleCount,
            fixedCenters,
            clusterCount,
            capacity
        )
        val float64Loss = fixedCenterFloat64Loss(directions, 0, first.labels, fixedCenters)
        val gap = float64Loss - exhaustiveFloat64.loss
        val bound = sampleCount / quantization.toDouble()
        require(gap <= bound + 1e-12) {
            "$id fixed-center float64 gap $gap exceeds N/Q bound $bound"
        }
        require(gap >= -1e-12)
        return linkedMapOf(
            "id" to id,
            "sampleCount" to sampleCount,
            "clusterCount" to clusterCount,
            "capacity" to capacity,
            "quantization" to quantization,
            "solverStatus" to first.status,
            "solverLabelsSha256" to labelsSha256(first.labels),
            "quantizedCost" to first.quantizedCost,
            "exhaustiveQuantizedCost" to exhaustiveQuantized.cost,
            "proofScale" to first.proofScale,
            "scaledDualLowerBound" to first.scaledDualLowerBound,
            "certificateGap" to first.certificateGap,
            "float64Loss" to float64Loss,
            "exhaustiveFloat64Loss" to exhaustiveFloat64.loss,
            "float64Gap" to gap,
            "nOverQBound" to bound,
            "transport" to transportStatsJson(first.stats)
        )
    }

    val qSweep = longArrayOf(10L, 1_000L, DEFAULT_QUANTIZATION)
    var previousBound = Double.POSITIVE_INFINITY
    for (quantization in qSweep) {
        checked {
            val receipt = verifyFixed("fixed-k2-q$quantization", centers2, 2, 3, quantization)
            val bound = receipt["nOverQBound"] as Double
            require(bound < previousBound)
            previousBound = bound
            fixedReceipts += receipt
        }
    }
    checked {
        fixedReceipts += verifyFixed("fixed-k3-q1000000", centers3, 3, 2, 1_000_000L)
    }
    val random = java.util.Random(0x5EEDC0DEL)
    var randomizedOracleCases = 0
    var randomizedMaximumCertificateGap = 0L
    var randomizedMaximumFloat64GapToBoundRatio = 0.0
    repeat(48) { caseIndex ->
        checked {
            val clusterCount = if (caseIndex % 2 == 0) 2 else 3
            val capacity = 6 / clusterCount
            val randomDirections = DoubleArray(6 * CHANNELS)
            val randomCenters = DoubleArray(clusterCount * CHANNELS)
            for (sample in 0 until 6) {
                unitVector(
                    random.nextDouble() * 2.0 - 1.0,
                    random.nextDouble() * 2.0 - 1.0,
                    0.05 + random.nextDouble()
                ).copyInto(randomDirections, sample * CHANNELS)
            }
            for (cluster in 0 until clusterCount) {
                unitVector(
                    random.nextDouble() * 2.0 - 1.0,
                    random.nextDouble() * 2.0 - 1.0,
                    0.05 + random.nextDouble()
                ).copyInto(randomCenters, cluster * CHANNELS)
            }
            val quantization = longArrayOf(1L, 17L, 10_000L, DEFAULT_QUANTIZATION)[caseIndex % 4]
            val result = balancedTransport(
                randomDirections,
                0,
                6,
                randomCenters,
                clusterCount,
                capacity,
                quantization
            )
            requireOptimalTransport(result, 6, clusterCount, capacity)
            val exhaustiveQuantized = exhaustiveBalancedQuantized(
                fixedCenterEdgeCosts(randomDirections, 0, 6, randomCenters, clusterCount, quantization),
                6,
                clusterCount,
                capacity
            )
            require(result.quantizedCost == exhaustiveQuantized.cost) {
                "Randomized case $caseIndex cost ${result.quantizedCost} != exhaustive ${exhaustiveQuantized.cost}"
            }
            val exhaustiveFloat64 = exhaustiveBalancedFixedFloat64(
                randomDirections,
                0,
                6,
                randomCenters,
                clusterCount,
                capacity
            )
            val float64Loss = fixedCenterFloat64Loss(randomDirections, 0, result.labels, randomCenters)
            val gap = max(0.0, float64Loss - exhaustiveFloat64.loss)
            val bound = 6.0 / quantization
            require(gap <= bound + 1e-12)
            randomizedMaximumCertificateGap = max(
                randomizedMaximumCertificateGap,
                result.certificateGap ?: error("Missing randomized certificate")
            )
            randomizedMaximumFloat64GapToBoundRatio = max(
                randomizedMaximumFloat64GapToBoundRatio,
                gap / bound
            )
            randomizedOracleCases++
        }
    }
    checked {
        val tiedDirections = unitVectors(
            unitVector(0.0, 0.0, 1.0),
            unitVector(0.0, 0.0, 1.0),
            unitVector(0.0, 0.0, 1.0),
            unitVector(0.0, 0.0, 1.0)
        )
        val tiedCenters = unitVectors(unitVector(0.0, 0.0, 1.0), unitVector(0.0, 0.0, 1.0))
        val first = balancedTransport(tiedDirections, 0, 4, tiedCenters, 2, 2, DEFAULT_QUANTIZATION)
        val second = balancedTransport(tiedDirections, 0, 4, tiedCenters, 2, 2, DEFAULT_QUANTIZATION)
        requireOptimalTransport(first, 4, 2, 2)
        require(first.quantizedCost == 0L)
        require(first.labels.contentEquals(second.labels)) { "Tied transport is not deterministic" }
    }
    checked {
        expectFailure("zero Q") { validateQuantization(6, 0L) }
        expectFailure("unsafe Q") { validateQuantization(6, Long.MAX_VALUE) }
    }
    for (k in intArrayOf(1, 2, 3, 6)) checked {
        val tiedDirections = DoubleArray(6 * CHANNELS) { if (it % CHANNELS == 2) 1.0 else 0.0 }
        val tiedCenters = tiedDirections.copyOf(k * CHANNELS)
        val result = balancedTransport(tiedDirections, 0, 6, tiedCenters, k, 6 / k, DEFAULT_QUANTIZATION)
        requireOptimalTransport(result, 6, k, 6 / k)
        require(result.labels.contentEquals(IntArray(6) { it / (6 / k) })) {
            "Equal-cost slots must retain ascending expanded object-ID order"
        }
    }
    checked {
        val result = balancedTransport(directions, 0, 1, centers2.copyOf(CHANNELS), 1, 1, DEFAULT_QUANTIZATION)
        requireOptimalTransport(result, 1, 1, 1)
        require(result.labels.contentEquals(intArrayOf(0)))
        require(result.quantizedCost == fixedCenterEdgeCosts(
            directions, 0, 1, centers2.copyOf(CHANNELS), 1, DEFAULT_QUANTIZATION
        )[0])
    }
    for (k in intArrayOf(1, 6)) checked {
        val fixedCenters = if (k == 1) centers2.copyOfRange(0, CHANNELS) else directions.copyOf()
        val result = balancedTransport(directions, 0, 6, fixedCenters, k, 6 / k, DEFAULT_QUANTIZATION)
        requireOptimalTransport(result, 6, k, 6 / k)
        val costs = fixedCenterEdgeCosts(directions, 0, 6, fixedCenters, k, DEFAULT_QUANTIZATION)
        val oracle = exhaustiveBalancedQuantized(costs, 6, k, 6 / k)
        require(result.quantizedCost == oracle.cost)
    }
    checked {
        val partial = balancedTransport(directions, 0, 6, centers2, 2, 3, DEFAULT_QUANTIZATION, 5)
        require(partial.status == "NONTERMINAL_LIMIT")
        expectFailure("nonterminal transport") { requireOptimalTransport(partial, 6, 2, 3) }
    }

    val joint = exhaustiveBalancedJointFloat64(directions, 0, 6, 2, 3)
    val lloyd = balancedLloyd(directions, 0, 6, 2, 3, DEFAULT_QUANTIZATION, BALANCED_MAX_ITERATIONS)
    checked {
        require(lloyd.objective + 1e-12 >= joint.loss)
        require(lloyd.finalCheckQuantizedCost <= lloyd.incumbentQuantizedCost)
        if (lloyd.converged) require(lloyd.finalCheckQuantizedCost == lloyd.incumbentQuantizedCost)
        if (lloyd.status == "CYCLE") require(!lloyd.converged)
        var acceptedObjective = Double.POSITIVE_INFINITY
        for (iteration in lloyd.iterations) {
            if (!iteration.accepted) continue
            require(iteration.updatedCenterFloat64Loss <= acceptedObjective)
            acceptedObjective = iteration.updatedCenterFloat64Loss
        }
    }
    checked {
        val limited = balancedLloyd(directions, 0, 6, 2, 3, DEFAULT_QUANTIZATION, 1)
        require(limited.status == "MAX_ITERATIONS" && !limited.converged)
    }
    checked {
        val opposite = unitVectors(unitVector(1.0, 0.0, 0.0), unitVector(-1.0, 0.0, 0.0))
        expectFailure("degenerate Lloyd center") {
            balancedLloyd(opposite, 0, 2, 1, 2, DEFAULT_QUANTIZATION, 2)
        }
    }

    val receipt = linkedMapOf<String, Any?>(
        "schema" to "alpha-piscium/stbn-clustering-transport-self-test/v1",
        "checks" to checks,
        "fixedCenterOracles" to fixedReceipts,
        "randomizedFixedCenterOracles" to linkedMapOf(
            "seed" to "0x5EEDC0DE",
            "cases" to randomizedOracleCases,
            "sampleCount" to 6,
            "clusterCounts" to listOf(2, 3),
            "quantizations" to listOf(1L, 17L, 10_000L, DEFAULT_QUANTIZATION),
            "allQuantizedCostsMatchedExhaustive" to true,
            "maximumCertificateGap" to randomizedMaximumCertificateGap,
            "maximumFloat64GapToNOverQBoundRatio" to randomizedMaximumFloat64GapToBoundRatio
        ),
        "jointOracle" to linkedMapOf(
            "sampleCount" to 6,
            "clusterCount" to 2,
            "capacity" to 3,
            "scope" to "complete labeled balanced partitions; diagnostic only, no full-size global-optimum claim",
            "exhaustiveFloat64Loss" to joint.loss,
            "lloydFloat64Loss" to lloyd.objective,
            "lloydGap" to lloyd.objective - joint.loss,
            "lloydStatus" to lloyd.status,
            "lloydConverged" to lloyd.converged,
            "lloydIterations" to lloyd.iterations.size,
            "finalCheckLabelsMatch" to lloyd.finalCheckLabelsMatch,
            "finalCheckFloat64Gap" to lloyd.finalCheckFloat64Gap
        )
    )
    val bytes = canonicalJson(receipt)
    println("transport-self-test.json|${bytes.size}|${sha256(bytes)}")
    print(bytes.toString(StandardCharsets.UTF_8))
    println("PASS transport self-test: $checks checks")
}

fun deleteExactTree(root: Path) {
    if (!Files.exists(root)) return
    Files.walk(root).use { paths ->
        paths.sorted(java.util.Comparator.reverseOrder()).forEach { Files.delete(it) }
    }
}

fun runInitializerSelfTests() {
    var checks = 0
    fun checked(action: () -> Unit) {
        action()
        checks++
    }

    val sampleCount = 16
    val clusterCount = 4
    val capacity = sampleCount / clusterCount
    val directions = DoubleArray(sampleCount * CHANNELS)
    for (sample in 0 until sampleCount) {
        val angle = 2.0 * PI * (sample + 0.173 * (sample % 3)) / sampleCount
        val radius = 0.15 + 0.75 * ((sample * 7) % sampleCount) / (sampleCount - 1.0)
        unitVector(cos(angle) * radius, sin(angle) * radius, 1.0 - 0.35 * radius)
            .copyInto(directions, sample * CHANNELS)
    }
    val originalDirections = directions.copyOf()
    val specs = multistartSpecs(9)
    val familyCounts = specs.groupingBy { it.family }.eachCount().toSortedMap()
    checked {
        require(familyCounts.values.all { it == 3 })
        require(specs.map { it.seed }.distinct().size == specs.size)
    }

    val initializerReceipts = ArrayList<Map<String, Any?>>()
    for (spec in specs) {
        checked {
            val first = initializationForSpec(directions, 0, sampleCount, clusterCount, spec)
            val second = initializationForSpec(directions, 0, sampleCount, clusterCount, spec)
            require(first.family == spec.family && first.seed == spec.seed)
            require(first.centers.contentEquals(second.centers)) { "${spec.family} is not deterministic" }
            require(first.balancedLabels?.contentEquals(second.balancedLabels) ?: (second.balancedLabels == null))
            for (center in 0 until clusterCount) {
                val offset = center * CHANNELS
                val length = sqrt(
                    first.centers[offset] * first.centers[offset] +
                        first.centers[offset + 1] * first.centers[offset + 1] +
                        first.centers[offset + 2] * first.centers[offset + 2]
                )
                require(abs(length - 1.0) <= 1e-12)
            }
            first.balancedLabels?.let { labels ->
                val counts = IntArray(clusterCount)
                for (label in labels) counts[label]++
                require(counts.all { it == capacity })
            }
            val transport = balancedTransport(
                directions,
                0,
                sampleCount,
                first.centers,
                clusterCount,
                capacity,
                DEFAULT_QUANTIZATION
            )
            requireOptimalTransport(transport, sampleCount, clusterCount, capacity)
            require(transport.stats.dotCostEvaluations == sampleCount.toLong() * clusterCount) {
                "${spec.family} did not use the complete assignment graph"
            }
            require(directions.contentEquals(originalDirections)) { "${spec.family} modified its input" }
            initializerReceipts += linkedMapOf(
                "ordinal" to spec.ordinal,
                "family" to spec.family,
                "seed" to unsignedHex(spec.seed),
                "balancedPartition" to (first.balancedLabels != null),
                "partitionSha256" to first.balancedLabels?.let { partitionSha256(it, clusterCount) },
                "completeEdgeEvaluations" to transport.stats.dotCostEvaluations
            )
        }
    }

    checked {
        val spec = specs.first { it.family == "squared-chord-d2" }
        val initialization = squaredChordD2Initialization(directions, 0, sampleCount, clusterCount, spec.seed)
        val random = DeterministicRandom(spec.seed)
        val firstSample = random.nextInt(sampleCount)
        val firstOffset = firstSample * CHANNELS
        require(initialization.centers.copyOfRange(0, CHANNELS).contentEquals(
            directions.copyOfRange(firstOffset, firstOffset + CHANNELS)
        ))
        val weights = DoubleArray(sampleCount)
        var total = 0.0
        for (sample in 0 until sampleCount) {
            if (sample == firstSample) continue
            val offset = sample * CHANNELS
            val dot = directions[offset] * directions[firstOffset] +
                directions[offset + 1] * directions[firstOffset + 1] +
                directions[offset + 2] * directions[firstOffset + 2]
            weights[sample] = 2.0 * (1.0 - dot.coerceIn(-1.0, 1.0))
            total += weights[sample]
        }
        val threshold = random.nextDouble() * total
        var cumulative = 0.0
        var secondSample = -1
        for (sample in 0 until sampleCount) {
            cumulative += weights[sample]
            if (secondSample < 0 && cumulative > threshold) secondSample = sample
        }
        require(secondSample >= 0)
        val secondOffset = secondSample * CHANNELS
        require(initialization.centers.copyOfRange(CHANNELS, 2 * CHANNELS).contentEquals(
            directions.copyOfRange(secondOffset, secondOffset + CHANNELS)
        )) { "D2 initializer did not sample proportional to unsquared squared-chord loss" }
    }

    val parent = recursiveTangentPcaInitialization(directions, 0, sampleCount, 2, 0L)
    val parentLabels = parent.balancedLabels ?: error("Parent initializer lacks labels")
    val parentObjective = computeClusterGeometry(directions, 0, parentLabels, 2).explicitSquaredChord
    val split = crossKSplitInitialization(directions, 0, parentLabels, 2, 0L)
    val splitLabels = split.balancedLabels ?: error("Cross-K initializer lacks labels")
    val splitObjective = computeClusterGeometry(directions, 0, splitLabels, 4).explicitSquaredChord
    checked {
        require(splitObjective <= parentObjective + OBJECTIVE_RELATIVE_TOLERANCE * max(1.0, parentObjective))
        val relabeled = IntArray(sampleCount) { clusterCount - 1 - splitLabels[it] }
        require(partitionSha256(splitLabels, clusterCount) == partitionSha256(relabeled, clusterCount))
    }
    checked {
        val crossSpec = StartSpec(9, "cross-k128-pca-split", 0, 0L)
        val outcome = evaluateMultistart(
            directions,
            0,
            sampleCount,
            clusterCount,
            capacity,
            DEFAULT_QUANTIZATION,
            8,
            crossSpec,
            split
        )
        require(outcome.initializerPartitionObjective == splitObjective)
        require(outcome.transport.dotCostEvaluations >= sampleCount.toLong() * clusterCount)
    }

    var canonicalResumeHash = ""
    var resumedCheckpointCount = 0
    val continuousRoot = Files.createTempDirectory("stbn-multistart-continuous-")
    val resumedRoot = Files.createTempDirectory("stbn-multistart-resumed-")
    try {
        fun execute(root: Path, maximumNewStarts: Int): ByteArray? {
            val checkpointRoot = root.resolve("checkpoint")
            val manifest = canonicalJson(linkedMapOf(
                "schema" to "alpha-piscium/stbn-clustering-mini-checkpoint/v1",
                "directionsSha256" to sha256(
                    java.nio.ByteBuffer.allocate(directions.size * java.lang.Double.BYTES).apply {
                        for (value in directions) putDouble(value)
                    }.array()
                ),
                "clusterCount" to clusterCount,
                "capacity" to capacity,
                "quantization" to DEFAULT_QUANTIZATION,
                "maxIterations" to 8,
                "starts" to specs.map {
                    linkedMapOf("ordinal" to it.ordinal, "family" to it.family, "seed" to unsignedHex(it.seed))
                }
            ))
            requireOrWriteExact(checkpointRoot.resolve("manifest.json"), manifest)
            val outcomes = ArrayList<MultistartOutcome>()
            var newStarts = 0
            for (spec in specs) {
                val path = multistartCheckpointPath(checkpointRoot, clusterCount, 0, spec.ordinal)
                if (!Files.exists(path) && newStarts < maximumNewStarts) {
                    val outcome = evaluateMultistart(
                        directions,
                        0,
                        sampleCount,
                        clusterCount,
                        capacity,
                        DEFAULT_QUANTIZATION,
                        8,
                        spec
                    )
                    writeMultistartCheckpoint(path, outcome)
                    newStarts++
                }
                if (Files.exists(path)) {
                    outcomes += readMultistartCheckpoint(
                        path,
                        clusterCount,
                        0,
                        spec,
                        directions,
                        0,
                        sampleCount,
                        capacity
                    )
                }
            }
            if (outcomes.size != specs.size) return null
            val elites = selectElites(outcomes, 8)
            return canonicalJson(linkedMapOf(
                "best" to multistartOutcomeJson(elites.first()),
                "elites" to elites.map(::multistartOutcomeJson),
                "diversity" to eliteDiversity(elites),
                "bestSoFar" to bestSoFar(outcomes),
                "starts" to outcomes.sortedBy { it.ordinal }.map(::multistartOutcomeJson)
            ))
        }

        val continuous = execute(continuousRoot, Int.MAX_VALUE) ?: error("Continuous mini search did not finish")
        checked {
            require(execute(resumedRoot, 4) == null) { "Interrupted mini search unexpectedly finished" }
            resumedCheckpointCount = 4
        }
        val resumed = execute(resumedRoot, Int.MAX_VALUE) ?: error("Resumed mini search did not finish")
        checked {
            require(continuous.contentEquals(resumed)) {
                "Interrupted/resumed mini search differs from continuous search"
            }
            canonicalResumeHash = sha256(continuous)
            val continuousHashes = specs.map {
                sha256(Files.readAllBytes(multistartCheckpointPath(continuousRoot.resolve("checkpoint"), 4, 0, it.ordinal)))
            }
            val resumedHashes = specs.map {
                sha256(Files.readAllBytes(multistartCheckpointPath(resumedRoot.resolve("checkpoint"), 4, 0, it.ordinal)))
            }
            require(continuousHashes == resumedHashes) { "Checkpoint bytes differ after resume" }
        }
        checked {
            expectFailure("checkpoint manifest mismatch") {
                requireOrWriteExact(resumedRoot.resolve("checkpoint/manifest.json"), "different\n".toByteArray())
            }
        }
    } finally {
        deleteExactTree(continuousRoot)
        deleteExactTree(resumedRoot)
    }

    checked {
        val laterStarted = java.util.concurrent.CountDownLatch(1)
        val work = specs.take(5).map { StartWork(0, it) }
        val completed = arrayListOf<MultistartOutcome>()
        runStartWork(work, 2, { item ->
            if (item.spec.ordinal == 0) {
                require(laterStarted.await(10, java.util.concurrent.TimeUnit.SECONDS)) {
                    "Scheduler stalled behind a slow task instead of replenishing work"
                }
            }
            if (item.spec.ordinal == 4) laterStarted.countDown()
            evaluateMultistart(directions, 0, sampleCount, clusterCount, capacity,
                DEFAULT_QUANTIZATION, 8, item.spec)
        }, { _, outcome -> completed += outcome })
        require(completed.map { it.ordinal }.sorted() == (0..4).toList())
        for (outcome in completed) {
            val serial = evaluateMultistart(directions, 0, sampleCount, clusterCount, capacity,
                DEFAULT_QUANTIZATION, 8, specs[outcome.ordinal])
            require(canonicalJson(multistartOutcomeJson(outcome)).contentEquals(
                canonicalJson(multistartOutcomeJson(serial))))
        }
    }

    val receipt = linkedMapOf<String, Any?>(
        "schema" to "alpha-piscium/stbn-clustering-initializer-self-test/v1",
        "checks" to checks,
        "sampleCount" to sampleCount,
        "clusterCount" to clusterCount,
        "capacity" to capacity,
        "initializerAllocation" to familyCounts,
        "initializers" to initializerReceipts,
        "d2Weight" to "2 * (1 - clamped dot), sampled directly without another square",
        "assignmentGraph" to "complete sample-center graph for every tested initializer",
        "crossK" to linkedMapOf(
            "parentObjective" to parentObjective,
            "splitObjective" to splitObjective,
            "nonIncrease" to true,
            "postInitializerAssignment" to "unrestricted complete graph"
        ),
        "checkpointResume" to linkedMapOf(
            "deliberatelyInterruptedAfterStarts" to resumedCheckpointCount,
            "canonicalOutputSha256" to canonicalResumeHash,
            "canonicalAndCheckpointBytesMatchContinuous" to true,
            "manifestMismatchRejected" to true
        )
    )
    val bytes = canonicalJson(receipt)
    println("initializer-self-test.json|${bytes.size}|${sha256(bytes)}")
    print(bytes.toString(StandardCharsets.UTF_8))
    println("PASS initializer self-test: $checks checks")
}

fun lloydIterationJson(iteration: LloydIteration): Map<String, Any?> = linkedMapOf(
    "index" to iteration.index,
    "labelsSha256" to iteration.labelsSha256,
    "transportStatus" to iteration.transportStatus,
    "quantizedAssignmentCost" to iteration.quantizedAssignmentCost,
    "transportProofScale" to iteration.transportProofScale,
    "transportCertificateGap" to iteration.transportCertificateGap,
    "fixedCenterFloat64Loss" to iteration.fixedCenterFloat64Loss,
    "previousFixedCenterFloat64Loss" to iteration.previousFixedCenterFloat64Loss,
    "fixedCenterQuantizationBound" to iteration.fixedCenterQuantizationBound,
    "updatedCenterFloat64Loss" to iteration.updatedCenterFloat64Loss,
    "deltaFromAccepted" to iteration.deltaFromAccepted,
    "accepted" to iteration.accepted,
    "transport" to transportStatsJson(iteration.transport)
)

fun evaluateMultistart(
    directions: DoubleArray,
    vectorBase: Int,
    sampleCount: Int,
    clusterCount: Int,
    capacity: Int,
    quantization: Long,
    maxIterations: Int,
    spec: StartSpec,
    overrideInitialization: SearchInitialization? = null
): MultistartOutcome {
    val initialization = overrideInitialization ?: initializationForSpec(
        directions,
        vectorBase,
        sampleCount,
        clusterCount,
        spec
    )
    require(initialization.family == spec.family && initialization.seed == spec.seed)
    val initializationPartitionSha256 = initialization.balancedLabels?.let {
        normalizedCentersForLabels(directions, vectorBase, it, clusterCount, capacity)
        partitionSha256(it, clusterCount)
    }
    val initializerPartitionObjective = initialization.balancedLabels?.let {
        computeClusterGeometry(directions, vectorBase, it, clusterCount).explicitSquaredChord
    }
    val result = balancedLloyd(
        directions,
        vectorBase,
        sampleCount,
        clusterCount,
        capacity,
        quantization,
        maxIterations,
        CenterInitialization(initialization.centers.copyOf(), initialization.dotEvaluations)
    )
    return MultistartOutcome(
        clusterCount,
        vectorBase / (SLICE_SAMPLES * CHANNELS),
        spec.ordinal,
        spec.family,
        spec.familyIndex,
        spec.seed,
        result.labels,
        partitionSha256(result.labels, clusterCount),
        result.objective,
        result.status,
        result.converged,
        result.iterations.size,
        result.iterations.count { it.accepted },
        initializerPartitionObjective,
        result.iterations.first().updatedCenterFloat64Loss,
        initializationPartitionSha256,
        result.firstAssignmentPartitionSha256,
        result.finalCheckQuantizedCost,
        result.finalCheckCertificateGap,
        result.incumbentQuantizedCost,
        result.finalCheckLabelsMatch,
        result.finalCheckFloat64Gap,
        result.quantizationBound,
        result.initializationDotEvaluations,
        result.transport
    )
}

fun compareOutcomes(first: MultistartOutcome, second: MultistartOutcome): Int {
    val objective = first.objective.compareTo(second.objective)
    if (objective != 0) return objective
    val partition = first.partitionSha256.compareTo(second.partitionSha256)
    if (partition != 0) return partition
    return first.ordinal.compareTo(second.ordinal)
}

fun selectElites(outcomes: List<MultistartOutcome>, eliteCount: Int): List<MultistartOutcome> {
    require(eliteCount in 8..16)
    val ordered = outcomes.sortedWith(::compareOutcomes)
    val canonicalByHash = linkedMapOf<String, IntArray>()
    val result = ArrayList<MultistartOutcome>(eliteCount)
    for (outcome in ordered) {
        val canonical = canonicalPartitionLabels(outcome.labels, outcome.clusterCount)
        val prior = canonicalByHash[outcome.partitionSha256]
        if (prior != null) {
            require(prior.contentEquals(canonical)) { "SHA-256 collision in elite partition identities" }
            continue
        }
        canonicalByHash[outcome.partitionSha256] = canonical
        result += outcome
        if (result.size == eliteCount) break
    }
    require(result.isNotEmpty())
    return result
}

fun normalizedVariationOfInformation(first: IntArray, second: IntArray, clusterCount: Int): Double {
    require(first.size == second.size && first.isNotEmpty())
    val firstCounts = IntArray(clusterCount)
    val secondCounts = IntArray(clusterCount)
    val contingency = IntArray(clusterCount * clusterCount)
    for (sample in first.indices) {
        val a = first[sample]
        val b = second[sample]
        require(a in 0 until clusterCount && b in 0 until clusterCount)
        firstCounts[a]++
        secondCounts[b]++
        contingency[a * clusterCount + b]++
    }
    val total = first.size.toDouble()
    var firstEntropy = 0.0
    var secondEntropy = 0.0
    for (count in firstCounts) if (count > 0) {
        val probability = count / total
        firstEntropy -= probability * ln(probability)
    }
    for (count in secondCounts) if (count > 0) {
        val probability = count / total
        secondEntropy -= probability * ln(probability)
    }
    var mutualInformation = 0.0
    for (firstCluster in 0 until clusterCount) {
        for (secondCluster in 0 until clusterCount) {
            val count = contingency[firstCluster * clusterCount + secondCluster]
            if (count == 0) continue
            val joint = count / total
            mutualInformation += joint * ln(
                count.toDouble() * first.size /
                    (firstCounts[firstCluster].toDouble() * secondCounts[secondCluster])
            )
        }
    }
    val normalization = max(firstEntropy + secondEntropy, 1e-300)
    return ((firstEntropy + secondEntropy - 2.0 * mutualInformation) / normalization).coerceIn(0.0, 1.0)
}

fun eliteDiversity(elites: List<MultistartOutcome>): Map<String, Any?> {
    val values = ArrayList<Double>()
    for (first in elites.indices) for (second in first + 1 until elites.size) {
        values += normalizedVariationOfInformation(
            elites[first].labels,
            elites[second].labels,
            elites[first].clusterCount
        )
    }
    return linkedMapOf(
        "metric" to "normalized variation of information; label-permutation invariant; 0 identical, 1 maximally distinct",
        "uniquePartitions" to elites.size,
        "pairCount" to values.size,
        "distribution" to if (values.isEmpty()) null else distribution(values.toDoubleArray())
    )
}

fun bestSoFar(outcomes: List<MultistartOutcome>): List<Map<String, Any?>> {
    var best: MultistartOutcome? = null
    val history = ArrayList<Map<String, Any?>>()
    for (outcome in outcomes.sortedBy { it.ordinal }) {
        if (best == null || compareOutcomes(outcome, best!!) < 0) {
            best = outcome
            history += linkedMapOf(
                "ordinal" to outcome.ordinal,
                "family" to outcome.family,
                "familyIndex" to outcome.familyIndex,
                "seed" to unsignedHex(outcome.seed),
                "objective" to outcome.objective,
                "partitionSha256" to outcome.partitionSha256
            )
        }
    }
    return history
}

fun multistartCheckpointPath(root: Path, clusterCount: Int, slice: Int, ordinal: Int): Path =
    root.resolve("k$clusterCount-s${slice.toString().padStart(2, '0')}-r${ordinal.toString().padStart(3, '0')}.bin")

fun writeMultistartCheckpoint(path: Path, outcome: MultistartOutcome) {
    Files.createDirectories(path.parent)
    val temporary = path.resolveSibling(path.fileName.toString() + ".tmp")
    DataOutputStream(BufferedOutputStream(Files.newOutputStream(temporary))).use { output ->
        output.writeInt(MULTISTART_CHECKPOINT_MAGIC)
        output.writeInt(MULTISTART_CHECKPOINT_VERSION)
        output.writeInt(outcome.clusterCount)
        output.writeInt(outcome.slice)
        output.writeInt(outcome.ordinal)
        output.writeUTF(outcome.family)
        output.writeInt(outcome.familyIndex)
        output.writeLong(outcome.seed)
        output.writeUTF(outcome.partitionSha256)
        output.writeDouble(outcome.objective)
        output.writeUTF(outcome.status)
        output.writeBoolean(outcome.converged)
        output.writeInt(outcome.iterationCount)
        output.writeInt(outcome.acceptedIterations)
        output.writeBoolean(outcome.initializerPartitionObjective != null)
        if (outcome.initializerPartitionObjective != null) output.writeDouble(outcome.initializerPartitionObjective)
        output.writeDouble(outcome.firstAssignmentObjective)
        output.writeBoolean(outcome.initializationPartitionSha256 != null)
        if (outcome.initializationPartitionSha256 != null) output.writeUTF(outcome.initializationPartitionSha256)
        output.writeUTF(outcome.firstAssignmentPartitionSha256)
        output.writeLong(outcome.finalCheckQuantizedCost)
        output.writeLong(outcome.finalCheckCertificateGap)
        output.writeLong(outcome.incumbentQuantizedCost)
        output.writeBoolean(outcome.finalCheckLabelsMatch)
        output.writeDouble(outcome.finalCheckFloat64Gap)
        output.writeDouble(outcome.quantizationBound)
        output.writeLong(outcome.initializationDotEvaluations)
        output.writeLong(outcome.transport.dotCostEvaluations)
        output.writeLong(outcome.transport.priceCandidateEvaluations)
        output.writeLong(outcome.transport.bids)
        output.writeLong(outcome.transport.evictions)
        output.writeLong(outcome.transport.scalingPhases)
        output.writeLong(outcome.transport.certificateCostEvaluations)
        output.writeInt(outcome.labels.size)
        output.write(labelsBytes(outcome.labels))
    }
    Files.move(temporary, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
}

fun readMultistartCheckpoint(
    path: Path,
    expectedClusterCount: Int,
    expectedSlice: Int,
    expectedSpec: StartSpec,
    directions: DoubleArray,
    vectorBase: Int,
    sampleCount: Int,
    capacity: Int
): MultistartOutcome {
    val outcome = DataInputStream(BufferedInputStream(Files.newInputStream(path))).use { input ->
        require(input.readInt() == MULTISTART_CHECKPOINT_MAGIC) { "Invalid checkpoint magic in ${normalizePath(path)}" }
        require(input.readInt() == MULTISTART_CHECKPOINT_VERSION) {
            "Unsupported checkpoint version in ${normalizePath(path)}"
        }
        val clusterCount = input.readInt()
        val slice = input.readInt()
        val ordinal = input.readInt()
        val family = input.readUTF()
        val familyIndex = input.readInt()
        val seed = input.readLong()
        val partitionHash = input.readUTF()
        val objective = input.readDouble()
        val status = input.readUTF()
        val converged = input.readBoolean()
        val iterationCount = input.readInt()
        val acceptedIterations = input.readInt()
        val initializerPartitionObjective = if (input.readBoolean()) input.readDouble() else null
        val firstAssignmentObjective = input.readDouble()
        val initializationPartitionHash = if (input.readBoolean()) input.readUTF() else null
        val firstAssignmentPartitionHash = input.readUTF()
        val finalCheckQuantizedCost = input.readLong()
        val finalCheckCertificateGap = input.readLong()
        val incumbentQuantizedCost = input.readLong()
        val finalCheckLabelsMatch = input.readBoolean()
        val finalCheckFloat64Gap = input.readDouble()
        val quantizationBound = input.readDouble()
        val initializationDotEvaluations = input.readLong()
        val transport = TransportStats(
            input.readLong(),
            input.readLong(),
            input.readLong(),
            input.readLong(),
            input.readLong(),
            input.readLong()
        )
        val labelCount = input.readInt()
        require(labelCount == sampleCount) { "Checkpoint label count $labelCount != $sampleCount" }
        val labelBytes = ByteArray(labelCount)
        input.readFully(labelBytes)
        require(input.read() == -1) { "Trailing data in ${normalizePath(path)}" }
        val labels = IntArray(labelCount) { labelBytes[it].toInt() and 0xFF }
        MultistartOutcome(
            clusterCount,
            slice,
            ordinal,
            family,
            familyIndex,
            seed,
            labels,
            partitionHash,
            objective,
            status,
            converged,
            iterationCount,
            acceptedIterations,
            initializerPartitionObjective,
            firstAssignmentObjective,
            initializationPartitionHash,
            firstAssignmentPartitionHash,
            finalCheckQuantizedCost,
            finalCheckCertificateGap,
            incumbentQuantizedCost,
            finalCheckLabelsMatch,
            finalCheckFloat64Gap,
            quantizationBound,
            initializationDotEvaluations,
            transport
        )
    }
    require(outcome.clusterCount == expectedClusterCount && outcome.slice == expectedSlice)
    require(outcome.ordinal == expectedSpec.ordinal && outcome.family == expectedSpec.family)
    require(outcome.familyIndex == expectedSpec.familyIndex && outcome.seed == expectedSpec.seed)
    val geometry = computeClusterGeometry(directions, vectorBase, outcome.labels, expectedClusterCount)
    require(geometry.counts.all { it == capacity }) { "Checkpoint capacities are invalid in ${normalizePath(path)}" }
    require(partitionSha256(outcome.labels, expectedClusterCount) == outcome.partitionSha256) {
        "Checkpoint partition hash mismatch in ${normalizePath(path)}"
    }
    require(abs(geometry.explicitSquaredChord - outcome.objective) <=
        OBJECTIVE_RELATIVE_TOLERANCE * max(1.0, outcome.objective)) {
        "Checkpoint objective mismatch in ${normalizePath(path)}"
    }
    require(outcome.iterationCount > 0 && outcome.acceptedIterations in 1..outcome.iterationCount)
    require(outcome.status in setOf(
        "CONVERGED_LABELS_AND_OBJECTIVE",
        "CYCLE",
        "ORIGINAL_OBJECTIVE_INCREASE",
        "MAX_ITERATIONS"
    ))
    return outcome
}

fun multistartOutcomeJson(outcome: MultistartOutcome): Map<String, Any?> = linkedMapOf(
    "ordinal" to outcome.ordinal,
    "family" to outcome.family,
    "familyIndex" to outcome.familyIndex,
    "seed" to unsignedHex(outcome.seed),
    "objective" to outcome.objective,
    "partitionSha256" to outcome.partitionSha256,
    "status" to outcome.status,
    "converged" to outcome.converged,
    "iterationCount" to outcome.iterationCount,
    "acceptedIterations" to outcome.acceptedIterations,
    "initializerPartitionObjective" to outcome.initializerPartitionObjective,
    "firstAssignmentObjective" to outcome.firstAssignmentObjective,
    "initializerPartitionSha256" to outcome.initializationPartitionSha256,
    "firstAssignmentPartitionSha256" to outcome.firstAssignmentPartitionSha256,
    "firstAssignmentChangedInitializerPartition" to outcome.initializationPartitionSha256?.let {
        it != outcome.firstAssignmentPartitionSha256
    },
    "finalCheckQuantizedCost" to outcome.finalCheckQuantizedCost,
    "finalCheckCertificateGap" to outcome.finalCheckCertificateGap,
    "incumbentQuantizedCost" to outcome.incumbentQuantizedCost,
    "finalCheckLabelsMatch" to outcome.finalCheckLabelsMatch,
    "finalCheckFloat64Gap" to outcome.finalCheckFloat64Gap,
    "quantizationBound" to outcome.quantizationBound,
    "initializationDotEvaluations" to outcome.initializationDotEvaluations,
    "transport" to transportStatsJson(outcome.transport)
)

fun runStartWork(
    work: List<StartWork>,
    workers: Int,
    evaluate: (StartWork) -> MultistartOutcome,
    accept: (StartWork, MultistartOutcome) -> Unit
) {
    val executor = Executors.newFixedThreadPool(workers)
    try {
        val completions = ExecutorCompletionService<Pair<StartWork, MultistartOutcome>>(executor)
        var submitted = 0
        fun submitNext() {
            val item = work[submitted++]
            completions.submit(Callable { item to evaluate(item) })
        }
        repeat(min(workers, work.size)) { submitNext() }
        repeat(work.size) {
            val (item, outcome) = completions.take().get()
            if (submitted < work.size) submitNext()
            accept(item, outcome)
        }
    } finally {
        executor.shutdownNow()
    }
}

fun executeMultistartPhase(
    directions: DoubleArray,
    checkpointRoot: Path,
    clusterCount: Int,
    specsBySlice: List<List<StartSpec>>,
    quantization: Long,
    maxIterations: Int,
    workers: Int,
    maximumNewStarts: Int,
    initializationOverride: (Int, StartSpec) -> SearchInitialization? = { _, _ -> null }
): StartPhaseResult {
    require(specsBySlice.size == DEPTH)
    require(workers > 0 && maximumNewStarts >= 0)
    val capacity = SLICE_SAMPLES / clusterCount
    val outcomesBySlice = List(DEPTH) { linkedMapOf<Int, MultistartOutcome>() }
    val pending = ArrayList<StartWork>()
    var loadedCheckpoints = 0
    for (slice in 0 until DEPTH) {
        require(specsBySlice[slice].map { it.ordinal }.distinct().size == specsBySlice[slice].size)
        val vectorBase = slice * SLICE_SAMPLES * CHANNELS
        for (spec in specsBySlice[slice].sortedBy { it.ordinal }) {
            val path = multistartCheckpointPath(checkpointRoot, clusterCount, slice, spec.ordinal)
            if (Files.exists(path)) {
                outcomesBySlice[slice][spec.ordinal] = readMultistartCheckpoint(
                    path,
                    clusterCount,
                    slice,
                    spec,
                    directions,
                    vectorBase,
                    SLICE_SAMPLES,
                    capacity
                )
                loadedCheckpoints++
            } else {
                pending += StartWork(slice, spec)
            }
        }
    }
    val selected = pending.take(maximumNewStarts)
    var computedStarts = 0
    runStartWork(selected, workers, { work ->
        val vectorBase = work.slice * SLICE_SAMPLES * CHANNELS
        val outcome = evaluateMultistart(
            directions, vectorBase, SLICE_SAMPLES, clusterCount, capacity, quantization,
            maxIterations, work.spec, initializationOverride(work.slice, work.spec)
        )
        writeMultistartCheckpoint(
            multistartCheckpointPath(checkpointRoot, clusterCount, work.slice, work.spec.ordinal), outcome
        )
        outcome
    }, { work, outcome ->
        outcomesBySlice[work.slice][work.spec.ordinal] = outcome
        computedStarts++
        println("checkpointed multistart K=$clusterCount new=$computedStarts/${selected.size} " +
            "remaining=${pending.size - computedStarts} slice=${work.slice} start=${work.spec.ordinal}")
    })
    val complete = (0 until DEPTH).all { slice ->
        outcomesBySlice[slice].size == specsBySlice[slice].size
    }
    return StartPhaseResult(
        outcomesBySlice.map { slice -> slice.values.sortedBy { it.ordinal } },
        complete,
        loadedCheckpoints,
        computedStarts
    )
}

fun readBalancedLabelSlices(path: Path, clusterCount: Int, sliceCount: Int = DEPTH): List<IntArray> {
    val sampleTotal = sliceCount * SLICE_SAMPLES
    val bytes = Files.readAllBytes(path)
    require(bytes.size == sampleTotal) {
        "Expected $sampleTotal one-byte labels in ${normalizePath(path)}, found ${bytes.size}"
    }
    val capacity = SLICE_SAMPLES / clusterCount
    return List(sliceCount) { slice ->
        val labels = IntArray(SLICE_SAMPLES)
        val counts = IntArray(clusterCount)
        for (sample in 0 until SLICE_SAMPLES) {
            val label = bytes[slice * SLICE_SAMPLES + sample].toInt() and 0xFF
            require(label < clusterCount) { "Slice $slice sample $sample has invalid label $label for K=$clusterCount" }
            labels[sample] = label
            counts[label]++
        }
        require(counts.all { it == capacity }) {
            "Slice $slice label capacities are ${counts.minOrNull()}..${counts.maxOrNull()}, expected $capacity"
        }
        labels
    }
}

fun searchBalancedLloyd(
    inputPath: Path,
    outputDirectory: Path,
    clusterCounts: IntArray,
    quantization: Long,
    maxIterations: Int
) {
    require(clusterCounts.isNotEmpty())
    require(clusterCounts.distinct().size == clusterCounts.size) { "Duplicate cluster count" }
    require(clusterCounts.all { it == 128 || it == 256 }) { "T03 search supports K=128 and K=256" }
    require(maxIterations > 0)
    validateQuantization(SLICE_SAMPLES, quantization)
    val totalStart = System.nanoTime()
    val sampler = PeakHeapSampler()
    try {
        val inputBytes = Files.readAllBytes(inputPath)
        val texture = decodeRgb8(inputBytes, WIDTH, HEIGHT, DEPTH)
        val inputSha256 = sha256(inputBytes)
        val availableProcessors = max(1, Runtime.getRuntime().availableProcessors())
        val largeHeap = Runtime.getRuntime().maxMemory() >= 2L * 1024L * 1024L * 1024L
        val workersByCase = linkedMapOf<String, Int>()
        val metricCases = ArrayList<Map<String, Any?>>()
        val searchCases = ArrayList<Map<String, Any?>>()
        val canonicalOutputs = linkedMapOf<String, Any?>()
        val caseTimings = ArrayList<TimedCase>()

        for (clusterCount in clusterCounts) {
            val caseStart = System.nanoTime()
            val capacity = SLICE_SAMPLES / clusterCount
            val caseId = "balanced-spherical-lloyd-k$clusterCount"
            val workerLimit = if (largeHeap) {
                if (clusterCount == 256) 12 else 16
            } else {
                if (clusterCount == 256) 3 else 8
            }
            val workers = min(workerLimit, availableProcessors)
            workersByCase[caseId] = workers
            val results = parallelMap(workers, (0 until DEPTH).toList()) { slice ->
                val result = balancedLloyd(
                    texture.directions,
                    slice * SLICE_SAMPLES * CHANNELS,
                    SLICE_SAMPLES,
                    clusterCount,
                    capacity,
                    quantization,
                    maxIterations
                )
                println(
                    "completed balanced-spherical-lloyd-k$clusterCount slice=$slice " +
                        "status=${result.status} iterations=${result.iterations.size}"
                )
                result
            }
            val labelBytes = ByteArray(TOTAL_SAMPLES)
            for (slice in 0 until DEPTH) {
                labelsBytes(results[slice].labels).copyInto(labelBytes, slice * SLICE_SAMPLES)
            }
            val labelName = "labels-k$clusterCount.bin"
            val labelPath = outputDirectory.resolve(labelName)
            writeBytes(labelPath, labelBytes)
            val readback = readBalancedLabelSlices(labelPath, clusterCount)
            require(Files.readAllBytes(labelPath).contentEquals(labelBytes)) { "$labelName changed during readback" }

            var maxObjectiveReadbackDifference = 0.0
            val clusterings = ArrayList<SliceClustering>(DEPTH)
            var aggregateTransport = TransportStats(0, 0, 0, 0, 0, 0)
            for (slice in 0 until DEPTH) {
                val vectorBase = slice * SLICE_SAMPLES * CHANNELS
                val geometry = computeClusterGeometry(texture.directions, vectorBase, readback[slice], clusterCount)
                maxObjectiveReadbackDifference = max(
                    maxObjectiveReadbackDifference,
                    abs(geometry.explicitSquaredChord - results[slice].objective)
                )
                require(
                    abs(geometry.explicitSquaredChord - results[slice].objective) <=
                        OBJECTIVE_RELATIVE_TOLERANCE * max(1.0, results[slice].objective)
                ) { "Slice $slice K=$clusterCount readback objective differs from search result" }
                aggregateTransport += results[slice].transport
                clusterings += SliceClustering(
                    readback[slice],
                    balancedPermutation(texture.directions, vectorBase, readback[slice], clusterCount),
                    AlgorithmCounts(
                        hilbertKeyEvaluations = (SLICE_SAMPLES + clusterCount).toLong(),
                        initializationDotEvaluations = results[slice].initializationDotEvaluations,
                        assignmentDotEvaluations = results[slice].transport.dotCostEvaluations,
                        lloydIterations = results[slice].iterations.size.toLong(),
                        centerUpdates = results[slice].iterations.count { it.accepted }.toLong() * clusterCount
                    )
                )
            }
            val labelHash = sha256(labelBytes)
            canonicalOutputs[labelName] = linkedMapOf("bytes" to labelBytes.size, "sha256" to labelHash)
            metricCases += caseMetrics(
                texture,
                "balanced-spherical-lloyd-k$clusterCount",
                "balanced-spherical-lloyd",
                "center/member Hilbert order packed into 16x16 Morton-lane groups",
                clusterCount,
                clusterings
            )

            val objectives = DoubleArray(DEPTH) { results[it].objective }
            val statusCounts = linkedMapOf<String, Int>()
            for (result in results) statusCounts[result.status] = (statusCounts[result.status] ?: 0) + 1
            searchCases += linkedMapOf(
                "clusterCount" to clusterCount,
                "capacity" to capacity,
                "labelFile" to labelName,
                "labelSha256" to labelHash,
                "objective" to distribution(objectives),
                "statusCounts" to statusCounts,
                "convergedSlices" to results.count { it.converged },
                "nonconvergedSlices" to results.count { !it.converged },
                "maxObjectiveReadbackAbsDifference" to maxObjectiveReadbackDifference,
                "finalCheckLabelMismatchSlices" to results.count { !it.finalCheckLabelsMatch },
                "maxFinalCheckFloat64Gap" to results.maxOf { it.finalCheckFloat64Gap },
                "transport" to transportStatsJson(aggregateTransport),
                "slices" to results.mapIndexed { slice, result ->
                    linkedMapOf(
                        "slice" to slice,
                        "labelsSha256" to labelsSha256(result.labels),
                        "objective" to result.objective,
                        "status" to result.status,
                        "converged" to result.converged,
                        "iterationCount" to result.iterations.size,
                        "acceptedIterations" to result.iterations.count { it.accepted },
                        "finalCheckQuantizedCost" to result.finalCheckQuantizedCost,
                        "finalCheckCertificateGap" to result.finalCheckCertificateGap,
                        "incumbentQuantizedCost" to result.incumbentQuantizedCost,
                        "finalCheckLabelsMatch" to result.finalCheckLabelsMatch,
                        "finalCheckFloat64Gap" to result.finalCheckFloat64Gap,
                        "quantizationBound" to result.quantizationBound,
                        "initializationDotEvaluations" to result.initializationDotEvaluations,
                        "transport" to transportStatsJson(result.transport),
                        "iterations" to result.iterations.map(::lloydIterationJson)
                    )
                }
            )
            caseTimings += TimedCase(caseId, System.nanoTime() - caseStart)
            println(
                "completed balanced-spherical-lloyd-k$clusterCount " +
                    "statuses=$statusCounts labels=$labelHash"
            )
        }

        val metrics = linkedMapOf<String, Any?>(
            "schema" to "alpha-piscium/stbn-clustering-metrics/v1",
            "input" to linkedMapOf(
                "path" to normalizePath(inputPath),
                "sha256" to inputSha256,
                "bytes" to inputBytes.size,
                "format" to "RGB8_UNORM",
                "width" to WIDTH,
                "height" to HEIGHT,
                "depth" to DEPTH,
                "channels" to CHANNELS,
                "sampleCount" to TOTAL_SAMPLES,
                "originalId" to "slice-local row-major XY index 0..16383",
                "decode" to "normalize((rgb8 / 255.0) * 2.0 - 1.0) in float64"
            ),
            "metricContract" to linkedMapOf(
                "precision" to "IEEE-754 float64",
                "center" to "normalized arithmetic sum; degenerate sums rejected",
                "squaredChord" to "2 * (1 - dot(sample, center))",
                "pairwiseSquaredChord" to "clusterSize^2 - squaredLength(clusterVectorSum)",
                "angularError" to "acos(clamped dot(sample, center)), degrees",
                "quantile" to "sorted linear interpolation at q*(n-1)",
                "minimumCap" to "deterministically shuffled incremental one/two/three-point support solver",
                "horizonStrataZ" to listOf("[0,0.1)", "[0.1,0.25)", "[0.25,0.5)", "[0.5,1]")
            ),
            "cases" to metricCases
        )
        val metricsBytes = canonicalJson(metrics)
        val metricsHash = sha256(metricsBytes)
        canonicalOutputs["metrics.json"] = linkedMapOf("bytes" to metricsBytes.size, "sha256" to metricsHash)

        val search = linkedMapOf<String, Any?>(
            "schema" to "alpha-piscium/stbn-clustering-balanced-lloyd-search/v1",
            "inputSha256" to inputSha256,
            "method" to "balanced-lloyd",
            "seeds" to listOf(1),
            "seedMeaning" to "single deterministic maximum-Z/farthest-first initialization; no stochastic state",
            "clusterCounts" to clusterCounts,
            "quantization" to linkedMapOf(
                "q" to quantization,
                "formula" to "round(2 * (1 - clampedDot) * Q)",
                "integerType" to "signed int64 with checked arithmetic",
                "fixedCenterFloat64SuboptimalityBoundPerSlice" to SLICE_SAMPLES / quantization.toDouble()
            ),
            "assignment" to linkedMapOf(
                "graph" to "complete sample-center bipartite residual graph; no edge pruning",
                "statusRequired" to "OPTIMAL",
                "capacityAndIntegrality" to "validated after every terminal solve",
                "exactCertificate" to "scale integer costs by N+1; require primal minus dual lower bound below N+1",
                "ties" to "ascending bidders; maximum net value then lowest expanded object ID; stable heap price/object ordering",
                "maxIterations" to maxIterations
            ),
            "optimalityScope" to linkedMapOf(
                "fixedCenters" to "global optimum of the quantized complete balanced assignment",
                "jointCentersAndLabels" to "deterministic one-start balanced spherical Lloyd local result; no global claim",
                "incumbent" to "only original-float64 non-increasing center updates are accepted"
            ),
            "readbackValidation" to "label files reread from disk, exact per-slice capacities checked, then existing float64 metric path evaluated",
            "cases" to searchCases
        )
        val searchBytes = canonicalJson(search)
        val searchHash = sha256(searchBytes)
        canonicalOutputs["search.json"] = linkedMapOf("bytes" to searchBytes.size, "sha256" to searchHash)

        val scriptPath = Paths.get("stbn-clustering.main.kts")
        require(Files.isRegularFile(scriptPath)) { "Run this entry point from scripts/" }
        val manifest = linkedMapOf<String, Any?>(
            "schema" to "alpha-piscium/stbn-clustering-manifest/v1",
            "tool" to linkedMapOf(
                "path" to "scripts/stbn-clustering.main.kts",
                "sha256" to sha256(Files.readAllBytes(scriptPath))
            ),
            "input" to linkedMapOf(
                "path" to normalizePath(inputPath),
                "sha256" to inputSha256,
                "bytes" to inputBytes.size,
                "dimensions" to listOf(WIDTH, HEIGHT, DEPTH, CHANNELS)
            ),
            "determinism" to linkedMapOf(
                "seed" to 1,
                "ties" to "ascending bidder queue and expanded object IDs with stable price heaps",
                "sliceParallelism" to "independent slices; serialized in ascending Z",
                "measuredFieldsExcluded" to listOf("wallTimeNanos", "peakJvmHeapBytesObserved", "workersByCase")
            ),
            "algorithm" to linkedMapOf(
                "id" to "epsilon-scaled capacitated auction min-cost flow",
                "clusterCounts" to clusterCounts,
                "quantization" to quantization,
                "maxIterations" to maxIterations,
                "fullAssignmentEdges" to true
            ),
            "canonicalOutputs" to canonicalOutputs
        )
        val manifestBytes = canonicalJson(manifest)
        val manifestHash = sha256(manifestBytes)
        val run = linkedMapOf<String, Any?>(
            "schema" to "alpha-piscium/stbn-clustering-run/v1",
            "canonicalMetricsSha256" to metricsHash,
            "canonicalSearchSha256" to searchHash,
            "manifestSha256" to manifestHash,
            "workersByCase" to workersByCase,
            "maximumJvmHeapBytes" to Runtime.getRuntime().maxMemory(),
            "peakJvmHeapBytesObserved" to sampler.peakBytes(),
            "peakRamMeasurement" to "JVM used heap sampled every 5 ms; excludes native/process overhead",
            "wallTimeNanos" to System.nanoTime() - totalStart,
            "caseWallTimeNanos" to caseTimings.map { linkedMapOf("id" to it.id, "value" to it.wallTimeNanos) }
        )
        val runBytes = canonicalJson(run)
        writeBytes(outputDirectory.resolve("metrics.json"), metricsBytes)
        writeBytes(outputDirectory.resolve("search.json"), searchBytes)
        writeBytes(outputDirectory.resolve("manifest.json"), manifestBytes)
        writeBytes(outputDirectory.resolve("run.json"), runBytes)
        println("metrics.json|${metricsBytes.size}|$metricsHash")
        println("search.json|${searchBytes.size}|$searchHash")
        println("manifest.json|${manifestBytes.size}|$manifestHash")
        println("run.json|${runBytes.size}|${sha256(runBytes)}")
    } finally {
        sampler.close()
    }
}

fun outcomeStatusCounts(outcomes: List<MultistartOutcome>): Map<String, Int> {
    val counts = linkedMapOf<String, Int>()
    for (outcome in outcomes) counts[outcome.status] = (counts[outcome.status] ?: 0) + 1
    return counts
}

fun aggregateTransport(outcomes: List<MultistartOutcome>): TransportStats {
    var result = TransportStats(0, 0, 0, 0, 0, 0)
    for (outcome in outcomes) result += outcome.transport
    return result
}

fun searchMultistart(
    inputPath: Path,
    outputDirectory: Path,
    requestedClusterCounts: IntArray,
    startCount: Int,
    eliteCount: Int,
    quantization: Long,
    maxIterations: Int,
    maximumNewStarts: Int?,
    sliceIds: IntArray = IntArray(DEPTH) { it },
    workerOverride: Int? = null
): Boolean {
    require(requestedClusterCounts.isNotEmpty())
    require(requestedClusterCounts.distinct().size == requestedClusterCounts.size) { "Duplicate cluster count" }
    require(requestedClusterCounts.all { it == 128 || it == 256 }) {
        "Multistart search supports K=128 and K=256"
    }
    require(sliceIds.isNotEmpty() && sliceIds.distinct().size == sliceIds.size && sliceIds.all { it in 0 until DEPTH })
    require(workerOverride == null || workerOverride > 0)
    val sampleTotal = sliceIds.size * SLICE_SAMPLES
    require(startCount >= 3) { "Multistart search requires at least 3 starts" }
    require(eliteCount in 8..16) { "Elite count must be in 8..16" }
    require(maxIterations > 0)
    require(maximumNewStarts == null || maximumNewStarts >= 0)
    validateQuantization(SLICE_SAMPLES, quantization)
    val clusterCounts = requestedClusterCounts.sorted().toIntArray()
    val baseSpecs = multistartSpecs(startCount)
    val crossKEnabled = clusterCounts.contentEquals(intArrayOf(128, 256))
    val totalStart = System.nanoTime()
    val sampler = PeakHeapSampler()
    try {
        val inputBytes = Files.readAllBytes(inputPath)
        val texture = decodeRgb8(inputBytes, WIDTH, HEIGHT, DEPTH)
        val inputSha256 = sha256(inputBytes)
        val scriptPath = Paths.get("stbn-clustering.main.kts")
        require(Files.isRegularFile(scriptPath)) { "Run this entry point from scripts/" }
        val scriptHash = sha256(Files.readAllBytes(scriptPath))
        val initializerAllocation = baseSpecs.groupingBy { it.family }.eachCount().toSortedMap()
        val checkpointRoot = outputDirectory.resolve("checkpoint")
        val checkpointManifest = canonicalJson(linkedMapOf(
            "schema" to "alpha-piscium/stbn-clustering-multistart-checkpoint/v1",
            "checkpointFormatVersion" to MULTISTART_CHECKPOINT_VERSION,
            "toolSha256" to scriptHash,
            "inputSha256" to inputSha256,
            "dimensions" to listOf(WIDTH, HEIGHT, DEPTH, CHANNELS),
            "clusterCounts" to clusterCounts,
            "baseStartCountPerSlice" to startCount,
            "sliceIds" to sliceIds,
            "eliteCount" to eliteCount,
            "quantization" to quantization,
            "maxIterations" to maxIterations,
            "initializerAllocationPerSlice" to initializerAllocation,
            "starts" to baseSpecs.map {
                linkedMapOf(
                    "ordinal" to it.ordinal,
                    "family" to it.family,
                    "familyIndex" to it.familyIndex,
                    "seed" to unsignedHex(it.seed)
                )
            },
            "crossK" to if (crossKEnabled) {
                "one additional K256 start per slice from the objective-best K128 partition split by balanced tangent/PCA"
            } else {
                "disabled because both K128 and K256 were not requested"
            },
            "measuredExecutionFieldsExcluded" to listOf("maximumNewStarts", "workers", "wallTime")
        ))
        requireOrWriteExact(checkpointRoot.resolve("manifest.json"), checkpointManifest)

        val availableProcessors = max(1, Runtime.getRuntime().availableProcessors())
        val maximumHeap = Runtime.getRuntime().maxMemory()
        val largeHeap = maximumHeap >= 2L * 1024L * 1024L * 1024L
        val searchHeap = maximumHeap >= 10L * 1024L * 1024L * 1024L
        val workersByPhase = linkedMapOf<String, Int>()
        val caseTimings = ArrayList<TimedCase>()
        val outcomesByCluster = linkedMapOf<Int, List<List<MultistartOutcome>>>()
        var remainingNewStarts = maximumNewStarts ?: Int.MAX_VALUE
        var totalLoadedCheckpoints = 0
        var totalComputedStarts = 0

        fun workerCount(clusterCount: Int): Int {
            if (workerOverride != null) return workerOverride
            val limit = if (searchHeap) {
                if (clusterCount == 256) 24 else 32
            } else if (largeHeap) {
                if (clusterCount == 256) 12 else 16
            } else {
                if (clusterCount == 256) 3 else 8
            }
            return min(limit, availableProcessors)
        }

        fun recordProgress(phase: String, complete: Boolean) {
            val progress = canonicalJson(linkedMapOf(
                "schema" to "alpha-piscium/stbn-clustering-multistart-progress/v1",
                "phase" to phase,
                "complete" to complete,
                "loadedCheckpointsThisInvocation" to totalLoadedCheckpoints,
                "computedStartsThisInvocation" to totalComputedStarts,
                "remainingNewStartAllowance" to if (maximumNewStarts == null) null else remainingNewStarts
            ))
            writeBytesAtomic(checkpointRoot.resolve("progress.json"), progress)
        }

        fun runBasePhase(clusterCount: Int): Boolean {
            val phaseId = "base-k$clusterCount"
            val phaseStart = System.nanoTime()
            val workers = workerCount(clusterCount)
            workersByPhase[phaseId] = workers
            val phase = executeMultistartPhase(
                texture.directions,
                checkpointRoot,
                clusterCount,
                List(DEPTH) { if (it in sliceIds) baseSpecs else emptyList() },
                quantization,
                maxIterations,
                workers,
                remainingNewStarts
            )
            totalLoadedCheckpoints += phase.loadedCheckpoints
            totalComputedStarts += phase.computedStarts
            if (remainingNewStarts != Int.MAX_VALUE) remainingNewStarts -= phase.computedStarts
            caseTimings += TimedCase(phaseId, System.nanoTime() - phaseStart)
            if (!phase.complete) {
                recordProgress(phaseId, false)
                println(
                    "INCOMPLETE_CHECKPOINT phase=$phaseId loaded=${phase.loadedCheckpoints} " +
                        "computed=${phase.computedStarts}; rerun the identical search command to resume"
                )
                return false
            }
            outcomesByCluster[clusterCount] = phase.outcomesBySlice
            return true
        }

        for (clusterCount in clusterCounts) {
            if (!runBasePhase(clusterCount)) return false
        }

        if (crossKEnabled) {
            val k128 = outcomesByCluster[128] ?: error("K128 base phase missing")
            val parents = sliceIds.associateWith { slice -> selectElites(k128[slice], eliteCount).first() }
            val crossSpecs = List(DEPTH) { slice ->
                if (slice !in sliceIds) return@List emptyList()
                val parent = parents.getValue(slice)
                listOf(StartSpec(
                    startCount,
                    "cross-k128-pca-split",
                    startCount,
                    mix64(parent.seed xor slice.toLong() xor 0x4B313238544F3235L)
                ))
            }
            val phaseId = "cross-k128-to-k256"
            val phaseStart = System.nanoTime()
            val workers = workerCount(256)
            workersByPhase[phaseId] = workers
            val cross = executeMultistartPhase(
                texture.directions,
                checkpointRoot,
                256,
                crossSpecs,
                quantization,
                maxIterations,
                workers,
                remainingNewStarts
            ) { slice, spec ->
                crossKSplitInitialization(
                    texture.directions,
                    slice * SLICE_SAMPLES * CHANNELS,
                    parents.getValue(slice).labels,
                    128,
                    spec.seed
                )
            }
            totalLoadedCheckpoints += cross.loadedCheckpoints
            totalComputedStarts += cross.computedStarts
            if (remainingNewStarts != Int.MAX_VALUE) remainingNewStarts -= cross.computedStarts
            caseTimings += TimedCase(phaseId, System.nanoTime() - phaseStart)
            if (!cross.complete) {
                recordProgress(phaseId, false)
                println(
                    "INCOMPLETE_CHECKPOINT phase=$phaseId loaded=${cross.loadedCheckpoints} " +
                        "computed=${cross.computedStarts}; rerun the identical search command to resume"
                )
                return false
            }
            val baseK256 = outcomesByCluster[256] ?: error("K256 base phase missing")
            outcomesByCluster[256] = List(DEPTH) { slice ->
                (baseK256[slice] + cross.outcomesBySlice[slice]).sortedBy { it.ordinal }
            }
        }

        val metricCases = ArrayList<Map<String, Any?>>()
        val searchCases = ArrayList<Map<String, Any?>>()
        val canonicalOutputs = linkedMapOf<String, Any?>()
        for (clusterCount in clusterCounts) {
            val capacity = SLICE_SAMPLES / clusterCount
            val sliceOutcomes = outcomesByCluster[clusterCount] ?: error("Missing K=$clusterCount outcomes")
            val elitesBySlice = sliceIds.associateWith { slice -> selectElites(sliceOutcomes[slice], eliteCount) }
            val bestBySlice = sliceIds.associateWith { slice -> elitesBySlice.getValue(slice).first() }
            val labelBytes = ByteArray(sampleTotal)
            for ((index, slice) in sliceIds.withIndex()) {
                labelsBytes(canonicalPartitionLabels(bestBySlice.getValue(slice).labels, clusterCount))
                    .copyInto(labelBytes, index * SLICE_SAMPLES)
            }
            val labelName = "labels-k$clusterCount.bin"
            val labelPath = outputDirectory.resolve(labelName)
            writeBytes(labelPath, labelBytes)
            val readback = readBalancedLabelSlices(labelPath, clusterCount, sliceIds.size)
            require(Files.readAllBytes(labelPath).contentEquals(labelBytes)) { "$labelName changed during readback" }
            val labelHash = sha256(labelBytes)
            canonicalOutputs[labelName] = linkedMapOf("bytes" to labelBytes.size, "sha256" to labelHash)

            val clusterings = ArrayList<SliceClustering>(sliceIds.size)
            var maxObjectiveReadbackDifference = 0.0
            for ((index, slice) in sliceIds.withIndex()) {
                val vectorBase = slice * SLICE_SAMPLES * CHANNELS
                val best = bestBySlice.getValue(slice)
                val geometry = computeClusterGeometry(texture.directions, vectorBase, readback[index], clusterCount)
                maxObjectiveReadbackDifference = max(
                    maxObjectiveReadbackDifference,
                    abs(geometry.explicitSquaredChord - best.objective)
                )
                require(abs(geometry.explicitSquaredChord - best.objective) <=
                    OBJECTIVE_RELATIVE_TOLERANCE * max(1.0, best.objective)) {
                    "Slice $slice K=$clusterCount best-label readback objective mismatch"
                }
                clusterings += SliceClustering(
                    readback[index],
                    balancedPermutation(texture.directions, vectorBase, readback[index], clusterCount),
                    AlgorithmCounts(
                        hilbertKeyEvaluations = (SLICE_SAMPLES + clusterCount).toLong(),
                        initializationDotEvaluations = best.initializationDotEvaluations,
                        assignmentDotEvaluations = best.transport.dotCostEvaluations,
                        lloydIterations = best.iterationCount.toLong(),
                        centerUpdates = best.acceptedIterations.toLong() * clusterCount
                    )
                )
            }
            metricCases += caseMetrics(
                texture,
                "multistart-balanced-spherical-lloyd-k$clusterCount",
                "multistart-balanced-spherical-lloyd",
                "canonical objective-best partition; center/member Hilbert order packed into 16x16 Morton-lane groups",
                clusterCount,
                clusterings,
                sliceIds
            )

            val allOutcomes = sliceOutcomes.flatten()
            val bestObjectives = DoubleArray(sliceIds.size) { bestBySlice.getValue(sliceIds[it]).objective }
            val familyAblation = allOutcomes.groupBy { it.family }.toSortedMap().map { (family, outcomes) ->
                val gaps = DoubleArray(outcomes.size) { index ->
                    outcomes[index].objective - bestBySlice.getValue(outcomes[index].slice).objective
                }
                linkedMapOf(
                    "family" to family,
                    "starts" to outcomes.size,
                    "startsPerSlice" to outcomes.groupingBy { it.slice }.eachCount().values.sorted().distinct(),
                    "objective" to distribution(DoubleArray(outcomes.size) { outcomes[it].objective }),
                    "objectiveGapToUnitBest" to distribution(gaps),
                    "statusCounts" to outcomeStatusCounts(outcomes),
                    "converged" to outcomes.count { it.converged },
                    "uniquePartitions" to outcomes.map { it.partitionSha256 }.distinct().size
                )
            }
            val aggregateTransport = aggregateTransport(allOutcomes)
            val crossOutcomes = allOutcomes.filter { it.family == "cross-k128-pca-split" }
            val crossReport = if (clusterCount == 256 && crossKEnabled) {
                val parents = outcomesByCluster[128] ?: error("K128 outcomes missing")
                val parentBest = sliceIds.associateWith { selectElites(parents[it], eliteCount).first() }
                require(crossOutcomes.size == sliceIds.size)
                val splitDeltas = DoubleArray(sliceIds.size)
                var changedAfterFirstAssignment = 0
                var changedAfterLloyd = 0
                for ((index, slice) in sliceIds.withIndex()) {
                    val outcome = crossOutcomes.single { it.slice == slice }
                    val splitObjective = outcome.initializerPartitionObjective
                        ?: error("Cross-K result lacks its split objective")
                    splitDeltas[index] = splitObjective - parentBest.getValue(slice).objective
                    val tolerance = OBJECTIVE_RELATIVE_TOLERANCE * max(1.0, parentBest.getValue(slice).objective)
                    require(splitDeltas[index] <= tolerance) {
                        "Slice $slice cross-K split increased parent objective by ${splitDeltas[index]}"
                    }
                    if (outcome.initializationPartitionSha256 != outcome.firstAssignmentPartitionSha256) {
                        changedAfterFirstAssignment++
                    }
                    if (outcome.initializationPartitionSha256 != outcome.partitionSha256) changedAfterLloyd++
                }
                linkedMapOf(
                    "construction" to "split each objective-best K128 cluster into two equal tangent/PCA children",
                    "additionalStarts" to crossOutcomes.size,
                    "splitObjectiveDeltaFromK128Parent" to distribution(splitDeltas),
                    "allSplitsNonIncreasing" to true,
                    "firstFullEdgeAssignmentChangedPartitionSlices" to changedAfterFirstAssignment,
                    "finalLloydPartitionChangedSlices" to changedAfterLloyd,
                    "hierarchyLocked" to false,
                    "postInitializationAssignmentGraph" to "complete sample-center graph"
                )
            } else null

            searchCases += linkedMapOf(
                "clusterCount" to clusterCount,
                "capacity" to capacity,
                "baseStartsPerSlice" to startCount,
                "additionalStartsPerSlice" to if (clusterCount == 256 && crossKEnabled) 1 else 0,
                "totalRecordedStarts" to allOutcomes.size,
                "labelFile" to labelName,
                "labelSha256" to labelHash,
                "bestObjective" to linkedMapOf(
                    "perSlice" to distribution(bestObjectives),
                    "total" to bestObjectives.sum(),
                    "meanPerSample" to bestObjectives.sum() / sampleTotal
                ),
                "allStartObjective" to distribution(DoubleArray(allOutcomes.size) { allOutcomes[it].objective }),
                "statusCounts" to outcomeStatusCounts(allOutcomes),
                "convergedStarts" to allOutcomes.count { it.converged },
                "nonconvergedStarts" to allOutcomes.count { !it.converged },
                "uniquePartitions" to allOutcomes.map { it.partitionSha256 }.distinct().size,
                "initializerPartitionsChangedByFirstFullEdgeAssignment" to allOutcomes.count {
                    it.initializationPartitionSha256 != null &&
                        it.initializationPartitionSha256 != it.firstAssignmentPartitionSha256
                },
                "maxObjectiveReadbackAbsDifference" to maxObjectiveReadbackDifference,
                "maxFinalCheckFloat64Gap" to allOutcomes.maxOf { it.finalCheckFloat64Gap },
                "transport" to transportStatsJson(aggregateTransport),
                "initializerAblation" to familyAblation,
                "crossK" to crossReport,
                "slices" to sliceIds.map { slice ->
                    val outcomes = sliceOutcomes[slice].sortedBy { it.ordinal }
                    val elites = elitesBySlice.getValue(slice)
                    linkedMapOf(
                        "slice" to slice,
                        "best" to multistartOutcomeJson(elites.first()),
                        "objective" to distribution(DoubleArray(outcomes.size) { outcomes[it].objective }),
                        "statusCounts" to outcomeStatusCounts(outcomes),
                        "bestSoFar" to bestSoFar(outcomes),
                        "eliteCapacity" to eliteCount,
                        "elites" to elites.map(::multistartOutcomeJson),
                        "eliteDiversity" to eliteDiversity(elites),
                        "starts" to outcomes.map(::multistartOutcomeJson)
                    )
                }
            )
            println(
                "completed multistart-balanced-spherical-lloyd-k$clusterCount " +
                    "bestTotal=${bestObjectives.sum()} labels=$labelHash"
            )
        }

        val metrics = linkedMapOf<String, Any?>(
            "schema" to "alpha-piscium/stbn-clustering-metrics/v1",
            "input" to linkedMapOf(
                "path" to normalizePath(inputPath),
                "sha256" to inputSha256,
                "bytes" to inputBytes.size,
                "format" to "RGB8_UNORM",
                "width" to WIDTH,
                "height" to HEIGHT,
                "depth" to DEPTH,
                "channels" to CHANNELS,
                "sampleCount" to TOTAL_SAMPLES,
                "evaluatedSliceIds" to sliceIds,
                "originalId" to "slice-local row-major XY index 0..16383",
                "decode" to "normalize((rgb8 / 255.0) * 2.0 - 1.0) in float64"
            ),
            "metricContract" to linkedMapOf(
                "precision" to "IEEE-754 float64",
                "center" to "normalized arithmetic sum; degenerate sums rejected",
                "squaredChord" to "2 * (1 - dot(sample, center))",
                "pairwiseSquaredChord" to "clusterSize^2 - squaredLength(clusterVectorSum)",
                "angularError" to "acos(clamped dot(sample, center)), degrees",
                "quantile" to "sorted linear interpolation at q*(n-1)",
                "minimumCap" to "deterministically shuffled incremental one/two/three-point support solver",
                "horizonStrataZ" to listOf("[0,0.1)", "[0.1,0.25)", "[0.25,0.5)", "[0.5,1]")
            ),
            "cases" to metricCases
        )
        val metricsBytes = canonicalJson(metrics)
        val metricsHash = sha256(metricsBytes)
        canonicalOutputs["metrics.json"] = linkedMapOf("bytes" to metricsBytes.size, "sha256" to metricsHash)

        val search = linkedMapOf<String, Any?>(
            "schema" to "alpha-piscium/stbn-clustering-multistart-search/v1",
            "inputSha256" to inputSha256,
            "method" to "multistart",
            "clusterCounts" to clusterCounts,
            "sliceIds" to sliceIds,
            "baseStartsPerSlice" to startCount,
            "initializerAllocationPerSlice" to initializerAllocation,
            "eliteCapacity" to eliteCount,
            "elitePolicy" to "objective order, partition SHA-256 tie break, one representative per label-permutation-invariant partition",
            "quantization" to linkedMapOf(
                "q" to quantization,
                "formula" to "round(2 * (1 - clampedDot) * Q)",
                "fixedCenterFloat64SuboptimalityBoundPerSlice" to SLICE_SAMPLES / quantization.toDouble()
            ),
            "initializers" to linkedMapOf(
                "squared-chord-d2" to "seeded weighted sampling proportional to 2 * (1 - dot), without squaring again",
                "recursive-balanced-tangent-pca" to "recursive equal halves along a seeded PCA-guided tangent axis",
                "structured-hilbert" to "seeded azimuth/reflection/cyclic offset of octahedral Hilbert order",
                "cross-k128-pca-split" to if (crossKEnabled) {
                    "one additional K256 start from equal PCA splits of the objective-best K128 result"
                } else null
            ),
            "lloyd" to linkedMapOf(
                "assignmentGraph" to "complete sample-center graph after every initializer; no boundary restriction or edge pruning",
                "fixedCenterStatusRequired" to "OPTIMAL",
                "maxIterations" to maxIterations,
                "incumbent" to "only original-float64 non-increasing center updates are accepted"
            ),
            "optimalityScope" to linkedMapOf(
                "fixedCenters" to "global optimum of the quantized complete balanced assignment",
                "multistart" to "budgeted deterministic local search; no global joint optimum claim"
            ),
            "checkpoint" to linkedMapOf(
                "directory" to "checkpoint",
                "manifestSha256" to sha256(checkpointManifest),
                "atomicUnit" to "one cluster-count/slice/start result",
                "resume" to "exact manifest match required; every loaded label/capacity/objective/partition hash revalidated"
            ),
            "cases" to searchCases
        )
        val searchBytes = canonicalJson(search)
        val searchHash = sha256(searchBytes)
        canonicalOutputs["search.json"] = linkedMapOf("bytes" to searchBytes.size, "sha256" to searchHash)

        val manifest = linkedMapOf<String, Any?>(
            "schema" to "alpha-piscium/stbn-clustering-manifest/v1",
            "tool" to linkedMapOf("path" to "scripts/stbn-clustering.main.kts", "sha256" to scriptHash),
            "input" to linkedMapOf(
                "path" to normalizePath(inputPath),
                "sha256" to inputSha256,
                "bytes" to inputBytes.size,
                "dimensions" to listOf(WIDTH, HEIGHT, DEPTH, CHANNELS)
            ),
            "determinism" to linkedMapOf(
                "seedDerivation" to "SplitMix64 from fixed domain, initializer family, and family-local ordinal",
                "baseStartsPerSlice" to startCount,
                "initializerAllocationPerSlice" to initializerAllocation,
                "crossKEnabled" to crossKEnabled,
                "sliceIds" to sliceIds,
                "ties" to "sample/cluster/expanded-object IDs and canonical partition SHA-256",
                "checkpointManifestSha256" to sha256(checkpointManifest),
                "measuredFieldsExcluded" to listOf(
                    "wallTimeNanos",
                    "peakJvmHeapBytesObserved",
                    "workersByPhase",
                    "loadedCheckpointsThisInvocation",
                    "computedStartsThisInvocation"
                )
            ),
            "algorithm" to linkedMapOf(
                "id" to "deterministic multistart exact-assignment balanced spherical Lloyd",
                "clusterCounts" to clusterCounts,
                "quantization" to quantization,
                "maxIterations" to maxIterations,
                "eliteCount" to eliteCount,
                "fullAssignmentEdges" to true
            ),
            "canonicalOutputs" to canonicalOutputs
        )
        val manifestBytes = canonicalJson(manifest)
        val manifestHash = sha256(manifestBytes)
        val run = linkedMapOf<String, Any?>(
            "schema" to "alpha-piscium/stbn-clustering-run/v1",
            "canonicalMetricsSha256" to metricsHash,
            "canonicalSearchSha256" to searchHash,
            "manifestSha256" to manifestHash,
            "workersByPhase" to workersByPhase,
            "loadedCheckpointsThisInvocation" to totalLoadedCheckpoints,
            "computedStartsThisInvocation" to totalComputedStarts,
            "maximumJvmHeapBytes" to Runtime.getRuntime().maxMemory(),
            "peakJvmHeapBytesObserved" to sampler.peakBytes(),
            "peakRamMeasurement" to "JVM used heap sampled every 5 ms; excludes native/process overhead",
            "wallTimeNanos" to System.nanoTime() - totalStart,
            "phaseWallTimeNanos" to caseTimings.map { linkedMapOf("id" to it.id, "value" to it.wallTimeNanos) }
        )
        val runBytes = canonicalJson(run)
        writeBytes(outputDirectory.resolve("metrics.json"), metricsBytes)
        writeBytes(outputDirectory.resolve("search.json"), searchBytes)
        writeBytes(outputDirectory.resolve("manifest.json"), manifestBytes)
        writeBytes(outputDirectory.resolve("run.json"), runBytes)
        recordProgress("complete", true)
        println("metrics.json|${metricsBytes.size}|$metricsHash")
        println("search.json|${searchBytes.size}|$searchHash")
        println("manifest.json|${manifestBytes.size}|$manifestHash")
        println("run.json|${runBytes.size}|${sha256(runBytes)}")
        return true
    } finally {
        sampler.close()
    }
}

fun refineMultistart(
    inputDirectory: Path,
    outputDirectory: Path,
    methods: Set<String>,
    fullSwapCertificate: Boolean,
    sliceIds: IntArray,
    eliteCount: Int,
    quantization: Long,
    startCount: Int,
    workers: Int
) {
    require(Files.isDirectory(inputDirectory)) { "Refinement input is not a directory: $inputDirectory" }
    require(!Files.exists(outputDirectory)) { "Refinement output already exists: $outputDirectory" }
    require(methods.isNotEmpty() && methods.all { it == "swap" || it == "pair" })
    require(fullSwapCertificate) { "T05 refinement requires --full-swap-certificate" }
    require(sliceIds.isNotEmpty() && sliceIds.distinct().size == sliceIds.size &&
        sliceIds.all { it in 0 until DEPTH })
    require(eliteCount in 1..16)
    require(startCount >= 3)
    validateQuantization(SLICE_SAMPLES, quantization)
    val totalStart = System.nanoTime()
    val inputPath = Paths.get("../shaders/textures/stbn_unitvec3_cosine_2Dx1D_128x128x64.bin")
    val inputBytes = Files.readAllBytes(inputPath)
    val texture = decodeRgb8(inputBytes, WIDTH, HEIGHT, DEPTH)
    val specs = multistartSpecs(startCount)
    val canonicalOutputs = linkedMapOf<String, Any?>()
    val cases = arrayListOf<Map<String, Any?>>()
    val clusterCounts = intArrayOf(128, 256)
    val units = clusterCounts.flatMap { clusterCount -> sliceIds.map { clusterCount to it } }
    val refinedSlices = parallelMap(workers, units) { (clusterCount, slice) ->
        val capacity = SLICE_SAMPLES / clusterCount
        val vectorBase = slice * SLICE_SAMPLES * CHANNELS
        val outcomes = specs.map { spec ->
            readMultistartCheckpoint(
                multistartCheckpointPath(inputDirectory.resolve("checkpoint"), clusterCount, slice, spec.ordinal),
                clusterCount,
                slice,
                spec,
                texture.directions,
                vectorBase,
                SLICE_SAMPLES,
                capacity
            )
        }
        val elites = selectElites(outcomes, max(8, eliteCount)).take(eliteCount)
        val ablationsByOrdinal = linkedMapOf<Int, Map<String, Any?>>()
        val refined = elites.map { source ->
            val swapOnly = refineLabels(
                texture.directions,
                vectorBase,
                source.labels,
                clusterCount,
                capacity,
                quantization,
                setOf("swap"),
                false
            )
            val pairOnly = refineLabels(
                texture.directions,
                vectorBase,
                source.labels,
                clusterCount,
                capacity,
                quantization,
                setOf("pair"),
                false
            )
            val combined = refineLabels(
                texture.directions,
                vectorBase,
                source.labels,
                clusterCount,
                capacity,
                quantization,
                methods,
                false
            )
            ablationsByOrdinal[source.ordinal] = linkedMapOf(
                "lloydSource" to linkedMapOf(
                    "objective" to source.objective,
                    "status" to source.status,
                    "converged" to source.converged
                ),
                "swapOnly" to linkedMapOf(
                    "finalObjective" to swapOnly.finalObjective,
                    "gain" to swapOnly.initialObjective - swapOnly.finalObjective,
                    "acceptedSwapMoves" to swapOnly.screenedSwapMoves,
                    "reconvergenceStatus" to swapOnly.reconvergenceStatus
                ),
                "pairOnly" to linkedMapOf(
                    "finalObjective" to pairOnly.finalObjective,
                    "gain" to pairOnly.initialObjective - pairOnly.finalObjective,
                    "acceptedPairMoves" to pairOnly.pairMoves,
                    "reconvergenceStatus" to pairOnly.reconvergenceStatus
                ),
                "swapThenPair" to linkedMapOf(
                    "finalObjective" to combined.finalObjective,
                    "gain" to combined.initialObjective - combined.finalObjective,
                    "acceptedSwapMoves" to combined.screenedSwapMoves,
                    "acceptedPairMoves" to combined.pairMoves,
                    "reconvergenceStatus" to combined.reconvergenceStatus
                )
            )
            source to combined
        }
        val winner = refined.minWithOrNull { first, second ->
            val objective = first.second.finalObjective.compareTo(second.second.finalObjective)
            if (objective != 0) objective else {
                val partition = partitionSha256(first.second.labels, clusterCount).compareTo(
                    partitionSha256(second.second.labels, clusterCount)
                )
                if (partition != 0) partition else first.first.ordinal.compareTo(second.first.ordinal)
            }
        } ?: error("No elite refinement for K=$clusterCount slice=$slice")
        val certificate = refineLabels(
            texture.directions,
            vectorBase,
            winner.second.labels,
            clusterCount,
            capacity,
            quantization,
            emptySet(),
            true
        )
        require(certificate.certified)
        val expectedPairs = (
            SLICE_SAMPLES.toLong() * SLICE_SAMPLES -
                clusterCount.toLong() * capacity * capacity
            ) / 2
        require(certificate.fullSweepPairs >= expectedPairs &&
            certificate.fullSweepPairs % expectedPairs == 0L) {
            "Incomplete swap sweep accounting for K=$clusterCount slice=$slice"
        }
        val canonical = canonicalPartitionLabels(certificate.labels, clusterCount)
        val finalGeometry = computeClusterGeometry(
            texture.directions, vectorBase, canonical, clusterCount
        )
        require(finalGeometry.counts.all { it == capacity })
        assertClose(
            "canonical refined objective K=$clusterCount slice=$slice",
            finalGeometry.explicitSquaredChord,
            certificate.finalObjective,
            1e-10
        )
        val record = linkedMapOf(
            "slice" to slice,
            "source" to linkedMapOf(
                "ordinal" to winner.first.ordinal,
                "family" to winner.first.family,
                "familyIndex" to winner.first.familyIndex,
                "seed" to unsignedHex(winner.first.seed),
                "status" to winner.first.status,
                "converged" to winner.first.converged,
                "objective" to winner.first.objective,
                "partitionSha256" to winner.first.partitionSha256,
                "checkpointSha256" to sha256(Files.readAllBytes(multistartCheckpointPath(
                    inputDirectory.resolve("checkpoint"), clusterCount, slice, winner.first.ordinal
                )))
            ),
            "eliteCandidates" to refined.map { (source, result) ->
                linkedMapOf(
                    "ordinal" to source.ordinal,
                    "family" to source.family,
                    "sourceObjective" to source.objective,
                    "sourceStatus" to source.status,
                    "sourceConverged" to source.converged,
                    "isolatedAblations" to ablationsByOrdinal.getValue(source.ordinal),
                    "refinement" to localRefinementJson(result)
                )
            },
            "selectedScreenedRefinement" to localRefinementJson(winner.second),
            "finalCertificateRefinement" to localRefinementJson(certificate),
            "finalObjective" to certificate.finalObjective,
            "totalObjectiveGain" to winner.first.objective - certificate.finalObjective,
            "finalLabelsSha256" to labelsSha256(canonical),
            "finalPartitionSha256" to partitionSha256(canonical, clusterCount),
            "completeSweepPairsPerPass" to expectedPairs
        )
        println(
            "refined K=$clusterCount slice=$slice source=${winner.first.ordinal} " +
                "objective=${winner.first.objective}->${certificate.finalObjective} " +
                "certificateSweeps=${certificate.fullSweepCount}"
        )
        canonical to record
    }
    for ((clusterIndex, clusterCount) in clusterCounts.withIndex()) {
        val capacity = SLICE_SAMPLES / clusterCount
        val labelBytes = ByteArray(sliceIds.size * SLICE_SAMPLES)
        val sliceRecords = arrayListOf<Map<String, Any?>>()
        for (sliceIndex in sliceIds.indices) {
            val (canonical, record) = refinedSlices[clusterIndex * sliceIds.size + sliceIndex]
            labelsBytes(canonical).copyInto(labelBytes, sliceIndex * SLICE_SAMPLES)
            sliceRecords += record
        }
        val labelName = "labels-k$clusterCount.bin"
        writeBytes(outputDirectory.resolve(labelName), labelBytes)
        val readback = readBalancedLabelSlices(
            outputDirectory.resolve(labelName), clusterCount, sliceIds.size
        )
        for (index in sliceIds.indices) {
            val geometry = computeClusterGeometry(
                texture.directions,
                sliceIds[index] * SLICE_SAMPLES * CHANNELS,
                readback[index],
                clusterCount
            )
            assertClose(
                "refinement readback K=$clusterCount slice=${sliceIds[index]}",
                geometry.explicitSquaredChord,
                sliceRecords[index]["finalObjective"] as Double,
                1e-10
            )
        }
        canonicalOutputs[labelName] = linkedMapOf(
            "bytes" to labelBytes.size,
            "sha256" to sha256(labelBytes)
        )
        cases += linkedMapOf(
            "clusterCount" to clusterCount,
            "capacity" to capacity,
            "labelFile" to labelName,
            "slices" to sliceRecords
        )
    }
    val refinement = linkedMapOf<String, Any?>(
        "schema" to "alpha-piscium/stbn-clustering-local-refinement/v1",
        "inputTextureSha256" to sha256(inputBytes),
        "inputSearchDirectory" to normalizePath(inputDirectory),
        "sliceIds" to sliceIds,
        "sourceStartsPerSlice" to startCount,
        "refinedEliteCountPerSlice" to eliteCount,
        "methods" to methods.sorted(),
        "objective" to "float64 squared chord with normalized vector-sum centers",
        "swap" to linkedMapOf(
            "delta" to "exact post-update two-cluster vector-sum center loss",
            "acceptance" to "serial full-objective revalidation; strict decrease beyond tolerance",
            "earlyScreen" to "neighbor clusters and boundary members only",
            "certificate" to "streamed complete unordered cross-cluster sample-pair sweep"
        ),
        "pairRepartition" to linkedMapOf(
            "assignment" to "exact fixed-center M-of-2M sorting",
            "freshSplitAxes" to listOf("current center-difference tangent", "deterministic tangent", "orthogonal tangent"),
            "acceptance" to "serial original-objective strict improvement only",
            "scope" to "screened neighboring two-cluster pools; no completeness claim and no r>2"
        ),
        "cases" to cases
    )
    val refinementBytes = canonicalJson(refinement)
    writeBytes(outputDirectory.resolve("refinement.json"), refinementBytes)
    canonicalOutputs["refinement.json"] = linkedMapOf(
        "bytes" to refinementBytes.size,
        "sha256" to sha256(refinementBytes)
    )
    val scriptPath = Paths.get("stbn-clustering.main.kts")
    val manifest = linkedMapOf<String, Any?>(
        "schema" to "alpha-piscium/stbn-clustering-manifest/v1",
        "tool" to linkedMapOf(
            "path" to "scripts/stbn-clustering.main.kts",
            "sha256" to sha256(Files.readAllBytes(scriptPath))
        ),
        "input" to linkedMapOf(
            "texturePath" to normalizePath(inputPath),
            "textureSha256" to sha256(inputBytes),
            "searchDirectory" to normalizePath(inputDirectory),
            "sliceIds" to sliceIds
        ),
        "determinism" to linkedMapOf(
            "eliteOrdering" to "objective, canonical partition SHA-256, source ordinal",
            "ties" to "ascending cluster pair and original sample ID",
            "measuredFieldsExcluded" to listOf("wallTimeNanos")
        ),
        "algorithm" to linkedMapOf(
            "id" to "exact-swap-and-two-cluster-local-refinement",
            "clusterCounts" to listOf(128, 256),
            "quantization" to quantization,
            "fullSwapCertificate" to fullSwapCertificate,
            "certificateTolerance" to 1e-12
        ),
        "canonicalOutputs" to canonicalOutputs
    )
    val manifestBytes = canonicalJson(manifest)
    writeBytes(outputDirectory.resolve("manifest.json"), manifestBytes)
    val runBytes = canonicalJson(linkedMapOf(
        "schema" to "alpha-piscium/stbn-clustering-run/v1",
        "refinementSha256" to sha256(refinementBytes),
        "manifestSha256" to sha256(manifestBytes),
        "workers" to workers,
        "wallTimeNanos" to System.nanoTime() - totalStart
    ))
    writeBytes(outputDirectory.resolve("run.json"), runBytes)
    println("refinement.json|${refinementBytes.size}|${sha256(refinementBytes)}")
    println("manifest.json|${manifestBytes.size}|${sha256(manifestBytes)}")
    println("run.json|${runBytes.size}|${sha256(runBytes)}")
}

fun loadSearchElites(
    searchDirectory: Path,
    directions: DoubleArray,
    vectorBase: Int,
    clusterCount: Int,
    slice: Int,
    capacity: Int,
    eliteCount: Int,
    startCount: Int = 64
): List<IntArray> {
    require(startCount >= 3)
    val outcomes = multistartSpecs(startCount).map { spec ->
        readMultistartCheckpoint(
            multistartCheckpointPath(searchDirectory.resolve("checkpoint"), clusterCount, slice, spec.ordinal),
            clusterCount,
            slice,
            spec,
            directions,
            vectorBase,
            SLICE_SAMPLES,
            capacity
        )
    }
    return selectElites(outcomes, max(8, eliteCount)).take(eliteCount).map { it.labels }
}

fun writeAcceptedNeighborhoodCheckpoints(
    outputDirectory: Path,
    prefix: String,
    result: NeighborhoodStageResult,
    directions: DoubleArray,
    vectorBase: Int,
    clusterCount: Int,
    capacity: Int,
    canonicalOutputs: MutableMap<String, Any?>
): List<Map<String, Any?>> = result.acceptedCheckpoints.mapIndexed { index, labels ->
    val canonical = canonicalPartitionLabels(labels, clusterCount)
    val geometry = computeClusterGeometry(directions, vectorBase, canonical, clusterCount)
    require(geometry.counts.all { it == capacity })
    val name = "checkpoint/$prefix-c${index.toString().padStart(2, '0')}.bin"
    val bytes = labelsBytes(canonical)
    writeBytes(outputDirectory.resolve(name), bytes)
    val hash = sha256(bytes)
    canonicalOutputs[name] = linkedMapOf("bytes" to bytes.size, "sha256" to hash)
    linkedMapOf(
        "path" to name,
        "sha256" to hash,
        "objective" to geometry.explicitSquaredChord,
        "capacity" to capacity,
        "originalIdCount" to labels.size
    )
}

data class RefinedNeighborhoodSlice(
    val labels: IntArray,
    val record: Map<String, Any?>,
    val measurement: Map<String, Any?>,
    val outputs: Map<String, Any?>
)

fun refineNeighborhoods(
    inputDirectories: List<Path>,
    searchDirectory: Path,
    outputDirectory: Path,
    sizes: IntArray,
    sliceIds: IntArray,
    quantization: Long,
    fullSwapCertificate: Boolean,
    startCount: Int,
    workers: Int
) {
    require(inputDirectories.isNotEmpty() && inputDirectories.distinct().size == inputDirectories.size)
    inputDirectories.forEach {
        require(Files.isDirectory(it)) { "Neighborhood input is not a directory: $it" }
    }
    val inputDirectory = inputDirectories.first()
    require(Files.isDirectory(searchDirectory)) { "Search input is not a directory: $searchDirectory" }
    require(!Files.exists(outputDirectory)) { "Neighborhood output already exists: $outputDirectory" }
    require(sizes.contentEquals(intArrayOf(2, 4, 8))) { "T06 requires ordered --sizes 2,4,8" }
    require(fullSwapCertificate) { "T06 requires --full-swap-certificate" }
    require(startCount >= 3)
    require(sliceIds.isNotEmpty() && sliceIds.distinct().size == sliceIds.size &&
        sliceIds.all { it in 0 until DEPTH })
    validateQuantization(SLICE_SAMPLES, quantization)
    val totalStart = System.nanoTime()
    val sampler = PeakHeapSampler()
    try {
        val inputPath = Paths.get("../shaders/textures/stbn_unitvec3_cosine_2Dx1D_128x128x64.bin")
        val inputBytes = Files.readAllBytes(inputPath)
        val texture = decodeRgb8(inputBytes, WIDTH, HEIGHT, DEPTH)
        val inputManifests = inputDirectories.map { Files.readString(it.resolve("manifest.json")) }
        val inputReceipts = inputDirectories.mapIndexed { index, directory ->
            require(jsonStringField(inputManifests[index], "textureSha256") == sha256(inputBytes)) {
                "Refinement texture differs from the current source: $directory"
            }
            linkedMapOf(
                "directory" to normalizePath(directory),
                "manifestSha256" to sha256(Files.readAllBytes(directory.resolve("manifest.json")))
            )
        }
        val inputRefinementManifestHash = inputReceipts.first().getValue("manifestSha256")
        val searchManifestHash = sha256(Files.readAllBytes(searchDirectory.resolve("manifest.json")))
        val canonicalOutputs = linkedMapOf<String, Any?>()
        val cases = arrayListOf<Map<String, Any?>>()
        val runCases = arrayListOf<Map<String, Any?>>()
        val clusterCounts = intArrayOf(128, 256)
        val inputSlicesByDirectory = inputDirectories.mapIndexed { directoryIndex, directory ->
            clusterCounts.map { clusterCount ->
                val labelName = "labels-k$clusterCount.bin"
                val inputLabelPath = directory.resolve(labelName)
                val labelMarker = "\"$labelName\":{"
                require(inputManifests[directoryIndex].contains(labelMarker)) {
                    "Manifest lacks $labelName: $directory"
                }
                require(sha256(Files.readAllBytes(inputLabelPath)) == jsonStringField(
                    inputManifests[directoryIndex].substringAfter(labelMarker), "sha256"
                )) { "Label hash differs from manifest: $inputLabelPath" }
                val labelBytes = Files.size(inputLabelPath)
                require(labelBytes % SLICE_SAMPLES == 0L) {
                    "Label input is not slice aligned: ${normalizePath(inputLabelPath)}"
                }
                val inputSliceCount = (labelBytes / SLICE_SAMPLES).toInt()
                require(inputSliceCount == sliceIds.size) {
                    "Expected ${sliceIds.size} label slices in ${normalizePath(inputLabelPath)}, found $inputSliceCount"
                }
                readBalancedLabelSlices(inputLabelPath, clusterCount, inputSliceCount)
            }
        }
        val units = clusterCounts.indices.flatMap { clusterIndex -> sliceIds.indices.map { clusterIndex to it } }
        val refinedSlices = parallelMap(workers, units) { (clusterIndex, sliceIndex) ->
            val clusterCount = clusterCounts[clusterIndex]
            val capacity = SLICE_SAMPLES / clusterCount
            val slice = sliceIds[sliceIndex]
            val sliceOutputs = linkedMapOf<String, Any?>()
            val vectorBase = slice * SLICE_SAMPLES * CHANNELS
            val inputObjectives = inputSlicesByDirectory.map {
                computeClusterGeometry(
                    texture.directions, vectorBase, it[clusterIndex][sliceIndex], clusterCount
                ).explicitSquaredChord
            }
            val inputIndex = inputObjectives.indices.minBy { inputObjectives[it] }
            val initialLabels = inputSlicesByDirectory[inputIndex][clusterIndex][sliceIndex]
            val initialObjective = inputObjectives[inputIndex]
            val elites = loadSearchElites(
                searchDirectory,
                texture.directions,
                vectorBase,
                clusterCount,
                slice,
                capacity,
                8,
                startCount
            )
            val scheduleSeed = mix64(
                0x5641524941424C45L xor clusterCount.toLong().shl(32) xor slice.toLong()
            )

            val ablationStart2 = System.nanoTime()
            val ablation2 = runPairStage(
                texture.directions, vectorBase, initialLabels, clusterCount, capacity,
                4, quantization
            )
            val ablationTime2 = System.nanoTime() - ablationStart2
            val ablationStart4 = System.nanoTime()
            val ablation4 = runMultiClusterStage(
                texture.directions, vectorBase, initialLabels, clusterCount, capacity,
                4, 4, elites, quantization, mix64(scheduleSeed xor 4L)
            )
            val ablationTime4 = System.nanoTime() - ablationStart4
            val ablationStart8 = System.nanoTime()
            val ablation8 = runMultiClusterStage(
                texture.directions, vectorBase, initialLabels, clusterCount, capacity,
                8, 4, elites, quantization, mix64(scheduleSeed xor 8L)
            )
            val ablationTime8 = System.nanoTime() - ablationStart8

            val stage2 = runPairStage(
                texture.directions, vectorBase, initialLabels, clusterCount, capacity,
                4, quantization
            )
            val stage4 = runMultiClusterStage(
                texture.directions, vectorBase, stage2.labels, clusterCount, capacity,
                4, 4, elites, quantization, mix64(scheduleSeed xor 4L)
            )
            val stage8 = runMultiClusterStage(
                texture.directions, vectorBase, stage4.labels, clusterCount, capacity,
                8, 4, elites, quantization, mix64(scheduleSeed xor 8L)
            )
            val finalLabels = stage8.labels.copyOf()
            val beforeFinalReconvergence = computeClusterGeometry(
                texture.directions, vectorBase, finalLabels, clusterCount
            ).explicitSquaredChord
            val finalReconvergenceStart = System.nanoTime()
            val finalReconvergence = reconvergeNeighborhoodStage(
                texture.directions, vectorBase, finalLabels, clusterCount, capacity, quantization
            )
            val finalReconvergenceTime = System.nanoTime() - finalReconvergenceStart
            val afterFinalReconvergence = finalReconvergence.third
            require(afterFinalReconvergence <= beforeFinalReconvergence)
            val certificateStart = System.nanoTime()
            val certificate = refineLabels(
                texture.directions,
                vectorBase,
                finalLabels,
                clusterCount,
                capacity,
                quantization,
                emptySet(),
                true
            )
            val certificateTime = System.nanoTime() - certificateStart
            require(certificate.certified)
            val expectedPairs = (
                SLICE_SAMPLES.toLong() * SLICE_SAMPLES -
                    clusterCount.toLong() * capacity * capacity
                ) / 2
            require(certificate.fullSweepPairs >= expectedPairs &&
                certificate.fullSweepPairs % expectedPairs == 0L)
            require(certificate.finalObjective <= initialObjective)
            val canonical = canonicalPartitionLabels(certificate.labels, clusterCount)
            val canonicalGeometry = computeClusterGeometry(
                texture.directions, vectorBase, canonical, clusterCount
            )
            require(canonicalGeometry.counts.all { it == capacity })
            assertClose(
                "VNS canonical objective K=$clusterCount slice=$slice",
                canonicalGeometry.explicitSquaredChord,
                certificate.finalObjective,
                1e-10
            )
            val checkpointPrefix = "k$clusterCount-s${slice.toString().padStart(2, '0')}"
            val checkpointRecords = linkedMapOf(
                "ablationR2" to writeAcceptedNeighborhoodCheckpoints(
                    outputDirectory, "$checkpointPrefix-ablation-r2", ablation2,
                    texture.directions, vectorBase, clusterCount, capacity, sliceOutputs
                ),
                "ablationR4" to writeAcceptedNeighborhoodCheckpoints(
                    outputDirectory, "$checkpointPrefix-ablation-r4", ablation4,
                    texture.directions, vectorBase, clusterCount, capacity, sliceOutputs
                ),
                "ablationR8" to writeAcceptedNeighborhoodCheckpoints(
                    outputDirectory, "$checkpointPrefix-ablation-r8", ablation8,
                    texture.directions, vectorBase, clusterCount, capacity, sliceOutputs
                ),
                "orderedR2" to writeAcceptedNeighborhoodCheckpoints(
                    outputDirectory, "$checkpointPrefix-ordered-r2", stage2,
                    texture.directions, vectorBase, clusterCount, capacity, sliceOutputs
                ),
                "orderedR4" to writeAcceptedNeighborhoodCheckpoints(
                    outputDirectory, "$checkpointPrefix-ordered-r4", stage4,
                    texture.directions, vectorBase, clusterCount, capacity, sliceOutputs
                ),
                "orderedR8" to writeAcceptedNeighborhoodCheckpoints(
                    outputDirectory, "$checkpointPrefix-ordered-r8", stage8,
                    texture.directions, vectorBase, clusterCount, capacity, sliceOutputs
                )
            )
            val record = linkedMapOf(
                "slice" to slice,
                "selectedInputDirectory" to normalizePath(inputDirectories[inputIndex]),
                "inputObjectives" to inputObjectives,
                "initialObjective" to initialObjective,
                "scheduleSeed" to unsignedHex(scheduleSeed),
                "eliteDisagreementPartitions" to elites.map {
                    partitionSha256(it, clusterCount)
                },
                "isolatedAblations" to linkedMapOf(
                    "r2" to neighborhoodStageJson(ablation2),
                    "r4" to neighborhoodStageJson(ablation4),
                    "r8" to neighborhoodStageJson(ablation8)
                ),
                "orderedStages" to listOf(
                    neighborhoodStageJson(stage2),
                    neighborhoodStageJson(stage4),
                    neighborhoodStageJson(stage8)
                ),
                "acceptedNeighborhoodCheckpoints" to checkpointRecords,
                "finalGlobalReconvergence" to linkedMapOf(
                    "objectiveBefore" to beforeFinalReconvergence,
                    "objectiveAfter" to afterFinalReconvergence,
                    "status" to finalReconvergence.first,
                    "accepted" to finalReconvergence.second
                ),
                "finalSwapCertificate" to localRefinementJson(certificate),
                "finalObjective" to certificate.finalObjective,
                "totalGain" to initialObjective - certificate.finalObjective,
                "labelsSha256" to labelsSha256(canonical),
                "partitionSha256" to partitionSha256(canonical, clusterCount),
                "completeSweepPairsPerPass" to expectedPairs
            )
            val measurement = linkedMapOf(
                "slice" to slice,
                "isolatedAblationWallTimeNanos" to linkedMapOf(
                    "r2" to ablationTime2,
                    "r4" to ablationTime4,
                    "r8" to ablationTime8
                ),
                "isolatedAblationAttempts" to listOf(
                    neighborhoodMeasurementsJson(ablation2),
                    neighborhoodMeasurementsJson(ablation4),
                    neighborhoodMeasurementsJson(ablation8)
                ),
                "orderedStageAttempts" to listOf(
                    neighborhoodMeasurementsJson(stage2),
                    neighborhoodMeasurementsJson(stage4),
                    neighborhoodMeasurementsJson(stage8)
                ),
                "finalGlobalReconvergenceWallTimeNanos" to finalReconvergenceTime,
                "finalSwapCertificateWallTimeNanos" to certificateTime
            )
            println(
                "VNS K=$clusterCount slice=$slice objective=$initialObjective->${certificate.finalObjective} " +
                    "accepted=${stage2.acceptedAttempts}/${stage4.acceptedAttempts}/${stage8.acceptedAttempts} " +
                    "certificateSweeps=${certificate.fullSweepCount}"
            )
            RefinedNeighborhoodSlice(canonical, record, measurement, sliceOutputs)
        }
        for ((clusterIndex, clusterCount) in clusterCounts.withIndex()) {
            val capacity = SLICE_SAMPLES / clusterCount
            val inputLabelPath = inputDirectory.resolve("labels-k$clusterCount.bin")
            val outputLabels = ByteArray(sliceIds.size * SLICE_SAMPLES)
            val sliceRecords = arrayListOf<Map<String, Any?>>()
            val sliceMeasurements = arrayListOf<Map<String, Any?>>()
            for (sliceIndex in sliceIds.indices) {
                val refined = refinedSlices[clusterIndex * sliceIds.size + sliceIndex]
                labelsBytes(refined.labels).copyInto(outputLabels, sliceIndex * SLICE_SAMPLES)
                sliceRecords += refined.record
                sliceMeasurements += refined.measurement
                canonicalOutputs.putAll(refined.outputs)
            }
            val labelName = "labels-k$clusterCount.bin"
            writeBytes(outputDirectory.resolve(labelName), outputLabels)
            val readback = readBalancedLabelSlices(
                outputDirectory.resolve(labelName), clusterCount, sliceIds.size
            )
            for (index in sliceIds.indices) {
                val geometry = computeClusterGeometry(
                    texture.directions,
                    sliceIds[index] * SLICE_SAMPLES * CHANNELS,
                    readback[index],
                    clusterCount
                )
                assertClose(
                    "VNS readback K=$clusterCount slice=${sliceIds[index]}",
                    geometry.explicitSquaredChord,
                    sliceRecords[index]["finalObjective"] as Double,
                    1e-10
                )
            }
            canonicalOutputs[labelName] = linkedMapOf(
                "bytes" to outputLabels.size,
                "sha256" to sha256(outputLabels)
            )
            cases += linkedMapOf(
                "clusterCount" to clusterCount,
                "capacity" to capacity,
                "inputLabelSha256" to sha256(Files.readAllBytes(inputLabelPath)),
                "outputLabelFile" to labelName,
                "slices" to sliceRecords
            )
            runCases += linkedMapOf(
                "clusterCount" to clusterCount,
                "slices" to sliceMeasurements
            )
        }
        val neighborhoods = linkedMapOf<String, Any?>(
            "schema" to "alpha-piscium/stbn-clustering-variable-neighborhood/v1",
            "inputTextureSha256" to sha256(inputBytes),
            "inputRefinementDirectory" to normalizePath(inputDirectory),
            "inputRefinementManifestSha256" to inputRefinementManifestHash,
            "inputRefinements" to inputReceipts,
            "sourceSearchDirectory" to normalizePath(searchDirectory),
            "sourceSearchManifestSha256" to searchManifestHash,
            "sliceIds" to sliceIds,
            "sizes" to sizes,
            "attemptBudgetPerSize" to linkedMapOf("r2" to 4, "r4" to 4, "r8" to 4),
            "freshStartsPerMultiClusterNeighborhood" to listOf(
                "recursive-balanced-tangent-pca seed 0",
                "recursive-balanced-tangent-pca seed 1",
                "squared-chord D2",
                "structured Hilbert"
            ),
            "incumbentStart" to "current normalized centers",
            "assignment" to "complete pooled sample-center graph through the exact quantized balanced transport oracle",
            "acceptance" to "serial strict float64 original squared-chord objective improvement only",
            "selectionSchedule" to listOf(
                "current center adjacency",
                "high-loss/elongated/horizon-heavy seed expanded by adjacency",
                "eight-elite disagreement seed expanded by adjacency",
                "recorded deterministic random cluster set"
            ),
            "optimalityScope" to "budgeted r=2/4/8 variable-neighborhood search; final certificate is one-swap local only",
            "cases" to cases
        )
        val neighborhoodsBytes = canonicalJson(neighborhoods)
        writeBytes(outputDirectory.resolve("neighborhoods.json"), neighborhoodsBytes)
        canonicalOutputs["neighborhoods.json"] = linkedMapOf(
            "bytes" to neighborhoodsBytes.size,
            "sha256" to sha256(neighborhoodsBytes)
        )
        val scriptPath = Paths.get("stbn-clustering.main.kts")
        val manifest = linkedMapOf<String, Any?>(
            "schema" to "alpha-piscium/stbn-clustering-manifest/v1",
            "tool" to linkedMapOf(
                "path" to "scripts/stbn-clustering.main.kts",
                "sha256" to sha256(Files.readAllBytes(scriptPath))
            ),
            "input" to linkedMapOf(
                "texturePath" to normalizePath(inputPath),
                "textureSha256" to sha256(inputBytes),
                "refinementDirectory" to normalizePath(inputDirectory),
                "refinementManifestSha256" to inputRefinementManifestHash,
                "refinements" to inputReceipts,
                "searchDirectory" to normalizePath(searchDirectory),
                "searchManifestSha256" to searchManifestHash,
                "sliceIds" to sliceIds
            ),
            "determinism" to linkedMapOf(
                "neighborhoodSchedule" to "fixed strategy cycle with manifest seeds",
                "ties" to "ascending cluster/sample IDs and canonical partition SHA-256",
                "measuredFieldsExcluded" to listOf("wallTimeNanos", "heapBytes")
            ),
            "algorithm" to linkedMapOf(
                "id" to "ordered-r2-r4-r8-variable-neighborhood-refinement",
                "sizes" to sizes,
                "attemptBudgetPerSize" to listOf(4, 4, 4),
                "freshStartsPerMultiClusterNeighborhood" to 4,
                "quantization" to quantization,
                "fullSwapCertificate" to true,
                "certificateTolerance" to 1e-12
            ),
            "canonicalOutputs" to canonicalOutputs
        )
        val manifestBytes = canonicalJson(manifest)
        writeBytes(outputDirectory.resolve("manifest.json"), manifestBytes)
        val run = linkedMapOf<String, Any?>(
            "schema" to "alpha-piscium/stbn-clustering-run/v1",
            "neighborhoodsSha256" to sha256(neighborhoodsBytes),
            "manifestSha256" to sha256(manifestBytes),
            "perNeighborhoodMeasurements" to runCases,
            "workers" to workers,
            "maximumJvmHeapBytes" to Runtime.getRuntime().maxMemory(),
            "peakJvmHeapBytesObserved" to sampler.peakBytes(),
            "peakRamMeasurement" to "JVM used heap sampled every 5 ms; per-attempt fields are endpoint measurements",
            "wallTimeNanos" to System.nanoTime() - totalStart
        )
        val runBytes = canonicalJson(run)
        writeBytes(outputDirectory.resolve("run.json"), runBytes)
        println("neighborhoods.json|${neighborhoodsBytes.size}|${sha256(neighborhoodsBytes)}")
        println("manifest.json|${manifestBytes.size}|${sha256(manifestBytes)}")
        println("run.json|${runBytes.size}|${sha256(runBytes)}")
    } finally {
        sampler.close()
    }
}

fun objectiveScores(
    directions: DoubleArray,
    vectorBase: Int,
    labels: IntArray,
    clusterCount: Int
): ObjectiveScores {
    val chord = computeClusterGeometry(directions, vectorBase, labels, clusterCount).explicitSquaredChord
    val angular = angularGeometry(directions, vectorBase, labels, clusterCount)
    require(angular.converged)
    val angles = DoubleArray(labels.size)
    for (sample in labels.indices) {
        val offset = vectorBase + sample * CHANNELS
        val centerOffset = labels[sample] * CHANNELS
        val dot = (directions[offset] * angular.centers[centerOffset] +
            directions[offset + 1] * angular.centers[centerOffset + 1] +
            directions[offset + 2] * angular.centers[centerOffset + 2]).coerceIn(-1.0, 1.0)
        angles[sample] = acos(dot) * RAD_TO_DEG
    }
    angles.sort()
    val maximumCap = capsForLabels(directions, vectorBase, labels, clusterCount).maxOf { it.radiusDeg }
    return ObjectiveScores(
        chord,
        angular.explicitSquaredAngle,
        quantileFromSorted(angles, 0.99),
        maximumCap
    )
}

fun aggregateObjectiveScores(
    directions: DoubleArray,
    sliceIds: IntArray,
    labelsBySlice: List<IntArray>,
    clusterCount: Int
): ObjectiveScores {
    val allAngles = DoubleArray(sliceIds.size * SLICE_SAMPLES)
    var chord = 0.0
    var angular2 = 0.0
    var maximumCap = 0.0
    for ((index, slice) in sliceIds.withIndex()) {
        val vectorBase = slice * SLICE_SAMPLES * CHANNELS
        val labels = labelsBySlice[index]
        chord += computeClusterGeometry(directions, vectorBase, labels, clusterCount).explicitSquaredChord
        val angular = angularGeometry(directions, vectorBase, labels, clusterCount)
        require(angular.converged)
        angular2 += angular.explicitSquaredAngle
        for (sample in labels.indices) {
            val offset = vectorBase + sample * CHANNELS
            val centerOffset = labels[sample] * CHANNELS
            val dot = (directions[offset] * angular.centers[centerOffset] +
                directions[offset + 1] * angular.centers[centerOffset + 1] +
                directions[offset + 2] * angular.centers[centerOffset + 2]).coerceIn(-1.0, 1.0)
            allAngles[index * SLICE_SAMPLES + sample] = acos(dot) * RAD_TO_DEG
        }
        maximumCap = max(
            maximumCap,
            capsForLabels(directions, vectorBase, labels, clusterCount).maxOf { it.radiusDeg }
        )
    }
    allAngles.sort()
    return ObjectiveScores(chord, angular2, quantileFromSorted(allAngles, 0.99), maximumCap)
}

fun objectiveScoresJson(scores: ObjectiveScores): Map<String, Any?> = linkedMapOf(
    "squaredChordTotal" to scores.chord,
    "trueSquaredAngularTotalRad2" to scores.angular2,
    "trueAngularP99Deg" to scores.p99AngleDeg,
    "maximumMinimumCapRadiusDeg" to scores.maximumCapRadiusDeg
)

fun trueAngularCaseMetrics(
    texture: DecodedTexture,
    sliceIds: IntArray,
    labelsBySlice: List<IntArray>,
    clusterCount: Int
): Map<String, Any?> {
    require(sliceIds.size == labelsBySlice.size)
    val allAngles = DoubleArray(sliceIds.size * SLICE_SAMPLES)
    val allSquaredAngles = DoubleArray(allAngles.size)
    val capRadii = DoubleArray(sliceIds.size * clusterCount)
    val horizonBounds = doubleArrayOf(0.1, 0.25, 0.5)
    val horizonCounts = IntArray(4)
    for ((sliceIndex, slice) in sliceIds.withIndex()) for (sample in 0 until SLICE_SAMPLES) {
        val z = texture.directions[(slice * SLICE_SAMPLES + sample) * CHANNELS + 2]
        val stratum = when {
            z < horizonBounds[0] -> 0
            z < horizonBounds[1] -> 1
            z < horizonBounds[2] -> 2
            else -> 3
        }
        horizonCounts[stratum]++
    }
    val horizonAngles = Array(4) { DoubleArray(horizonCounts[it]) }
    val horizonSquared = Array(4) { DoubleArray(horizonCounts[it]) }
    val horizonCursors = IntArray(4)
    var total = 0.0
    var capTests = 0L
    var maxCapViolation = 0.0
    var nonconvergedSlices = 0
    val perSlice = arrayListOf<Map<String, Any?>>()
    for ((sliceIndex, slice) in sliceIds.withIndex()) {
        val vectorBase = slice * SLICE_SAMPLES * CHANNELS
        val labels = labelsBySlice[sliceIndex]
        val geometry = angularGeometry(texture.directions, vectorBase, labels, clusterCount)
        if (!geometry.converged) nonconvergedSlices++
        total += geometry.explicitSquaredAngle
        val sliceAngles = DoubleArray(SLICE_SAMPLES)
        val sliceSquared = DoubleArray(SLICE_SAMPLES)
        for (sample in labels.indices) {
            val offset = vectorBase + sample * CHANNELS
            val centerOffset = labels[sample] * CHANNELS
            val dot = (texture.directions[offset] * geometry.centers[centerOffset] +
                texture.directions[offset + 1] * geometry.centers[centerOffset + 1] +
                texture.directions[offset + 2] * geometry.centers[centerOffset + 2]).coerceIn(-1.0, 1.0)
            val angle = acos(dot)
            val angleDeg = angle * RAD_TO_DEG
            val global = sliceIndex * SLICE_SAMPLES + sample
            sliceAngles[sample] = angleDeg
            sliceSquared[sample] = angle * angle
            allAngles[global] = angleDeg
            allSquaredAngles[global] = angle * angle
            val z = texture.directions[offset + 2]
            val stratum = when {
                z < horizonBounds[0] -> 0
                z < horizonBounds[1] -> 1
                z < horizonBounds[2] -> 2
                else -> 3
            }
            val cursor = horizonCursors[stratum]++
            horizonAngles[stratum][cursor] = angleDeg
            horizonSquared[stratum][cursor] = angle * angle
        }
        val caps = capsForLabels(texture.directions, vectorBase, labels, clusterCount)
        for (cluster in 0 until clusterCount) {
            capRadii[sliceIndex * clusterCount + cluster] = caps[cluster].radiusDeg
            capTests += caps[cluster].containmentTests
            maxCapViolation = max(maxCapViolation, caps[cluster].maxCosViolation)
        }
        perSlice += linkedMapOf(
            "slice" to slice,
            "trueSquaredAngularTotalRad2" to geometry.explicitSquaredAngle,
            "centersConverged" to geometry.converged,
            "centerIterations" to geometry.centerIterations,
            "centerStatusCounts" to geometry.centerStatusCounts,
            "centersFloat64" to geometry.centers,
            "angularErrorDeg" to distribution(sliceAngles),
            "squaredAngularRad2" to distribution(sliceSquared),
            "minimumCapRadiusDeg" to distribution(DoubleArray(clusterCount) {
                capRadii[sliceIndex * clusterCount + it]
            }),
            "minimumCapsFloat64" to caps.map { cap -> linkedMapOf(
                "center" to cap.center,
                "cosRadius" to cap.cosRadius,
                "radiusDeg" to cap.radiusDeg,
                "supportSize" to cap.supportSize,
                "maxContainmentViolationCos" to cap.maxCosViolation
            ) }
        )
    }
    require(horizonCursors.contentEquals(horizonCounts))
    return linkedMapOf(
        "center" to "intrinsic Frechet/Karcher center estimates; convergence reported per slice",
        "nonconvergedSlices" to nonconvergedSlices,
        "trueSquaredAngularTotalRad2" to total,
        "squaredAngularRad2" to distribution(allSquaredAngles),
        "angularErrorDeg" to distribution(allAngles),
        "minimumCapRadiusDeg" to distribution(capRadii),
        "minimumCapContainmentTests" to capTests,
        "maximumCapContainmentViolationCos" to maxCapViolation,
        "horizonStrata" to (0..3).map { stratum ->
            linkedMapOf(
                "z" to arrayOf("[0,0.1)", "[0.1,0.25)", "[0.25,0.5)", "[0.5,1]")[stratum],
                "squaredAngularRad2" to distribution(horizonSquared[stratum]),
                "angularErrorDeg" to distribution(horizonAngles[stratum])
            )
        },
        "perSlice" to perSlice
    )
}

fun comparisonStarts(
    searchDirectory: Path,
    directions: DoubleArray,
    vectorBase: Int,
    clusterCount: Int,
    slice: Int,
    capacity: Int,
    chordLabels: IntArray
): List<Pair<String, IntArray>> {
    val starts = arrayListOf("t06-chord-final" to chordLabels)
    val seen = linkedSetOf(partitionSha256(chordLabels, clusterCount))
    for (elite in loadSearchElites(
        searchDirectory, directions, vectorBase, clusterCount, slice, capacity, 8
    )) {
        val partition = partitionSha256(elite, clusterCount)
        if (seen.add(partition)) starts += "t04-chord-elite-${starts.size}" to elite
        if (starts.size == 4) break
    }
    require(starts.size == 4) { "T07 requires four distinct chord starts" }
    return starts
}

fun compareObjectives(
    inputDirectory: Path,
    searchDirectory: Path,
    outputDirectory: Path,
    objectives: Set<String>,
    sliceIds: IntArray,
    quantization: Long
) {
    require(Files.isDirectory(inputDirectory) && Files.isDirectory(searchDirectory))
    require(!Files.exists(outputDirectory)) { "Objective comparison output already exists: $outputDirectory" }
    require(objectives == setOf("chord", "angular2", "radius")) {
        "T07 requires --objectives chord,angular2,radius"
    }
    require(sliceIds.contentEquals(intArrayOf(0, 9, 18))) { "T07 is restricted to pilot slices 0,9,18" }
    validateQuantization(SLICE_SAMPLES, quantization)
    val totalStart = System.nanoTime()
    val sampler = PeakHeapSampler()
    try {
        val inputPath = Paths.get("../shaders/textures/stbn_unitvec3_cosine_2Dx1D_128x128x64.bin")
        val inputBytes = Files.readAllBytes(inputPath)
        val texture = decodeRgb8(inputBytes, WIDTH, HEIGHT, DEPTH)
        val canonicalOutputs = linkedMapOf<String, Any?>()
        val comparisonCases = arrayListOf<Map<String, Any?>>()
        val metricCases = arrayListOf<Map<String, Any?>>()
        val runCases = arrayListOf<Map<String, Any?>>()
        for (clusterCount in intArrayOf(128, 256)) {
            val capacity = SLICE_SAMPLES / clusterCount
            val chordSlices = readBalancedLabelSlices(
                inputDirectory.resolve("labels-k$clusterCount.bin"), clusterCount, sliceIds.size
            )
            val labelsByCandidate = linkedMapOf(
                "chord-best" to chordSlices.map { it.copyOf() }.toMutableList(),
                "angular-best" to arrayListOf<IntArray>(),
                "radius-best" to arrayListOf<IntArray>(),
                "mean-tail" to arrayListOf<IntArray>()
            )
            val sliceRecords = arrayListOf<Map<String, Any?>>()
            val sliceMeasurements = arrayListOf<Map<String, Any?>>()
            for ((sliceIndex, slice) in sliceIds.withIndex()) {
                val vectorBase = slice * SLICE_SAMPLES * CHANNELS
                val chordLabels = chordSlices[sliceIndex]
                val starts = comparisonStarts(
                    searchDirectory, texture.directions, vectorBase, clusterCount, slice, capacity, chordLabels
                )
                val angularRuns = arrayListOf<Pair<String, AngularLloydResult>>()
                val angularMeasurements = arrayListOf<Map<String, Any?>>()
                for ((id, startLabels) in starts) {
                    val start = System.nanoTime()
                    val result = angularBalancedLloyd(
                        texture.directions, vectorBase, startLabels, clusterCount, capacity,
                        quantization, 4
                    )
                    angularMeasurements += linkedMapOf(
                        "start" to id,
                        "wallTimeNanos" to System.nanoTime() - start,
                        "heapBytesAfter" to usedJvmHeapBytes()
                    )
                    angularRuns += id to result
                }
                var angularWinner = angularRuns.first()
                for (candidate in angularRuns.drop(1)) {
                    val tolerance = 1e-12 * max(
                        1.0, angularWinner.second.geometry.explicitSquaredAngle
                    )
                    if (candidate.second.geometry.explicitSquaredAngle <
                        angularWinner.second.geometry.explicitSquaredAngle - tolerance) {
                        angularWinner = candidate
                    }
                }
                labelsByCandidate.getValue("angular-best") += angularWinner.second.labels.copyOf()

                val chordScore = objectiveScores(texture.directions, vectorBase, chordLabels, clusterCount)
                val radiusFactors = doubleArrayOf(1.0, 0.9975, 0.995, 0.99)
                val radiusRuns = arrayListOf<Pair<String, RadiusLloydResult>>()
                val radiusMeasurements = arrayListOf<Map<String, Any?>>()
                for (index in starts.indices) {
                    val target = chordScore.maximumCapRadiusDeg * radiusFactors[index]
                    val start = System.nanoTime()
                    val result = radiusConstrainedLloyd(
                        texture.directions,
                        vectorBase,
                        starts[index].second,
                        clusterCount,
                        capacity,
                        target,
                        4
                    )
                    radiusMeasurements += linkedMapOf(
                        "start" to starts[index].first,
                        "targetFactor" to radiusFactors[index],
                        "wallTimeNanos" to System.nanoTime() - start,
                        "heapBytesAfter" to usedJvmHeapBytes()
                    )
                    radiusRuns += starts[index].first to result
                }
                var radiusWinnerId = "t06-chord-incumbent"
                var radiusWinnerLabels = chordLabels
                var radiusWinnerScore = chordScore
                for ((id, result) in radiusRuns) if (result.feasible) {
                    val score = objectiveScores(texture.directions, vectorBase, result.labels, clusterCount)
                    if (score.maximumCapRadiusDeg < radiusWinnerScore.maximumCapRadiusDeg - 1e-12 ||
                        abs(score.maximumCapRadiusDeg - radiusWinnerScore.maximumCapRadiusDeg) <= 1e-12 &&
                        score.chord < radiusWinnerScore.chord) {
                        radiusWinnerId = id
                        radiusWinnerLabels = result.labels
                        radiusWinnerScore = score
                    }
                }
                labelsByCandidate.getValue("radius-best") += radiusWinnerLabels.copyOf()

                val candidates = linkedMapOf(
                    "chord-best" to chordLabels,
                    "angular-best" to angularWinner.second.labels,
                    "radius-best" to radiusWinnerLabels
                )
                val scored = candidates.mapValues {
                    objectiveScores(texture.directions, vectorBase, it.value, clusterCount)
                }
                val chordLimit = chordScore.chord * 1.0005
                val tradeoff = scored.entries.filter { it.value.chord <= chordLimit }.minWith { first, second ->
                    var comparison = first.value.p99AngleDeg.compareTo(second.value.p99AngleDeg)
                    if (comparison == 0) comparison = first.value.maximumCapRadiusDeg.compareTo(
                        second.value.maximumCapRadiusDeg
                    )
                    if (comparison == 0) comparison = first.value.angular2.compareTo(second.value.angular2)
                    if (comparison == 0) comparison = first.key.compareTo(second.key)
                    comparison
                }
                labelsByCandidate.getValue("mean-tail") += candidates.getValue(tradeoff.key).copyOf()
                sliceRecords += linkedMapOf(
                    "slice" to slice,
                    "chordStarts" to starts.map { linkedMapOf(
                        "id" to it.first,
                        "partitionSha256" to partitionSha256(it.second, clusterCount),
                        "scores" to objectiveScoresJson(
                            objectiveScores(texture.directions, vectorBase, it.second, clusterCount)
                        )
                    ) },
                    "angularRuns" to angularRuns.map { (id, result) -> linkedMapOf(
                        "start" to id,
                        "status" to result.status,
                        "converged" to result.converged,
                        "iterations" to result.iterations,
                        "scores" to objectiveScoresJson(
                            objectiveScores(texture.directions, vectorBase, result.labels, clusterCount)
                        ),
                        "partitionSha256" to partitionSha256(result.labels, clusterCount),
                        "history" to result.history
                    ) },
                    "angularWinner" to angularWinner.first,
                    "radiusRuns" to radiusRuns.map { (id, result) -> linkedMapOf(
                        "start" to id,
                        "targetRadiusDeg" to result.targetRadiusDeg,
                        "status" to result.status,
                        "feasible" to result.feasible,
                        "iterations" to result.iterations,
                        "maximumRadiusDeg" to result.maximumRadiusDeg,
                        "chordObjective" to result.chordObjective,
                        "partitionSha256" to if (result.feasible) partitionSha256(result.labels, clusterCount) else null,
                        "history" to result.history
                    ) },
                    "radiusWinner" to radiusWinnerId,
                    "meanTailContract" to linkedMapOf(
                        "chordRelativeLimit" to 0.0005,
                        "selection" to "minimum true-angular p99, then minimum cap radius, then angular2; no weighted sum",
                        "chordLimit" to chordLimit,
                        "winner" to tradeoff.key
                    ),
                    "candidateScores" to scored.mapValues { objectiveScoresJson(it.value) }
                )
                sliceMeasurements += linkedMapOf(
                    "slice" to slice,
                    "angularRuns" to angularMeasurements,
                    "radiusRuns" to radiusMeasurements
                )
                println(
                    "objectives K=$clusterCount slice=$slice angular=${angularWinner.first} " +
                        "radius=$radiusWinnerId meanTail=${tradeoff.key}"
                )
            }

            val unbalancedSlices = sliceIds.map { slice ->
                unbalancedSphericalClustering(texture.directions, slice, clusterCount)
            }
            val aggregateScores = linkedMapOf<String, ObjectiveScores>()
            for ((id, candidateSlices) in labelsByCandidate) {
                val canonicalSlices = candidateSlices.map {
                    canonicalPartitionLabels(it, clusterCount)
                }
                val bytes = ByteArray(sliceIds.size * SLICE_SAMPLES)
                for (index in sliceIds.indices) labelsBytes(canonicalSlices[index]).copyInto(
                    bytes, index * SLICE_SAMPLES
                )
                val name = "labels-$id-k$clusterCount.bin"
                writeBytes(outputDirectory.resolve(name), bytes)
                canonicalOutputs[name] = linkedMapOf("bytes" to bytes.size, "sha256" to sha256(bytes))
                val clusterings = canonicalSlices.mapIndexed { index, labels -> SliceClustering(
                    labels,
                    balancedPermutation(
                        texture.directions,
                        sliceIds[index] * SLICE_SAMPLES * CHANNELS,
                        labels,
                        clusterCount
                    ),
                    AlgorithmCounts()
                ) }
                metricCases += linkedMapOf(
                    "id" to "$id-k$clusterCount",
                    "chordCenterMetrics" to caseMetrics(
                        texture, "$id-k$clusterCount", id, "balanced cluster/Hilbert member placement",
                        clusterCount, clusterings, sliceIds
                    ),
                    "trueAngularCenterMetrics" to trueAngularCaseMetrics(
                        texture, sliceIds, canonicalSlices, clusterCount
                    )
                )
                aggregateScores[id] = aggregateObjectiveScores(
                    texture.directions, sliceIds, canonicalSlices, clusterCount
                )
            }
            val unbalancedLabels = unbalancedSlices.map { it.labels }
            val unbalancedScores = aggregateObjectiveScores(
                texture.directions, sliceIds, unbalancedLabels, clusterCount
            )
            metricCases += linkedMapOf(
                "id" to "unbalanced-reference-k$clusterCount",
                "chordCenterMetrics" to caseMetrics(
                    texture,
                    "unbalanced-reference-k$clusterCount",
                    "unbalanced-spherical-lloyd",
                    "unbalanced research reference",
                    clusterCount,
                    unbalancedSlices,
                    sliceIds
                ),
                "trueAngularCenterMetrics" to trueAngularCaseMetrics(
                    texture, sliceIds, unbalancedLabels, clusterCount
                )
            )
            val chordAggregate = aggregateScores.getValue("chord-best")
            val angularAggregate = aggregateScores.getValue("angular-best")
            val radiusAggregate = aggregateScores.getValue("radius-best")
            val meanTailAggregate = aggregateScores.getValue("mean-tail")
            fun samePartitions(first: String, second: String): Boolean =
                labelsByCandidate.getValue(first).indices.all { index ->
                    canonicalPartitionLabels(
                        labelsByCandidate.getValue(first)[index], clusterCount
                    ).contentEquals(canonicalPartitionLabels(
                        labelsByCandidate.getValue(second)[index], clusterCount
                    ))
                }
            comparisonCases += linkedMapOf(
                "clusterCount" to clusterCount,
                "capacity" to capacity,
                "sameBudget" to linkedMapOf(
                    "startsPerObjective" to 4,
                    "maximumAlternationsPerStart" to 4,
                    "assignmentGraph" to "complete sample-center graph",
                    "angularAssignmentQuantization" to quantization,
                    "radiusAssignmentQuantization" to 10_000
                ),
                "candidates" to aggregateScores.mapValues { (id, score) -> linkedMapOf(
                    "scores" to objectiveScoresJson(score),
                    "versusUnbalanced" to linkedMapOf(
                        "squaredChordTotalDelta" to score.chord - unbalancedScores.chord,
                        "trueSquaredAngularTotalDeltaRad2" to score.angular2 - unbalancedScores.angular2,
                        "trueAngularP99DeltaDeg" to score.p99AngleDeg - unbalancedScores.p99AngleDeg,
                        "maximumCapRadiusDeltaDeg" to score.maximumCapRadiusDeg -
                            unbalancedScores.maximumCapRadiusDeg
                    )
                ) },
                "unbalancedReference" to objectiveScoresJson(unbalancedScores),
                "objectiveSelections" to linkedMapOf(
                    "chord" to linkedMapOf(
                        "selected" to "chord-best",
                        "reason" to "lowest preserved squared-chord incumbent"
                    ),
                    "angular2" to linkedMapOf(
                        "selected" to "angular-best",
                        "distinctFromChord" to !samePartitions("angular-best", "chord-best"),
                        "angular2DeltaFromChord" to angularAggregate.angular2 - chordAggregate.angular2,
                        "reason" to if (samePartitions("angular-best", "chord-best"))
                            "no distinct angular partition improved the chord incumbent under the fixed budget"
                        else "lowest true squared-angular incumbent under the fixed budget"
                    ),
                    "radius" to linkedMapOf(
                        "selected" to "radius-best",
                        "distinctFromChord" to !samePartitions("radius-best", "chord-best"),
                        "maximumCapDeltaFromChordDeg" to radiusAggregate.maximumCapRadiusDeg -
                            chordAggregate.maximumCapRadiusDeg,
                        "chordDeltaFromChord" to radiusAggregate.chord - chordAggregate.chord,
                        "p99DeltaFromChordDeg" to radiusAggregate.p99AngleDeg - chordAggregate.p99AngleDeg,
                        "reason" to if (radiusAggregate.maximumCapRadiusDeg <
                            chordAggregate.maximumCapRadiusDeg)
                            "preserved as the hard-radius incumbent; not substituted for chord without its recorded mean/tail costs"
                        else "no distinct hard-radius candidate improved the chord incumbent cap"
                    ),
                    "meanTail" to linkedMapOf(
                        "selected" to "mean-tail",
                        "distinctFromChord" to !samePartitions("mean-tail", "chord-best"),
                        "chordRelativeDelta" to meanTailAggregate.chord / chordAggregate.chord - 1.0,
                        "reason" to if (samePartitions("mean-tail", "chord-best"))
                            "no distinct candidate improved the lexicographic tail tuple inside the 0.05% chord envelope"
                        else "best lexicographic p99/cap/angular2 tuple inside the 0.05% chord envelope"
                    ),
                    "unbalancedReference" to linkedMapOf(
                        "selected" to false,
                        "reason" to "research-only unequal-capacity reference; never replaces a balanced incumbent"
                    )
                ),
                "slices" to sliceRecords
            )
            runCases += linkedMapOf("clusterCount" to clusterCount, "slices" to sliceMeasurements)
        }

        val comparison = linkedMapOf<String, Any?>(
            "schema" to "alpha-piscium/stbn-clustering-objective-comparison/v1",
            "inputTextureSha256" to sha256(inputBytes),
            "inputNeighborhoodDirectory" to normalizePath(inputDirectory),
            "inputNeighborhoodManifestSha256" to sha256(Files.readAllBytes(inputDirectory.resolve("manifest.json"))),
            "sourceSearchDirectory" to normalizePath(searchDirectory),
            "sourceSearchManifestSha256" to sha256(Files.readAllBytes(searchDirectory.resolve("manifest.json"))),
            "sliceIds" to sliceIds,
            "objectives" to objectives.sorted(),
            "angularObjective" to "sum of squared geodesic angles in radians with converged intrinsic Karcher centers",
            "radiusContract" to linkedMapOf(
                "assignment" to "exact lexicographic minimum of hard-radius violations then quantized chord cost for fixed cap centers",
                "centerUpdate" to "deterministic minimum enclosing spherical cap",
                "feasibility" to "every assigned decoded float64 direction is enclosed by its updated cap at the recorded target",
                "notFoundScope" to "not found for the attempted centers and budget; never a global infeasibility claim",
                "encodedRadius" to null
            ),
            "meanTailContract" to "chord loss within 0.05% of chord incumbent, then lexicographic p99/cap/angular2; no scalar blend",
            "cases" to comparisonCases
        )
        val comparisonBytes = canonicalJson(comparison)
        writeBytes(outputDirectory.resolve("objectives.json"), comparisonBytes)
        canonicalOutputs["objectives.json"] = linkedMapOf(
            "bytes" to comparisonBytes.size, "sha256" to sha256(comparisonBytes)
        )
        val metrics = linkedMapOf<String, Any?>(
            "schema" to "alpha-piscium/stbn-clustering-cross-objective-metrics/v1",
            "cases" to metricCases
        )
        val metricsBytes = canonicalJson(metrics)
        writeBytes(outputDirectory.resolve("metrics.json"), metricsBytes)
        canonicalOutputs["metrics.json"] = linkedMapOf(
            "bytes" to metricsBytes.size, "sha256" to sha256(metricsBytes)
        )
        val scriptPath = Paths.get("stbn-clustering.main.kts")
        val manifest = linkedMapOf<String, Any?>(
            "schema" to "alpha-piscium/stbn-clustering-manifest/v1",
            "tool" to linkedMapOf(
                "path" to "scripts/stbn-clustering.main.kts",
                "sha256" to sha256(Files.readAllBytes(scriptPath))
            ),
            "input" to linkedMapOf(
                "texturePath" to normalizePath(inputPath),
                "textureSha256" to sha256(inputBytes),
                "neighborhoodDirectory" to normalizePath(inputDirectory),
                "neighborhoodManifestSha256" to sha256(Files.readAllBytes(inputDirectory.resolve("manifest.json"))),
                "searchDirectory" to normalizePath(searchDirectory),
                "searchManifestSha256" to sha256(Files.readAllBytes(searchDirectory.resolve("manifest.json"))),
                "sliceIds" to sliceIds
            ),
            "algorithm" to linkedMapOf(
                "objectives" to objectives.sorted(),
                "startsPerObjective" to 4,
                "maximumAlternationsPerStart" to 4,
                "angularQuantization" to quantization,
                "radiusQuantization" to 10_000,
                "radiusTargetFactors" to listOf(1.0, 0.9975, 0.995, 0.99),
                "meanTailChordRelativeLimit" to 0.0005
            ),
            "determinism" to linkedMapOf(
                "ties" to "objective tuple, canonical partition SHA-256, source ID",
                "measuredFieldsExcluded" to listOf("wallTimeNanos", "heapBytes")
            ),
            "canonicalOutputs" to canonicalOutputs
        )
        val manifestBytes = canonicalJson(manifest)
        writeBytes(outputDirectory.resolve("manifest.json"), manifestBytes)
        val runBytes = canonicalJson(linkedMapOf(
            "schema" to "alpha-piscium/stbn-clustering-run/v1",
            "objectivesSha256" to sha256(comparisonBytes),
            "metricsSha256" to sha256(metricsBytes),
            "manifestSha256" to sha256(manifestBytes),
            "perStartMeasurements" to runCases,
            "maximumJvmHeapBytes" to Runtime.getRuntime().maxMemory(),
            "peakJvmHeapBytesObserved" to sampler.peakBytes(),
            "wallTimeNanos" to System.nanoTime() - totalStart
        ))
        writeBytes(outputDirectory.resolve("run.json"), runBytes)
        println("objectives.json|${comparisonBytes.size}|${sha256(comparisonBytes)}")
        println("metrics.json|${metricsBytes.size}|${sha256(metricsBytes)}")
        println("manifest.json|${manifestBytes.size}|${sha256(manifestBytes)}")
        println("run.json|${runBytes.size}|${sha256(runBytes)}")
    } finally {
        sampler.close()
    }
}

fun totalChordObjective(
    directions: DoubleArray,
    labelsBySlice: List<IntArray>,
    clusterCount: Int
): Double {
    require(labelsBySlice.size == DEPTH)
    var total = 0.0
    for (slice in 0 until DEPTH) total += computeClusterGeometry(
        directions, slice * SLICE_SAMPLES * CHANNELS, labelsBySlice[slice], clusterCount
    ).explicitSquaredChord
    return total
}

fun finalSearchBudgetCurve(
    searchDirectory: Path,
    directions: DoubleArray,
    clusterCount: Int,
    startCount: Int,
    quantization: Long,
    workers: Int
): Map<String, Any?> {
    require(startCount >= 64) { "T08 requires at least the recorded 64-start base budget" }
    val specs = multistartSpecs(startCount)
    val budgets = (if (startCount == 64) {
        intArrayOf(16, 32, 48, 64)
    } else {
        intArrayOf(16, 32, 64, startCount / 2, startCount)
    }).distinct().sorted().toIntArray()
    val baselineStarts = if (startCount == 64) 32 else startCount / 2
    val baselineIndex = budgets.indexOf(baselineStarts)
    require(baselineIndex >= 0 && baselineIndex < budgets.lastIndex)
    val totals = DoubleArray(budgets.size)
    var improvementsAfterBaseline = 0
    val perSlice = ArrayList<Map<String, Any?>>(DEPTH)
    val objectivesBySlice = parallelMap(workers, (0 until DEPTH).toList()) { slice ->
        val vectorBase = slice * SLICE_SAMPLES * CHANNELS
        val outcomes = specs.map { spec -> readMultistartCheckpoint(
            multistartCheckpointPath(searchDirectory.resolve("checkpoint"), clusterCount, slice, spec.ordinal),
            clusterCount,
            slice,
            spec,
            directions,
            vectorBase,
            SLICE_SAMPLES,
            SLICE_SAMPLES / clusterCount
        ) }
        DoubleArray(budgets.size) { index ->
            outcomes.take(budgets[index]).minWith(::compareOutcomes).objective
        }
    }
    for (slice in 0 until DEPTH) {
        val objectives = objectivesBySlice[slice]
        for (index in budgets.indices) totals[index] += objectives[index]
        if (objectives.last() < objectives[baselineIndex]) improvementsAfterBaseline++
        val objectiveByStarts = linkedMapOf<String, Any?>()
        for (index in budgets.indices) objectiveByStarts[budgets[index].toString()] = objectives[index]
        perSlice += linkedMapOf(
            "slice" to slice,
            "objectiveByStarts" to objectiveByStarts,
            "gainBaselineToFinal" to objectives[baselineIndex] - objectives.last(),
            "gainPreviousToFinal" to objectives[budgets.lastIndex - 1] - objectives.last()
        )
    }
    val gainBaselineToFinal = totals[baselineIndex] - totals.last()
    val relativeGainBaselineToFinal = gainBaselineToFinal / totals[baselineIndex]
    val gainPreviousToFinal = totals[budgets.lastIndex - 1] - totals.last()
    val relativeGainPreviousToFinal = gainPreviousToFinal / totals[budgets.lastIndex - 1]
    val expansionThreshold = 0.0001
    val thresholdSatisfied = relativeGainPreviousToFinal <= expansionThreshold
    return linkedMapOf(
        "baseStartsPerSlice" to startCount,
        "aggregate" to budgets.indices.map { index -> linkedMapOf(
            "starts" to budgets[index], "squaredChordTotal" to totals[index]
        ) },
        "expansionBaselineStarts" to baselineStarts,
        "gainBaselineToFinal" to gainBaselineToFinal,
        "relativeGainBaselineToFinal" to relativeGainBaselineToFinal,
        "gainPreviousToFinal" to gainPreviousToFinal,
        "relativeGainPreviousToFinal" to relativeGainPreviousToFinal,
        "slicesImprovedAfterBaseline" to improvementsAfterBaseline,
        "expansionThresholdRelative" to expansionThreshold,
        "expansionThresholdSatisfied" to thresholdSatisfied,
        "expandedBeyond64" to (startCount > 64),
        "decision" to if (thresholdSatisfied) {
            "stop at $startCount because the final marginal gain does not exceed the predeclared 0.01% threshold"
        } else {
            "budget-limited candidate at $startCount for runtime evaluation; the predeclared 0.01% threshold is not satisfied"
        },
        "perSlice" to perSlice
    )
}

data class ExportedSlice(
    val permutation: IntArray,
    val centers: DoubleArray,
    val caps: DoubleArray,
    val validation: Map<String, Any?>,
    val assignmentMatches: Boolean,
    val fixedCenterGap: Double,
    val compensatedDifference: Double,
    val swapPairs: Long,
    val maximumCapViolation: Double
)

fun exportFinalCandidates(
    inputDirectory: Path,
    pairDirectory: Path,
    searchDirectory: Path,
    outputDirectory: Path,
    trackedOutput: Path,
    startCount: Int,
    quantization: Long,
    workers: Int
) {
    require(Files.isDirectory(inputDirectory) && Files.isDirectory(pairDirectory) &&
        Files.isDirectory(searchDirectory))
    require(startCount >= 64)
    validateQuantization(SLICE_SAMPLES, quantization)
    require(!Files.exists(outputDirectory.resolve("selected-manifest.json"))) {
        "Final export already exists: ${outputDirectory.resolve("selected-manifest.json")}"
    }
    val totalStart = System.nanoTime()
    val sampler = PeakHeapSampler()
    try {
        val inputPath = Paths.get("../shaders/textures/stbn_unitvec3_cosine_2Dx1D_128x128x64.bin")
        val inputBytes = Files.readAllBytes(inputPath)
        val texture = decodeRgb8(inputBytes, WIDTH, HEIGHT, DEPTH)
        val scriptPath = Paths.get("stbn-clustering.main.kts")
        require(Files.isRegularFile(scriptPath)) { "Run this entry point from scripts/" }
        val scriptHash = sha256(Files.readAllBytes(scriptPath))
        val sourceArtifacts = linkedMapOf(
            "searchManifest" to searchDirectory.resolve("manifest.json"),
            "searchHistory" to searchDirectory.resolve("search.json"),
            "pairManifest" to pairDirectory.resolve("manifest.json"),
            "pairHistory" to pairDirectory.resolve("refinement.json"),
            "neighborhoodManifest" to inputDirectory.resolve("manifest.json"),
            "neighborhoodHistory" to inputDirectory.resolve("neighborhoods.json")
        )
        val sourceRunArtifacts = linkedMapOf(
            "searchRun" to searchDirectory.resolve("run.json"),
            "pairRun" to pairDirectory.resolve("run.json"),
            "neighborhoodRun" to inputDirectory.resolve("run.json")
        )
        sourceArtifacts.values.forEach { require(Files.isRegularFile(it)) { "Missing source artifact $it" } }
        sourceRunArtifacts.values.forEach { require(Files.isRegularFile(it)) { "Missing source run artifact $it" } }
        val sourceArtifactReceipts = sourceArtifacts.mapValues { artifactJson(it.value) }
        val candidates = arrayListOf<ExportedCandidate>()

        for (clusterCount in intArrayOf(128, 256)) {
            val capacity = SLICE_SAMPLES / clusterCount
            val candidateId = "k$clusterCount"
            val candidateDirectory = outputDirectory.resolve("candidate-$candidateId")
            require(!Files.exists(candidateDirectory)) { "Candidate output already exists: $candidateDirectory" }
            val searchLabels = readBalancedLabelSlices(
                searchDirectory.resolve("labels-k$clusterCount.bin"), clusterCount
            )
            val pairLabels = readBalancedLabelSlices(
                pairDirectory.resolve("labels-k$clusterCount.bin"), clusterCount
            )
            val labelsBySlice = readBalancedLabelSlices(
                inputDirectory.resolve("labels-k$clusterCount.bin"), clusterCount
            ).map { canonicalPartitionLabels(it, clusterCount) }
            val permutations = ArrayList<IntArray>(DEPTH)
            val centers = DoubleArray(DEPTH * clusterCount * CHANNELS)
            val caps = DoubleArray(DEPTH * clusterCount * 4)
            val sliceValidation = ArrayList<Map<String, Any?>>(DEPTH)
            var maximumCompensatedDifference = 0.0
            var maximumFixedCenterFloat64Gap = 0.0
            var maximumCapAngleViolation = Double.NEGATIVE_INFINITY
            var fullSwapPairs = 0L
            var assignmentPartitionMismatches = 0

            val exportedSlices = parallelMap(workers, (0 until DEPTH).toList()) { slice ->
                val vectorBase = slice * SLICE_SAMPLES * CHANNELS
                val labels = labelsBySlice[slice]
                val geometry = computeClusterGeometry(texture.directions, vectorBase, labels, clusterCount)
                require(geometry.counts.all { it == capacity })
                val permutation = balancedPermutation(
                    texture.directions, vectorBase, labels, clusterCount
                )
                requireLaneGrouping(labels, permutation, clusterCount)

                val incumbentQuantizedCost = quantizedCostForLabels(
                    texture.directions, vectorBase, labels, geometry.centers, quantization
                )
                val assignment = balancedTransport(
                    texture.directions, vectorBase, SLICE_SAMPLES, geometry.centers,
                    clusterCount, capacity, quantization
                )
                requireOptimalTransport(assignment, SLICE_SAMPLES, clusterCount, capacity)
                require(assignment.quantizedCost == incumbentQuantizedCost) {
                    "K=$clusterCount slice=$slice is not a fixed point of the exact quantized full-edge assignment"
                }
                val assignmentMatches = assignment.labels.contentEquals(labels)
                val incumbentFixedLoss = fixedCenterFloat64Loss(
                    texture.directions, vectorBase, labels, geometry.centers
                )
                val assignedFixedLoss = fixedCenterFloat64Loss(
                    texture.directions, vectorBase, assignment.labels, geometry.centers
                )
                val fixedCenterGap = incumbentFixedLoss - assignedFixedLoss
                val quantizationBound = SLICE_SAMPLES / quantization.toDouble()
                require(fixedCenterGap <= quantizationBound + 1e-12)

                val compensated = compensatedFixedCenterLoss(
                    texture.directions, vectorBase, labels, geometry.centers
                )
                val compensatedDifference = abs(compensated - geometry.explicitSquaredChord)

                val swap = completeSwapScan(texture.directions, vectorBase, labels, clusterCount)
                val expectedPairs = (
                    SLICE_SAMPLES.toLong() * SLICE_SAMPLES - clusterCount.toLong() * capacity * capacity
                    ) / 2
                require(swap.pairsEvaluated == expectedPairs)
                require(swap.firstSample < 0 || swap.delta >= -1e-12) {
                    "K=$clusterCount slice=$slice still has one-swap improvement ${swap.delta}"
                }

                val members = buildMembers(labels, geometry.counts)
                val sliceCaps = DoubleArray(clusterCount * 4)
                var sliceMaximumCapViolation = Double.NEGATIVE_INFINITY
                for (cluster in 0 until clusterCount) {
                    val cap = minimumSphericalCap(texture.directions, vectorBase, members[cluster])
                    var maximumMemberAngle = 0.0
                    for (sample in members[cluster]) {
                        val offset = vectorBase + sample * CHANNELS
                        val dot = (texture.directions[offset] * cap.center[0] +
                            texture.directions[offset + 1] * cap.center[1] +
                            texture.directions[offset + 2] * cap.center[2]).coerceIn(-1.0, 1.0)
                        maximumMemberAngle = max(maximumMemberAngle, acos(dot) * RAD_TO_DEG)
                    }
                    val conservativeRadius = Math.nextUp(max(cap.radiusDeg, maximumMemberAngle))
                    val capOffset = cluster * 4
                    sliceCaps[capOffset] = cap.center[0]
                    sliceCaps[capOffset + 1] = cap.center[1]
                    sliceCaps[capOffset + 2] = cap.center[2]
                    sliceCaps[capOffset + 3] = conservativeRadius
                    val violation = maximumMemberAngle - conservativeRadius
                    sliceMaximumCapViolation = max(sliceMaximumCapViolation, violation)
                }
                val validation = linkedMapOf(
                    "slice" to slice,
                    "partitionSha256" to partitionSha256(labels, clusterCount),
                    "squaredChordTotal" to geometry.explicitSquaredChord,
                    "compensatedSquaredChordTotal" to compensated,
                    "compensatedAbsDifference" to compensatedDifference,
                    "fixedCenterAssignmentStatus" to assignment.status,
                    "fixedCenterQuantizedCost" to incumbentQuantizedCost,
                    "fixedCenterCertificateGap" to assignment.certificateGap,
                    "fixedCenterLabelsMatch" to assignmentMatches,
                    "fixedCenterFloat64Gap" to fixedCenterGap,
                    "fixedCenterFloat64Bound" to quantizationBound,
                    "completeOneSwapPairs" to swap.pairsEvaluated,
                    "completeOneSwapBestDelta" to swap.delta,
                    "maximumConservativeCapViolationDeg" to sliceMaximumCapViolation
                )
                println("export verified K=$clusterCount slice=$slice")
                ExportedSlice(
                    permutation, geometry.centers, sliceCaps, validation, assignmentMatches, fixedCenterGap,
                    compensatedDifference, swap.pairsEvaluated, sliceMaximumCapViolation
                )
            }
            for ((slice, exported) in exportedSlices.withIndex()) {
                permutations += exported.permutation
                exported.centers.copyInto(centers, slice * clusterCount * CHANNELS)
                exported.caps.copyInto(caps, slice * clusterCount * 4)
                sliceValidation += exported.validation
                if (!exported.assignmentMatches) assignmentPartitionMismatches++
                maximumFixedCenterFloat64Gap = max(maximumFixedCenterFloat64Gap, exported.fixedCenterGap)
                maximumCompensatedDifference = max(maximumCompensatedDifference, exported.compensatedDifference)
                fullSwapPairs += exported.swapPairs
                maximumCapAngleViolation = max(maximumCapAngleViolation, exported.maximumCapViolation)
            }

            val labelsBytes = ByteArray(TOTAL_SAMPLES)
            for (slice in 0 until DEPTH) labelsBytes(labelsBySlice[slice]).copyInto(
                labelsBytes, slice * SLICE_SAMPLES
            )
            val permutationBytes = unsignedShortsLittleEndian(permutations)
            val centerBytes = doublesLittleEndian(centers)
            val capBytes = doublesLittleEndian(caps)
            val indirectionBytes = indirectionRg8(permutations)
            val decodedIndirection = decodeIndirectionRg8(indirectionBytes)
            require(decodedIndirection.indices.all { decodedIndirection[it].contentEquals(permutations[it]) })

            val labelsName = "labels-original-id-u8.bin"
            val permutationName = "permutation-destination-to-original-id-u16le.bin"
            val centersName = "centers-xyz-f64le.bin"
            val capsName = "minimum-caps-xyz-radius-deg-f64le.bin"
            val indirectionName = "indirection-destination-to-source-rg8.bin"
            writeBytes(candidateDirectory.resolve(labelsName), labelsBytes)
            writeBytes(candidateDirectory.resolve(permutationName), permutationBytes)
            writeBytes(candidateDirectory.resolve(centersName), centerBytes)
            writeBytes(candidateDirectory.resolve(capsName), capBytes)
            writeBytes(candidateDirectory.resolve(indirectionName), indirectionBytes)

            val stageObjectives = linkedMapOf(
                "searchBase" to totalChordObjective(texture.directions, searchLabels, clusterCount),
                "swapPair" to totalChordObjective(texture.directions, pairLabels, clusterCount),
                "variableNeighborhoodFinal" to totalChordObjective(texture.directions, labelsBySlice, clusterCount)
            )
            require(stageObjectives.getValue("swapPair") <= stageObjectives.getValue("searchBase") + 1e-10)
            require(stageObjectives.getValue("variableNeighborhoodFinal") <=
                stageObjectives.getValue("swapPair") + 1e-10)
            val budgetCurve = finalSearchBudgetCurve(
                searchDirectory, texture.directions, clusterCount, startCount, quantization, workers
            )
            val clusterings = labelsBySlice.mapIndexed { slice, labels -> SliceClustering(
                labels, permutations[slice], AlgorithmCounts()
            ) }
            val chordMetrics = caseMetrics(
                texture,
                "final-balanced-chord-$candidateId",
                "${startCount}-start exact-assignment balanced spherical Lloyd plus swap/pair/r2-r4-r8 refinement",
                "center/member Hilbert order packed into 16x16 Morton-lane groups",
                clusterCount,
                clusterings
            )
            val angularMetrics = trueAngularCaseMetrics(
                texture, IntArray(DEPTH) { it }, labelsBySlice, clusterCount
            )
            val metrics = linkedMapOf<String, Any?>(
                "schema" to "alpha-piscium/stbn-clustering-final-metrics/v1",
                "candidate" to candidateId,
                "chordCenterMetrics" to chordMetrics,
                "trueAngularCenterMetrics" to angularMetrics
            )
            val metricsBytes = canonicalJson(metrics)
            writeBytes(candidateDirectory.resolve("metrics.json"), metricsBytes)
            val history = linkedMapOf<String, Any?>(
                "schema" to "alpha-piscium/stbn-clustering-final-history/v1",
                "candidate" to candidateId,
                "stageSquaredChordTotals" to stageObjectives,
                "searchBudgetCurve" to budgetCurve,
                "finalValidation" to linkedMapOf(
                    "assignmentGraph" to "complete sample-center graph",
                    "assignmentQuantization" to quantization,
                    "assignmentPartitionMismatchSlicesAtEqualOptimalCost" to assignmentPartitionMismatches,
                    "maximumFixedCenterFloat64Gap" to maximumFixedCenterFloat64Gap,
                    "maximumCompensatedObjectiveAbsDifference" to maximumCompensatedDifference,
                    "completeOneSwapPairs" to fullSwapPairs,
                    "oneSwapTolerance" to 1e-12,
                    "maximumConservativeCapViolationDeg" to maximumCapAngleViolation,
                    "slices" to sliceValidation
                ),
                "sourceArtifacts" to sourceArtifactReceipts
            )
            val historyBytes = canonicalJson(history)
            writeBytes(candidateDirectory.resolve("history.json"), historyBytes)

            val canonicalArtifacts = linkedMapOf(
                labelsName to artifactJson(candidateDirectory.resolve(labelsName)),
                permutationName to artifactJson(candidateDirectory.resolve(permutationName)),
                centersName to artifactJson(candidateDirectory.resolve(centersName)),
                capsName to artifactJson(candidateDirectory.resolve(capsName)),
                indirectionName to artifactJson(candidateDirectory.resolve(indirectionName)),
                "metrics.json" to artifactJson(candidateDirectory.resolve("metrics.json")),
                "history.json" to artifactJson(candidateDirectory.resolve("history.json"))
            )
            val candidateManifest = linkedMapOf<String, Any?>(
                "schema" to "alpha-piscium/stbn-clustering-final-candidate-manifest/v1",
                "candidate" to candidateId,
                "clusterCount" to clusterCount,
                "capacity" to capacity,
                "input" to linkedMapOf(
                    "path" to normalizePath(inputPath),
                    "sha256" to sha256(inputBytes),
                    "dimensions" to listOf(WIDTH, HEIGHT, DEPTH, CHANNELS),
                    "decode" to "normalize((rgb8 / 255.0) * 2.0 - 1.0) in float64"
                ),
                "tool" to linkedMapOf("path" to "scripts/stbn-clustering.main.kts", "sha256" to scriptHash),
                "algorithm" to linkedMapOf(
                    "mainObjective" to "squared chord loss",
                    "baseStartsPerSlice" to startCount,
                    "initializers" to listOf(
                        "squared-chord-d2", "recursive-balanced-tangent-pca", "structured-hilbert"
                    ),
                    "seedDerivation" to "SplitMix64 from fixed domain, initializer family, and family-local ordinal",
                    "quantization" to quantization,
                    "fullAssignmentEdges" to true,
                    "localRefinement" to listOf(
                        "screened swap", "two-cluster repartition", "ordered r2/r4/r8 variable neighborhoods"
                    ),
                    "neighborhoodAttemptsPerSize" to 4,
                    "completeOneSwapTolerance" to 1e-12,
                    "optimalityScope" to "exact fixed-center quantized assignments and complete one-swap local certificate; no global joint optimum claim"
                ),
                "layout" to linkedMapOf(
                    "forwardMap" to "destination XY to same-slice source XY",
                    "workgroup" to "16x16",
                    "laneOrder" to "8-bit Morton",
                    "groupsPerWorkgroup" to 256 / capacity,
                    "membersPerGroup" to capacity,
                    "clusterPlacement" to "angular-neighbor centers and Hilbert-ordered members"
                ),
                "binaryFormats" to linkedMapOf(
                    labelsName to "one canonical cluster label per original row-major source ID",
                    permutationName to "little-endian uint16 original source ID per destination",
                    centersName to "little-endian float64 normalized chord center XYZ per canonical cluster",
                    capsName to "little-endian float64 minimum-cap center XYZ and outward-rounded radius in degrees",
                    indirectionName to "RG8 unsigned source XY per destination; Z remains unchanged"
                ),
                "sourceArtifacts" to sourceArtifactReceipts,
                "canonicalOutputs" to canonicalArtifacts
            )
            val manifestBytes = canonicalJson(candidateManifest)
            writeBytes(candidateDirectory.resolve("manifest.json"), manifestBytes)
            val relativePrefix = "candidate-$candidateId"
            candidates += ExportedCandidate(
                clusterCount,
                capacity,
                stageObjectives.getValue("variableNeighborhoodFinal"),
                "$relativePrefix/manifest.json",
                sha256(manifestBytes),
                "$relativePrefix/$labelsName",
                sha256(labelsBytes),
                "$relativePrefix/$indirectionName",
                sha256(indirectionBytes),
                indirectionBytes
            )
        }

        val selected = candidates.single { it.clusterCount == 256 }
        val selection = linkedMapOf<String, Any?>(
            "schema" to "alpha-piscium/stbn-clustering-offline-selection/v1",
            "selected" to "k256",
            "provisional" to true,
            "reason" to "both candidates obey the locked 16x16 Morton-lane layout; K256 uses four 64-member lane groups and provides the finer offline directional partition",
            "runtimeWinnerClaim" to false,
            "nextGate" to "target-Iris compile, visual checks, and comparable 720p Vibris A/B",
            "candidates" to candidates.map { linkedMapOf(
                "id" to "k${it.clusterCount}",
                "clusterCount" to it.clusterCount,
                "capacity" to it.capacity,
                "squaredChordTotal" to it.objective,
                "manifestSha256" to it.manifestSha256,
                "indirectionSha256" to it.indirectionSha256
            ) }
        )
        val selectionBytes = canonicalJson(selection)
        writeBytes(outputDirectory.resolve("selection.json"), selectionBytes)
        val selectedManifest = linkedMapOf<String, Any?>(
            "schema" to "alpha-piscium/stbn-clustering-selected-manifest/v1",
            "selectedCandidate" to "k${selected.clusterCount}",
            "selectedClusterCount" to selected.clusterCount,
            "selectedCapacity" to selected.capacity,
            "inputTexturePath" to normalizePath(inputPath),
            "inputTextureSha256" to sha256(inputBytes),
            "labelsFile" to selected.labelsRelativePath,
            "labelsSha256" to selected.labelsSha256,
            "candidateManifestFile" to selected.manifestRelativePath,
            "candidateManifestSha256" to selected.manifestSha256,
            "indirectionFile" to selected.indirectionRelativePath,
            "indirectionSha256" to selected.indirectionSha256,
            "selectionSha256" to sha256(selectionBytes),
            "toolSha256" to scriptHash,
            "regeneration" to "decode input RGB8 in float64, read source-ID labels, rebuild balancedPermutation, encode source XY as RG8"
        )
        val selectedManifestBytes = canonicalJson(selectedManifest)
        writeBytes(outputDirectory.resolve("selected-manifest.json"), selectedManifestBytes)
        if (Files.exists(trackedOutput)) {
            require(Files.readAllBytes(trackedOutput).contentEquals(selected.indirectionBytes)) {
                "Existing tracked output differs from the selected deterministic candidate: $trackedOutput"
            }
        } else {
            writeBytes(trackedOutput, selected.indirectionBytes)
        }
        val runBytes = canonicalJson(linkedMapOf(
            "schema" to "alpha-piscium/stbn-clustering-final-export-run/v1",
            "selectedManifestSha256" to sha256(selectedManifestBytes),
            "sourceRunArtifacts" to sourceRunArtifacts.mapValues { artifactJson(it.value) },
            "workers" to workers,
            "peakJvmHeapBytesObserved" to sampler.peakBytes(),
            "wallTimeNanos" to System.nanoTime() - totalStart
        ))
        writeBytes(outputDirectory.resolve("export-run.json"), runBytes)
        println("selected-manifest.json|${selectedManifestBytes.size}|${sha256(selectedManifestBytes)}")
        println("tracked-indirection|${selected.indirectionBytes.size}|${selected.indirectionSha256}")
    } finally {
        sampler.close()
    }
}

fun jsonStringField(text: String, name: String): String {
    val match = Regex("\\\"${Regex.escape(name)}\\\":\\\"([^\\\"]+)\\\"").find(text)
        ?: error("Manifest lacks string field $name")
    return match.groupValues[1]
}

fun jsonIntField(text: String, name: String): Int {
    val match = Regex("\\\"${Regex.escape(name)}\\\":([0-9]+)").find(text)
        ?: error("Manifest lacks integer field $name")
    return match.groupValues[1].toInt()
}

fun resolveManifestArtifact(manifestPath: Path, relative: String): Path {
    require(!Paths.get(relative).isAbsolute) { "Manifest artifact must be relative: $relative" }
    val root = manifestPath.toAbsolutePath().normalize().parent
    val path = root.resolve(relative).normalize()
    require(path.startsWith(root)) { "Manifest artifact escapes export directory: $relative" }
    return path
}

fun verifyFinalExport(manifestPath: Path, trackedOutput: Path) {
    val manifestBytes = Files.readAllBytes(manifestPath)
    val manifest = manifestBytes.toString(StandardCharsets.UTF_8)
    require(jsonStringField(manifest, "schema") == "alpha-piscium/stbn-clustering-selected-manifest/v1")
    val clusterCount = jsonIntField(manifest, "selectedClusterCount")
    val capacity = jsonIntField(manifest, "selectedCapacity")
    require(clusterCount == 128 || clusterCount == 256)
    require(capacity == SLICE_SAMPLES / clusterCount)
    val scriptPath = Paths.get("stbn-clustering.main.kts")
    require(sha256(Files.readAllBytes(scriptPath)) == jsonStringField(manifest, "toolSha256")) {
        "Current generator does not match the selected manifest"
    }
    val inputPath = Paths.get(jsonStringField(manifest, "inputTexturePath"))
    val inputBytes = Files.readAllBytes(inputPath)
    require(sha256(inputBytes) == jsonStringField(manifest, "inputTextureSha256"))
    val labelsPath = resolveManifestArtifact(manifestPath, jsonStringField(manifest, "labelsFile"))
    val labelsBytes = Files.readAllBytes(labelsPath)
    require(sha256(labelsBytes) == jsonStringField(manifest, "labelsSha256"))
    val labelsBySlice = readBalancedLabelSlices(labelsPath, clusterCount)
    val texture = decodeRgb8(inputBytes, WIDTH, HEIGHT, DEPTH)
    val permutations = labelsBySlice.mapIndexed { slice, labels ->
        val permutation = balancedPermutation(
            texture.directions, slice * SLICE_SAMPLES * CHANNELS, labels, clusterCount
        )
        requireLaneGrouping(labels, permutation, clusterCount)
        permutation
    }
    val regenerated = indirectionRg8(permutations)
    val indirectionPath = resolveManifestArtifact(manifestPath, jsonStringField(manifest, "indirectionFile"))
    val exported = Files.readAllBytes(indirectionPath)
    require(sha256(exported) == jsonStringField(manifest, "indirectionSha256"))
    require(exported.contentEquals(regenerated)) { "Manifest regeneration differs from exported indirection" }
    val tracked = Files.readAllBytes(trackedOutput)
    require(tracked.contentEquals(regenerated)) { "Tracked indirection differs from manifest regeneration" }
    val candidateManifestPath = resolveManifestArtifact(
        manifestPath, jsonStringField(manifest, "candidateManifestFile")
    )
    require(sha256(Files.readAllBytes(candidateManifestPath)) ==
        jsonStringField(manifest, "candidateManifestSha256"))
    val selectionPath = manifestPath.toAbsolutePath().normalize().parent.resolve("selection.json")
    require(sha256(Files.readAllBytes(selectionPath)) == jsonStringField(manifest, "selectionSha256"))
    decodeIndirectionRg8(tracked)
    println("PASS final export verify: candidate=k$clusterCount bytes=${tracked.size} sha256=${sha256(tracked)}")
}

fun runFinalExportSelfTests() {
    var checks = 0
    fun checked(action: () -> Unit) {
        action()
        checks++
    }
    val identity = IntArray(SLICE_SAMPLES) { it }
    val maps = List(DEPTH) { identity }
    val encoded = indirectionRg8(maps)
    checked {
        require(encoded.size == TOTAL_SAMPLES * 2)
        val decoded = decodeIndirectionRg8(encoded)
        require(decoded.all { it.contentEquals(identity) })
    }
    checked {
        val values = listOf(intArrayOf(0, 1, 255, 256, 16383))
        val bytes = unsignedShortsLittleEndian(values)
        require(bytes.contentEquals(byteArrayOf(0, 0, 1, 0, -1, 0, 0, 1, -1, 63)))
    }
    checked {
        val values = doubleArrayOf(0.0, -1.0, PI)
        val bytes = doublesLittleEndian(values)
        require(bytes.size == values.size * 8)
        for (index in values.indices) {
            var bits = 0L
            for (byte in 0 until 8) bits = bits or
                ((bytes[index * 8 + byte].toLong() and 0xFFL) shl (byte * 8))
            require(java.lang.Double.longBitsToDouble(bits) == values[index])
        }
    }
    checked {
        for (clusterCount in intArrayOf(128, 256)) {
            val capacity = SLICE_SAMPLES / clusterCount
            val destinations = destinationsByCluster(clusterCount)
            val labels = IntArray(SLICE_SAMPLES)
            val permutation = IntArray(SLICE_SAMPLES)
            for (cluster in 0 until clusterCount) for (member in 0 until capacity) {
                val source = cluster * capacity + member
                labels[source] = cluster
                permutation[destinations[source]] = source
            }
            requireLaneGrouping(labels, permutation, clusterCount)
        }
    }
    checked {
        val text = canonicalJson(linkedMapOf(
            "schema" to "test", "selectedClusterCount" to 256
        )).toString(StandardCharsets.UTF_8)
        require(jsonStringField(text, "schema") == "test")
        require(jsonIntField(text, "selectedClusterCount") == 256)
    }
    val receipt = canonicalJson(linkedMapOf(
        "schema" to "alpha-piscium/stbn-clustering-final-export-self-test/v1",
        "checks" to checks,
        "rg8RoundTripSlices" to DEPTH,
        "laneLayouts" to listOf("K128/M128", "K256/M64")
    ))
    println("final-export-self-test.json|${receipt.size}|${sha256(receipt)}")
    print(receipt.toString(StandardCharsets.UTF_8))
    println("PASS final-export self-test: $checks checks")
}

fun analyze(inputPath: Path, outputDirectory: Path) {
    val totalStart = System.nanoTime()
    val sampler = PeakHeapSampler()
    try {
        val inputBytes = Files.readAllBytes(inputPath)
        val texture = decodeRgb8(inputBytes, WIDTH, HEIGHT, DEPTH)
        val inputSha256 = sha256(inputBytes)
        val workers = min(DEPTH, max(1, Runtime.getRuntime().availableProcessors()))
        val cases = ArrayList<Map<String, Any?>>()
        val timings = ArrayList<TimedCase>()

        fun runCase(
            id: String,
            algorithm: String,
            placement: String,
            clusterCount: Int,
            build: (Int) -> SliceClustering
        ) {
            val start = System.nanoTime()
            val slices = parallelMap(workers, (0 until DEPTH).toList(), build)
            cases += caseMetrics(texture, id, algorithm, placement, clusterCount, slices)
            timings += TimedCase(id, System.nanoTime() - start)
            println("completed $id")
        }

        for (clusterCount in intArrayOf(128, 256)) {
            runCase(
                "unchanged-k$clusterCount",
                "unchanged",
                "identity source coordinates evaluated in 16x16 Morton-lane groups",
                clusterCount
            ) { identityClustering(clusterCount) }
            runCase(
                "hilbert-octahedral-k$clusterCount",
                "hilbert-octahedral-balanced-order",
                "direction-Hilbert order packed into 16x16 Morton-lane groups",
                clusterCount
            ) { slice -> hilbertClustering(texture.directions, slice, clusterCount) }
            runCase(
                "unbalanced-spherical-lloyd-k$clusterCount",
                "unbalanced-spherical-lloyd",
                "variable-size cluster runs ordered by center/member Hilbert keys; research-only",
                clusterCount
            ) { slice -> unbalancedSphericalClustering(texture.directions, slice, clusterCount) }
        }

        val metrics = linkedMapOf<String, Any?>(
            "schema" to "alpha-piscium/stbn-clustering-metrics/v1",
            "input" to linkedMapOf(
                "path" to normalizePath(inputPath),
                "sha256" to inputSha256,
                "bytes" to inputBytes.size,
                "format" to "RGB8_UNORM",
                "width" to WIDTH,
                "height" to HEIGHT,
                "depth" to DEPTH,
                "channels" to CHANNELS,
                "sampleCount" to TOTAL_SAMPLES,
                "originalId" to "slice-local row-major XY index 0..16383",
                "decode" to "normalize((rgb8 / 255.0) * 2.0 - 1.0) in float64",
                "encodedLength" to linkedMapOf(
                    "min" to texture.stats.encodedLengthMin,
                    "mean" to texture.stats.encodedLengthMean,
                    "max" to texture.stats.encodedLengthMax,
                    "maxAbsDeviationFromOne" to texture.stats.encodedLengthMaxAbsDeviation
                ),
                "decodedZ" to linkedMapOf(
                    "min" to texture.stats.decodedZMin,
                    "max" to texture.stats.decodedZMax,
                    "negativeCount" to 0
                )
            ),
            "metricContract" to linkedMapOf(
                "precision" to "IEEE-754 float64",
                "center" to "normalized arithmetic sum; degenerate sums rejected",
                "squaredChord" to "2 * (1 - dot(sample, center))",
                "eliminatedSquaredChord" to "2 * (clusterSize - length(clusterVectorSum))",
                "objectiveIdentityRelativeTolerance" to OBJECTIVE_RELATIVE_TOLERANCE,
                "pairwiseSquaredChord" to "clusterSize^2 - squaredLength(clusterVectorSum)",
                "angularError" to "acos(clamped dot(sample, center)), degrees",
                "quantile" to "sorted linear interpolation at q*(n-1)",
                "minimumCap" to "deterministically shuffled incremental one/two/three-point support solver",
                "minimumCapContainmentToleranceCos" to CAP_CONTAINMENT_TOLERANCE,
                "horizonStrataZ" to listOf("[0,0.1)", "[0.1,0.25)", "[0.25,0.5)", "[0.5,1]")
            ),
            "cases" to cases
        )
        val metricsBytes = canonicalJson(metrics)
        val metricsHash = sha256(metricsBytes)
        val scriptPath = Paths.get("stbn-clustering.main.kts")
        require(Files.isRegularFile(scriptPath)) { "Run this entry point from scripts/" }
        val scriptHash = sha256(Files.readAllBytes(scriptPath))
        val manifest = linkedMapOf<String, Any?>(
            "schema" to "alpha-piscium/stbn-clustering-manifest/v1",
            "tool" to linkedMapOf(
                "path" to "scripts/stbn-clustering.main.kts",
                "sha256" to scriptHash
            ),
            "input" to linkedMapOf(
                "path" to normalizePath(inputPath),
                "sha256" to inputSha256,
                "bytes" to inputBytes.size,
                "dimensions" to listOf(WIDTH, HEIGHT, DEPTH, CHANNELS)
            ),
            "determinism" to linkedMapOf(
                "seeds" to "none",
                "ties" to "lowest original sample or cluster ID",
                "sliceParallelism" to "independent slices; serialized in ascending Z",
                "measuredFieldsExcluded" to listOf("wallTimeNanos", "peakJvmHeapBytesObserved", "workers")
            ),
            "algorithms" to listOf(
                linkedMapOf("id" to "unchanged", "clusterCounts" to listOf(128, 256)),
                linkedMapOf(
                    "id" to "hilbert-octahedral-balanced-order",
                    "clusterCounts" to listOf(128, 256),
                    "coordinateBits" to 16,
                    "destinationLayout" to "16x16 Morton-lane groups"
                ),
                linkedMapOf(
                    "id" to "unbalanced-spherical-lloyd",
                    "clusterCounts" to listOf(128, 256),
                    "initialization" to "deterministic farthest-first from maximum-Z sample",
                    "maxIterations" to UNBALANCED_MAX_ITERATIONS,
                    "emptyClusterPolicy" to "move lowest-ID maximum-loss sample from a cluster with size > 1"
                )
            ),
            "canonicalOutputs" to linkedMapOf(
                "metrics.json" to linkedMapOf("bytes" to metricsBytes.size, "sha256" to metricsHash)
            )
        )
        val manifestBytes = canonicalJson(manifest)
        val manifestHash = sha256(manifestBytes)
        val run = linkedMapOf<String, Any?>(
            "schema" to "alpha-piscium/stbn-clustering-run/v1",
            "canonicalMetricsSha256" to metricsHash,
            "manifestSha256" to manifestHash,
            "workers" to workers,
            "peakJvmHeapBytesObserved" to sampler.peakBytes(),
            "peakRamMeasurement" to "JVM used heap sampled every 5 ms; excludes native/process overhead",
            "wallTimeNanos" to System.nanoTime() - totalStart,
            "caseWallTimeNanos" to timings.map { linkedMapOf("id" to it.id, "value" to it.wallTimeNanos) }
        )
        val runBytes = canonicalJson(run)
        writeBytes(outputDirectory.resolve("metrics.json"), metricsBytes)
        writeBytes(outputDirectory.resolve("manifest.json"), manifestBytes)
        writeBytes(outputDirectory.resolve("run.json"), runBytes)
        println("metrics.json|${metricsBytes.size}|$metricsHash")
        println("manifest.json|${manifestBytes.size}|$manifestHash")
        println("run.json|${runBytes.size}|${sha256(runBytes)}")
    } finally {
        sampler.close()
    }
}

val arguments = args.toList()
require(arguments.isNotEmpty()) {
    "Usage: kotlin stbn-clustering.main.kts self-test --suite core|transport|initializers|local-refinement|neighborhoods|angular-tail|final-export | " +
        "analyze --input PATH --output DIRECTORY | " +
        "search --method balanced-lloyd --clusters 128,256 --seeds 1 --output DIRECTORY | " +
        "search --method multistart --clusters 128,256 --seeds 64 --elite 16 --slices 0,9,18 --output DIRECTORY | " +
        "refine --input SEARCH_DIRECTORY --starts 64 --methods swap,pair --full-swap-certificate [--workers N] " +
        "--output DIRECTORY | " +
        "refine --input DIRECTORY[,DIRECTORY...] --search-input SEARCH_DIRECTORY --methods neighborhoods " +
        "--starts 64 --sizes 2,4,8 --full-swap-certificate [--workers N] --output DIRECTORY | " +
        "compare-objectives --input NEIGHBORHOOD_DIRECTORY --search-input SEARCH_DIRECTORY " +
        "--objectives chord,angular2,radius --output DIRECTORY | " +
        "export --input NEIGHBORHOOD_DIRECTORY --pair-input PAIR_DIRECTORY --search-input SEARCH_DIRECTORY " +
        "--starts 64 [--workers N] --output DIRECTORY --tracked-output PATH | " +
        "verify --manifest SELECTED_MANIFEST --tracked-output PATH"
}
when (arguments.first()) {
    "benchmark" -> {
        val options = parseOptions(arguments.drop(1), setOf(
            "--slices", "--starts", "--clusters", "--workers", "--repeats", "--max-iterations", "--output"
        ), setOf("--slices", "--starts", "--clusters"))
        fun ints(name: String) = (options[name] ?: error("benchmark requires $name"))
            .split(',').map(String::toInt)
        val slices = ints("--slices")
        val ordinals = ints("--starts")
        val clusters = ints("--clusters")
        val workers = options["--workers"]?.toInt() ?: 1
        val repeats = options["--repeats"]?.toInt() ?: 2
        val iterations = options["--max-iterations"]?.toInt() ?: BALANCED_MAX_ITERATIONS
        require(slices.isNotEmpty() && slices.distinct().size == slices.size && slices.all { it in 0 until DEPTH })
        require(ordinals.isNotEmpty() && ordinals.distinct().size == ordinals.size && ordinals.all { it in 0 until 64 })
        require(clusters.isNotEmpty() && clusters.distinct().size == clusters.size && clusters.all { it == 128 || it == 256 })
        require(workers > 0 && repeats > 0 && iterations > 0)
        val output = Paths.get(options["--output"] ?: error("benchmark requires --output"))
        require(!Files.exists(output)) { "Benchmark output already exists: $output" }
        val input = Files.readAllBytes(Paths.get("../shaders/textures/stbn_unitvec3_cosine_2Dx1D_128x128x64.bin"))
        val texture = decodeRgb8(input, WIDTH, HEIGHT, DEPTH)
        val specs = multistartSpecs(64)
        val work = slices.flatMap { slice -> ordinals.map { StartWork(slice, specs[it]) } }
        val timings = java.util.concurrent.ConcurrentHashMap<String, Map<String, Any?>>()
        val records = arrayListOf<Map<String, Any?>>()
        val runs = arrayListOf<Map<String, Any?>>()
        val threadBean = java.lang.management.ManagementFactory.getThreadMXBean()
        // Compile/JIT the same full-size assignment path before recording repetitions.
        for (k in clusters) {
            val base = slices.first() * SLICE_SAMPLES * CHANNELS
            val initial = initializationForSpec(texture.directions, base, SLICE_SAMPLES, k, specs[ordinals.first()])
            val warmup = balancedTransport(texture.directions, base, SLICE_SAMPLES, initial.centers,
                k, SLICE_SAMPLES / k, DEFAULT_QUANTIZATION)
            requireOptimalTransport(warmup, SLICE_SAMPLES, k, SLICE_SAMPLES / k)
        }
        for (repeat in 0 until repeats) for (k in clusters) {
            val start = System.nanoTime()
            runStartWork(work, workers, { item ->
                val cpuStart = threadBean.currentThreadCpuTime
                val taskStart = System.nanoTime()
                val outcome = evaluateMultistart(texture.directions, item.slice * SLICE_SAMPLES * CHANNELS,
                    SLICE_SAMPLES, k, SLICE_SAMPLES / k, DEFAULT_QUANTIZATION, iterations, item.spec)
                timings["${item.slice}:${item.spec.ordinal}"] = linkedMapOf(
                    "wallTimeNanos" to System.nanoTime() - taskStart,
                    "threadCpuTimeNanos" to threadBean.currentThreadCpuTime - cpuStart
                )
                outcome
            }, { item, outcome ->
                val record = linkedMapOf<String, Any?>(
                    "repeat" to repeat, "clusterCount" to k, "slice" to item.slice,
                    "result" to multistartOutcomeJson(outcome),
                    "labelsSha256" to labelsSha256(outcome.labels),
                    "timing" to timings.getValue("${item.slice}:${item.spec.ordinal}")
                )
                records += record
                println("benchmark K=$k repeat=$repeat slice=${item.slice} start=${item.spec.ordinal} " +
                    "iterations=${outcome.iterationCount} objective=${outcome.objective} " +
                    "timing=${record["timing"]}")
            })
            runs += linkedMapOf("repeat" to repeat, "clusterCount" to k, "wallTimeNanos" to System.nanoTime() - start)
        }
        writeBytes(output, canonicalJson(linkedMapOf(
            "inputSha256" to sha256(input), "slices" to slices, "starts" to ordinals,
            "workers" to workers, "repeats" to repeats, "maxIterations" to iterations,
            "quantization" to DEFAULT_QUANTIZATION, "fullAssignmentEdges" to true,
            "javaVersion" to System.getProperty("java.runtime.version"),
            "runs" to runs, "records" to records
        )))
        println("benchmark receipt: $output")
    }
    "self-test" -> {
        val options = parseOptions(arguments.drop(1), setOf("--suite"))
        when (options["--suite"]) {
            "core" -> runCoreSelfTests()
            "transport" -> runTransportSelfTests()
            "initializers" -> runInitializerSelfTests()
            "local-refinement" -> runLocalRefinementSelfTests()
            "neighborhoods" -> runNeighborhoodSelfTests()
            "angular-tail" -> runAngularTailSelfTests()
            "final-export" -> runFinalExportSelfTests()
            else -> error(
                "self-test requires --suite core, transport, initializers, local-refinement, neighborhoods, angular-tail, or final-export"
            )
        }
    }
    "analyze" -> {
        val options = parseOptions(arguments.drop(1), setOf("--input", "--output"))
        val input = options["--input"] ?: error("analyze requires --input PATH")
        val output = options["--output"] ?: error("analyze requires --output DIRECTORY")
        analyze(Paths.get(input), Paths.get(output))
    }
    "compare-objectives" -> {
        val options = parseOptions(arguments.drop(1), setOf(
            "--input", "--search-input", "--output", "--objectives", "--slices", "--q"
        ), setOf("--objectives", "--slices"))
        val input = options["--input"] ?: error("compare-objectives requires --input DIRECTORY")
        val searchInput = options["--search-input"]
            ?: error("compare-objectives requires --search-input DIRECTORY")
        val output = options["--output"] ?: error("compare-objectives requires --output DIRECTORY")
        val objectives = (options["--objectives"]
            ?: error("compare-objectives requires --objectives")).split(',').toSet()
        val slices = (options["--slices"] ?: "0,9,18").split(',').map(String::toInt).sorted().toIntArray()
        val quantization = options["--q"]?.toLong() ?: DEFAULT_QUANTIZATION
        compareObjectives(
            Paths.get(input), Paths.get(searchInput), Paths.get(output), objectives, slices, quantization
        )
    }
    "export" -> {
        val options = parseOptions(arguments.drop(1), setOf(
            "--input", "--pair-input", "--search-input", "--output", "--tracked-output", "--starts", "--q", "--workers"
        ))
        val input = options["--input"] ?: error("export requires --input NEIGHBORHOOD_DIRECTORY")
        val pairInput = options["--pair-input"] ?: error("export requires --pair-input PAIR_DIRECTORY")
        val searchInput = options["--search-input"] ?: error("export requires --search-input SEARCH_DIRECTORY")
        val output = options["--output"] ?: error("export requires --output DIRECTORY")
        val trackedOutput = options["--tracked-output"] ?: error("export requires --tracked-output PATH")
        val starts = options["--starts"]?.toInt() ?: error("export requires --starts 64")
        val quantization = options["--q"]?.toLong() ?: DEFAULT_QUANTIZATION
        val workers = options["--workers"]?.toInt() ?: defaultWorkers()
        exportFinalCandidates(
            Paths.get(input), Paths.get(pairInput), Paths.get(searchInput), Paths.get(output),
            Paths.get(trackedOutput), starts, quantization, workers
        )
    }
    "verify" -> {
        val options = parseOptions(arguments.drop(1), setOf("--manifest", "--tracked-output"))
        val manifest = options["--manifest"] ?: error("verify requires --manifest SELECTED_MANIFEST")
        val trackedOutput = options["--tracked-output"] ?: error("verify requires --tracked-output PATH")
        verifyFinalExport(Paths.get(manifest), Paths.get(trackedOutput))
    }
    "search" -> {
        val options = parseOptions(
            arguments.drop(1),
            setOf(
                "--method",
                "--clusters",
                "--seeds",
                "--elite",
                "--input",
                "--output",
                "--q",
                "--max-iterations",
                "--stop-after-starts",
                "--slices",
                "--workers"
            ),
            setOf("--clusters", "--seeds", "--slices")
        )
        val clusterCounts = (options["--clusters"] ?: error("search requires --clusters"))
            .split(',')
            .map { it.toInt() }
            .toIntArray()
        val seeds = (options["--seeds"] ?: error("search requires --seeds"))
            .split(',')
            .map { it.toInt() }
            .toIntArray()
        val input = options["--input"]
            ?: "../shaders/textures/stbn_unitvec3_cosine_2Dx1D_128x128x64.bin"
        val output = options["--output"] ?: error("search requires --output DIRECTORY")
        val quantization = options["--q"]?.toLong() ?: DEFAULT_QUANTIZATION
        val maxIterations = options["--max-iterations"]?.toInt() ?: BALANCED_MAX_ITERATIONS
        when (options["--method"]) {
            "balanced-lloyd" -> {
                require(options["--elite"] == null && options["--stop-after-starts"] == null &&
                    options["--slices"] == null && options["--workers"] == null) {
                    "balanced-lloyd does not accept multistart options"
                }
                require(seeds.contentEquals(intArrayOf(1))) {
                    "Balanced Lloyd requires the single deterministic seed --seeds 1"
                }
                searchBalancedLloyd(Paths.get(input), Paths.get(output), clusterCounts, quantization, maxIterations)
            }
            "multistart" -> {
                require(seeds.size == 1) { "Multistart requires one start count, for example --seeds 64" }
                val elite = options["--elite"]?.toInt() ?: 16
                val stopAfterStarts = options["--stop-after-starts"]?.toInt()
                val slices = (options["--slices"] ?: error("multistart requires explicit --slices")).split(',')
                    .map(String::toInt).sorted().toIntArray()
                val workers = options["--workers"]?.toInt()
                searchMultistart(
                    Paths.get(input),
                    Paths.get(output),
                    clusterCounts,
                    seeds[0],
                    elite,
                    quantization,
                    maxIterations,
                    stopAfterStarts,
                    slices,
                    workers
                )
                Unit
            }
            else -> error("search requires --method balanced-lloyd or multistart")
        }
    }
    "refine" -> {
        val expanded = arguments.drop(1).flatMap {
            if (it == "--full-swap-certificate") listOf(it, "true") else listOf(it)
        }
        val options = parseOptions(expanded, setOf(
            "--input", "--search-input", "--output", "--methods", "--sizes",
            "--full-swap-certificate", "--slices", "--elite", "--starts", "--q", "--workers"
        ), setOf("--input", "--methods", "--sizes", "--slices"))
        val input = options["--input"] ?: error("refine requires --input SEARCH_DIRECTORY")
        val output = options["--output"] ?: error("refine requires --output DIRECTORY")
        val methods = (options["--methods"] ?: error("refine requires --methods")).split(',').toSet()
        val fullCertificate = options["--full-swap-certificate"]?.toBooleanStrictOrNull()
            ?: error("refine requires --full-swap-certificate")
        val slices = (options["--slices"] ?: "0,9,18").split(',').map(String::toInt).sorted().toIntArray()
        val elite = options["--elite"]?.toInt() ?: 2
        val starts = options["--starts"]?.toInt() ?: 64
        val quantization = options["--q"]?.toLong() ?: DEFAULT_QUANTIZATION
        val workers = options["--workers"]?.toInt() ?: defaultWorkers()
        if (methods == setOf("neighborhoods")) {
            require(options["--elite"] == null) { "Neighborhood refinement does not accept --elite" }
            val searchInput = options["--search-input"]
                ?: error("neighborhood refinement requires --search-input SEARCH_DIRECTORY")
            val sizes = (options["--sizes"] ?: error("neighborhood refinement requires --sizes"))
                .split(',').map(String::toInt).toIntArray()
            refineNeighborhoods(
                input.split(',').map { Paths.get(it) }, Paths.get(searchInput), Paths.get(output),
                sizes, slices, quantization, fullCertificate, starts, workers
            )
        } else {
            require(options["--search-input"] == null && options["--sizes"] == null) {
                "Swap/pair refinement does not accept neighborhood options"
            }
            refineMultistart(
                Paths.get(input), Paths.get(output), methods, fullCertificate, slices, elite, quantization, starts,
                workers
            )
        }
    }
    else -> error("Unknown command ${arguments.first()}")
}
