import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.math.sqrt

data class V3(val x: Double, val y: Double, val z: Double) {
    operator fun minus(v: V3) = V3(x - v.x, y - v.y, z - v.z)
    fun cross(v: V3) = V3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x)
    fun component(axis: Int) = when (axis) { 0 -> x; 1 -> y; else -> z }
}
fun expandDiscreteRotation(marker: Int): Int {
    val rotation = marker
    val x = rotation and 7
    val y = rotation shr 3 and 7
    val xAxis = x and 3
    val yAxis = y and 3
    val zAxis = 3 - xAxis - yAxis
    val z = zAxis or (rotation shr 4 and 4)
    return x or (y shl 3) or (z shl 6)
}
fun rotateDiscrete(marker: Int, value: V3): V3 {
    val rotation = expandDiscreteRotation(marker)
    fun component(transform: Int) = value.component(transform and 3) * if (transform and 4 == 0) 1.0 else -1.0
    return V3(component(rotation), component(rotation shr 3), component(rotation shr 6))
}
fun unrotateDiscrete(marker: Int, value: V3): V3 {
    var rotation = expandDiscreteRotation(marker)
    val result = DoubleArray(3)
    listOf(value.x, value.y, value.z).forEach { component ->
        result[rotation and 3] = component * if (rotation and 4 == 0) 1.0 else -1.0
        rotation = rotation shr 3
    }
    return V3(result[0], result[1], result[2])
}
fun rayBox(o: V3, d: V3, lo: V3, hi: V3): Pair<Double, V3>? {
    val os = doubleArrayOf(o.x, o.y, o.z)
    val ds = doubleArrayOf(d.x, d.y, d.z)
    val ls = doubleArrayOf(lo.x, lo.y, lo.z)
    val hs = doubleArrayOf(hi.x, hi.y, hi.z)
    var enter = Double.NEGATIVE_INFINITY
    var exit = Double.POSITIVE_INFINITY
    var normal = V3(0.0, 0.0, 0.0)
    for (axis in 0..2) {
        if (abs(ds[axis]) < 1e-12) {
            if (os[axis] !in ls[axis]..hs[axis]) return null
            continue
        }
        val a = (ls[axis] - os[axis]) / ds[axis]
        val b = (hs[axis] - os[axis]) / ds[axis]
        val near = minOf(a, b)
        if (near > enter) {
            enter = near
            val s = if (ds[axis] > 0) -1.0 else 1.0
            normal = when (axis) { 0 -> V3(s,0.0,0.0); 1 -> V3(0.0,s,0.0); else -> V3(0.0,0.0,s) }
        }
        exit = minOf(exit, maxOf(a, b))
    }
    return if (enter <= exit && exit >= 0) maxOf(enter, 0.0) to normal else null
}
val root = Path.of("..").toAbsolutePath().normalize()
fun read(path: String) = Files.readString(root.resolve(path))
val failures = mutableListOf<String>()
fun expect(text: String, token: String, label: String) { if (!text.contains(token)) failures += label + ": missing " + token }

val mappings = read("shaders/block.properties")
val models = read("shaders/techniques/voxel/BlockModels.glsl")
val hardcoded = read("shaders/util/HardcodedPBR.glsl")
val shadow = read("shaders/pass/geometry/ShadowPass.vert.glsl")
val trace = read("shaders/techniques/voxel/VoxelTrace.glsl")
val surface = read("shaders/techniques/voxel/SurfaceData.glsl")
val texcoords = read("shaders/techniques/voxel/VoxelFaceTexcoords.glsl")
val clear = read("shaders/pass/setup/ClearVoxelFaceTexcoords.comp.glsl")
val properties = read("scripts/shaders.properties")
val textures = read("shaders/base/Textures.glsl")

val states = listOf(
    "oak_slab:type=bottom", "oak_slab:type=top",
    "oak_stairs:facing=east:half=bottom:shape=inner_left",
    "oak_door:facing=east:half=lower:hinge=left:open=true",
    "oak_fence:east=false:north=false:south=false:west=false"
)
fun id(state: String) = Regex("(?m)^block\\.(\\d+) = " + Regex.escape(state) + "$")
    .find(mappings)?.groupValues?.get(1)?.toInt()
val ids = states.map { state -> id(state) ?: run { failures += "mapping missing: " + state; -1 } }
val allIds = Regex("(?m)^block\\.(\\d+) =").findAll(mappings).map { it.groupValues[1].toInt() }.toList()
val maxId = allIds.maxOrNull()!!
if (maxId >= 16384) failures += "material ID >= 16384"
val lutBytes = Files.readAllBytes(root.resolve("shaders/textures/pbr_lut_0.bin"))
val modelLutBytes = Files.readAllBytes(root.resolve("shaders/textures/pbr_lut_1.bin"))
val aabbPath = root.resolve("shaders/textures/block_model_aabbs.bin")
val aabbBytes = if (Files.exists(aabbPath)) {
    Files.readAllBytes(aabbPath)
} else {
    failures += "block-model AABB texture missing: shaders/textures/block_model_aabbs.bin"
    ByteArray(0)
}
if (lutBytes.size % 4 != 0) failures += "PBR LUT byte size not divisible by 4"
val lutWidth = lutBytes.size / 4L
if (lutWidth != maxId + 1L) failures += "PBR LUT width $lutWidth != max material ID + 1 (${maxId + 1})"
if (modelLutBytes.size != lutBytes.size) failures += "model LUT width differs from packed PBR LUT"
if (aabbBytes.size % 12 != 0) failures += "block-model AABB texture size is not 12 bytes per AABB"
val aabbCount = aabbBytes.size / 12
val aabbTexelWidth = aabbBytes.size / 4
val discreteAABBCount = (aabbBytes.indices step 12).count { offset ->
    aabbBytes[offset + 7].toUByte().toInt() != 255
}
if ((aabbBytes.indices step 12).any { offset ->
        val marker = aabbBytes[offset + 7].toUByte().toInt()
        marker != 255 && (marker > 118 || run {
            val rotation = marker
            (rotation and 3) == (rotation shr 3 and 3)
        })
    }) failures += "discrete AABB marker is invalid"
fun lutUInt(bytes: ByteArray, materialId: Int) = ByteBuffer.wrap(bytes)
    .order(ByteOrder.LITTLE_ENDIAN).getInt(materialId * 4).toUInt()
val modelRotationBits = 9
val modelRotationMask = 0x1FFu
fun aabbOffset(modelData: UInt) = (modelData shr modelRotationBits) and 0x7FFFu
fun aabbCount(modelData: UInt) = modelData shr 24
val selectedModelData = ids.filter { it >= 0 }.map { lutUInt(modelLutBytes, it) }
if (selectedModelData.any { it == 0u }) failures += "selected non-full state has model data zero"
if (models.contains("switch")) failures += "generated model dispatch still uses switch"
val lutModelData = (modelLutBytes.indices step 4).map { lutUInt(modelLutBytes, it / 4) }.filter { it != 0u }
val lutAABBMetadata = lutModelData.map { it and modelRotationMask.inv() }
if (lutModelData.size <= lutAABBMetadata.distinct().size) failures += "model LUT does not deduplicate shared AABB geometry"
if (lutModelData.distinct().size <= lutAABBMetadata.distinct().size) failures += "model LUT does not preserve rotated instances"
if (lutModelData.any { aabbCount(it) == 0u || aabbOffset(it) + aabbCount(it) > aabbCount.toUInt() }) {
    failures += "model LUT contains invalid AABB offset or count"
}
if (lutModelData.any { it and modelRotationMask == 0u }) {
    failures += "model LUT contains invalid packed rotation"
}
if (selectedModelData.any { aabbCount(it) == 0u || aabbOffset(it) + aabbCount(it) > aabbCount.toUInt() }) {
    failures += "representative slab/stair/door/fence metadata is invalid"
}
expect(models, "Generated by Shadesmith from Minecraft 26.2", "provenance")
expect(models, "bool voxel_intersectBlockModel(", "model API")
expect(models, "uint aabbOffset = (modelData >> 9u) & 0x7FFFu;", "packed AABB offset decode")
expect(models, "uint aabbCount = modelData >> 24u;", "packed AABB count decode")
expect(models, "for (uint i = 0u; i < aabbCount; ++i)", "AABB loop")
expect(models, "_voxel_rotateBlockModelVector", "packed model rotation")
expect(models, "_voxel_unrotateBlockModelVector", "model normal inverse rotation")
expect(models, "uint discreteRotation = uint(originData.w * 255.0 + 0.5);", "discrete AABB fast path")
expect(models, "_voxel_unrotateBlockModelVector(discreteRotation, localNormal)", "discrete normal rotation")
if (discreteAABBCount <= aabbCount / 2) failures += "discrete AABB encoding does not cover the majority of models"
val modelFunction = models.substringAfter("bool voxel_intersectBlockModel(")
if (Regex("texelFetch\\(usam_blockModelAABBs").findAll(models).count() != 3) failures += "generated model code does not fetch exactly three AABB texels"
if (models.contains("_voxel_intersectBlockModelQuad") || models.contains("modelID")) failures += "generated model code still hardcodes quad/model ID dispatch"
if (Regex("(?m)^    bool hit = false;$").findAll(modelFunction).count() != 1) {
    failures += "model function does not have exactly one hit accumulator"
}
if (Regex("_voxel_unrotateBlockModelVector").findAll(modelFunction).count() != 1) {
    failures += "model function duplicates final normal rotation"
}
if (modelFunction.lineSequence().count { it.trimStart().startsWith("return ") } != 1) {
    failures += "model function contains early or duplicate returns"
}
if (Regex("(?m)\\b(?:const\\s+)?(?:vec[234]|u?int|float|bool)\\s+\\w+\\s*\\[").containsMatchIn(models)) failures += "generated local/const array"

expect(hardcoded, "uint blockModelMetadata;", "PBR model metadata member")
expect(hardcoded, "texelFetch(usam_pbrLUT1", "PBR model LUT fetch")
expect(hardcoded, "pbr.blockModelMetadata =", "PBR model metadata decode")
expect(hardcoded, "pbr.roughness = unpackU8(bitfieldExtract(rawData.x, 16, 8));", "PBR roughness bit 16")
if (hardcoded.contains("pbr.roughness = unpackU8(bitfieldExtract(rawData.x, 24, 8));")) failures += "PBR roughness still decodes obsolete bit 24"
expect(shadow, "hardcoded.isFullCube || hardcoded.blockModelMetadata != 0u", "voxelization gate")
if (shadow.contains("hardcoded.emissive > 0.0")) failures += "obsolete emissive voxelization"
expect(shadow, "materialID != MATERIAL_ID_WATER", "water exclusion")
expect(trace, "#include \"/util/HardcodedPBR.glsl\"", "trace PBR include")
expect(trace, "#include \"/techniques/voxel/BlockModels.glsl\"", "trace model include")
expect(trace, "voxel_intersectBlockModel(", "trace model call")
expect(trace, "hardcoded.isFullCube", "full-cube fast path")
expect(trace, "hardcoded.blockModelMetadata != 0u", "unsupported exclusion")
listOf("ray.lastT = lastT;", "ray.level = level;", "ray.fullMorton = fullMorton;").forEach { expect(trace, it, "resumable state") }
val emptyStep = trace.indexOf("// ---- Empty child")
val blockPosUpdate = trace.indexOf("blockPos = exitBlockPos;", emptyStep)
val finalAxisUpdate = trace.indexOf("blockPos.x = clamp", blockPosUpdate)
val updatedBounds = trace.indexOf("if (uint(blockPos.x | blockPos.y | blockPos.z) >= uint(GRID_BLOCKS))", finalAxisUpdate)
val updatedPack = trace.indexOf("fullMorton = _voxel_packBlockPos(blockPos);", blockPosUpdate)
if (emptyStep < 0 || blockPosUpdate < 0 || finalAxisUpdate < 0 || updatedBounds < 0 || updatedPack < 0 ||
    !(blockPosUpdate < finalAxisUpdate && finalAxisUpdate < updatedBounds && updatedBounds < updatedPack)) {
    failures += "updated blockPos bounds check must precede pack after all axis updates"
}
val fullCubeStart = trace.indexOf("if (hardcoded.isFullCube)")
val modelStart = trace.indexOf("if (hardcoded.blockModelMetadata", fullCubeStart)
if (fullCubeStart < 0 || modelStart < 0 || !trace.substring(fullCubeStart, modelStart).contains("ray.level = 0;")) {
    failures += "full-cube hit does not clear ray.level"
}
val modelHitStart = trace.indexOf("if (voxel_intersectBlockModel(", modelStart)
val modelMissStart = trace.indexOf("isHit = false;", modelHitStart)
if (modelHitStart < 0 || modelMissStart < 0 || !trace.substring(modelHitStart, modelMissStart).contains("ray.level = 0;")) {
    failures += "block-model hit does not clear ray.level"
}
val writebackStart = trace.indexOf("// Write back state")
val resultStart = trace.indexOf("VoxelHit result;", writebackStart)
if (writebackStart < 0 || resultStart < 0) {
    failures += "trace writeback section missing"
} else {
    val writeback = trace.substring(writebackStart, resultStart)
    val levelWrite = writeback.indexOf("ray.level = level;")
    val activeGuard = writeback.indexOf("if (level != 0)")
    val mortonWrite = writeback.indexOf("ray.fullMorton = fullMorton;")
    if (levelWrite < 0 || activeGuard < levelWrite || mortonWrite < activeGuard) failures += "terminal writeback must propagate level and guard fullMorton"
}
expect(surface, "gData.geomNormal = hit.normal;", "exact normal")
expect(texcoords, "#define VOXEL_FACE_TEXCOORD_MATERIALS 16384", "texcoord capacity")
expect(clear, "const ivec3 workGroups = ivec3(768, 1, 1);", "clear size")
expect(properties, "customTexture.usam_pbrLUT0=textures/pbr_lut_0.bin TEXTURE_1D R32UI $lutWidth ", "LUT width")
expect(properties, "customTexture.usam_pbrLUT1=textures/pbr_lut_1.bin TEXTURE_1D R32UI $lutWidth ", "model LUT width")
expect(properties, "customTexture.usam_blockModelAABBs=textures/block_model_aabbs.bin TEXTURE_1D RGBA8 $aabbTexelWidth RGBA UNSIGNED_BYTE", "block-model AABB texture")
expect(textures, "uniform sampler1D usam_blockModelAABBs;", "block-model AABB sampler")
expect(properties, "bufferObject.9=1572864", "SSBO size")

val down = rayBox(V3(.25,1.0,.25), V3(0.0,-1.0,0.0), V3(0.0,0.0,0.0), V3(1.0,.5,1.0))
val up = rayBox(V3(.25,0.0,.25), V3(0.0,1.0,0.0), V3(0.0,.5,0.0), V3(1.0,1.0,1.0))
if (down != (0.5 to V3(0.0,1.0,0.0))) failures += "bottom slab numeric hit"
if (up != (0.5 to V3(0.0,-1.0,0.0))) failures += "top slab numeric hit"
val tieMarker = 2
val tieHit = rayBox(
    rotateDiscrete(tieMarker, V3(0.0, 2.0, 2.0)),
    rotateDiscrete(tieMarker, V3(0.0, -1.0, -1.0)),
    V3(-0.4375, -0.03125, -0.4375),
    V3(0.4375, 0.03125, 0.4375)
)
if (tieHit?.first != 1.5625 || tieHit.second.let { unrotateDiscrete(tieMarker, it) } != V3(0.0, 0.0, 1.0)) {
    failures += "discrete AABB axis-tie normal"
}
val n = (V3(1.0,0.0,1.0) - V3(0.0,0.0,0.0)).cross(V3(0.0,1.0,0.0) - V3(0.0,0.0,0.0))
val len = sqrt(n.x*n.x + n.y*n.y + n.z*n.z)
if (abs(abs(n.x/len) - sqrt(.5)) > 1e-9 || abs(abs(n.z/len) - sqrt(.5)) > 1e-9) failures += "rotated normal"
val first = rayBox(V3(-1.0,.75,.5), V3(1.0,0.0,0.0), V3(0.0,0.0,0.0), V3(1.0,.5,1.0))
val later = rayBox(V3(-1.0,.75,.5), V3(1.0,0.0,0.0), V3(2.0,0.0,0.0), V3(3.0,1.0,1.0))
if (first != null || later?.first != 3.0) failures += "model-miss continuation"
if (rayBox(V3(-1.0,.25,.25), V3(1.0,0.0,0.0), V3(0.0,0.0,0.0), V3(1.0,1.0,1.0))?.first != 1.0) failures += "full-cube entry"
fun voxelized(id: Int, full: Boolean, model: Boolean) = id != 0 && id != 1 && (full || model)
if (voxelized(1,false,true) || voxelized(0,true,false) || voxelized(2,false,false)) failures += "water/zero/unsupported exclusion"
data class DdaStep(val level: Int, val packedBlockPos: Int?)
fun packBlockPos(blockPos: Int, gridBlocks: Int): Int {
    check(blockPos in 0 until gridBlocks)
    return blockPos
}
fun stepEmptyCell(blockPos: Int, stepDir: Int, gridBlocks: Int): DdaStep {
    val updatedBlockPos = blockPos + stepDir
    if (updatedBlockPos !in 0 until gridBlocks) return DdaStep(0, null)
    return DdaStep(1, packBlockPos(updatedBlockPos, gridBlocks))
}
val GRID_BLOCKS = 64
val positiveEdge = stepEmptyCell(GRID_BLOCKS - 1, 1, GRID_BLOCKS)
val negativeEdge = stepEmptyCell(0, -1, GRID_BLOCKS)
val interior = stepEmptyCell(17, 1, GRID_BLOCKS)
if (positiveEdge.level != 0 || positiveEdge.packedBlockPos != null) failures += "positive edge must terminate without packing"
if (negativeEdge.level != 0 || negativeEdge.packedBlockPos != null) failures += "negative edge must terminate without packing"
if (interior.level == 0 || interior.packedBlockPos != 18) failures += "interior step must remain active and pack"

check(failures.isEmpty()) { "Voxel trace contract failed:\n" + failures.joinToString("\n") { "- " + it } }
println("Voxel trace contract PASS: " + allIds.size + " mappings, " + aabbCount + " AABBs (" + discreteAABBCount + " discrete), max " + lutModelData.maxOf(::aabbCount) + "/model, " + aabbTexelWidth + " texels, Minecraft 26.2")
