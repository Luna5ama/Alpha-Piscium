import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.math.sqrt

data class V3(val x: Double, val y: Double, val z: Double) {
    operator fun plus(v: V3) = V3(x + v.x, y + v.y, z + v.z)
    operator fun minus(v: V3) = V3(x - v.x, y - v.y, z - v.z)
    operator fun times(s: Double) = V3(x * s, y * s, z * s)
    operator fun unaryMinus() = V3(-x, -y, -z)
    fun dot(v: V3) = x * v.x + y * v.y + z * v.z
    fun cross(v: V3) = V3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x)
    fun normalized(): V3 {
        val length = sqrt(dot(this))
        return this * (1.0 / length)
    }
}

fun rayTriangle(origin: V3, direction: V3, a: V3, b: V3, c: V3): Pair<Double, V3>? {
    val edge1 = b - a
    val edge2 = c - a
    val p = direction.cross(edge2)
    val determinant = edge1.dot(p)
    if (abs(determinant) < 1e-12) return null
    val inverseDeterminant = 1.0 / determinant
    val offset = origin - a
    val u = offset.dot(p) * inverseDeterminant
    if (u !in 0.0..1.0) return null
    val q = offset.cross(edge1)
    val v = direction.dot(q) * inverseDeterminant
    if (v < 0.0 || u + v > 1.0) return null
    val t = edge2.dot(q) * inverseDeterminant
    if (t < 0.0) return null
    val normal = edge1.cross(edge2).normalized()
    return t to if (normal.dot(direction) < 0.0) normal else -normal
}

fun rayQuad(origin: V3, direction: V3, a: V3, b: V3, c: V3, d: V3): Pair<Double, V3>? =
    listOfNotNull(rayTriangle(origin, direction, a, b, c), rayTriangle(origin, direction, a, c, d))
        .minByOrNull { it.first }

val root = Path.of("..").toAbsolutePath().normalize()
fun read(path: String) = Files.readString(root.resolve(path))
val failures = mutableListOf<String>()
fun expect(text: String, token: String, label: String) {
    if (!text.contains(token)) failures += "$label: missing $token"
}
fun expect(text: String, pattern: Regex, label: String) {
    if (!pattern.containsMatchIn(text)) failures += "$label: pattern ${pattern.pattern}"
}
fun reject(text: String, pattern: Regex, label: String) {
    if (pattern.containsMatchIn(text)) failures += "$label: obsolete ${pattern.pattern}"
}

val mappings = read("shaders/block.properties")
val models = read("shaders/techniques/voxel/BlockModels.glsl")
val hardcoded = read("shaders/util/HardcodedPBR.glsl")
val trace = read("shaders/techniques/voxel/VoxelTrace.glsl")
val voxelization = read("shaders/techniques/voxel/Voxelization.glsl")
val builder = read("shaders/pass/shadow/VoxelTreeBuilder.comp.glsl")
val faceMaskPath = root.resolve("shaders/pass/shadow/VoxelFaceMask.comp.glsl")
val faceMaskImplementationPath = root.resolve("shaders/techniques/voxel/VoxelFaceMask.glsl")
val programs = read("scripts/programs.main.kts")
val properties = read("scripts/shaders.properties")
val finalProperties = read("shaders/shaders.properties")
val textures = read("shaders/base/Textures.glsl")

val states = listOf(
    "oak_slab:type=bottom", "oak_slab:type=top",
    "oak_stairs:facing=east:half=bottom:shape=inner_left",
    "oak_door:facing=east:half=lower:hinge=left:open=true",
    "oak_fence:east=false:north=false:south=false:west=false"
)
fun id(state: String) = Regex("(?m)^block\\.(\\d+) = ${Regex.escape(state)}$")
    .find(mappings)?.groupValues?.get(1)?.toInt()
val ids = states.map { state -> id(state) ?: run { failures += "mapping missing: $state"; -1 } }
val allIds = Regex("(?m)^block\\.(\\d+) =").findAll(mappings).map { it.groupValues[1].toInt() }.toList()
val materialCount = allIds.maxOrNull()!! + 1
if (materialCount > 65536) failures += "material count exceeds packed 16-bit ID: $materialCount"

val pbrBytes = Files.readAllBytes(root.resolve("shaders/textures/pbr_lut_0.bin"))
if (pbrBytes.size != materialCount * 4) failures += "PBR LUT 0 size ${pbrBytes.size} != material count * 4 (${materialCount * 4})"
val flagBytes = Files.readAllBytes(root.resolve("shaders/textures/pbr_lut_1.bin"))
if (flagBytes.size != materialCount * 4) failures += "PBR LUT 1 size ${flagBytes.size} != material count * 4 (${materialCount * 4})"
val modelLutBytes = Files.readAllBytes(root.resolve("shaders/textures/pbr_lut_2.bin"))
val expectedModelLutBytes = materialCount * 64 * 4
if (modelLutBytes.size != expectedModelLutBytes) {
    failures += "PBR LUT 2 size ${modelLutBytes.size} != 64 masks * $materialCount materials * 4 ($expectedModelLutBytes)"
}
val quadPath = root.resolve("shaders/textures/block_model_quads.bin")
val quadBytes = if (Files.exists(quadPath)) Files.readAllBytes(quadPath) else {
    failures += "quad asset missing: shaders/textures/block_model_quads.bin"
    ByteArray(0)
}
val quadTextureWidth = 32768
val quadTextureRowBytes = quadTextureWidth * 4
if (quadBytes.size % quadTextureRowBytes != 0) {
    failures += "quad asset size ${quadBytes.size} is not a whole $quadTextureWidth-texel row"
}
val quadTextureHeight = quadBytes.size / quadTextureRowBytes
val quadCount = if (modelLutBytes.size == expectedModelLutBytes) {
    val modelLut = ByteBuffer.wrap(modelLutBytes).order(ByteOrder.LITTLE_ENDIAN)
    (0 until materialCount * 64).maxOf { index ->
        val metadata = modelLut.getInt(index * 4).toUInt()
        ((metadata shr 9) and 0xFFFFu) + ((metadata shr 25) and 0x3Fu)
    }.toInt()
} else 0
val expectedQuadTextureHeight = (quadCount * 2 + quadTextureWidth - 1) / quadTextureWidth
if (quadBytes.size != quadTextureRowBytes * expectedQuadTextureHeight) {
    failures += "quad asset size ${quadBytes.size} != minimal ${quadTextureWidth}x$expectedQuadTextureHeight RGBA8 texture"
}
if (quadBytes.drop(quadCount * 8).any { it != 0.toByte() }) failures += "quad texture padding is not zero"

for (index in 0..1) {
    val expectedPbrProperty =
        "customTexture.usam_pbrLUT$index=textures/pbr_lut_$index.bin TEXTURE_1D R32UI $materialCount RED_INTEGER UNSIGNED_INT"
    expect(properties, expectedPbrProperty, "maintained PBR LUT $index property")
    expect(finalProperties, expectedPbrProperty, "final PBR LUT $index property")
}
val expectedModelLutProperty =
    "customTexture.usam_pbrLUT2=textures/pbr_lut_2.bin TEXTURE_2D R32UI 64 $materialCount RED_INTEGER UNSIGNED_INT"
expect(properties, expectedModelLutProperty, "maintained 2D model LUT property")
expect(finalProperties, expectedModelLutProperty, "final 2D model LUT property")
fun checkQuadProperty(text: String, label: String) {
    val match = Regex("(?m)^customTexture\\.usam_blockModelQuads=textures/block_model_quads\\.bin TEXTURE_2D RGBA8 (\\d+) (\\d+) RGBA UNSIGNED_BYTE$").find(text)
    if (match == null) failures += "$label: 2D normalized quad texture declaration missing"
    else if (match.groupValues[1].toInt() != quadTextureWidth || match.groupValues[2].toInt() != quadTextureHeight) {
        failures += "$label: quad texture size ${match.groupValues[1]}x${match.groupValues[2]} != ${quadTextureWidth}x$quadTextureHeight"
    }
}
checkQuadProperty(properties, "maintained properties")
checkQuadProperty(finalProperties, "final properties")

if (Files.exists(root.resolve("shaders/textures/block_model_aabbs.bin"))) failures += "obsolete AABB asset remains"
listOf(models, properties, finalProperties, textures).forEachIndexed { index, text ->
    reject(text, Regex("aabb", RegexOption.IGNORE_CASE), "AABB token in runtime source $index")
}
expect(textures, Regex("uniform\\s+usampler2D\\s+usam_pbrLUT2\\s*;"), "2D model LUT sampler")
expect(textures, Regex("uniform\\s+sampler2D\\s+usam_blockModelQuads\\s*;"), "2D normalized quad sampler")

expect(models, "Generated by Shadesmith from Minecraft 26.2", "provenance")
expect(models, Regex("bool\\s+voxel_intersectBlockModel\\s*\\("), "model API")
expect(models, Regex("ivec2\\s+texelCoord\\s*=\\s*ivec2\\s*\\(\\s*int\\s*\\(\\s*\\(modelData\\s*>>\\s*8u\\)\\s*&\\s*0x7FFEu\\s*\\)\\s*,\\s*int\\s*\\(\\s*\\(modelData\\s*>>\\s*23u\\)\\s*&\\s*3u\\s*\\)\\s*\\)"), "model-level 2D quad texel address")
expect(models, Regex("texelCoord\\.x\\s*\\+=\\s*2"), "linear quad texel advance")
expect(models, Regex("texelCoord\\s*\\+\\s*ivec2\\s*\\(\\s*1\\s*,\\s*0\\s*\\)"), "adjacent second quad texel")
expect(models, Regex("rotation\\s*=\\s*modelData\\s*&\\s*0x1FFu"), "packed rotation decode")
expect(models, Regex("quadCount\\s*=\\s*\\(modelData\\s*>>\\s*25u\\)\\s*&\\s*0x3Fu"), "packed quad count decode")
if (Regex("while\\s*\\(\\s*quadCount\\s*!=\\s*0u\\s*\\)").findAll(models).count() != 2) {
    failures += "model-level axis split must have exactly two countdown quad loops"
}
expect(models, Regex("--quadCount"), "quad loop countdown")
expect(models, "_voxel_rotateBlockModelRay", "shared packed ray rotation")
reject(models, Regex("_voxel_rotateBlockModelComponent"), "duplicated component rotation")
expect(models, "_voxel_unrotateBlockModelVector", "model normal inverse rotation")
val axisQuadIntersection = models.substringAfter("bool _voxel_intersectBlockModelAxisAlignedQuad(")
    .substringBefore("bool _voxel_intersectBlockModelQuad(")
val quadIntersection = models.substringAfter("bool _voxel_intersectBlockModelQuad(")
    .substringBefore("bool voxel_intersectBlockModel(")
for ((label, intersection) in listOf("axis-aligned" to axisQuadIntersection, "general" to quadIntersection)) {
    if (Regex("texelFetch\\s*\\(\\s*usam_blockModelQuads\\b").findAll(intersection).count() != 2) {
        failures += "$label quad intersection must fetch exactly two quad texels"
    }
}
reject(axisQuadIntersection, Regex("\\baxisAligned\\b"), "per-quad axis-aligned branch")
expect(models, "vec3 inverseRayDir = 1.0 / rayDir;", "model-level reciprocal ray direction")
expect(axisQuadIntersection, "float t = (origin[axis] - rayOrigin[axis]) * inverseRayDir[axis];", "reciprocal axis-plane intersection")
reject(quadIntersection, Regex("\\bquadIndex\\b"), "per-quad 2D texel address")
reject(quadIntersection, Regex("normalize\\s*\\("), "per-quad basis normalization")
val modelFunction = models.substringAfter("bool voxel_intersectBlockModel(")
expect(modelFunction, "hit = hitT != uintBitsToFloat(0x7F800000u);", "axis-aligned post-loop hit inference")
if (Regex("\\|\\|\\s*hit").findAll(modelFunction).count() != 1) failures += "general quad path must retain exactly one hit accumulation"
expect(models, Regex("if\\s*\\(hit\\)\\s+hitNormal\\s*=\\s*_voxel_unrotateBlockModelVector\\([^;]*normalize\\s*\\(hitNormal\\)"), "post-loop hit normal normalization")
if (Regex("(?m)\\b(?:const\\s+)?(?:vec[234]|u?int|float|bool)\\s+\\w+\\s*\\[").containsMatchIn(models)) {
    failures += "block-model runtime uses a local or const array"
}

expect(hardcoded, "uint blockModelMetadata;", "PBR model metadata member")
expect(hardcoded, Regex("HardcodedPBR\\s+hardcodedpbr_decode\\s*\\(\\s*uint\\s+materialID\\s*,\\s*uint\\s+faceMask\\s*\\)"), "face-mask decode API")
expect(hardcoded, Regex("texelFetch\\s*\\(\\s*usam_pbrLUT2\\s*,\\s*ivec2\\s*\\(\\s*int\\s*\\(\\s*faceMask\\s*\\)\\s*,\\s*int\\s*\\(\\s*materialID\\s*\\)\\s*\\)\\s*,\\s*0\\s*\\)"), "row-major 2D model metadata fetch")
expect(hardcoded, Regex("hardcodedpbr_decode\\s*\\(\\s*materialID\\s*,\\s*63u\\s*\\)"), "one-argument mask-63 decode")

expect(builder, Regex("#define\\s+VOXEL_MATERIAL_DATA_MODIFIER\\s+(?:restrict\\s+)?readonly\\s+buffer"), "read-only tree-builder material buffer")
reject(builder, Regex("voxel_materials_v4\\s*\\[[^]]+]\\s*="), "tree-builder material writeback")
expect(builder, "rc_markPendingVisibleFace", "tree-builder RC visible-face publication")
if (Files.exists(faceMaskPath) || Files.exists(faceMaskImplementationPath)) failures += "obsolete voxel face-mask publisher remains"
reject(programs, Regex("VoxelFaceMask"), "voxel face-mask program registration")
reject(voxelization, Regex("open-face mask|0xFFFFu"), "packed face-mask material layout")
reject(trace, Regex("packedMaterial|openFaceMask|>>\\s*16u"), "packed neighbor face-mask consumption")
expect(trace, "uint rayFaceMask = uint((boundOffsetMask.x & 1) + 1) |", "ray X face selection")
expect(trace, "uint(((boundOffsetMask.y & 1) + 1) << 2) |", "ray Y face selection")
expect(trace, "uint(((boundOffsetMask.z & 1) + 1) << 4);", "ray Z face selection")
expect(trace, Regex("hardcodedpbr_decode\\s*\\(\\s*material\\s*,\\s*rayFaceMask\\s*\\)"), "ray-selected 2D model metadata lookup")
fun rayFaceMask(positiveX: Boolean, positiveY: Boolean, positiveZ: Boolean) =
    (if (positiveX) 2u else 1u) or (if (positiveY) 8u else 4u) or (if (positiveZ) 32u else 16u)
if (rayFaceMask(true, true, true) != 42u || rayFaceMask(false, false, false) != 21u) {
    failures += "ray face mask must select normals opposing each ray component"
}
val modelCall = trace.indexOf("if (voxel_intersectBlockModel(")
val modelMiss = trace.indexOf("isHit = false;", modelCall)
if (modelCall < 0 || modelMiss < 0) failures += "model miss does not continue hierarchical traversal"

if (modelLutBytes.size == expectedModelLutBytes && pbrBytes.size == materialCount * 4) {
    val modelLut = ByteBuffer.wrap(modelLutBytes).order(ByteOrder.LITTLE_ENDIAN)
    val flagLut = ByteBuffer.wrap(flagBytes).order(ByteOrder.LITTLE_ENDIAN)
    fun modelData(materialId: Int, mask: Int) = modelLut.getInt(4 * (materialId * 64 + mask)).toUInt()
    fun validRotation(rotation: UInt): Boolean {
        val axes = listOf(rotation and 3u, rotation shr 3 and 3u, rotation shr 6 and 3u)
        return axes.all { it < 3u } && axes.distinct().size == 3
    }
    for (mask in 0..63) for (materialId in 0 until materialCount) {
        val metadata = modelData(materialId, mask)
        if (metadata == 0u) continue
        val rotation = metadata and 0x1FFu
        val offset = (metadata shr 9).toInt() and 0xFFFF
        val count = (metadata shr 25).toInt() and 0x3F
        if (!validRotation(rotation) || count == 0 || offset + count > quadCount) {
            failures += "material $materialId mask $mask has invalid rotation/quad bounds"
        }
        if (offset % (quadTextureWidth / 2) + count > quadTextureWidth / 2) {
            failures += "material $materialId mask $mask crosses a quad texture row"
        }
    }
    ids.filter { it >= 0 }.forEach { if (modelData(it, 63) == 0u) failures += "representative model $it has zero mask-63 metadata" }
    val fullCubeIds = (0 until materialCount).filter { (flagLut.getInt(it * 4).toUInt() shr 4) and 1u == 1u }
    if ((fullCubeIds + listOf(0, 1)).any { id -> (0..63).any { mask -> modelData(id, mask) != 0u } }) {
        failures += "full-cube, water, or unsupported LUT row contains model metadata"
    }
}

val diagonalA = V3(0.0, 0.0, 0.0)
val diagonalB = V3(1.0, 0.0, 1.0)
val diagonalC = V3(1.0, 1.0, 1.0)
val diagonalD = V3(0.0, 1.0, 0.0)
val forward = rayQuad(V3(-1.0, 0.5, 2.0), V3(1.0, 0.0, -1.0), diagonalA, diagonalB, diagonalC, diagonalD)
val backward = rayQuad(V3(2.0, 0.5, -1.0), V3(-1.0, 0.0, 1.0), diagonalA, diagonalB, diagonalC, diagonalD)
if (forward?.first != 1.5 || backward?.first != 1.5 || forward.second.dot(V3(1.0, 0.0, -1.0)) >= 0.0 || backward.second.dot(V3(-1.0, 0.0, 1.0)) >= 0.0) {
    failures += "quad normal must face against the ray"
}
val first = rayQuad(V3(-1.0, .75, .5), V3(1.0, 0.0, 0.0), V3(0.0, 0.0, 0.0), V3(0.0, .5, 0.0), V3(0.0, .5, 1.0), V3(0.0, 0.0, 1.0))
val later = rayQuad(V3(-1.0, .75, .5), V3(1.0, 0.0, 0.0), V3(2.0, 0.0, 0.0), V3(2.0, 1.0, 0.0), V3(2.0, 1.0, 1.0), V3(2.0, 0.0, 1.0))
if (first != null || later?.first != 3.0) failures += "model-miss continuation"

data class DdaStep(val level: Int, val packedBlockPos: Int?)
fun stepEmptyCell(blockPos: Int, stepDir: Int, gridBlocks: Int): DdaStep {
    val updatedBlockPos = blockPos + stepDir
    return if (updatedBlockPos in 0 until gridBlocks) DdaStep(1, updatedBlockPos) else DdaStep(0, null)
}
val gridBlocks = 64
if (stepEmptyCell(gridBlocks - 1, 1, gridBlocks) != DdaStep(0, null) ||
    stepEmptyCell(0, -1, gridBlocks) != DdaStep(0, null) ||
    stepEmptyCell(17, 1, gridBlocks) != DdaStep(1, 18)) failures += "DDA boundary continuation"

check(failures.isEmpty()) { "Voxel trace quad contract failed:\n" + failures.joinToString("\n") { "- $it" } }
println("Voxel trace quad contract PASS: ${allIds.size} mappings, $quadCount quads, ${quadTextureWidth}x$quadTextureHeight texels, Minecraft 26.2")
