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
val initialTrace = read("shaders/pass/composite/GIReSTIRInitalSampleVoxelFallback.comp.glsl")
val voxelization = read("shaders/techniques/voxel/Voxelization.glsl")
val shadowVertex = read("shaders/pass/geometry/ShadowPass.vert.glsl")
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
fun voxelMaterialData(materialID: Int, fullCube: Boolean) = (materialID shl 1) or if (fullCube) 1 else 0
for (materialID in 0 until 65536) {
    for (fullCube in listOf(false, true)) {
        val materialData = voxelMaterialData(materialID, fullCube)
        if ((materialData ushr 1) != materialID || ((materialData and 1) != 0) != fullCube) {
            failures += "voxel material round trip failed: $materialID $fullCube"
        }
    }
    if (materialID != 65535 && voxelMaterialData(materialID, true) >= voxelMaterialData(materialID + 1, false)) {
        failures += "voxel material atomic ordering failed: $materialID"
    }
}
if (voxelMaterialData(1, false) != 2) failures += "voxel placeholder encoding changed"

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
expect(models, Regex("ivec2\\s+texelCoord\\s*=\\s*ivec2\\s*\\(\\s*int\\s*\\(\\s*bitfieldExtract\\s*\\(\\s*modelData\\s*,\\s*9\\s*,\\s*14\\s*\\)\\s*<<\\s*1u\\s*\\)\\s*,\\s*int\\s*\\(\\s*bitfieldExtract\\s*\\(\\s*modelData\\s*,\\s*23\\s*,\\s*2\\s*\\)\\s*\\)\\s*\\)"), "bitfield-extracted 2D quad texel address")
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
expect(models, "uvec3 transform = uvec3(rotation, rotation >> 3u, rotation >> 6u);", "vectorized inverse rotation decode")
expect(models, "ivec3 axis = ivec3(transform & 3u);", "inverse rotation axis extraction")
expect(models, "vec3 signedValue = value * mix(vec3(1.0), vec3(-1.0), notEqual(transform & 4u, uvec3(0u)));", "inverse rotation sign extraction")
expect(models, "result[axis.x] = signedValue.x;", "inverse rotation X indexed write")
expect(models, "result[axis.y] = signedValue.y;", "inverse rotation Y indexed write")
expect(models, "result[axis.z] = signedValue.z;", "inverse rotation Z indexed write")
reject(models, Regex("rotation\\s*>>=\\s*3u"), "serial inverse rotation decode")
expect(models, Regex("void\\s+_voxel_intersectBlockModelAxisAlignedQuad\\s*\\("), "axis-aligned quad helper API")
val axisQuadIntersection = models.substringAfter("void _voxel_intersectBlockModelAxisAlignedQuad(")
    .substringBefore("bool _voxel_intersectBlockModelQuad(")
val quadIntersection = models.substringAfter("bool _voxel_intersectBlockModelQuad(")
    .substringBefore("bool voxel_intersectBlockModel(")
for ((label, intersection) in listOf("axis-aligned" to axisQuadIntersection, "general" to quadIntersection)) {
    if (Regex("texelFetch\\s*\\(\\s*usam_blockModelQuads\\b").findAll(intersection).count() != 2) {
        failures += "$label quad intersection must fetch exactly two quad texels"
    }
}
reject(axisQuadIntersection, Regex("\\baxisAligned\\b"), "per-quad axis-aligned branch")
expect(axisQuadIntersection, "if (originNormalX.w < 253.5 / 255.0)", "direct X axis encoding threshold")
expect(axisQuadIntersection, "else if (originNormalX.w < 254.5 / 255.0)", "direct Y axis encoding threshold")
reject(axisQuadIntersection, Regex("int\\s+axis\\s*=\\s*int\\s*\\(\\s*originNormalX\\.w"), "integer axis encoding decode")
expect(models, "vec3 inverseRayDir = 1.0 / rayDir;", "model-level reciprocal ray direction")
expect(axisQuadIntersection, "t = (origin.x - rayOrigin.x) * inverseRayDir.x;", "reciprocal X axis-plane intersection")
expect(axisQuadIntersection, "t = (origin.y - rayOrigin.y) * inverseRayDir.y;", "reciprocal Y axis-plane intersection")
expect(axisQuadIntersection, "t = (origin.z - rayOrigin.z) * inverseRayDir.z;", "reciprocal Z axis-plane intersection")
reject(quadIntersection, Regex("\\bquadIndex\\b"), "per-quad 2D texel address")
reject(quadIntersection, Regex("normalize\\s*\\("), "per-quad basis normalization")
if (Regex("projected\\.x > halfSize\\.x \\|\\| projected\\.y > halfSize\\.y").findAll(models).count() != 1) {
    failures += "axis-aligned quad intersection does not short-circuit projected bounds"
}
expect(quadIntersection, "float projectedU = abs(dot(offset, u));", "general quad U projection")
expect(quadIntersection, "if (projectedU > halfSize.x) return false;", "general quad U bound short-circuit")
expect(quadIntersection, "if (abs(dot(offset, v)) > halfSize.y) return false;", "general quad V bound short-circuit")
reject(models, Regex("greaterThan\\(projected"), "eager projected quad bounds")
val modelFunction = models.substringAfter("bool voxel_intersectBlockModel(")
expect(modelFunction, "hit = hitT != uintBitsToFloat(0x7F800000u);", "axis-aligned post-loop hit inference")
if (Regex("\\|\\|\\s*hit").findAll(modelFunction).count() != 1) failures += "general quad path must retain exactly one hit accumulation"
expect(
    models,
    Regex("""if \(!axisAligned\) \{\s*hitNormal = normalize\(hitNormal\);\s*if \(dot\(hitNormal, rayDir\) > 0\.0\) hitNormal = -hitNormal;\s*\}"""),
    "general-only post-loop hit normal orientation"
)
expect(models, "hitNormal = _voxel_unrotateBlockModelVector(rotation, hitNormal);", "post-loop hit normal inverse rotation")
if (Regex("(?m)\\b(?:const\\s+)?(?:vec[234]|u?int|float|bool)\\s+\\w+\\s*\\[").containsMatchIn(models)) {
    failures += "block-model runtime uses a local or const array"
}

reject(hardcoded, Regex("blockModelMetadata|usam_pbrLUT2"), "unconditional model metadata decode")
expect(shadowVertex, "uint lookupMaterial = hardcoded.isKnown ? materialID : 0u;", "safe voxelization model lookup")
expect(shadowVertex, "uint blockModelMetadata = texelFetch(usam_pbrLUT2, ivec2(63, int(lookupMaterial)), 0).x;", "voxelization model metadata lookup")

expect(builder, Regex("#define\\s+VOXEL_MATERIAL_DATA_MODIFIER\\s+(?:restrict\\s+)?readonly\\s+buffer"), "read-only tree-builder material buffer")
reject(builder, Regex("voxel_materials_v4\\s*\\[[^]]+]\\s*="), "tree-builder material writeback")
expect(voxelization, "uint voxel_decodeMaterialID(uint materialData)", "voxel material decode API")
expect(shadowVertex, "uint materialData = (materialID << 1u) | uint(hardcoded.isFullCube);", "voxel full-cube material encoding")
expect(shadowVertex, "atomicMax(voxel_materials[matIdx], 1u << 1u);", "voxel placeholder encoding")
expect(voxelization, "return materialData >= 4u;", "encoded voxel opacity classification")
expect(builder, "bool voxel_isGIOpaqueMaterial(uint materialData)", "encoded tree-builder material classification")
expect(builder, "return materialData >= 4u;", "tree-builder encoded opacity classification")
expect(builder, "rc_markPendingVisibleFace", "tree-builder RC visible-face publication")
if (Files.exists(faceMaskPath) || Files.exists(faceMaskImplementationPath)) failures += "obsolete voxel face-mask publisher remains"
reject(programs, Regex("VoxelFaceMask"), "voxel face-mask program registration")
reject(voxelization, Regex("open-face mask|0xFFFFu"), "packed face-mask material layout")
reject(trace, Regex("packedMaterial|openFaceMask|>>\\s*16u"), "packed neighbor face-mask consumption")
expect(trace, "uint rayFaceMask = uint(42 + directionSign.x +", "ray X face selection")
expect(trace, "(directionSign.y << 2) + (directionSign.z << 4));", "ray YZ face selection")
expect(trace, "bool isHit = bool((maskPart >> (mortonPrefix & 31u)) & 1u);", "direct child occupancy bit index")
reject(trace, Regex("uint\\s+childIdx\\s*="), "redundant 6-bit child index")
expect(initialTrace, "#define VOXEL_TRACE_TRUST_MATERIAL_ID", "trusted initial trace material ID")
expect(trace, "#ifndef VOXEL_TRACE_TRUST_MATERIAL_ID", "selectable material ID validation")
expect(trace, "bool isFullCube = bool(materialData & 1u);", "cached trace full-cube lookup")
reject(trace, Regex("texelFetch\\s*\\(\\s*usam_pbrLUT1"), "trace full-cube texture lookup")
expect(trace, "usam_pbrLUT2, ivec2(int(rayFaceMask), int(lookupMaterial)), 0", "ray-selected model lookup")
reject(trace, Regex("#include\\s+\"/util/HardcodedPBR\\.glsl\""), "unused hardcoded PBR include")
reject(trace, Regex("HardcodedPBR\\s+hardcoded\\s*=\\s*hardcodedpbr_decode\\s*\\(\\s*material\\s*,\\s*rayFaceMask\\s*\\)"), "full PBR decode in voxel tracing")
expect(trace, "bool isKnown = material < textureSize(usam_pbrLUT0, 0).x &&\n                    material < textureSize(usam_pbrLUT1, 0).x &&\n                    material < textureSize(usam_pbrLUT2, 0).y;", "short-circuit PBR LUT bounds")
if (Regex("textureSize\\s*\\(\\s*usam_pbrLUT[012]").findAll(trace).count() != 3) {
    failures += "voxel trace must retain exactly three local PBR LUT size queries"
}
expect(trace, "uint lookupMaterial = isKnown ? material : 0u;", "safe default model lookup material")
fun rayFaceMask(positiveX: Boolean, positiveY: Boolean, positiveZ: Boolean) =
    (if (positiveX) 2u else 1u) or (if (positiveY) 8u else 4u) or (if (positiveZ) 32u else 16u)
fun directionSignFaceMask(positiveX: Boolean, positiveY: Boolean, positiveZ: Boolean) =
    (42 + (if (positiveX) 0 else -1) + (if (positiveY) 0 else -4) + (if (positiveZ) 0 else -16)).toUInt()
val directions = listOf(false, true)
if (directions.any { x -> directions.any { y -> directions.any { z ->
        rayFaceMask(x, y, z) != directionSignFaceMask(x, y, z)
    } } }) {
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
    fun serialUnrotate(rotation: UInt, value: V3): V3 {
        val result = DoubleArray(3)
        var packed = rotation
        val components = listOf(value.x, value.y, value.z)
        for (component in 0..2) {
            val axis = (packed and 3u).toInt()
            val sign = if ((packed and 4u) == 0u) 1.0 else -1.0
            result[axis] = components[component] * sign
            packed = packed shr 3
        }
        return V3(result[0], result[1], result[2])
    }
    fun vectorizedUnrotate(rotation: UInt, value: V3): V3 {
        val transform = listOf(rotation, rotation shr 3, rotation shr 6)
        val result = DoubleArray(3)
        val components = listOf(value.x, value.y, value.z)
        for (component in 0..2) {
            val encoded = transform[component]
            val axis = (encoded and 3u).toInt()
            val sign = if ((encoded and 4u) == 0u) 1.0 else -1.0
            result[axis] = components[component] * sign
        }
        return V3(result[0], result[1], result[2])
    }
    val normalSamples = listOf(
        V3(1.0, 0.0, 0.0),
        V3(0.0, 1.0, 0.0),
        V3(0.0, 0.0, 1.0),
        V3(1.0, 2.0, 3.0),
        V3(-2.5, 0.75, -4.0),
    )
    for (rotationValue in 0 until 512) {
        val rotation = rotationValue.toUInt()
        if (!validRotation(rotation)) continue
        for (normal in normalSamples) {
            val expected = serialUnrotate(rotation, normal)
            val actual = vectorizedUnrotate(rotation, normal)
            if (abs(expected.x - actual.x) > 1e-12 || abs(expected.y - actual.y) > 1e-12 || abs(expected.z - actual.z) > 1e-12) {
                failures += "packed inverse rotation mismatch: rotation=$rotationValue normal=$normal expected=$expected actual=$actual"
            }
        }
    }
    for (mask in 0..63) for (materialId in 0 until materialCount) {
        val metadata = modelData(materialId, mask)
        if (metadata == 0u) continue
        val rotation = metadata and 0x1FFu
        val offset = (metadata shr 9).toInt() and 0xFFFF
        val count = (metadata shr 25).toInt() and 0x3F
        val legacyTexelX = ((metadata shr 8) and 0x7FFEu).toInt()
        val legacyTexelY = ((metadata shr 23) and 3u).toInt()
        val extractedTexelX = (((metadata shr 9) and 0x3FFFu) shl 1).toInt()
        val extractedTexelY = ((metadata shr 23) and 3u).toInt()
        if (legacyTexelX != extractedTexelX || legacyTexelY != extractedTexelY) {
            failures += "material $materialId mask $mask changes 2D texel address: legacy=($legacyTexelX,$legacyTexelY) extracted=($extractedTexelX,$extractedTexelY)"
        }
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
