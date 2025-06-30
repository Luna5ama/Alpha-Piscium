@file:OptIn(ExperimentalUnsignedTypes::class)

import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.StandardOpenOption
import kotlin.io.path.Path

// Constants from Hash.glsl
val HASH_XXHASH32_PRIME32_2 = 2246822519U;
val HASH_XXHASH32_PRIME32_3 = 3266489917U;
val HASH_XXHASH32_PRIME32_4 = 668265263U;
val HASH_XXHASH32_PRIME32_5 = 374761393U;

fun hash_pcg4d_44(vIn: UIntArray): UIntArray {
    require(vIn.size == 4)
    val v = vIn.copyOf()
    // v = v * 1664525u + 1013904223u
    for (i in 0..3) v[i] = v[i] * 1664525u + 1013904223u
    v[0] += v[1] * v[3]
    v[1] += v[2] * v[0]
    v[2] += v[0] * v[1]
    v[3] += v[1] * v[2]
    for (i in 0..3) v[i] = v[i] xor (v[i] shr 16)
    v[0] += v[1] * v[3]
    v[1] += v[2] * v[0]
    v[2] += v[0] * v[1]
    v[3] += v[1] * v[2]
    return v
}

val whiteNoiseTexName = "white_noise_64x64x64.bin"
val whiteNoiseTexDimensions = 64
val whiteNoiseTexPath = Path("../shaders/textures/$whiteNoiseTexName")
val outputSize = whiteNoiseTexDimensions * whiteNoiseTexDimensions * whiteNoiseTexDimensions * 4 * 2

FileChannel.open(
    whiteNoiseTexPath,
    StandardOpenOption.CREATE,
    StandardOpenOption.READ,
    StandardOpenOption.WRITE,
    StandardOpenOption.TRUNCATE_EXISTING
).use { outputChannel ->
    val mapped = outputChannel.map(FileChannel.MapMode.READ_WRITE, 0L, outputSize.toLong())
        .order(ByteOrder.nativeOrder())
    for (z in 0 until whiteNoiseTexDimensions) {
        for (y in 0 until whiteNoiseTexDimensions) {
            for (x in 0 until whiteNoiseTexDimensions) {
                val v = uintArrayOf(
                    x.toUInt() + HASH_XXHASH32_PRIME32_2,
                    y.toUInt() + HASH_XXHASH32_PRIME32_3,
                    z.toUInt() + HASH_XXHASH32_PRIME32_4,
                    HASH_XXHASH32_PRIME32_5
                )
                val hash = hash_pcg4d_44(v)
                // Remap each UInt (0..0xFFFFFFFFu) to UInt16 (0..0xFFFF)
                for (c in 0..3) {
                    val u16 = (hash[c] shr 16).toUShort() // Take upper 16 bits
                    mapped.putShort(u16.toShort())
                }
            }
        }
    }
}