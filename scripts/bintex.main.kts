import java.awt.image.BufferedImage
import java.awt.image.ComponentColorModel
import java.awt.image.DataBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.StandardOpenOption
import javax.imageio.ImageIO
import kotlin.io.path.Path
import kotlin.io.path.absolute
import kotlin.io.path.nameWithoutExtension
import kotlin.system.exitProcess

if (args.size < 4) {
    println("Usage: bintex.main.kts <dimensions (separated by underscore '_')> <channels> <output file> <input file(s)>")
    exitProcess(1)
}

val dimensions = args[0].splitToSequence("_")
    .map { it.toInt() }
    .toList()

if (dimensions.size !in 1..3) {
    println("Dimension size must be 1, 2, or 3")
    exitProcess(1)
}

val channels = args[1].toInt()

if (channels !in 1..4) {
    println("Channels must be 1, 2, 3, or 4")
    exitProcess(1)
}

val numberRegex = """.+?(\d+)$""".toRegex()
val outputFile = Path(args[2]).absolute()
val inputFiles = args.asSequence()
    .drop(3)
    .map { Path(it).absolute() }
    .sortedBy { path ->
        val matchResult = numberRegex.matchEntire(path.nameWithoutExtension)
        matchResult?.let {
            it.groupValues[1].toIntOrNull()
        } ?: Int.MAX_VALUE
    }
    .toList()

fun BufferedImage.copyDataTo(buffer: ByteBuffer) {
    var dataArray: Any? = null

    for (y in 0..<height) {
        for (x in 0..<width) {
            dataArray = raster.getDataElements(x, y, dataArray)

            if (colorModel is ComponentColorModel) {
                when (dataArray) {
                    is ByteArray -> {
                        buffer.put(dataArray, 0, channels)
                    }
                    is ShortArray -> {
                        for (i in 0..<channels) {
                            buffer.putShort(dataArray[i])
                        }
                    }
                    is IntArray -> {
                        for (i in 0..<channels) {
                            buffer.putInt(dataArray[i])
                        }
                    }
                    is FloatArray -> {
                        for (i in 0..<channels) {
                            buffer.putFloat(dataArray[i])
                        }
                    }
                    is DoubleArray -> {
                        for (i in 0..<channels) {
                            buffer.putDouble(dataArray[i])
                        }
                    }
                    else -> {
                        throw IllegalArgumentException("Unsupported data array type: ${dataArray?.javaClass?.name}")
                    }
                }
            } else {
                when (channels) {
                    1 -> {
                        buffer.put(colorModel.getRed(dataArray).toByte())
                    }
                    2 -> {
                        buffer.put(colorModel.getRed(dataArray).toByte())
                        buffer.put(colorModel.getGreen(dataArray).toByte())
                    }
                    3 -> {
                        buffer.put(colorModel.getRed(dataArray).toByte())
                        buffer.put(colorModel.getGreen(dataArray).toByte())
                        buffer.put(colorModel.getBlue(dataArray).toByte())
                    }
                    4 -> {
                        buffer.put(colorModel.getRed(dataArray).toByte())
                        buffer.put(colorModel.getGreen(dataArray).toByte())
                        buffer.put(colorModel.getBlue(dataArray).toByte())
                        buffer.put(colorModel.getAlpha(dataArray).toByte())
                    }
                }
            }
        }
    }
}

val colorModel = ImageIO.read(inputFiles.first().toFile()).colorModel!!
val pixelSizeByte = colorModel.pixelSize / colorModel.numComponents / 8
val outputSize = dimensions.fold(1L) { acc, dim -> acc * dim } * channels * pixelSizeByte

FileChannel.open(
    outputFile,
    StandardOpenOption.CREATE,
    StandardOpenOption.READ,
    StandardOpenOption.WRITE,
    StandardOpenOption.TRUNCATE_EXISTING
).use { outputChannel ->
    val mapped = outputChannel.map(FileChannel.MapMode.READ_WRITE, 0L, outputSize)
        .order(ByteOrder.nativeOrder())
    inputFiles.forEach {
        ImageIO.read(it.toFile()).copyDataTo(mapped)
    }
}