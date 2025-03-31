import java.awt.image.BufferedImage
import java.awt.image.DataBuffer
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.StandardOpenOption
import javax.imageio.ImageIO
import kotlin.io.path.Path
import kotlin.io.path.absolute
import kotlin.io.path.createDirectories
import kotlin.io.path.exists
import kotlin.io.path.listDirectoryEntries
import kotlin.io.path.nameWithoutExtension
import kotlin.system.exitProcess

if (args.size != 5) {
    println("Usage: bintex.main.kts <dimensions (separated by underscore '_')> <channels> <input directory> <input prefix> <output file>")
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

val inputDirectory = Path(args[2]).absolute()
if (!inputDirectory.exists()) {
    println("Input directory does not exist")
    exitProcess(1)
}

val inputPrefix = args[3]
if (inputPrefix.isEmpty()) {
    println("Input prefix cannot be empty")
    exitProcess(1)
}

val outputFile = Path(args[4]).absolute()
outputFile.parent.createDirectories()

val inputFiles = inputDirectory.listDirectoryEntries("$inputPrefix*")
    .filter { it.nameWithoutExtension.removePrefix(inputPrefix).toIntOrNull() != null }
    .sortedBy { it.nameWithoutExtension.removePrefix(inputPrefix).toInt() }
    .toList()

if (inputFiles.isEmpty()) {
    println("No input files found with prefix $inputPrefix")
    exitProcess(1)
}

val outputSize = dimensions.fold(1L) { acc, dim -> acc * dim } * channels

fun BufferedImage.copyDataTo(buffer: ByteBuffer) {
    fun BufferedImage.createDataArray(): Any {
        val numBands = raster.numBands
        return when (val dataType = raster.dataBuffer.dataType) {
            DataBuffer.TYPE_BYTE -> ByteArray(numBands)
            DataBuffer.TYPE_USHORT -> ShortArray(numBands)
            DataBuffer.TYPE_INT -> IntArray(numBands)
            DataBuffer.TYPE_FLOAT -> FloatArray(numBands)
            DataBuffer.TYPE_DOUBLE -> DoubleArray(numBands)
            else -> throw IllegalArgumentException("Unknown data buffer type: $dataType")
        }
    }

    val data = createDataArray()

    when (channels) {
        1 -> {
            for (y in 0..<height) {
                for (x in 0..<width) {
                    val dataElement = raster.getDataElements(x, y, data)
                        buffer.put(colorModel.getRed(dataElement).toByte())
                }
            }
        }
        2 -> {
            for (y in 0..<height) {
                for (x in 0..<width) {
                    val dataElement = raster.getDataElements(x, y, data)
                    buffer.put(colorModel.getRed(dataElement).toByte())
                    buffer.put(colorModel.getGreen(dataElement).toByte())
                }
            }
        }
        3 -> {
            for (y in 0..<height) {
                for (x in 0..<width) {
                    val dataElement = raster.getDataElements(x, y, data)
                    buffer.put(colorModel.getRed(dataElement).toByte())
                    buffer.put(colorModel.getGreen(dataElement).toByte())
                    buffer.put(colorModel.getBlue(dataElement).toByte())
                }
            }
        }
        4 -> {
            for (y in 0..<height) {
                for (x in 0..<width) {
                    val dataElement = raster.getDataElements(x, y, data)
                    buffer.put(colorModel.getRed(dataElement).toByte())
                    buffer.put(colorModel.getGreen(dataElement).toByte())
                    buffer.put(colorModel.getBlue(dataElement).toByte())
                    buffer.put(colorModel.getAlpha(dataElement).toByte())
                }
            }
        }
    }
}

FileChannel.open(outputFile, StandardOpenOption.CREATE, StandardOpenOption.READ, StandardOpenOption.WRITE).use {
    val mapped = it.map(FileChannel.MapMode.READ_WRITE, 0L, outputSize)
    inputFiles.forEach {
        ImageIO.read(it.toFile()).copyDataTo(mapped)
    }
}