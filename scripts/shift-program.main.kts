import java.nio.channels.FileChannel
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import java.security.MessageDigest
import kotlin.io.path.*
import kotlin.math.abs
import kotlin.math.sign
import kotlin.system.exitProcess

if (args.size != 3) {
    println("Usage: shift-program.main.kts <prefix>> <center> <delta>")
    exitProcess(1)
}

val prefix = args[0]
val validPrefixes = setOf("setup", "begin", "shadowcomp", "prepare", "deferred", "composite")
check(prefix in validPrefixes) { "Invalid prefix: $prefix, must be one of $validPrefixes" }

val center = args[1].toInt()
check(center in 1..99) { "Center must be in range 1..99" }
val delta = args[2].toInt()
check(delta != 0) { "Delta must be non-zero" }

val shadersDir = Path("../shaders")

val entries = shadersDir.listDirectoryEntries("$prefix*").asSequence()
    .map {
        val justName = it.nameWithoutExtension
        var underscoreIndex = justName.indexOf('_')
        if (underscoreIndex == -1) underscoreIndex = justName.length
        justName.substring(prefix.length, underscoreIndex).toInt() to it
    }
    .groupBy({ it.first }, { it.second })

check(center in entries) { "Center index not found in shader names" }


var spaces = 0
val toMove = entries.keys.asSequence()
    .run {
        if (delta > 0) {
            filter { it >= center }.sorted()
        } else {
            filter { it <= center }.sortedDescending()
        }
    }
    .windowed(2, 1, false)
    .map { (a, b) ->
        val diff = abs(a - b)
        val move = abs(delta) - spaces
        spaces += diff - 1
        a to move
    }
    .takeWhile { it.second > 0 }
    .map { it.first to it.first + it.second * delta.sign }
    .run {
        if (delta > 0) {
            sortedByDescending { it.first }
        } else {
            sortedBy { it.first }
        }
    }
    .toList()

val toMoveIndices = toMove.mapTo(mutableSetOf()) { it.first }

if (delta > 0) {
    check(99 !in toMoveIndices) { "shift too big" }
} else {
    check(1 !in toMoveIndices) { "shift too big" }
}

fun hash(path: Path): ByteArray {
    return FileChannel.open(path, StandardOpenOption.READ).use { channel ->
        val mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update(mapped)
        digest.digest()!!
    }
}

val hashes = entries.asSequence()
    .filter { it.key in toMoveIndices }
    .flatMap { (_, paths) ->
        paths.map {
            it to hash(it)
        }
    }
    .toMap()

val moved = toMove.flatMap { (oldIndex, newIndex) ->
    val oldIndexStr = oldIndex.toString()
    val newIndexStr = newIndex.toString()
    entries[oldIndex]!!.map { oldPath ->
        val startIndex = prefix.length
        val endIndex = startIndex + oldIndexStr.length
        val newName = oldPath.name.replaceRange(prefix.length, endIndex, newIndexStr)
        val newPath = oldPath.parent.resolve(newName)
        println("Renaming $oldPath to $newPath")
        var retry = 0
        while (!oldPath.toFile().renameTo(newPath.toFile())) {
            Thread.sleep(100)
            if (++retry >= 3) {
                throw RuntimeException("Failed to rename $oldPath to $newPath")
            }
        }
        oldPath to newPath
    }
}

Thread.sleep(500)

moved.forEach { (oldPath, newPath) ->
    val oldHash = hashes[oldPath]!!
    val newHash = hash(newPath)
    check(oldHash.contentEquals(newHash)) { "Hashes do not match for $oldPath and $newPath, revert your files using git" }
}

val updated = moved.asSequence()
    .map { it.second }
    .distinct()
    .map { it.pathString }
    .toList()

ProcessBuilder(listOf("git", "add") + updated)
    .inheritIO()
    .start()
    .waitFor()

