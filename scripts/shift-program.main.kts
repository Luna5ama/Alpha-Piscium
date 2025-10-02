import java.nio.channels.FileChannel
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import java.security.MessageDigest
import kotlin.io.path.*
import kotlin.system.exitProcess

if (args.size != 3) {
    println("Usage: shift-program.main.kts <prefix> <center> <delta>")
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

val movingSrc = if (delta > 0) entries.keys.sorted() else entries.keys.sortedDescending()
val movingDst = movingSrc.toMutableList()
val toMove = mutableListOf<Pair<Int, Int>>()

for (i in movingSrc.indices) {
    val src = movingSrc[i]
    if (delta > 0 && src < center) continue
    if (delta < 0 && src > center) continue
    if (src == center) {
        movingDst[i] += delta
    } else if (i > 0) {
        movingDst[i] = if (delta > 0) maxOf(movingDst[i], movingDst[i - 1] + 1)
        else minOf(movingDst[i], movingDst[i - 1] - 1)
    }
    val newDst = movingDst[i]
    if (newDst != src) {
        check(newDst in 1..99) { "shift too big" }
        toMove.add(src to newDst)
    }
}

toMove.reverse()
toMove.forEach(::println)

fun hash(path: Path): ByteArray {
    return FileChannel.open(path, StandardOpenOption.READ).use { channel ->
        val mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update(mapped)
        digest.digest()!!
    }
}

val hashes = toMove.asSequence()
    .mapNotNull { entries[it.first] }
    .flatMap { paths ->
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

val scriptsShadersProperites = Path("shaders.properties")
val shadersShaderProperties = shadersDir.resolve(scriptsShadersProperites.name)

val movedRenames = moved.map { (oldPath, newPath) ->
    "(?<!\\w)${oldPath.nameWithoutExtension}(?!\\w)".toRegex() to newPath.nameWithoutExtension
}

listOf(scriptsShadersProperites, shadersShaderProperties).forEach { path ->
    path.writeLines(
        path.readLines()
            .map {
                var line = it
                movedRenames.forEach { (old, new) ->
                    line = line.replace(old, new)
                }
                line
            }
    )
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

