#!/usr/bin/env kotlin

/*
    Copyright 2025 Luna

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

import java.nio.channels.FileChannel
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import java.security.MessageDigest
import kotlin.io.path.*
import kotlin.system.exitProcess

if (args.size !in 3..5) {
    println("Usage: shift-program.main.kts <prefix> <center> <delta> <world path from shaders (.)> <shaders path (../shaders)>")
    exitProcess(1)
}

val prefix = args[0]
val validPrefixes = setOf("setup", "begin", "shadowcomp", "prepare", "deferred", "composite")
check(prefix in validPrefixes) { "Invalid prefix: $prefix, must be one of $validPrefixes" }

val center = args[1].toInt()
check(center in 0..99) { "Center must be in range 0..99" }
val delta = args[2].toInt()
check(delta != 0) { "Delta must be non-zero" }

val shadersDirStr = args.getOrElse(4) { "../shaders" }
val shadersDir = Path(shadersDirStr)

val worldStr = args.getOrElse(3) { "." }
val worldDir = shadersDir.resolve(worldStr).normalize()

val entries = worldDir.listDirectoryEntries("$prefix*").asSequence()
    .map {
        val justName = it.nameWithoutExtension
        var underscoreIndex = justName.indexOf('_')
        if (underscoreIndex == -1) underscoreIndex = justName.length
        val index = if (prefix.length == underscoreIndex) 0 else justName.substring(prefix.length, underscoreIndex).toInt()
        index to it
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
        check(newDst in 0..99) { "shift too big" }
        toMove.add(src to newDst)
    }
}

fun indexToString(index: Int): String {
    return if (index == 0) "" else index.toString()
}

toMove.reverse()
println("Pending renames:")
toMove.forEachIndexed { i, it ->
    println("$i: $prefix${indexToString(it.first)} -> $prefix${indexToString(it.second)}")
}

println("This can break your code, make sure to commit all changes before proceeding. Type 'Y' to continue:")
val confirmation = readLine().toString().lowercase()
if (confirmation != "y" && confirmation != "yes") {
    println("Aborting")
    exitProcess(0)
}

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
    val oldIndexStr = indexToString(oldIndex)
    val newIndexStr = indexToString(newIndex)
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

sequenceOf(scriptsShadersProperites, shadersShaderProperties)
    .filter { it.exists() }
    .forEach { path ->
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
    .directory(worldDir.toFile())
    .inheritIO()
    .start()
    .waitFor()

