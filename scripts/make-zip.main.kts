import java.io.File
import java.util.zip.Deflater
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

val included = setOf("changelogs", "shaders", "licenses", "LICENSE", "README.md")
val branchName = Runtime.getRuntime().exec(arrayOf("git", "rev-parse", "--abbrev-ref", "HEAD")).inputStream.bufferedReader().readText().trim()
val commitTag = Runtime.getRuntime().exec(arrayOf("git", "rev-parse", "--short", "HEAD")).inputStream.bufferedReader().readText().trim()

val currDir = File("")
val rootDir = currDir.absoluteFile.parentFile
ZipOutputStream(File(rootDir, "${rootDir.name.replace("-", " ")} $branchName $commitTag.zip").outputStream(), Charsets.UTF_8).use { zipOut ->
    zipOut.setLevel(Deflater.DEFAULT_COMPRESSION)
    zipOut.setMethod(ZipOutputStream.DEFLATED)

    rootDir.walk().asSequence()
        .filter { it != rootDir }
        .filter { file ->
            val baseDirName = file.relativeTo(rootDir).path.substringBefore(File.separator)
            if (baseDirName.startsWith('.')) return@filter false
            baseDirName in included
        }
        .forEach { file ->
            val relativePath = file.relativeTo(rootDir).invariantSeparatorsPath
            if (file.isDirectory) {
                if (file.listFiles()?.isNotEmpty() == true) {
                    zipOut.putNextEntry(ZipEntry("$relativePath/"))
                    zipOut.closeEntry()
                }
                return@forEach
            }
            zipOut.putNextEntry(ZipEntry(relativePath))
            file.inputStream().use { input ->
                input.copyTo(zipOut)
            }
            zipOut.closeEntry()
        }
}