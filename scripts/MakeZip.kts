import java.io.File
import java.util.zip.Deflater
import java.util.zip.ZipOutputStream

val excluded = setOf("scripts", ".*\\.zip").map { it.toRegex() }

val currDir = File("")
val rootDir = currDir.absoluteFile.parentFile
ZipOutputStream(File(rootDir, "${rootDir.name}.zip").outputStream(), Charsets.UTF_8).use { zipOut ->
    zipOut.setLevel(Deflater.BEST_COMPRESSION)
    zipOut.setMethod(ZipOutputStream.DEFLATED)

    rootDir.walk().asSequence()
        .filter { file ->
            if (file.isDirectory) return@filter false
            val baseDirName = file.relativeTo(rootDir).path.substringBefore(File.separator)
            if (baseDirName.startsWith('.')) return@filter false
            if (excluded.any { it.matches(baseDirName) }) return@filter false

            return@filter true
        }
        .forEach { file ->
            val relativePath = file.relativeTo(rootDir)
            zipOut.putNextEntry(java.util.zip.ZipEntry(relativePath.toString()))
            file.inputStream().use { input ->
                input.copyTo(zipOut)
            }
            zipOut.closeEntry()
        }
}