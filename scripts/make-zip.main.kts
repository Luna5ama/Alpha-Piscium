import java.nio.file.Path
import java.util.*
import java.util.zip.Deflater
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.io.path.*


val config = Properties().apply {
    runCatching {
        Path("config.properties").inputStream().use {
            load(it)
        }
    }
}

val currDirPath = Path("").absolute()
val projectRootPath = currDirPath.parent
val shadesmithJarPath = currDirPath.resolve("shadesmith-0.0.1-SNAPSHOT-fatjar-optimized.jar")
val shadersPath = projectRootPath.resolve("shaders")

val shdesmithOutputPathStr = config.getOrDefault("SHADESMITH_OUTPUT", "./shadesmitth").toString()
val shadesmithOutputPath = Path(shdesmithOutputPathStr).normalize().absolute()
val java = System.getProperty("java.home")

val shadesmithRun = ProcessBuilder()
    .command(
        "$java/bin/java",
        "-jar",
        shadesmithJarPath.toString(),
        shadersPath.toString(),
        shadesmithOutputPath.resolve("shaders").toString()
    )
    .inheritIO()
    .start()

val included = setOf("changelogs", "licenses", "shaders/lang", "shaders/textures", "LICENSE", "README.md")
val branchName =
    Runtime.getRuntime().exec(arrayOf("git", "rev-parse", "--abbrev-ref", "HEAD")).inputStream.bufferedReader()
        .readText().trim()
val commitTag =
    Runtime.getRuntime().exec(arrayOf("git", "rev-parse", "--short", "HEAD")).inputStream.bufferedReader().readText()
        .trim()
val zipFileName = "${projectRootPath.name.replace("-", " ")} $branchName $commitTag.zip"
val zipFilePath = projectRootPath.resolve("builds").resolve(zipFileName)

ZipOutputStream(zipFilePath.outputStream(), Charsets.UTF_8).use { zipOut ->
    zipOut.setLevel(Deflater.DEFAULT_COMPRESSION)
    zipOut.setMethod(ZipOutputStream.DEFLATED)

    fun addStuff(rootDir: Path, sequence: Sequence<Path>) {
        sequence
            .filter { it != rootDir }
            .forEach { file ->
                val relativePath = file.relativeTo(rootDir).invariantSeparatorsPathString
                if (file.isDirectory()) {
                    if (file.listDirectoryEntries().isNotEmpty()) {
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

    addStuff(projectRootPath, projectRootPath.walk(PathWalkOption.FOLLOW_LINKS).filter { file ->
        if (file.extension == "properties") return@filter true
        val baseDirName = file.relativeTo(projectRootPath).invariantSeparatorsPathString
        if (baseDirName.startsWith('.')) return@filter false
        included.any {
            baseDirName.startsWith(it)
        }
    })
    shadesmithRun.waitFor()
    addStuff(shadesmithOutputPath, shadesmithOutputPath.walk())
}