import java.nio.file.Path
import java.util.*
import java.util.zip.Deflater
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.io.path.*

fun makeZip(zipFilePath: Path) {
    val java = System.getProperty("java.home")

    val currDirPath = Path("").absolute()
    val projectRootPath = currDirPath.parent
    val shadesmithJarPath = currDirPath.resolve("shadesmith.jar")
    val shadesmithAotCachePath = currDirPath.resolve("shadesmith.aot")

    val shadersPath = projectRootPath.resolve("shaders")

    val config = Properties().apply {
        runCatching {
            Path("config.properties").inputStream().use {
                load(it)
            }
        }
    }

    val shdesmithOutputPathStr = config.getOrDefault("SHADESMITH_OUTPUT", "./shadesmitth").toString()
    val shadesmithOutputPath = Path(shdesmithOutputPathStr).normalize().absolute()
    val shadesmithShadersPath = shadesmithOutputPath.resolve("shaders")

    if (!shadesmithAotCachePath.exists()) {
        println("Shadesmith AOT cache not found, running once to generate it...")
        ProcessBuilder()
            .command(
                "$java/bin/java",
                "--add-modules",
                "jdk.internal.vm.ci",
                "-XX:AOTCacheOutput=${shadesmithAotCachePath}",
                "-jar",
                shadesmithJarPath.toString(),
                shadersPath.toString(),
                shadesmithShadersPath.toString()
            )
            .inheritIO()
            .start()
            .waitFor()
    }


    val shadesmithRun = ProcessBuilder()
        .command(
            "$java/bin/java",
            "-jar",
            "-XX:AOTCache=${shadesmithAotCachePath}",
            shadesmithJarPath.toString(),
            shadersPath.toString(),
            shadesmithShadersPath.toString()
        )
        .inheritIO()
        .start()

    val included = setOf(
        "changelogs",
        "licenses",
        "shaders/lang",
        "shaders/textures",
        "shaders",
        "LICENSE",
        "README.md"
    )

    ZipOutputStream(zipFilePath.outputStream(), Charsets.UTF_8).use { zipOut ->
        zipOut.setLevel(Deflater.DEFAULT_COMPRESSION)
        zipOut.setMethod(ZipOutputStream.DEFLATED)

        val added = mutableSetOf<String>()

        fun addStuff(rootDir: Path, sequence: Sequence<Path>) {
            sequence
                .filter { it != rootDir }
                .forEach { file ->
                    val relativePath = file.relativeTo(rootDir).invariantSeparatorsPathString
                    if (!added.add(relativePath)) return@forEach
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

        shadesmithRun.waitFor()
        addStuff(shadesmithOutputPath, shadesmithShadersPath.walk())
        addStuff(projectRootPath, projectRootPath.walk(PathWalkOption.FOLLOW_LINKS).filter { file ->
            val baseDirName = file
                .relativeTo(projectRootPath)
                .invariantSeparatorsPathString
                .substringBeforeLast('/')
            if (baseDirName.contains('.')) return@filter false
            baseDirName in included
        })
    }
}