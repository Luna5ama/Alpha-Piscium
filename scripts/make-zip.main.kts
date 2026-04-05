@file:Import("make-zip.kts")

import kotlin.io.path.Path
import kotlin.io.path.absolute
import kotlin.io.path.name

var versionArg = args.getOrNull(0)
val noCommitHash = "--no-commit-hash" in args
val version = if (versionArg == null) {
    ""
} else {
    " v$versionArg"
}

val currDirPath = Path("").absolute()
val projectRootPath = currDirPath.parent

val branchName =
    Runtime.getRuntime().exec(arrayOf("git", "rev-parse", "--abbrev-ref", "HEAD")).inputStream.bufferedReader()
        .readText().trim().takeIf { it != "main" && it != "dev" }
val commitTag =
    Runtime.getRuntime().exec(arrayOf("git", "rev-parse", "--short", "HEAD")).inputStream.bufferedReader().readText()
        .trim()
val suffix = if (noCommitHash) {
    listOf(version)
} else {
    listOfNotNull(version, commitTag, branchName)
}
val suffixStr = suffix.joinToString(" ")
val zipFileName = "${projectRootPath.name.replace("-", " ")}$suffixStr.zip"
val zipFilePath = projectRootPath.resolve("builds").resolve(zipFileName)

makeZip(zipFilePath)

println("Created zip file at: $zipFilePath")