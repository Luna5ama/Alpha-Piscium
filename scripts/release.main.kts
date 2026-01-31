#!/usr/bin/env kotlin

@file:Repository("https://repo.maven.apache.org/maven2/")
@file:DependsOn("com.squareup.okhttp3:okhttp:4.12.0")
@file:DependsOn("org.json:json:20231013")
@file:OptIn(ExperimentalPathApi::class)

import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.util.Properties
import java.util.zip.Deflater
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.io.path.ExperimentalPathApi
import kotlin.io.path.Path
import kotlin.io.path.PathWalkOption
import kotlin.io.path.absolute
import kotlin.io.path.extension
import kotlin.io.path.inputStream
import kotlin.io.path.invariantSeparatorsPathString
import kotlin.io.path.isDirectory
import kotlin.io.path.listDirectoryEntries
import kotlin.io.path.name
import kotlin.io.path.outputStream
import kotlin.io.path.relativeTo
import kotlin.io.path.walk
import kotlin.system.exitProcess

// Check if version argument is provided
if (args.isEmpty()) {
    println("Usage: kotlin release.main.kts <version> [-1] [-2] ...")
    println("Example: kotlin release.main.kts 1.7.2 -1 -2")
    exitProcess(1)
}

val version = args[0]
val skippedSteps = args.drop(1)
    .filter { it.startsWith("-") }
    .mapNotNull { it.replace("-", "").toIntOrNull() }
    .toSet()

if (skippedSteps.isNotEmpty()) {
    println("Skipping steps: ${skippedSteps.sorted().joinToString(", ")}")
}

val isBeta = version.contains("beta", true)
val rootDir = File("").absoluteFile.parentFile
val buildDir = File(rootDir, "builds")
val changelogFile = File(rootDir, "changelogs/$version.md")

// Validate changelog exists
if (!changelogFile.exists()) {
    println("Error: Changelog file not found at ${changelogFile.absolutePath}")
    exitProcess(1)
}

// Read changelog content
val changelogFullContent = changelogFile.readText().trim()
// Remove the first line (header) for GitHub/Modrinth releases
val changelogContent = changelogFullContent.lines().drop(1).joinToString("\n").trim()

// Read tokens from tokens.properties file
val tokensFile = File(File("").absoluteFile, "tokens.properties")
if (!tokensFile.exists()) {
    println("Error: tokens.properties file not found at ${tokensFile.absolutePath}")
    exitProcess(1)
}

val tokens = Properties().apply {
    tokensFile.inputStream().use { load(it) }
}

val githubToken = tokens.getProperty("GITHUB_TOKEN")
if (githubToken.isNullOrBlank()) {
    println("Error: GITHUB_TOKEN not found in tokens.properties")
    exitProcess(1)
}

val modrinthToken = tokens.getProperty("MODRINTH_TOKEN")
if (modrinthToken.isNullOrBlank()) {
    println("Error: MODRINTH_TOKEN not found in tokens.properties")
    exitProcess(1)
}

val curseForgeToken = tokens.getProperty("CURSEFORGE_TOKEN")
if (curseForgeToken.isNullOrBlank()) {
    println("Error: CURSEFORGE_TOKEN not found in tokens.properties")
    exitProcess(1)
}

val curseForgeProjectId = tokens.getProperty("CURSEFORGE_PROJECT_ID")
if (curseForgeProjectId.isNullOrBlank()) {
    println("Error: CURSEFORGE_PROJECT_ID not found in tokens.properties")
    exitProcess(1)
}

val client = OkHttpClient()


println("=== Starting Release Process for v$version ===\n")

var githubReleaseUrl = "Skipped"
var modrinthReleaseUrl = "Skipped"
var curseForgeReleaseUrl = "Skipped"

// Pre-define file paths
val versionZipName = "Alpha-Piscium_v$version.zip"
val versionZip = File(buildDir, versionZipName)
val spaceVersionZipName = "Alpha Piscium v$version.zip"
val spaceVersionZip = File(buildDir, spaceVersionZipName)

// Step 1: Create zip file (inline version of make-zip.main.kts)
var gitHashZip: File? = null
if (1 !in skippedSteps) {
    gitHashZip = run {
        println("Step 1: Creating zip file...")
        // Create the zip file
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

        println("Creating $zipFileName")

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
            addStuff(shadesmithOutputPath, shadesmithOutputPath.walk(PathWalkOption.FOLLOW_LINKS))
            zipFilePath.toFile()
        }
    }

    println("Zip file created successfully")
} else {
    println("Step 1: Skipped")
}

// Step 2: Rename to version format
if (2 !in skippedSteps) {
    println("\nStep 2: Renaming to version format...")
    if (gitHashZip == null || !gitHashZip!!.exists()) {
        println("Warning: Source zip from Step 1 not found. Skipping rename if you skipped Step 1.")
    } else {
        Files.move(gitHashZip!!.toPath(), versionZip.toPath(), StandardCopyOption.REPLACE_EXISTING)
        println("Renamed to: $versionZipName")
    }
} else {
    println("Step 2: Skipped")
}

// Step 3: Create and push git tag
if (3 !in skippedSteps) {
    println("\nStep 3: Creating and pushing git tag...")
    val githubReleaseTag = "v$version"
    val targetBranch = if (isBeta) "dev" else "main"

    // Get current branch
    val currentBranch = Runtime.getRuntime()
        .exec(arrayOf("git", "rev-parse", "--abbrev-ref", "HEAD"))
        .inputStream.bufferedReader().readText().trim()

    println("Current branch: $currentBranch")
    println("Target branch for tag: $targetBranch")

    // Checkout target branch if needed
    if (currentBranch != targetBranch) {
        println("Checking out $targetBranch branch...")
        val checkoutProcess = Runtime.getRuntime().exec(arrayOf("git", "checkout", targetBranch))
        checkoutProcess.waitFor()
        if (checkoutProcess.exitValue() != 0) {
            println("Error: Failed to checkout $targetBranch branch")
            exitProcess(1)
        }
    }

    // Create the tag
    println("Creating tag $githubReleaseTag on $targetBranch branch...")
    val createTagProcess = Runtime.getRuntime().exec(arrayOf("git", "tag", "-a", githubReleaseTag, "-m", "Alpha Piscium v$version"))
    createTagProcess.waitFor()
    if (createTagProcess.exitValue() != 0) {
        println("Error: Failed to create tag")
        exitProcess(1)
    }

    // Push the tag
    println("Pushing tag to remote...")
    val pushTagProcess = Runtime.getRuntime().exec(arrayOf("git", "push", "origin", githubReleaseTag))
    pushTagProcess.waitFor()
    if (pushTagProcess.exitValue() != 0) {
        println("Error: Failed to push tag")
        exitProcess(1)
    }

    println("Tag $githubReleaseTag created and pushed successfully")

    // Checkout back to original branch if needed
    if (currentBranch != targetBranch) {
        println("Checking out back to $currentBranch branch...")
        val checkoutBackProcess = Runtime.getRuntime().exec(arrayOf("git", "checkout", currentBranch))
        checkoutBackProcess.waitFor()
    }
} else {
    println("Step 3: Skipped")
}

// Step 4: Create GitHub Release
if (4 !in skippedSteps) {
    println("\nStep 4: Creating GitHub release...")
    val githubReleaseTag = "v$version"
    val targetBranch = if (isBeta) "dev" else "main"
    val githubReleaseName = "Alpha Piscium v$version"

    // Create the release
    val createReleaseJson = JSONObject().apply {
        put("tag_name", githubReleaseTag)
        put("name", githubReleaseName)
        put("body", changelogContent)
        put("draft", false)
        put("prerelease", isBeta)
        put("target_commitish", targetBranch)
    }

    val createReleaseRequest = Request.Builder()
        .url("https://api.github.com/repos/Luna5ama/Alpha-Piscium/releases")
        .header("Authorization", "Bearer $githubToken")
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .post(createReleaseJson.toString().toRequestBody("application/json".toMediaType()))
        .build()

    val createReleaseResponse = client.newCall(createReleaseRequest).execute()
    if (!createReleaseResponse.isSuccessful) {
        println("Error creating GitHub release: ${createReleaseResponse.code} - ${createReleaseResponse.body?.string()}")
        exitProcess(1)
    }

    val releaseJson = JSONObject(createReleaseResponse.body?.string() ?: "{}")
    val releaseId = releaseJson.getLong("id")
    val uploadUrl = releaseJson.getString("upload_url").substringBefore("{")
    githubReleaseUrl = releaseJson.getString("html_url")

    println("GitHub release created: $githubReleaseUrl")

    // Upload asset to GitHub release
    println("Uploading asset to GitHub release...")
    val assetUploadRequest = Request.Builder()
        .url("$uploadUrl?name=$versionZipName")
        .header("Authorization", "Bearer $githubToken")
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .header("Content-Type", "application/zip")
        .post(versionZip.asRequestBody("application/zip".toMediaType()))
        .build()

    val assetUploadResponse = client.newCall(assetUploadRequest).execute()
    if (!assetUploadResponse.isSuccessful) {
        println("Warning: Failed to upload asset to GitHub: ${assetUploadResponse.code} - ${assetUploadResponse.body?.string()}")
    } else {
        println("Asset uploaded successfully")
    }

    // Rename back to version with spaces
    println("Renaming back to original format with spaces...")
    if (versionZip.exists()) {
        Files.move(versionZip.toPath(), spaceVersionZip.toPath(), StandardCopyOption.REPLACE_EXISTING)
        println("Renamed to: $spaceVersionZipName")
    } else {
        println("Warning: $versionZipName not found to rename.")
    }
} else {
    println("Step 4: Skipped")
}

// Prepare supported versions (used in Step 5 and Step 6)
val supportedVersions = mutableListOf<String>()
if (5 !in skippedSteps || 6 !in skippedSteps) {
    println("\nFetching supported game versions from Modrinth API...")
    try {
        val getVersionsRequest = Request.Builder()
            .url("https://api.modrinth.com/v2/tag/game_version")
            .get()
            .build()

        val getVersionsResponse = client.newCall(getVersionsRequest).execute()
        val versionsArray = JSONArray(getVersionsResponse.body?.string() ?: "[]")

        for (i in 0 until versionsArray.length()) {
            val versionObj = versionsArray.getJSONObject(i)
            val versionName = versionObj.getString("version")
            if (versionName.startsWith("1.21.") || versionName == "1.20.5" || versionName == "1.20.6") {
                supportedVersions.add(versionName)
            }
        }
        println("Supported Minecraft versions: ${supportedVersions.joinToString(", ")}")
    } catch (e: Exception) {
        println("Warning: Failed to fetch versions from Modrinth: ${e.message}")
    }
}

// Step 5: Create Modrinth Release
if (5 !in skippedSteps) {
    println("\nStep 5: Creating Modrinth release...")

    // Get project ID first
    val projectSlug = "alpha-piscium"
    val getProjectRequest = Request.Builder()
        .url("https://api.modrinth.com/v2/project/$projectSlug")
        .header("Authorization", modrinthToken)
        .get()
        .build()

    val getProjectResponse = client.newCall(getProjectRequest).execute()
    if (!getProjectResponse.isSuccessful) {
        println("Error getting Modrinth project: ${getProjectResponse.code} - ${getProjectResponse.body?.string()}")
        exitProcess(1)
    }

    val projectJson = JSONObject(getProjectResponse.body?.string() ?: "{}")
    val projectId = projectJson.getString("id")


    // Create multipart form data for Modrinth
    val modrinthVersionData = JSONObject().apply {
        put("project_id", projectId)
        put("version_number", version)
        put("version_title", "Alpha Piscium v$version")
        put("changelog", changelogContent)
        put("version_type", if (isBeta) "beta" else "release")
        put("loaders", JSONArray(listOf("iris")))
        put("game_versions", JSONArray(supportedVersions))
        put("dependencies", JSONArray())
        put("featured", true)
        put("file_parts", JSONArray(listOf("file")))
    }

    val modrinthRequestBody = MultipartBody.Builder()
        .setType(MultipartBody.FORM)
        .addFormDataPart("data", modrinthVersionData.toString())
        .addFormDataPart(
            "file",
            spaceVersionZipName,
            spaceVersionZip.asRequestBody("application/zip".toMediaType())
        )
        .build()

    val createModrinthVersionRequest = Request.Builder()
        .url("https://api.modrinth.com/v2/version")
        .header("Authorization", modrinthToken)
        .post(modrinthRequestBody)
        .build()

    val createModrinthVersionResponse = client.newCall(createModrinthVersionRequest).execute()
    if (!createModrinthVersionResponse.isSuccessful) {
        println("Error creating Modrinth version: ${createModrinthVersionResponse.code} - ${createModrinthVersionResponse.body?.string()}")
        exitProcess(1)
    }

    modrinthReleaseUrl = "https://modrinth.com/shader/alpha-piscium/version/$version"
    println("Modrinth version created: $modrinthReleaseUrl")
} else {
    println("Step 5: Skipped")
}

// Step 6: Create CurseForge Release
if (6 !in skippedSteps) {
    println("\nStep 6: Creating CurseForge release...")

    // Get available versions from CurseForge
    println("Fetching CurseForge game versions...")
    val getCfVersionsRequest = Request.Builder()
        .url("https://minecraft.curseforge.com/api/game/versions")
        .header("X-Api-Token", curseForgeToken)
        .get()
        .build()

    val getCfVersionsResponse = client.newCall(getCfVersionsRequest).execute()
    if (!getCfVersionsResponse.isSuccessful) {
        println("Error getting CurseForge versions: ${getCfVersionsResponse.code} - ${getCfVersionsResponse.body?.string()}")
        exitProcess(1)
    }

    // Get version types to act as a filter (we only want "Minecraft x.y" types, not ModLoaders)
    println("Fetching CurseForge version types...")
    val getVersionTypesRequest = Request.Builder()
        .url("https://minecraft.curseforge.com/api/game/version-types")
        .header("X-Api-Token", curseForgeToken)
        .get()
        .build()

    val versionTypesMap = mutableMapOf<Int, String>()
    val getVersionTypesResponse = client.newCall(getVersionTypesRequest).execute()
    if (getVersionTypesResponse.isSuccessful) {
        val typesArray = JSONArray(getVersionTypesResponse.body?.string() ?: "[]")
        for (i in 0 until typesArray.length()) {
            val typeObj = typesArray.getJSONObject(i)
            versionTypesMap[typeObj.getInt("id")] = typeObj.getString("name")
        }
    } else {
        println("Warning: Failed to fetch version types. Proceeding without type filtering (risky).")
    }

    val cfVersionsArray = JSONArray(getCfVersionsResponse.body?.string() ?: "[]")
    val cfVersionIds = mutableListOf<Int>()

    for (i in 0 until cfVersionsArray.length()) {
        val versionObj = cfVersionsArray.getJSONObject(i)
        val versionName = versionObj.getString("name")
        val typeId = versionObj.optInt("gameVersionTypeID")
        val typeName = versionTypesMap[typeId] ?: "Unknown"

        if (supportedVersions.contains(versionName)) {
            // Filter out types that are likely ModLoaders or Java versions if the name matches
            // We expect types like "Minecraft 1.21", "Minecraft 1.20", etc.
            // valid if: typeName matches "Minecraft.*" and NOT "Forge"/"Fabric"/"Quilt"/"NeoForge"
            // Note: Sometimes the type name is just "Minecraft 1.20", which is good.

            val isLikelyLoader = typeName.contains("Forge", true) ||
                                 typeName.contains("Fabric", true) ||
                                 typeName.contains("Quilt", true) ||
                                 typeName.contains("NeoForge", true) ||
                                 typeName.contains("Snapshot", true) // Optional: restrict snapshots?

            if (versionTypesMap.isNotEmpty() && !typeName.startsWith("Minecraft", true)) {
                 // Not a Minecraft version type (e.g. "Java 17")
                 continue
            }

            if (isLikelyLoader) {
                // println("Skipping version $versionName (ID: ${versionObj.getInt("id")}) because type is '$typeName'")
                continue
            }

            cfVersionIds.add(versionObj.getInt("id"))
        }
    }

    if (cfVersionIds.isEmpty()) {
        println("Warning: No matching CurseForge version IDs found for: ${supportedVersions.joinToString(", ")}")
    } else {
        println("Mapped CurseForge version IDs: ${cfVersionIds.joinToString(", ")}")
    }

    val cfMetadata = JSONObject().apply {
        put("changelog", changelogContent)
        put("changelogType", "markdown")
        put("displayName", "Alpha Piscium v$version")
        put("gameVersions", JSONArray(cfVersionIds))
        put("releaseType", if (isBeta) "beta" else "release")
    }

    val cfRequestBody = MultipartBody.Builder()
        .setType(MultipartBody.FORM)
        .addFormDataPart("metadata", cfMetadata.toString())
        .addFormDataPart(
            "file",
            spaceVersionZipName,
            spaceVersionZip.asRequestBody("application/zip".toMediaType())
        )
        .build()

    val uploadCfFileRequest = Request.Builder()
        .url("https://minecraft.curseforge.com/api/projects/$curseForgeProjectId/upload-file")
        .header("X-Api-Token", curseForgeToken)
        .post(cfRequestBody)
        .build()

    val uploadCfFileResponse = client.newCall(uploadCfFileRequest).execute()
    if (!uploadCfFileResponse.isSuccessful) {
        println("Error uploading to CurseForge: ${uploadCfFileResponse.code} - ${uploadCfFileResponse.body?.string()}")
        exitProcess(1)
    }

    val cfResponseJson = JSONObject(uploadCfFileResponse.body?.string() ?: "{}")
    val cfFileId = cfResponseJson.getLong("id")
    curseForgeReleaseUrl = "https://www.curseforge.com/minecraft/shaders/alpha-piscium/files/$cfFileId"

    println("CurseForge release created: $curseForgeReleaseUrl")
} else {
    println("Step 6: Skipped")
}

// Step 7: Print Discord announcement
if (7 !in skippedSteps) {
    println("\n" + "=".repeat(60))
    println("DISCORD ANNOUNCEMENT")
    println("=".repeat(60))
    println()
    println(changelogFullContent)
    println()
    println(githubReleaseUrl)
    println(modrinthReleaseUrl)
    println(curseForgeReleaseUrl)
    println("@everyone")
    println()
    println("=".repeat(60))
    println("\n✅ Release process completed successfully!")
} else {
    println("Step 7: Skipped")
}
