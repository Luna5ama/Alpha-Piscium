import kotlin.io.path.*

val dataDir = Path("../data")
val opacDataDir = dataDir.resolve("opac_raw")
val outputDir = dataDir.resolve("opac")
outputDir.createDirectories()
val spacesRegex = """\s+""".toRegex()
val scientificRegex = """E([+-\\d]+)""".toRegex()

opacDataDir.useDirectoryEntries { entries ->
    entries.filter { it.isRegularFile() && it.extension == "txt" }
        .forEach { filepath ->
            val textLines = filepath.readLines()
            val opticalParameterIndex = textLines.indexOf("# optical parameters:")

            val name = filepath.nameWithoutExtension
            outputDir.resolve("${name}_optical_parameters.tsv").bufferedWriter().use { writer ->
                writer.appendLine(
                    textLines[opticalParameterIndex + 3]
                        .substring(2)
                        .replace(spacesRegex, "\t")
                )

                textLines.asSequence()
                    .drop(opticalParameterIndex + 6)
                    .takeWhile { it.length > 1 }
                    .map { it.substring(3) }
                    .map { it.split(spacesRegex) }
                    .filter { it.first().toDouble() < 1.2 }
                    .map { row ->
                        row.map { element ->
                            element.replace(scientificRegex) {
                                "\\times10^{${it.groupValues[1]}}"
                            }
                        }
                    }
                    .forEach {
                        writer.appendLine(it.joinToString("\t"))
                    }
            }
        }
}
