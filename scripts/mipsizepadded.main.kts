var maxRatioX = 0.0
var maxRatioY = 0.0

for (xSize in 320..<65536) {
    var totalXSize = 0
    var mipsize = xSize / 2
    while (mipsize > 0) {
        totalXSize += (mipsize + 1)
        mipsize /= 2
    }
    maxRatioX = maxOf(maxRatioX, totalXSize.toDouble() / xSize.toDouble())
}

for (ySize in 240..<65536) {
    val totalYSize = (ySize + 1) + (ySize / 2 + 1) * 2
    maxRatioY = maxOf(maxRatioY, totalYSize.toDouble() / ySize.toDouble())
}

println("Max Ratios: X=$maxRatioX Y=$maxRatioY")