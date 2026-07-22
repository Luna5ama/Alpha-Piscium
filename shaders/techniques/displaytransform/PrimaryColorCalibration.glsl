#ifndef INCLUDE_techniques_displaytransform_PrimaryColorCalibration_glsl
#define INCLUDE_techniques_displaytransform_PrimaryColorCalibration_glsl a

vec2 _displaytransform_primarycolorcalibration_XYZ2xy(vec3 XYZ) {
    float sum = XYZ.x + XYZ.y + XYZ.z;
    return XYZ.xy / sum;
}

vec3 _displaytransform_primarycolorcalibration_xy2XYZ(vec2 xy, float Y) {
    return vec3(xy.x * Y / xy.y, Y, (1.0 - xy.x - xy.y) * Y / xy.y);
}

vec2 _displaytransform_primarycolorcalibration_rotatePrimaryXy(vec2 primaryXy, vec2 whiteXy, float hueDeg, float satMult) {
    vec2 fromWhite = primaryXy - whiteXy;
    float angle = radians(hueDeg);
    float cosA = cos(angle);
    float sinA = sin(angle);
    fromWhite = vec2(fromWhite.x * cosA - fromWhite.y * sinA, fromWhite.x * sinA + fromWhite.y * cosA) * satMult;
    return whiteXy + fromWhite;
}

vec3 displaytransform_primarycolorcalibration_apply(vec3 color) {
    vec3 srgbColor = colors2_colorspaces_convert(COLORS2_WORKING_COLORSPACE, COLORS2_COLORSPACES_SRGB, color);
    vec3 whiteXYZ = colors2_colorspaces_convert(COLORS2_COLORSPACES_SRGB, COLORS2_COLORSPACES_CIE_XYZ, vec3(1.0));
    vec3 redXYZ = colors2_colorspaces_convert(COLORS2_COLORSPACES_SRGB, COLORS2_COLORSPACES_CIE_XYZ, vec3(1.0, 0.0, 0.0));
    vec3 greenXYZ = colors2_colorspaces_convert(COLORS2_COLORSPACES_SRGB, COLORS2_COLORSPACES_CIE_XYZ, vec3(0.0, 1.0, 0.0));
    vec3 blueXYZ = colors2_colorspaces_convert(COLORS2_COLORSPACES_SRGB, COLORS2_COLORSPACES_CIE_XYZ, vec3(0.0, 0.0, 1.0));
    vec2 whiteXy = _displaytransform_primarycolorcalibration_XYZ2xy(whiteXYZ);
    const float HUE_RANGE_DEG = 25.0;
    vec3 rXYZ = _displaytransform_primarycolorcalibration_xy2XYZ(_displaytransform_primarycolorcalibration_rotatePrimaryXy(_displaytransform_primarycolorcalibration_XYZ2xy(redXYZ), whiteXy, SETTING_COLOR_CALIBRATION_RED_HUE * 0.01 * HUE_RANGE_DEG, 1.0 + SETTING_COLOR_CALIBRATION_RED_SAT * 0.01), redXYZ.y);
    vec3 gXYZ = _displaytransform_primarycolorcalibration_xy2XYZ(_displaytransform_primarycolorcalibration_rotatePrimaryXy(_displaytransform_primarycolorcalibration_XYZ2xy(greenXYZ), whiteXy, SETTING_COLOR_CALIBRATION_GREEN_HUE * 0.01 * HUE_RANGE_DEG, 1.0 + SETTING_COLOR_CALIBRATION_GREEN_SAT * 0.01), greenXYZ.y);
    vec3 bXYZ = _displaytransform_primarycolorcalibration_xy2XYZ(_displaytransform_primarycolorcalibration_rotatePrimaryXy(_displaytransform_primarycolorcalibration_XYZ2xy(blueXYZ), whiteXy, SETTING_COLOR_CALIBRATION_BLUE_HUE * 0.01 * HUE_RANGE_DEG, 1.0 + SETTING_COLOR_CALIBRATION_BLUE_SAT * 0.01), blueXYZ.y);
    vec3 whiteCorrection = (whiteXYZ - rXYZ - gXYZ - bXYZ) / 3.0;
    mat3 calibMatrix = mat3(rXYZ + whiteCorrection, gXYZ + whiteCorrection, bXYZ + whiteCorrection);
    vec3 calibratedXYZ = calibMatrix * srgbColor;
    vec3 calibrated = colors2_colorspaces_convert(COLORS2_COLORSPACES_CIE_XYZ, COLORS2_COLORSPACES_SRGB, calibratedXYZ);
    return colors2_colorspaces_convert(COLORS2_COLORSPACES_SRGB, COLORS2_WORKING_COLORSPACE, calibrated);
}

#endif
