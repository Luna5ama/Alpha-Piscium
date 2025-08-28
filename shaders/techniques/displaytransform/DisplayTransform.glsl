#include "Exposure.glsl"
#include "DRT.glsl"

void displaytransform_init() {
    _displaytransform_exposure_init();
}

void displaytransform_apply(inout vec4 color) {
    _displaytransform_exposure_apply(color);
    _displaytransform_DRT_apply(color);
    _displaytransform_exposure_update(color);
}