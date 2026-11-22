// OpenColorIO 2.6.0 generated from Blender 5.0.0, sRGB / AgX.

// Declaration of all textures

uniform sampler3D usam_agx;

// Declaration of the OCIO shader function

vec4 blenderAgX(vec4 inPixel)
{
  vec4 outColor = inPixel;

  // Add Matrix processing

  {
    vec4 res = vec4(outColor.rgb.r, outColor.rgb.g, outColor.rgb.b, outColor.a);
    vec4 tmp = res;
    res = mat4(1.3242029294284472, -0.027742099703214177, 0.10790395802043705, 0., -0.23679925241972027, 0.97444442615979698, 0.033780806885502751, 0., -0.087403676492011903, 0.053297664728828048, 0.85831518053479061, 0., 0., 0., 0., 1.) * tmp;
    outColor.rgb = vec3(res.x, res.y, res.z);
    outColor.a = res.w;
  }

  // Add Log processing

  {
    outColor.rgb = max( vec3(1.17549435e-38, 1.17549435e-38, 1.17549435e-38), outColor.rgb);
    outColor.rgb = log2(outColor.rgb);
  }

  // Add Matrix processing

  {
    vec4 res = vec4(outColor.rgb.r, outColor.rgb.g, outColor.rgb.b, outColor.a);
    res = vec4(0.0400000028, 0.0400000028, 0.0400000028, 1.) * res;
    res = vec4(0.498957247, 0.498957247, 0.498957247, 0.) + res;
    outColor.rgb = vec3(res.x, res.y, res.z);
    outColor.a = res.w;
  }

  // Add LUT 3D processing for ocio_lut3d_0

  {
    vec3 coords = outColor.rgb * vec3(56., 56., 56.);
    vec3 baseInd = floor(coords);
    vec3 frac = coords - baseInd;
    vec3 f1, f4;
    baseInd = ( baseInd.zyx + vec3(0.5, 0.5, 0.5) ) / vec3(57., 57., 57.);
    vec3 v1 = texelFetch(usam_agx, clamp(ivec3(floor(baseInd * 57.0)), ivec3(0), ivec3(56)), 0).rgb;
    vec3 nextInd = baseInd + vec3(0.0175438598, 0.0175438598, 0.0175438598);
    vec3 v4 = texelFetch(usam_agx, clamp(ivec3(floor(nextInd * 57.0)), ivec3(0), ivec3(56)), 0).rgb;
    if (frac.r >= frac.g)
    {
      if (frac.g >= frac.b)
      {
        nextInd = baseInd + vec3(0., 0., 0.0175438598);
        vec3 v2 = texelFetch(usam_agx, clamp(ivec3(floor(nextInd * 57.0)), ivec3(0), ivec3(56)), 0).rgb;
        nextInd = baseInd + vec3(0., 0.0175438598, 0.0175438598);
        vec3 v3 = texelFetch(usam_agx, clamp(ivec3(floor(nextInd * 57.0)), ivec3(0), ivec3(56)), 0).rgb;
        f1 = vec3(1. - frac.r, 1. - frac.r, 1. - frac.r);
        f4 = vec3(frac.b, frac.b, frac.b);
        vec3 f2 = vec3(frac.r - frac.g, frac.r - frac.g, frac.r - frac.g);
        vec3 f3 = vec3(frac.g - frac.b, frac.g - frac.b, frac.g - frac.b);
        outColor.rgb = (f2 * v2) + (f3 * v3);
      }
      else if (frac.r >= frac.b)
      {
        nextInd = baseInd + vec3(0., 0., 0.0175438598);
        vec3 v2 = texelFetch(usam_agx, clamp(ivec3(floor(nextInd * 57.0)), ivec3(0), ivec3(56)), 0).rgb;
        nextInd = baseInd + vec3(0.0175438598, 0., 0.0175438598);
        vec3 v3 = texelFetch(usam_agx, clamp(ivec3(floor(nextInd * 57.0)), ivec3(0), ivec3(56)), 0).rgb;
        f1 = vec3(1. - frac.r, 1. - frac.r, 1. - frac.r);
        f4 = vec3(frac.g, frac.g, frac.g);
        vec3 f2 = vec3(frac.r - frac.b, frac.r - frac.b, frac.r - frac.b);
        vec3 f3 = vec3(frac.b - frac.g, frac.b - frac.g, frac.b - frac.g);
        outColor.rgb = (f2 * v2) + (f3 * v3);
      }
      else
      {
        nextInd = baseInd + vec3(0.0175438598, 0., 0.);
        vec3 v2 = texelFetch(usam_agx, clamp(ivec3(floor(nextInd * 57.0)), ivec3(0), ivec3(56)), 0).rgb;
        nextInd = baseInd + vec3(0.0175438598, 0., 0.0175438598);
        vec3 v3 = texelFetch(usam_agx, clamp(ivec3(floor(nextInd * 57.0)), ivec3(0), ivec3(56)), 0).rgb;
        f1 = vec3(1. - frac.b, 1. - frac.b, 1. - frac.b);
        f4 = vec3(frac.g, frac.g, frac.g);
        vec3 f2 = vec3(frac.b - frac.r, frac.b - frac.r, frac.b - frac.r);
        vec3 f3 = vec3(frac.r - frac.g, frac.r - frac.g, frac.r - frac.g);
        outColor.rgb = (f2 * v2) + (f3 * v3);
      }
    }
    else
    {
      if (frac.g <= frac.b)
      {
        nextInd = baseInd + vec3(0.0175438598, 0., 0.);
        vec3 v2 = texelFetch(usam_agx, clamp(ivec3(floor(nextInd * 57.0)), ivec3(0), ivec3(56)), 0).rgb;
        nextInd = baseInd + vec3(0.0175438598, 0.0175438598, 0.);
        vec3 v3 = texelFetch(usam_agx, clamp(ivec3(floor(nextInd * 57.0)), ivec3(0), ivec3(56)), 0).rgb;
        f1 = vec3(1. - frac.b, 1. - frac.b, 1. - frac.b);
        f4 = vec3(frac.r, frac.r, frac.r);
        vec3 f2 = vec3(frac.b - frac.g, frac.b - frac.g, frac.b - frac.g);
        vec3 f3 = vec3(frac.g - frac.r, frac.g - frac.r, frac.g - frac.r);
        outColor.rgb = (f2 * v2) + (f3 * v3);
      }
      else if (frac.r >= frac.b)
      {
        nextInd = baseInd + vec3(0., 0.0175438598, 0.);
        vec3 v2 = texelFetch(usam_agx, clamp(ivec3(floor(nextInd * 57.0)), ivec3(0), ivec3(56)), 0).rgb;
        nextInd = baseInd + vec3(0., 0.0175438598, 0.0175438598);
        vec3 v3 = texelFetch(usam_agx, clamp(ivec3(floor(nextInd * 57.0)), ivec3(0), ivec3(56)), 0).rgb;
        f1 = vec3(1. - frac.g, 1. - frac.g, 1. - frac.g);
        f4 = vec3(frac.b, frac.b, frac.b);
        vec3 f2 = vec3(frac.g - frac.r, frac.g - frac.r, frac.g - frac.r);
        vec3 f3 = vec3(frac.r - frac.b, frac.r - frac.b, frac.r - frac.b);
        outColor.rgb = (f2 * v2) + (f3 * v3);
      }
      else
      {
        nextInd = baseInd + vec3(0., 0.0175438598, 0.);
        vec3 v2 = texelFetch(usam_agx, clamp(ivec3(floor(nextInd * 57.0)), ivec3(0), ivec3(56)), 0).rgb;
        nextInd = baseInd + vec3(0.0175438598, 0.0175438598, 0.);
        vec3 v3 = texelFetch(usam_agx, clamp(ivec3(floor(nextInd * 57.0)), ivec3(0), ivec3(56)), 0).rgb;
        f1 = vec3(1. - frac.g, 1. - frac.g, 1. - frac.g);
        f4 = vec3(frac.r, frac.r, frac.r);
        vec3 f2 = vec3(frac.g - frac.b, frac.g - frac.b, frac.g - frac.b);
        vec3 f3 = vec3(frac.b - frac.r, frac.b - frac.r, frac.b - frac.r);
        outColor.rgb = (f2 * v2) + (f3 * v3);
      }
    }
    outColor.rgb = outColor.rgb + (f1 * v1) + (f4 * v4);
  }

  // Add Gamma 'basicFwd' processing

  {
    vec4 gamma = vec4(2.3999999999999999, 2.3999999999999999, 2.3999999999999999, 1.);
    vec4 res = pow( max( vec4(0., 0., 0., 0.), outColor ), gamma );
    outColor.rgb = vec3(res.x, res.y, res.z);
    outColor.a = res.w;
  }

  // Add Gamma 'monCurveRev' processing

  {
    vec4 breakPnt = vec4(0.00303993467, 0.00303993467, 0.00303993467, 1.);
    vec4 slope = vec4(12.9232101, 12.9232101, 12.9232101, 1.);
    vec4 scale = vec4(1.05499995, 1.05499995, 1.05499995, 1.00000095);
    vec4 offset = vec4(0.0549999997, 0.0549999997, 0.0549999997, 9.99999997e-07);
    vec4 gamma = vec4(0.416666657, 0.416666657, 0.416666657, 0.999998987);
    vec4 isAboveBreak = vec4(greaterThan( outColor, breakPnt));
    vec4 linSeg = outColor * slope;
    vec4 powSeg = pow( max( vec4(0., 0., 0., 0.), outColor ), gamma ) * scale - offset;
    vec4 res = isAboveBreak * powSeg + ( vec4(1., 1., 1., 1.) - isAboveBreak ) * linSeg;
    outColor.rgb = vec3(res.x, res.y, res.z);
    outColor.a = res.w;
  }

  return outColor;
}
