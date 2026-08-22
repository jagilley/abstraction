// style: tumbling_blocks
// title: Tumbling blocks
// description: The rhombille tiling read as a heap of isometric cubes in cut stone — bone, slate, terracotta and ochre — where some cubes turn inside out into hollow corners and a few collapse to a flat lozenge, all under fine incised joint lines.
// tags: isometric, rhombille, cubes, intarsia, tiling, geometric, illusion
// brief: constructed geometric ornament; the seed reassigns cube stones and flips which ones read as hollow or flat
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const float S = 8.0;                                   // cubes across the canvas
const float PI = 3.14159265;
const float TAU = 6.28318531;

const vec3 JOINT = vec3(0.157, 0.145, 0.137);

float h11(float x) { return fract(sin(x * 127.1 + 11.3) * 43758.5453123); }
float h21(vec2 p)  { return fract(sin(dot(floor(p + 0.5), vec2(127.1, 311.7))) * 43758.5453123); }
float h22(vec2 p)  { return fract(sin(dot(floor(p + 0.5), vec2(74.7, 219.3))) * 24634.6345); }
float h23(vec2 p)  { return fract(sin(dot(floor(p + 0.5), vec2(21.9, 97.7))) * 15731.743); }

vec3 stone(float i) {
    if (i < 0.5) return vec3(0.878, 0.851, 0.784);     // bone marble
    if (i < 1.5) return vec3(0.310, 0.341, 0.373);     // slate
    if (i < 2.5) return vec3(0.667, 0.337, 0.243);     // terracotta
    if (i < 3.5) return vec3(0.780, 0.620, 0.290);     // ochre
    if (i < 4.5) return vec3(0.318, 0.400, 0.361);     // verde
    return vec3(0.361, 0.243, 0.298);                  // aubergine
}

void main() {
    vec2 p = uv * S;
    float px = fwidth(uv.x) * S + 1e-5;
    vec2 so = floor(vec2(h11(seed + 1.1), h11(seed * 8.3 + 5.9)) * 71.0);

    vec2 r = vec2(1.0, 1.7320508);
    vec2 a = mod(p, r) - r * 0.5;
    vec2 b = mod(p - r * 0.5, r) - r * 0.5;
    float useA = dot(a, a) < dot(b, b) ? 1.0 : 0.0;
    vec2 gv = mix(b, a, useA);
    vec2 base = mix(floor((p - r * 0.5) / r), floor(p / r), useA);
    vec2 id = vec2(base.x * 2.0 + (1.0 - useA), base.y) + so;

    // which of the three rhombi of this hexagon
    float ang = atan(gv.y, gv.x);
    float sect = floor(mod(ang - PI / 6.0, TAU) / (TAU / 3.0));

    float hs = h21(id);
    float hf = h22(id);
    float hg = h23(id);

    // hollow cubes swap the two side faces; flat ones drop the shading altogether
    bool isHollow = hf > 0.74;
    bool isFlat = hg > 0.935;

    float lum;
    if (isFlat) lum = 1.02 - 0.14 * (gv.y + 0.5);
    else {
        float s3 = isHollow ? mod(sect + 1.0, 3.0) : sect;
        lum = s3 < 0.5 ? 1.14 : (s3 < 1.5 ? 0.60 : 0.84);
    }

    vec3 col = stone(floor(hs * 6.0)) * lum + vec3(0.045, 0.038, 0.030) * max(0.0, 1.0 - lum);

    // incised joints: the hexagon rim and the three rays that meet at its centre
    float hd = max(dot(abs(gv), vec2(0.5, 0.8660254)), abs(gv).x);
    float dRim = 0.5 - hd;
    float ia = mod(ang - PI / 2.0, TAU / 3.0);
    float dRay = sin(min(ia, TAU / 3.0 - ia)) * length(gv);
    float dJoint = min(dRim, isFlat ? dRim : dRay);

    float w = 0.026;
    float line = smoothstep(w + px, w - px, dJoint);
    col = mix(col, JOINT, line * 0.85);
    // a lit chamfer just inside every joint
    col *= 1.0 + 0.16 * smoothstep(0.075, w, dJoint) * (lum > 0.9 ? 1.0 : -0.6);

    // stone: speckle plus a broad vein wash
    float sp = fract(sin(dot(uv * 1451.0, vec2(12.99, 78.23))) * 43758.545) - 0.5;
    float sp2 = fract(sin(dot(floor(uv * 380.0), vec2(39.3, 11.1))) * 12345.678) - 0.5;
    col += (sp * 0.030 + sp2 * 0.026);
    col *= 1.0 + 0.05 * sin(uv.x * 11.0 + uv.y * 7.0) * sin(uv.y * 5.0 - 1.0);

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
