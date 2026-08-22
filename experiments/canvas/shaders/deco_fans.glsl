// style: deco_fans
// title: Deco fan tiles
// description: A brick-laid grid of square tiles, each holding a quarter-circle fan of concentric bands anchored at one of its corners, in lacquer black, cream, brass and jade with fine dark rules.
// tags: art-deco, fan, tiling, brick-bond, geometric, ornament
// brief: constructed geometric ornament; the seed picks each tile's anchor corner, band colours and motif
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const float T = 6.0;                                  // tiles across
const float RINGS = 4.0;
const float PI = 3.14159265;

const vec3 INK   = vec3(0.075, 0.066, 0.062);         // lacquer black
const vec3 CREAM = vec3(0.937, 0.902, 0.827);
const vec3 BRASS = vec3(0.796, 0.639, 0.204);
const vec3 JADE  = vec3(0.118, 0.435, 0.365);
const vec3 ROSE  = vec3(0.792, 0.435, 0.353);
const vec3 NAVY  = vec3(0.098, 0.208, 0.290);

float h11(float x) { return fract(sin(x * 127.1 + 11.3) * 43758.5453123); }
float h21(vec2 p)  { return fract(sin(dot(floor(p), vec2(127.1, 311.7))) * 43758.5453123); }
float h22(vec2 p)  { return fract(sin(dot(floor(p), vec2(74.7, 219.3))) * 24634.6345); }

vec3 pal(float t) {
    if (t < 0.20) return CREAM;
    if (t < 0.40) return JADE;
    if (t < 0.58) return BRASS;
    if (t < 0.76) return ROSE;
    if (t < 0.90) return NAVY;
    return INK;
}

void main() {
    vec2 pp = uv * T;
    float px = fwidth(uv.x) * T + 1e-5;

    vec2 so = floor(vec2(h11(seed + 2.3), h11(seed * 5.7 + 4.1)) * 89.0);

    // brick bond: every other row shifts half a tile
    float row = floor(pp.y);
    pp.x += 0.5 * mod(row + so.y, 2.0);
    vec2 tile = floor(pp);
    vec2 q = fract(pp);

    vec2 ts = tile + so;
    float ha = h21(ts);
    float hb = h22(ts);
    float hc = h21(ts + vec2(31.0, 53.0));
    float hd = h22(ts + vec2(11.0, 7.0));

    // anchor corner
    vec2 qa = vec2(ha < 0.5 ? q.x : 1.0 - q.x, hb < 0.5 ? q.y : 1.0 - q.y);
    float r = length(qa);
    float ang = atan(qa.y, max(qa.x, 1e-4)) / (0.5 * PI);

    float a = r * RINGS;
    float k = floor(a);

    // per-tile band colours: two hues alternating, on a ground
    vec3 c1 = pal(hc);
    vec3 c2 = pal(fract(hc * 3.7 + 0.31));
    if (distance(c1, c2) < 0.15) c2 = BRASS;
    vec3 ground = hd < 0.34 ? INK : (hd < 0.68 ? CREAM : NAVY);

    vec3 col = ground;
    float inFan = smoothstep(px, -px, r - 1.0);
    vec3 band = mod(k, 2.0) < 0.5 ? c1 : c2;
    // metallic sheen sweeping across the quarter
    band *= 0.91 + 0.16 * sin(ang * PI * (1.0 + step(0.5, hd)));
    col = mix(col, band, inFan);

    // fine fluting in the negative corner, ruled along one axis per tile
    float flute = ha < 0.5 ? qa.x + qa.y : qa.x - qa.y;
    const float FP = 6.0;
    float fl = abs(fract(flute * FP) - 0.5) * 2.0;
    float fw = clamp(px * FP * 2.2, 0.02, 1.0);
    float flines = smoothstep(0.52 - fw, 0.52 + fw, fl) * (1.0 - inFan);
    float fade = smoothstep(0.55, 0.15, fw);
    col = mix(col, mix(col, INK, 0.50) + BRASS * 0.09, flines * 0.55 * fade);

    // ring rules and the fan's outer arc
    float dRing = abs(fract(a) - 0.5) / RINGS;          // distance to nearest ring edge, in tile units
    float rules = smoothstep(0.011 + px, 0.011 - px, abs(dRing - 0.5 / RINGS));
    col = mix(col, INK, rules * inFan * 0.9);
    float outer = smoothstep(0.014 + px, 0.014 - px, abs(r - 1.0));
    col = mix(col, INK, outer);

    // second motif: radial spokes over the fan
    if (hd > 0.72) {
        float sp = abs(fract(ang * 5.0) - 0.5) * 2.0;
        float spk = smoothstep(0.86, 0.94, sp) * smoothstep(px, -px, r - 0.98) * smoothstep(0.10, 0.22, r);
        col = mix(col, BRASS, spk * 0.85);
    }
    // a brass pip at the anchor
    float pip = length(qa) - 0.13;
    col = mix(col, BRASS, smoothstep(px, -px, pip));
    col = mix(col, INK, smoothstep(0.012 + px, 0.012 - px, abs(pip)));

    // tile grid rules
    vec2 e = min(q, 1.0 - q);
    float grid = smoothstep(0.016 + px, 0.016 - px, min(e.x, e.y));
    col = mix(col, mix(INK, BRASS, 0.25), grid * 0.85);

    // fine grain
    float grain = fract(sin(dot(uv * 943.0, vec2(12.99, 78.23))) * 43758.545) - 0.5;
    col += grain * 0.022;

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
