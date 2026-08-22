// style: sediment_strata
// title: Sediment strata
// description: A cut face of sedimentary rock — beds of bone, sand, clay and rust stacked and gently folded, each graded from a gritty base to a fine top, and truncated here and there by an old erosion surface.
// tags: rock, strata, layers, sediment, geology, horizontal, gritty, desert
// brief: organic and natural textures; the gritty, matte, horizontally banded one
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const vec3 BONE  = vec3(0.914, 0.859, 0.769);
const vec3 SAND  = vec3(0.812, 0.686, 0.482);
const vec3 CLAY  = vec3(0.647, 0.451, 0.302);
const vec3 RUST  = vec3(0.482, 0.278, 0.176);
const vec3 SHALE = vec3(0.286, 0.259, 0.239);
const vec3 ASH   = vec3(0.671, 0.647, 0.588);

float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash21(i), b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0)), d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbm(vec2 p) {
    float s = 0.0, a = 0.5;
    for (int k = 0; k < 4; k++) { s += a * vnoise(p); p = p * 2.03 + vec2(13.7, 2.9); a *= 0.5; }
    return s;
}

vec3 bed_colour(float h) {
    if (h < 0.20) return SHALE;
    if (h < 0.40) return CLAY;
    if (h < 0.62) return SAND;
    if (h < 0.78) return RUST;
    if (h < 0.92) return BONE;
    return ASH;
}

// one stack of beds: returns colour for point p given the package's tilt and phase
vec3 package_colour(vec2 p, float tilt, float phase, float dens) {
    float y = p.y + tilt * p.x
            + 0.075 * (fbm(vec2(p.x * 0.7, phase)) - 0.5)
            + 0.022 * (fbm(vec2(p.x * 3.3, phase + 5.0)) - 0.5);
    float t = y * (19.0 * dens) + phase * 7.0;
    t += 0.45 * sin(t * 0.53 + phase) + 0.20 * sin(t * 1.27 - phase);   // uneven bed thickness
    float bi = floor(t), bf = fract(t);

    float h = hash21(vec2(bi, floor(phase * 31.0)));
    vec3 col = bed_colour(h);

    // graded bedding: dark coarse base fining upward to a paler top
    col *= 0.80 + 0.30 * bf;
    // a thin parting at the top of each bed
    col *= 1.0 - 0.30 * smoothstep(0.90, 1.0, bf);

    // fine laminae inside the quieter beds
    float lam = 0.5 + 0.5 * sin(6.2832 * (t * 7.0));
    col *= 1.0 - 0.10 * lam * step(h, 0.62);

    // grain: coarse beds are speckly, fine beds are smooth
    float coarse = step(0.5, h) * (1.0 - bf);
    float grit = vnoise(vec2(p.x * 430.0, p.y * 430.0));
    col *= 0.93 + 0.16 * grit * (0.35 + 1.1 * coarse);
    // pebbles in the coarsest beds
    float pb = vnoise(vec2(p.x * 130.0, p.y * 290.0));
    col *= 1.0 - 0.22 * smoothstep(0.80, 0.95, pb) * coarse;
    return col;
}

void main() {
    vec2 so = vec2(hash21(vec2(seed, 7.7)), hash21(vec2(seed, 2.1)));
    vec2 p = uv + so * 37.0;

    // two erosion surfaces cut the outcrop into three packages with different dips
    float s1 = 0.63 + 0.09 * (fbm(vec2(p.x * 1.3, 3.0)) - 0.5) * 2.0;
    float s2 = 0.29 + 0.07 * (fbm(vec2(p.x * 1.7, 9.0)) - 0.5) * 2.0;

    vec3 a = package_colour(p, 0.030, so.x + 0.11, 1.00);
    vec3 b = package_colour(p, -0.075, so.y + 0.53, 1.35);
    vec3 c = package_colour(p, 0.010, so.x + 0.87, 0.78);

    vec3 col = c;
    col = mix(col, b, smoothstep(s2 - 0.004, s2 + 0.004, uv.y));
    col = mix(col, a, smoothstep(s1 - 0.004, s1 + 0.004, uv.y));
    // dark seam along each erosion surface
    col *= 1.0 - 0.35 * smoothstep(0.010, 0.0, abs(uv.y - s1));
    col *= 1.0 - 0.35 * smoothstep(0.010, 0.0, abs(uv.y - s2));

    // weathering stain washing down the face
    col *= 0.88 + 0.24 * fbm(vec2(p.x * 1.6, p.y * 0.5) + 27.0);
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
