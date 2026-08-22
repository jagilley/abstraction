// style: mycelium_mat
// title: Mycelium mat
// description: A mat of fungal threads spreading over dark forest loam — pale hyphae at three scales crossing and breaking off, brighter where they knot together, with spores caught in the litter.
// tags: mycelium, fungus, hyphae, threads, network, dark, loam, filaments
// brief: organic and natural textures; the dark, fine-threaded one
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const vec3 LOAM_A = vec3(0.106, 0.078, 0.055);  // wet leaf litter
const vec3 LOAM_B = vec3(0.208, 0.157, 0.106);  // dry loam
const vec3 LOAM_C = vec3(0.263, 0.208, 0.129);  // exposed grit
const vec3 HYPHA  = vec3(0.898, 0.859, 0.769);  // bone-white thread
const vec3 HYPHA2 = vec3(0.596, 0.545, 0.455);  // older, duller thread

float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
vec2 hash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
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
    for (int k = 0; k < 4; k++) { s += a * vnoise(p); p = p * 2.07 + vec2(11.9, 6.1); a *= 0.5; }
    return s;
}

// perpendicular distance to the nearest cell border: threads run along these borders
float edge_dist(vec2 p, float jit) {
    vec2 i = floor(p), f = p - i;
    vec2 mr = vec2(0.0);
    float md = 8.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 g = vec2(float(x), float(y));
            vec2 r = g + 0.5 + jit * (hash22(i + g) - 0.5) - f;
            float d = dot(r, r);
            if (d < md) { md = d; mr = r; }
        }
    }
    float me = 8.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 g = vec2(float(x), float(y));
            vec2 r = g + 0.5 + jit * (hash22(i + g) - 0.5) - f;
            vec2 dv = r - mr;
            if (dot(dv, dv) > 1e-5) me = min(me, dot(0.5 * (mr + r), normalize(dv)));
        }
    }
    return me;
}

void main() {
    vec2 so = vec2(hash21(vec2(seed, 5.1)), hash21(vec2(seed, 8.3))) * 47.0;
    vec2 p = uv + so;

    // damp loam underneath
    float m = fbm(p * 3.0);
    vec3 col = mix(LOAM_A, LOAM_B, smoothstep(0.32, 0.62, m));
    col = mix(col, LOAM_C, smoothstep(0.66, 0.86, m) * 0.8);
    col *= 0.86 + 0.28 * vnoise(p * 190.0);                 // grit
    col *= 0.80 + 0.40 * fbm(p * 1.1 + 17.0);               // shadowed hollows

    // three generations of hyphae, each warped its own way so the mat is not a lattice
    vec2 w1 = vec2(fbm(p * 1.9), fbm(p * 1.9 + 5.0)) - 0.5;
    vec2 w2 = vec2(fbm(p * 4.3 + 12.0), fbm(p * 4.3 + 27.0)) - 0.5;
    // a small high-frequency wiggle on top: hyphae wander, they do not run straight
    vec2 w3 = vec2(fbm(p * 11.0 + 41.0), fbm(p * 11.0 + 63.0)) - 0.5;

    float e1 = edge_dist((p + 0.40 * w1 + 0.030 * w3) * 5.5, 1.10);
    float e2 = edge_dist((p + 0.22 * w2 + 0.022 * w3) * 11.0 + 33.0, 1.10);
    float e3 = edge_dist((p + 0.10 * w1 + 0.08 * w2 + 0.012 * w3) * 23.0 + 71.0, 1.10);

    // threads break off: each generation fades in and out along its length
    float b1 = smoothstep(0.36, 0.60, fbm(p * 3.7 + 61.0));
    float b2 = smoothstep(0.34, 0.58, fbm(p * 6.1 + 83.0));
    float b3 = smoothstep(0.38, 0.64, fbm(p * 9.7 + 97.0));

    float t1 = smoothstep(0.030, 0.010, e1) * b1;
    float t2 = smoothstep(0.026, 0.009, e2) * b2 * 0.85;
    float t3 = smoothstep(0.024, 0.008, e3) * b3 * 0.6;

    float th = max(t1, max(t2, t3));
    float knot = t1 * t2 + t2 * t3 + t1 * t3;               // brighter where they cross

    col = mix(col, HYPHA2, th * 0.85);
    col = mix(col, HYPHA, th * th * 0.9);
    col += HYPHA * 0.35 * clamp(knot, 0.0, 1.0);
    col += 0.10 * th * vec3(0.9, 0.85, 0.7);                // faint bloom along the threads

    // spores in the litter: round, sparse specks
    vec2 sg = p * 70.0;
    vec2 si = floor(sg), sf = fract(sg);
    vec2 sc = 0.25 + 0.5 * hash22(si + 3.0);
    float spore = smoothstep(0.16, 0.05, length(sf - sc)) * step(0.86, hash21(si + 9.0));
    col += vec3(0.26, 0.24, 0.19) * spore;

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
