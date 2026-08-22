// style: watercolor_blooms
// title: Watercolour blooms
// description: Transparent washes pooled wet-into-wet on cold-press paper, each puddle ringed by the dark hard edge pigment leaves as it dries, with granulation settling into the paper tooth.
// tags: watercolour, wash, bloom, granulation, paper, transparent
// brief: painterly - a wash reads as a wash because of the hard rim, the glaze multiplying where layers overlap, and pigment sinking into the tooth
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

// pigment transmittances (what the paper light looks like after passing the glaze)
const vec3 PAPER   = vec3(0.972, 0.965, 0.943);
const vec3 PAPER_D = vec3(0.905, 0.893, 0.862);
const vec3 INDIGO  = vec3(0.30, 0.42, 0.66);
const vec3 MADDER  = vec3(0.94, 0.52, 0.55);
const vec3 SIENNA  = vec3(0.90, 0.70, 0.34);
const vec3 SAP     = vec3(0.58, 0.74, 0.48);
const mat2 ROT = mat2(0.80, 0.60, -0.60, 0.80);

float hash21(vec2 p) {
    p = fract(p * vec2(443.897, 441.423));
    p += dot(p, p.yx + 19.19);
    return fract((p.x + p.y) * p.x);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i), b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0)), d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float fbm(vec2 p) {
    float s = 0.0, a = 0.55;
    for (int i = 0; i < 5; i++) { s += a * vnoise(p); p = ROT * p * 2.03; a *= 0.5; }
    return s / 1.06;
}
float fbm3(vec2 p) {
    float s = 0.0, a = 0.6;
    for (int i = 0; i < 3; i++) { s += a * vnoise(p); p = ROT * p * 2.07; a *= 0.5; }
    return s / 1.05;
}

// one transparent wash: irregular puddle with a dark rim where the edge dried
// x = interior alpha, y = rim alpha
vec2 puddle(vec2 p, float scale, vec2 off, float th, float feather) {
    vec2 w = vec2(fbm3(p * scale * 0.7 + off), fbm3(p * scale * 0.7 + off + 9.3));
    float v = fbm(p * scale + off + 0.9 * (w - 0.5));
    float a = smoothstep(th, th + feather, v);
    float rim = a * (1.0 - smoothstep(th + feather * 0.9, th + feather * 3.4, v));
    return vec2(a, rim);
}

void main() {
    vec2 so = vec2(seed * 21.13, seed * 33.71) + seed * seed * 0.023;
    vec2 p = uv;
    float px = fwidth(uv.x);

    // cold-press tooth: chunky low peaks, fine fibre on top
    float tooth = vnoise(p * vec2(150.0, 168.0)) * 0.60 + vnoise(p * 340.0) * 0.25
                + vnoise(p * vec2(88.0, 640.0)) * 0.15;
    float valley = 1.0 - tooth;                       // pigment settles in the low places
    vec3 col = mix(PAPER, PAPER_D, 0.35 * tooth + 0.12 * fbm3(p * 4.0 + 3.0));

    // four glazes, coarse to fine; overlaps multiply the way real washes do
    vec2 a1 = puddle(p, 4.2, so + 1.7, 0.47, 0.085);
    float g1 = (0.34 * a1.x + 0.48 * a1.y) * (0.70 + 0.62 * valley);
    col = mix(col, col * INDIGO, clamp(g1, 0.0, 1.0));

    vec2 a2 = puddle(p, 5.6, so * 1.31 + 14.2, 0.545, 0.070);
    float g2 = (0.30 * a2.x + 0.50 * a2.y) * (0.75 + 0.50 * valley);
    col = mix(col, col * MADDER, clamp(g2, 0.0, 1.0));

    vec2 a3 = puddle(p, 7.3, so * 0.77 + 31.5, 0.565, 0.055);
    float g3 = (0.28 * a3.x + 0.52 * a3.y) * (0.72 + 0.58 * valley);
    col = mix(col, col * SIENNA, clamp(g3, 0.0, 1.0));

    vec2 a4 = puddle(p, 9.6, so * 1.83 + 52.9, 0.590, 0.045);
    float g4 = (0.22 * a4.x + 0.46 * a4.y) * (0.78 + 0.44 * valley);
    col = mix(col, col * SAP, clamp(g4, 0.0, 1.0));

    // a deep pooled accent where two heavy washes sat on top of one another
    float pool = a1.x * a3.x * smoothstep(0.55, 0.80, fbm3(p * 6.5 + so + 7.0));
    col = mix(col, col * INDIGO * vec3(0.82, 0.86, 0.98), 0.45 * pool);

    // dry-brush skips over the tooth at the fastest strokes
    float skip = smoothstep(0.62, 0.95, tooth) * smoothstep(0.55, 0.85, fbm3(p * 11.0 + so + 21.0));
    col = mix(col, PAPER, 0.30 * skip);

    // fine spatter flicked off the brush
    vec2 sp = p * 46.0 + so;
    vec2 si = floor(sp);
    float h = hash21(si + 3.7);
    if (h > 0.90) {
        float r = 0.09 + 0.16 * hash21(si + 8.1);
        vec2 fp = fract(sp) - vec2(hash21(si + 1.1), hash21(si + 2.2));
        float d = length(fp) * (0.85 + 0.35 * vnoise(fp * 9.0 + si));
        float dot0 = 1.0 - smoothstep(r * 0.55, r, d);
        col = mix(col, col * mix(INDIGO, MADDER, hash21(si + 4.4)), 0.55 * dot0);
    }

    col *= 0.955 + 0.09 * tooth;
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
