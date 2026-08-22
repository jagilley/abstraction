// style: woodcut_gouge
// title: Woodcut and gouge
// description: A two-block relief print: flat oxide-red under warm black, both carved into faceted angular shapes, with parallel gouge slashes cut through the blacks and the wood grain showing wherever the block inked unevenly on cream paper.
// tags: woodcut, linocut, relief-print, carved, gouge, black, red, graphic
// brief: mark-making with a blade - the carved edge is locally straight with hard corners, and the gouge leaves a pointed slash of bare paper
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const vec3 PAPER = vec3(0.930, 0.903, 0.836);
const vec3 PAPER_D = vec3(0.856, 0.822, 0.740);
const vec3 INK   = vec3(0.098, 0.090, 0.086);
const vec3 OXIDE = vec3(0.700, 0.245, 0.165);
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
    for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = ROT * p * 2.03; a *= 0.5; }
    return s / 1.03;
}

// piecewise-LINEAR noise on a triangular lattice: its level sets are straight
// segments meeting at corners, which is what a blade leaves behind
float tnoise(vec2 p) {
    const float K1 = 0.3660254, K2 = 0.2113249;
    vec2 s = p + (p.x + p.y) * K1;
    vec2 i = floor(s);
    vec2 f = s - i;
    float h00 = hash21(i), h11 = hash21(i + 1.0);
    if (f.x > f.y) {
        float h10 = hash21(i + vec2(1.0, 0.0));
        return (1.0 - f.x) * h00 + (f.x - f.y) * h10 + f.y * h11;
    }
    float h01 = hash21(i + vec2(0.0, 1.0));
    return (1.0 - f.y) * h00 + (f.y - f.x) * h01 + f.x * h11;
}
float tfbm(vec2 p) {
    float s = 0.0, a = 0.66;
    for (int i = 0; i < 3; i++) { s += a * tnoise(p); p = ROT * p * 2.17; a *= 0.38; }
    return s / 1.00;
}

// parallel gouge slashes: pointed at both ends, following a slow curve
float gouges(vec2 p, float ang, float freq, vec2 so, float len, float wide) {
    float ca = cos(ang), sa = sin(ang);
    vec2 q = vec2(p.x * ca + p.y * sa, -p.x * sa + p.y * ca);
    float ph = q.y * freq + 1.6 * fbm(p * 3.0 + so);
    float line = floor(ph);
    float f = abs(fract(ph) - 0.5);
    float lj = hash21(vec2(line, 3.3) + so);
    float sc = q.x / len + lj * 7.0;
    float si = floor(sc), u = fract(sc);
    float sh = hash21(vec2(line, si) + so + 5.1);
    float on = step(0.30, sh);
    float taper = pow(max(0.0, sin(3.14159 * u)), 0.85);         // pointed both ends
    float w = wide * (0.70 + 0.55 * sh) * taper * on;
    return (1.0 - smoothstep(w * 0.55, w, f)) * step(0.004, w);
}

void main() {
    vec2 so = vec2(seed * 17.91, seed * 29.37) + seed * seed * 0.021;
    vec2 p = uv;

    // paper
    float fib = vnoise(p * 250.0) * 0.6 + vnoise(p * 520.0) * 0.4;
    vec3 col = mix(PAPER, PAPER_D, 0.30 * (1.0 - fib) + 0.14 * fbm(p * 3.5 + 9.0));

    // wood grain of the block, which shows in the inking
    vec2 gp = p * vec2(1.4, 5.0) + 0.55 * vec2(fbm(p * 2.0 + so), 0.0);
    float rings = fract(gp.y * 3.4 + 2.2 * fbm(gp * 1.6 + so));
    float grain = 0.55 + 0.45 * smoothstep(0.10, 0.55, abs(rings - 0.5) * 2.0);
    grain *= 0.82 + 0.30 * vnoise(p * vec2(18.0, 240.0) + so);

    // red block, printed first and a hair out of register
    vec2 rp = p + vec2(0.0075, -0.0055);
    float red = step(0.470, tfbm(rp * 4.1 + so * 1.19 + 30.0));
    red *= 1.0 - 0.80 * gouges(rp, 1.05, 27.0, so + 12.0, 0.19, 0.26)
                 * smoothstep(0.34, 0.58, fbm(rp * 2.6 + so + 170.0));
    float redInk = red * (0.72 + 0.42 * grain) * step(0.10, vnoise(p * 300.0) + 0.55);
    col = mix(col, OXIDE, clamp(0.94 * redInk, 0.0, 1.0));

    // black block
    float k = tfbm(p * 3.6 + so);
    float blackMask = step(0.470, k);
    // fine chipping along the carved edge
    float edge = 1.0 - smoothstep(0.0, 0.030, abs(k - 0.470));
    blackMask = clamp(blackMask + edge * (step(0.55, vnoise(p * 190.0 + so)) - 0.5) * 1.2, 0.0, 1.0);

    // gouges cut through the black in two directions
    // a carver treats different passages differently: some flats are left solid,
    // some are cut in one direction, only a few are crossed
    float r1 = smoothstep(0.36, 0.60, fbm(p * 2.3 + so + 100.0));
    float r2 = smoothstep(0.52, 0.76, fbm(p * 2.9 + so + 140.0));
    float g1 = gouges(p, -0.35, 34.0, so + 21.0, 0.22, 0.30) * r1;
    float g2 = gouges(p, 1.42, 44.0, so + 44.0, 0.15, 0.22) * r2;
    float carved = clamp(g1 + 0.80 * g2, 0.0, 1.0);
    float black = blackMask * (1.0 - 0.96 * carved);

    // uneven inking of the block: grain and speckle let paper through
    float ink = black * (0.70 + 0.44 * grain);
    ink *= 0.80 + 0.35 * vnoise(p * vec2(130.0, 150.0) + so);
    col = mix(col, INK, clamp(1.15 * ink, 0.0, 1.0));

    // chips of ink left standing where the blade skipped in the cleared areas
    float chip = (1.0 - blackMask) * step(0.74, vnoise(p * 105.0 + so + 70.0))
               * smoothstep(0.34, 0.62, tfbm(p * 3.6 + so));
    col = mix(col, INK, 0.55 * chip);

    col *= 0.96 + 0.08 * fib;
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
