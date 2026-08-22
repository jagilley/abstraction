// style: chalk_trois_crayons
// title: Three chalks on toned paper
// description: Black charcoal, red sanguine and white chalk worked over warm grey paper: broad smudged masses, gritty stick strokes that only catch the top of the tooth, and pale eraser lifts.
// tags: charcoal, chalk, sanguine, drawing, toned-paper, grain, monochrome-plus
// brief: mark-making - dry media reads dry because pigment sits on the peaks of the paper tooth and only fills the valleys under pressure
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const vec3 PAPER    = vec3(0.615, 0.578, 0.522);
const vec3 PAPER_HI = vec3(0.700, 0.664, 0.604);
const vec3 CHARCOAL = vec3(0.105, 0.098, 0.104);
const vec3 SANGUINE = vec3(0.585, 0.288, 0.222);
const vec3 CHALK    = vec3(0.955, 0.941, 0.912);
const mat2 ROT = mat2(0.80, 0.60, -0.60, 0.80);

float hash21(vec2 p) {
    p = fract(p * vec2(443.897, 441.423));
    p += dot(p, p.yx + 19.19);
    return fract((p.x + p.y) * p.x);
}
vec2 hash22(vec2 p) { return vec2(hash21(p), hash21(p + 71.31)); }
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

// stick strokes: the chalk is held at an angle, so the mark is a blunt band with
// abrupt ends, grainy all the way across
float sticks(vec2 p, float N, vec2 so, float halfLen, float halfW, float skew) {
    vec2 gi = floor(p * N);
    float m = 0.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 cell = gi + vec2(float(i), float(j));
            vec2 h = hash22(cell + so);
            vec2 h2 = hash22(cell * 1.9 + so + 23.7);
            if (h2.x > 0.70) continue;
            vec2 c = (cell + 0.15 + 0.7 * h) / N;
            float ang = skew + (h2.y - 0.5) * 1.7 + 1.1 * fbm3(c * 1.6 + so * 0.13);
            float ca = cos(ang), sa = sin(ang);
            vec2 d = p - c;
            vec2 q = vec2(d.x * ca + d.y * sa, -d.x * sa + d.y * ca);
            float L = halfLen * (0.5 + h2.x);
            float t = q.x / L;
            float y = q.y - 0.5 * (h.x - 0.5) * L * (t * t - 0.333);
            float w = halfW * (0.40 + 1.30 * h.y) * (0.80 + 0.35 * vnoise(vec2(q.x * 30.0, h.x * 20.0)));
            float body = (1.0 - smoothstep(0.55, 1.0, abs(y) / max(w, 1e-5)))
                       * (1.0 - smoothstep(0.72, 1.0, abs(t)));
            m = max(m, body * (0.55 + 0.65 * h2.y));
        }
    }
    return clamp(m, 0.0, 1.0);
}

void main() {
    vec2 so = vec2(seed * 31.17, seed * 15.53) + seed * seed * 0.027;
    vec2 p = uv;

    // paper tooth: coarse laid grain the chalk will catch on
    float tooth = vnoise(p * vec2(190.0, 205.0)) * 0.52 + vnoise(p * 96.0) * 0.28
                + vnoise(p * 430.0) * 0.20;
    float fib = vnoise(p * vec2(700.0, 120.0));
    vec3 col = mix(PAPER, PAPER_HI, 0.45 * tooth + 0.10 * fib);
    col = mix(col, PAPER * 0.92, 0.18 * fbm3(p * 3.3 + 7.0));

    // smudged tonal masses: the domain is stretched along a slowly turning
    // direction, which is what a thumb dragged across charcoal looks like
    float dir = 0.8 + 1.6 * fbm3(p * 1.2 + so);
    float cd = cos(dir), sd = sin(dir);
    vec2 r = vec2(p.x * cd + p.y * sd, -p.x * sd + p.y * cd);
    float smear = fbm(vec2(r.x * 3.1, r.y * 12.0) + so * 1.7);
    float mass = smoothstep(0.33, 0.82, 0.60 * smear + 0.40 * fbm3(p * 4.2 + so + 11.0));

    // dry pigment only bridges the tooth where it was pressed hard
    float catchK = smoothstep(1.02 - 1.25 * mass, 1.02 - 1.25 * mass + 0.30, tooth + 0.10 * fib);
    col = mix(col, CHARCOAL, 0.86 * catchK * (0.55 + 0.55 * mass));

    // sanguine passage, drawn in a different direction
    float smear2 = fbm(vec2(r.y * 3.4, r.x * 10.0) + so * 0.9 + 40.0);
    float mass2 = smoothstep(0.52, 0.88, 0.62 * smear2 + 0.38 * fbm3(p * 5.1 + so + 29.0));
    float catchS = smoothstep(1.02 - 1.25 * mass2, 1.02 - 1.25 * mass2 + 0.32, tooth);
    col = mix(col, SANGUINE, 0.80 * catchS);

    // gritty accent strokes in charcoal, then chalk on top
    float sc = sticks(p, 4.7, so + 3.0, 0.145, 0.021, 0.5);
    col = mix(col, CHARCOAL, 0.92 * sc * smoothstep(0.16, 0.62, tooth + 0.22 * vnoise(p * 520.0)));

    float sr = sticks(p, 6.4, so + 61.0, 0.100, 0.014, -0.9);
    col = mix(col, SANGUINE * 0.92, 0.80 * sr * smoothstep(0.22, 0.68, tooth));

    float sw = sticks(p, 4.1, so + 88.0, 0.160, 0.023, 2.1);
    col = mix(col, CHALK, 0.86 * sw * smoothstep(0.20, 0.70, tooth + 0.18 * vnoise(p * 380.0)));

    // eraser lifts and the bloom of chalk dust
    float lift = smoothstep(0.62, 0.94, fbm(vec2(r.x * 2.4, r.y * 9.0) + so + 71.0));
    col = mix(col, PAPER_HI, 0.42 * lift);
    col = mix(col, CHALK, 0.16 * smoothstep(0.70, 1.0, fbm3(p * 3.0 + so + 95.0)) * tooth);

    col *= 0.94 + 0.12 * tooth;
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
