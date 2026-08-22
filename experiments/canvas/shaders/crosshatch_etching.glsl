// style: crosshatch_etching
// title: Crosshatched etching
// description: An engraver's crosshatch in sepia and slate ink on laid ivory paper: flicked burin lines that swell and taper, curving with the form, crossing in two and three directions wherever the tone goes dark.
// tags: engraving, crosshatch, etching, line, ink, sepia, monochrome
// brief: mark-making - the burin line as the unit: it tapers at both ends, swells under pressure, and tone is built by crossing layers rather than by grey
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const vec3 PAPER  = vec3(0.938, 0.914, 0.856);
const vec3 PAPER_D= vec3(0.862, 0.828, 0.752);
const vec3 SEPIA  = vec3(0.235, 0.150, 0.098);
const vec3 SLATE  = vec3(0.180, 0.205, 0.268);
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

// one family of engraved lines: they follow a slow flow, break into flicks with
// tapered ends, and swell where the burin bit deeper.
float hatch(vec2 p, float ang, float freq, float tone, float bend, vec2 so, float ew) {
    float ca = cos(ang), sa = sin(ang);
    vec2 q = vec2(p.x * ca + p.y * sa, -p.x * sa + p.y * ca);
    float ph = q.y * freq + bend * fbm(p * 4.4 + so) + 0.35 * bend * fbm(p * 1.5 + so);
    float line = floor(ph);
    float f = abs(fract(ph) - 0.5);
    float lj = hash21(vec2(line, ang * 13.7) + so);
    float segL = 0.095 + 0.105 * hash21(vec2(line, 7.3) + so);
    float sc = q.x / segL + lj * 5.0;
    float si = floor(sc);
    float u = fract(sc);
    float sh = hash21(vec2(line, si) + so + 2.31);
    float on = step(0.16, sh);
    float taper = pow(max(0.0, sin(3.14159 * u)), 0.55);
    float w = 0.155 * tone * (0.72 + 0.56 * sh) * taper * on;
    w *= 0.66 + 0.70 * vnoise(vec2(q.x * 34.0 + lj * 30.0, line * 3.1));   // burin swell
    return (1.0 - smoothstep(max(w - ew * freq, 0.0), w, f)) * step(0.002, w);
}

void main() {
    vec2 so = vec2(seed * 19.31, seed * 41.17) + seed * seed * 0.031;
    vec2 p = uv;
    float px = fwidth(uv.x);
    float ew = max(0.0011, 0.85 * px);

    // laid paper: chain lines, fine laid lines, fibre
    float laid = 0.5 + 0.5 * sin(p.y * 520.0);
    float chain = 0.5 + 0.5 * sin(p.x * 47.0);
    float fib = vnoise(p * 300.0) * 0.6 + vnoise(p * 620.0) * 0.4;
    vec3 col = mix(PAPER, PAPER_D, 0.10 * laid + 0.05 * chain + 0.22 * (1.0 - fib)
                                  + 0.14 * fbm(p * 3.7 + 5.0));

    // the tonal drawing the hatching describes
    vec2 wp = p + 0.16 * vec2(fbm(p * 2.4 + so), fbm(p * 2.4 + so + 4.4));
    float t = fbm(wp * 3.4 + so * 0.7);
    t = smoothstep(0.26, 0.74, t);
    float t2 = smoothstep(0.30, 0.95, t);
    float t3 = smoothstep(0.55, 1.00, t);
    float t4 = smoothstep(0.76, 1.00, t);

    float h1 = hatch(p, 0.42, 44.0, 0.50 + 0.50 * t,  1.1, so + 1.0, ew);
    float h2 = hatch(p, 1.31, 39.0, 0.14 + 0.90 * t2, 0.9, so + 13.0, ew);
    float h3 = hatch(p, -0.62, 50.0, 0.08 + 0.94 * t3, 1.5, so + 27.0, ew);
    float h4 = hatch(p, 2.16, 35.0, 0.04 + 0.98 * t4, 0.8, so + 41.0, ew);

    // two plates: warm sepia carries the drawing, cool slate the deepest crossings
    float warm = 1.0 - (1.0 - h1) * (1.0 - h3);
    float cool = 1.0 - (1.0 - h2) * (1.0 - h4);
    col = mix(col, SEPIA, 0.93 * warm);
    col = mix(col, SLATE, 0.86 * cool);

    // plate tone left by the wiping rag, and the ink sitting on the fibre
    col = mix(col, PAPER_D * 0.94, 0.16 * smoothstep(0.35, 0.85, fbm(p * 2.2 + so + 60.0)));
    col *= 0.955 + 0.09 * fib;
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
