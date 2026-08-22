// style: riso_misprint
// title: Risograph misprint
// description: Three spot inks — fluorescent pink, medium blue and yellow — screened as chunky halftone dots at different angles, each plate printed slightly out of register so the colours fringe and overprint into new hues on cheap warm paper.
// tags: risograph, halftone, print, misregistration, spot-colour, pink, blue
// brief: mark-making by machine - the dot is the mark; misregistration, uneven roller ink and posterised separations are what make a print look printed
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const vec3 PAPER = vec3(0.945, 0.930, 0.898);
const vec3 PINK  = vec3(0.98, 0.23, 0.52);   // fluorescent pink
const vec3 BLUE  = vec3(0.13, 0.34, 0.70);   // medium blue
const vec3 YELL  = vec3(0.99, 0.83, 0.16);   // yellow
const float CELL = 0.0265;                    // halftone pitch
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

// one screened plate: rotate, offset (the misregistration), dot area = tone
float plate(vec2 p, float tone, float ang, vec2 shift, float ew) {
    float ca = cos(ang), sa = sin(ang);
    vec2 q = p + shift;
    q = vec2(q.x * ca + q.y * sa, -q.x * sa + q.y * ca) / CELL;
    vec2 f = fract(q) - 0.5;
    float d = length(f);
    float r = 0.735 * sqrt(clamp(tone, 0.0, 1.0));
    return 1.0 - smoothstep(r - ew, r + ew, d);
}

float posterize(float t, float n) {
    return mix(t, floor(t * n + 0.5) / n, 0.65);
}

// riso ink is thin but not a pure filter: an overprint darkens toward the
// product of the two inks while keeping some of the top ink's own colour
vec3 over(vec3 below, vec3 ink, float a) {
    return mix(below, mix(below * ink, ink, 0.44), a);
}

void main() {
    vec2 so = vec2(seed * 23.71, seed * 11.13) + seed * seed * 0.019;
    vec2 p = uv;
    float px = fwidth(uv.x);
    float ew = max(0.035, 0.9 * px / CELL);

    // the separations: three plates of one drawing, so they relate to each other
    vec2 wp = p + 0.20 * vec2(fbm(p * 2.1 + so), fbm(p * 2.1 + so + 5.5));
    float f = fbm(wp * 3.2 + so);
    float g = fbm(wp * 5.4 + so * 1.3 + 17.0);
    float h = fbm(p * 7.1 + so * 0.7 + 39.0);

    float tPink = 0.92 * posterize(smoothstep(0.44, 0.86, f), 4.0);
    float tBlue = posterize(smoothstep(0.42, 0.79, 0.62 * g + 0.38 * (1.0 - f)), 4.0);
    float tYell = posterize(smoothstep(0.36, 0.74, 0.70 * h + 0.30 * f), 3.0);

    // uneven roller ink and drum streaks
    float roll = 0.82 + 0.30 * fbm(p * vec2(1.6, 9.0) + so + 3.0);
    float streak = 1.0 - 0.18 * smoothstep(0.55, 0.95, vnoise(p * vec2(2.0, 130.0) + so));
    tPink *= roll * streak;
    tBlue *= (1.66 - roll) * streak;
    tYell *= 0.82 * (0.86 + 0.24 * fbm(p * vec2(9.0, 1.7) + so + 8.0));

    // misregistration: each plate lands a hair off
    vec2 jp = 0.0035 * vec2(vnoise(so * 0.31), vnoise(so * 0.53 + 4.0)) - 0.00175;
    float aP = plate(p, tPink, 0.2618, vec2(-0.0075, 0.0042) + jp, ew);
    float aB = plate(p, tBlue, 1.3090, vec2(0.0068, 0.0055) - jp, ew);
    float aY = plate(p, tYell, 0.7854, vec2(0.0021, -0.0080) + 0.6 * jp.yx, ew);

    // paper, then translucent inks laid down one plate at a time
    float fib = vnoise(p * 260.0) * 0.6 + vnoise(p * 520.0) * 0.4;
    vec3 col = mix(PAPER, PAPER * 0.94, 0.35 * (1.0 - fib) + 0.10 * fbm(p * 4.0 + 21.0));
    col = over(col, YELL, 0.92 * aY);
    col = over(col, BLUE, 0.90 * aB);
    col = over(col, PINK, 0.90 * aP);

    col *= 0.965 + 0.07 * fib;
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
