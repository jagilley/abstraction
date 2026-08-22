// style: pointillist_dabs
// title: Pointillist dabs
// description: Small loaded brush dabs of unmixed colour laid side by side, warm and cool notes vibrating against each other, with the ochre-primed canvas weave showing in the gaps between touches.
// tags: pointillism, divisionism, dabs, impasto, canvas, colour, painterly
// brief: painterly - the dab is the mark: an oval of one pure pigment with a loaded ridge on one side, and colour mixed in the eye rather than on the palette
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const vec3 GROUND = vec3(0.795, 0.712, 0.572);
const vec3 GROUND_D = vec3(0.700, 0.612, 0.470);
// warm notes
const vec3 W1 = vec3(0.980, 0.805, 0.215);   // cadmium yellow
const vec3 W2 = vec3(0.945, 0.520, 0.155);   // cadmium orange
const vec3 W3 = vec3(0.870, 0.270, 0.195);   // vermilion
const vec3 W4 = vec3(0.855, 0.435, 0.400);   // rose madder
// cool notes
const vec3 C1 = vec3(0.095, 0.495, 0.420);   // viridian
const vec3 C2 = vec3(0.155, 0.335, 0.680);   // cobalt
const vec3 C3 = vec3(0.360, 0.275, 0.615);   // violet
const vec3 C4 = vec3(0.420, 0.605, 0.280);   // sap green
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
    for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = ROT * p * 2.03; a *= 0.5; }
    return s / 1.03;
}
vec3 pick4(vec3 a, vec3 b, vec3 c, vec3 d, float h) {
    vec3 r = a;
    r = mix(r, b, step(0.25, h));
    r = mix(r, c, step(0.50, h));
    r = mix(r, d, step(0.75, h));
    return r;
}

// one layer of loaded dabs; each keeps its own pure colour
vec4 dabs(vec2 p, float N, vec2 so, float la, float lb, float warmField, float lum) {
    vec2 gi = floor(p * N);
    vec3 col = vec3(0.0);
    float cov = 0.0, top = -1.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 cell = gi + vec2(float(i), float(j));
            vec2 h = hash22(cell + so);
            vec2 h2 = hash22(cell * 1.61 + so + 13.9);
            if (h2.x > 0.94) continue;                       // a few bare gaps
            vec2 c = (cell + 0.12 + 0.76 * h) / N;
            float ang = 1.9 * fbm(c * 2.2 + so * 0.17) + (h2.y - 0.5) * 1.5;
            float ca = cos(ang), sa = sin(ang);
            vec2 d = p - c;
            vec2 q = vec2(d.x * ca + d.y * sa, -d.x * sa + d.y * ca);
            float A = la * (0.70 + 0.75 * h2.x);
            float B = lb * (0.75 + 0.55 * h.y);
            B *= 1.0 - 0.45 * smoothstep(-0.2, 1.0, q.x / A);   // comma taper
            float e = length(vec2(q.x / A, q.y / max(B, 1e-5)));
            float a0 = 1.0 - smoothstep(0.72, 1.02, e);
            a0 *= 0.80 + 0.30 * vnoise(vec2(q.x * 130.0, q.y * 190.0) + cell);
            if (a0 <= 0.0) continue;
            // warm or cool, with the odd contrary note thrown in
            float wf = clamp(warmField + 0.20 * (h.x - 0.5), 0.0, 1.0);
            float flip = step(0.88, hash21(cell * 2.7 + so + 5.5));
            float useWarm = abs(flip - step(0.5, wf));
            vec3 cw = pick4(W1, W2, W3, W4, h2.y);
            vec3 cc = pick4(C1, C2, C3, C4, h.y);
            vec3 pig = mix(cc, cw, useWarm);
            // the loaded ridge catches the light on one side of the dab
            float ridge = 0.5 + 0.5 * (q.y / max(B, 1e-5)) * 0.9 - 0.35 * e;
            pig *= 0.80 + 0.42 * clamp(ridge, 0.0, 1.0);
            // light and shade: the same pigments, run darker in the shadow passages
            pig *= 0.63 + 0.66 * clamp(lum + 0.22 * (h2.y - 0.5), 0.0, 1.0);
            float z = h2.x + 0.3 * h.y;                       // paint order
            if (z > top) { top = z; col = mix(col, pig, a0); cov = max(cov, a0); }
            else { col = mix(pig, col, cov); cov = max(cov, a0); }
        }
    }
    return vec4(col, cov);
}

void main() {
    vec2 so = vec2(seed * 25.13, seed * 37.71) + seed * seed * 0.029;
    vec2 p = uv;

    // primed canvas: a woven ground the dabs sit on
    float wx = 0.5 + 0.5 * sin(p.x * 380.0);
    float wy = 0.5 + 0.5 * sin(p.y * 380.0);
    float weave = mix(wx, wy, 0.5) * 0.55 + vnoise(p * 270.0) * 0.45;
    vec3 col = mix(GROUND, GROUND_D, 0.40 * (1.0 - weave) + 0.20 * fbm(p * 3.4 + 6.0));

    // the light the picture is painted under: warm passages against cool ones
    vec2 wp = p + 0.14 * vec2(fbm(p * 2.0 + so), fbm(p * 2.0 + so + 3.3));
    float warmField = smoothstep(0.34, 0.66, fbm(wp * 2.4 + so));
    float lum = smoothstep(0.28, 0.74, 0.62 * fbm(wp * 2.0 + so + 9.1) + 0.38 * warmField);

    vec4 d1 = dabs(p, 16.0, so + 2.0, 0.038, 0.0190, warmField, lum);
    col = mix(col, d1.rgb, 0.94 * d1.a);
    vec4 d2 = dabs(p, 22.0, so + 51.0, 0.026, 0.0130, warmField, lum);
    col = mix(col, d2.rgb, 0.90 * d2.a);
    vec4 d3 = dabs(p, 31.0, so + 97.0, 0.017, 0.0086, warmField, lum);
    col = mix(col, d3.rgb, 0.84 * d3.a);

    col *= 0.93 + 0.13 * weave;
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
