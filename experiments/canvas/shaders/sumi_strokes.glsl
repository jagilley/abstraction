// style: sumi_strokes
// title: Sumi-e brush strokes
// description: Black ink brush strokes on warm rice paper: broad diluted washes underneath, dry-brush strokes with tapered ends and bristle streaks on top, and the occasional small vermilion seal.
// tags: ink, sumi-e, brush, calligraphy, monochrome, paper
// brief: painterly / mark-making - what makes a mark read as a mark: pressure taper, bristle streaks, ink pooling at the wet edge
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const vec3 PAPER    = vec3(0.928, 0.903, 0.842);
const vec3 PAPER_SH = vec3(0.836, 0.800, 0.722);
const vec3 INK      = vec3(0.055, 0.052, 0.062);
const vec3 WASH     = vec3(0.300, 0.305, 0.330);
const vec3 SEAL     = vec3(0.700, 0.150, 0.110);
const mat2 ROT = mat2(0.80, 0.60, -0.60, 0.80);

float hash21(vec2 p) {
    p = fract(p * vec2(443.897, 441.423));
    p += dot(p, p.yx + 19.19);
    return fract((p.x + p.y) * p.x);
}
vec2 hash22(vec2 p) {
    float a = hash21(p);
    float b = hash21(p + 71.31);
    return vec2(a, b);
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
    for (int i = 0; i < 5; i++) { s += a * vnoise(p); p = ROT * p * 2.02; a *= 0.5; }
    return s / 1.06;
}

// One layer of brush strokes on a jittered grid. Returns vec2(body, rim).
// The mark reads as a mark because of the pressure profile (heavy head, long
// tapering tail), the ragged wet edge, and bristle hairs that open up as the
// brush runs out of ink.
vec3 strokes(vec2 p, float N, vec2 so, float halfLen, float halfW, float dry, float ew) {
    vec2 g = p * N;
    vec2 gi = floor(g);
    float body = 0.0, rim = 0.0, halo = 0.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 cell = gi + vec2(float(i), float(j));
            vec2 h = hash22(cell + so);
            vec2 h2 = hash22(cell * 1.7 + so + 31.1);
            if (h2.x > 0.90) continue;                       // a few gaps in the field
            vec2 c = (cell + 0.2 + 0.6 * h) / N;
            float flow = 6.2831 * fbm(c * 1.3 + so * 0.11);
            float ang = flow + (h2.y - 0.5) * 2.6;
            float ca = cos(ang), sa = sin(ang);
            vec2 d = p - c;
            vec2 q = vec2(d.x * ca + d.y * sa, -d.x * sa + d.y * ca);
            float L = halfLen * (0.55 + 0.9 * h2.x);
            float t = q.x / L;
            float u = 0.5 * (t + 1.0);                       // 0 at head, 1 at tail
            if (u < -0.05 || u > 1.05) continue;
            float curv = (h.x - 0.5) * 1.3;
            float y = q.y - curv * L * (t * t - 0.333);
            // pressure: snap to full width at the head, bleed away at the tail
            float prof = smoothstep(0.0, 0.13, u) * pow(max(0.0, 1.0 - u), 0.52);
            float press = 0.72 + 0.60 * vnoise(vec2(u * 3.4 + h.y * 40.0, h.x * 17.0));
            float w = halfW * (0.6 + 0.8 * h.y) * prof * press;
            w *= 0.86 + 0.34 * vnoise(vec2(q.x * 55.0 + h.x * 23.0, h.y * 9.0));  // ragged edge
            float e = abs(y) / max(w, 1e-5);
            float cov = 1.0 - smoothstep(1.0 - ew / max(w, 1e-5), 1.0, e);
            // bristle hairs, splitting apart as the brush dries out along the stroke
            float run = dry * smoothstep(-0.10, 0.95, u);
            float hair = vnoise(vec2(q.x * 14.0 + h.x * 60.0, y * 620.0));
            float hair2 = vnoise(vec2(q.x * 40.0, y * 240.0 + h.y * 30.0));
            float bristle = smoothstep(0.10, 0.55, 0.55 * hair + 0.45 * hair2);
            cov *= mix(1.0, bristle, run);
            cov *= step(0.0005, w);
            body = body + cov - body * cov;
            float r = cov * smoothstep(0.28, 0.99, e);
            rim = rim + r - rim * r;
            // ink wicking into the paper fibres just outside the mark
            float hc = (1.0 - smoothstep(0.30, 1.0, abs(y) / max(w * 3.0, 1e-5))) * step(0.0005, w);
            hc *= 0.55 + 0.45 * vnoise(vec2(q.x * 60.0, y * 60.0 + h.x * 12.0));
            halo = halo + hc - halo * hc;
        }
    }
    return vec3(body, rim, halo);
}

// sparse carved seals
float seals(vec2 p, vec2 so) {
    vec2 g = p * 2.3;
    vec2 gi = floor(g);
    float m = 0.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 cell = gi + vec2(float(i), float(j));
            vec2 h = hash22(cell + so * 0.37 + 5.5);
            if (hash21(cell * 3.3 + so) > 0.20) continue;
            vec2 c = (cell + 0.25 + 0.5 * h) / 2.3;
            vec2 d = abs(p - c);
            float r = 0.030 * (0.85 + 0.3 * h.x);
            float box = 1.0 - smoothstep(r - 0.002, r, max(d.x, d.y));
            // carved marks inside the seal
            vec2 lp = (p - c) / r;
            float bars = step(0.35, vnoise(lp * vec2(2.6, 2.6) + h * 11.0));
            float frame = smoothstep(0.72, 0.80, max(abs(lp.x), abs(lp.y)));
            float ink = max(frame, bars * 0.9);
            m = max(m, box * ink * (0.75 + 0.25 * vnoise(p * 300.0)));
        }
    }
    return m;
}

void main() {
    vec2 so = vec2(seed * 27.31, seed * 13.77) + seed * seed * 0.017;
    vec2 p = uv;
    float px = fwidth(uv.x);
    float ew = max(0.0018, 1.4 * px);

    // paper: fibre grain plus soft mottling
    float grain = vnoise(p * vec2(230.0, 260.0)) * 0.55 + vnoise(p * 470.0) * 0.25
                + vnoise(p * vec2(700.0, 90.0)) * 0.20;
    float mottle = fbm(p * 3.1 + 11.0);
    vec3 col = mix(PAPER, PAPER_SH, 0.30 * mottle + 0.26 * (1.0 - grain));

    // dilute background wash sweeps
    vec3 w1 = strokes(p, 3.6, so + 3.0, 0.17, 0.040, 0.35, ew * 2.5);
    col = mix(col, mix(PAPER_SH, WASH, 0.5), 0.30 * w1.z);
    col = mix(col, WASH, clamp(0.30 * w1.x + 0.26 * w1.y, 0.0, 1.0));

    // mid-tone strokes
    vec3 w2 = strokes(p, 4.6, so + 19.0, 0.135, 0.026, 0.62, ew);
    col = mix(col, WASH, 0.30 * w2.z);
    col = mix(col, mix(WASH, INK, 0.55), clamp(0.74 * w2.x + 0.24 * w2.y, 0.0, 1.0));

    // dark accent strokes, driest
    vec3 w3 = strokes(p, 6.5, so + 47.0, 0.100, 0.014, 0.92, ew);
    col = mix(col, mix(WASH, INK, 0.4), 0.34 * w3.z);
    col = mix(col, INK, clamp(0.93 * w3.x, 0.0, 1.0));

    // seals
    col = mix(col, SEAL, 0.92 * seals(p, so));

    // paper tooth showing through the ink
    col *= 0.94 + 0.10 * grain;
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
