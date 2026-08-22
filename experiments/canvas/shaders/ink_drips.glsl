// style: ink_drips
// title: Ink drips and runs
// description: Ink poured at the top and left to run: narrow vertical trails that wobble, thin as they travel and swell into a bead where they stopped, in black, ultramarine and burnt orange over a stained bone ground.
// tags: ink, drip, pour, run, gravity, vertical, black, ultramarine
// brief: mark-making without a hand - gravity draws the line; a run reads as a run because of the wobble, the taper and the bead of pigment at the tip
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const vec3 GROUND  = vec3(0.906, 0.898, 0.876);
const vec3 GROUND_D= vec3(0.798, 0.788, 0.762);
const vec3 INKB    = vec3(0.088, 0.096, 0.108);
const vec3 ULTRA   = vec3(0.140, 0.185, 0.520);
const vec3 RUST    = vec3(0.760, 0.345, 0.130);
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

// runs of ink falling from spawn points on a coarse grid. Gravity is -y.
// returns rgb of the topmost run, .a its coverage; halo comes back separately.
vec4 runs(vec2 p, float N, float M, vec2 so, float maxL, float baseW, out float halo) {
    vec2 gi = floor(p * vec2(N, M));
    vec3 col = vec3(0.0);
    float cov = 0.0;
    halo = 0.0;
    for (int jj = 0; jj <= 3; jj++) {
        for (int ii = -1; ii <= 1; ii++) {
            vec2 cell = gi + vec2(float(ii), float(jj));
            vec2 h = hash22(cell + so);
            vec2 h2 = hash22(cell * 1.37 + so + 7.7);
            float cx = (cell.x + 0.12 + 0.76 * h.x) / N;
            float cy = (cell.y + 0.10 + 0.80 * h.y) / M;
            // the pour is heavier in some passages than others
            float dens = fbm(vec2(cx * 2.6, cy * 1.1) + so + 3.3);
            if (h2.x > 0.30 + 1.05 * dens) continue;
            float L = maxL * (0.22 + 1.00 * h2.y * h2.y * 1.35);
            float dy = cy - p.y;
            if (dy < 0.0 || dy > L) continue;
            float u = dy / L;
            float wob = 0.013 * (vnoise(vec2(dy * 8.0 + h.x * 60.0, h.y * 25.0)) - 0.5) * (0.25 + u);
            float x = p.x - cx - wob;
            float W = baseW * (0.40 + 1.25 * h2.x);
            float w = W * (1.0 - 0.45 * u);
            w *= smoothstep(0.0, 0.035, u);                               // the run gathers itself
            float bd = (u - 0.90) / 0.055;
            w *= 1.0 + 1.10 * exp(-bd * bd);                              // bead at the tip
            w *= 0.90 + 0.20 * vnoise(vec2(dy * 20.0, h.x * 13.0));
            float e = abs(x) / max(w, 1e-5);
            float a0 = 1.0 - smoothstep(0.70, 1.00, e);
            float hc = 1.0 - smoothstep(0.55, 1.00, abs(x) / max(w * 2.6, 1e-5));
            halo = max(halo, hc * (0.6 + 0.4 * vnoise(vec2(x * 220.0, dy * 60.0))));
            if (a0 <= 0.0) continue;
            float ci = hash21(cell * 3.1 + so + 2.9);
            vec3 pig = INKB;
            pig = mix(pig, ULTRA, step(0.56, ci));
            pig = mix(pig, RUST, step(0.82, ci));
            pig *= 0.86 + 0.26 * smoothstep(0.55, 1.0, e);                // wet edge sits darker
            pig *= 0.90 + 0.18 * vnoise(vec2(x * 260.0, dy * 90.0));
            col = mix(col, pig, a0 * (1.0 - 0.35 * cov));
            cov = max(cov, a0);
        }
    }
    return vec4(col, cov);
}

void main() {
    vec2 so = vec2(seed * 29.11, seed * 19.83) + seed * seed * 0.033;
    vec2 p = uv;

    // paper, with old stains that ran the same way
    float fib = vnoise(p * 240.0) * 0.6 + vnoise(p * 500.0) * 0.4;
    float stain = fbm(vec2(p.x * 7.0, p.y * 1.1) + so * 0.7);
    vec3 col = mix(GROUND, GROUND_D, 0.28 * (1.0 - fib) + 0.34 * smoothstep(0.42, 0.85, stain));
    col = mix(col, mix(GROUND_D, ULTRA, 0.14), 0.38 * smoothstep(0.56, 0.92, stain));
    // ghosts of earlier runs, dried into the ground
    float ghost = smoothstep(0.58, 0.95, fbm(vec2(p.x * 30.0, p.y * 1.5) + so + 55.0));
    col = mix(col, mix(GROUND_D, INKB, 0.40), 0.42 * ghost);
    float ghost2 = smoothstep(0.66, 0.98, fbm(vec2(p.x * 60.0, p.y * 2.4) + so + 88.0));
    col = mix(col, mix(GROUND_D, ULTRA, 0.30), 0.26 * ghost2);

    float h1, h2, h3;
    vec4 r1 = runs(p, 8.0, 2.0, so + 1.0, 0.85, 0.0115, h1);
    vec4 r2 = runs(p, 14.0, 3.6, so + 33.0, 0.50, 0.0058, h2);
    vec4 r3 = runs(p, 24.0, 6.5, so + 71.0, 0.26, 0.0028, h3);

    col = mix(col, mix(GROUND_D, INKB, 0.34), 0.26 * max(h1, max(h2, h3)));
    col = mix(col, r1.rgb, 0.95 * r1.a);
    col = mix(col, r2.rgb, 0.93 * r2.a);
    col = mix(col, r3.rgb, 0.90 * r3.a);

    // fine spatter thrown off the pour
    vec2 sp = p * 40.0 + so;
    vec2 si = floor(sp);
    if (hash21(si + 4.3) > 0.955) {
        vec2 fp = fract(sp) - vec2(hash21(si + 1.7), hash21(si + 2.9));
        float r = 0.10 + 0.14 * hash21(si + 6.1);
        float d = length(fp * vec2(1.0, 0.65 + 0.5 * hash21(si + 9.2)));
        col = mix(col, INKB, 0.85 * (1.0 - smoothstep(r * 0.6, r, d)));
    }

    col *= 0.955 + 0.09 * fib;
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
