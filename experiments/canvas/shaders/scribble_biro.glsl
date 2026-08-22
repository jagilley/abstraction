// style: scribble_biro
// title: Ballpoint scribble
// description: Overlapping loops of ballpoint pen worked over faintly ruled paper, the line skipping where the ball ran dry and gathering into small gobs where it paused, in blue and black with the odd red pass.
// tags: ballpoint, biro, scribble, loops, line, drawing, blue, paper
// brief: mark-making with a bad pen - a constant-width line whose only expression is speed, overlap and ink starvation
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const vec3 PAPER  = vec3(0.955, 0.952, 0.938);
const vec3 PAPER_D= vec3(0.884, 0.880, 0.860);
const vec3 RULE   = vec3(0.680, 0.720, 0.780);
const vec3 BIRO   = vec3(0.145, 0.215, 0.520);
const vec3 BLACK  = vec3(0.115, 0.112, 0.128);
const vec3 REDPEN = vec3(0.640, 0.155, 0.175);
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

// a pass of looping scribble; the loops are elongated and share a direction
// with their neighbours, the way a hand scribbling back and forth does
vec4 loops(vec2 p, float N, vec2 so, float rad, float wid, float redP) {
    vec2 gi = floor(p * N);
    vec3 csum = vec3(0.0);
    float acc = 0.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 cell = gi + vec2(float(i), float(j));
            vec2 h = hash22(cell + so);
            vec2 h2 = hash22(cell * 1.53 + so + 9.1);
            if (h2.x > 0.94) continue;
            vec2 c = (cell + 0.15 + 0.70 * h) / N;
            float ang = 3.1416 * fbm(c * 1.7 + so * 0.19) + (h.x - 0.5) * 1.0;
            float ca = cos(ang), sa = sin(ang);
            vec2 dr = p - c;
            float asp = 1.75 + 0.75 * h2.y;
            vec2 q = vec2((dr.x * ca + dr.y * sa) / asp, -dr.x * sa + dr.y * ca);
            float rr = length(q);
            float th = atan(q.y, q.x);
            float r = rad * (0.55 + 0.80 * h2.x);
            float wob = 1.0 + 0.20 * sin(3.0 * th + 6.2831 * h.x)
                            + 0.11 * sin(5.0 * th + 6.2831 * h2.y);
            float ddn = abs(rr - r * wob);
            if (ddn > rad * 0.6) continue;
            float dd = ddn * length(vec2(asp * cos(th), sin(th)));
            // the pen lifts for part of every loop, and the ball runs dry
            float gate = 0.5 + 0.5 * sin(2.0 * th + 6.2831 * h2.y);
            float on = smoothstep(0.02, 0.26, gate + 0.30);
            float w = wid * (0.70 + 0.60 * h.y) * on;
            w *= 0.34 + 1.00 * vnoise(vec2(th * 6.0 + h.x * 30.0, h2.x * 21.0));
            float cov = (1.0 - smoothstep(w * 0.40, w, dd)) * step(0.00025, w);
            // a gob of ink where the pen turned and slowed
            float gth = 6.2831 * h.y - 3.14159;
            float ga = abs(atan(sin(th - gth), cos(th - gth)));
            float gob = (1.0 - smoothstep(0.0, 0.20, ga)) * (1.0 - smoothstep(w * 1.2, w * 3.0, dd));
            cov = max(cov, 0.90 * gob * step(0.62, h2.y));
            if (cov <= 0.0) continue;
            float ci = hash21(cell * 2.3 + so + 4.7);
            vec3 pen = mix(BIRO, BLACK, step(0.55, ci));
            pen = mix(pen, REDPEN, step(1.0 - redP, ci));
            csum += pen * cov;
            acc += cov;
        }
    }
    if (acc <= 0.0) return vec4(0.0);
    return vec4(csum / acc, acc);
}

void main() {
    vec2 so = vec2(seed * 13.37, seed * 27.19) + seed * seed * 0.037;
    vec2 p = uv;

    float fib = vnoise(p * 300.0) * 0.55 + vnoise(p * 640.0) * 0.45;
    vec3 col = mix(PAPER, PAPER_D, 0.26 * (1.0 - fib) + 0.10 * fbm(p * 3.0 + 4.0));
    // faint ruling
    float rl = abs(fract(p.y * 13.0 + 0.31) - 0.5) * 2.0;
    col = mix(col, RULE, 0.38 * (1.0 - smoothstep(0.955, 0.992, rl)));

    // three passes at different sizes, ink building where they overlap
    vec4 l1 = loops(p, 5.5, so + 1.0, 0.070, 0.0033, 0.03);
    vec4 l2 = loops(p, 9.0, so + 37.0, 0.043, 0.0024, 0.07);
    vec4 l3 = loops(p, 15.0, so + 83.0, 0.026, 0.0017, 0.02);

    float a1 = clamp(l1.a, 0.0, 1.0), a2 = clamp(l2.a, 0.0, 1.0), a3 = clamp(l3.a, 0.0, 1.0);
    col = mix(col, l1.rgb, 0.93 * a1);
    col = mix(col, l2.rgb, 0.93 * a2);
    col = mix(col, l3.rgb, 0.90 * a3);

    // pressure burnish: heavily worked passages go darker and slightly glossy
    float work = clamp(0.35 * (l1.a + l2.a + l3.a) - 0.30, 0.0, 1.0);
    col = mix(col, col * vec3(0.72, 0.74, 0.82), 0.55 * work);

    col *= 0.965 + 0.07 * fib;
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
