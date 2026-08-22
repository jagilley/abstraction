// style: crackle_glaze
// title: Crackle glaze
// description: A field of soft blue and teal ceramic panels crazed apart by a fine dark network of cracks, with a few pale brush strokes laid over the top like a glaze that ran before it set.
// tags: ceramic, crackle, glaze, voronoi, blue, teal, strokes
// author: claude (lluminate smoke2 b2c6d300)
precision mediump float;

varying vec2 uv;
uniform float seed;

// ---------- fixed palette (seed must not touch these) ----------
const vec3 PAL_A = vec3(0.55, 0.50, 0.50);
const vec3 PAL_B = vec3(0.50, 0.45, 0.50);
const vec3 PAL_C = vec3(1.00, 0.90, 0.70);
const vec3 PAL_D = vec3(0.10, 0.42, 0.71);

// ---------- fixed structure ----------
const float WARP_SCALE = 3.2;
const float CELL_SCALE = 9.0;
const int STROKES = 22;

// ---------- hashing (seed enters only as a domain offset) ----------
float hash(vec2 p) {
    p = p + seed * 37.13;
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

vec2 hash2(vec2 p) {
    vec2 q = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    q += seed * 13.13;
    return fract(sin(q) * 43758.5453123);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm(vec2 p) {
    float sum = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    for (int i = 0; i < 5; i++) {
        sum += amp * vnoise(p * freq);
        freq *= 2.03;
        amp *= 0.55;
        p = p * mat2(0.8, 0.6, -0.6, 0.8);
    }
    return sum;
}

vec3 palette(float t) {
    return PAL_A + PAL_B * cos(6.28318 * (PAL_C * t + PAL_D));
}

vec3 voronoi(vec2 p) {
    vec2 ip = floor(p);
    vec2 fp = fract(p);
    float minDist = 8.0;
    float secondDist = 8.0;
    vec2 cellId = vec2(0.0);
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));
            vec2 rnd = hash2(ip + neighbor);
            vec2 diff = neighbor + rnd - fp;
            float d = dot(diff, diff);
            if (d < minDist) {
                secondDist = minDist;
                minDist = d;
                cellId = ip + neighbor + rnd;
            } else if (d < secondDist) {
                secondDist = d;
            }
        }
    }
    return vec3(sqrt(minDist), sqrt(secondDist), hash(cellId));
}

float segDist(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0);
    return length(pa - ba * h);
}

// distance to a quadratic bezier, polyline-approximated so strokes stay
// continuous under the zoom a crop applies
float strokeDist(vec2 p, vec2 p0, vec2 p1, vec2 p2) {
    float best = 1e5;
    vec2 prev = p0;
    const int N = 12;
    for (int i = 1; i <= N; i++) {
        float t = float(i) / float(N);
        vec2 pt = mix(mix(p0, p1, t), mix(p1, p2, t), t);
        best = min(best, segDist(p, prev, pt));
        prev = pt;
    }
    return best;
}

void main() {
    vec2 p = uv;

    // seed slides the whole field: arrangement, not identity
    vec2 shift = vec2(sin(seed * 1.31), cos(seed * 1.87)) * 13.0 + seed * 0.71;

    vec2 wp = p * WARP_SCALE + shift;
    vec2 warp = vec2(fbm(wp + vec2(1.7, 9.2)), fbm(wp + vec2(8.3, 2.8)));
    vec2 warped = wp + (warp - 0.5) * 1.6;

    float base = fbm(warped);

    // voronoi crazing layer
    vec2 vp = p * CELL_SCALE + shift * 2.0 + (warp - 0.5) * 1.2;
    vec3 vor = voronoi(vp);
    float cellEdge = smoothstep(0.0, 0.06, vor.y - vor.x);
    float cellShade = mix(0.15, 1.0, vor.z);

    float t = base * 0.7 + vor.z * 0.3;
    vec3 col = palette(t);
    col *= mix(0.5, 1.0, cellShade);
    col *= mix(0.35, 1.0, cellEdge);

    // fine grain
    col += (hash(uv * 800.0) - 0.5) * 0.03;

    // calligraphic strokes overlay: seeded positions, fixed weight
    float strokeAccum = 0.0;
    for (int i = 0; i < STROKES; i++) {
        float fi = float(i);
        vec2 c0 = hash2(vec2(fi * 3.17, fi * 7.91));
        vec2 dir = normalize(hash2(vec2(fi * 1.3 + 2.0, fi * 5.7 + 1.0)) - 0.5);
        float len = 0.15 + 0.25 * hash(vec2(fi * 9.3, fi * 2.1));
        vec2 c1 = c0 + dir * len * 0.5 + (hash2(vec2(fi * 4.4, fi * 8.8)) - 0.5) * 0.2;
        vec2 c2 = c0 + dir * len;

        float d = strokeDist(p, c0, c1, c2);
        float w = 0.0026 + 0.004 * hash(vec2(fi * 6.6, fi * 3.3));
        strokeAccum += smoothstep(w, 0.0, d);
    }
    strokeAccum = clamp(strokeAccum, 0.0, 1.0);

    col = mix(col, palette(t + 0.5), strokeAccum * 0.65);

    // shallow vignette so off-centre crops stay usable
    vec2 centered = uv - 0.5;
    col *= 1.0 - dot(centered, centered) * 0.35;

    col = pow(clamp(col, 0.0, 1.0), vec3(0.95));

    gl_FragColor = vec4(col, 1.0);
}
