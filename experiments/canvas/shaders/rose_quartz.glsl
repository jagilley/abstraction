// style: rose_quartz
// title: Rose quartz
// description: Clouded lobes of hot magenta, blush and pale periwinkle flow into one another with dark seams between them and tiny white glints where the grains meet, like a polished slab of rose quartz.
// tags: quartz, marble, blush, pink, violet, stone, soft, glints
// author: claude (lluminate smoke2 e57eef89)
precision mediump float;

varying vec2 uv;
uniform float seed;

// ---------- fixed palette (seed must not touch these) ----------
const vec3 PAL_A = vec3(0.7349, 0.2906, 0.7709);
const vec3 PAL_B = vec3(0.4300, 0.4300, 0.4500);
const vec3 PAL_C = vec3(1.0000, 1.0000, 1.0000);
const vec3 PAL_D = vec3(0.3737, 0.5851, 0.5683);

// ---------- fixed structure ----------
const float CELL_SCALE = 5.5;

float hash1(float n) {
    return fract(sin(n) * 43758.5453123);
}

// seed enters the hash lattice only as a domain offset
vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)) + seed * 17.0,
             dot(p, vec2(269.5, 183.3)) + seed * 31.0);
    return fract(sin(p) * 43758.5453123);
}

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y + seed * 0.618);
}

float noise(vec2 p) {
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
    float v = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 6; i++) {
        v += amp * noise(p);
        p *= 2.02;
        amp *= 0.55;
    }
    return v;
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
            vec2 cell = ip + neighbor;
            vec2 pt = hash2(cell);
            pt = 0.5 + 0.5 * sin(6.2831 * pt + seed * 3.0);
            vec2 diff = neighbor + pt - fp;
            float dist = dot(diff, diff);
            if (dist < minDist) {
                secondDist = minDist;
                minDist = dist;
                cellId = cell;
            } else if (dist < secondDist) {
                secondDist = dist;
            }
        }
    }
    return vec3(sqrt(minDist), sqrt(secondDist), hash1(dot(cellId, vec2(12.9898, 78.233)) + seed));
}

vec3 palette(float t) {
    return PAL_A + PAL_B * cos(6.28318 * (PAL_C * t + PAL_D));
}

void main() {
    vec2 p = uv * 2.0 - 1.0;

    // domain warp; seed only slides where in the field we are
    vec2 warp;
    warp.x = fbm(p * 3.0 + seed * 5.0);
    warp.y = fbm(p * 3.0 + seed * 5.0 + 100.0);
    vec2 warped = p + (warp - 0.5) * 0.8;

    vec3 vor = voronoi(warped * CELL_SCALE);

    float edge = smoothstep(0.0, 0.08, vor.y - vor.x);
    float cellShade = vor.z;

    float f = fbm(warped * CELL_SCALE * 0.5 + vor.z * 10.0);

    float t = cellShade * 0.6 + f * 0.4 + vor.x * 0.3;
    vec3 baseColor = palette(t);

    // grain interior shading
    baseColor *= mix(0.6, 1.15, cellShade);

    // dark seams between grains
    vec3 edgeColor = mix(vec3(0.02, 0.02, 0.04), baseColor, edge);

    // fine clouded streaks
    float streak = smoothstep(0.45, 0.55, fbm(warped * CELL_SCALE * 4.0 + vec2(seed * 2.0, 0.0)));
    vec3 col = mix(edgeColor, edgeColor * 1.3 + 0.05, streak * 0.3);

    // shallow vignette so off-centre crops stay usable
    col *= 1.0 - 0.18 * dot(p * 0.7, p * 0.7);

    // pinpoint glints where grains meet
    col += smoothstep(0.02, 0.0, vor.x) * vec3(1.0, 0.95, 0.85) * 0.6;

    col = pow(clamp(col, 0.0, 1.0), vec3(0.9));

    gl_FragColor = vec4(col, 1.0);
}
