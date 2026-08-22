// style: veined_continents
// title: Veined continents
// description: Broad soft-edged landmasses of moss green, cobalt and magenta pushed up against one another and outlined by thin dark veins, like a marbled endpaper or a weather map of somewhere that does not exist.
// tags: marbled, continents, veined, warp, green, magenta, organic
// author: claude (lluminate smoke2 3bf33502)
precision mediump float;

varying vec2 uv;
uniform float seed;

// ---------- fixed palette (seed must not touch these) ----------
const vec3 PAL_A = vec3(0.50, 0.70, 0.50);
const vec3 PAL_B = vec3(0.50, 0.45, 0.50);
const vec3 PAL_C = vec3(1.00, 0.90, 0.90);
const vec3 PAL_D = vec3(0.00, 0.33, 0.67);

// ---------- fixed structure ----------
const float SPAN = 8.0;      // world units across the canvas
const float CELL_SCALE = 1.7;

float hash1(float n) {
    return fract(sin(n) * 43758.5453123);
}

// seed enters the hash lattice only as a domain offset
vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)) + seed * 13.13,
             dot(p, vec2(269.5, 183.3)) + seed * 7.77);
    return fract(sin(p) * 43758.5453123);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = dot(hash2(i) - 0.5, f - vec2(0.0, 0.0));
    float b = dot(hash2(i + vec2(1.0, 0.0)) - 0.5, f - vec2(1.0, 0.0));
    float c = dot(hash2(i + vec2(0.0, 1.0)) - 0.5, f - vec2(0.0, 1.0));
    float d = dot(hash2(i + vec2(1.0, 1.0)) - 0.5, f - vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
    float total = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    for (int i = 0; i < 5; i++) {
        total += noise(p * freq) * amp;
        freq *= 2.02;
        amp *= 0.55;
    }
    return total;
}

vec2 warp(vec2 p) {
    vec2 q = vec2(fbm(p), fbm(p + vec2(5.2, 1.3)));
    vec2 r = vec2(fbm(p + 4.0 * q + vec2(1.7, 9.2)),
                  fbm(p + 4.0 * q + vec2(8.3, 2.8)));
    return p + 2.5 * r;
}

vec3 voronoi(vec2 p) {
    vec2 ip = floor(p);
    vec2 fp = fract(p);
    float minD1 = 8.0;
    float minD2 = 8.0;
    vec2 cellId = vec2(0.0);

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));
            vec2 pointPos = hash2(ip + neighbor);
            pointPos = 0.5 + 0.45 * sin(6.2831 * pointPos + seed);
            vec2 diff = neighbor + pointPos - fp;
            float d = dot(diff, diff);
            if (d < minD1) {
                minD2 = minD1;
                minD1 = d;
                cellId = ip + neighbor;
            } else if (d < minD2) {
                minD2 = d;
            }
        }
    }
    float edge = minD2 - minD1;
    return vec3(sqrt(minD1), edge, hash1(dot(cellId, vec2(12.9898, 78.233)) + seed));
}

vec3 palette(float t) {
    return PAL_A + PAL_B * cos(6.28318 * (PAL_C * t + PAL_D));
}

void main() {
    vec2 p = (uv - 0.5) * SPAN;

    // seed only rotates and slides the field: arrangement, not identity
    float rot = seed * 0.7;
    mat2 rmat = mat2(cos(rot), -sin(rot), sin(rot), cos(rot));
    p = rmat * p + vec2(sin(seed * 1.9), cos(seed * 2.7)) * 9.0;

    vec2 wp = warp(p * 0.6);

    vec3 vor = voronoi(wp * CELL_SCALE);
    float cellDist = vor.x;
    float edgeDist = vor.y;
    float cellRand = vor.z;

    float fine = fbm(p * 3.0 + wp * 0.5);

    float veinLine = pow(1.0 - smoothstep(0.0, 0.04, edgeDist), 1.5);
    float cellShade = smoothstep(0.0, 1.2, cellDist);

    float base = mix(cellRand, fine * 0.5 + 0.5, 0.35);
    base = clamp(base + 0.15 * fine, 0.0, 1.0);

    vec3 col = palette(base);
    col *= mix(0.6, 1.0, cellShade);

    vec3 lineColor = palette(fract(base + 0.5)) * 0.3;
    col = mix(col, lineColor, veinLine * 0.85);

    float grain = noise(p * 40.0) * 0.04;
    col += grain;

    // scattered inverted speckle, like flecks in handmade paper
    float speck = fbm(p * 8.0);
    float dots = smoothstep(0.85, 0.95, speck);
    col = mix(col, vec3(1.0) - col, dots * 0.3);

    // shallow vignette so off-centre crops stay usable
    vec2 vig = uv - 0.5;
    col *= 1.0 - dot(vig, vig) * 0.35;

    col = pow(clamp(col, 0.0, 1.0), vec3(0.9));

    gl_FragColor = vec4(col, 1.0);
}
