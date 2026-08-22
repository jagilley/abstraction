// style: amber_lacquer
// title: Amber lacquer
// description: Molten orange, rust and cream churn through a marbled turbulent field with veins of slate blue, split by fine ink cracks and dusted with metallic flecks and pinpoint highlights, like a gilded lacquer panel.
// tags: marble, amber, rust, lacquer, turbulent, ink, flecks, warm
// author: claude (lluminate smoke2 c8ce3351)
precision mediump float;

varying vec2 uv;
uniform float seed;

// ---------- fixed palette (seed must not touch these) ----------
const vec3 PAL_A = vec3(0.6225, 0.2840, 0.2013);
const vec3 PAL_B = vec3(0.5000, 0.5000, 0.5000);
const vec3 PAL_C = vec3(1.0000, 1.0000, 1.0000);
const vec3 PAL_D = vec3(0.4535, 0.3242, 0.1341);
const vec3 INK = vec3(0.03, 0.02, 0.04);

// ---------- fixed structure ----------
const float CELL_SCALE = 5.0;

// seed enters the hash lattice only as a domain offset
vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)) + seed * 17.0,
             dot(p, vec2(269.5, 183.3)) + seed * 31.0);
    return fract(sin(p) * 43758.5453123);
}

float hash1(vec2 p) {
    return fract(sin(dot(p, vec2(41.3, 289.1)) + seed * 13.7) * 43758.5453);
}

float hashf(float n) {
    return fract(sin(n) * 43758.5453123);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash1(i);
    float b = hash1(i + vec2(1.0, 0.0));
    float c = hash1(i + vec2(0.0, 1.0));
    float d = hash1(i + vec2(1.0, 1.0));
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

vec4 voronoi(vec2 p) {
    vec2 ip = floor(p);
    vec2 fp = fract(p);
    float minD = 8.0;
    float secD = 8.0;
    vec2 minCell = vec2(0.0);
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 cell = vec2(float(x), float(y));
            vec2 pt = hash2(ip + cell);
            pt = 0.5 + 0.5 * sin(6.2831 * pt + seed * 3.0);
            vec2 diff = cell + pt - fp;
            float d = dot(diff, diff);
            if (d < minD) {
                secD = minD;
                minD = d;
                minCell = ip + cell;
            } else if (d < secD) {
                secD = d;
            }
        }
    }
    float cellHash = hashf(dot(minCell, vec2(12.9898, 78.233)) + seed);
    return vec4(sqrt(minD), sqrt(secD), cellHash, 0.0);
}

vec3 palette(float t) {
    return PAL_A + PAL_B * cos(6.28318 * (PAL_C * t + PAL_D));
}

void main() {
    vec2 p = uv * 2.0 - 1.0;

    // two independent warp fields collided into one another
    vec2 wp = p * 3.2 + vec2(seed * 3.0, -seed * 2.0);
    vec2 warpA = vec2(fbm(wp + vec2(1.7, 4.2)), fbm(wp + vec2(9.2, 2.1)));

    vec2 warpB;
    warpB.x = fbm(p * 3.0 + seed * 5.0 + 50.0);
    warpB.y = fbm(p * 3.0 + seed * 5.0 + 150.0);

    vec2 warped = p + (warpA - 0.5) * 1.1 + (warpB - 0.5) * 0.7;

    vec4 vor = voronoi(warped * CELL_SCALE);
    float dist = vor.x;
    float cellH = vor.z;

    float edge = smoothstep(0.0, 0.07, vor.y - vor.x);

    float f = fbm(warped * CELL_SCALE * 0.5 + cellH * 10.0);
    float t = cellH * 0.55 + f * 0.35 + dist * 0.35;

    vec3 base = palette(fract(t));
    float ring = pow(sin(dist * 26.0 - fbm(warped * 1.5) * 4.0) * 0.5 + 0.5, 2.0);
    vec3 col = base * mix(0.6, 1.15, cellH) * (0.7 + 0.3 * ring);

    // ink cracks between cells
    col = mix(col, INK, (1.0 - edge) * 0.8);

    // calligraphic streak overlay
    float streak = smoothstep(0.45, 0.55, fbm(warped * CELL_SCALE * 4.0 + vec2(seed * 2.0, 0.0)));
    col = mix(col, col * 1.3 + 0.05, streak * 0.3);

    // fine flow lines
    float flow = fbm(p * 8.0 + warpA * 2.0 + warpB * 1.3 + seed);
    float lines = smoothstep(0.85, 1.0, abs(sin(flow * 20.0)));
    col = mix(col, col * 0.4 + vec3(0.9, 0.85, 0.7) * 0.3, lines * 0.22);

    // sparkle dots at cell centres
    col += smoothstep(0.025, 0.0, dist) * vec3(1.0, 0.95, 0.85) * 0.55;

    // metallic flecks
    float fleck = step(0.997, hash1(floor(uv * 220.0) + seed * 11.0));
    col = mix(col, vec3(0.95, 0.8, 0.35), fleck * 0.55);

    // grain / paper texture
    col += (hash1(uv * 800.0 + seed * 5.0) - 0.5) * 0.035;

    // shallow vignette so off-centre crops stay usable
    float vig = smoothstep(1.5, 0.2, length(p));
    col *= mix(0.87, 1.04, vig);

    col = pow(clamp(col, 0.0, 1.0), vec3(0.92));

    gl_FragColor = vec4(col, 1.0);
}
