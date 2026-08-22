// style: paper_collage
// title: Torn paper collage
// description: Overlapping scraps of coloured paper in brick red, ochre, teal and bone, each ringed with faint contour lines and outlined in ink, layered edge over edge with occasional gold flecks.
// tags: collage, paper, torn, ochre, teal, ink, contour, gold
// author: claude (lluminate smoke2 bdbbf7af)
precision mediump float;

varying vec2 uv;
uniform float seed;

// ---------- fixed palette (seed must not touch these) ----------
const vec3 C0 = vec3(0.08, 0.06, 0.12);
const vec3 C1 = vec3(0.78, 0.25, 0.22);
const vec3 C2 = vec3(0.85, 0.65, 0.20);
const vec3 C3 = vec3(0.10, 0.35, 0.38);
const vec3 C4 = vec3(0.92, 0.90, 0.82);
const vec3 C5 = vec3(0.35, 0.15, 0.35);

// ---------- fixed structure ----------
const float WARP_SCALE = 3.4;
const float CELL_SCALE = 2.2;

// seed enters the hash lattice only as a domain offset
vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)) + seed * 17.0,
             dot(p, vec2(269.5, 183.3)) + seed * 7.0);
    return fract(sin(p) * 43758.5453123);
}

float hash1(vec2 p) {
    return fract(sin(dot(p, vec2(41.3, 289.1)) + seed * 13.7) * 43758.5453);
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
            pt = 0.5 + 0.5 * sin(6.2831 * pt + seed);
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
    return vec4(sqrt(minD), sqrt(secD), hash1(minCell * 1.13), 0.0);
}

// fixed six-stop ramp: which scrap gets which colour may change, the inks may not
vec3 palette(float t) {
    if (t < 0.2) return mix(C0, C1, t / 0.2);
    if (t < 0.4) return mix(C1, C2, (t - 0.2) / 0.2);
    if (t < 0.6) return mix(C2, C3, (t - 0.4) / 0.2);
    if (t < 0.8) return mix(C3, C4, (t - 0.6) / 0.2);
    return mix(C4, C5, (t - 0.8) / 0.2);
}

void main() {
    vec2 p = uv * 2.0 - 1.0;

    // domain warp; seed only slides where in the field we are
    vec2 wp = p * WARP_SCALE + vec2(seed * 3.0, -seed * 2.0);
    vec2 warp = vec2(fbm(wp + vec2(1.7, 4.2)), fbm(wp + vec2(9.2, 2.1)));
    vec2 warped = wp + (warp - 0.5) * 1.6;

    vec4 vor = voronoi(warped * CELL_SCALE);
    float dist = vor.x;
    float edge = smoothstep(0.0, 0.06, vor.y - vor.x);
    float cellH = vor.z;

    // scrap colour: a per-cell draw from the fixed ramp
    vec3 base = palette(fract(cellH * 1.7 + seed * 0.13));

    // contour rings inside each scrap
    float ring = pow(sin(dist * 28.0 - fbm(warped * 1.5) * 4.0) * 0.5 + 0.5, 2.0);
    vec3 col = base * (0.6 + 0.4 * ring);

    // ink edges between scraps
    col = mix(col, vec3(0.03, 0.02, 0.04), (1.0 - edge) * 0.85);

    // fine flow lines overlay
    float flow = fbm(p * 8.0 + warp * 2.0 + seed);
    float lines = smoothstep(0.85, 1.0, abs(sin(flow * 20.0)));
    col = mix(col, col * 0.4 + vec3(0.9, 0.85, 0.7) * 0.3, lines * 0.25);

    // grain / paper texture
    col += (hash1(uv * 800.0 + seed * 5.0) - 0.5) * 0.04;

    // subtle gold flecks
    float fleck = step(0.997, hash1(floor(uv * 220.0) + seed * 11.0));
    col = mix(col, vec3(0.95, 0.8, 0.35), fleck * 0.6);

    // shallow vignette so off-centre crops stay usable
    float vig = smoothstep(1.45, 0.2, length(p));
    col *= mix(0.87, 1.03, vig);

    col = pow(clamp(col, 0.0, 1.0), vec3(0.92));

    gl_FragColor = vec4(col, 1.0);
}
