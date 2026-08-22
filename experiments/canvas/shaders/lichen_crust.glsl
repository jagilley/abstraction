// style: lichen_crust
// title: Lichen crust
// description: Irregular patches of acid green, bone white and rust orange packed edge to edge, separated by ragged black crack lines, with a gritty speckle over the whole surface like lichen growing on stone.
// tags: lichen, crust, organic, voronoi, speckled, green, crackle
// author: claude (lluminate smoke2 274dc41b)
precision mediump float;

varying vec2 uv;
uniform float seed;

// ---------- fixed palette (seed must not touch these) ----------
const vec3 COL_A = vec3(0.50, 1.00, 0.50);
const vec3 COL_B = vec3(0.50, 0.45, 0.50);
const vec3 COL_C = vec3(1.00, 0.90, 0.70);
const vec3 COL_D = vec3(0.10, 0.50, 0.40);

// ---------- fixed structure ----------
const float DENSITY = 10.0;   // cells across the canvas
const float LINE_FREQ = 40.0;

// ---------- hashing / noise (seed enters only as a domain offset) ----------
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21) + seed * 17.13);
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

vec2 hash22(vec2 p) {
    float n = sin(dot(p, vec2(41.3, 289.1)) + seed * 7.77);
    return fract(vec2(262144.0, 32768.0) * n);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm(vec2 p) {
    float total = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    for (int i = 0; i < 6; i++) {
        total += noise(p * freq) * amp;
        freq *= 2.02;
        amp *= 0.55;
    }
    return total;
}

// ---------- voronoi ----------
vec3 voronoi(vec2 p) {
    vec2 ip = floor(p);
    vec2 fp = fract(p);
    float minDist1 = 8.0;
    float minDist2 = 8.0;
    vec2 cellId = vec2(0.0);

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));
            vec2 point = hash22(ip + neighbor);
            point = 0.5 + 0.5 * sin(seed * 3.0 + 6.2831 * point);
            vec2 diff = neighbor + point - fp;
            float dist = dot(diff, diff);
            if (dist < minDist1) {
                minDist2 = minDist1;
                minDist1 = dist;
                cellId = ip + neighbor;
            } else if (dist < minDist2) {
                minDist2 = dist;
            }
        }
    }
    float edge = sqrt(minDist2) - sqrt(minDist1);
    return vec3(sqrt(minDist1), edge, hash21(cellId));
}

vec3 palette(float t) {
    return COL_A + COL_B * cos(6.28318 * (COL_C * t + COL_D));
}

void main() {
    vec2 p = uv * 2.0 - 1.0;

    // seed shifts where in the pattern we are looking, never what it is made of
    vec2 shift = vec2(sin(seed * 2.17), cos(seed * 1.61)) * 11.0 + seed * 0.83;
    vec2 st = uv * DENSITY + shift;

    // domain warp using fbm for organic flow
    vec2 warp;
    warp.x = fbm(st);
    warp.y = fbm(st + vec2(5.2, 1.3));
    vec2 warped = st + (warp - 0.5) * 1.8;

    // secondary smaller warp for detail
    vec2 warp2;
    warp2.x = fbm(warped * 2.3);
    warp2.y = fbm(warped * 2.3 + 3.0);
    warped += (warp2 - 0.5) * 0.6;

    vec3 vor = voronoi(warped);

    float t = fract(vor.z + fbm(warped * 0.7) * 0.6);
    vec3 baseColor = palette(t);

    // shading based on cell distance (bump-like)
    float shade = smoothstep(0.0, 0.7, vor.x);
    baseColor *= mix(0.6, 1.15, shade);

    // edges (organic cracked look)
    float edgeWidth = 0.035 + 0.02 * fbm(warped * 3.0);
    float edgeMask = smoothstep(edgeWidth, 0.0, vor.y);
    vec3 edgeColor = mix(vec3(0.02, 0.02, 0.03), COL_C * 0.3, 0.5);
    baseColor = mix(baseColor, edgeColor, edgeMask);

    // fine circuit-like linework overlay using sine grids modulated by fbm
    float lines = abs(sin((warped.x + warped.y * 0.5 + fbm(warped * 1.5) * 4.0) * LINE_FREQ));
    float lineMask = smoothstep(0.97, 1.0, lines) * (1.0 - edgeMask) * 0.4;
    baseColor += lineMask * vec3(1.0, 0.95, 0.85);

    // subtle grain
    float grain = hash21(uv * 800.0);
    baseColor += (grain - 0.5) * 0.03;

    // gentle vignette (kept shallow so off-centre crops stay usable)
    float vig = smoothstep(1.45, 0.2, length(p));
    baseColor *= mix(0.88, 1.04, vig);

    baseColor = pow(clamp(baseColor, 0.0, 1.0), vec3(0.9));

    gl_FragColor = vec4(baseColor, 1.0);
}
