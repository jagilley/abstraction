// style: stained_glass_cells
// title: Stained glass cells
// description: Rounded organic panes of olive, cream and burnt umber, each holding its own set of concentric rings and faint radiating spokes, held together by heavy black leading with a thin highlight along every seam.
// tags: stained-glass, cells, voronoi, rings, leading, olive, cream
// author: claude (lluminate smoke2 bf43c880)
precision mediump float;

varying vec2 uv;
uniform float seed;

// ---------- fixed palette (seed must not touch these) ----------
const vec3 PAL_A = vec3(0.500, 0.814, 0.197);
const vec3 PAL_B = vec3(0.500, 0.500, 0.500);
const vec3 PAL_C = vec3(1.000, 0.900, 0.600);
const vec3 PAL_D = vec3(0.150, 0.400, 0.600);

// ---------- fixed structure ----------
const float SCALE = 8.0;     // panes across the canvas
const float RING_FREQ = 22.0;

// seed enters the hash lattice only as a domain offset
vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453123 + seed * 17.13);
}

float hash1(vec2 p) {
    return fract(sin(dot(p, vec2(41.3, 289.1)) + seed * 7.919) * 43758.5453);
}

// returns vec4(edgeDist, cellId.xy, centerDist)
vec4 voronoi(vec2 x) {
    vec2 n = floor(x);
    vec2 f = fract(x);

    vec2 mg = vec2(0.0), mr = vec2(0.0);
    float md = 8.0;

    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 g = vec2(float(i), float(j));
            vec2 o = hash2(n + g);
            vec2 r = g + o - f;
            float d = dot(r, r);
            if (d < md) { md = d; mr = r; mg = g; }
        }
    }

    float centerDist = sqrt(md);
    float med = 8.0;

    for (int j = -2; j <= 2; j++) {
        for (int i = -2; i <= 2; i++) {
            vec2 g = mg + vec2(float(i), float(j));
            vec2 o = hash2(n + g);
            vec2 r = g + o - f;
            vec2 diff = r - mr;
            float dd = dot(diff, diff);
            if (dd > 0.00001) {
                med = min(med, dot(0.5 * (mr + r), normalize(diff)));
            }
        }
    }

    return vec4(med, mg + n, centerDist);
}

vec3 pal(float t, vec3 d) {
    return PAL_A + PAL_B * cos(6.28318 * (PAL_C * t + d));
}

void main() {
    // seed slides the lattice: arrangement, not identity
    vec2 shift = vec2(sin(seed * 1.43), cos(seed * 2.11)) * 7.0 + seed * 0.61;
    vec2 p = uv * SCALE + shift;

    // domain warp for organic flow
    vec2 warp = vec2(
        sin(p.y * 2.1 + seed * 3.0) * 0.35 + sin(p.x * 0.7 + seed) * 0.15,
        cos(p.x * 2.3 + seed * 5.0) * 0.35 + cos(p.y * 0.6 + seed * 2.0) * 0.15
    );
    vec2 pw = p + warp;

    vec4 vor = voronoi(pw);
    float edgeDist = vor.x;
    vec2 cellId = vor.yz;
    float centerDist = vor.w;

    float cellHash = hash1(cellId);
    float cellHash2 = hash1(cellId + 13.7);

    vec3 baseColor = pal(cellHash, PAL_D);

    // concentric organic rings inside each pane
    float rings = sin(centerDist * RING_FREQ + cellHash2 * 6.2831) * 0.5 + 0.5;
    vec3 col = mix(baseColor * 0.75, baseColor * 1.25, rings);

    // radial fine lines for texture
    float ang = atan(pw.y - floor(pw.y) - 0.5, pw.x - floor(pw.x) - 0.5);
    col *= 0.9 + 0.1 * (sin(ang * 10.0 + cellHash * 20.0) * 0.5 + 0.5);

    // leading between panes
    float edge = smoothstep(0.0, 0.06, edgeDist);
    col = mix(vec3(0.03, 0.03, 0.05), col, edge);

    // subtle highlight along the seam
    float highlight = smoothstep(0.06, 0.02, edgeDist) * (1.0 - edge);
    col += highlight * vec3(0.9, 0.85, 0.7) * 0.15;

    // slow tint drift, tied to the pattern rather than to screen position
    vec3 tint = pal(0.06 * (pw.x + pw.y), PAL_D + 0.3);
    col = mix(col, col * tint, 0.15);

    // fine grain
    col += (hash1(gl_FragCoord.xy * 0.7 + seed * 91.0) - 0.5) * 0.035;

    // shallow vignette so off-centre crops stay usable
    vec2 c = uv - 0.5;
    col *= 1.0 - dot(c, c) * 0.35;

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
