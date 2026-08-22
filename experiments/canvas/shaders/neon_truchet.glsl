// style: neon_truchet
// title: Neon Truchet
// description: Fat glowing quarter-circle ribbons in mint, coral and violet snap together across a grid on near-black, weaving over and under each other so the eye keeps finding loops and long meandering runs.
// tags: truchet, tiles, arcs, neon, dark, ribbon, geometric
// author: claude (lluminate smoke2 a3a0ba7f)
precision mediump float;

varying vec2 uv;
uniform float seed;

// ---------- fixed palette (seed must not touch these) ----------
const vec3 PAL_A = vec3(0.50, 0.70, 0.50);
const vec3 PAL_B = vec3(0.50, 0.45, 0.50);
const vec3 PAL_C = vec3(1.00, 0.90, 0.90);
const vec3 PAL_D = vec3(0.00, 0.33, 0.67);
const vec3 BG = vec3(0.055, 0.05, 0.065);

// ---------- fixed structure ----------
const float GRID_N = 10.0;   // cells across the canvas

float hash1(float n) {
    return fract(sin(n) * 43758.5453123);
}

// seed enters the hash lattice only as a domain offset
vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)) + seed * 13.13,
             dot(p, vec2(269.5, 183.3)) + seed * 7.77);
    return fract(sin(p) * 43758.5453123);
}

vec3 palette(float t) {
    return PAL_A + PAL_B * cos(6.28318 * (PAL_C * t + PAL_D));
}

void main() {
    // seed tilts and slides the tiling: arrangement, not identity
    float rot = (hash1(seed * 5.7 + 1.3) - 0.5) * 0.6;
    mat2 rmat = mat2(cos(rot), -sin(rot), sin(rot), cos(rot));
    vec2 uvr = rmat * (uv - 0.5) + 0.5;
    uvr += vec2(hash1(seed * 2.1 + 4.0), hash1(seed * 3.9 + 8.0));

    vec2 p = uvr * GRID_N;
    vec2 id = floor(p);
    vec2 f = fract(p) - 0.5;

    // per-cell random values (seeded hash -> which way each tile turns)
    vec2 rnd = hash2(id);
    float orient = step(0.5, rnd.x);

    vec2 c1 = orient > 0.5 ? vec2(-0.5, -0.5) : vec2(-0.5, 0.5);
    vec2 c2 = orient > 0.5 ? vec2(0.5, 0.5) : vec2(0.5, -0.5);

    float r = 0.5;
    float d1 = abs(length(f - c1) - r);
    float d2 = abs(length(f - c2) - r);

    float thickness = 0.10 + 0.06 * hash2(id + 3.7).y;
    float edgeSoft = thickness * 0.35;

    float ribbon1 = smoothstep(thickness, thickness - edgeSoft, d1);
    float ribbon2 = smoothstep(thickness, thickness - edgeSoft, d2);

    // over/under weave alternation
    float parity = mod(id.x + id.y, 2.0);

    // colour per cell, drawn from the fixed palette
    vec3 colA = palette(hash1(dot(id, vec2(12.9, 78.2)) + seed));
    vec3 colB = palette(hash1(dot(id, vec2(45.1, 91.7)) + seed + 3.3));

    vec3 bg = BG + 0.03 * hash2(id * 0.13).x;
    vec3 col = bg;

    if (parity < 1.0) {
        col = mix(col, colB * 0.6, ribbon2);
        col = mix(col, colA, ribbon1);
    } else {
        col = mix(col, colA * 0.6, ribbon1);
        col = mix(col, colB, ribbon2);
    }

    // rounded thread shading (fake cylindrical highlight)
    float shade1 = 1.0 - smoothstep(0.0, thickness, d1);
    float shade2 = 1.0 - smoothstep(0.0, thickness, d2);
    col += 0.18 * shade1 * ribbon1;
    col += 0.18 * shade2 * ribbon2;

    // subtle woven fabric grain
    col += (hash2(p * 41.0).x - 0.5) * 0.035;

    // faint diagonal striping along the ribbons
    float stripe = sin((f.x + f.y) * 60.0) * 0.02;
    col += stripe * (ribbon1 + ribbon2);

    // shallow vignette so off-centre crops stay usable
    vec2 vig = uv - 0.5;
    col *= 1.0 - dot(vig, vig) * 0.3;

    col = pow(clamp(col, 0.0, 1.0), vec3(0.95));

    gl_FragColor = vec4(col, 1.0);
}
