// style: girih_stars
// title: Girih star lattice
// description: An Islamic-geometry tile panel of eight-pointed stars set inside octagons, with small rotated squares at every corner and thin gold strapwork running over an indigo ground.
// tags: islamic, geometric, star, tiling, ornament, strapwork
// brief: constructed geometric ornament; the seed permutes which tiles take which accent colour and which of two motifs
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const float T = 8.0;                                  // tiles across the canvas

const vec3 GROUND  = vec3(0.070, 0.110, 0.255);       // deep indigo
const vec3 CREAM   = vec3(0.941, 0.894, 0.784);       // ivory plaster
const vec3 GOLD    = vec3(0.858, 0.667, 0.278);       // brass strapwork
const vec3 INK     = vec3(0.055, 0.078, 0.165);       // outline

float h11(float x) { return fract(sin(x * 127.1 + 11.3) * 43758.5453123); }
float h21(vec2 p)  { return fract(sin(dot(floor(p), vec2(127.1, 311.7))) * 43758.5453123); }
float h22(vec2 p)  { return fract(sin(dot(floor(p), vec2(269.5, 183.3))) * 24634.6345); }

vec3 accent(float t) {
    if (t < 0.28) return vec3(0.706, 0.318, 0.212);   // terracotta
    if (t < 0.52) return vec3(0.157, 0.463, 0.451);   // verdigris
    if (t < 0.76) return vec3(0.858, 0.667, 0.278);   // gold
    return vec3(0.478, 0.157, 0.263);                 // wine
}

void main() {
    vec2 p = uv * T;
    float px = fwidth(uv.x) * T + 1e-5;

    // seed -> integer shuffle of the hash domain (arrangement only)
    vec2 so = floor(vec2(h11(seed + 1.7), h11(seed * 3.1 + 9.4)) * 97.0);

    vec2 cell = floor(p);
    vec2 f = fract(p) - 0.5;

    float s1 = max(abs(f.x), abs(f.y));
    float s2 = (abs(f.x) + abs(f.y)) * 0.70710678;
    float dOct  = max(s1, s2) - 0.5;                  // octagon tile  (<0 inside)
    float dStar = min(s1, s2) - 0.295;                // 8-pointed star inside it

    // rotated square tile sitting on every lattice corner
    vec2 g = fract(p + 0.5) - 0.5;
    float dDia = (abs(g.x) + abs(g.y)) - 0.29289;

    float ha = h21(cell + so);
    float hb = h22(cell + so * 1.31);
    float hc = h21(cell + so + vec2(41.0, 17.0));
    bool swap = hb > 0.55;                            // motif variant: star / ring inverted

    vec3 acc = accent(ha);
    vec3 starCol = swap ? CREAM : acc;
    vec3 ringCol = swap ? acc * 0.92 : CREAM;

    vec3 col = GROUND;

    // octagon body
    float inOct = smoothstep(px, -px, dOct);
    col = mix(col, ringCol, inOct);

    // star body
    float inStar = smoothstep(px, -px, dStar);
    col = mix(col, starCol, inStar);

    // small rosette centre for the swapped motif
    float rc = length(f) - 0.085;
    col = mix(col, swap ? acc : GROUND, smoothstep(px, -px, rc) * (hc > 0.4 ? 1.0 : 0.0));

    // corner square gets its own ground tone, occasionally a gold pip
    float inDia = smoothstep(px, -px, dDia);
    vec3 diaCol = mix(GROUND, GROUND * 1.9 + vec3(0.02, 0.01, 0.0), step(0.62, hc));
    col = mix(col, diaCol, inDia);
    float pip = length(g) - 0.075;
    col = mix(col, GOLD, smoothstep(px, -px, pip) * step(0.62, hc));

    // glaze: soft bevel shading so the flats read as fired ceramic, not vector fills
    float bevOct = smoothstep(0.0, 0.085, -dOct);
    col *= mix(0.80, 1.06, bevOct);
    float bevStar = smoothstep(0.0, 0.060, -dStar);
    col *= mix(1.0, 1.10, bevStar * inStar);
    float bevDia = smoothstep(0.0, 0.070, -dDia);
    col *= mix(1.0, 1.25, bevDia * inDia);

    // strapwork: gold band on every tile edge, ink hairline on the star
    float band = smoothstep(0.022 + px, 0.022 - px, abs(dOct));
    col = mix(col, GOLD * (0.92 + 0.20 * smoothstep(0.022, 0.0, abs(dOct))), band);
    float hair = smoothstep(0.013 + px, 0.013 - px, abs(dStar));
    col = mix(col, INK, hair * 0.85);
    float diaEdge = smoothstep(0.018 + px, 0.018 - px, abs(dDia));
    col = mix(col, GOLD, diaEdge * 0.75);

    // plaster grain + gentle low-frequency shading so the flats are not dead
    float grain = fract(sin(dot(uv * 811.0, vec2(12.99, 78.23))) * 43758.545) - 0.5;
    col += grain * 0.030;
    float lf = sin(uv.x * 5.3 + 1.2) * sin(uv.y * 4.1 - 0.4);
    col *= 1.0 + 0.045 * lf;

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
