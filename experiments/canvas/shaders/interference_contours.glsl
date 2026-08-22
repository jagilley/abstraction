// style: interference_contours
// title: Interference contours
// description: Overlapping wavefronts settle into smooth pools of lime, jade and violet, and the whole surface is ruled with tight black contour lines so it reads like a topographic map of a soap film.
// tags: interference, contour, topographic, iridescent, waves, isolines, green
// author: claude (lluminate smoke2 bccd1028)
precision mediump float;

varying vec2 uv;
uniform float seed;

// ---------- fixed palette (seed must not touch these) ----------
const vec3 PAL_A = vec3(0.55, 0.50, 0.50);
const vec3 PAL_B = vec3(0.45, 0.45, 0.50);
const vec3 PAL_C = vec3(1.00, 1.00, 1.00);
const vec3 PAL_D = vec3(0.00, 0.33, 0.67);

// ---------- fixed structure ----------
const float TAU = 6.2831853;
const float ISO_FREQ = 36.0;
const float WEAVE_FREQ = 40.0;
const int NW = 7;   // plane waves
const int NP = 4;   // circular sources

// seed-free hash: fixes the structural constants (wave directions, frequencies)
float hashc(float n) {
    return fract(sin(n) * 43758.5453123);
}

// seeded hash: arrangement only (phases, source positions, grain)
float hashs(float n) {
    return fract(sin(n) * 43758.5453123 + seed * 17.13);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hashs(dot(i, vec2(1.0, 57.0)));
    float b = hashs(dot(i + vec2(1.0, 0.0), vec2(1.0, 57.0)));
    float c = hashs(dot(i + vec2(0.0, 1.0), vec2(1.0, 57.0)));
    float d = hashs(dot(i + vec2(1.0, 1.0), vec2(1.0, 57.0)));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

// thin-film / diffraction-grating iridescence
vec3 iridescent(float t) {
    return PAL_A + PAL_B * cos(TAU * (PAL_C * t + PAL_D));
}

vec2 rot(vec2 v, float ang) {
    float s = sin(ang), c = cos(ang);
    return mat2(c, -s, s, c) * v;
}

void main() {
    vec2 p = uv * 2.0 - 1.0;

    // superposition of plane waves: directions and frequencies are fixed,
    // seed only slides their phases, which reorganises the interference
    float field = 0.0;
    for (int i = 0; i < NW; i++) {
        float fi = float(i);
        float ang = hashc(fi * 7.77 + 1.0) * TAU;
        float freq = 4.0 + hashc(fi * 2.19 + 2.0) * 6.0;
        float phase = hashs(fi * 4.44 + 3.0) * TAU;
        vec2 dir = vec2(cos(ang), sin(ang));
        field += cos(dot(p, dir) * freq + phase);
    }
    field /= float(NW);

    // circular sources -> concentric ring interference (Newton rings)
    float rings = 0.0;
    for (int i = 0; i < NP; i++) {
        float fi = float(i);
        vec2 center = vec2(hashs(fi * 5.5 + 4.0) - 0.5, hashs(fi * 8.8 + 5.0) - 0.5) * 2.0;
        float freq = 18.0 + hashc(fi * 2.2 + 6.0) * 26.0;
        rings += cos(length(p - center) * freq - hashs(fi * 9.9 + 7.0) * TAU);
    }
    rings /= float(NP);

    // organic domain warp
    float warp = noise(p * 2.5 + 10.0) - 0.5;
    float combo = field * 0.55 + rings * 0.35 + warp * 0.4;

    vec3 col = iridescent(combo * 0.5 + 0.5);

    // sharp isolines: grooves of a diffraction grating / stitched thread
    float iso = abs(sin(combo * ISO_FREQ));
    float threadLine = smoothstep(0.88, 1.0, iso);
    col = mix(col, vec3(0.02, 0.02, 0.03), threadLine * 0.55);

    // finer weave crossing at an angle, like warp/weft threads
    vec2 pr = rot(p, 0.9 + seed * 0.3);
    float weave = abs(sin((pr.x + pr.y) * WEAVE_FREQ + combo * 6.0));
    col = mix(col, col * 0.5 + vec3(0.9, 0.85, 0.75) * 0.25, smoothstep(0.9, 1.0, weave) * 0.3);

    // bright highlights near field extrema
    col += smoothstep(0.94, 1.0, abs(field)) * 0.18 * vec3(1.0, 0.95, 0.8);

    // cellular speckle overlay
    float speck = smoothstep(0.7, 0.9, noise(p * 40.0 + 50.0));
    col = mix(col, col * 1.15, speck * 0.15);

    // paper-like grain
    col += (hashs(dot(floor(uv * 700.0), vec2(12.9898, 78.233))) - 0.5) * 0.035;

    // occasional metallic flecks, like light catching foil threads
    float fleck = step(0.996, hashs(dot(floor(uv * 260.0), vec2(3.1, 7.7)) + 90.0));
    col = mix(col, vec3(0.95, 0.9, 0.7), fleck * 0.5);

    // shallow vignette so off-centre crops stay usable
    float vig = smoothstep(1.45, 0.15, length(p));
    col *= mix(0.85, 1.03, vig);

    col = pow(max(col, 0.0), vec3(0.9));

    gl_FragColor = vec4(col, 1.0);
}
