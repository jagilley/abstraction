// style: screenprint_tints
// title: Screenprinted tints
// description: A two-colour screenprint on manila stock: hard-edged circles, bars, rings, wedges and quarter discs on a grid, each pulled either solid or at a stepped 25, 50 or 75 per cent tint, so a coarse square dot screen shows through the vermilion and the ink blue.
// tags: screenprint, halftone, square-dot, tints, geometric, grid, two-colour, poster
// brief: constructed geometric ornament; the seed picks each cell's shape, rotation, ink and tint step
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const float G = 5.0;                                   // shape cells across
const float FREQ = 40.0;                               // screen rulings across the canvas

const vec3 PAPER = vec3(0.882, 0.831, 0.710);          // manila
const vec3 TA    = vec3(0.878, 0.294, 0.133);          // vermilion, as a transmittance
const vec3 TB    = vec3(0.075, 0.145, 0.310);          // ink blue

float h11(float x) { return fract(sin(x * 127.1 + 11.3) * 43758.5453123); }
float h21(vec2 p)  { return fract(sin(dot(floor(p), vec2(127.1, 311.7))) * 43758.5453123); }
float h22(vec2 p)  { return fract(sin(dot(floor(p), vec2(74.7, 219.3))) * 24634.6345); }

vec2 rot(vec2 p, float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c) * p; }

// coverage demanded of each of the two inks, already stepped to printable tints
vec2 covers(vec2 w, vec2 so) {
    vec2 p = w * G;
    vec2 cell = floor(p);
    vec2 f = fract(p) - 0.5;
    vec2 cs = cell + so;
    float h1 = h21(cs), h2 = h22(cs), h3 = h21(cs + vec2(7.0, 3.0)), h4 = h22(cs + vec2(23.0, 5.0));

    vec2 g = rot(f, floor(h2 * 4.0) * 1.5707963);

    float d;
    float t = h1 * 6.0;
    if      (t < 1.0) d = length(g) - 0.36;                                  // disc
    else if (t < 2.0) d = abs(length(g) - 0.31) - 0.095;                     // ring
    else if (t < 3.0) d = max(length(g) - 0.44, -g.y);                       // half disc
    else if (t < 4.0) d = max(abs(g.x) - 0.15, abs(g.y) - 0.46);             // bar
    else if (t < 5.0) d = length(g + vec2(0.5, 0.5)) - 0.82;                 // quarter
    else              d = max(max(-g.x - 0.44, -g.y - 0.44), (g.x + g.y) - 0.04);  // wedge

    float shape = step(d, 0.0);

    // stepped tints, the way a printer would specify them
    float tint = floor(h4 * 3.0) * 0.25 + 0.50;        // 50 / 75 / 100 %
    float ground = 0.10 + 0.11 * step(0.62, h3);       // the cell's flat backing tint

    float toA = step(h3, 0.50);
    vec2 c = vec2(shape * tint * toA, shape * tint * (1.0 - toA));
    if (h4 > 0.82) c = vec2(shape * 0.75, shape * 0.50);   // a few cells overprint both
    c += vec2(ground * (1.0 - toA), ground * toA) * 0.85;
    return clamp(c, 0.0, 1.0);
}

// square dots on a rotated screen
float screenSq(vec2 w, float cov, float ang, float px) {
    vec2 r = rot(w, ang) * FREQ;
    vec2 g = fract(r) - 0.5;
    float rad = 0.525 * sqrt(clamp(cov, 0.0, 1.0));
    float d = (max(abs(g.x), abs(g.y)) - rad) / FREQ;
    return smoothstep(px * 0.8, -px * 0.8, d);
}

void main() {
    float px = fwidth(uv.x) + 1e-6;
    vec2 so = floor(vec2(h11(seed + 4.3), h11(seed * 1.9 + 8.8)) * 61.0);

    vec2 cA = covers(uv, so);
    vec2 cB = covers(uv + vec2(0.0028, -0.0022), so);   // the second pull is a hair out

    float a = screenSq(uv, cA.x, 0.7854, px);           // 45 degrees
    float b = screenSq(uv + vec2(0.0025, 0.0018), cB.y, 0.2618, px);   // 15 degrees

    // squeegee: ink lays down a little unevenly across the sheet
    float pull = 0.94 + 0.10 * sin(uv.y * 5.0 + sin(uv.x * 2.0) * 1.5);
    vec3 col = PAPER;
    col *= mix(vec3(1.0), TA, a * pull);
    col *= mix(vec3(1.0), TB, b * pull);

    // stock: visible fibre and a soft mottle
    float grain = fract(sin(dot(uv * 1201.0, vec2(12.99, 78.23))) * 43758.545) - 0.5;
    float fib = fract(sin(dot(floor(uv * 300.0), vec2(39.3, 11.1))) * 12345.678) - 0.5;
    col += grain * 0.030 + fib * 0.022;
    col *= 1.0 - 0.045 * (0.5 + 0.5 * sin(uv.x * 8.0) * sin(uv.y * 6.0));

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
