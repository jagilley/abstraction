// style: truchet_meander
// title: Truchet meander
// description: Pen-and-ink knotwork on warm paper: quarter-circle ribbons that join across a grid whose cells are subdivided to two or three sizes, each strand outlined and echoed by a hairline, a few picked out in madder red or prussian blue.
// tags: truchet, knot, meander, linework, ink, paper, multiscale
// brief: constructed geometric ornament; the seed chooses which cells subdivide, each cell's arc orientation and which strands take an accent
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const float BASE = 4.0;                                // coarsest cells across the canvas

const vec3 PAPER = vec3(0.925, 0.890, 0.808);
const vec3 SHADE = vec3(0.855, 0.812, 0.722);          // ribbon fill
const vec3 INK   = vec3(0.129, 0.114, 0.098);
const vec3 MADDER= vec3(0.643, 0.216, 0.169);
const vec3 PRUSS = vec3(0.129, 0.271, 0.376);

float h11(float x) { return fract(sin(x * 127.1 + 11.3) * 43758.5453123); }
float h21(vec2 p)  { return fract(sin(dot(floor(p), vec2(127.1, 311.7))) * 43758.5453123); }
float h22(vec2 p)  { return fract(sin(dot(floor(p), vec2(74.7, 219.3))) * 24634.6345); }

void main() {
    float px = fwidth(uv.x) + 1e-6;
    vec2 so = floor(vec2(h11(seed + 0.7), h11(seed * 2.9 + 6.2)) * 73.0);

    // multi-scale Truchet: subdivide a cell if its hash says so, up to two extra levels
    vec2 q = uv * BASE;
    float sc = 1.0;
    vec2 id = floor(q) + so;
    if (h21(id) < 0.62) { q *= 2.0; sc *= 2.0; id = floor(q) + so * 1.7; }
    if (h22(id) < 0.55) { q *= 2.0; sc *= 2.0; id = floor(q) + so * 2.3; }

    vec2 f = fract(q);
    float cs = 1.0 / (BASE * sc);                      // cell size in canvas units
    float ho = h21(id + vec2(13.0, 29.0));
    float ha = h22(id + vec2(5.0, 41.0));

    // two quarter arcs joining opposite edge midpoints
    vec2 c0 = ho < 0.5 ? vec2(0.0, 0.0) : vec2(1.0, 0.0);
    vec2 c1 = ho < 0.5 ? vec2(1.0, 1.0) : vec2(0.0, 1.0);
    float a0 = abs(length(f - c0) - 0.5);
    float a1 = abs(length(f - c1) - 0.5);
    float d = min(a0, a1) * cs;                        // canvas-unit distance to the strand centre line

    float W  = 0.150 * cs;                             // ribbon half width
    float EO = 0.290 * cs;                             // echo hairline offset

    // strand tint: most strands are the paper's own shade, a few are picked out
    vec3 fill = SHADE;
    if (ha > 0.90)      fill = mix(SHADE, MADDER, 0.72);
    else if (ha > 0.78) fill = mix(SHADE, PRUSS, 0.68);
    else if (ha > 0.66) fill = mix(SHADE, INK, 0.22);

    vec3 col = PAPER;

    // echo hairlines flanking each strand
    float echo = smoothstep(0.0016 + px, 0.0016 - px, abs(min(a0, a1) * cs - EO));
    col = mix(col, mix(PAPER, INK, 0.42), echo);

    // ribbon body, then its ink outline, then a hairline down the middle
    float body = smoothstep(px, -px, d - W);
    col = mix(col, fill, body);
    float edge = smoothstep(0.0026 + px, 0.0026 - px, abs(d - W));
    col = mix(col, INK, edge);
    float spine = smoothstep(0.0014 + px, 0.0014 - px, d) * 0.55;
    col = mix(col, INK, spine);

    // paper: fibre grain plus a faint blotchy wash
    float g1 = fract(sin(dot(uv * 1013.0, vec2(12.99, 78.23))) * 43758.545) - 0.5;
    float g2 = fract(sin(dot(floor(uv * 260.0), vec2(39.3, 11.1))) * 12345.678) - 0.5;
    col += (g1 * 0.030 + g2 * 0.020);
    float wash = sin(uv.x * 7.1 + 0.6) * sin(uv.y * 5.7 - 1.1) + 0.5 * sin(uv.x * 13.0 - uv.y * 9.0);
    col *= 1.0 + 0.028 * wash;

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
