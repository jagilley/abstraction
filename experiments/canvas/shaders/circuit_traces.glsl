// style: circuit_traces
// title: Circuit traces
// description: A printed-circuit panel in dark solder-mask green: gold traces routed between octagonal pads and drilled vias, black chip bodies with white silkscreen outlines, and a hatched copper pour behind it all.
// tags: circuit, pcb, technical, grid, routing, gold, geometric
// brief: constructed geometric ornament; the seed rewires which edges of the grid are routed and where pads, vias and chips fall
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const float T = 11.0;                                   // routing cells across the canvas

const vec3 MASK  = vec3(0.043, 0.196, 0.145);          // solder mask
const vec3 POUR  = vec3(0.075, 0.267, 0.196);          // copper pour under the mask
const vec3 GOLD  = vec3(0.855, 0.686, 0.290);
const vec3 GOLD2 = vec3(0.973, 0.878, 0.588);          // specular on the plating
const vec3 SILK  = vec3(0.902, 0.914, 0.878);
const vec3 BODY  = vec3(0.086, 0.082, 0.086);          // chip package

float h11(float x) { return fract(sin(x * 127.1 + 11.3) * 43758.5453123); }
float h21(vec2 p)  { return fract(sin(dot(floor(p), vec2(127.1, 311.7))) * 43758.5453123); }
float h22(vec2 p)  { return fract(sin(dot(floor(p), vec2(74.7, 219.3))) * 24634.6345); }
float h23(vec2 p)  { return fract(sin(dot(floor(p), vec2(21.9, 97.7))) * 15731.743); }

float sdSeg(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

void main() {
    vec2 p = uv * T;
    float px = fwidth(uv.x) * T + 1e-5;
    vec2 so = floor(vec2(h11(seed + 3.3), h11(seed * 4.7 + 1.6)) * 83.0);

    vec2 cell = floor(p);
    vec2 f = fract(p) - 0.5;
    vec2 cs = cell + so;

    // edge hashes are shared with the neighbour, so every routed trace actually connects
    float eL = h22(cs);
    float eR = h22(cs + vec2(1.0, 0.0));
    float eD = h23(cs);
    float eU = h23(cs + vec2(0.0, 1.0));
    float OPEN = 0.57;

    float d = 1e3;
    if (eL < OPEN) d = min(d, sdSeg(f, vec2(0.0), vec2(-0.5, 0.0)));
    if (eR < OPEN) d = min(d, sdSeg(f, vec2(0.0), vec2( 0.5, 0.0)));
    if (eD < OPEN) d = min(d, sdSeg(f, vec2(0.0), vec2(0.0, -0.5)));
    if (eU < OPEN) d = min(d, sdSeg(f, vec2(0.0), vec2(0.0,  0.5)));
    float deg = step(eL, OPEN) + step(eR, OPEN) + step(eD, OPEN) + step(eU, OPEN);

    // a family of coarse diagonal buses, only some of which are populated
    float bu = (uv.x + uv.y) * (T * 0.5) + 0.31;
    float bidx = floor(bu);
    float bfr = abs(fract(bu) - 0.5) * 2.0;
    float bw = clamp(fwidth(uv.x) * T, 0.02, 1.0);
    float bus = smoothstep(0.80 - bw, 0.86 + bw, bfr) * step(0.74, h21(vec2(bidx, 7.0) + so));

    float hc = h21(cs + vec2(9.0, 17.0));
    float hd = h21(cs + vec2(41.0, 3.0));

    // copper pour: broad regions of 45-degree hatch behind everything
    float pourMask = step(0.0, sin(uv.x * 6.1 + 1.1) + sin(uv.y * 4.7 - 0.5) + 0.95);
    float hatch = abs(fract((uv.x + uv.y) * 56.0) - 0.5) * 2.0;
    float hw = clamp(fwidth(uv.x) * 56.0 * 2.2, 0.03, 1.0);
    vec3 col = mix(MASK, POUR, pourMask * smoothstep(0.5 - hw, 0.5 + hw, hatch) * smoothstep(0.6, 0.15, hw));
    col = mix(col, POUR * 0.9, pourMask * 0.35);

    // diagonal buses run under the routing layer
    col = mix(col, GOLD * 0.60, bus * 0.85);

    // traces
    float trace = smoothstep(px, -px, d - 0.082);
    col = mix(col, GOLD * 0.92, trace);
    float traceHi = smoothstep(px, -px, d - 0.030);
    col = mix(col, GOLD2 * 0.85, traceHi * 0.5);

    // octagonal pad wherever a trace turns or branches
    float s1 = max(abs(f.x), abs(f.y));
    float s2 = (abs(f.x) + abs(f.y)) * 0.70710678;
    float pad = max(s1, s2) - (hd < 0.30 ? 0.26 : 0.17);
    float hasPad = step(1.5, deg) * step(0.18, hc);
    float inPad = smoothstep(px, -px, pad) * hasPad;
    col = mix(col, GOLD, inPad);
    col = mix(col, GOLD2, inPad * smoothstep(0.10, -0.06, pad) * 0.55);
    // drilled via
    float hole = length(f) - 0.075;
    col = mix(col, MASK * 0.55, smoothstep(px, -px, hole) * hasPad * step(hd, 0.42));

    // chip packages with silkscreen outlines and gold legs
    if (hc > 0.90) {
        vec2 e = abs(f) - (hd < 0.5 ? vec2(0.36, 0.22) : vec2(0.22, 0.36));
        float box = min(max(e.x, e.y), 0.0) + length(max(e, 0.0));
        col = mix(col, BODY, smoothstep(px, -px, box));
        col = mix(col, SILK, smoothstep(0.012 + px, 0.012 - px, abs(box - 0.055)) * 0.85);
    }

    // silkscreen tick marks and reference circles, sparse
    if (hc > 0.62 && hc < 0.70) {
        float ring = abs(length(f) - 0.30) - 0.012;
        col = mix(col, SILK, smoothstep(px, -px, ring) * 0.8);
    }

    // fibreglass speckle and a soft sheen
    float grain = fract(sin(dot(uv * 1301.0, vec2(12.99, 78.23))) * 43758.545) - 0.5;
    col += grain * 0.026;
    col *= 1.0 + 0.10 * sin(uv.x * 3.1 - uv.y * 2.3);

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
