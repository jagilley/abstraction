// style: quilt_patch
// title: Patchwork quilt blocks
// description: A pieced cotton quilt: a grid of blocks — pinwheels, nine-patches, squares-in-squares, Ohio stars and flying geese — cut from printed calicoes in indigo, madder and mustard, with pressed seams and running-stitch quilting.
// tags: quilt, patchwork, textile, blocks, triangles, geometric, folk
// brief: constructed geometric ornament; the seed chooses each block's pattern, its rotation and which three fabrics it is pieced from
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const float T = 5.0;                                   // blocks across the canvas

float h11(float x) { return fract(sin(x * 127.1 + 11.3) * 43758.5453123); }
float h21(vec2 p)  { return fract(sin(dot(floor(p), vec2(127.1, 311.7))) * 43758.5453123); }
float h22(vec2 p)  { return fract(sin(dot(floor(p), vec2(74.7, 219.3))) * 24634.6345); }
float h23(vec2 p)  { return fract(sin(dot(floor(p), vec2(21.9, 97.7))) * 15731.743); }

vec2 rot90(vec2 p, float k) {
    p -= 0.5;
    float c = cos(k * 1.5707963), s = sin(k * 1.5707963);
    return mat2(c, -s, s, c) * p + 0.5;
}

// ---- block patterns: each returns a fabric slot in {0,1,2} ----------------

float bPinwheel(vec2 f) {
    vec2 q = f * 2.0;
    vec2 u = floor(q);
    float qi = (u.y < 0.5) ? (u.x < 0.5 ? 0.0 : 1.0) : (u.x < 0.5 ? 3.0 : 2.0);
    vec2 s = rot90(fract(q), qi);
    return step(1.0, s.x + s.y);
}

float bNine(vec2 f) {
    vec2 u = floor(f * 3.0);
    return mod(u.x + u.y, 2.0);
}

float bSquares(vec2 f) {
    vec2 c = abs(f - 0.5);
    float r = max(max(c.x, c.y), (c.x + c.y) * 0.70710678);
    return mod(floor(r * 3.5), 2.0);
}

float bStar(vec2 f) {
    vec2 q = f * 3.0;
    vec2 u = floor(q);
    vec2 s = fract(q);
    float cx = abs(u.x - 1.0) < 0.5 ? 1.0 : 0.0;
    float cy = abs(u.y - 1.0) < 0.5 ? 1.0 : 0.0;
    if (cx > 0.5 && cy > 0.5) return 2.0;              // centre square
    if (cx < 0.5 && cy < 0.5) return 0.0;              // corner squares
    // edge unit: a triangle with its apex pointing at the centre
    vec2 d = vec2(1.0, 1.0) - u;                       // axis-aligned, length 1
    float k = (d.y > 0.5) ? 0.0 : (d.x < -0.5 ? 1.0 : (d.y < -0.5 ? 2.0 : 3.0));
    vec2 t = rot90(s, k);
    return step(t.y, 1.0 - 2.0 * abs(t.x - 0.5));
}

float bGeese(vec2 f) {
    vec2 q = vec2(f.x * 2.0, f.y * 4.0);
    vec2 u = floor(q);
    vec2 s = fract(q);
    if (mod(u.y, 2.0) > 0.5) s.y = 1.0 - s.y;
    return step(s.y, 1.0 - 2.0 * abs(s.x - 0.5));
}

float blockSlot(vec2 f, float ht, float hr) {
    f = clamp(rot90(f, floor(hr * 4.0)), 0.0, 0.99999);
    float t = ht * 5.0;
    if (t < 1.0) return bPinwheel(f);
    if (t < 2.0) return bNine(f);
    if (t < 3.0) return bSquares(f);
    if (t < 4.0) return bStar(f);
    return bGeese(f);
}

// ---- fabrics: a fixed bolt of cloth per index, colour and print together ---

vec3 fabricBase(float i) {
    if (i < 0.5) return vec3(0.933, 0.906, 0.831);     // muslin
    if (i < 1.5) return vec3(0.855, 0.827, 0.729);     // ecru
    if (i < 2.5) return vec3(0.769, 0.796, 0.784);     // pale grey-blue
    if (i < 3.5) return vec3(0.118, 0.180, 0.337);     // indigo
    if (i < 4.5) return vec3(0.588, 0.196, 0.169);     // madder
    if (i < 5.5) return vec3(0.157, 0.161, 0.176);     // charcoal
    if (i < 6.5) return vec3(0.176, 0.376, 0.396);     // teal
    if (i < 7.5) return vec3(0.294, 0.365, 0.259);     // sage
    if (i < 8.5) return vec3(0.780, 0.596, 0.208);     // mustard
    return vec3(0.706, 0.494, 0.475);                  // dusty rose
}

vec3 fabric(float i, vec2 w, float px) {
    vec3 c = fabricBase(i);
    float pt = mod(i, 4.0);
    float tint = dot(c, vec3(0.33)) > 0.5 ? -0.16 : 0.20;
    float m = 0.0;
    if (pt < 0.5) {
        m = 0.0;                                       // solid
    } else if (pt < 1.5) {
        vec2 g = fract(w * 46.0 + vec2(0.0, 0.5 * step(0.5, fract(w.y * 23.0)))) - 0.5;
        float dd = length(g) / 46.0 - 0.006;
        m = smoothstep(px, -px, dd);                   // sprigged dots
    } else if (pt < 2.5) {
        float st = abs(fract((w.x + w.y) * 34.0) - 0.5) * 2.0;
        float sw = clamp(px * 34.0 * 2.2, 0.03, 1.0);
        m = smoothstep(0.45 - sw, 0.45 + sw, st) * smoothstep(0.7, 0.2, sw);
    } else {
        float a = abs(fract(w.x * 30.0) - 0.5) * 2.0;
        float b = abs(fract(w.y * 30.0) - 0.5) * 2.0;
        float sw = clamp(px * 30.0 * 2.2, 0.03, 1.0);
        m = max(smoothstep(0.80 - sw, 0.80 + sw, a), smoothstep(0.80 - sw, 0.80 + sw, b))
            * smoothstep(0.7, 0.2, sw);
    }
    return c * (1.0 + tint * m);
}

void main() {
    float px = fwidth(uv.x) + 1e-6;
    vec2 so = floor(vec2(h11(seed + 5.1), h11(seed * 6.3 + 2.4)) * 67.0);

    vec2 p = uv * T;
    vec2 blk = floor(p);
    vec2 f = fract(p);
    vec2 bs = blk + so;

    float ht = h21(bs);
    float hr = h22(bs);
    float f0 = floor(h23(bs) * 3.0);                       // a light
    float f1 = 3.0 + floor(h21(bs + vec2(13.0, 7.0)) * 5.0);  // a dark
    float f2 = 6.0 + floor(h22(bs + vec2(3.0, 29.0)) * 4.0);  // an accent

    float slot = blockSlot(f, ht, hr);
    float fi = slot < 0.5 ? f0 : (slot < 1.5 ? f1 : f2);
    vec3 col = fabric(fi, uv, px);

    // seams: sample the slot field around this point; a change means two patches meet
    float e = px * T * 1.0;
    float sd = abs(blockSlot(f + vec2(e, 0.0), ht, hr) - slot)
             + abs(blockSlot(f - vec2(e, 0.0), ht, hr) - slot)
             + abs(blockSlot(f + vec2(0.0, e), ht, hr) - slot)
             + abs(blockSlot(f - vec2(0.0, e), ht, hr) - slot);
    float seam = clamp(sd, 0.0, 1.0);
    col *= 1.0 - 0.30 * seam;

    // block-to-block sashing seam
    vec2 be = min(f, 1.0 - f);
    float bseam = smoothstep(0.020 + px * T, 0.020 - px * T, min(be.x, be.y));
    col *= 1.0 - 0.22 * bseam;

    // running-stitch quilting: dashed thread a little inside every block edge
    float ring = abs(min(be.x, be.y) - 0.11);
    float along = (be.x < be.y) ? f.y : f.x;
    float dash = step(0.42, fract(along * 26.0));
    float stitch = smoothstep(0.012 + px * T, 0.012 - px * T, ring) * dash;
    col = mix(col, vec3(0.949, 0.929, 0.878), stitch * 0.45);
    col *= 1.0 - 0.25 * smoothstep(0.030 + px * T, 0.030 - px * T, ring) * (1.0 - stitch) * 0.35;

    // cloth: fine weave and a soft quilted loft
    float wv = (abs(fract(uv.x * 320.0) - 0.5) + abs(fract(uv.y * 320.0) - 0.5)) - 0.5;
    col *= 1.0 + 0.05 * wv * smoothstep(0.006, 0.001, px);
    float loft = sin((f.x - 0.5) * 3.1416) * sin((f.y - 0.5) * 3.1416);
    col *= 0.94 + 0.10 * loft;
    float grain = fract(sin(dot(uv * 877.0, vec2(12.99, 78.23))) * 43758.545) - 0.5;
    col += grain * 0.028;

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
