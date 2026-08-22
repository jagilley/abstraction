// style: mosaic_tesserae
// title: Glass tesserae
// description: A Byzantine wall mosaic of small square glass tesserae bedded in dark grout, laid in rows that change direction from patch to patch, in lapis, verdigris, marble and gold leaf that catches the light.
// tags: mosaic, tesserae, tiling, byzantine, glass, gold, grout
// brief: constructed geometric ornament; the seed turns the rows in each patch and moves the colour fields the tesserae are drawn from
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const float R = 4.0;                                   // andamento patches across the canvas
const float F = 22.0;                                  // tesserae across the canvas
const float PI = 3.14159265;

const vec3 GROUT = vec3(0.129, 0.118, 0.110);

float h11(float x) { return fract(sin(x * 127.1 + 11.3) * 43758.5453123); }
float h21(vec2 p)  { return fract(sin(dot(floor(p), vec2(127.1, 311.7))) * 43758.5453123); }
float h22(vec2 p)  { return fract(sin(dot(floor(p), vec2(74.7, 219.3))) * 24634.6345); }
float h23(vec2 p)  { return fract(sin(dot(floor(p), vec2(21.9, 97.7))) * 15731.743); }

// five families of closely related glasses; k picks the member
vec3 glass(float fam, float k) {
    if (fam < 1.0) {                                   // lapis
        if (k < 0.34) return vec3(0.078, 0.153, 0.396);
        if (k < 0.67) return vec3(0.153, 0.271, 0.541);
        return vec3(0.310, 0.435, 0.671);
    } else if (fam < 2.0) {                            // gold leaf
        if (k < 0.34) return vec3(0.804, 0.639, 0.243);
        if (k < 0.67) return vec3(0.910, 0.769, 0.361);
        return vec3(0.667, 0.482, 0.180);
    } else if (fam < 3.0) {                            // verdigris
        if (k < 0.34) return vec3(0.129, 0.325, 0.302);
        if (k < 0.67) return vec3(0.220, 0.451, 0.400);
        return vec3(0.376, 0.569, 0.475);
    } else if (fam < 4.0) {                            // porphyry
        if (k < 0.34) return vec3(0.435, 0.169, 0.184);
        if (k < 0.67) return vec3(0.588, 0.271, 0.224);
        return vec3(0.361, 0.212, 0.243);
    }
    if (k < 0.34) return vec3(0.855, 0.831, 0.769);    // marble
    if (k < 0.67) return vec3(0.749, 0.729, 0.678);
    return vec3(0.918, 0.902, 0.855);
}

void main() {
    float px = fwidth(uv.x) * F + 1e-5;
    vec2 so = floor(vec2(h11(seed + 6.6), h11(seed * 9.7 + 2.2)) * 59.0);
    float sph = h11(seed * 3.7 + 0.4) * 6.2831;

    // which andamento patch, and which way its rows run
    vec2 rc = floor(uv * R) + so;
    float th = floor(h21(rc) * 6.0) * (PI / 6.0) + 0.09;
    vec2 dir = vec2(cos(th), sin(th));
    vec2 nor = vec2(-dir.y, dir.x);

    float across = dot(uv, nor) * F;
    float row = floor(across);
    float jit = h22(vec2(row, dot(rc, vec2(1.0, 7.0))));
    float along = dot(uv, dir) * F + jit;
    float col_i = floor(along);

    vec2 f = vec2(fract(along), fract(across)) - 0.5;
    vec2 tid = vec2(col_i, row) + rc * 3.0;

    // tessera outline: a slightly irregular rounded square
    float hw = 0.355 + 0.055 * h21(tid + vec2(3.0, 11.0));
    float hh = 0.355 + 0.055 * h22(tid + vec2(7.0, 5.0));
    vec2 e = abs(f) - vec2(hw, hh);
    float d = min(max(e.x, e.y), 0.0) + length(max(e, 0.0)) - 0.045;

    // colour field: broad drifts of one family into the next
    vec2 wf = uv + vec2(h11(seed * 2.1 + 0.9), h11(seed * 4.3 + 7.7)) * 9.0;
    float fieldv = sin(wf.x * 6.9 + sph) + sin(wf.y * 5.3 - sph * 0.6)
                 + 0.85 * sin((wf.x + wf.y) * 9.7) + 0.65 * sin((wf.x - wf.y) * 7.9);
    float fam = fieldv < -1.55 ? 0.0 : (fieldv < -0.50 ? 1.0 : (fieldv < 0.50 ? 2.0 :
               (fieldv < 1.55 ? 3.0 : 4.0)));
    if (h23(tid) > 0.87) fam = mod(fam + 2.0, 5.0);     // stray tesserae from another pot
    vec3 base = glass(fam, h21(tid + vec2(23.0, 2.0)));
    base *= 0.90 + 0.20 * h22(tid + vec2(17.0, 31.0));

    vec3 col = GROUT * (0.85 + 0.35 * h21(tid + vec2(41.0, 13.0)));

    float inTile = smoothstep(px, -px, d);
    // bedded: the grout darkens right against each tessera
    col *= 1.0 - 0.45 * smoothstep(0.10, 0.0, d) * (1.0 - inTile);

    // the glass, with a bevelled edge lit from the upper left
    float bev = smoothstep(0.0, 0.10, -d);
    vec3 tile = base * (0.72 + 0.36 * bev);
    float lip = smoothstep(0.09, 0.0, -d) * clamp(dot(normalize(f + 1e-4), normalize(vec2(-1.0, 1.0))), 0.0, 1.0);
    tile += (0.18 + 0.30 * step(1.0, fam) * step(fam, 1.0)) * lip;
    // a specular glint, strongest on the gold
    float gl = smoothstep(0.26, 0.0, length(f - vec2(-0.13, 0.13)));
    tile += gl * (fam < 1.5 && fam > 0.5 ? 0.38 : 0.10) * (0.5 + 0.5 * h23(tid + vec2(9.0, 9.0)));
    col = mix(col, tile, inTile);

    // wall: uneven light across the whole surface, plus grit
    col *= 0.90 + 0.18 * (0.5 + 0.5 * sin(uv.x * 2.7 + 1.0) * sin(uv.y * 2.1 - 0.3));
    float grit = fract(sin(dot(uv * 1601.0, vec2(12.99, 78.23))) * 43758.545) - 0.5;
    col += grit * 0.026;

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
