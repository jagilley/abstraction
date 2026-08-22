// style: agate_bands
// title: Banded agate
// description: Slices of banded agate — nested rings of ochre, rust and cream crowding around irregular cavities, with the occasional cool chalcedony band.
// tags: mineral, agate, bands, concentric, warm, geode
// brief: organic and natural textures; a softly shaded mineral cross-section
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const vec3 C0 = vec3(0.216, 0.129, 0.086);  // umber
const vec3 C1 = vec3(0.451, 0.243, 0.129);  // sienna
const vec3 C2 = vec3(0.671, 0.412, 0.180);  // amber
const vec3 C3 = vec3(0.851, 0.639, 0.353);  // honey
const vec3 C4 = vec3(0.949, 0.894, 0.796);  // cream
const vec3 C5 = vec3(0.663, 0.769, 0.796);  // chalcedony blue

float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
vec2 hash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash21(i), b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0)), d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbm(vec2 p) {
    float s = 0.0, a = 0.5;
    for (int k = 0; k < 5; k++) { s += a * vnoise(p); p = p * 2.03 + vec2(11.3, 7.7); a *= 0.5; }
    return s;
}

vec3 band_colour(float h) {
    if (h < 0.22) return C1;
    if (h < 0.44) return C2;
    if (h < 0.62) return C3;
    if (h < 0.78) return C0;
    if (h < 0.955) return C4;
    return C5;
}

void main() {
    vec2 so = vec2(hash21(vec2(seed, 3.7)), hash21(vec2(seed, 9.1))) * 64.0;
    vec2 p = uv * 3.0 + so;

    // irregular cavity outlines: warp the lattice before nearest-site search
    vec2 w = vec2(fbm(p * 1.7), fbm(p * 1.7 + vec2(23.1, 5.4))) - 0.5;
    vec2 q = p + 0.55 * w;

    vec2 ci = floor(q), cf = q - ci;
    float md = 8.0; vec2 mid = vec2(0.0);
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 g = vec2(float(x), float(y));
            vec2 o = hash22(ci + g);
            vec2 r = g + 0.25 + 0.5 * o - cf;
            float d = dot(r, r);
            if (d < md) { md = d; mid = ci + g; }
        }
    }
    float d = sqrt(md);

    // banding coordinate: distance plus a fine ripple, per-cavity phase
    float ph = hash21(mid + 0.5) * 7.0;
    float t = d * 17.0 + ph + 0.9 * fbm(p * 4.0 + mid);
    t += 0.42 * sin(t * 1.31 + ph) + 0.22 * sin(t * 0.53 - ph);  // uneven band widths
    float bi = floor(t), bf = fract(t);

    float h = hash21(vec2(bi, dot(mid, vec2(1.0, 17.0))));
    vec3 col = band_colour(h);

    // sub-band gloss: bright core, darker at the seams
    float gloss = smoothstep(0.0, 0.28, bf) * smoothstep(1.0, 0.72, bf);
    col *= 0.74 + 0.36 * gloss;
    // a thin bright seam every few bands reads as chalcedony rind
    float rind = smoothstep(0.9, 0.99, hash21(vec2(bi + 41.0, dot(mid, vec2(3.0, 7.0)))));
    col = mix(col, C4, rind * gloss * 0.75);

    // mineral grain
    float g2 = vnoise(p * 90.0);
    col *= 0.94 + 0.12 * g2;
    // broad tonal drift so cavities don't all read alike
    col *= 0.86 + 0.28 * fbm(p * 0.9 + 13.0);

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
