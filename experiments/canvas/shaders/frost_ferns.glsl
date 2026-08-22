// style: frost_ferns
// title: Window frost
// description: Ice ferns creeping over dark glass — pale stems sweeping in slow curves with short barbs combed off either side, thinning out to bare pane.
// tags: frost, ice, crystal, fern, dendritic, cool, dark, line
// brief: organic and natural textures; the fine line-drawn one, light on dark
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const vec3 GLASS_A = vec3(0.055, 0.086, 0.145);  // deep night glass
const vec3 GLASS_B = vec3(0.129, 0.196, 0.286);  // lit haze
const vec3 ICE     = vec3(0.902, 0.961, 0.988);  // crystal white
const vec3 ICE_C   = vec3(0.502, 0.769, 0.882);  // cold cyan

float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
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
    for (int k = 0; k < 4; k++) { s += a * vnoise(p); p = p * 2.11 + vec2(4.7, 8.9); a *= 0.5; }
    return s;
}

void main() {
    vec2 so = vec2(hash21(vec2(seed, 4.1)), hash21(vec2(seed, 8.7))) * 73.0;
    vec2 p = uv + so;

    // a slowly turning growth direction; ferns comb along it.
    // NOTE rotate the local uv, not p: p carries a large seed offset, and rotating it by a
    // spatially varying angle would multiply the pattern frequency by |p|.
    float ang = 6.2832 * fbm(p * 0.55);
    float ca = cos(ang), sa = sin(ang);
    vec2 q = mat2(ca, -sa, sa, ca) * uv;

    float across = q.y * 12.0 + 1.1 * (fbm(p * 2.0) - 0.5);     // across the stems
    float along  = q.x * 12.0 + 0.8 * (fbm(p * 2.6 + 9.0) - 0.5);

    float ni = floor(across);
    float nf = abs(fract(across) - 0.5) * 2.0;                 // 0 on a stem, 1 midway
    // stems come and go along their length
    float grow = smoothstep(0.18, 0.52, fbm(vec2(along * 0.30, ni * 2.7) + 51.0));

    float stem = smoothstep(0.13, 0.02, nf);
    float ph = hash21(vec2(ni, 3.0)) * 3.0;
    float bf = abs(fract(along * 2.4 + ph) - 0.5) * 2.0;
    float barb = smoothstep(0.40, 0.08, bf) * smoothstep(0.72, 0.04, nf);
    float bf2 = abs(fract(along * 4.8 + ph) - 0.5) * 2.0;
    float barb2 = smoothstep(0.32, 0.08, bf2) * smoothstep(0.52, 0.04, nf) * 0.8;

    float fern = max(stem, max(barb, barb2)) * grow;

    // patchy coverage so bare glass shows through
    fern *= 0.30 + 0.70 * smoothstep(0.14, 0.46, fbm(p * 1.3 + 40.0));

    vec3 col = mix(GLASS_A, GLASS_B, smoothstep(0.25, 0.80, fbm(p * 2.2 + 11.0)));
    col += ICE_C * fern * 0.40;                       // glow
    col = mix(col, ICE_C, smoothstep(0.14, 0.50, fern));
    col = mix(col, ICE, smoothstep(0.45, 0.85, fern));

    // frozen dust on the pane
    col += 0.05 * vnoise(p * 220.0) * (0.35 + fern);

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
