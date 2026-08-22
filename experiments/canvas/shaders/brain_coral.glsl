// style: brain_coral
// title: Brain coral
// description: The meandering ridges of a bleached brain coral — rounded chalk-white walls winding in near-parallel lanes, splitting and rejoining, with cool sea-green grooves and fine ribs between them.
// tags: coral, ridges, labyrinth, meander, relief, chalk, sea green, shaded
// brief: organic and natural textures; the sculptural, softly lit one
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const vec3 VALLEY = vec3(0.176, 0.267, 0.259);  // shaded groove
const vec3 FLANK  = vec3(0.573, 0.663, 0.596);  // sea-green flank
const vec3 CREST  = vec3(0.949, 0.937, 0.882);  // bleached chalk crest
const vec3 TINT   = vec3(0.855, 0.812, 0.706);  // sun-bleached warmth

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
    for (int k = 0; k < 4; k++) { s += a * vnoise(p); p = p * 2.13 + vec2(7.3, 19.1); a *= 0.5; }
    return s;
}

// height of the coral surface: stripes whose coordinate is warped hard enough to meander,
// split and rejoin. Smooth by construction, so central differences are safe.
float height(vec2 p) {
    vec2 w = vec2(fbm(p * 1.55), fbm(p * 1.55 + vec2(21.7, 4.3))) - 0.5;
    vec2 w2 = vec2(fbm(p * 3.7 + 8.0), fbm(p * 3.7 + 30.0)) - 0.5;
    float s = (p.x + 1.55 * w.x + 0.30 * w2.x + 0.55 * (p.y + 1.55 * w.y)) * 11.0;
    float raw = 0.5 - 0.5 * cos(6.2832 * s);
    return pow(raw, 0.68);            // broad rounded crests, narrow grooves
}

void main() {
    vec2 so = vec2(hash21(vec2(seed, 3.3)), hash21(vec2(seed, 9.7))) * 61.0;
    vec2 p = uv + so;

    const float E = 0.0016;
    float h  = height(p);
    float hx = height(p + vec2(E, 0.0));
    float hy = height(p + vec2(0.0, E));
    vec3 n = normalize(vec3((h - hx) * 17.0, (h - hy) * 17.0, 1.0));

    vec3 L = normalize(vec3(-0.55, 0.72, 0.62));
    float diff = clamp(dot(n, L), 0.0, 1.0);
    float amb = 0.34 + 0.66 * h;                      // valleys sit in their own shadow

    vec3 col = mix(VALLEY, FLANK, smoothstep(0.10, 0.62, h));
    col = mix(col, CREST, smoothstep(0.68, 0.97, h));
    col *= 0.60 + 0.34 * diff + 0.20 * amb;

    // fine ribs running across the lanes, and a chalky surface grain
    float rib = 0.5 + 0.5 * sin(6.2832 * (p.y * 46.0 + 6.0 * fbm(p * 1.55)));
    col *= 0.94 + 0.11 * rib * (0.25 + 0.75 * h);
    col *= 0.93 + 0.14 * vnoise(p * 230.0);

    // slow bleaching across the colony
    col = mix(col, col * TINT * 1.12, 0.35 * fbm(p * 0.9 + 44.0));
    // specular sheen on the crests
    col += 0.05 * pow(clamp(dot(n, normalize(L + vec3(0.0, 0.0, 1.0))), 0.0, 1.0), 18.0);

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
