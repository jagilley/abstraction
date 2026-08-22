// style: wood_grain
// title: Planed wood grain
// description: A planed board of dark hardwood — growth rings of honey and walnut sweeping across the grain, fine fibre streaks, open pores and pale ray flecks.
// tags: wood, grain, timber, stripes, grainy, warm, matte
// brief: organic and natural textures; the matte, grainy, near-parallel one
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const vec3 EARLY = vec3(0.792, 0.643, 0.451);  // earlywood honey
const vec3 MID   = vec3(0.396, 0.235, 0.118);  // heartwood
const vec3 LATE  = vec3(0.125, 0.071, 0.039);  // latewood
const vec3 FLECK = vec3(0.925, 0.831, 0.671);  // medullary ray

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
    for (int k = 0; k < 4; k++) { s += a * vnoise(p); p = p * 2.07 + vec2(9.1, 3.3); a *= 0.5; }
    return s;
}

void main() {
    vec2 so = vec2(hash21(vec2(seed, 1.9)), hash21(vec2(seed, 6.3))) * 57.0;
    vec2 p = uv + so;

    // growth rings: sweeping across the board, wide and narrow by turns
    float t = p.x * 12.5
            + 6.5 * (fbm(vec2(p.x * 0.55, p.y * 0.30)) - 0.5)
            + 1.1 * (fbm(vec2(p.x * 2.2, p.y * 0.9)) - 0.5);
    t += 0.62 * sin(t * 0.47 + so.x) + 0.30 * sin(t * 1.19 - so.y);  // uneven ring widths
    float ring = fract(t);
    float ri = floor(t);

    // some rings are far darker than others
    float rh = hash21(vec2(ri, 5.0));
    float depth = 0.55 + 0.85 * rh;

    // ring profile: a near-black boundary line, a bright earlywood band, then a slow
    // fade back through heartwood into the next dark line
    float dens = smoothstep(0.10, 0.85, ring);
    vec3 midv = mix(MID, vec3(0.573, 0.318, 0.145), rh);   // some rings run redder
    vec3 col = mix(LATE, EARLY, smoothstep(0.0, 0.13, ring));
    col = mix(col, midv, smoothstep(0.13, 0.72, ring) * (0.60 + 0.40 * rh));
    col = mix(col, LATE, smoothstep(0.68 - 0.18 * rh, 0.96, ring));   // latewood band, width varies
    col *= 1.0 - 0.22 * smoothstep(0.90, 1.0, ring);                    // hard line at the boundary

    // fibre streaks: noise stretched hard along the grain
    float fib = vnoise(vec2(p.x * 210.0, p.y * 4.0)) * 0.55
              + vnoise(vec2(p.x * 62.0, p.y * 1.6)) * 0.45;
    col *= 0.85 + 0.28 * fib;

    // open pores: short dark dashes lying along the grain
    float pr = vnoise(vec2(p.x * 150.0 + 40.0, p.y * 22.0));
    col *= 1.0 - 0.5 * smoothstep(0.78, 0.95, pr) * (0.3 + 0.7 * dens);

    // medullary ray flecks: pale lenses lying across the grain
    float fl = vnoise(vec2(p.x * 7.0 + 17.0, p.y * 40.0));
    float flm = smoothstep(0.80, 0.93, fl) * smoothstep(0.995, 0.88, fl);
    col = mix(col, FLECK, 0.16 * flm * (0.35 + 0.65 * (1.0 - dens)));

    // board-scale tonal drift + light dust
    col *= 0.80 + 0.26 * fbm(p * 1.1 + 21.0);
    col *= 0.96 + 0.08 * vnoise(p * 340.0);

    col = pow(clamp(col, 0.0, 1.0), vec3(1.35)) * 1.22;   // deepen the darks, keep the honey
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
