// style: leaf_venation
// title: Leaf venation
// description: The vein network of a leaf held to the light — flat mottled greens divided by dark veins that branch from thick to hair-fine and thicken at every junction.
// tags: leaf, veins, network, botanical, flat, green, hierarchical
// brief: organic and natural textures; the flat-colour, line-network one
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const vec3 LAM_A = vec3(0.416, 0.529, 0.290);  // deep leaf
const vec3 LAM_B = vec3(0.588, 0.686, 0.400);  // mid green
const vec3 LAM_C = vec3(0.749, 0.796, 0.522);  // sunlit green
const vec3 LAM_D = vec3(0.855, 0.831, 0.588);  // chlorotic yellow
const vec3 VEIN  = vec3(0.169, 0.239, 0.153);  // vein green-black
const vec3 VEIN2 = vec3(0.310, 0.365, 0.216);  // minor vein

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
    for (int k = 0; k < 4; k++) { s += a * vnoise(p); p = p * 2.09 + vec2(3.7, 15.1); a *= 0.5; }
    return s;
}

// perpendicular distance to the nearest cell border (Quilez's two-pass form): unlike F2-F1
// this keeps the line width even, so junctions stay crisp instead of blooming into blobs
float edge_dist(vec2 p, float jit) {
    vec2 i = floor(p), f = p - i;
    vec2 mr = vec2(0.0);
    float md = 8.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 g = vec2(float(x), float(y));
            vec2 r = g + 0.5 + jit * (hash22(i + g) - 0.5) - f;
            float d = dot(r, r);
            if (d < md) { md = d; mr = r; }
        }
    }
    float me = 8.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 g = vec2(float(x), float(y));
            vec2 r = g + 0.5 + jit * (hash22(i + g) - 0.5) - f;
            vec2 dv = r - mr;
            if (dot(dv, dv) > 1e-5) me = min(me, dot(0.5 * (mr + r), normalize(dv)));
        }
    }
    return me;
}

void main() {
    vec2 so = vec2(hash21(vec2(seed, 6.7)), hash21(vec2(seed, 1.3))) * 51.0;
    vec2 p = uv + so;
    // veins wander: warp the lattice so nothing looks like a grid
    vec2 w = vec2(fbm(p * 2.2), fbm(p * 2.2 + 8.4)) - 0.5;
    vec2 q = p + 0.16 * w;

    float g1 = edge_dist(q * 3.6, 1.05);          // primary veins
    float g2 = edge_dist(q * 8.5 + 21.0, 1.05);   // secondaries
    float g3 = edge_dist(q * 16.0 + 47.0, 1.05);  // areole mesh

    float v1 = smoothstep(0.062, 0.024, g1);
    float v2 = smoothstep(0.052, 0.020, g2);
    float v3 = smoothstep(0.048, 0.018, g3);

    // lamina: flat patches of green from a smooth field, unrelated to the vein cells
    float m = fbm(p * 2.4 + 63.0) + 0.25 * fbm(p * 6.0);
    vec3 col = LAM_A;
    col = mix(col, LAM_B, smoothstep(0.44, 0.47, m));
    col = mix(col, LAM_C, smoothstep(0.58, 0.61, m));
    col = mix(col, LAM_D, smoothstep(0.72, 0.75, m));

    col = mix(col, VEIN2, v3 * 0.70);
    col = mix(col, VEIN2, v2 * 0.92);
    col = mix(col, VEIN, v1);

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
