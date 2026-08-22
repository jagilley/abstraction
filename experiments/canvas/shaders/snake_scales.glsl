// style: snake_scales
// title: Snake scales
// description: Overlapping keeled scales lying in offset rows like roof tiles, each one a single flat olive or bronze tone lit from above, with dark shadow in the gaps.
// tags: scales, reptile, snake, tiles, overlap, shaded, olive, bronze
// brief: organic and natural textures; the softly shaded one, built from discrete units
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const vec3 GAP    = vec3(0.129, 0.137, 0.098);  // shadow between scales
const vec3 OLIVE  = vec3(0.353, 0.396, 0.239);  // body olive
const vec3 DARK   = vec3(0.235, 0.267, 0.161);  // blotch
const vec3 BRONZE = vec3(0.549, 0.451, 0.235);  // warm bronze
const vec3 BONE   = vec3(0.784, 0.769, 0.639);  // pale belly scale

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
    for (int k = 0; k < 4; k++) { s += a * vnoise(p); p = p * 2.05 + vec2(6.1, 12.7); a *= 0.5; }
    return s;
}

vec3 scale_colour(vec2 c, vec2 so) {
    float m = fbm(c * vec2(0.18, 0.22) + so);          // slow blotch field
    float j = hash21(c + 7.0);
    float t = m + 0.07 * (j - 0.5);
    vec3 col = mix(DARK, OLIVE, smoothstep(0.30, 0.46, t));
    col = mix(col, BRONZE, smoothstep(0.50, 0.62, t));
    col = mix(col, BONE, smoothstep(0.66, 0.78, t));
    return col * (0.88 + 0.24 * j);
}

void main() {
    vec2 so = vec2(hash21(vec2(seed, 2.3)), hash21(vec2(seed, 5.9))) * 43.0;
    // gentle body curvature: rows bow across the frame
    vec2 p = uv + 0.035 * vec2(fbm(uv * 1.6 + so), fbm(uv * 1.6 + so + 19.0));
    const vec2 G = vec2(11.0, 15.0);                    // scales per canvas
    vec2 q = p * G + so * G;

    float best = -1e9;
    vec3 col = GAP;
    // rows overlap the row above, like tiles; walk back-to-front so the front row wins
    for (int dy = 1; dy >= -1; dy--) {
        float row = floor(q.y) + float(dy);
        float roff = 0.5 * mod(row, 2.0);
        for (int dx = -1; dx <= 1; dx++) {
            float cx = floor(q.x - roff) + float(dx);
            vec2 c = vec2(cx + roff + 0.5, row + 0.5);
            vec2 d = q - c;
            // teardrop: wide, with a rounded tip pointing down-frame
            vec2 e = d / vec2(0.72, 0.95);
            e.x /= 0.62 + 0.38 * smoothstep(-1.0, 0.7, e.y);
            float r = length(e);
            if (r < 1.0 && -row > best) {
                best = -row;
                vec3 base = scale_colour(vec2(cx, row), so);
                // shading: lit from the top edge, shadowed under the overlap
                float lit = 0.84 + 0.42 * smoothstep(1.0, -0.4, e.y) * (1.0 - 0.5 * r * r);
                lit -= 0.30 * smoothstep(0.62, 1.0, r);
                float keel = 1.0 + 0.16 * smoothstep(0.30, 0.0, abs(e.x));   // ridge down the middle
                float spec = 0.30 * smoothstep(0.55, 0.0, length(e - vec2(0.0, -0.30)));
                col = base * lit * keel + spec * vec3(0.30, 0.31, 0.26);
                // rim darkening so neighbours read apart
                col = mix(col, GAP, smoothstep(0.86, 1.0, r) * 0.85);
            }
        }
    }
    col *= 0.94 + 0.12 * vnoise(q * 22.0);              // scale-surface grain
    col *= 0.94 + 0.16 * fbm(p * 1.3 + 31.0);           // broad sheen
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
