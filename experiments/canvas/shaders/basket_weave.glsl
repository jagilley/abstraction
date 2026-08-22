// style: basket_weave
// title: Basket weave
// description: Two families of fat glossy ribbons cross on the diagonal and pass over and under each other in a checkerboard, each strip a different mint, butter or coral and finely cross-hatched, with dark backing showing through the gaps.
// tags: weave, basket, ribbon, diagonal, textile, glossy, pastel
// author: claude (lluminate smoke2 b20d10e4)
precision mediump float;

varying vec2 uv;
uniform float seed;

// ---------- fixed palette (seed must not touch these) ----------
const vec3 PAL_A = vec3(0.50, 1.00, 0.50);
const vec3 PAL_B = vec3(0.50, 0.50, 0.50);
const vec3 PAL_C = vec3(1.00, 0.85, 0.65);
const vec3 PAL_D1 = vec3(0.02, 0.18, 0.38);
const vec3 PAL_D2 = vec3(0.42, 0.10, 0.02);
const vec3 BG = vec3(0.045, 0.045, 0.06);

// ---------- fixed structure ----------
const float SCALE = 6.5;     // ribbon density
const float WIDTH = 0.678;   // ribbon width as a fraction of the period
const float PERIOD = 1.0;

// seed enters the hash lattice only as a domain offset
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21) + seed * 17.13);
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

vec3 palette(float t, vec3 d) {
    return PAL_A + PAL_B * cos(6.28318 * (PAL_C * t + d));
}

void main() {
    vec2 p = uv * 2.0 - 1.0;

    // seed turns and slides the weave: arrangement, not identity
    float ang = seed * 0.73;
    float ca = cos(ang), sa = sin(ang);
    vec2 rp = vec2(p.x * ca - p.y * sa, p.x * sa + p.y * ca);
    vec2 q = rp * SCALE + vec2(sin(seed * 1.7), cos(seed * 2.3)) * 5.0;

    // diagonal weave coordinates (two crossing families of ribbons)
    float u = (q.x + q.y) * 0.7071;
    float v = (q.x - q.y) * 0.7071;

    float cu = floor(u / PERIOD);
    float cv = floor(v / PERIOD);

    float du = abs(fract(u / PERIOD) - 0.5);
    float dv = abs(fract(v / PERIOD) - 0.5);

    bool presentU = du < WIDTH * 0.5;
    bool presentV = dv < WIDTH * 0.5;

    // checkerboard parity decides which ribbon passes on top at a crossing
    float parity = mod(cu + cv, 2.0);

    vec3 col = BG;

    float idU = cu * 0.1234 + hash21(vec2(cu, 1.0)) * 3.0;
    float idV = cv * 0.4321 + hash21(vec2(cv, 2.0)) * 3.0;

    vec3 colU = palette(fract(idU), PAL_D1);
    vec3 colV = palette(fract(idV), PAL_D2);

    bool drawU = false;
    bool drawV = false;

    if (presentU && presentV) {
        if (parity < 1.0) { drawU = true; } else { drawV = true; }
    } else if (presentU) {
        drawU = true;
    } else if (presentV) {
        drawV = true;
    }

    if (drawU) {
        float shade = sqrt(max(0.0, 1.0 - pow(du / (WIDTH * 0.5), 2.0)));
        col = colU * (0.45 + 0.65 * shade);
        col += 0.18 * pow(shade, 6.0);         // gloss highlight along the ribbon
        col *= 1.0 + 0.04 * sin((u - v) * 30.0); // woven cross-hatch texture
    } else if (drawV) {
        float shade = sqrt(max(0.0, 1.0 - pow(dv / (WIDTH * 0.5), 2.0)));
        col = colV * (0.45 + 0.65 * shade);
        col += 0.18 * pow(shade, 6.0);
        col *= 1.0 + 0.04 * sin((u + v) * 30.0);
    } else {
        // gap between ribbons: small stippled backing texture
        float g = hash21(floor(q * 3.0));
        col = BG + 0.035 * g;
        float edgeDist = min(WIDTH * 0.5 - du, WIDTH * 0.5 - dv);
        col *= 1.0 - 0.25 * smoothstep(0.12, 0.0, abs(edgeDist));
    }

    // fine grain
    col += (hash21(uv * 900.0) - 0.5) * 0.02;

    // shallow vignette so off-centre crops stay usable
    float vig = smoothstep(1.45, 0.2, length(p));
    col *= mix(0.87, 1.03, vig);

    col = pow(clamp(col, 0.0, 1.0), vec3(0.9));

    gl_FragColor = vec4(col, 1.0);
}
