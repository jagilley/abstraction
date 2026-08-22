// style: op_lattice
// title: Op-art lens field
// description: Hard black-and-bone stripes pulled out of true by a scattered lattice of invisible lenses, so the ruling seems to swell and dent; a few of the swellings are printed in vermilion or cobalt instead of black.
// tags: op-art, moire, stripes, optical, lattice, high-contrast
// brief: constructed geometric ornament; the seed relocates the lenses, flips their sign and moves the coloured plates
// author: claude
precision mediump float;
varying vec2 uv;
uniform float seed;

const float FREQ = 24.0;                               // stripe cycles across the canvas
const float G    = 5.0;                                // lens lattice cells across

const vec3 INK   = vec3(0.075, 0.071, 0.067);
const vec3 BONE  = vec3(0.941, 0.929, 0.898);
const vec3 VERM  = vec3(0.827, 0.271, 0.153);
const vec3 COBAL = vec3(0.114, 0.235, 0.588);

float h11(float x) { return fract(sin(x * 127.1 + 11.3) * 43758.5453123); }
float h21(vec2 p)  { return fract(sin(dot(floor(p), vec2(127.1, 311.7))) * 43758.5453123); }
float h22(vec2 p)  { return fract(sin(dot(floor(p), vec2(74.7, 219.3))) * 24634.6345); }
float h23(vec2 p)  { return fract(sin(dot(floor(p), vec2(21.9, 97.7))) * 15731.743); }

// x = stripe phase in cycles, y = vermilion weight, z = cobalt weight
vec3 field(vec2 w, vec2 so) {
    float ph = w.x * FREQ + 0.62 * sin(w.y * 10.6 + 1.0) + 0.28 * sin(w.y * 25.1 - 2.2);
    float wv = 0.0, wc = 0.0;

    vec2 p = w * G;
    vec2 c = floor(p);
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            vec2 cc = c + vec2(float(i), float(j)) + so;
            vec2 off = vec2(h21(cc), h22(cc)) * 0.66 + 0.17;
            vec2 cp = (c + vec2(float(i), float(j)) + off) / G;
            float rr = length(w - cp) * G;
            float b = exp(-rr * rr * 1.7);
            float sgn = h23(cc) < 0.5 ? -1.0 : 1.0;
            ph += sgn * (1.1 + 1.7 * h21(cc + vec2(3.0, 5.0))) * b;
            float hp = h22(cc + vec2(7.0, 11.0));
            wv += b * step(0.90, hp);
            wc += b * step(0.79, hp) * step(hp, 0.90);
        }
    }
    return vec3(ph, wv, wc);
}

void main() {
    vec2 so = floor(vec2(h11(seed + 8.2), h11(seed * 7.1 + 3.5)) * 53.0);
    vec3 fd = field(uv, so);

    // band-limited square wave: it goes soft, then grey, instead of aliasing
    float fw = fwidth(fd.x);
    float e = clamp(fw * 1.6, 0.004, 0.5);
    float tri = abs(fract(fd.x) - 0.5) * 2.0;
    float band = smoothstep(0.5 - e, 0.5 + e, tri);

    vec3 dark = INK;
    dark = mix(dark, VERM, smoothstep(0.10, 0.85, fd.y));
    dark = mix(dark, COBAL, smoothstep(0.10, 0.85, fd.z));

    vec3 col = mix(dark, BONE, band);

    // the paper itself: press grain and a very slight unevenness of inking
    float grain = fract(sin(dot(uv * 991.0, vec2(12.99, 78.23))) * 43758.545) - 0.5;
    col += grain * 0.028;
    col *= 1.0 + 0.030 * sin(uv.x * 4.3 + 0.7) * sin(uv.y * 3.1 - 1.4);

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
