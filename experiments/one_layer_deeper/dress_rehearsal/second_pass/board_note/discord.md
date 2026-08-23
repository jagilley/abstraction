*Draft for `#one-layer-deeper`. Not posted; Jasper decides whether and when.*

---

A few pieces of arithmetic about the Hard ladder that might save people some time. All of it
checks out from the public repo at `4ceff95`.

Some prompts never reduce. When `x^(2^T) < N` the answer is the unreduced power, so squaring small
integers as digit strings gets them right with no knowledge of the modulus. On the public
`m5` config, whose moduli are 12, 14 and 16 bits, that comes to exactly 6 of 768 at `T=1` on the
seen-`N` profile, which is what ranks 2 through 7 read today. Shifting every width one
bit down gives 17 and one bit up gives 1, which is roughly how you can pin what Hard uses. The count
collapses with depth: 6 at `T=1`, 1 at `T=2`, none after.

Coverage differs enormously across the three widths. Only 14 valid RSA moduli exist at 12 bits,
with 35,624 `(N, x)` pairs among them, so a 10k-per-setting dataset trains on 58% of the whole
space. Per-modulus base coverage runs 0.589 / 0.063 / 0.004 at 12 / 14 / 16 bits, and each rung
draws 256 of its 768 prompts from each width. Half of one cell is a cheaper score than it looks.

Depth degenerates too, since `x^(2^T)` goes eventually periodic in `T`. On `m5`, 191 of 768 prompts
at `T=16` and 357 at `T=64` repeat an answer from a lower rung. Certification wants 100%, so nobody
gets a free rung, but deep-rung accuracy stops measuring depth cleanly.

Lastly, if you can re-ground a rollout step to step at test time, certifying rung `T` needs
one-step exact error below roughly `9e-4/T`. Each doubling costs a factor of two, the whole ladder
comes to 64×, and rung zero carries nearly all of it.

Full note with sources and repro steps: [LINK]
