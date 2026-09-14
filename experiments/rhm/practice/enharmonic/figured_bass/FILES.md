# figured_bass — FILES

**Up**: [`../FILES.md`](../FILES.md) (enharmonic). **Spec**: [`SPEC.md`](SPEC.md). Facts only until
discussed: [`sizing/SIZING.md`](sizing/SIZING.md), `../figures/fb_s0_reduction.txt`,
`../figures/fb_s1_reduction.txt`.

## Code files

| file | purpose |
|---|---|
| `fb_open.patch` | The diff of record against `../enharmonic.py` and `../quotient.py` (every addition `# [figured_bass]`): `open_inventory` (an adopted level's operative table is the live build; the move set re-materialised once per cycle with `len(ms)`, slot ids and priced width unchanged by construction), `ungate_l5` (the committable L5 miner mines every cycle), exact-replay instrumentation (`ClassMiner` picks and spell counts; blocked-key counts; operative twins of the frozen readouts), gates E-7 (knob-off identity with the bits named False), E-8/E-8b (flag-vs-table invariant, exercised against a table that moves), E-9/E-9b (the ungated miner accrues exactly the mined rows and identically to the panel; nothing before era 4 with the bit off). Revision notes inline. |
| `results/RUN_fb_s0.sh`, `RUN_fb_s1.sh` | The commands of record with the gate record appended. `fb_s0`: `given_cat_tok_open`, `given_cat_tok_ung5`, `given_cat_tok_open_ung5`, `flat_open`, `flat_yk_open`, `flat_yk_open_ung5`. `fb_s1`: `given_cat_tok_open_yk`, `given_cat_tok_open_ung5_yk` clock-yoked to the banked `given_cat_tok`. Logs beside them are untracked. |

## Children

| folder | what |
|---|---|
| [`sizing/`](sizing/SIZING.md) | Q0, offline: `phase0_open.py` replays `en_s0`/`en_s1` from the logged keys at support under frozen / L4-open / all-open regimes — class coverage, L5 lookup-ability, the era gate on the committable L5 miner, executor exposure. |
