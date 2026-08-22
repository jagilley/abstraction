"""Pairwise preference clicker: the ~100 human clicks that seed the taste gauge.

Serves a local page showing two swatches; press ← / → (or click) for the one you prefer,
↓ for "no preference", ␣ to skip. Each answer is appended as one JSON line:

    {"t": ..., "rater": ..., "a": "<style>/<nnn>.png", "b": "...", "choice": "a"|"b"|"tie",
     "kind": "within"|"across", "ms": reaction-time}

Pairs are sampled half within-style (two exemplars of one style: taste *inside* the valid
set, the organ the memo says RHM never had) and half across-style (taste between styles).
Nothing here is consumed by the learner; it is teacher-slot data.

    cd experiments && python -m canvas.taste.clicker --corpus canvas/corpora/data/<tag> \
        --rater jasper [--port 8765] [--n 120]
then open http://localhost:8765

Headless: `--manifest <dir>` writes `pairs.jsonl` and one side-by-side `pair_NNNN.png` per
pair (A left, B right) and exits; a Claude rater then reads each image and appends records in
the same schema to `taste/clicks/<rater>_<tag>.jsonl` (`rater` = e.g. "claude-opus").
"""

import argparse
import glob
import http.server
import json
import os
import random
import time
import urllib.parse

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>taste</title>
<style>
body{margin:0;background:#111;color:#ddd;font:14px system-ui;display:flex;flex-direction:column;align-items:center}
#row{display:flex;gap:24px;margin-top:24px}
img{width:384px;height:384px;image-rendering:auto;border:4px solid #222;cursor:pointer}
img:hover{border-color:#888}
#hud{margin:14px;color:#888}
kbd{background:#222;padding:2px 6px;border-radius:4px}
</style></head><body>
<div id="hud"></div>
<div id="row"><img id="a"><img id="b"></div>
<div id="hud2"><kbd>←</kbd> left &nbsp; <kbd>→</kbd> right &nbsp; <kbd>↓</kbd> no preference &nbsp; <kbd>space</kbd> skip</div>
<script>
let cur=null, t0=0, done=0;
async function next(){
  const r=await fetch('/pair'); cur=await r.json();
  if(cur.end){document.getElementById('hud').textContent='done — '+done+' answers recorded. thanks.';return;}
  document.getElementById('a').src='/img/'+cur.a; document.getElementById('b').src='/img/'+cur.b;
  document.getElementById('hud').textContent=(done+1)+' / '+cur.n+'   ('+cur.kind+')';
  t0=performance.now();
}
async function answer(choice){
  if(!cur) return;
  const ms=Math.round(performance.now()-t0);
  await fetch('/choice',{method:'POST',body:JSON.stringify({...cur,choice,ms})});
  if(choice!=='skip') done++;
  next();
}
document.addEventListener('keydown',e=>{
  if(e.key==='ArrowLeft')answer('a'); else if(e.key==='ArrowRight')answer('b');
  else if(e.key==='ArrowDown')answer('tie'); else if(e.key===' '){e.preventDefault();answer('skip');}
});
document.getElementById('a').onclick=()=>answer('a');
document.getElementById('b').onclick=()=>answer('b');
next();
</script></body></html>"""


def make_pairs(corpus, n, rng, within_frac=0.5):
    styles = {}
    for sdir in sorted(glob.glob(os.path.join(corpus, "*/"))):
        sid = os.path.basename(os.path.normpath(sdir))
        pngs = sorted(glob.glob(os.path.join(sdir, "*.png")))
        if len(pngs) >= 2:
            styles[sid] = [os.path.relpath(p, corpus) for p in pngs]
    sids = sorted(styles)
    pairs = []
    for i in range(n):
        if rng.random() < within_frac:
            s = rng.choice(sids)
            a, b = rng.sample(styles[s], 2)
            kind = "within"
        else:
            s1, s2 = rng.sample(sids, 2)
            a, b = rng.choice(styles[s1]), rng.choice(styles[s2])
            kind = "across"
        if rng.random() < 0.5:
            a, b = b, a
        pairs.append({"a": a, "b": b, "kind": kind, "n": n})
    return pairs


def write_manifest(corpus, pairs, out_dir, th=384):
    """Side-by-side pair images + pairs.jsonl for a headless rater."""
    from PIL import Image, ImageDraw
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "pairs.jsonl"), "w") as f:
        for i, p in enumerate(pairs):
            rec = dict(p, idx=i, png=f"pair_{i:04d}.png")
            f.write(json.dumps(rec) + "\n")
            im = Image.new("RGB", (2 * th + 24, th + 24), (17, 17, 17))
            for k, key in enumerate(("a", "b")):
                sw = Image.open(os.path.join(corpus, p[key])).convert("RGB").resize((th, th))
                im.paste(sw, (8 + k * (th + 8), 16))
            d = ImageDraw.Draw(im)
            d.text((8, 2), "A", fill=(220, 220, 220)); d.text((th + 16, 2), "B", fill=(220, 220, 220))
            im.save(os.path.join(out_dir, rec["png"]))
    print(f"wrote {len(pairs)} pairs to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--rater", default="jasper")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest", default=None, help="write pair images + pairs.jsonl here and exit")
    a = ap.parse_args()
    rng = random.Random(a.seed if a.seed is not None else int(time.time()))
    corpus = os.path.abspath(a.corpus)
    pairs = make_pairs(corpus, a.n, rng)
    if a.manifest:
        write_manifest(corpus, pairs, a.manifest)
        return
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "clicks",
                                f"{a.rater}_{os.path.basename(corpus)}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    state = {"i": 0}

    class H(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            if u.path == "/":
                body = HTML.encode()
                self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers()
                self.wfile.write(body)
            elif u.path == "/pair":
                if state["i"] >= len(pairs):
                    body = json.dumps({"end": True}).encode()
                else:
                    body = json.dumps(pairs[state["i"]]).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(body)
            elif u.path.startswith("/img/"):
                p = os.path.join(corpus, urllib.parse.unquote(u.path[5:]))
                if not os.path.abspath(p).startswith(corpus) or not os.path.exists(p):
                    self.send_response(404); self.end_headers(); return
                data = open(p, "rb").read()
                self.send_response(200); self.send_header("Content-Type", "image/png"); self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404); self.end_headers()

        def do_POST(self):
            if self.path == "/choice":
                n = int(self.headers.get("Content-Length", 0))
                rec = json.loads(self.rfile.read(n))
                rec = {"t": time.time(), "rater": a.rater, "corpus": os.path.basename(corpus),
                       "a": rec["a"], "b": rec["b"], "choice": rec["choice"], "kind": rec["kind"],
                       "ms": rec.get("ms")}
                if rec["choice"] != "skip":
                    with open(out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                state["i"] += 1
                self.send_response(204); self.end_headers()
            else:
                self.send_response(404); self.end_headers()

    print(f"{len(pairs)} pairs over {len(set(p['a'].split('/')[0] for p in pairs))} styles -> {out}")
    print(f"open http://localhost:{a.port}")
    http.server.ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
