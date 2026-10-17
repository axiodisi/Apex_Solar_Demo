# eval.py - ask every test question and check the answers.
#
# Usage:
#   python eval.py                                  # against a local server on :8080
#   python eval.py https://apex-solar-demo.onrender.com
#   python eval.py http://localhost:8080 law-firm   # one assistant only
#
# A question passes when every expected phrase appears in the answer (case-insensitive)
# and, if a source is expected, at least one cited file name contains it.
import json
import sys
import time
import urllib.request
from pathlib import Path

base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080").rstrip("/")
only = sys.argv[2] if len(sys.argv) > 2 else None
tests = json.loads((Path(__file__).parent / "eval_questions.json").read_text(encoding="utf-8"))
sys.stdout.reconfigure(encoding="utf-8")

passed = total = 0
times = []
for assistant, questions in tests.items():
    if only and assistant != only:
        continue
    print(f"\n== {assistant}")
    for t in questions:
        total += 1
        body = json.dumps({"question": t["q"], "assistant": assistant}).encode()
        req = urllib.request.Request(f"{base}/query/", data=body, headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read())
        seconds = time.perf_counter() - started
        times.append(seconds)
        answer = data["answer"].lower()
        files = [s["filename"] for s in data["sources"]]
        ok_text = all(e.lower() in answer for e in t["expect"])
        ok_source = not t["source"] or any(any(part in f for part in t["source"].split("|")) for f in files)
        ok = ok_text and ok_source
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'} {seconds:4.1f}s  {t['q']}")
        if not ok:
            print(f"      answer: {data['answer'][:300]}")
            print(f"      sources: {files}")

times.sort()
print(f"\n{passed}/{total} passed. Median {times[len(times)//2]:.1f}s, slowest {times[-1]:.1f}s.")
