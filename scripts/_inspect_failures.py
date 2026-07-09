import json
from pathlib import Path

d = json.load(open(Path(__file__).resolve().parent.parent / "eval-report.json", encoding="utf-8"))
print("MISS (answerable, not retrieved @k):")
for i in d["items"]:
    if i["answerable"] and not i["hit"]:
        print(" ", i["id"], [k.split("::")[1][:24] for k in i["retrieved"]])
print("HALLUCINATED (out-of-corpus but answered):")
for i in d["items"]:
    if not i["answerable"] and not i.get("abstained"):
        print(" ", i["id"], repr(i.get("answer", "")[:150]))
print("LOW CORRECTNESS (<8):")
for i in d["items"]:
    if i["answerable"] and isinstance(i.get("correctness"), int) and i["correctness"] < 8:
        print(" ", i["id"], "c=%d f=%s" % (i["correctness"], i.get("faithfulness")), repr(i.get("answer", "")[:110]))
