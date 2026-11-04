"""Compare fixed-prefix teacher-forced GLM output files, without NumPy."""
from __future__ import annotations

import argparse
import array
import hashlib
import heapq
import json
import math
from pathlib import Path
import struct
import sys


def _read(file, size):
    data = file.read(size)
    if len(data) != size:
        raise ValueError("truncated teacher-forced output")
    return data


def _header(file):
    magic, version, vocab, count, prefix, flags = struct.unpack("<6i", _read(file, 24))
    if (magic != 0x474C4D50 or version != 1 or not 2 <= vocab <= 1000000
            or not 1 <= count <= 256 or not 1 <= prefix <= 131072 - count or not 0 <= flags <= 5):
        raise ValueError("invalid teacher-forced output header")
    return vocab, count, prefix, flags


def _floats(file, count):
    values = array.array("f")
    values.frombytes(_read(file, count * 4))
    if sys.byteorder != "little":
        values.byteswap()
    if not all(map(math.isfinite, values)):
        raise ValueError("non-finite teacher-forced logits")
    return values


def compare(reference: Path, candidate: Path, low_margin: float = 0.5):
    if not math.isfinite(low_margin) or low_margin < 0:
        raise ValueError("low-margin threshold must be finite and nonnegative")
    rows = []
    with reference.open("rb") as ref, candidate.open("rb") as other:
        rh, ch = _header(ref), _header(other)
        if rh[:3] != ch[:3]:
            raise ValueError("teacher-forced shapes or prefix lengths differ")
        vocab, count, prefix, _ = rh
        token_bytes = _read(ref, (prefix + count) * 4)
        if token_bytes != _read(other, len(token_bytes)):
            raise ValueError("teacher-forced conditioning tokens differ")
        tokens = struct.unpack(f"<{prefix + count}i", token_bytes)
        if any(t < 0 or t >= vocab for t in tokens):
            raise ValueError("conditioning token outside vocabulary")
        for i in range(count):
            target = struct.unpack("<i", _read(ref, 4))[0]
            if target != tokens[prefix + i] or target != struct.unpack("<i", _read(other, 4))[0]:
                raise ValueError("teacher-forced target token mismatch")
            a, b = _floats(ref, vocab), _floats(other, vocab)
            at = heapq.nlargest(min(10, vocab), range(vocab), key=a.__getitem__)
            bt = heapq.nlargest(min(10, vocab), range(vocab), key=b.__getitem__)
            am, bm = a[at[0]], b[bt[0]]
            ae = [math.exp(x - am) for x in a]
            be = [math.exp(x - bm) for x in b]
            az, bz = math.fsum(ae), math.fsum(be)
            al, bl = math.log(az), math.log(bz)
            ap, bp = [x / az for x in ae], [x / bz for x in be]
            kl = math.fsum(p * ((x - am - al) - (y - bm - bl)) for p, x, y in zip(ap, a, b))
            nll_a, nll_b = (am - a[target]) + al, (bm - b[target]) + bl
            rows.append({
                "target_position": prefix + i, "target_id": target,
                "reference_top1": at[0], "candidate_top1": bt[0],
                "argmax_changed": at[0] != bt[0],
                "reference_margin": a[at[0]] - a[at[1]],
                "candidate_margin": b[bt[0]] - b[bt[1]],
                "top10_overlap": len(set(at) & set(bt)),
                "reference_nll": nll_a, "candidate_nll": nll_b, "nll_delta": nll_b - nll_a,
                "kl": kl, "total_variation": math.fsum(abs(x - y) for x, y in zip(ap, bp)) / 2,
                "logit_rmse": math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(a, b)) / vocab),
                "max_logit_error": max(abs(x - y) for x, y in zip(a, b)),
            })
        if ref.read(1) or other.read(1):
            raise ValueError("trailing teacher-forced output data")
    low = [r for r in rows if r["reference_margin"] <= low_margin]
    summary = {
        "positions": count, "prefix": prefix, "vocab": vocab,
        "reference_flags": rh[3], "candidate_flags": ch[3],
        "conditioning_sha256": hashlib.sha256(token_bytes).hexdigest(),
        "low_margin_threshold": low_margin, "low_margin_positions": len(low),
        "argmax_changes": sum(r["argmax_changed"] for r in rows),
        "low_margin_argmax_changes": sum(r["argmax_changed"] for r in low),
        "reference_top1_correct": sum(r["reference_top1"] == r["target_id"] for r in rows),
        "candidate_top1_correct": sum(r["candidate_top1"] == r["target_id"] for r in rows),
        "mean_kl": math.fsum(r["kl"] for r in rows) / count,
        "max_kl": max(r["kl"] for r in rows),
        "mean_tv": math.fsum(r["total_variation"] for r in rows) / count,
        "max_tv": max(r["total_variation"] for r in rows),
        "mean_reference_nll": math.fsum(r["reference_nll"] for r in rows) / count,
        "mean_candidate_nll": math.fsum(r["candidate_nll"] for r in rows) / count,
        "mean_nll_delta": math.fsum(r["nll_delta"] for r in rows) / count,
    }
    return {"summary": summary, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--low-margin", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.output and args.output.resolve() in (args.reference.resolve(), args.candidate.resolve()):
            raise ValueError("comparison output must not overwrite a logit input")
        result = compare(args.reference, args.candidate, args.low_margin)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result["summary"], indent=2))
        else:
            print(json.dumps(result, indent=2))
    except (OSError, ValueError, struct.error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
