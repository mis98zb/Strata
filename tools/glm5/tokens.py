"""Encode a formatted GLM prompt, or decode the standalone engine's CSV token IDs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gguf_reader import GGUFFile
from strata_tokenizer import Tokenizer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="first GGUF shard, containing tokenizer metadata")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--text-file", help="UTF-8 prompt, already formatted with the model's chat template")
    group.add_argument("--decode", help="comma-separated generated token IDs")
    args = ap.parse_args()
    try:
        metadata = GGUFFile(Path(args.model)).metadata
        if metadata.get("general.architecture") != "glm5-next":
            ap.error("--model must declare general.architecture = glm5-next")
        if metadata.get("tokenizer.ggml.pre") not in ("glm4", "chatglm-bpe"):
            ap.error("unsupported GLM pre-tokenizer")
        tk = Tokenizer.from_gguf(args.model)
        if args.text_file is not None:
            text = Path(args.text_file).read_text(encoding="utf-8")
            print(",".join(map(str, tk.encode(text, parse_special=True))))
        else:
            ids = [int(x) for x in args.decode.split(",")] if args.decode else []
            if any(x < 0 or x >= len(tk.tokens) for x in ids):
                ap.error("token ID outside the model vocabulary")
            print(tk.decode(ids), end="")
    except (OSError, ValueError, KeyError) as error:
        ap.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
