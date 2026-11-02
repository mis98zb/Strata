import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_tokenizer as ST


class GlmTokenizerTests(unittest.TestCase):
    def tokenizer(self, pre):
        tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)] + ["12", "123", "1234"]
        return ST.Tokenizer(tokens, ["1 2", "12 3", "123 4"], [1] * len(tokens), pre=pre)

    def test_numeric_ids_not_just_roundtrip(self):
        for pre in ("glm4", "chatglm-bpe"):
            tk = self.tokenizer(pre)
            self.assertEqual(tk.encode("1234"), [257, ord("4")])
            self.assertEqual(tk.decode(tk.encode("1234")), "1234")
        self.assertEqual(self.tokenizer("qwen35").encode("1234"), list(b"1234"))

    def test_combining_marks_use_glm_pattern(self):
        self.assertEqual(self.tokenizer("glm4")._split("e\u0301?"), ["e", "\u0301?"])
        self.assertEqual(self.tokenizer("qwen35")._split("e\u0301?"), ["e\u0301", "?"])

    def test_unicode_and_whitespace_roundtrip(self):
        tk = self.tokenizer("glm4")
        for text in ("", "Hello 1234567", "\u4f60\u597d\n\n  ", "e\u0301", "\U0001f600", "\x00\r\n"):
            with self.subTest(text=text):
                self.assertEqual(tk.decode(tk.encode(text)), text)

    def test_extract_records_glm_pattern(self):
        tk = self.tokenizer("glm4")
        with tempfile.TemporaryDirectory() as tmp, patch.object(ST.Tokenizer, "from_gguf", return_value=tk), \
                patch("gguf_reader.GGUFFile") as reader:
            reader.return_value.metadata = {}
            cfg = ST.extract("fixture.gguf", tmp)
            saved = json.loads((Path(tmp) / "tokenizer" / "tokenizer.json").read_text())
            self.assertEqual(cfg["pre_pattern"], ST.CHATGLM4_PATTERN)
            self.assertEqual(saved["pre"], "glm4")
            self.assertEqual(saved["pre_pattern"], ST.CHATGLM4_PATTERN)

    def test_cli_help_imports_without_shadowing_stdlib(self):
        script = Path(__file__).parent / "glm5" / "tokens.py"
        result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--text-file", result.stdout)

    def test_cli_encodes_and_decodes_exact_ids(self):
        from tools.glm5 import tokens as cli
        tk = self.tokenizer("glm4")
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp) / "prompt.txt"
            prompt.write_text("1234", encoding="utf-8")
            for arguments, want in ((["--text-file", str(prompt)], "257,52\n"),
                                    (["--decode", "257,52"], "1234")):
                output = StringIO()
                with patch.object(cli, "GGUFFile") as reader, \
                        patch.object(cli.Tokenizer, "from_gguf", return_value=tk), \
                        patch.object(sys, "argv", ["tokens.py", "--model", "fake.gguf", *arguments]), \
                        redirect_stdout(output):
                    reader.return_value.metadata = {"general.architecture": "glm5-next",
                                                    "tokenizer.ggml.pre": "glm4"}
                    self.assertEqual(cli.main(), 0)
                self.assertEqual(output.getvalue(), want)

    def test_cli_rejects_out_of_vocabulary_ids(self):
        from tools.glm5 import tokens as cli
        with patch.object(cli, "GGUFFile") as reader, \
                patch.object(cli.Tokenizer, "from_gguf", return_value=self.tokenizer("glm4")), \
                patch.object(sys, "argv", ["tokens.py", "--model", "fake.gguf", "--decode", "-1"]), \
                redirect_stderr(StringIO()), self.assertRaises(SystemExit) as stopped:
            reader.return_value.metadata = {"general.architecture": "glm5-next",
                                            "tokenizer.ggml.pre": "glm4"}
            cli.main()
        self.assertEqual(stopped.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
