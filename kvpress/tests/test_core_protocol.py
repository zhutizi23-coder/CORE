"""CPU regression tests for CORE's allocation and inference wiring.

No model downloads or checkpoints are required. Run from the repository root:
PYTHONPATH=.:kvpress python -m unittest discover -s kvpress/tests -p test_core_protocol.py
"""

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from core import infer_with_core_indexer as inference
from core.kvpress_adapter import COREDecodingPress, COREScorerPress


class COREProtocolTests(unittest.TestCase):
    def test_sink_protection_preserves_memory_allocation(self):
        for query_length in (8, 1):  # Prefill and decode scoring.
            with self.subTest(query_length=query_length):
                press = COREScorerPress(
                    indexer=torch.nn.Identity(),
                    feature_extractor=SimpleNamespace(),
                    compression_ratio=0.5,
                    scoring_layer_idx=-1,
                    n_sink_protect=4,
                )
                raw = torch.zeros(1, 2, 8)
                keys = torch.zeros(1, 2, 8, 2)
                with patch.object(press, "_ensure_device"), patch.object(
                    press, "_compute_scores_from_features", return_value=raw
                ):
                    selected = press.score(
                        SimpleNamespace(layer_idx=0),
                        torch.zeros(1, query_length, 4), keys, keys, None, {},
                    )
                self.assertTrue(torch.all(selected[:, :, :4] > selected[:, :, 4:]))
                torch.testing.assert_close(raw, torch.zeros_like(raw))
                torch.testing.assert_close(press._cached_memory_scores, raw[:, 0])
                mass = press._cached_memory_scores.softmax(-1)[:, 4:].sum()
                self.assertAlmostEqual(mass.item(), 0.5)

    def test_default_buffer_covers_entire_compression_interval(self):
        for interval in (64, 128, 256, 512):
            with self.subTest(interval=interval):
                press = COREDecodingPress(
                    base_press=SimpleNamespace(compress=Mock()),
                    compression_interval=interval, target_size=4096,
                )
                self.assertEqual(press.hidden_states_buffer_size, interval)
                keys = torch.zeros(1, 1, 4097, 1)
                cache = SimpleNamespace(layers=[SimpleNamespace(keys=keys, values=keys)])
                with patch.object(press, "compress", return_value=(keys, keys)) as compress, patch(
                    "kvpress.presses.decoding_press.extract_keys_and_values", return_value=(keys, keys)
                ):
                    for token in range(interval):
                        press.forward_hook(SimpleNamespace(layer_idx=0), [], {
                            "hidden_states": torch.tensor([[[float(token)]]]),
                            "past_key_values": cache,
                            "cache_position": torch.tensor([8192 + token]),
                        }, (keys, None))
                compress.assert_called_once()
                queries = compress.call_args.args[1]
                torch.testing.assert_close(queries.flatten(), torch.arange(interval).float())

    def test_short_explicit_buffer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "entire compression interval"):
            COREDecodingPress(
                base_press=SimpleNamespace(compress=Mock()),
                compression_interval=256, target_size=1024,
                hidden_states_buffer_size=128,
            )

    def test_standalone_defaults_and_pipeline(self):
        model = torch.nn.Linear(1, 1)
        press = object()
        answer_pipeline = Mock(return_value={"answer": "test answer"})
        with patch("sys.argv", ["core.infer", "--context", "test context", "--device", "cpu"]), patch.object(
            inference, "load_core_prefill_decoding_press", return_value=press
        ) as load_press, patch.object(
            inference.AutoTokenizer, "from_pretrained", return_value=object()
        ), patch.object(
            inference.AutoModelForCausalLM, "from_pretrained", return_value=model
        ), patch.object(inference, "pipeline", return_value=answer_pipeline) as pipeline, contextlib.redirect_stdout(io.StringIO()):
            inference.main()
        self.assertEqual(pipeline.call_args.args[0], "core-kv-press-text-generation")
        self.assertEqual(load_press.call_args.kwargs["decoding_compression_interval"], 128)
        self.assertIsNone(load_press.call_args.kwargs["decoding_target_size"])
        self.assertIsNone(load_press.call_args.kwargs["decoding_hidden_states_buffer_size"])
        self.assertIs(answer_pipeline.call_args.kwargs["press"], press)

    def test_prefill_only_remains_an_explicit_option(self):
        with patch("sys.argv", ["core.infer", "--no-enable_decoding"]):
            self.assertFalse(inference.parse_args().enable_decoding)


if __name__ == "__main__":
    unittest.main()
