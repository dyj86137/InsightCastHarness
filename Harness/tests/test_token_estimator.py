from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from context.token import (  # noqa: E402
    HeuristicTokenEstimator,
    TiktokenTokenEstimator,
    create_token_estimator_from_env,
)
from infra.exception import ValidationError as HarnessValidationError  # noqa: E402


class TokenEstimatorTests(unittest.TestCase):
    def test_env_factory_creates_heuristic_estimator(self) -> None:
        estimator = create_token_estimator_from_env(
            {
                "MYHARNESS_TOKENIZER_PROVIDER": "heuristic",
                "MYHARNESS_TOKENIZER_CHARS_PER_TOKEN": "2",
            }
        )

        self.assertIsInstance(estimator, HeuristicTokenEstimator)
        self.assertEqual(estimator.chars_per_token, 2)
        self.assertEqual(estimator.estimate_text("abcd").text_tokens, 3)

    def test_tiktoken_estimator_falls_back_when_not_strict(self) -> None:
        estimator = create_token_estimator_from_env(
            {
                "MYHARNESS_TOKENIZER_PROVIDER": "tiktoken",
                "MYHARNESS_TOKENIZER_ENCODING": "missing_test_encoding",
                "MYHARNESS_TOKENIZER_STRICT": "0",
            }
        )

        self.assertIsInstance(estimator, TiktokenTokenEstimator)
        self.assertFalse(estimator.available)
        estimate = estimator.estimate_text("abcd")
        self.assertEqual(estimate.metadata["estimator"], "heuristic")
        self.assertEqual(estimate.metadata["requested_estimator"], "tiktoken")
        self.assertIn("fallback_reason", estimate.metadata)

    def test_tiktoken_estimator_strict_mode_raises(self) -> None:
        with self.assertRaises(HarnessValidationError):
            create_token_estimator_from_env(
                {
                    "MYHARNESS_TOKENIZER_PROVIDER": "tiktoken",
                    "MYHARNESS_TOKENIZER_ENCODING": "missing_test_encoding",
                    "MYHARNESS_TOKENIZER_STRICT": "1",
                }
            )


if __name__ == "__main__":
    unittest.main()
