from __future__ import annotations

import unittest
from typing import Any

from rag_server.utils import coerce_bool, coerce_message_content, parse_json_object


class CoerceMessageContentTests(unittest.TestCase):
    def test_string_passthrough(self) -> None:
        self.assertEqual(coerce_message_content("hello"), "hello")

    def test_list_of_dicts_with_text(self) -> None:
        content = [{"text": "part1"}, {"text": "part2"}]
        self.assertEqual(coerce_message_content(content), "part1\npart2")

    def test_list_of_mixed(self) -> None:
        content = [{"text": "a"}, "b", 42]
        self.assertEqual(coerce_message_content(content), "a\nb\n42")

    def test_non_string_non_list(self) -> None:
        self.assertEqual(coerce_message_content(123), "123")


class ParseJsonObjectTests(unittest.TestCase):
    def test_plain_json(self) -> None:
        result = parse_json_object('{"key": "value"}')
        self.assertEqual(result, {"key": "value"})

    def test_json_in_surrounding_text(self) -> None:
        result = parse_json_object('Here is: {"key": "value"} done.')
        self.assertEqual(result, {"key": "value"})

    def test_empty_string_returns_empty_dict(self) -> None:
        self.assertEqual(parse_json_object(""), {})

    def test_invalid_json_returns_empty_dict(self) -> None:
        self.assertEqual(parse_json_object("not json"), {})

    def test_json_array_returns_empty_dict(self) -> None:
        self.assertEqual(parse_json_object("[1,2,3]"), {})


class CoerceBoolTests(unittest.TestCase):
    def test_bool_passthrough(self) -> None:
        self.assertTrue(coerce_bool(True))
        self.assertFalse(coerce_bool(False))

    def test_string_true_values(self) -> None:
        for value in ("true", "True", "yes", "on", "1", "是", "需要"):
            self.assertTrue(coerce_bool(value), msg=f"Expected True for {value!r}")

    def test_string_false_values(self) -> None:
        for value in ("false", "False", "no", "off", "0"):
            self.assertFalse(coerce_bool(value), msg=f"Expected False for {value!r}")

    def test_none_uses_default(self) -> None:
        self.assertFalse(coerce_bool(None))
        self.assertTrue(coerce_bool(None, default=True))

    def test_unknown_string_uses_bool(self) -> None:
        self.assertTrue(coerce_bool("unknown"))


if __name__ == "__main__":
    unittest.main()
