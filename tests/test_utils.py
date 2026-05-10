"""工具函数单元测试。

测试消息内容类型转换（coerce_message_content）、JSON 对象解析（parse_json_object）、
以及布尔值强制转换（coerce_bool）等实用工具函数。
"""

from __future__ import annotations

import unittest

from rag_server.utils import coerce_bool, coerce_message_content, parse_json_object


class CoerceMessageContentTests(unittest.TestCase):
    # 验证字符串输入直接透传
    def test_string_passthrough(self) -> None:
        self.assertEqual(coerce_message_content("hello"), "hello")

    # 验证包含 text 字段的字典列表被正确拼接为多行文本
    def test_list_of_dicts_with_text(self) -> None:
        content = [{"text": "part1"}, {"text": "part2"}]
        self.assertEqual(coerce_message_content(content), "part1\npart2")

    # 验证混合类型列表（字典、字符串、数字）被正确转换为文本
    def test_list_of_mixed(self) -> None:
        content = [{"text": "a"}, "b", 42]
        self.assertEqual(coerce_message_content(content), "a\nb\n42")

    # 验证非字符串非列表输入（如数字）被转为字符串
    def test_non_string_non_list(self) -> None:
        self.assertEqual(coerce_message_content(123), "123")


class ParseJsonObjectTests(unittest.TestCase):
    # 验证纯 JSON 字符串被正确解析
    def test_plain_json(self) -> None:
        result = parse_json_object('{"key": "value"}')
        self.assertEqual(result, {"key": "value"})

    # 验证从混合文本中提取 JSON 对象（忽略前后非 JSON 内容）
    def test_json_in_surrounding_text(self) -> None:
        result = parse_json_object('Here is: {"key": "value"} done.')
        self.assertEqual(result, {"key": "value"})

    # 验证空字符串返回空字典
    def test_empty_string_returns_empty_dict(self) -> None:
        self.assertEqual(parse_json_object(""), {})

    # 验证非法 JSON 返回空字典
    def test_invalid_json_returns_empty_dict(self) -> None:
        self.assertEqual(parse_json_object("not json"), {})

    # 验证 JSON 数组输入返回空字典（仅接受 JSON 对象）
    def test_json_array_returns_empty_dict(self) -> None:
        self.assertEqual(parse_json_object("[1,2,3]"), {})


class CoerceBoolTests(unittest.TestCase):
    # 验证布尔值输入直接透传
    def test_bool_passthrough(self) -> None:
        self.assertTrue(coerce_bool(True))
        self.assertFalse(coerce_bool(False))

    # 验证常见真值字符串（true/yes/on/1/是/需要）被识别为 True
    def test_string_true_values(self) -> None:
        for value in ("true", "True", "yes", "on", "1", "是", "需要"):
            self.assertTrue(coerce_bool(value), msg=f"Expected True for {value!r}")

    # 验证常见假值字符串（false/no/off/0）被识别为 False
    def test_string_false_values(self) -> None:
        for value in ("false", "False", "no", "off", "0"):
            self.assertFalse(coerce_bool(value), msg=f"Expected False for {value!r}")

    # 验证 None 输入使用默认值
    def test_none_uses_default(self) -> None:
        self.assertFalse(coerce_bool(None))
        self.assertTrue(coerce_bool(None, default=True))

    # 验证未知字符串回退到 Python 内置 bool() 判断（非空字符串为 True）
    def test_unknown_string_uses_bool(self) -> None:
        self.assertTrue(coerce_bool("unknown"))


if __name__ == "__main__":
    unittest.main()
