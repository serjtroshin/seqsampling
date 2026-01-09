from seqsampling.parsing.extractors import RemoveThinkingParser


def test_remove_thinking_parser_strips_thinking_block():
    parser = RemoveThinkingParser()
    text = "<think>This is chain-of-thought.</think>\n Final answer."

    assert parser.extract(text) == "Final answer."


def test_remove_thinking_parser_handles_missing_tag():
    parser = RemoveThinkingParser()
    text = "No thinking tag here "

    assert parser.extract(text) == "No thinking tag here"


def test_remove_thinking_parser_is_case_insensitive_and_custom_tag():
    parser = RemoveThinkingParser()
    text = "<THINK>Reasoning</THINK> Answer"

    assert parser.extract(text, end_del="</THINK>") == "Answer"
