"""Unit tests for brain.py pure logic (no arm needed)."""

import brain


def test_parse_command_grab():
    assert brain.parse_command("grab this") == "grab"
    assert brain.parse_command("PICK UP the block") == "grab"
    assert brain.parse_command("take it") == "grab"


def test_parse_command_place():
    assert brain.parse_command("put it there") == "place"
    assert brain.parse_command("drop it in the bin") == "place"


def test_parse_command_emotion():
    assert brain.parse_command("wave") == "wave"
    assert brain.parse_command("be sad") == "sad"
    assert brain.parse_command("happy") == "happy"


def test_parse_command_quit():
    assert brain.parse_command("quit") == "quit"
    assert brain.parse_command("bye") == "quit"


def test_parse_command_howareyou():
    assert brain.parse_command("how are you") == "howareyou"


def test_parse_command_chat_fallback():
    assert brain.parse_command("blah blah") == "chat"
