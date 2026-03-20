"""
Test script for claude_cli.py — verifies VSCode-like behavior.

Run: python test_claude_cli.py
"""

from claude_cli import ClaudeSession


def test_multi_turn():
    """
    Test 1: Multi-turn memory (like VSCode's persistent chat panel).

    We tell Claude a made-up fact in turn 1, then ask about it in turn 2.
    If session continuity works, it remembers. If not, it won't.
    """
    print("=" * 60)
    print("TEST: Multi-turn session memory")
    print("=" * 60)

    session = ClaudeSession(model="haiku")

    r1 = session.text("Remember this secret code: PINEAPPLE-742. Just confirm you noted it.")
    print(f"Turn 1: {r1}\n")

    r2 = session.text("What was the secret code I just told you?")
    print(f"Turn 2: {r2}\n")

    assert session.session_id is not None, "Session ID should be set after first call"

    if "PINEAPPLE-742" in r2.upper():
        print("PASS — Claude remembered across turns (multi-turn works)\n")
    else:
        print("FAIL — Claude did not recall the secret code\n")


def test_session_resume():
    """
    Test 2: Session resume (like reopening a VSCode conversation).

    Start a session, save its ID, create a new ClaudeSession from that ID,
    and verify the conversation context carries over.
    """
    print("=" * 60)
    print("TEST: Session resume")
    print("=" * 60)

    session1 = ClaudeSession(model="haiku")
    session1.text("My favorite color is cerulean. Just acknowledge.")
    saved_id = session1.session_id
    print(f"Session 1 ID: {saved_id}\n")

    # Resume from a fresh object — simulates a new Python process picking up
    # where a previous run (or VSCode session) left off.
    session2 = ClaudeSession.resume(saved_id, model="haiku")
    r = session2.text("What is my favorite color?")
    print(f"Resumed turn: {r}\n")

    if "cerulean" in r.lower():
        print("PASS — Resumed session retained context\n")
    else:
        print("FAIL — Resumed session lost context\n")


def test_tool_use():
    """
    Test 3: Tool use (like VSCode running Bash/Read on your behalf).

    Ask Claude to read a file using the Read tool. If tools work,
    it will return actual file contents.
    """
    print("=" * 60)
    print("TEST: Tool use (Read)")
    print("=" * 60)

    session = ClaudeSession(model="haiku", tools=["Read"], max_turns=3)
    r = session.text("Read the file claude_cli.py and tell me the first line of the docstring.")
    print(f"Response: {r}\n")

    if "mirror" in r.lower() or "vscode" in r.lower() or "cli" in r.lower():
        print("PASS — Claude read the file and reported its contents\n")
    else:
        print("UNCERTAIN — Check response manually\n")


def test_json_metadata():
    """
    Test 4: Verify the JSON response includes metadata (session_id, usage).

    This mirrors VSCode's internal tracking of context usage and costs.
    """
    print("=" * 60)
    print("TEST: JSON response metadata")
    print("=" * 60)

    session = ClaudeSession(model="haiku")
    response = session.prompt("Say hello in exactly 3 words.")

    print(f"Keys returned: {list(response.keys())}")
    print(f"session_id: {response.get('session_id')}")
    print(f"result: {response.get('result')}")
    print()

    has_session = response.get("session_id") is not None
    has_result = response.get("result") is not None

    if has_session and has_result:
        print("PASS — JSON response has session_id and result\n")
    else:
        print(f"FAIL — missing session_id={has_session}, result={has_result}\n")


if __name__ == "__main__":
    tests = [test_multi_turn, test_session_resume, test_tool_use, test_json_metadata]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"ERROR in {test.__name__}: {e}\n")
        print()
