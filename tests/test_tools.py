from pathlib import Path

from hx import tools
from hx.tools import ToolContext


def ctx(workdir: Path, **kw) -> ToolContext:
    return ToolContext(workdir=workdir.resolve(), auto_approve=True, **kw)


def test_write_then_read_numbers_lines(workdir):
    c = ctx(workdir)
    r = tools.write_file(c, "pkg/a.py", "x = 1\ny = 2\n")
    assert r.ok and "Created pkg/a.py (2 lines)" in r.output
    r = tools.read_file(c, "pkg/a.py")
    assert r.ok
    assert r.output.splitlines()[1] == "    2\ty = 2"


def test_overwrite_requires_read_first(workdir):
    (workdir / "a.txt").write_text("old\n")
    c = ctx(workdir)
    r = tools.write_file(c, "a.txt", "new\n")
    assert not r.ok and "read_file" in r.output
    tools.read_file(c, "a.txt")
    assert tools.write_file(c, "a.txt", "new\n").ok
    assert (workdir / "a.txt").read_text() == "new\n"


def test_write_rejects_file_changed_since_read(workdir):
    p = workdir / "a.txt"
    p.write_text("one\n")
    c = ctx(workdir)
    tools.read_file(c, "a.txt")
    import os
    import time

    time.sleep(0.01)
    p.write_text("changed elsewhere\n")
    os.utime(p, None)
    r = tools.edit_file(c, "a.txt", "changed", "x")
    assert not r.ok and "changed on disk" in r.output


def test_edit_must_be_unique(workdir):
    (workdir / "a.py").write_text("a = 1\na = 1\n")
    c = ctx(workdir)
    tools.read_file(c, "a.py")
    r = tools.edit_file(c, "a.py", "a = 1", "a = 2")
    assert not r.ok and "occurs 2 times" in r.output
    r = tools.edit_file(c, "a.py", "a = 1", "a = 2", replace_all=True)
    assert r.ok
    assert (workdir / "a.py").read_text() == "a = 2\na = 2\n"


def test_edit_whitespace_hint(workdir):
    (workdir / "a.py").write_text("def f():\n    return 1\n")
    c = ctx(workdir)
    tools.read_file(c, "a.py")
    r = tools.edit_file(c, "a.py", "return 1  ", "return 2")
    assert not r.ok and "whitespace" in r.output


def test_read_pagination(workdir):
    (workdir / "big.txt").write_text("\n".join(str(i) for i in range(1, 501)))
    r = tools.read_file(ctx(workdir), "big.txt", offset=10, limit=5)
    lines = r.output.splitlines()
    assert lines[0] == "   10\t10" and "continue with offset=15" in lines[-1]


def test_bash_exit_codes_and_timeout(workdir):
    c = ctx(workdir, bash_timeout=2)
    ok = tools.bash(c, "echo hi")
    assert ok.ok and ok.output == "exit code: 0\nhi"
    bad = tools.bash(c, "python3 -c 'import sys; print(\"boom\"); sys.exit(3)'")
    assert not bad.ok and bad.kind == "nonzero_exit" and "exit code: 3" in bad.output
    slow = tools.bash(c, "sleep 10", timeout=1)
    assert not slow.ok and slow.kind == "timeout"


def test_bash_stdin_is_closed(workdir):
    r = tools.bash(ctx(workdir, bash_timeout=5), "python3 -c 'input()'")
    assert not r.ok and r.kind == "nonzero_exit" and "EOFError" in r.output


def test_bash_needs_confirmation(workdir):
    c = ToolContext(workdir=workdir, auto_approve=False, confirm=lambda p: False)
    r = tools.bash(c, "echo hi")
    assert r.kind == "denied"


def test_grep_and_list_skip_venv(workdir):
    (workdir / "src").mkdir()
    (workdir / "src" / "m.py").write_text("def snake():\n    pass\n")
    (workdir / ".venv").mkdir()
    (workdir / ".venv" / "x.py").write_text("def snake(): pass\n")
    c = ctx(workdir)
    assert tools.grep(c, r"def snake").output == "src/m.py:1: def snake():"
    assert tools.list_files(c, ".", "*.py").output == "src/m.py"


def test_benign_failures():
    r = tools.ToolResult(False, "exit code: 1 (no output)", "nonzero_exit")
    assert tools.is_benign("bash", {"command": "grep -n foo x.py"}, r)
    assert not tools.is_benign("bash", {"command": "python -m pytest"}, r)
    assert tools.is_benign("bash", {"command": "x"}, tools.ToolResult(False, "", "denied"))


def test_execute_unknown_and_bad_args(workdir):
    c = ctx(workdir)
    assert "Unknown tool" in tools.execute(c, "nope", {}).output
    assert "Bad arguments" in tools.execute(c, "read_file", {"wat": 1}).output
    assert "not valid JSON" in tools.execute(c, "read_file", {"_unparsed": "{"}).output


def test_bash_strips_terminal_escapes(workdir):
    r = tools.bash(ctx(workdir), r"printf '\033[?1049h\033[1;24r\033(Bhello\033[0m\n'")
    assert r.output == "exit code: 0\nhello"
