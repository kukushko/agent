import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from sandbox import SandboxIO, SandboxPolicy, SandboxResult
from sandbox.base import SandboxStreamResult
from sandbox.docker import build_context_fingerprint
from tools import ToolEnvironment, discover_tool_packages
from workpaths import WorkPathResolver


class FakeSandbox:
    def __init__(self, available: bool = True, exit_code: int = 0) -> None:
        self._available = available
        self.exit_code = exit_code
        self.calls = []

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def unavailable_reason(self) -> str | None:
        return None if self._available else "test sandbox is unavailable"

    def refresh_availability(self) -> bool:
        return self._available

    def run(
        self,
        argv: list[str],
        policy: SandboxPolicy,
        io: SandboxIO = SandboxIO(),
    ) -> SandboxResult:
        self.calls.append((argv, policy, io))
        return SandboxResult(
            exit_code=self.exit_code,
            stdout=SandboxStreamResult("ok\n", None, 3, False),
            stderr=SandboxStreamResult("", None, 0, False),
            timed_out=False,
            output_limit_exceeded=False,
            duration_ms=5,
        )


class WorkPathResolverTest(unittest.TestCase):
    def test_normalizes_relative_path_and_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            resolver = WorkPathResolver(Path(tmp) / "work")
            self.assertEqual(resolver.relative("src/../result.py"), "result.py")
            with self.assertRaisesRegex(ValueError, "escapes the work directory"):
                resolver.resolve("../outside.py")


class SandboxFingerprintTest(unittest.TestCase):
    def test_fingerprint_tracks_content_not_modification_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = Path(tmp)
            dockerfile = context / "Dockerfile"
            dockerfile.write_text("FROM scratch\n", encoding="utf-8")
            original_stat = dockerfile.stat()
            first = build_context_fingerprint(context)

            dockerfile.write_text("FROM busybox\n", encoding="utf-8")
            os.utime(
                dockerfile,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            second = build_context_fingerprint(context)

            self.assertNotEqual(first, second)


class PythonToolsTest(unittest.TestCase):
    def test_evaluator_accepts_statements_with_a_named_result(self) -> None:
        evaluator = (
            Path(__file__).parent / "sandbox" / "python312" / "bin" / "eval_python.py"
        )
        completed = subprocess.run(
            [
                "python3",
                str(evaluator),
                "x = radians(52); result = {'exact': sin(x), 'input': x}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("'exact':", completed.stdout)
        self.assertIn("'input':", completed.stdout)

    def test_evaluator_requires_result_for_statement_snippets(self) -> None:
        evaluator = (
            Path(__file__).parent / "sandbox" / "python312" / "bin" / "eval_python.py"
        )
        completed = subprocess.run(
            ["python3", str(evaluator), "x = radians(52)"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("must assign the value to 'result'", completed.stderr)

    def test_package_is_hidden_when_sandbox_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            environment = ToolEnvironment(
                work_dir=Path(tmp) / "work",
                sandbox=FakeSandbox(available=False),
            )
            with self.assertLogs("tools.runtime", level="WARNING") as logs:
                registry = discover_tool_packages(environment)

            names = [definition.name for definition in registry.definitions()]
            self.assertFalse(any(name.startswith("python.") for name in names))
            self.assertIn("test sandbox is unavailable", "\n".join(logs.output))

    def test_run_uses_relative_paths_io_and_no_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            work_dir.mkdir()
            (work_dir / "script.py").write_text("print('ok')\n", encoding="utf-8")
            (work_dir / "input.txt").write_text("hello\n", encoding="utf-8")
            sandbox = FakeSandbox()
            registry = discover_tool_packages(
                ToolEnvironment(work_dir=work_dir, sandbox=sandbox)
            )

            result = registry.execute(
                "python.run",
                {
                    "path": "script.py",
                    "arguments": ["--value", "2"],
                    "stdin_path": "input.txt",
                    "stdout_path": "output.txt",
                },
            )

            self.assertTrue(result.ok)
            argv, policy, io = sandbox.calls[0]
            self.assertEqual(argv, ["python", "./script.py", "--value", "2"])
            self.assertFalse(policy.network)
            self.assertEqual(io.stdin_path, "input.txt")
            self.assertEqual(io.stdout_path, "output.txt")

    def test_syntax_and_eval_use_fixed_sandbox_programs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            work_dir.mkdir()
            (work_dir / "script.py").write_text("value = 1\n", encoding="utf-8")
            sandbox = FakeSandbox()
            registry = discover_tool_packages(
                ToolEnvironment(work_dir=work_dir, sandbox=sandbox)
            )

            syntax = registry.execute("python.check_syntax", {"path": "script.py"})
            evaluated = registry.execute("python.eval", {"expression": "6 * 7"})

            self.assertTrue(syntax.ok)
            self.assertTrue(evaluated.ok)
            self.assertTrue(syntax.result["valid"])
            self.assertEqual(sandbox.calls[0][0][-1], "script.py")
            self.assertEqual(sandbox.calls[1][0][-1], "6 * 7")
            self.assertFalse(sandbox.calls[0][1].network)
            self.assertFalse(sandbox.calls[1][1].network)

    def test_eval_returns_a_short_redirected_value_inline(self) -> None:
        class RedirectSandbox(FakeSandbox):
            def __init__(self, work_dir: Path) -> None:
                super().__init__()
                self.work_dir = work_dir

            def run(self, argv, policy, io=SandboxIO()):
                assert io.stdout_path is not None
                value = "51954.0876\n"
                (self.work_dir / io.stdout_path).write_text(value, encoding="utf-8")
                return SandboxResult(
                    exit_code=0,
                    stdout=SandboxStreamResult(None, io.stdout_path, len(value), False),
                    stderr=SandboxStreamResult("", None, 0, False),
                    timed_out=False,
                    output_limit_exceeded=False,
                    duration_ms=5,
                )

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            work_dir.mkdir()
            registry = discover_tool_packages(
                ToolEnvironment(work_dir=work_dir, sandbox=RedirectSandbox(work_dir))
            )

            result = registry.execute(
                "python.eval",
                {"expression": "6 * 7", "stdout_path": "result.txt"},
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.result["value"], "51954.0876")

    def test_output_cannot_overwrite_executed_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            work_dir.mkdir()
            (work_dir / "script.py").write_text("print('ok')\n", encoding="utf-8")
            registry = discover_tool_packages(
                ToolEnvironment(work_dir=work_dir, sandbox=FakeSandbox())
            )

            result = registry.execute(
                "python.run",
                {"path": "script.py", "stdout_path": "script.py"},
            )

            self.assertFalse(result.ok)
            self.assertIn("must not overwrite", result.result["error"])

    def test_nonzero_python_exit_marks_tool_result_as_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = discover_tool_packages(
                ToolEnvironment(
                    work_dir=Path(tmp) / "work",
                    sandbox=FakeSandbox(exit_code=1),
                )
            )

            result = registry.execute("python.eval", {"expression": "1 / 0"})

            self.assertFalse(result.ok)
            self.assertEqual(result.result["exit_code"], 1)


if __name__ == "__main__":
    unittest.main()
