import tempfile
import unittest
from pathlib import Path

from jobruntime.service import (
    JobPolicy,
    JobService,
    JobTool,
    JobUsage,
    delegation_from_metadata,
    require_safe_directory,
)
from tools import Delegation
from tools.runtime import ToolResult
from tools.jobs import normalize_generated_code


class UnusedSandbox:
    @property
    def is_available(self):
        return True

    @property
    def unavailable_reason(self):
        return None


class JobPolicyTest(unittest.TestCase):
    def test_repairs_one_extra_json_escape_layer_in_generated_code(self) -> None:
        escaped = 'value = tools.example.read(path=\\"result.py\\")\\nresult = value.content'
        self.assertEqual(
            normalize_generated_code(escaped),
            'value = tools.example.read(path="result.py")\nresult = value.content',
        )

        valid = "result = 'first\\nsecond'"
        self.assertEqual(normalize_generated_code(valid), valid)

    def test_external_metadata_controls_delegation_without_tool_names(self) -> None:
        denied = delegation_from_metadata(
            {"delegation": {"allowed": False, "costs": {}, "reason": "unsafe"}}
        )
        default = delegation_from_metadata(None)

        self.assertFalse(denied.allowed)
        self.assertEqual(denied.reason, "unsafe")
        self.assertTrue(default.allowed)
        self.assertEqual(default.costs, {"tool_calls": 1})

    def test_budget_is_charged_before_execution(self) -> None:
        service = JobService(
            Path("unused"),
            UnusedSandbox(),
            JobPolicy(quotas={"tool_calls": 2, "records": 1}),
        )
        service._tools = {
            "alpha.fetch": JobTool(
                "alpha.fetch",
                "Fetch one record.",
                {"type": "object"},
                Delegation(costs={"tool_calls": 1, "records": 1}),
            ),
            "beta.transform": JobTool(
                "beta.transform",
                "Transform records.",
                {"type": "object"},
                Delegation(costs={"tool_calls": 1}),
            ),
        }
        usage = JobUsage()

        service._charge("alpha.fetch", usage)
        service._charge("beta.transform", usage)

        self.assertEqual(usage.as_dict(), {"records": 1, "tool_calls": 2})
        with self.assertRaisesRegex(RuntimeError, "tool_calls"):
            service._charge("beta.transform", usage)

    def test_measured_resource_usage_is_charged_generically(self) -> None:
        service = JobService(
            Path("unused"),
            UnusedSandbox(),
            JobPolicy(quotas={"model_tokens": 10}),
        )
        usage = JobUsage()
        service._charge_usage({"model_tokens": 7}, usage)
        self.assertEqual(usage.as_dict(), {"model_tokens": 7})
        with self.assertRaisesRegex(RuntimeError, "model_tokens"):
            service._charge_usage({"model_tokens": 4}, usage)
        self.assertEqual(usage.as_dict(), {"model_tokens": 11})

    def test_broker_rejects_symlinked_ipc_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "job"
            root.mkdir()
            outside = Path(tmp) / "outside"
            outside.mkdir()
            unsafe = root / "responses"
            unsafe.symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(RuntimeError, "unsafe"):
                require_safe_directory(root, unsafe)


if __name__ == "__main__":
    unittest.main()
