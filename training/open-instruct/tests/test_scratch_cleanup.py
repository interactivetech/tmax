"""Cleanup must leave instances from other runs alone."""

import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import call, patch

spec = importlib.util.spec_from_file_location(
    "scratch_cleanup", Path(__file__).parents[1] / "scripts/tmax/cleanup_scratch_instances.py"
)
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


def test_only_instances_started_in_this_log_are_stopped(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("Starting Apptainer instance (name=swerl-123-abc123, image=/tmp/task.sif, flags=...)\n")
    listing = {"instances": [{"instance": "swerl-123-abc123"}, {"instance": "swerl-456-def456"}]}
    with patch.object(
        cleanup.subprocess,
        "run",
        side_effect=[subprocess.CompletedProcess([], 0, json.dumps(listing)), subprocess.CompletedProcess([], 0)],
    ) as run:
        assert cleanup.cleanup_instances(log) == 1
    assert run.call_args_list == [
        call(["apptainer", "instance", "list", "--json"], capture_output=True, text=True, check=True),
        call(["apptainer", "instance", "stop", "swerl-123-abc123"], check=True),
    ]


def test_missing_log_does_not_touch_instances(tmp_path):
    with patch.object(cleanup.subprocess, "run") as run:
        assert cleanup.cleanup_instances(tmp_path / "missing.log") == 0
        run.assert_not_called()


def test_run_without_instances_does_not_touch_other_runs(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("Training failed before sandbox startup.\n")
    with patch.object(cleanup.subprocess, "run") as run:
        assert cleanup.cleanup_instances(log) == 0
        run.assert_not_called()
