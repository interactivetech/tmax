"""Stop surviving Apptainer instances whose exact names appear in one run's log."""

import argparse
import json
import re
import subprocess
from pathlib import Path

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def cleanup_instances(log_path: Path) -> int:
    if not log_path.is_file():
        return 0
    started = set(re.findall(r"Starting Apptainer instance \(name=(swerl-\d+-[0-9a-f]+),", log_path.read_text()))
    if not started:
        return 0
    listing = subprocess.run(["apptainer", "instance", "list", "--json"], capture_output=True, text=True, check=True)
    current = json.loads(listing.stdout)["instances"]
    stopped = 0
    for instance in current:
        name = instance["instance"]
        if name in started:
            subprocess.run(["apptainer", "instance", "stop", name], check=True)
            stopped += 1
    logger.info("Stopped %d surviving instances from %s", stopped, log_path)
    return stopped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_log", type=Path)
    args = parser.parse_args()
    cleanup_instances(args.run_log)


if __name__ == "__main__":
    main()
