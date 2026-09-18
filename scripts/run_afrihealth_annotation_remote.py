"""Run on the VPS: launch an isolated annotation job using the live app's parsed secret."""

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="/opt/ghana-health-ai/research-annotation-v2")
    parser.add_argument("--name", default="gha-research-annotator-v2")
    parser.add_argument("--script", choices=["scripts/annotate-afrihealth-akan.ts", "scripts/annotate-source-corpus.ts"], default="scripts/annotate-afrihealth-akan.ts")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    directory = Path(options.directory).resolve(strict=True)
    config = json.loads(subprocess.check_output(["docker", "inspect", "ghana-health-ai-web"]))[0]["Config"]
    active = dict(entry.split("=", 1) for entry in config["Env"] if "=" in entry)
    if not active.get("OPENAI_API_KEY"):
        raise SystemExit("The live application has no annotation provider key configured.")
    environment = dict(os.environ)
    command = ["docker", "run", "--rm", "--user", "root", "--name", options.name]
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
        if active.get(key):
            environment[key] = active[key]
            command.extend(["--env", key])
    arguments = options.args[1:] if options.args[:1] == ["--"] else options.args
    command.extend([
        "--volume", f"{directory}:/app/research-run",
        "--workdir", "/app/research-run",
        config["Image"],
        "/app/node_modules/.bin/tsx", options.script,
        *arguments,
    ])
    raise SystemExit(subprocess.call(command, env=environment))


if __name__ == "__main__":
    main()
