#!/usr/bin/env python3
"""Apply pre-verified timeline ranges through one persistent SpliceKit session."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Scripts"))

from splicekit_client import SpliceKit  # noqa: E402


def fail(message: str) -> None:
    raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--expected-duration", type=float, required=True)
    parser.add_argument("--expected-items", type=int, required=True)
    ranges_group = parser.add_mutually_exclusive_group(required=True)
    ranges_group.add_argument("--ranges-json")
    ranges_group.add_argument("--plan", help="Verified JSON plan containing a cuts array")
    parser.add_argument("--apply", action="store_true", help="Required acknowledgement for destructive edits")
    args = parser.parse_args()

    if not args.apply:
        fail("Refusing to edit without --apply")

    if args.plan:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        if not plan.get("verification", {}).get("passed"):
            fail("Refusing an unverified plan")
        if plan.get("sequence") != args.sequence:
            fail(f"Plan sequence mismatch: {plan.get('sequence')}")
        if abs(float(plan.get("timelineDuration", -1)) - args.expected_duration) > 0.001:
            fail("Plan duration does not match the expected timeline")
        if int(plan.get("itemCount", -1)) != args.expected_items:
            fail("Plan item count does not match the expected timeline")
        raw_ranges = [[item["start"], item["end"]] for item in plan.get("cuts", [])]
    else:
        raw_ranges = json.loads(args.ranges_json)
    ranges = [(float(item[0]), float(item[1])) for item in raw_ranges]
    if not ranges:
        fail("No ranges supplied")
    for start, end in ranges:
        if not (math.isfinite(start) and math.isfinite(end)):
            fail("Ranges must be finite")
        if start < 0 or end <= start or end > args.expected_duration + 1e-6:
            fail(f"Invalid range: {start}-{end}")

    ranges.sort(key=lambda item: (item[0], item[1]), reverse=True)
    ascending = list(reversed(ranges))
    for previous, current in zip(ascending, ascending[1:]):
        if current[0] < previous[1] - 1e-7:
            fail(f"Overlapping ranges: {previous} and {current}")

    client = SpliceKit()
    try:
        state = client.call("timeline.getDetailedState", limit=1)
        if state.get("sequenceName") != args.sequence:
            fail(f"Sequence changed: {state.get('sequenceName')}")
        duration = float(state.get("duration", {}).get("seconds", -1))
        if abs(duration - args.expected_duration) > 0.001:
            fail(f"Duration changed: {duration:.6f}s")
        if int(state.get("itemCount", -1)) != args.expected_items:
            fail(f"Item count changed: {state.get('itemCount')}")

        panel = client.call(
            "system.callMethodWithArgs",
            target="SpliceKitTranscriptPanel",
            selector="sharedPanel",
            args=[],
            classMethod=True,
            returnHandle=True,
        )
        panel_handle = panel.get("handle")
        if not panel_handle:
            fail(f"Unable to resolve transcript panel: {panel}")

        for index, (start, end) in enumerate(ranges, 1):
            result = client.call(
                "system.callMethodWithArgs",
                target=panel_handle,
                selector="deleteTimelineRange:end:",
                args=[
                    {"type": "double", "value": start},
                    {"type": "double", "value": end},
                ],
                classMethod=False,
                returnHandle=False,
            )
            if isinstance(result, dict) and result.get("error"):
                fail(f"Edit {index} failed at {start:.6f}-{end:.6f}: {result['error']}")
            if index % 10 == 0 or index == len(ranges):
                print(f"PROGRESS {index}/{len(ranges)}", flush=True)

        expected_final = args.expected_duration - sum(end - start for start, end in ranges)
        final_state = client.call("timeline.getDetailedState", limit=1)
        final_duration = float(final_state.get("duration", {}).get("seconds", -1))
        response = {
            "status": "ok" if abs(final_duration - expected_final) <= 0.001 else "mismatch",
            "sequence": final_state.get("sequenceName"),
            "appliedCount": len(ranges),
            "removedDuration": args.expected_duration - final_duration,
            "expectedDuration": expected_final,
            "actualDuration": final_duration,
            "itemCount": final_state.get("itemCount"),
        }
        print(json.dumps(response, indent=2))
        if response["status"] != "ok":
            fail("Final duration did not match the verified plan")
    finally:
        client.close()


if __name__ == "__main__":
    main()
