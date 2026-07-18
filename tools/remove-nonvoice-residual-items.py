#!/usr/bin/env python3
"""Delete exact one-frame non-voice objects left by an older loaded dylib."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Scripts"))

from splicekit_client import SpliceKit  # noqa: E402


def seconds(value: str | None) -> float:
    raw = (value or "0s").removesuffix("s")
    if "/" in raw:
        numerator, denominator = raw.split("/", 1)
        return float(numerator) / float(denominator)
    return float(raw or 0)


def overlaps(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return min(left[1], right[1]) > max(left[0], right[0]) + 1e-9


def object_handle(client: SpliceKit, target: str, selector: str, *, class_method: bool) -> str:
    result = client.call(
        "system.callMethodWithArgs",
        target=target,
        selector=selector,
        args=[],
        classMethod=class_method,
        returnHandle=True,
    )
    handle = result.get("handle")
    if not handle:
        raise RuntimeError(f"Unable to resolve {target} {selector}: {result}")
    return handle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", required=True)
    parser.add_argument("--current-plan", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        raise SystemExit("Refusing to edit without --apply")

    source_plan = json.loads(Path(args.source_plan).read_text(encoding="utf-8"))
    current_plan = json.loads(Path(args.current_plan).read_text(encoding="utf-8"))
    if not source_plan.get("verification", {}).get("passed"):
        raise SystemExit("Source voice plan is not verified")
    if not current_plan.get("verification", {}).get("passed"):
        raise SystemExit("Current edit plan is not verified")
    voice = [(float(item["start"]), float(item["end"])) for item in source_plan["clearVoice"]]
    frame = float(current_plan["frameDuration"])

    client = SpliceKit()
    try:
        state = client.call("timeline.getDetailedState", limit=1000)
        if state.get("sequenceName") != args.sequence:
            raise SystemExit(f"Sequence changed: {state.get('sequenceName')}")
        actual_duration = float(state["duration"]["seconds"])
        expected_duration = float(current_plan["expectedDuration"])
        expected_count = int(current_plan["expectedItemCount"])
        residual_duration = actual_duration - expected_duration
        residual_count = int(state["itemCount"]) - expected_count
        if residual_duration <= 1e-6 or residual_count <= 0:
            raise SystemExit("No positive residual remains")

        with tempfile.NamedTemporaryFile(prefix="splicekit-zero-buffer-audit-", suffix=".fcpxml") as temp:
            export = client.call("fcpxml.export", path=temp.name)
            if export.get("status") != "ok":
                raise RuntimeError(f"FCPXML export failed: {export}")
            root = ET.parse(temp.name).getroot()

        project = next(
            (node for node in root.findall(".//project") if node.get("name") == args.sequence),
            None,
        )
        if project is None:
            raise RuntimeError("Active sequence missing from audit FCPXML")
        spine = project.find("./sequence/spine")
        xml_items = [
            item for item in list(spine if spine is not None else [])
            if item.tag in {"sync-clip", "asset-clip"}
        ]
        live_items = state.get("items", [])
        if len(xml_items) != len(live_items) or len(live_items) != int(state["itemCount"]):
            raise RuntimeError("FCPXML/live item ordering could not be proven")

        candidates = []
        for index, (xml_item, live_item) in enumerate(zip(xml_items, live_items)):
            source_start = seconds(xml_item.get("start"))
            xml_duration = seconds(xml_item.get("duration"))
            live_duration = float(live_item["duration"]["seconds"])
            if abs(xml_duration - live_duration) > 1e-6:
                raise RuntimeError(f"FCPXML/live duration mismatch at item {index}")
            source_range = (source_start, source_start + xml_duration)
            has_voice = any(overlaps(source_range, protected) for protected in voice)
            if not has_voice:
                candidates.append({
                    "index": index,
                    "handle": live_item["handle"],
                    "duration": live_duration,
                    "timelineStart": float(live_item["startTime"]["seconds"]),
                    "sourceStart": source_start,
                })

        candidate_duration = sum(item["duration"] for item in candidates)
        if len(candidates) != residual_count:
            raise RuntimeError(f"Residual count mismatch: found {len(candidates)}, expected {residual_count}")
        if abs(candidate_duration - residual_duration) > 1e-6:
            raise RuntimeError(
                f"Residual duration mismatch: found {candidate_duration:.9f}, expected {residual_duration:.9f}"
            )
        if any(item["duration"] > frame + 1e-6 for item in candidates):
            raise RuntimeError("A residual candidate is longer than one frame")

        app = object_handle(client, "NSApplication", "sharedApplication", class_method=True)
        delegate = object_handle(client, app, "delegate", class_method=False)
        container = object_handle(client, delegate, "activeEditorContainer", class_method=False)
        timeline = object_handle(client, container, "timelineModule", class_method=False)

        for item in reversed(candidates):
            selection = client.call(
                "system.callMethodWithArgs",
                target="NSArray",
                selector="arrayWithObject:",
                args=[{"type": "handle", "value": item["handle"]}],
                classMethod=True,
                returnHandle=True,
            )
            selection_handle = selection.get("handle")
            if not selection_handle:
                raise RuntimeError(f"Unable to build exact selection: {selection}")
            client.call(
                "system.callMethodWithArgs",
                target=timeline,
                selector="setSelectedItems:",
                args=[{"type": "handle", "value": selection_handle}],
                classMethod=False,
                returnHandle=False,
            )
            client.call(
                "system.callMethodWithArgs",
                target=timeline,
                selector="delete:",
                args=[{"type": "nil"}],
                classMethod=False,
                returnHandle=False,
            )

        final_state = client.call("timeline.getDetailedState", limit=1000)
        final_duration = float(final_state["duration"]["seconds"])
        passed = (
            final_state.get("sequenceName") == args.sequence
            and abs(final_duration - expected_duration) <= 1e-6
            and int(final_state["itemCount"]) == expected_count
        )
        print(json.dumps({
            "status": "ok" if passed else "mismatch",
            "sequence": final_state.get("sequenceName"),
            "deleted": candidates,
            "actualDuration": final_duration,
            "expectedDuration": expected_duration,
            "itemCount": final_state.get("itemCount"),
            "expectedItemCount": expected_count,
        }, indent=2))
        if not passed:
            raise SystemExit("Final residual audit failed")
    finally:
        client.close()


if __name__ == "__main__":
    main()
