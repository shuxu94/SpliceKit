#!/usr/bin/env python3
"""Map a source-time clear-voice mask through an edited FCPXML timeline."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def seconds(value: str | None) -> float:
    raw = (value or "0s").removesuffix("s")
    if "/" in raw:
        numerator, denominator = raw.split("/", 1)
        return float(numerator) / float(denominator)
    return float(raw or 0)


def merge(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if result and start <= result[-1][1] + 1e-9:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def complement(ranges: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merge(ranges):
        if start > cursor:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        result.append((cursor, duration))
    return result


def overlaps(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return min(left[1], right[1]) > max(left[0], right[0]) + 1e-9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--fcpxml", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    source_plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if source_plan.get("mode") != "voice" or not source_plan.get("verification", {}).get("passed"):
        raise SystemExit("Source plan is not a verified voice-mode plan")
    source_voice = [
        (float(item["start"]), float(item["end"]))
        for item in source_plan["clearVoice"]
    ]
    frame_duration = float(source_plan["frameDuration"])

    root = ET.parse(args.fcpxml).getroot()
    project = next(
        (node for node in root.findall(".//project") if node.get("name") == args.sequence),
        None,
    )
    if project is None:
        raise SystemExit(f"Sequence not found in FCPXML: {args.sequence}")
    spine = project.find("./sequence/spine")
    if spine is None:
        raise SystemExit("Sequence has no spine")

    items: list[tuple[float, float, float]] = []
    for item in list(spine):
        if item.tag not in {"sync-clip", "asset-clip"}:
            continue
        timeline_start = seconds(item.get("offset"))
        source_start = seconds(item.get("start"))
        duration = seconds(item.get("duration"))
        if duration > 0:
            items.append((timeline_start, source_start, duration))
    if not items:
        raise SystemExit("No editable spine clips found")

    timeline_duration = max(start + duration for start, _, duration in items)
    mapped_voice: list[tuple[float, float]] = []
    for timeline_start, source_start, duration in items:
        source_end = source_start + duration
        for voice_start, voice_end in source_voice:
            overlap_start = max(source_start, voice_start)
            overlap_end = min(source_end, voice_end)
            if overlap_end > overlap_start:
                mapped_voice.append((
                    timeline_start + overlap_start - source_start,
                    timeline_start + overlap_end - source_start,
                ))
    protected = merge(mapped_voice)

    logical_cuts: list[tuple[float, float]] = []
    for raw_start, raw_end in complement(protected, timeline_duration):
        start = math.ceil((raw_start / frame_duration) - 1e-9) * frame_duration
        end = math.floor((raw_end / frame_duration) + 1e-9) * frame_duration
        start = max(0.0, start)
        end = min(timeline_duration, end)
        if end - start >= frame_duration - 1e-6:
            logical_cuts.append((start, end))

    # deleteTimelineRange selects one exact bladed timeline object. A logical
    # non-voice range may cross an existing synchronized-clip edit, so split it
    # into single-object operations before applying it in descending order.
    cuts: list[tuple[float, float]] = []
    for cut_start, cut_end in logical_cuts:
        for item_start, _, item_duration in items:
            start = max(cut_start, item_start)
            end = min(cut_end, item_start + item_duration)
            if end - start >= frame_duration - 1e-6:
                cuts.append((start, end))

    retained_durations: list[float] = []
    for item_start, _, item_duration in items:
        item_end = item_start + item_duration
        cursor = item_start
        for cut_start, cut_end in cuts:
            if cut_end <= cursor + 1e-9:
                continue
            if cut_start >= item_end - 1e-9:
                break
            if cut_start > cursor + 1e-9:
                retained_durations.append(min(cut_start, item_end) - cursor)
            cursor = max(cursor, cut_end)
            if cursor >= item_end - 1e-9:
                break
        if cursor < item_end - 1e-9:
            retained_durations.append(item_end - cursor)

    violations = [
        {"cut": cut, "voice": voice}
        for cut in cuts
        for voice in protected
        if overlaps(cut, voice)
    ]
    result = {
        "sequence": args.sequence,
        "timelineDuration": timeline_duration,
        "itemCount": len(items),
        "frameDuration": frame_duration,
        "voicePadding": 0.0,
        "protectedVoice": [{"start": a, "end": b} for a, b in protected],
        "logicalCutCount": len(logical_cuts),
        "cuts": [{"start": a, "end": b, "duration": b - a} for a, b in cuts],
        "cutCount": len(cuts),
        "totalRemoval": sum(b - a for a, b in cuts),
        "expectedDuration": timeline_duration - sum(b - a for a, b in cuts),
        "expectedItemCount": len(retained_durations),
        "retainedDurations": retained_durations,
        "verification": {
            "passed": not violations,
            "violations": violations,
            "rules": [
                "current clip source ranges are read from native FCPXML",
                "no padding is added to clear voice",
                "cuts are rounded inward to the current frame grid",
                "no cut overlaps mapped clear voice",
            ],
        },
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    if args.quiet:
        print(json.dumps({
            "sequence": result["sequence"],
            "timelineDuration": result["timelineDuration"],
            "itemCount": result["itemCount"],
            "logicalCutCount": result["logicalCutCount"],
            "cutCount": result["cutCount"],
            "totalRemoval": result["totalRemoval"],
            "expectedDuration": result["expectedDuration"],
            "expectedItemCount": result["expectedItemCount"],
            "verification": result["verification"],
        }))
    else:
        print(rendered)


if __name__ == "__main__":
    main()
