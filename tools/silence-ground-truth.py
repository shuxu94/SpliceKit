#!/usr/bin/env python3
"""Build a verified silence-removal plan from every active audio source.

The planner deliberately separates two questions:
1. FFmpeg: is the audio energy actually below the silence threshold?
2. Silero VAD: is there any speech in either active source?

A range is removable only when every active source reports silence and no VAD
speech segment overlaps it. Source offsets are timeline placements in seconds.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


SILENCE_START_RE = re.compile(r"silence_start: ([0-9.]+)")
SILENCE_END_RE = re.compile(r"silence_end: ([0-9.]+)")
VAD_SEGMENT_RE = re.compile(r"VAD_SEGMENT .* start=([0-9.]+) end=([0-9.]+)")
VAD_SUMMARY_RE = re.compile(r"VAD_SUMMARY .* audio_seconds=([0-9.]+)")


@dataclass(frozen=True)
class Source:
    path: str
    offset: float
    duration: float


def run(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return completed.stdout + "\n" + completed.stderr


def merge(ranges: list[tuple[float, float]], gap: float = 0.0) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if result and start <= result[-1][1] + gap:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def intersect(left: list[tuple[float, float]], right: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            result.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def subtract(base: list[tuple[float, float]], veto: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for base_start, base_end in base:
        cursor = base_start
        for veto_start, veto_end in veto:
            if veto_end <= cursor:
                continue
            if veto_start >= base_end:
                break
            if veto_start > cursor:
                result.append((cursor, min(veto_start, base_end)))
            cursor = max(cursor, veto_end)
            if cursor >= base_end:
                break
        if cursor < base_end:
            result.append((cursor, base_end))
    return result


def complement(ranges: list[tuple[float, float]], start: float, end: float) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    cursor = start
    for range_start, range_end in merge(ranges):
        range_start = max(start, range_start)
        range_end = min(end, range_end)
        if range_end <= range_start:
            continue
        if range_start > cursor:
            result.append((cursor, range_start))
        cursor = max(cursor, range_end)
    if cursor < end:
        result.append((cursor, end))
    return result


def pad_ranges(
    ranges: list[tuple[float, float]], padding: float, start: float, end: float
) -> list[tuple[float, float]]:
    return merge([
        (max(start, range_start - padding), min(end, range_end + padding))
        for range_start, range_end in ranges
    ])


def ffmpeg_silences(ffmpeg: str, source: Source, threshold_db: float, minimum: float) -> list[tuple[float, float]]:
    output = run([
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-i",
        source.path,
        "-vn",
        "-af",
        f"silencedetect=noise={threshold_db:.1f}dB:d={minimum:.3f}",
        "-f",
        "null",
        "-",
    ])
    ranges: list[tuple[float, float]] = []
    pending: float | None = None
    for line in output.splitlines():
        start_match = SILENCE_START_RE.search(line)
        if start_match:
            pending = float(start_match.group(1))
        end_match = SILENCE_END_RE.search(line)
        if end_match and pending is not None:
            end = min(float(end_match.group(1)), source.duration)
            if end > pending:
                ranges.append((pending + source.offset, end + source.offset))
            pending = None
    if pending is not None and source.duration > pending:
        ranges.append((pending + source.offset, source.duration + source.offset))
    return merge(ranges, gap=0.010)


def vad_speech(vad: str, source: Source, threshold: float, padding_ms: int) -> list[tuple[float, float]]:
    output = run([
        vad,
        "vad-analyze",
        source.path,
        "--threshold",
        str(threshold),
        "--neg-threshold",
        "0.10",
        "--min-speech-ms",
        "50",
        "--min-silence-ms",
        "300",
        "--max-speech-s",
        "1000",
        "--pad-ms",
        str(padding_ms),
        "--compute-units",
        "all",
    ])
    summary = VAD_SUMMARY_RE.search(output)
    if not summary:
        raise RuntimeError(f"VAD did not return a summary for {source.path}")
    ranges = [
        (float(match.group(1)) + source.offset, float(match.group(2)) + source.offset)
        for match in VAD_SEGMENT_RE.finditer(output)
    ]
    return merge(ranges)


def source_silence_coverage(
    silences: list[tuple[float, float]], source: Source, timeline_duration: float
) -> list[tuple[float, float]]:
    coverage = list(silences)
    if source.offset > 0:
        coverage.append((0.0, min(source.offset, timeline_duration)))
    source_end = source.offset + source.duration
    if source_end < timeline_duration:
        coverage.append((max(0.0, source_end), timeline_duration))
    return merge(coverage)


def quantize_inward(
    ranges: list[tuple[float, float]],
    frame_duration: float,
    edge_guard: float,
    minimum: float,
    timeline_duration: float,
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for raw_start, raw_end in ranges:
        start_guard = 0.0 if raw_start <= 1e-6 else edge_guard
        end_guard = 0.0 if raw_end >= timeline_duration - 1e-6 else edge_guard
        guarded_start = raw_start + start_guard
        guarded_end = raw_end - end_guard
        start = math.ceil((guarded_start / frame_duration) - 1e-9) * frame_duration
        end = math.floor((guarded_end / frame_duration) + 1e-9) * frame_duration
        if end - start >= minimum:
            result.append((start, end))
    return result


def parse_source(raw: str) -> Source:
    data = json.loads(raw)
    return Source(path=data["path"], offset=float(data["offset"]), duration=float(data["duration"]))


def overlaps(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return min(left[1], right[1]) > max(left[0], right[0]) + 1e-9


def main() -> None:
    sources = [parse_source(raw) for raw in ARGS.source]
    all_silence: list[tuple[float, float]] | None = None
    source_reports = []
    all_speech: list[tuple[float, float]] = []
    all_clear_voice: list[tuple[float, float]] = []

    for source in sources:
        if not Path(source.path).is_file():
            raise FileNotFoundError(source.path)
        energy_silence = ffmpeg_silences(ARGS.ffmpeg, source, ARGS.silence_db, ARGS.ffmpeg_minimum)
        coverage = source_silence_coverage(energy_silence, source, ARGS.timeline_duration)
        all_silence = coverage if all_silence is None else intersect(all_silence, coverage)
        speech = vad_speech(ARGS.vad, source, ARGS.vad_threshold, ARGS.vad_padding_ms)
        all_speech.extend(speech)
        source_start = max(0.0, source.offset)
        source_end = min(ARGS.timeline_duration, source.offset + source.duration)
        energy_active = complement(energy_silence, source_start, source_end)
        # Silero alone is not a clear-voice classifier: room tone and handling
        # noise can produce long false-positive speech segments.  Voice mode
        # therefore requires temporal agreement between VAD and FFmpeg energy
        # on the same enabled source.  Padding is added only after agreement.
        clear_voice = [
            (start, end)
            for start, end in intersect(speech, energy_active)
            if end - start >= ARGS.voice_min_evidence
        ]
        all_clear_voice.extend(pad_ranges(
            clear_voice,
            ARGS.voice_padding,
            0.0,
            ARGS.timeline_duration,
        ))
        source_reports.append({
            "path": source.path,
            "offset": source.offset,
            "duration": source.duration,
            "energySilenceCount": len(energy_silence),
            "speechSegmentCount": len(speech),
            "clearVoiceEvidenceCount": len(clear_voice),
        })

    speech_union = merge(all_speech)
    clear_voice_union = merge(all_clear_voice)
    if ARGS.mode == "voice":
        safe = complement(clear_voice_union, 0.0, ARGS.timeline_duration)
    else:
        safe = subtract(all_silence or [], speech_union)
    cuts = quantize_inward(
        safe,
        ARGS.frame_duration,
        ARGS.edge_guard,
        ARGS.minimum_cut,
        ARGS.timeline_duration,
    )

    violations = []
    for cut in cuts:
        protected = clear_voice_union if ARGS.mode == "voice" else speech_union
        if any(overlaps(cut, speech) for speech in protected):
            violations.append({"cut": cut, "reason": "protected voice overlap"})
        if ARGS.mode == "silence" and not any(
            start <= cut[0] + 1e-9 and end >= cut[1] - 1e-9
            for start, end in (all_silence or [])
        ):
            violations.append({"cut": cut, "reason": "not silent in every active source"})

    result = {
        "timelineDuration": ARGS.timeline_duration,
        "mode": ARGS.mode,
        "frameDuration": ARGS.frame_duration,
        "silenceThresholdDB": ARGS.silence_db,
        "vadThreshold": ARGS.vad_threshold,
        "vadPaddingMS": ARGS.vad_padding_ms,
        "edgeGuard": ARGS.edge_guard,
        "minimumCut": ARGS.minimum_cut,
        "sources": source_reports,
        "speechUnion": [{"start": start, "end": end} for start, end in speech_union],
        "clearVoice": [{"start": start, "end": end} for start, end in clear_voice_union],
        "cuts": [
            {"start": start, "end": end, "duration": end - start}
            for start, end in cuts
        ],
        "cutCount": len(cuts),
        "totalRemoval": sum(end - start for start, end in cuts),
        "verification": {
            "passed": not violations,
            "violations": violations,
            "rules": [
                (
                    "clear voice requires Silero VAD and FFmpeg energy agreement on an enabled source"
                    if ARGS.mode == "voice"
                    else "every cut is below the FFmpeg threshold in every active source"
                ),
                "no cut overlaps the protected voice mask",
                "cut boundaries are rounded inward to the FCP frame grid",
            ],
        },
    }
    rendered = json.dumps(result, indent=2)
    if ARGS.output:
        Path(ARGS.output).write_text(rendered + "\n", encoding="utf-8")
    if ARGS.quiet:
        print(json.dumps({
            "mode": result["mode"],
            "cutCount": result["cutCount"],
            "totalRemoval": result["totalRemoval"],
            "verification": result["verification"],
        }))
    else:
        print(rendered)


PARSER = argparse.ArgumentParser()
PARSER.add_argument("--mode", choices=("silence", "voice"), default="silence")
PARSER.add_argument("--source", action="append", required=True, help='JSON: {"path":"...","offset":0,"duration":1}')
PARSER.add_argument("--timeline-duration", type=float, required=True)
PARSER.add_argument("--frame-duration", type=float, required=True)
PARSER.add_argument("--ffmpeg", default="/opt/homebrew/bin/ffmpeg")
PARSER.add_argument("--vad", required=True)
PARSER.add_argument("--silence-db", type=float, default=-35.0)
PARSER.add_argument("--ffmpeg-minimum", type=float, default=0.15)
PARSER.add_argument("--vad-threshold", type=float, default=0.25)
PARSER.add_argument("--vad-padding-ms", type=int, default=200)
PARSER.add_argument("--voice-padding", type=float, default=0.0)
PARSER.add_argument("--voice-min-evidence", type=float, default=0.050)
PARSER.add_argument("--edge-guard", type=float, default=0.080)
PARSER.add_argument("--minimum-cut", type=float, default=0.300)
PARSER.add_argument("--output", help="Optional path for the complete JSON plan")
PARSER.add_argument("--quiet", action="store_true", help="Print only a compact summary")
ARGS = PARSER.parse_args()


if __name__ == "__main__":
    main()
