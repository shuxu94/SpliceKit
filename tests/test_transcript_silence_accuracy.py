#!/usr/bin/env python3
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL = REPO_ROOT / "Sources" / "SpliceKitTranscriptPanel.m"
HEADER = REPO_ROOT / "Sources" / "SpliceKitTranscriptPanel.h"
SERVER = REPO_ROOT / "Sources" / "SpliceKitServer.m"


def source(path):
    return path.read_text(encoding="utf-8")


def method_body(text, signature):
    start = text.index(signature)
    brace = text.index("{", start)
    depth = 0
    for index in range(brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1:index]
    raise AssertionError(f"Unterminated method: {signature}")


class TranscriptSilenceAccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = source(PANEL)
        cls.header = source(HEADER)
        cls.server = source(SERVER)
        cls.batch_delete = method_body(
            cls.panel,
            "- (NSDictionary *)deleteSilencesLongerThan:(double)minDuration",
        )

    def test_audio_first_mode_uses_one_frame_minimum_and_safe_voice_edges(self):
        self.assertIn("minimum cut: one frame", self.header)
        detector = method_body(
            self.panel,
            "- (void)detectAudioSilencesWithCompletion:(void (^)(NSDictionary *result))completion",
        )
        self.assertIn("self.silenceThreshold = 1.0 / MAX(self.frameRate, 1.0)", detector)
        self.assertIn("MAX(kSpliceKitMinimumVoiceEdgeProtectionSeconds", detector)
        self.assertIn(
            "kSpliceKitMinimumVoiceEdgeProtectionFrames * exactFrameDuration",
            detector,
        )
        self.assertIn("self.audioVoicePadding = voicePadding", detector)
        self.assertNotIn("self.audioVoicePadding = 0.0", detector)
        # The protection is already part of the detected voice mask. The delete
        # path must apply those exact confirmed ranges without adding another,
        # unrelated silence-side buffer.
        self.assertIn(
            "deleteTimelineRange:silence.startTime end:silence.endTime",
            self.batch_delete,
        )
        self.assertNotIn("boundaryPadding", self.header)

    def test_ui_and_rpc_report_voice_edge_protection(self):
        preset = method_body(self.panel, "- (void)silencePresetChanged:(NSPopUpButton *)sender")
        self.assertIn("self.audioVoicePadding = 0.150", preset)
        self.assertIn("kSpliceKitMinimumVoiceEdgeProtectionSeconds", preset)
        self.assertIn("voice-edge protection", preset)

        remove_clicked = method_body(
            self.panel,
            "- (void)deleteSilencesClicked:(id)sender {",
        )
        self.assertIn("three video frames of edge protection", remove_clicked)

        state = method_body(self.panel, "- (NSDictionary *)getState {")
        self.assertIn('@"voiceEdgeProtection"', state)
        self.assertIn('@"voiceEdgeProtectionFrames"', state)

    def test_descending_deletes_keep_original_earlier_coordinates(self):
        self.assertIn("a.startTime > b.startTime", self.batch_delete)
        self.assertIn(
            "deleteTimelineRange:silence.startTime end:silence.endTime",
            self.batch_delete,
        )
        self.assertNotIn("silence.startTime - totalTimeRemoved", self.batch_delete)
        self.assertNotIn("silence.endTime - totalTimeRemoved", self.batch_delete)

    def test_one_frame_delete_selects_exact_bladed_object(self):
        delete_range = method_body(
            self.panel,
            "- (NSDictionary *)deleteTimelineRange:(double)deleteStart end:(double)deleteEnd",
        )
        self.assertIn("effectiveRangeOfObject:", delete_range)
        self.assertIn("setSelectedItems:", delete_range)
        self.assertIn("exactItem", delete_range)
        self.assertNotIn("selectClipAtPlayhead:", delete_range)
        self.assertIn("indexOfObjectIdenticalTo:exactItem", delete_range)

    def test_timeline_delete_verifies_actual_duration_change(self):
        delete_range = method_body(
            self.panel,
            "- (NSDictionary *)deleteTimelineRange:(double)deleteStart end:(double)deleteEnd",
        )
        self.assertIn("durationBefore", delete_range)
        self.assertIn("durationAfter", delete_range)
        self.assertIn("actualRemoved", delete_range)
        self.assertIn("durationMatches", delete_range)
        self.assertIn("targetRemoved", delete_range)
        self.assertIn('@"verified": @YES', delete_range)
        self.assertIn("Timeline deletion did not remove the requested duration", delete_range)

    def test_timeline_duration_uses_spine_geometry_for_synchronized_clips(self):
        duration = method_body(
            self.panel,
            "- (double)durationSecondsForSequence:(id)sequence {",
        )
        self.assertIn("primaryObject", duration)
        self.assertIn("containedItems", duration)
        self.assertIn("effectiveRangeOfObject:", duration)
        self.assertIn("timelineEnd = MAX(timelineEnd, start + itemDuration)", duration)
        self.assertNotIn("@selector(duration)", duration)

    def test_reported_removed_time_uses_verified_timeline_duration(self):
        self.assertIn('totalTimeRemoved += [result[@"duration"] doubleValue]', self.batch_delete)
        self.assertNotIn("totalTimeRemoved += silence.duration", self.batch_delete)

    def test_batch_delete_verifies_audio_without_retranscription(self):
        self.assertIn("[self scheduleAudioSilenceRedetection]", self.batch_delete)
        self.assertIn('response[@"audioVerificationScheduled"] = @YES', self.batch_delete)
        self.assertNotIn("[self scheduleRetranscribe]", self.batch_delete)
        self.assertNotIn("[self resyncTimestampsFromTimeline]", self.batch_delete)
        refresh = method_body(self.panel, "- (void)scheduleAudioSilenceRedetection {")
        self.assertIn("[self detectAudioSilencesWithCompletion:nil]", refresh)
        self.assertNotIn("transcribeTimeline", refresh)

    def test_bulk_delete_never_uses_transcript_only_pauses(self):
        self.assertGreaterEqual(self.panel.count("silence.inferred = YES"), 3)
        self.assertIn("!silence.audioDetected", self.batch_delete)
        self.assertNotIn("includeInferred", self.header)
        self.assertNotIn("includeInferred", self.batch_delete)

    def test_destructive_path_requires_ffmpeg_confirmation(self):
        self.assertIn("!silence.audioDetected", self.batch_delete)
        self.assertIn("!silence.selectedForRemoval", self.batch_delete)
        self.assertIn("silence.confidence < 0.9", self.batch_delete)

    def test_audio_detector_requires_vad_and_ffmpeg_energy_agreement(self):
        detector = method_body(
            self.panel,
            "- (void)detectAudioSilencesWithCompletion:(void (^)(NSDictionary *result))completion",
        )
        self.assertIn("ffmpegSilencesForPath", detector)
        self.assertIn("vadSpeechForPath", detector)
        self.assertIn("SpliceKitTranscript_intersectTimeRanges(speech, energyActive)", detector)
        self.assertIn("self.audioVoiceMinimumEvidence", detector)
        self.assertIn("silence.selectedForRemoval = YES", detector)

    def test_synced_clip_offset_combines_parent_trim_and_sync_offset(self):
        offset = method_body(self.panel, "- (double)fileRelativeStartForClipInfo:(NSDictionary *)clipInfo")
        self.assertIn('clipInfo[@"mediaObject"]', offset)
        self.assertIn('clipInfo[@"timelineObject"]', offset)
        self.assertIn("collectionRelative + mediaRelative", offset)
        self.assertIn("timelineObject != mediaObject", offset)

    def test_vad_alone_cannot_protect_room_tone(self):
        detector = method_body(
            self.panel,
            "- (void)detectAudioSilencesWithCompletion:(void (^)(NSDictionary *result))completion",
        )
        self.assertIn("NSArray *energyActive", detector)
        self.assertIn("NSArray *agreement", detector)
        self.assertNotIn("mappedClearVoice addObjectsFromArray:speech", detector)

    def test_detector_prefers_fcpxml_audio_plan(self):
        detector = method_body(
            self.panel,
            "- (void)detectAudioSilencesWithCompletion:(void (^)(NSDictionary *result))completion",
        )
        self.assertIn("audioClipInfosFromFCPXMLWithError", detector)
        plan = method_body(
            self.panel,
            "- (NSArray<NSDictionary *> *)audioClipInfosFromFCPXMLWithError:(NSString **)errorOut",
        )
        self.assertIn("SpliceKit_handleFCPXMLExport", plan)
        self.assertIn('if ([enabled isEqualToString:@"0"]) continue', plan)
        self.assertIn("for (NSXMLElement *reference in references)", plan)
        self.assertIn("sourcePlacementStart = topOffset + nestedOffset - topReferenceStart", plan)
        self.assertIn("overlapStart = MAX(syncStart, sourcePlacementStart)", plan)
        self.assertIn("mappedTimelineStart = timelineStart + overlapStart - syncStart", plan)
        self.assertIn("fileStart = referenceStart - assetOrigin + overlapStart - sourcePlacementStart", plan)
        self.assertNotIn("syncStart + nestedOffset - topOrigin - assetOrigin", plan)
        self.assertIn('@"sourceDuration": @(sourceDuration)', plan)
        self.assertIn('@"sequenceDuration": @(sequenceDuration)', plan)

    def test_unsupported_audio_structures_fail_closed(self):
        plan = method_body(
            self.panel,
            "- (NSArray<NSDictionary *> *)audioClipInfosFromFCPXMLWithError:(NSString **)errorOut",
        )
        for marker in (
            "timeMap", "mute", "filter-audio", "adjust-volume",
            "audio-channel-source", "audioStart", "audioDuration",
        ):
            self.assertIn(marker, plan)
        self.assertIn('![elementName isEqualToString:@"sync-clip"]', plan)
        self.assertIn('![elementName isEqualToString:@"asset-clip"]', plan)
        self.assertIn('@"Silence removal does not yet support %@ timeline items"', plan)
        self.assertIn('nodesForXPath:@".//*[@lane]"', plan)
        self.assertIn("reference.parent != topReference", plan)
        self.assertIn("has no enabled, readable audio source", plan)

        detector = method_body(
            self.panel,
            "- (void)detectAudioSilencesWithCompletion:(void (^)(NSDictionary *result))completion",
        )
        self.assertIn("clipInfos.count > 0 || collectError.length > 0", detector)

    def test_candidates_split_at_primary_item_boundaries(self):
        detector = method_body(
            self.panel,
            "- (void)detectAudioSilencesWithCompletion:(void (^)(NSDictionary *result))completion",
        )
        self.assertIn('clipInfo[@"timelineItemStart"]', detector)
        self.assertIn('clipInfo[@"timelineItemDuration"]', detector)
        self.assertIn("itemBoundaries", detector)
        self.assertIn("rawCandidates addObject", detector)

    def test_audio_mask_is_bound_to_sequence_and_duration(self):
        detector = method_body(
            self.panel,
            "- (void)detectAudioSilencesWithCompletion:(void (^)(NSDictionary *result))completion",
        )
        self.assertIn("activeSequence != detectionSequence", detector)
        self.assertIn("self.audioSilenceDetectionSequence = detectionSequence", detector)
        self.assertIn("self.audioSilenceDetectionTimelineFingerprint = detectionFingerprint", detector)
        self.assertIn("Timeline changed during audio analysis", detector)

        self.assertIn("[self audioSilencePlanValidationError]", self.batch_delete)
        validation = method_body(
            self.panel,
            "- (NSString *)audioSilencePlanValidationError {",
        )
        self.assertIn("!activeSequence", validation)
        self.assertIn("SpliceKitTranscript_audioPlanFingerprint(currentPlan)", validation)
        self.assertIn("self.audioSilenceDetectionTimelineFingerprint", validation)
        self.assertIn("fingerprintMatches", validation)
        self.assertNotIn("audioSilenceDetectionSequenceDuration", validation)
        self.assertIn("Audio plan validation refused", validation)
        self.assertIn("Timeline changed after audio detection", validation)

        plan = method_body(
            self.panel,
            "- (NSArray<NSDictionary *> *)audioClipInfosFromFCPXMLWithError:(NSString **)errorOut",
        )
        self.assertIn('@"sequenceUID": sequenceUID', plan)
        self.assertIn('@"sequenceName": sequenceName', plan)

    def test_persisted_audio_mask_is_not_destructive_after_restore(self):
        restore = method_body(
            self.panel,
            "- (void)restorePersistedStateForCurrentSequenceIfNeeded",
        )
        self.assertIn("silence.audioDetected = NO", restore)
        self.assertIn("silence.selectedForRemoval = NO", restore)
        self.assertIn("silence.confidence = 0.0", restore)
        self.assertIn("self.audioSilenceDetectionTimelineFingerprint = nil", restore)

    def test_selected_pause_delete_uses_confirmed_original_coordinates(self):
        selected_delete = method_body(
            self.panel,
            "- (void)handleDeleteKeyInTextView {",
        )
        self.assertIn("!silence.audioDetected", selected_delete)
        self.assertIn("[self audioSilencePlanValidationError]", selected_delete)
        self.assertIn(
            "[self deleteTimelineRange:silence.startTime end:silence.endTime]",
            selected_delete,
        )
        self.assertNotIn("silence.startTime - totalRemoved", selected_delete)
        self.assertIn("[self scheduleAudioSilenceRedetection]", selected_delete)

    def test_remove_button_surfaces_validation_errors(self):
        remove_clicked = method_body(
            self.panel,
            "- (void)deleteSilencesClicked:(id)sender {",
        )
        self.assertIn("NSDictionary *result", remove_clicked)
        self.assertIn('result[@"error"]', remove_clicked)
        self.assertIn("[self updateStatusUI:message]", remove_clicked)
        self.assertIn("Remove Non-Voice refused", remove_clicked)

    def test_fcpxml_excludes_explicitly_inactive_storyline_audio(self):
        inactive_check = method_body(
            self.panel,
            "static BOOL SpliceKitTranscript_syncSourceAudioIsExplicitlyInactive",
        )
        self.assertIn("sourceID isEqualToString:requestedSourceID", inactive_check)
        self.assertIn('active isEqualToString:@"0"', inactive_check)
        self.assertIn('enabled isEqualToString:@"0"', inactive_check)
        self.assertIn("foundRoleState && !foundEnabledActiveRole", inactive_check)

        plan = method_body(
            self.panel,
            "- (NSArray<NSDictionary *> *)audioClipInfosFromFCPXMLWithError:(NSString **)errorOut",
        )
        self.assertIn('SpliceKitTranscript_syncSourceAudioIsExplicitlyInactive(item, @"storyline")', plan)
        self.assertIn('SpliceKitTranscript_syncSourceAudioIsExplicitlyInactive(item, @"connected")', plan)
        self.assertIn("reference == topReference && storylineAudioInactive", plan)
        self.assertIn("reference != topReference && connectedAudioInactive", plan)
        self.assertIn("skippedInactiveAudioRanges++", plan)

    def test_synced_external_source_coverage_uses_fcpxml_parent_time_domain(self):
        # Regression fixture from a synchronized clip whose camera timecode is
        # 25633.2409667s and whose external WAV is placed at 25633.611925s.
        # The WAV therefore starts 0.3709583s into the synchronized clip.
        top_offset = 0.0
        top_reference_start = 768997229 / 30000
        nested_offset = 1025344477 / 40000
        source_placement_start = top_offset + nested_offset - top_reference_start
        self.assertAlmostEqual(source_placement_start, 0.370958333333, places=9)

        # An untrimmed clip must shift source coverage on the timeline instead
        # of clamping a negative source time and pretending WAV time zero is at
        # timeline zero.
        sync_start = 0.0
        overlap_start = max(sync_start, source_placement_start)
        mapped_timeline_start = overlap_start - sync_start
        file_start = overlap_start - source_placement_start
        self.assertAlmostEqual(mapped_timeline_start, 0.370958333333, places=9)
        self.assertAlmostEqual(file_start, 0.0, places=9)

        # Once the synchronized clip is trimmed past the WAV placement, its
        # timeline position stays fixed and the source advances normally.
        sync_start = 1.5015
        overlap_start = max(sync_start, source_placement_start)
        mapped_timeline_start = overlap_start - sync_start
        file_start = overlap_start - source_placement_start
        self.assertAlmostEqual(mapped_timeline_start, 0.0, places=9)
        self.assertAlmostEqual(file_start, 1.130541666667, places=9)

    def test_ffmpeg_audio_asset_seek_returns_clip_relative_timestamps(self):
        runner = method_body(
            self.panel,
            "- (NSArray<NSDictionary *> *)ffmpegSilencesForPath:(NSString *)path",
        )
        self.assertLess(runner.index('@"-ss"'), runner.index('@"-i", path'))

    def test_rpc_uses_single_confirmed_range_deletion_path(self):
        handler = method_body(
            self.server,
            "static NSDictionary *SpliceKit_handleTranscriptDeleteSilences",
        )
        self.assertIn('params[@"minDuration"]', handler)
        self.assertIn("deleteSilencesLongerThan:minDuration", handler)
        self.assertNotIn("boundaryPadding", handler)
        self.assertNotIn("includeInferred", handler)

    def test_removed_experimental_settings_do_not_reappear(self):
        for setting in (
            "audioSilenceConfirmDB",
            "audioSilenceMergeGap",
            "audioSilenceRoomTone",
            "kSpliceKitDefaultSilenceBoundaryPadding",
            "kSpliceKitDefaultSilenceNoiseDB",
            "kSpliceKitDefaultSilenceConfirmDB",
        ):
            self.assertNotIn(setting, self.panel)


if __name__ == "__main__":
    unittest.main()
