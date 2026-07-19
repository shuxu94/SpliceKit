"""Static regression tests for the social-caption Apple Speech integration."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PANEL = (ROOT / "Sources/SpliceKitCaptionPanel.m").read_text()
HEADER = (ROOT / "Sources/SpliceKitCaptionPanel.h").read_text()
SERVER = (ROOT / "Sources/SpliceKitServer.m").read_text()
MCP = (ROOT / "mcp/server.py").read_text()
DIAGNOSTICS = (ROOT / "Sources/SpliceKitTranscriptDiagnostics.m").read_text()


class CaptionAppleSpeechTests(unittest.TestCase):
    def test_engine_is_available_and_persisted(self):
        self.assertIn('@"Apple Speech (On-device)"', PANEL)
        self.assertIn('representedObject = @"appleSpeech"', PANEL)
        self.assertIn('setObject:engineID forKey:@"SpliceKitCaptionEngine"', PANEL)
        self.assertIn('- (BOOL)setTranscriptionEngine:(NSString *)engineID;', HEADER)

    def test_dispatch_does_not_require_helper_binary(self):
        dispatch = PANEL.index('if ([engineID isEqualToString:@"appleSpeech"])')
        binary_check = PANEL.index('if (!binaryPath)', dispatch)
        self.assertLess(dispatch, binary_check)
        self.assertIn('[self performAppleSpeechCaptionTranscription]', PANEL[dispatch:binary_check])

    def test_timestamps_are_mapped_back_to_timeline_and_source_space(self):
        self.assertIn('appleSpeechAudioRangeForURL:', PANEL)
        self.assertIn('@"startTime": @(rangeStart + timestamp)', PANEL)
        self.assertIn('double relativeStart = sourceTimestamp - fileRelativeTrimStart;', PANEL)
        self.assertIn('word.startTime = timelineStart + wordTimeOffset + relativeStart;', PANEL)
        self.assertIn('word.sourceMediaOffset = trimStart;', PANEL)
        self.assertIn('word.sourceMediaTime = sourceTimestamp + mediaOrigin;', PANEL)
        self.assertIn('[self transcriptionFinishedWithWords:allWords];', PANEL)

    def test_bridge_and_mcp_expose_caption_engine(self):
        self.assertIn('@"captions.setEngine"', SERVER)
        self.assertIn('SpliceKit_handleCaptionsSetEngine', SERVER)
        self.assertIn('def set_caption_engine(engine: str)', MCP)
        self.assertIn('bridge.call("captions.setEngine", engine=engine)', MCP)
        self.assertIn('engine: str = ""', MCP)

    def test_diagnostics_never_use_invalid_url_request_initializer(self):
        self.assertNotIn('[[requestClass alloc] init]', DIAGNOSTICS)
        self.assertIn('instancesRespondToSelector:', DIAGNOSTICS)

    def test_fresh_caption_generation_cancels_restore_retries(self):
        self.assertNotIn('self.lastHealedSequenceKey = nil;', PANEL)
        self.assertIn('self.lastHealedSequenceKey = sequenceKey;', PANEL)
        self.assertIn('self.automaticRestoreGeneration += 1;', PANEL)


if __name__ == "__main__":
    unittest.main()
