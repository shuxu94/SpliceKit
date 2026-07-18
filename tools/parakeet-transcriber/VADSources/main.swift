import AVFoundation
import CoreML
import Foundation
import FluidAudio

enum VoiceActivityDetectorError: Error, LocalizedError {
    case usage(String)
    case noAudioTrack
    case reader(String)

    var errorDescription: String? {
        switch self {
        case .usage(let message): return message
        case .noAudioTrack: return "No audio track found"
        case .reader(let message): return message
        }
    }
}

struct Options {
    var path = ""
    var threshold: Float = 0.25
    var negativeThreshold: Float = 0.10
    var minSpeechMS = 50.0
    var minSilenceMS = 300.0
    var maxSpeechSeconds = 1000.0
    var paddingMS = 0.0
}

func parseOptions() throws -> Options {
    var arguments = Array(CommandLine.arguments.dropFirst())
    if arguments.first == "vad-analyze" { arguments.removeFirst() }
    guard let path = arguments.first, !path.hasPrefix("--") else {
        throw VoiceActivityDetectorError.usage(
            "Usage: voice-activity-detector [vad-analyze] <audio-file> [--threshold N] "
            + "[--neg-threshold N] [--min-speech-ms N] [--min-silence-ms N] "
            + "[--max-speech-s N] [--pad-ms N]"
        )
    }

    var options = Options()
    options.path = path
    var index = 1
    while index < arguments.count {
        let flag = arguments[index]
        guard index + 1 < arguments.count else {
            throw VoiceActivityDetectorError.usage("Missing value for \(flag)")
        }
        let value = arguments[index + 1]
        switch flag {
        case "--threshold":
            guard let parsed = Float(value) else { throw VoiceActivityDetectorError.usage("Invalid threshold") }
            options.threshold = parsed
        case "--neg-threshold":
            guard let parsed = Float(value) else { throw VoiceActivityDetectorError.usage("Invalid negative threshold") }
            options.negativeThreshold = parsed
        case "--min-speech-ms":
            guard let parsed = Double(value) else { throw VoiceActivityDetectorError.usage("Invalid minimum speech") }
            options.minSpeechMS = parsed
        case "--min-silence-ms":
            guard let parsed = Double(value) else { throw VoiceActivityDetectorError.usage("Invalid minimum silence") }
            options.minSilenceMS = parsed
        case "--max-speech-s":
            guard let parsed = Double(value) else { throw VoiceActivityDetectorError.usage("Invalid maximum speech") }
            options.maxSpeechSeconds = parsed
        case "--pad-ms":
            guard let parsed = Double(value) else { throw VoiceActivityDetectorError.usage("Invalid padding") }
            options.paddingMS = parsed
        case "--compute-units":
            // Accepted for compatibility with the ground-truth planner. The
            // production helper always uses CPU + Neural Engine.
            break
        default:
            throw VoiceActivityDetectorError.usage("Unknown option: \(flag)")
        }
        index += 2
    }
    return options
}

func decodeContainerAudioTo16kMono(path: String) async throws -> [Float] {
    let asset = AVURLAsset(url: URL(fileURLWithPath: path))
    let tracks = try await asset.loadTracks(withMediaType: .audio)
    guard !tracks.isEmpty else { throw VoiceActivityDetectorError.noAudioTrack }

    let reader = try AVAssetReader(asset: asset)
    let settings: [String: Any] = [
        AVFormatIDKey: kAudioFormatLinearPCM,
        AVLinearPCMBitDepthKey: 32,
        AVLinearPCMIsFloatKey: true,
        AVLinearPCMIsBigEndianKey: false,
        AVLinearPCMIsNonInterleaved: false,
        AVSampleRateKey: 16000,
        AVNumberOfChannelsKey: 1,
    ]
    let output = AVAssetReaderAudioMixOutput(audioTracks: tracks, audioSettings: settings)
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else {
        throw VoiceActivityDetectorError.reader("Cannot add audio reader output")
    }
    reader.add(output)
    guard reader.startReading() else {
        throw VoiceActivityDetectorError.reader(reader.error?.localizedDescription ?? "Audio reader failed")
    }

    var samples: [Float] = []
    while let sampleBuffer = output.copyNextSampleBuffer() {
        defer { CMSampleBufferInvalidate(sampleBuffer) }
        guard let block = CMSampleBufferGetDataBuffer(sampleBuffer) else { continue }
        let length = CMBlockBufferGetDataLength(block)
        guard length > 0 else { continue }
        var pointer: UnsafeMutablePointer<Int8>?
        var lengthAtOffset = 0
        var totalLength = 0
        let status = CMBlockBufferGetDataPointer(
            block,
            atOffset: 0,
            lengthAtOffsetOut: &lengthAtOffset,
            totalLengthOut: &totalLength,
            dataPointerOut: &pointer
        )
        guard status == kCMBlockBufferNoErr, let pointer else { continue }
        let count = length / MemoryLayout<Float>.size
        pointer.withMemoryRebound(to: Float.self, capacity: count) { floatPointer in
            samples.append(contentsOf: UnsafeBufferPointer(start: floatPointer, count: count))
        }
    }
    if reader.status == .failed {
        throw VoiceActivityDetectorError.reader(reader.error?.localizedDescription ?? "Audio decode failed")
    }
    return samples
}

func decodeAudioTo16kMono(path: String) async throws -> [Float] {
    // FluidAudio's converter is sample-exact for WAV/AIFF dialogue sources.
    // AVAssetReader is the fallback for video containers such as MOV, which
    // AVAudioFile cannot open directly.
    do {
        return try AudioConverter().resampleAudioFile(path: path)
    } catch {
        return try await decodeContainerAudioTo16kMono(path: path)
    }
}

@main
struct VoiceActivityDetector {
    static func main() async {
        do {
            let options = try parseOptions()
            let samples = try await decodeAudioTo16kMono(path: options.path)
            let manager = try await VadManager(config: VadConfig(
                defaultThreshold: options.threshold,
                debugMode: false,
                computeUnits: .cpuAndNeuralEngine
            ))
            let results = try await manager.process(samples)
            let config = VadSegmentationConfig(
                minSpeechDuration: max(0, options.minSpeechMS) / 1000.0,
                minSilenceDuration: max(0, options.minSilenceMS) / 1000.0,
                maxSpeechDuration: max(0.001, options.maxSpeechSeconds),
                speechPadding: max(0, options.paddingMS) / 1000.0,
                negativeThreshold: options.negativeThreshold
            )
            let segments = await manager.segmentSpeech(
                from: results,
                totalSamples: samples.count,
                config: config
            )
            let seconds = Double(samples.count) / Double(VadManager.sampleRate)
            print("VAD_SUMMARY count=\(segments.count) audio_seconds=\(String(format: "%.6f", seconds))")
            for (index, segment) in segments.enumerated() {
                print("VAD_SEGMENT index=\(index + 1) start=\(String(format: "%.6f", segment.startTime)) "
                    + "end=\(String(format: "%.6f", segment.endTime))")
            }
        } catch {
            fputs("ERROR: \(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }
}
