import AVFoundation
import Foundation
import WhisperKit

// Usage:
//   whisper-transcriber <audio-file>   [--progress] [--model large-v3|large-v3-turbo]
//   whisper-transcriber --batch <json> [--progress] [--model large-v3|large-v3-turbo]
//
// Output contract matches parakeet-transcriber:
//   Single mode: JSON array of word dicts to stdout
//   Batch mode:  JSON array of {"file":path, "words":[...]} to stdout
//   Progress:    "PROGRESS:<fraction>:<message>" lines to stderr when --progress set
//
// Runs Whisper via CoreML on the Apple Neural Engine using WhisperKit. The
// CoreML encoder + decoder is downloaded from HuggingFace on first use into
// ~/Library/Application Support/SpliceKit/Models/whisper/.

let progressLock = NSLock()

func reportProgress(_ fraction: Double, _ message: String) {
    progressLock.lock()
    let line = "PROGRESS:\(fraction):\(message)\n"
    FileHandle.standardError.write(line.data(using: .utf8)!)
    progressLock.unlock()
}

func printError(_ message: String) {
    progressLock.lock()
    FileHandle.standardError.write("ERROR:\(message)\n".data(using: .utf8)!)
    progressLock.unlock()
}

// A transcription request. When `start`/`duration` are set (per-clip captioning),
// only that source range is decoded and word timestamps are relative to `start`
// (i.e. 0-based within the clip). When nil, the whole file is transcribed
// (transcript-panel / whole-file behavior).
struct BatchEntry {
    let file: String
    let start: Double?
    let duration: Double?
}

// Parse one manifest entry into a BatchEntry, reading optional start/duration.
func parseBatchEntry(_ entry: [String: Any]) -> BatchEntry? {
    guard let file = entry["file"] as? String else { return nil }
    let start = (entry["start"] as? NSNumber)?.doubleValue
    let duration = (entry["duration"] as? NSNumber)?.doubleValue
    return BatchEntry(file: file, start: start, duration: duration)
}

let args = CommandLine.arguments
guard args.count >= 2 else {
    printError("Usage: whisper-transcriber <audio-file> [--progress] [--model large-v3|large-v3-turbo]")
    printError("       whisper-transcriber --batch <manifest.json> [--progress] [--model large-v3|large-v3-turbo]")
    exit(1)
}

let showProgress = args.contains("--progress")
let batchMode = args.contains("--batch")
// Persistent server mode: load the model once, then service transcription
// requests (manifest paths, one per line) from stdin. This avoids the cold-start
// CoreML/ANE specialization that otherwise happens on every subprocess launch.
let serveMode = args.contains("--serve")

// Default to large-v3-turbo (much faster, nearly identical quality for captions).
// WhisperKit variant names match HuggingFace repo subdirs under argmaxinc/whisperkit-coreml.
// Turbo variant is "large-v3_turbo" (underscore), not "large-v3-turbo" (hyphen).
var modelVariant = "large-v3_turbo"
var prettyName = "Whisper large-v3-turbo"
var approxSizeMB = 950
if let idx = args.firstIndex(of: "--model"), idx + 1 < args.count {
    let choice = args[idx + 1].lowercased()
    switch choice {
    case "large-v3-turbo", "large-v3_turbo", "turbo", "v3-turbo":
        modelVariant = "large-v3_turbo"; prettyName = "Whisper large-v3-turbo"; approxSizeMB = 950
    case "large-v3", "v3":
        modelVariant = "large-v3"; prettyName = "Whisper large-v3"; approxSizeMB = 1550
    default:
        printError("Unknown model '\(choice)' — using large-v3_turbo")
    }
}

var batchEntries: [BatchEntry] = []
if batchMode {
    guard let batchIdx = args.firstIndex(of: "--batch"), batchIdx + 1 < args.count else {
        printError("--batch requires a manifest JSON file path")
        exit(1)
    }
    let manifestPath = args[batchIdx + 1]
    guard let data = FileManager.default.contents(atPath: manifestPath),
          let arr = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
        printError("Failed to read batch manifest: \(manifestPath)")
        exit(1)
    }
    for entry in arr {
        if let be = parseBatchEntry(entry) { batchEntries.append(be) }
    }
    if batchEntries.isEmpty {
        printError("No files in batch manifest"); exit(1)
    }
} else if serveMode {
    // No audio path on the command line; requests arrive on stdin after load.
} else {
    let audioPath = args[1]
    guard FileManager.default.fileExists(atPath: audioPath) else {
        printError("File not found: \(audioPath)"); exit(1)
    }
    batchEntries.append(BatchEntry(file: audioPath, start: nil, duration: nil))
}

// Models live under ~/Library/Application Support/SpliceKit/Models/whisper/
let modelsDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
    .appendingPathComponent("SpliceKit/Models/whisper", isDirectory: true)
try? FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)

// WhisperKit cache layout under downloadBase:
//   <downloadBase>/models/<modelRepo>/openai_whisper-<variant>/{MelSpectrogram,AudioEncoder,TextDecoder}.mlmodelc
let modelRepo = "argmaxinc/whisperkit-coreml"
let variantFolder = "openai_whisper-\(modelVariant)"
let fullModelPath = modelsDir
    .appendingPathComponent("models")
    .appendingPathComponent(modelRepo)
    .appendingPathComponent(variantFolder)
// A fully-downloaded .mlmodelc always contains weights/weight.bin. Checking just the directory
// misses interrupted downloads, producing a "Could not open weight.bin" load failure later.
let requiredComponents = ["MelSpectrogram.mlmodelc", "AudioEncoder.mlmodelc", "TextDecoder.mlmodelc"]
let isCached = requiredComponents.allSatisfy { component -> Bool in
    let mlmodelc = fullModelPath.appendingPathComponent(component)
    guard FileManager.default.fileExists(atPath: mlmodelc.path) else { return false }
    let weight = mlmodelc.appendingPathComponent("weights/weight.bin")
    return FileManager.default.fileExists(atPath: weight.path)
}

func floatValue(_ value: Any?) -> Float? {
    if let value = value as? Float { return value }
    if let value = value as? Double { return Float(value) }
    if let value = value as? NSNumber { return value.floatValue }
    if let value = value as? String { return Float(value) }
    return nil
}

func normalizedWordTimings(_ words: [[String: Any]], minimumDuration: Float = 1.0 / 30.0) -> [[String: Any]] {
    var normalized = words.sorted {
        (floatValue($0["startTime"]) ?? 0) < (floatValue($1["startTime"]) ?? 0)
    }

    var previousEnd: Float = 0
    for index in normalized.indices {
        var start = floatValue(normalized[index]["startTime"]) ?? previousEnd
        var end = floatValue(normalized[index]["endTime"]) ?? (start + minimumDuration)

        if !start.isFinite { start = previousEnd }
        if !end.isFinite { end = start + minimumDuration }
        if start < previousEnd { start = previousEnd }
        if end <= start { end = start + minimumDuration }

        normalized[index]["startTime"] = start
        normalized[index]["endTime"] = end
        previousEnd = end
    }

    return normalized
}

enum AudioDecodeError: Error, LocalizedError {
    case noAudioTrack
    case readerSetupFailed(String)
    case readFailed(String)
    var errorDescription: String? {
        switch self {
        case .noAudioTrack: return "No audio track found in file"
        case .readerSetupFailed(let m): return "Audio reader setup failed: \(m)"
        case .readFailed(let m): return "Audio decode failed: \(m)"
        }
    }
}

// Decode any AVFoundation-readable container (including video files like .mov/.mp4)
// to 16 kHz mono Float PCM. WhisperKit's built-in loader uses AVAudioFile(forReading:),
// which rejects video containers with kAudioFileUnsupportedFileTypeError ('typ?',
// 1954115647). AVAssetReader decodes them the same way Parakeet/FluidAudio does, so
// any clip that transcribes under Parakeet also works here.
func decodeAudioTo16kMono(path: String, start: Double? = nil, duration: Double? = nil) async throws -> [Float] {
    let asset = AVURLAsset(url: URL(fileURLWithPath: path))

    let audioTracks = try await asset.loadTracks(withMediaType: .audio)
    guard !audioTracks.isEmpty else { throw AudioDecodeError.noAudioTrack }

    let reader: AVAssetReader
    do {
        reader = try AVAssetReader(asset: asset)
    } catch {
        throw AudioDecodeError.readerSetupFailed(error.localizedDescription)
    }

    // Restrict decoding to [start, start+duration] for per-clip transcription.
    // The collected samples become a 0-based [Float] window, so model word
    // timestamps come out relative to the clip start.
    if let start = start, start >= 0, let duration = duration, duration > 0 {
        let t0 = CMTimeMakeWithSeconds(start, preferredTimescale: 600)
        let dur = CMTimeMakeWithSeconds(duration, preferredTimescale: 600)
        reader.timeRange = CMTimeRange(start: t0, duration: dur)
    }

    // Ask AVFoundation to resample/downmix to WhisperKit's required 16 kHz mono Float32.
    let outputSettings: [String: Any] = [
        AVFormatIDKey: kAudioFormatLinearPCM,
        AVLinearPCMBitDepthKey: 32,
        AVLinearPCMIsFloatKey: true,
        AVLinearPCMIsBigEndianKey: false,
        AVLinearPCMIsNonInterleaved: false,
        AVSampleRateKey: 16000,
        AVNumberOfChannelsKey: 1,
    ]
    let output = AVAssetReaderAudioMixOutput(audioTracks: audioTracks, audioSettings: outputSettings)
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else {
        throw AudioDecodeError.readerSetupFailed("cannot add audio mix output")
    }
    reader.add(output)

    guard reader.startReading() else {
        throw AudioDecodeError.readFailed(reader.error?.localizedDescription ?? "startReading returned false")
    }

    var samples: [Float] = []
    while let sampleBuffer = output.copyNextSampleBuffer() {
        defer { CMSampleBufferInvalidate(sampleBuffer) }
        guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { continue }
        let length = CMBlockBufferGetDataLength(blockBuffer)
        if length == 0 { continue }
        var dataPointer: UnsafeMutablePointer<Int8>? = nil
        var lengthAtOffset = 0
        var totalLength = 0
        let status = CMBlockBufferGetDataPointer(
            blockBuffer, atOffset: 0,
            lengthAtOffsetOut: &lengthAtOffset,
            totalLengthOut: &totalLength,
            dataPointerOut: &dataPointer)
        guard status == kCMBlockBufferNoErr, let dataPointer = dataPointer else { continue }
        let floatCount = length / MemoryLayout<Float>.size
        dataPointer.withMemoryRebound(to: Float.self, capacity: floatCount) { floatPtr in
            samples.append(contentsOf: UnsafeBufferPointer(start: floatPtr, count: floatCount))
        }
    }

    if reader.status == .failed {
        throw AudioDecodeError.readFailed(reader.error?.localizedDescription ?? "unknown reader failure")
    }

    return samples
}

func extractWords(from transcription: [TranscriptionResult]) -> [[String: Any]] {
    var words: [[String: Any]] = []
    for result in transcription {
        for segment in result.segments {
            if let wt = segment.words, !wt.isEmpty {
                for w in wt {
                    let trimmed = w.word.trimmingCharacters(in: .whitespacesAndNewlines)
                    if trimmed.isEmpty { continue }
                    words.append([
                        "word": trimmed,
                        "startTime": Float(w.start),
                        "endTime": Float(w.end),
                        "confidence": Float(w.probability),
                    ])
                }
            } else {
                // Fallback: emit the whole segment as one word if no per-word timing.
                let trimmed = segment.text.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    words.append([
                        "word": trimmed,
                        "startTime": Float(segment.start),
                        "endTime": Float(segment.end),
                        "confidence": Float(1.0),
                    ])
                }
            }
        }
    }
    return normalizedWordTimings(words)
}

// Shared decode options for every transcription path (serve, batch, single).
//
// chunkingStrategy: .none — Use .none rather than .vad: VAD was dropping long
// stretches of the source (e.g. first 83s of a 4-min Tim Keller meditation,
// ~70% fewer words than Parakeet). .none slides a 30s window across the whole
// file so nothing is skipped. For a 4-min file this costs seconds of extra
// inference.
//
// Whisper produced large gaps where audio wasn't captioned on real footage. Two
// independent WhisperKit mechanisms cause this; both are disabled here.
//
// 1) chunkingStrategy: .vad — With .none, WhisperKit runs one long-form seek loop
//    across the whole file. The decoder can emit a far-ahead end-timestamp and
//    `seek` jumps past audio, silently dropping it. Verified on this repo's
//    footage: a ~10s stretch of clear voiceover ("Everyone knows NVIDIA TSMC and
//    ASML, who makes the machines that make the chips") was emitted nothing in
//    the full-file run but transcribed fine when that region was clipped out and
//    fed standalone. .vad splits the audio into CONTIGUOUS chunks at silence
//    midpoints (AudioChunker.chunkAll drops no audio) and resets seek per chunk,
//    so the seek-skip can't accumulate. This lifted the file from 301 -> 430
//    words and filled the dropped voiceover.
//
// 2) noSpeechThreshold: nil — WhisperKit discards an entire window
//    (SegmentSeeker.findSeekPointAndSegments) when noSpeechProb > noSpeechThreshold
//    AND avgLogProb <= logProbThreshold. A music bed inflates noSpeechProb, so
//    real speech under music gets gated out. This was also the likely cause of an
//    earlier regression where VAD "dropped" 83s of a quiet meditation — quiet
//    chunks were gated, not lost by chunking. Setting it to nil keeps whatever the
//    decoder produced for every chunk. compressionRatioThreshold (default 2.4) +
//    temperature fallback still guard against runaway repetition/hallucination on
//    genuinely silent stretches; logProbThreshold (slightly loosened) is retained
//    for the temperature-fallback quality check only.
func makeDecodeOptions() -> DecodingOptions {
    DecodingOptions(
        verbose: false,
        task: .transcribe,
        language: nil,
        temperature: 0.0,
        wordTimestamps: true,
        logProbThreshold: -1.5,
        noSpeechThreshold: nil,
        chunkingStrategy: ChunkingStrategy.vad
    )
}

let semaphore = DispatchSemaphore(value: 0)
var exitCode: Int32 = 0

Task {
    do {
        if showProgress {
            if isCached {
                reportProgress(0.05, "Loading \(prettyName) model (cached)...")
            } else {
                reportProgress(0.03, "Downloading \(prettyName) CoreML model (~\(approxSizeMB) MB)... First run only.")
            }
        }

        let config = WhisperKitConfig(
            model: modelVariant,
            downloadBase: modelsDir,
            modelRepo: modelRepo,
            verbose: false,
            logLevel: .error,
            prewarm: false,
            load: false,
            download: false
        )

        let whisper: WhisperKit
        do {
            whisper = try await WhisperKit(config)
        } catch {
            printError("Failed to initialize WhisperKit: \(error.localizedDescription)")
            throw error
        }

        if !isCached {
            do {
                let downloadedURL = try await WhisperKit.download(
                    variant: modelVariant,
                    downloadBase: modelsDir,
                    useBackgroundSession: false,
                    from: modelRepo,
                    progressCallback: { progress in
                        if showProgress {
                            let frac = 0.05 + 0.55 * progress.fractionCompleted
                            reportProgress(frac, "Downloading \(prettyName)... \(Int(progress.fractionCompleted * 100))%")
                        }
                    }
                )
                whisper.modelFolder = downloadedURL
            } catch {
                let msg = error.localizedDescription
                if msg.contains("rate") || msg.contains("429") || msg.contains("503") {
                    printError("Model download rate-limited by HuggingFace. Wait a few minutes and try again.")
                } else if msg.contains("network") || msg.contains("connect") || msg.contains("NSURL") {
                    printError("Network error downloading model. Check internet connection: \(msg)")
                } else if msg.contains("space") || msg.contains("disk") {
                    printError("Not enough disk space for \(prettyName) (~\(approxSizeMB) MB).")
                } else {
                    printError("Model download failed: \(msg)")
                }
                printError("TIP: Delete \(fullModelPath.path) and retry.")
                throw error
            }
        }

        // When model is cached, WhisperKit's init() didn't discover it (download: false).
        // Set modelFolder to the cached path so loadModels() can find it.
        if isCached && whisper.modelFolder == nil {
            whisper.modelFolder = fullModelPath
        }

        if showProgress { reportProgress(0.62, "Compiling CoreML models for your device...") }

        do {
            try await whisper.loadModels()
        } catch {
            printError("Failed to load \(prettyName): \(error.localizedDescription)")
            printError("TIP: Whisper CoreML requires Apple Silicon. Delete \(fullModelPath.path) and retry if models are corrupt.")
            throw error
        }

        if showProgress { reportProgress(0.68, "\(prettyName) ready") }

        // Shared decode options (see note below on chunkingStrategy: .none and
        // the loosened no-speech/logprob thresholds).
        let sharedOptions = makeDecodeOptions()

        // ── Persistent server mode ──────────────────────────────────────────
        // Model is now loaded/specialized once. Loop over stdin requests, keeping
        // the model warm so no further CoreML compilation happens per transcription.
        if serveMode {
            // Tell the host the model is loaded and we're ready for requests.
            FileHandle.standardError.write("READY\n".data(using: .utf8)!)
            if showProgress { reportProgress(0.7, "\(prettyName) ready — awaiting requests") }

            while let raw = readLine(strippingNewline: true) {
                let manifestPath = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                if manifestPath.isEmpty { continue }
                if manifestPath == "__QUIT__" { break }

                var entries: [BatchEntry] = []
                if let data = FileManager.default.contents(atPath: manifestPath),
                   let arr = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                    for e in arr { if let be = parseBatchEntry(e) { entries.append(be) } }
                } else {
                    printError("Failed to read request manifest: \(manifestPath)")
                }

                var results: [[String: Any]] = []
                let total = Double(max(entries.count, 1))
                for (idx, entry) in entries.enumerated() {
                    if showProgress {
                        let pct = 0.05 + 0.9 * Double(idx) / total
                        reportProgress(pct, "Transcribing \(idx + 1)/\(entries.count): \((entry.file as NSString).lastPathComponent)...")
                    }
                    let startEcho = entry.start ?? -1
                    guard FileManager.default.fileExists(atPath: entry.file) else {
                        results.append(["file": entry.file, "start": startEcho, "words": [] as [Any], "error": "File not found"])
                        continue
                    }
                    do {
                        let samples = try await decodeAudioTo16kMono(path: entry.file, start: entry.start, duration: entry.duration)
                        if samples.isEmpty {
                            results.append(["file": entry.file, "start": startEcho, "words": [] as [Any], "error": "No audio samples"])
                            continue
                        }
                        let r: [TranscriptionResult] = try await whisper.transcribe(audioArray: samples, decodeOptions: sharedOptions)
                        results.append(["file": entry.file, "start": startEcho, "words": extractWords(from: r)])
                    } catch {
                        results.append(["file": entry.file, "start": startEcho, "words": [] as [Any], "error": error.localizedDescription])
                    }
                }

                if showProgress {
                    let totalWords = results.reduce(0) { $0 + (($1["words"] as? [[String: Any]])?.count ?? 0) }
                    reportProgress(1.0, "Done — \(totalWords) words")
                }

                // Emit the result framed by a unique token so any CoreML/E5RT noise
                // printed to stdout can't be mistaken for a response. Written via the
                // raw fd because stdout is fully buffered when it's a pipe.
                let payload: Data
                if let jsonData = try? JSONSerialization.data(withJSONObject: results, options: [.sortedKeys]) {
                    payload = jsonData
                } else {
                    payload = "[]".data(using: .utf8)!
                }
                var line = "__SK_JSON__".data(using: .utf8)!
                line.append(payload)
                line.append("\n".data(using: .utf8)!)
                FileHandle.standardOutput.write(line)
            }

            exitCode = 0
            semaphore.signal()
            return
        }

        let totalFiles = Double(batchEntries.count)
        var allResults: [[String: Any]] = []

        for (index, entry) in batchEntries.enumerated() {
            let startEcho = entry.start ?? -1
            guard FileManager.default.fileExists(atPath: entry.file) else {
                printError("File not found: \(entry.file)")
                if batchMode {
                    allResults.append(["file": entry.file, "start": startEcho, "words": [] as [Any], "error": "File not found"])
                }
                continue
            }
            if showProgress {
                let pct = 0.68 + (0.30 * Double(index) / totalFiles)
                let name = (entry.file as NSString).lastPathComponent
                reportProgress(pct, "Transcribing \(index + 1)/\(Int(totalFiles)): \(name)...")
            }

            let options = makeDecodeOptions()

            do {
                // Decode through AVAssetReader instead of letting WhisperKit open the
                // file with AVAudioFile — the latter fails on video containers with
                // 'typ?' (kAudioFileUnsupportedFileTypeError). Hand it raw 16 kHz samples.
                let samples = try await decodeAudioTo16kMono(path: entry.file, start: entry.start, duration: entry.duration)
                if samples.isEmpty {
                    printError("No audio samples decoded from \(entry.file) (silent or no audio track)")
                    if batchMode {
                        allResults.append(["file": entry.file, "start": startEcho, "words": [] as [Any], "error": "No audio samples"])
                    }
                    continue
                }
                let results: [TranscriptionResult] = try await whisper.transcribe(audioArray: samples, decodeOptions: options)
                let words = extractWords(from: results)
                if batchMode {
                    allResults.append(["file": entry.file, "start": startEcho, "words": words])
                } else {
                    allResults = words
                }
            } catch {
                printError("Transcription failed for \(entry.file): \(error.localizedDescription)")
                if batchMode {
                    allResults.append(["file": entry.file, "start": startEcho, "words": [] as [Any], "error": error.localizedDescription])
                } else {
                    throw error
                }
            }
        }

        if showProgress {
            let totalWords = allResults.reduce(0) { sum, r in
                if let words = r["words"] as? [[String: Any]] { return sum + words.count }
                if r["word"] != nil { return sum + 1 }
                return sum
            }
            reportProgress(1.0, "Done — \(totalWords) words from \(batchEntries.count) file(s)")
        }

        let jsonData = try JSONSerialization.data(withJSONObject: allResults, options: [.sortedKeys])
        if let jsonString = String(data: jsonData, encoding: .utf8) { print(jsonString) }

    } catch {
        let errMsg = error.localizedDescription
        printError("Whisper transcription failed: \(errMsg)")
        if errMsg.contains("memory") || errMsg.contains("Memory") {
            printError("TIP: Close other apps to free RAM — Whisper large-v3 needs ~3 GB available.")
        }
        if errMsg.contains("CoreML") || errMsg.contains("mlmodel") {
            printError("TIP: Delete \(fullModelPath.path) and re-run to redownload the model.")
        }
        exitCode = 1
    }
    semaphore.signal()
}

semaphore.wait()
exit(exitCode)
