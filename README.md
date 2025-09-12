Here’s a detailed `README.md` tailored for your Rust Whisper + Gemma summarization project:
# Rust Voice Agent: Whisper + Gemma Summarizer

This Rust project implements a **local agent flow** that:

1. Takes an audio file (`.mp3`), converts it to `.wav`.
2. Runs **Whisper** locally to transcribe the audio to text.
3. Sends the transcription to **Gemma** (via Ollama) to generate a summary.
4. Saves the summarized transcript to disk.

It supports processing transcripts in **chunks** to handle long audio files.

---

## 🚀 Prerequisites

- Rust (1.70+ recommended)
- `ffmpeg` (for audio conversion)
- [Ollama](https://ollama.ai) installed
- Internet access (for downloading models)

---

## 1️⃣ Clone the project

```bash
git clone <your-repo-url>
cd meeting_summarizer
````

---

## 2️⃣ Install Rust dependencies

```bash
cargo build
```

Dependencies are in `Cargo.toml`:

* `whisper-rs` for Whisper ASR
* `hound` for WAV handling
* `reqwest` + `serde_json` for Ollama API calls
* `tokio` for async runtime

---

## 3️⃣ Download the Whisper model

1. Go to [Whisper.cpp models](https://huggingface.co/ggerganov/whisper.cpp/tree/main)
2. Download the **English base model**: `ggml-base.en.bin`
3. Place it in `models/`:

```
meeting_summarizer/
└── models/
    └── ggml-base.en.bin
```

---

## 4️⃣ Install and run Ollama with Gemma

1. Install Ollama:

```bash
brew install ollama   # macOS
# or follow instructions for Linux/Windows
```

2. Pull Gemma model:

```bash
ollama pull gemma
```

3. Start the Ollama API server:

```bash
ollama serve
```

By default, it listens on `http://localhost:11434/api/generate`.

4. Optional: verify it’s running:

```bash
curl http://localhost:11434/api/tags
ollama run gemma "Hello!"
```

---

## 5️⃣ Prepare audio input

Whisper expects **16 kHz mono PCM WAV**:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le input.wav
```

Place `input.wav` in the project root.

---

## 6️⃣ Run the Rust agent

```bash
cargo run -- <chunk_size_in_seconds>
```

* `chunk_size_in_seconds` (optional) determines how large each audio chunk is for transcription.
  Example: `cargo run -- 60` will process audio in 60-second chunks.
* If no parameter is passed, the program will transcribe the whole file at once.

---

## 7️⃣ Output

* Transcription printed to console.
* Summary printed to console.
* Summary saved to `summary.txt` in project root.

---

## 8️⃣ Notes & Tips

* Whisper is **CPU-intensive**; for longer audio, consider using smaller models (`ggml-small.en.bin`) or GPU acceleration if available.
* Gemma’s LLM has a **fixed context window**. For long transcripts, either:

  * Summarize in chunks and feed previous summaries into the next chunk.
  * Use hierarchical summarization: chunk → summarize → summarize summaries.
* You can batch multiple files by creating multiple `WhisperState`s with the same `WhisperContext`.

---

## ⚡ Example Workflow

```bash
# Convert audio
ffmpeg -i meeting.mp3 -ar 16000 -ac 1 -c:a pcm_s16le input.wav

# Pull Gemma
ollama pull gemma

# Start Ollama server
ollama serve

# Run summarizer with 60-second chunks
cargo run -- 60

# Check output
cat summary.txt
```

---

## 📚 References

* [whisper-rs](https://crates.io/crates/whisper-rs)
* [Whisper.cpp models](https://huggingface.co/ggerganov/whisper.cpp/tree/main)
* [Ollama](https://ollama.ai)
* [ffmpeg](https://ffmpeg.org/)

---

## License

MIT License
