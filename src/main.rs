use anyhow::Result;
use hound;
use reqwest::Client;
use std::env;
use std::fs;
use std::path::Path;
use tokio;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Pretty-print a transcript with one line per segment
fn pretty_transcript(segments: &[String]) -> String {
    segments.join("\n")
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse audio file from argv
    let args: Vec<String> = env::args().collect();
    let audio_path = args.get(1).expect("Usage: cargo run -- <input.wav>");

    if !audio_path.ends_with(".wav") {
        anyhow::bail!("Input file must be a .wav file");
    }

    // Derive output filenames
    let audio_stem = Path::new(audio_path)
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let transcript_path = format!("{}.transcript", audio_stem);
    let summary_path = format!("{}.summary", audio_stem);

    // 1. Load Whisper model
    let ctx = WhisperContext::new_with_params(
        "models/ggml-base.en.bin",
        WhisperContextParameters::default(),
    )?;

    // 2. Load audio file
    let audio_data = load_wav_mono_f32(audio_path)?;

    // 3. Transcribe with Whisper
    let mut state = ctx.create_state()?;
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("en"));

    state.full(params, &audio_data)?;
    let num_segments = state.full_n_segments();

    let mut segments = Vec::new();

    for i in 0..num_segments {
        if let Some(segment) = state.get_segment(i) {
            let text = segment.to_str_lossy().unwrap_or_default().to_string();
            segments.push(text);
        }
    }

    let transcription = segments.join(" ");
    let pretty_transcript = pretty_transcript(&segments);

    println!("--- TRANSCRIPTION ---\n{}", pretty_transcript);

    // 4. Load summarization prompt template
    let prompt_template =
        fs::read_to_string("prompt.txt").expect("Missing prompt.txt file in project folder");
    let prompt = prompt_template.replace("{}", &transcription);

    // 5. Summarize transcript with Ollama
    let client = Client::new();
    let body = serde_json::json!({
        "model": "gemma",
        "prompt": prompt,
    });

    let res = client
        .post("http://localhost:11434/api/generate")
        .json(&body)
        .send()
        .await?;

    let mut summary = String::new();
    let text = res.text().await?;
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(resp) = value["response"].as_str() {
                summary.push_str(resp);
            }
        }
    }

    println!("--- FINAL SUMMARY ---\n{}", summary);

    // 6. Save transcript and summary
    fs::write(&transcript_path, pretty_transcript)?;
    fs::write(&summary_path, summary)?;

    println!("Transcript saved to {}", transcript_path);
    println!("Summary saved to {}", summary_path);

    Ok(())
}

/// Load WAV file as f32 mono PCM samples
fn load_wav_mono_f32(path: &str) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    if spec.channels != 1 {
        anyhow::bail!("Expected mono audio (1 channel), got {}", spec.channels);
    }

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    Ok(samples)
}
