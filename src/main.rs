use anyhow::Result;
use hound;
use reqwest::Client;
use serde_json::Value;
use std::env;
use std::fs;
use tokio;
use tokio_stream::StreamExt;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[tokio::main]
async fn main() -> Result<()> {
    // Parse chunk count from argv, default = 1
    let args: Vec<String> = env::args().collect();
    let num_chunks: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);

    // 1. Load Whisper model
    let ctx = WhisperContext::new_with_params(
        "models/ggml-base.en.bin",
        WhisperContextParameters::default(),
    )?;

    // 2. Load audio file
    let audio_data = load_wav_mono_f32("input.wav")?;

    // 3. Transcribe with Whisper
    let mut state = ctx.create_state()?;
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("en"));

    state.full(params, &audio_data)?;
    let num_segments = state.full_n_segments();
    let mut transcription = String::new();

    for i in 0..num_segments {
        if let Some(segment) = state.get_segment(i) {
            let text = segment.to_str_lossy().unwrap_or_default();
            transcription.push_str(&text);
            transcription.push(' ');
        }
    }

    println!("--- TRANSCRIPTION ---\n{}", transcription);

    // 4. Chunk transcript into N pieces
    let chunks = split_into_chunks(&transcription, num_chunks);

    // 5. Summarize each chunk with Ollama
    let client = Client::new();
    let mut summaries = Vec::new();

    for (i, chunk) in chunks.iter().enumerate() {
        println!("--- Summarizing chunk {}/{} ---", i + 1, num_chunks);

        let body = serde_json::json!({
            "model": "gemma",
            "prompt": format!("Thoroughly summarize the following part of a transcript:\n{}", chunk),
        });

        let res = client
            .post("http://localhost:11434/api/generate")
            .json(&body)
            .send()
            .await?;

        // Stream the response line by line and collect all "response" fields
        let mut summary = String::new();

        let text = res.text().await?; // get the full response body
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

        summaries.push(summary);
    }

    // 6. Combine summaries
    let final_summary = summaries.join("\n\n");
    println!("--- FINAL SUMMARY ---\n{}", final_summary);

    // 7. Save to file
    fs::write("summary.txt", final_summary)?;

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

/// Split text into N roughly equal chunks
fn split_into_chunks(text: &str, n: usize) -> Vec<String> {
    if n <= 1 {
        return vec![text.to_string()];
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    let chunk_size = (words.len() + n - 1) / n; // ceiling division

    words.chunks(chunk_size).map(|c| c.join(" ")).collect()
}
