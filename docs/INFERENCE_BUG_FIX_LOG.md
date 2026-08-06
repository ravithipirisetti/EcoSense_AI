# EcoSense AI — Inference Bug Fix Log (Aug 2026)

## Issue
Live and file predictions returned wrong species (e.g. *Asian Koel* → *Spotted Owlet*) despite reasonable training metrics (Val/Test Acc ≈ 52%–56%).

## Root Cause
Model (`audio_model_yamnet.keras`) and label encoder (`label_encoder.pkl`) got out of sync. Different training scripts (`train_birdnet.py`, `train_completed_only.py`) overwrote the same file paths after training on different species subsets (70 vs 17 vs 10 classes). As a result, model output index `i` no longer matched what the encoder thought that index meant.

## Contributing Factors
1. **Unnormalized Mic Audio:** Live mic audio via `sounddevice` was not peak-normalized like training audio via `librosa.load`, causing amplitude scale mismatch in YAMNet log-mel spectrogram features.
2. **Silence / Noise Dilution:** Mean-pooling YAMNet embeddings over long quiet segments diluted the vocalization signal.
3. **No Low-Volume Guard:** Microphone input was passed to inference without checking for silence or low audio energy.
4. **No Validation Guardrails:** No validation existed to check `model.output_shape[-1] == len(label_encoder.classes_)` before decoding predictions.

## Fixes Applied
1. **Unified Feature Extractor:** Implemented `extract_yamnet_embedding_from_signal` in `ai/yamnet_extractor.py` with peak normalization (`sig = sig / max_val`), silence trimming (`top_db=25`), padding, and mean-pooling. Used identically for both file loading and mic input.
2. **Class-Count Validation Guard:** Added runtime check `if model.output_shape[-1] != len(label_encoder.classes_): raise ValueError(...)` in `live_predict.py`, `predict_audio.py`, and `test_yamnet.py`.
3. **RMS Low-Volume Guard:** Added RMS-based silence check (`rms < 0.003`) in `live_predict.py` to prevent false triggering on background silence.
4. **Clean Retraining & Atomic Artifact Saving:** Cleared stale `.keras` and `.pkl` artifacts and retrained on a clean 53-species dataset ($\ge 25$ clips/species) using `train_completed_only.py`.

## Verification & Results
- **Model / Encoder Synchronization:** 53 output classes / 53 encoder classes (100% Match).
- **Test Accuracy:** 40.69% across 53 multi-class species (2,031 audio files).
- ** Asian Koel Test File (`S14_1.wav`):** Correctly predicted **Asian Koel (#1) at 72.0% confidence** (previously predicted Spotted Owlet at 65%).
- ** White-throated Kingfisher Test File (`S17_1.wav`):** Correctly predicted **White-throated Kingfisher (#1) at 26.7% confidence**.

## Protocol Going Forward
Always train and save model + encoder together in a single script run. Never allow multiple training scripts to write to shared artifact paths independently.
