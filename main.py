"""
VocalTwin – CLI
===============

Personalised voice cloning & TTS:
1) Train on your MP3 recordings.
2) Synthesis any text as your own voice.

Commands
--------
train                   – extract speaker-embedding only
synthesize              – generate speech from .txt files
train_and_synthesize    – do both in sequence
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.trainer import VoiceTrainer
from src.synthesizer import TextToSpeechSynthesizer


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="VocalTwin",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Voice cloning & text-to-speech with OpenVoice-V2 + MeloTTS",
    )
    p.add_argument(
        "command",
        choices=("train", "synthesize", "train_and_synthesize"),
        help="Action to perform",
    )

    # Common paths
    p.add_argument("--audio_dir", default="audio_samples", type=str,
                   help="Directory containing MP3 training recordings")
    p.add_argument("--text_dir", default="texts", type=str,
                   help="Directory containing input .txt files")
    p.add_argument("--checkpoint_dir", default="checkpoints", type=str,
                   help="Where to save / load target_se.pth")
    p.add_argument("--output_dir", default="outputs", type=str,
                   help="Where to write generated WAV files")

    # TTS options
    p.add_argument("--language", default="EN", type=str,
                   help="TTS language code understood by MeloTTS")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    # Resolve to absolute paths for robustness
    audio_dir = Path(args.audio_dir).resolve()
    text_dir = Path(args.text_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if args.command == "train":
        VoiceTrainer().train(audio_dir, checkpoint_dir)

    elif args.command == "synthesize":
        TextToSpeechSynthesizer(
            checkpoint_dir=checkpoint_dir,
            language=args.language,
        ).synthesize(text_dir, output_dir)

    elif args.command == "train_and_synthesize":
        VoiceTrainer().train(audio_dir, checkpoint_dir)
        TextToSpeechSynthesizer(
            checkpoint_dir=checkpoint_dir,
            language=args.language,
        ).synthesize(text_dir, output_dir)

    else:  # pragma: no cover – argparse guarantees we never land here
        print("Unknown command.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
