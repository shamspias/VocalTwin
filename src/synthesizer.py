"""
Generate speech from text using a target-speaker embedding previously
computed by `VoiceTrainer`.  Pipeline:

    1.  Use **MeloTTS** to synthesise “base-voice” audio from text.
    2.  Extract SE (speaker embedding) of that base voice.
    3.  Use **OpenVoice ToneColorConverter** to swap tone-colour to
        the *target* speaker (your voice).
    4.  Save final audio to `outputs/`.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
import nltk
import torch
from melo.api import TTS
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)


class TextToSpeechSynthesizer:
    """
    Convert every *.txt file in `text_dir` to speech using the user’s voice,
    writing WAV files to `output_dir`.
    """

    def __init__(
            self,
            checkpoint_dir: os.PathLike,
            language: str = "EN",
            device: str | None = None,
            base_speaker_id: int = 0,  # default neutral voice in each MeloTTS model
    ) -> None:
        # ── paths ─────────────────────────────────────────────────────────
        self.checkpoint_dir = Path(checkpoint_dir)
        self.language = language.upper()
        self.device = (
            device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        # ── OpenVoice tone-colour converter ───────────────────────────────
        conv_dir = Path("checkpoints_v2/converter")
        self.converter = ToneColorConverter(str(conv_dir / "config.json"), device=self.device)
        self.converter.load_ckpt(str(conv_dir / "checkpoint.pth"))

        # ── target-speaker embedding ──────────────────────────────────────
        tgt_path = self.checkpoint_dir / "target_se.pth"
        if not tgt_path.exists():
            raise FileNotFoundError(
                f"{tgt_path} not found – run `main.py train` first."
            )
        self.target_se = torch.load(tgt_path, map_location=self.device)["se"]
        logger.info("🎯 Loaded target speaker embedding from %s", tgt_path)

        # ── MeloTTS initialisation ────────────────────────────────────────
        self.tts = TTS(language=self.language, device=self.device)
        self.base_speaker_id = base_speaker_id  # always an *int*
        logger.debug(
            "TextToSpeechSynthesizer ready (lang=%s | device=%s | base_speaker_id=%d)",
            self.language,
            self.device,
            self.base_speaker_id,
        )

    # ======================================================================
    # Public API
    # ======================================================================

    def synthesize(self, text_dir: os.PathLike, output_dir: os.PathLike) -> None:
        """Generate WAVs for every .txt in `text_dir`."""
        text_dir = Path(text_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        txt_files = [p for p in text_dir.rglob("*.txt") if p.is_file()]
        if not txt_files:
            logger.error("No .txt files found in %s – nothing to synthesise.", text_dir)
            return

        logger.info("📝 Found %d text files – starting synthesis …", len(txt_files))
        for tf in txt_files:
            self._process_one(tf, output_dir)

        logger.info("✅ All done – audio written to %s", output_dir.resolve())

    # ======================================================================
    # Internal helpers
    # ======================================================================

    def _process_one(self, txt_path: Path, out_dir: Path) -> None:
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            logger.warning("⚠️  %s is empty – skipping.", txt_path.name)
            return

        logger.info("🔊 Synthesising %s …", txt_path.name)

        with tempfile.TemporaryDirectory() as tmp:
            base_wav = Path(tmp) / "base.wav"

            # 1) Generate base voice
            self.tts.tts_to_file(
                text=text,
                speaker_id=self.base_speaker_id,  # ← single int, per new API
                output_path=str(base_wav),
            )

            # 2) Extract SE of base voice
            src_se, _ = se_extractor.get_se(
                str(base_wav),
                self.converter,
                vad=True,
            )

            # 3) Tone-colour conversion
            final_wav = out_dir / txt_path.with_suffix(".wav").name
            self.converter.convert(
                audio_src_path=str(base_wav),
                src_se=src_se,
                tgt_se=self.target_se,
                output_path=str(final_wav),
            )

        logger.debug("   ↳ saved %s", final_wav.name)
