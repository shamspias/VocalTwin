"""
Generate speech from text using a target-speaker embedding previously
computed by `VoiceTrainer`.  Pipeline:

    1.  Use **MeloTTS** to synthesise 'base voice' audio from text.
    2.  Extract SE (speaker embedding) of that base voice.
    3.  Use **OpenVoice ToneColorConverter** to swap tone-colour to
        the *target* speaker (your voice).
    4.  Save final audio to `outputs/`.

Expected directory layout (see README):

    checkpoints_v2/
        converter/           <-- same as in trainer
    checkpoints/target_se.pth   <-- produced by trainer

Author: you
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import List

import torch
from melo.api import TTS  # pip install git+https://github.com/myshell-ai/MeloTTS.git
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


class TextToSpeechSynthesizer:
    """
    Convert all *.txt files in `text_dir` to speech in the user’s voice,
    writing WAV files into `output_dir`.

    Parameters
    ----------
    checkpoint_dir : Path-like
        Folder containing *target_se.pth* from VoiceTrainer.
    language : str, optional
        TTS language code from MeloTTS (`EN`, `ES`, `FR`, `ZH`, `JA`, `KO`…)
    device : str, optional
        PyTorch device; default is CUDA if available.
    """

    def __init__(
            self,
            checkpoint_dir: os.PathLike,
            language: str = "EN",
            device: str | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.language = language.upper()
        self.device = (
            device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        # ---- Load tone-colour converter (same as trainer) ----
        conv_dir = Path("checkpoints_v2/converter")  # keep fixed path for v2
        self.converter = ToneColorConverter(
            str(conv_dir / "config.json"), device=self.device
        )
        self.converter.load_ckpt(str(conv_dir / "checkpoint.pth"))

        # ---- Load user’s target-speaker embedding ----
        target_ckpt = self.checkpoint_dir / "target_se.pth"
        if not target_ckpt.exists():
            raise FileNotFoundError(
                f"target_se.pth not found in {self.checkpoint_dir}. "
                "Run trainer first."
            )
        self.target_se = torch.load(target_ckpt, map_location=self.device)["se"]
        logger.info("🎯 Loaded target speaker embedding from %s", target_ckpt)

        # ---- Init MeloTTS ----
        self.tts = TTS(language=self.language, device=self.device)
        # Choose the first native speaker voice for the language
        self.base_speaker_id = self.tts.get_speaker_ids()[0]

        logger.debug(
            "TextToSpeechSynthesizer initialised "
            "(language=%s, device=%s, base_speaker_id=%s)",
            self.language,
            self.device,
            self.base_speaker_id,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def synthesize(
            self,
            text_dir: os.PathLike,
            output_dir: os.PathLike,
    ) -> None:
        """
        Convert each .txt file within `text_dir` to speech audio and store
        the resulting .wav in `output_dir` (same filenames, .txt → .wav).
        """

        text_dir = Path(text_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        txt_files = self._collect_txts(text_dir)
        if not txt_files:
            logger.error("No .txt files found in %s – nothing to synthesise.", text_dir)
            return

        logger.info("📝 Found %d text files – starting synthesis …", len(txt_files))
        for tf in txt_files:
            self._process_file(tf, output_dir)

        logger.info("✅ All done – generated audio saved in %s", output_dir.resolve())

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _collect_txts(root: Path) -> List[Path]:
        """Recursively gather *.txt files."""
        return [p for p in root.rglob("*.txt") if p.is_file()]

    def _process_file(self, txt_path: Path, output_dir: Path) -> None:
        """Generate voice-cloned speech for one text file."""
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            logger.warning("⚠️  %s is empty – skipping.", txt_path.name)
            return

        logger.info("🔊 Synthesising %s …", txt_path.name)

        with tempfile.TemporaryDirectory() as tmpdir:
            base_audio = Path(tmpdir) / "base.wav"

            # 1) Use MeloTTS to speak with a neutral/base voice
            self.tts.tts_to_file(
                text=text,
                speaker_ids=[self.base_speaker_id],
                output_path=str(base_audio),
            )

            # 2) Extract SE of the base voice
            src_se, _ = se_extractor.get_se(
                str(base_audio),
                self.converter,
                vad=True,
            )

            # 3) Convert tone-colour to the target speaker
            final_out = output_dir / txt_path.with_suffix(".wav").name
            self.converter.convert(
                audio_src_path=str(base_audio),
                src_se=src_se,
                tgt_se=self.target_se,
                output_path=str(final_out),
            )

        logger.debug("   ↳ saved %s", final_out.name)
