import sys
from src.trainer import VoiceTrainer
from src.synthesizer import TextToSpeechSynthesizer


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [train | synthesize | train_and_synthesize]")
        return

    command = sys.argv[1]
    if command == "train":
        trainer = VoiceTrainer()
        trainer.train(audio_dir="audio_samples", checkpoint_dir="checkpoints")
    elif command == "synthesize":
        synthesizer = TextToSpeechSynthesizer()
        synthesizer.synthesize(text_dir="texts", checkpoint_dir="checkpoints", output_dir="outputs")
    elif command == "train_and_synthesize":
        trainer = VoiceTrainer()
        trainer.train(audio_dir="audio_samples", checkpoint_dir="checkpoints")
        synthesizer = TextToSpeechSynthesizer()
        synthesizer.synthesize(text_dir="texts", checkpoint_dir="checkpoints", output_dir="outputs")
    else:
        print("Unknown command. Use: train, synthesize, or train_and_synthesize")


if __name__ == "__main__":
    main()
