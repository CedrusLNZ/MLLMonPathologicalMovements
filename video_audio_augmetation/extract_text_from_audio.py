import os
import whisper

def transcribe_audio_directory(source_dir, target_dir, model_name='base'):
    """
    Transcribe all .wav files in a directory using Whisper
    
    Args:
        source_dir: Path to directory containing .wav files
        target_dir: Path to directory where text files will be saved
        model_name: Whisper model size (default: 'base')
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Load the Whisper model
    model = whisper.load_model(model_name)
    
    # Process each WAV file in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith(".wav"):
            # Build full file paths
            audio_path = os.path.join(source_dir, filename)
            base_name = os.path.splitext(filename)[0]
            txt_path = os.path.join(target_dir, f"{base_name}.txt")
            
            # Transcribe audio
            print(f"Transcribing {filename}...")
            result = model.transcribe(audio_path)
            
            # Save transcription to text file
            with open(txt_path, "w") as txt_file:
                txt_file.write(result["text"])
            
            print(f"Saved transcription to {txt_path}")

if __name__ == "__main__":
    # Configuration - modify these paths as needed
    SOURCE_DIR = "/mnt/SSD1/prateik/seizure_audition/data/new_patients/audio"
    TARGET_DIR = "/mnt/SSD1/prateik/seizure_audition/data/new_patients/text"
    MODEL_SIZE = 'large'  # Choices: 'tiny', 'base', 'small', 'medium', 'large'
    
    # Run transcription
    transcribe_audio_directory(SOURCE_DIR, TARGET_DIR, MODEL_SIZE)