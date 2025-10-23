import librosa
import numpy as np
import whisper
from jiwer import wer


def extract_mcep(y, sr, n_mfcc=13):
    # MCD (Mel Cepstral Distortion)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=40
    )
    log_mel_spec = librosa.power_to_db(mel_spec)
    mfcc = librosa.feature.mfcc(S=log_mel_spec, n_mfcc=n_mfcc)
    return mfcc.T


def val_mcd(ref_path, conv_path, sr=16000):
    ref_audio, _ = librosa.load(ref_path, sr=sr)
    conv_audio, _ = librosa.load(conv_path, sr=sr)

    # MFCC
    mcep_ref = extract_mcep(ref_audio, sr)
    mcep_conv = extract_mcep(conv_audio, sr)

    min_len = min(len(mcep_ref), len(mcep_conv))
    mcep_ref = mcep_ref[:min_len]
    mcep_conv = mcep_conv[:min_len]

    diff = mcep_ref - mcep_conv
    sq_diff = np.square(diff)
    sum_sq = np.sum(sq_diff, axis=1)
    mcd = np.mean(np.sqrt(2) * np.sqrt(sum_sq))
    return mcd


def val_wer(ref_path, conv_path, whisper_model="base"):
    # WER (Word Error Rate)

    model = whisper.load_model(whisper_model)

    ref_result = model.transcribe(ref_path)
    conv_result = model.transcribe(conv_path)

    ref_text = ref_result["text"].strip().lower()
    conv_text = conv_result["text"].strip().lower()

    # print(ref_text)
    # print(conv_text)

    return wer(ref_text, conv_text)


if __name__ == "__main__":
    ref_wav = "ref.wav"
    conv_wav = "converted.wav"

    mcd_score = val_mcd(ref_wav, conv_wav)
    wer_score = val_wer(ref_wav, conv_wav)

    print(f"MCD: {mcd_score:.3f} dB")
    print(f"WER: {wer_score:.3f}")
