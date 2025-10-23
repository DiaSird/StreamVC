import argparse
import time
import safetensors.torch
import soundfile as sf
import numpy as np
import torch
import torchaudio.functional as F
from tqdm import tqdm
from streamvc import StreamVC
from validate import val_mcd, val_wer

SAMPLE_RATE = 16_000
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
DTYPE = torch.float32
# CHUNK_SIZE = 320  # 20ms chunk
CHUNK_SIZE = 1024  # 64ms chunk


@torch.no_grad()
def streaming_inference(model, source, target):
    """
    source, target: 1D torch tensors on DEVICE
    """
    model.eval()
    source_len = source.shape[0]
    output_chunks = []

    for start in tqdm(range(0, source_len, CHUNK_SIZE), desc="Streaming Inference"):
        end = min(start + CHUNK_SIZE, source_len)

        source_chunk = source[start:end]
        target_chunk = target[start:end]

        if source_chunk.numel() == 0 or target_chunk.numel() == 0:
            continue

        t0 = time.time()

        out_chunk = model(source_chunk, target_chunk)

        t1 = time.time()
        elapsed_ms = (t1 - t0) * 1000

        print(f"Chunk [{start}:{end}] - Inference time: {elapsed_ms:.2f} ms")

        out_chunk_np = out_chunk.cpu().numpy()
        output_chunks.append(out_chunk_np)

    return np.concatenate(output_chunks, axis=0)


@torch.no_grad()
def main(args):
    # Load StreamVC
    model = StreamVC().to(device=DEVICE, dtype=DTYPE)
    encoder_state_dict = safetensors.torch.load_file(args.checkpoint, device=DEVICE)
    model.load_state_dict(encoder_state_dict)
    print("StreamVC model loaded.")

    source_speech, orig_sr = sf.read(args.source_speech)
    source_speech = torch.from_numpy(source_speech).to(device=DEVICE, dtype=DTYPE)
    if orig_sr != SAMPLE_RATE:
        source_speech = F.resample(source_speech, orig_sr, SAMPLE_RATE)

    target_speech, orig_sr = sf.read(args.target_speech)
    target_speech = torch.from_numpy(target_speech).to(device=DEVICE, dtype=DTYPE)
    if orig_sr != SAMPLE_RATE:
        target_speech = F.resample(target_speech, orig_sr, SAMPLE_RATE)

    start = time.time()

    if args.stream:
        print("Streaming inference...")
        output = streaming_inference(model, source_speech, target_speech)
    else:
        print("Full audio inference...")
        with torch.no_grad():
            output = model(source_speech, target_speech).cpu().numpy()

    end = time.time()
    print(f"Time: {end - start:.3f} [s]")

    sf.write(args.output_path, output, SAMPLE_RATE)
    print(f"Saved output to {args.output_path}")

    mcd_score = val_mcd(args.source_speech, args.output_path)
    wer_score = val_wer(args.source_speech, args.output_path)

    print(f"MCD: {mcd_score:.3f} dB")
    print(f"WER: {wer_score:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="StreamVC Inference Script",
        description="Inference script for StreamVC model, performs voice conversion on a single audio file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="./model.safetensors",
        help="Path to a pretrained StreamVC model checkpoint (safetensors).",
    )
    parser.add_argument(
        "-s", "--source-speech", type=str, help="Path to source speech audio file."
    )
    parser.add_argument(
        "-t", "--target-speech", type=str, help="Path to target speech audio file."
    )
    parser.add_argument(
        "-o", "--output-path", type=str, default="./out.wav", help="Output file path."
    )
    parser.add_argument(
        "--stream", action="store_true", help="Enable streaming inference mode."
    )

    main(parser.parse_args())
