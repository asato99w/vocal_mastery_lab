#!/usr/bin/env python3
"""
Demucs PyTorch → CoreML conversion PoC
"""

import torch
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
import coremltools as ct


def main():
    print("Loading Demucs model from TorchAudio...")
    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model()
    model.eval()

    print(f"Model loaded: {type(model)}")
    print(f"Sample rate: {bundle.sample_rate}")

    # Example input: 2 channels (stereo), 5 seconds at 44.1kHz
    sample_rate = bundle.sample_rate
    duration = 5
    batch_size = 1
    channels = 2
    samples = int(sample_rate * duration)

    example_input = torch.randn(batch_size, channels, samples)

    print(f"Example input shape: {example_input.shape}")

    # Trace model
    print("Tracing model with example input...")
    traced_model = torch.jit.trace(model, example_input)

    print("Converting to CoreML...")
    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=example_input.shape)],
        outputs=[ct.TensorType(name="separated_sources")],
        minimum_deployment_target=ct.target.iOS16,
    )

    output_path = "../models/demucs_vocals.mlpackage"
    print(f"Saving CoreML model to {output_path}...")
    mlmodel.save(output_path)

    print("Conversion complete!")
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    main()
