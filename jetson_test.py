import argparse
import sys
import torch


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = "/home/nrashvan/TSGAD"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Project imports
# ============================================================

import lib.dist as dist
from models import VAE


def main():

    # ============================================================
    # Arguments
    # ============================================================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pretrained VAE checkpoint",
    )

    args = parser.parse_args()

    # ============================================================
    # Device
    # ============================================================

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print("========================================")
    print("DEVICE")
    print("========================================")
    print(device)

    # ============================================================
    # Model configuration
    #
    # From your arguments.yaml
    # ============================================================

    latent_dim = 10
    seg_len = 24

    conv = False
    graph = True

    headless = False

    tcvae = True
    mss = True

    dropout = 0.3

    conv_oper = "sagc"
    act = "relu"

    mse = False

    alpha = 1
    gamma = 1

    exclude_mutinfo = False

    # ============================================================
    # Distributions
    # ============================================================

    prior_dist = dist.Normal()
    q_dist = dist.Normal()

    # ============================================================
    # Create VAE
    # ============================================================

    vae = VAE(
        z_dim=latent_dim,
        use_cuda=torch.cuda.is_available(),
        device=str(device),
        prior_dist=prior_dist,
        q_dist=q_dist,
        include_mutinfo=not exclude_mutinfo,
        tcvae=tcvae,
        conv=conv,
        graph=graph,
        mss=mss,
        drop_out=dropout,
        conv_oper=conv_oper,
        act=act,
        headless=headless,
        input_frames=seg_len,
        mse=mse,
        alpha=alpha,
        gamma=gamma,
    )

    vae = vae.to(device)

    # ============================================================
    # Parameter count
    # ============================================================

    total_params = sum(
        p.numel()
        for p in vae.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in vae.parameters()
        if p.requires_grad
    )

    print("\n========================================")
    print("MODEL INFORMATION")
    print("========================================")

    print(
        f"Total parameters     : {total_params:,}"
    )

    print(
        f"Trainable parameters : {trainable_params:,}"
    )

    print(
        f"Total parameters (M) : "
        f"{total_params / 1e6:.4f} M"
    )

    # ============================================================
    # Load pretrained checkpoint
    # ============================================================

    print("\n========================================")
    print("LOADING CHECKPOINT")
    print("========================================")

    print(args.checkpoint)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    print(
        "Checkpoint type:",
        type(checkpoint),
    )

    if isinstance(checkpoint, dict):

        print(
            "Checkpoint keys:",
            checkpoint.keys(),
        )

    # ============================================================
    # Extract state_dict
    # ============================================================

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "model" in checkpoint:

            state_dict = checkpoint[
                "model"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # ============================================================
    # Remove "module." prefix if necessary
    # ============================================================

    clean_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        clean_state_dict[key] = value

    # ============================================================
    # Load pretrained weights
    # ============================================================

    vae.load_state_dict(
        clean_state_dict,
        strict=True,
    )

    vae.eval()

    print(
        "\nPretrained VAE loaded successfully."
    )

    # ============================================================
    # Create random input
    #
    # Same pose shape used by your original model:
    #
    # [Batch, Channels, Frames, Keypoints]
    #
    # [1, 2, 24, 18]
    # ============================================================

    torch.manual_seed(0)

    x = torch.randn(
        1,
        2,
        seg_len,
        18,
        dtype=torch.float32,
        device=device,
    )

    print("\n========================================")
    print("INPUT")
    print("========================================")

    print(
        "Input shape:",
        x.shape,
    )

    print(
        "Input dtype:",
        x.dtype,
    )

    print(
        "Input device:",
        x.device,
    )

    print(
        "Input min:",
        x.min().item(),
    )

    print(
        "Input max:",
        x.max().item(),
    )

    print(
        "Input mean:",
        x.mean().item(),
    )

    # ============================================================
    # Inference
    #
    # Same encoder call used in your original main:
    #
    # _, z_params, _ = vae.encode(data)
    # ============================================================

    print("\n========================================")
    print("INFERENCE")
    print("========================================")

    with torch.inference_mode():

        output1, z_params, output3 = vae.encode(
            x
        )

    # ============================================================
    # Print outputs
    # ============================================================

    print("\n========================================")
    print("MODEL OUTPUT")
    print("========================================")

    print(
        "Output 1 type:",
        type(output1),
    )

    if torch.is_tensor(output1):

        print(
            "Output 1 shape:",
            output1.shape,
        )

    print(
        "\nz_params shape:",
        z_params.shape,
    )

    print(
        "z_params min:",
        z_params.min().item(),
    )

    print(
        "z_params max:",
        z_params.max().item(),
    )

    print(
        "z_params mean:",
        z_params.mean().item(),
    )

    print(
        "z_params std:",
        z_params.std().item(),
    )

    print(
        "\nOutput 3 type:",
        type(output3),
    )

    if torch.is_tensor(output3):

        print(
            "Output 3 shape:",
            output3.shape,
        )

    # ============================================================
    # Flatten latent output
    # ============================================================

    z_flat = z_params.view(
        z_params.shape[0],
        -1,
    )

    print("\n========================================")
    print("FINAL ENCODER OUTPUT")
    print("========================================")

    print(
        "z_params shape:",
        z_params.shape,
    )

    print(
        "Flattened shape:",
        z_flat.shape,
    )

    print(
        "Flattened output:"
    )

    print(z_flat)

    print("\n========================================")
    print("TEST COMPLETED SUCCESSFULLY")
    print("========================================")


if __name__ == "__main__":
    main()