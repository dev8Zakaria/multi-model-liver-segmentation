"""Convert the five trained vessel checkpoints (.pth) to ONNX.

One-off step, run on a machine that has PyTorch installed. The web app itself
only needs onnxruntime, which keeps the deployed image small and CPU inference
fast.

    python tools/export_onnx.py --downloads C:/Users/zakar/Downloads

Every architecture is rebuilt exactly as in the training notebooks and loaded
with strict=True, so a mismatch fails loudly here instead of producing a
silently wrong model. Each export is then checked against the PyTorch output.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 5
PATCH_SIZE = 256


# --- Attention U-Net (Run 3): same layer names/widths as the trained checkpoint ---
class AttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.W_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionUNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=NUM_CLASSES, base=32):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8, base * 16]
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(in_ch, chs[0])
        self.enc2 = ConvBlock(chs[0], chs[1])
        self.enc3 = ConvBlock(chs[1], chs[2])
        self.enc4 = ConvBlock(chs[2], chs[3])
        self.bottleneck = ConvBlock(chs[3], chs[4])
        self.up4 = nn.ConvTranspose2d(chs[4], chs[3], 2, stride=2)
        self.att4 = AttentionGate(chs[3], chs[3], chs[3] // 2)
        self.dec4 = ConvBlock(chs[4], chs[3])
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.att3 = AttentionGate(chs[2], chs[2], chs[2] // 2)
        self.dec3 = ConvBlock(chs[3], chs[2])
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.att2 = AttentionGate(chs[1], chs[1], chs[1] // 2)
        self.dec2 = ConvBlock(chs[2], chs[1])
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.att1 = AttentionGate(chs[0], chs[0], chs[0] // 2)
        self.dec1 = ConvBlock(chs[1], chs[0])
        self.out_conv = nn.Conv2d(chs[0], num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, self.att4(d4, e4)], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, self.att3(d3, e3)], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self.att2(d2, e2)], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self.att1(d1, e1)], dim=1))
        return self.out_conv(d1)


# --- SegFormer (Run 7): the decode head works at 1/4 resolution, upsample back ---
class SegformerWrapper(nn.Module):
    def __init__(self, hf_model, target_size):
        super().__init__()
        self.hf_model = hf_model
        self.target_size = target_size

    def forward(self, pixel_values):
        logits = self.hf_model(pixel_values=pixel_values).logits
        return F.interpolate(logits, size=self.target_size, mode="bilinear", align_corners=False)


def load_state(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    return state


def build_unetpp(ckpt):
    import segmentation_models_pytorch as smp
    model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, classes=NUM_CLASSES)
    model.load_state_dict(load_state(ckpt), strict=True)
    return model


def build_attention_unet(ckpt):
    model = AttentionUNet()
    model.load_state_dict(load_state(ckpt), strict=True)
    return model


def build_deeplab(ckpt):
    import segmentation_models_pytorch as smp
    model = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, classes=NUM_CLASSES)
    model.load_state_dict(load_state(ckpt), strict=True)
    return model


def build_segformer(ckpt):
    # Loaded from the save_pretrained() folder written next to the .pth at the end of
    # training (same weights): from_pretrained() maps the layer names of the
    # transformers version used for training onto the installed one, which a raw
    # load_state_dict of the .pth cannot do.
    from transformers import SegformerForSemanticSegmentation
    hf_dir = os.path.join(os.path.dirname(ckpt), "vessel_run7_segformer_hf")
    hf_model, info = SegformerForSemanticSegmentation.from_pretrained(hf_dir, output_loading_info=True)
    problems = {k: v for k, v in info.items() if k in ("missing_keys", "unexpected_keys", "mismatched_keys") and v}
    if problems:
        raise RuntimeError(f"SegFormer weights did not load cleanly: {problems}")
    return SegformerWrapper(hf_model, (PATCH_SIZE, PATCH_SIZE))


RUNS = [
    ("unetpp_noaug", build_unetpp,
     "vessel_results/kaggle/working/checkpoints/vessel_run1_unetplusplus_augOFF.pth"),
    ("unetpp_aug", build_unetpp,
     "vessel_results/kaggle/working/checkpoints/vessel_run2_unetplusplus_augON.pth"),
    ("attention_unet", build_attention_unet,
     "vessel_results/kaggle/working/checkpoints/vessel_run3_attentionunet_augON.pth"),
    ("deeplabv3plus", build_deeplab,
     "vessel_run4_results/kaggle/working/checkpoints/vessel_run4_deeplabv3plus_augON.pth"),
    ("segformer_b0", build_segformer,
     "vessel_run7_results/kaggle/working/checkpoints/vessel_run7_segformer_augON.pth"),
]


def export(model, out_path):
    model.eval()
    dummy = torch.randn(2, 3, PATCH_SIZE, PATCH_SIZE)
    torch.onnx.export(
        model, (dummy,), out_path,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )


def verify(model, out_path):
    """Compare PyTorch and ONNX Runtime on random input; returns (max |diff|, argmax agreement)."""
    import onnxruntime as ort
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 3, PATCH_SIZE, PATCH_SIZE)).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": x})[0]
    agree = float((ref.argmax(1) == got.argmax(1)).mean())
    return float(np.abs(ref - got).max()), agree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", required=True, help="folder holding the vessel_*_results directories")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "models"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    failed = False
    for name, builder, rel in RUNS:
        ckpt = os.path.join(args.downloads, rel)
        out_path = os.path.join(args.out, f"{name}.onnx")
        try:
            model = builder(ckpt)
            export(model, out_path)
            diff, agree = verify(model, out_path)
            size = os.path.getsize(out_path) / 1e6
            print(f"{name:16s} ok  {size:6.1f} MB  max|diff|={diff:.2e}  argmax agreement={agree:.4%}")
        except Exception as exc:  # keep going so one bad checkpoint doesn't hide the others
            failed = True
            print(f"{name:16s} FAILED: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
