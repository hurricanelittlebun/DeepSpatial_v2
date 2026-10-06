"""Frozen UNI2-h only. Official architecture/preprocess: MahmoodLab/UNI2-h.

Weights require the user's own approved access. Never downloads during training.
"""

import torch
from torchvision import transforms


class UNI2Encoder:
    feature_dim = 1536

    def __init__(self, checkpoint=None, device="cuda"):
        import timm

        kwargs = dict(
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1e-5,
            embed_dim=1536,
            mlp_ratio=2.66667 * 2,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
        )
        if checkpoint is None:
            from timm.data import resolve_data_config, create_transform

            self.model = timm.create_model(
                "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kwargs
            )
            self.transform = create_transform(
                **resolve_data_config(self.model.pretrained_cfg, model=self.model)
            )
        else:
            self.model = timm.create_model(
                "vit_giant_patch14_224", pretrained=False, **kwargs
            )
            self.model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True),
                strict=True,
            )
            self.transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        self.device = torch.device(device)
        self.model.requires_grad_(False).eval().to(self.device)

    @torch.inference_mode()
    def encode(self, patches, batch_size=32):
        """PIL RGB patches -> CPU float32 [N,1536]; no autograd or fine-tuning."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model.eval()
        results = []
        for start in range(0, len(patches), batch_size):
            batch = torch.stack(
                [
                    self.transform(p.convert("RGB"))
                    for p in patches[start : start + batch_size]
                ]
            )
            out = self.model(batch.to(self.device))
            if out.ndim != 2 or out.shape[1] != self.feature_dim:
                raise ValueError("Expected UNI2-h CLS embeddings [B,1536]")
            results.append(out.float().cpu())
        return torch.cat(results) if results else torch.empty((0, self.feature_dim))
