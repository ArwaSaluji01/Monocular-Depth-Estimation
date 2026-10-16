import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)


MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class MonocularDepthEstimator:
    def __init__(self):
        self.device = torch.device("cpu")

        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            MODEL_ID
        ).to(self.device)

        self.model.eval()

    @torch.inference_mode()
    def predict(self, image):
        inputs = self.processor(
            images=image,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)

        result = self.processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )

        depth = result[0]["predicted_depth"]

        return depth.squeeze(0).cpu().numpy()