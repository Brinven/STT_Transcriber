"""
Medical image analysis engine for STT Transcriber.

Uses MedGemma (google/medgemma-1.5-4b-it) for vision-language inference
on medical images (X-rays, CT, MRI, dermatology, etc.).

**Not for clinical diagnosis** — research/development tool only.
"""

import logging
from typing import Optional

from backend.errors import VisionEngineError

logger = logging.getLogger(__name__)

VISION_PRESETS: dict[str, str] = {
    "General": (
        "Describe this medical image. Identify any abnormalities, "
        "notable findings, or areas of concern."
    ),
    "Chest X-Ray": (
        "Analyze this chest X-ray. Evaluate heart size, lung fields, "
        "mediastinum, costophrenic angles, and note any effusions, "
        "opacities, or abnormalities."
    ),
    "CT Scan": (
        "Analyze this CT scan. Evaluate anatomical structures, identify "
        "any lesions, masses, or abnormal contrast enhancement patterns."
    ),
    "MRI": (
        "Analyze this MRI. Evaluate signal characteristics, structural "
        "changes, edema, and any abnormal findings."
    ),
    "Dermatology": (
        "Analyze this dermatological image. Describe the lesion morphology, "
        "borders, color distribution, and provide differential diagnoses."
    ),
    "Fundoscopy": (
        "Analyze this fundoscopic image. Evaluate the optic disc, macula, "
        "retinal vessels, and note any hemorrhages or abnormalities."
    ),
    "Histopathology": (
        "Analyze this histopathology slide. Evaluate tissue architecture, "
        "cell morphology, staining patterns, and any pathological findings."
    ),
}

_MODEL_NAME = "google/medgemma-1.5-4b-it"


class MedGemmaVisionEngine:
    """Medical image analysis using MedGemma vision-language model.

    Uses ``google/medgemma-1.5-4b-it`` (4B params, bf16) for analyzing
    medical images with natural-language queries.

    Args:
        device: ``"auto"`` (CUDA if available, else CPU), ``"cpu"``,
            or ``"cuda"``.
        hf_token: HuggingFace token for gated model access.
    """

    def __init__(self, device: str = "auto", hf_token: str = "") -> None:
        self._device_str = device
        self._hf_token = hf_token
        self._model = None
        self._processor = None
        self._device = None  # resolved torch.device

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        """Download (if needed) and load the MedGemma vision model.

        This is a blocking call — run it on a worker thread.
        First-time use will download from HuggingFace (requires
        accepting the Health AI Developer Foundations terms).

        Raises:
            VisionEngineError: If model loading fails.
        """
        if self._model is not None:
            logger.info("MedGemma vision model already loaded, skipping")
            return

        device_str = self._resolve_device()
        logger.info("Loading MedGemma vision model on device: %s", device_str)

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            kwargs: dict = {}
            if self._hf_token:
                kwargs["token"] = self._hf_token

            self._processor = AutoProcessor.from_pretrained(
                _MODEL_NAME, **kwargs,
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                _MODEL_NAME,
                torch_dtype=torch.bfloat16,
                **kwargs,
            )
            self._device = torch.device(device_str)
            self._model.to(self._device)
            self._model.eval()
            logger.info("MedGemma vision model loaded successfully")
        except Exception as exc:
            self._model = None
            self._processor = None
            raise VisionEngineError(
                f"Failed to load MedGemma vision model: {exc}"
            ) from exc

    def analyze(
        self,
        image: "PIL.Image.Image",  # noqa: F821
        query: str,
        max_new_tokens: int = 2000,
    ) -> str:
        """Analyze a medical image with a text query.

        Args:
            image: PIL Image (RGB).
            query: Natural-language question about the image.
            max_new_tokens: Maximum tokens to generate in the response.

        Returns:
            The model's analysis text.

        Raises:
            VisionEngineError: If the model is not loaded or analysis fails.
        """
        if self._model is None or self._processor is None:
            raise VisionEngineError(
                "Vision model not loaded — call load_model() first"
            )

        try:
            import torch

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": query},
                    ],
                }
            ]

            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                generation = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            text = self._processor.tokenizer.decode(
                generation[0][input_len:],
                skip_special_tokens=True,
            )
            return text.strip()
        except Exception as exc:
            raise VisionEngineError(
                f"Image analysis failed: {exc}"
            ) from exc

    def unload_model(self) -> None:
        """Release the model from memory."""
        if self._model is not None:
            self._model = None
            self._processor = None
            self._device = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("MedGemma vision model unloaded")

    def _resolve_device(self) -> str:
        """Resolve the device string to use for inference."""
        if self._device_str == "auto":
            try:
                import torch
                return "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self._device_str
