import io
from string import digits
from typing import List, Optional, Tuple

import streamlit as st
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


@st.cache_resource
def load_model(model_name: str = "microsoft/trocr-small-handwritten") -> Tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    """Load and cache the TrOCR processor and model."""
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    return processor, model


def build_bad_words_ids(processor: TrOCRProcessor, allow_digits_only: bool) -> Optional[List[List[int]]]:
    """Return bad_words_ids list to enforce digit-only outputs."""
    if not allow_digits_only:
        return None
    tokenizer = processor.tokenizer
    specials = set(tokenizer.all_special_tokens)
    return [
        [tid]
        for tok, tid in tokenizer.get_vocab().items()
        if tok not in specials and tok not in digits + ' '
    ]


def load_image(uploaded) -> Image.Image:
    """Load a Streamlit-uploaded file or PIL image into an RGB Image."""
    if hasattr(uploaded, "read"):
        data = uploaded.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    return uploaded.convert("RGB")


def main():
    st.set_page_config(page_title="TrOCR Quick Tester", layout="wide")
    st.title("TrOCR Quick Tester")
    st.write("Upload or paste an image, then run TrOCR to transcribe the text.")

    model_name = st.sidebar.selectbox(
        "Model",
        options=[
            "microsoft/trocr-small-handwritten",
            "microsoft/trocr-large-handwritten",
            "microsoft/trocr-small-printed",
            "microsoft/trocr-base-handwritten",
            "microsoft/trocr-base-printed",
        ],
        help="Select a TrOCR checkpoint",
    )
    digits_only = st.sidebar.checkbox("Restrict output to digits", value=False)

    processor, model = load_model(model_name)
    bad_words_ids = build_bad_words_ids(processor, digits_only)

    uploaded = st.file_uploader("Drop an image (PNG/JPG) or paste one", type=["png", "jpg", "jpeg"])
    if not uploaded:
        st.info("Waiting for an image")
        return

    image = load_image(uploaded)
    st.subheader("Input Preview")
    st.image(image, caption="Uploaded image", width='stretch')

    with st.spinner("Running TrOCR"):
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(
            pixel_values,
            bad_words_ids=bad_words_ids,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(generated_text)

    st.subheader("Recognized Text")
    if generated_text:
        st.code(generated_text, language="text")
    else:
        st.warning("No text recognized.")


if __name__ == "__main__":
    main()
