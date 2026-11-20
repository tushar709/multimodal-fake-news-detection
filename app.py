import streamlit as st
import torch
import numpy as np
import joblib
import re
import pandas as pd
import shap
import pandas as pd

from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModel,
    CLIPModel,
    CLIPProcessor
)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="🔎",
    layout="centered"
)

# ---------------- SIMPLE UI STYLE ----------------

st.markdown("""
<style>
.main {
    max-width: 900px;
    padding-top: 2rem;
}
h1 {
    text-align: center;
    color: #1E293B;
}
.subtitle {
    text-align: center;
    color: #64748B;
    margin-bottom: 30px;
}
.stButton > button {
    width: 100%;
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
}
.result {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DEVICE ----------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ---------------- LOAD MODELS ----------------

@st.cache_resource
def load_models():

    classifier = joblib.load(
        "models/corrected_text_image_domain_model.pkl"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    text_model = AutoModel.from_pretrained(
        "distilbert-base-uncased"
    ).to(DEVICE)

    text_model.eval()

    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(DEVICE)

    clip_model.eval()

    return (
        classifier,
        tokenizer,
        text_model,
        clip_processor,
        clip_model
    )


# ---------------- TEXT FEATURE EXTRACTION ----------------

def extract_text_features(text, tokenizer, text_model):

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.inference_mode():

        outputs = text_model(**encoded)

        features = outputs.last_hidden_state[:, 0, :]

    return features.cpu().numpy()


# ---------------- IMAGE FEATURE EXTRACTION ----------------

def extract_image_features(image, processor, model):

    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.inference_mode():

        vision_output = model.vision_model(
            pixel_values=pixel_values
        )

        features = model.visual_projection(
            vision_output.pooler_output
        )

        features = features / features.norm(
            dim=-1,
            keepdim=True
        )

    return features.cpu().numpy()


# ---------------- DOMAIN CATEGORY ----------------

IMAGE_HOSTING = {
    "i.redd.it",
    "i.imgur.com",
    "imgur.com",
    "i.reddituploads.com",
    "v.redd.it"
}

NEWS_PUBLISHERS = {
    "reuters.com",
    "theguardian.com",
    "bbc.com",
    "bbc.co.uk",
    "washingtonpost.com",
    "cnn.com",
    "nytimes.com",
    "cbc.ca",
    "independent.co.uk",
    "foxnews.com"
}


def get_domain_category(domain):

    domain = domain.strip().lower()
    domain = re.sub(r"^https?://", "", domain)
    domain = domain.split("/")[0]
    domain = domain.split(":")[0]
    domain = domain.removeprefix("www.")

    if not domain:
        return "Unknown"

    if domain in IMAGE_HOSTING:
        return "Image Hosting"

    if domain in NEWS_PUBLISHERS:
        return "News Publisher"

    return "other"


def extract_domain_features(domain):

    category = get_domain_category(domain)

    categories = [
        "Image Hosting",
        "News Publisher",
        "Unknown",
        "other"
    ]

    features = [
        1.0 if category == item else 0.0
        for item in categories
    ]

    return np.array([features], dtype=np.float32), category





# ---------------- SHAP EXPLAINABLE AI ----------------

@st.cache_resource
def load_shap_explainer():

    classifier = joblib.load(
        "models/corrected_text_image_domain_model.pkl"
    )

    # The zero background represents the mean
    # after StandardScaler transformation.
    explainer = shap.LinearExplainer(
        classifier.named_steps["classifier"],
        np.zeros((1, 1284))
    )

    return explainer


def explain_modalities(combined_features, classifier):

    # Apply the same scaling used during model training
    scaled_features = classifier.named_steps[
        "scaler"
    ].transform(combined_features)

    explainer = load_shap_explainer()

    shap_values = explainer.shap_values(
        scaled_features
    )

    shap_values = np.asarray(shap_values).reshape(-1)

    # Group SHAP values by modality
    contributions = {
        "Text": float(np.sum(shap_values[:768])),
        "Image": float(np.sum(shap_values[768:1280])),
        "Domain": float(np.sum(shap_values[1280:1284]))
    }

    return contributions

# ---------------- VERIFICATION & MITIGATION ----------------

def show_verification_guidance(prediction):

    st.divider()
    st.subheader("🛡️ Verification & Mitigation")

    if int(prediction) == 1:

        st.warning(
            "The model flagged this news as potentially fake. "
            "Do not share it until you verify the claim."
        )

        st.markdown("**Recommended actions:**")

        st.markdown("""
        1. Check whether the same claim appears in official sources.
        2. Compare the information with multiple independent news outlets.
        3. Check the original publication date and context.
        4. Reverse-search the image to identify older or unrelated uses.
        5. Consult independent fact-checking organizations.
        """)

    else:

        st.success(
            "The model classified this news as Real. "
            "Independent verification is still recommended."
        )

        st.markdown("**Recommended actions:**")

        st.markdown("""
        1. Check the original source of the information.
        2. Confirm important claims through official announcements.
        3. Compare the report with other independent news sources.
        4. Check whether the image matches the reported event.
        5. Verify the publication date and surrounding context.
        """)

    st.info(
        "These are general verification suggestions. "
        "The application does not independently confirm the "
        "truth of a news story."
    )

# ---------------- PREDICTION ----------------

def predict_news(
    text,
    image,
    domain,
    classifier,
    tokenizer,
    text_model,
    clip_processor,
    clip_model
):

    text_features = extract_text_features(
        text,
        tokenizer,
        text_model
    )

    image_features = extract_image_features(
        image,
        clip_processor,
        clip_model
    )

    domain_features, category = extract_domain_features(
        domain
    )

    combined_features = np.concatenate(
        [
            text_features,
            image_features,
            domain_features
        ],
        axis=1
    )

    if combined_features.shape[1] != 1284:
        raise ValueError(
            f"Expected 1284 features, got {combined_features.shape[1]}"
        )

    prediction = classifier.predict(
        combined_features
    )[0]

    probabilities = classifier.predict_proba(
        combined_features
    )[0]

    class_probabilities = {
        int(label): float(prob)
        for label, prob in zip(
            classifier.classes_,
            probabilities
        )
    }

    return prediction, class_probabilities, category, combined_features


# ---------------- MAIN INTERFACE ----------------

st.title("🔎 Fake News Detection")

st.markdown(
    '<p class="subtitle">Multimodal AI-powered news verification</p>',
    unsafe_allow_html=True
)

st.markdown("### Enter News Details")

news_text = st.text_area(
    "News headline or text",
    placeholder="Paste the news headline or article text here...",
    height=140
)

uploaded_image = st.file_uploader(
    "Upload news image",
    type=["jpg", "jpeg", "png"]
)

domain = st.text_input(
    "Source domain (optional)",
    placeholder="Example: bbc.com"
)

st.caption(
    "Source category is based on the domain lists used during model training. "
    "It is not an independently verified credibility score."
)

analyze = st.button("Analyze News")

# ---------------- ANALYSIS ----------------

if analyze:

    if not news_text.strip():
        st.warning("Please enter a news headline or text.")

    elif uploaded_image is None:
        st.warning("Please upload a news image.")

    else:

        try:

            with st.spinner(
                "Analyzing text, image and source information..."
            ):

                (
                    classifier,
                    tokenizer,
                    text_model,
                    clip_processor,
                    clip_model
                ) = load_models()

                image = Image.open(uploaded_image)

                prediction, probabilities, category ,  combined_features = predict_news(
                    news_text,
                    image,
                    domain,
                    classifier,
                    tokenizer,
                    text_model,
                    clip_processor,
                    clip_model
                )

            st.divider()

            st.subheader("Analysis Result")

            # Assumes the project's label mapping:
            # 0 = Real, 1 = Fake

            if int(prediction) == 1:

                st.markdown(
                    """
                    <div class="result"
                         style="background-color:#FEE2E2;">
                        <h2 style="color:#B91C1C;">⚠️ Fake News</h2>
                        <p style="color:#7F1D1D;">
                            The model classified this news as Fake.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="result"
                         style="background-color:#DCFCE7;">
                        <h2 style="color:#15803D;">✅ Real News</h2>
                        <p style="color:#166534;">
                            The model classified this news as Real.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            real_probability = probabilities.get(0, 0.0) * 100
            fake_probability = probabilities.get(1, 0.0) * 100

            st.markdown("### Prediction Confidence")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Real News",
                    f"{real_probability:.2f}%"
                )

            with col2:
                st.metric(
                    "Fake News",
                    f"{fake_probability:.2f}%"
                )

            st.markdown("### Source Information")

            st.write(
                f"Detected source category: **{category}**"
            )


# ---------------- SHAP EXPLANATION DISPLAY ----------------

            st.divider()
            st.subheader("🤖 Explainable AI - SHAP Analysis")

            with st.spinner("Generating explanation..."):

                contributions = explain_modalities(
                combined_features,
                classifier
            )

            st.write(
                "The chart shows how each modality influenced "
                "the model's Fake News prediction."
            )



            shap_df = pd.DataFrame({
                "Modality": list(contributions.keys()),
                "SHAP Contribution": list(contributions.values())
            })

            st.bar_chart(
                shap_df.set_index("Modality")
            )

            for modality, value in contributions.items():

                if value > 0:
                    direction = "Towards Fake News"
                elif value < 0:
                    direction = "Towards Real News"
                else:
                    direction = "Neutral"

                st.write(
                    f"**{modality}:** {value:+.5f} — {direction}"
                )
            
            st.caption(
                "Positive SHAP values push towards Fake (class 1); "
                "negative values push towards Real (class 0). "
                "These are contribution scores, not percentages."
            )

            show_verification_guidance(prediction)

            st.caption(
                "These probabilities are model outputs, not proof that "
                "a news story is true or false. Verify important claims "
                "using reliable independent sources."
            )

        except Exception as error:

            st.error("Something went wrong during analysis.")

            st.exception(error)
