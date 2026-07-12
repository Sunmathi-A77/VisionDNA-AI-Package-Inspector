import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import time

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="VisionDNA AI Inspector",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# Custom CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main{
    background:#f5f7fb;
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#1E3A8A;
}

.subtitle{
    text-align:center;
    color:gray;
    margin-bottom:30px;
}

.block{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,.15);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "models/package_classifier.keras"
    )

    return model

model = load_model()

# ---------------------------------------------------
# Prediction Function
# ---------------------------------------------------

def predict(img):

    img = img.resize((224,224))

    arr = img_to_array(img)

    arr = np.expand_dims(arr, axis=0)

    arr = preprocess_input(arr)

    pred = model.predict(arr, verbose=0)[0][0]

    if pred >= 0.5:

        label = "✅ Intact"

        confidence = pred*100

        recommendation = "Package Accepted"

        color = "green"

    else:

        label = "❌ Damaged"

        confidence = (1-pred)*100

        recommendation = "Package Rejected"

        color = "red"

    return label, confidence, recommendation, color

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

st.sidebar.title("📦 VisionDNA")

st.sidebar.markdown("---")

st.sidebar.success("Industrial Package Inspection")

st.sidebar.info("""
### Model

EfficientNetB0

Transfer Learning

TensorFlow

Computer Vision
""")

st.sidebar.markdown("---")

st.sidebar.write("HackZen 2026")

# ---------------------------------------------------
# Header
# ---------------------------------------------------

st.markdown(
    "<div class='title'>📦 VisionDNA AI Inspector</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Industrial Package Damage Detection using Computer Vision</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# =====================================================
# Upload Image
# =====================================================

uploaded_file = st.file_uploader(
    "📤 Upload Package Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:

        st.markdown("### 📦 Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.markdown("### 🤖 AI Inspection")

        if st.button(
            "🔍 Inspect Package",
            use_container_width=True,
            type="primary"
        ):

            with st.spinner("Analyzing Package..."):

                time.sleep(1)

                label, confidence, recommendation, color = predict(image)

            st.success("Inspection Completed!")

            st.markdown("---")

            if color == "green":

                st.success(label)

            else:

                st.error(label)

            st.metric(

                "Confidence",

                f"{confidence:.2f}%"

            )

            st.progress(
                float(confidence/100)
            )

            st.info(recommendation)

            st.markdown("---")

            st.markdown("### 📊 Prediction Details")

            damaged_prob = 100 - confidence if "Intact" in label else confidence
            intact_prob = confidence if "Intact" in label else 100 - confidence

            metric1, metric2 = st.columns(2)

            metric1.metric(
                "Damaged Probability",
                f"{damaged_prob:.2f}%"
            )

            metric2.metric(
                "Intact Probability",
                f"{intact_prob:.2f}%"
            )

            st.markdown("---")

            if "Damaged" in label:

                st.warning("""
### Recommendation

- Reject Package
- Manual Quality Inspection Recommended
- Possible transportation damage detected.
                """)

            else:

                st.success("""
### Recommendation

- Package Approved
- Ready for Shipment
- No visible external damage detected.
                """)

else:

    st.info("👈 Upload a package image to start inspection.")


