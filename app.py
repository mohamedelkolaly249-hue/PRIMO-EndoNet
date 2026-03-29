import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import base64
import time
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# =========================
# CONFIG
# =========================
st.set_page_config(layout="wide", page_title="PRIMO-EndoNet", initial_sidebar_state="expanded")

# =========================
# LOAD HERO BACKGROUND
# =========================
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

hero_bg = get_base64("background2.webp")

# =========================
# GLOBAL STYLE (FINAL)
# =========================
st.markdown(f"""
<style>

/* FONT */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: #0f172a;
    background: #f8fafc;
}}

/* HERO SECTION */
.hero {{
    background-image: url("data:image/webp;base64,{hero_bg}");
    height: 320px;
    border-radius: 20px;
    background-size: cover;
    background-position: center;
    position: relative;
    margin-bottom: 30px;
}}

.hero::before {{
    content: "";
    position: absolute;
    width:100%;
    height:100%;
    background: rgba(0,0,0,0.6);
    border-radius: 20px;
}}

.hero-content {{
    position: absolute;
    top:50%;
    left:50%;
    transform: translate(-50%,-50%);
    text-align:center;
    color:white;
}}

.hero-title {{
    font-size: 48px;
    font-weight: 700;
}}

.hero-sub {{
    font-size: 18px;
    opacity:0.9;
}}

/* WHITE PANELS */
.box {{
    background:white;
    padding:30px;
    border-radius:18px;
    margin-bottom:25px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.08);
}}

/* REPORT */
.report {{
    background:#f8fafc;
    padding:25px;
    border-radius:16px;
    border:1px solid #e2e8f0;
}}

/* STATUS */
.high {{ color:#16a34a; font-weight:600; }}
.medium {{ color:#ca8a04; font-weight:600; }}
.low {{ color:#dc2626; font-weight:600; }}

.section-title {{
    font-size:20px;
    font-weight:600;
    margin-top:15px;
}}

</style>
""", unsafe_allow_html=True)

# =========================
# MODEL
# =========================
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2048, 2)
    )
    model.load_state_dict(torch.load("glenda_model_inference.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# =========================
# FUNCTIONS
# =========================
def predict(image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = model(img)
        prob = torch.softmax(out, dim=1)
    return int(prob[0][1] > 0.5), prob[0][1].item(), prob[0][0].item()

def gradcam(image):
    rgb = np.array(image.resize((224,224))).astype(np.float32)/255.0
    tensor = transform(image).unsqueeze(0)
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    mask = cam(input_tensor=tensor)[0]
    return show_cam_on_image(rgb, mask, use_rgb=True)

def clinical(prob):
    if 0.45 <= prob <= 0.55:
        return "Uncertain","low","Model unstable. Avoid reliance.","Low reliability"
    elif prob > 0.85:
        return "High Confidence","high","Strong pathological evidence.","High reliability"
    elif prob > 0.65:
        return "Moderate Confidence","medium","Possible abnormal patterns.","Moderate reliability"
    else:
        return "Low Risk","low","No strong pathological evidence.","Acceptable reliability"

def attention_strength(prob):
    return "Strong focus" if prob>0.8 else "Moderate focus" if prob>0.6 else "Weak attention"

# =========================
# NAV
# =========================
page = st.sidebar.radio("Navigation", ["Home","Detection","Research"])

# =========================
# HERO HEADER
# =========================
st.markdown(f"""
<div class="hero">
    <div class="hero-content">
        <div class="hero-title">PRIMO-EndoNet</div>
        <div class="hero-sub">AI Clinical Decision Support System</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# HOME
# =========================
if page=="Home":
    st.markdown('<div class="box">', unsafe_allow_html=True)

    st.markdown("## Clinical Background")
    st.markdown("""
Endometriosis is a chronic gynecological condition affecting approximately 10% of women of reproductive age. 
Diagnosis remains challenging due to variability in lesion appearance and reliance on subjective intraoperative interpretation.

Delayed diagnosis significantly impacts patient outcomes, including chronic pelvic pain and infertility.
""")

    st.markdown("## System Overview")
    st.markdown("""
PRIMO-EndoNet is a clinical decision support system designed to assist in detecting endometriosis from laparoscopic images.

The system provides:
- Automated classification (Endometriosis vs Normal)
- Confidence-based interpretation
- Visual explanation of relevant regions
""")

    st.markdown("## Clinical Objective")
    st.markdown("""
The system aims to support surgeons by:

- Improving detection of pathological regions  
- Reducing variability in interpretation  
- Providing objective and reproducible analysis  

The model prioritizes **high sensitivity**, minimizing missed pathological cases.
""")

    st.markdown("## System Workflow")
    st.markdown("""
1. Laparoscopic image acquisition  
2. AI-based image analysis  
3. Prediction (Endometriosis / Normal)  
4. Confidence estimation  
5. Visual explanation (heatmap)  
""")

    st.markdown("## Clinical Considerations")
    st.markdown("""
This system is intended as a **decision-support tool** and does not replace clinical judgment.

Performance depends on image quality and requires further validation before clinical deployment.
""")

    st.markdown('</div>', unsafe_allow_html=True)
# =========================
# DETECTION
# =========================
if page=="Detection":
    st.markdown('<div class="box">', unsafe_allow_html=True)

    file = st.file_uploader("Upload laparoscopic image", type=["jpg","png","jpeg"])

    if file:
        image = Image.open(file).convert("RGB")

        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i+1)

        pred,p_endo,p_norm = predict(image)
        status,level,interp,reliability = clinical(p_endo)

        col1,col2 = st.columns(2)

        with col1:
            st.image(image, use_container_width=True)

        with col2:
            st.markdown('<div class="report">', unsafe_allow_html=True)

            diagnosis = "Endometriosis" if pred else "No Endometriosis"
            st.markdown(f"### Diagnosis: {diagnosis}")

            st.markdown('<div class="section-title">Probability</div>', unsafe_allow_html=True)
            st.progress(p_endo)
            st.write("Endometriosis:", round(p_endo,2))
            st.progress(p_norm)
            st.write("Normal:", round(p_norm,2))

            st.markdown('<div class="section-title">Confidence</div>', unsafe_allow_html=True)
            st.markdown(f'<span class="{level}">{status}</span>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Interpretation</div>', unsafe_allow_html=True)
            st.write(interp)

            st.markdown('<div class="section-title">Recommended Action</div>', unsafe_allow_html=True)
            st.write("Further clinical evaluation advised.")

            st.markdown('<div class="section-title">Reliability</div>', unsafe_allow_html=True)
            st.write(reliability)

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("## Model Explainability")
        cam = gradcam(image)

        c1,c2 = st.columns(2)
        c1.image(image, caption="Original", use_container_width=True)
        c2.image(cam, caption="Attention Map", use_container_width=True)

        st.write("Attention:", attention_strength(p_endo))

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# RESEARCH
# =========================
if page=="Research":
    st.markdown('<div class="box">', unsafe_allow_html=True)

    st.markdown("## Dataset and Experimental Design")
    st.markdown("""
Dataset: GLENDA

- Positive (Endometriosis): 302 images  
- Negative (Normal): 302 images  
- Total: 604 samples  

To ensure reliability:
- Group-aware splitting was applied (no data leakage)
- Dataset was balanced to prevent bias
- Evaluation performed on a held-out test set (n = 102)
""")

    st.markdown("## Diagnostic Performance")

    st.markdown("""
- Accuracy: 0.78  
- F1-score: 0.78  

The confusion matrix summarizes model performance:
""")

    st.image("confusion_matrix.png", use_container_width=True)

    st.markdown("""
### Clinical Interpretation

- True positives: 41 / 46  
- False negatives: 5  

Sensitivity (Recall): **0.89**  
Specificity: **0.70**

The model demonstrates high sensitivity, successfully detecting most endometriosis cases while maintaining acceptable specificity.

This behavior is clinically desirable, as missing disease is more critical than over-detection.
""")

    st.markdown("## Model Reliability")

    st.markdown("""
Robustness analysis under different image conditions:

- Stable under brightness variations  
- Moderate degradation under blur  
- Significant degradation under noise  

This indicates sensitivity to image quality, particularly noise.
""")

    st.markdown("## Confidence Analysis")

    st.markdown("""
Temperature scaling (T ≈ 1.25) was applied.

- Reliable at high and low confidence  
- Unstable at mid-confidence  

This supports the use of uncertainty warnings in clinical use.
""")

    st.markdown("## Explainability")

    st.markdown("""
Grad-CAM visualization shows that the model focuses on lesion-relevant regions.

Causal validation confirms that predictions depend on these regions, supporting interpretability.
""")

    st.markdown("## Key Contributions")

    st.markdown("""
- Leakage-free dataset construction  
- Balanced dataset using real negative samples  
- Clinically meaningful evaluation  
- Robustness analysis  
- Calibration analysis  
- Explainability with causal validation  
""")

    st.markdown("## Limitations")

    st.markdown("""
- Limited dataset size  
- Sensitivity to image noise  
- Requires external clinical validation  
""")

    st.markdown("## Conclusion")

    st.markdown("""
PRIMO-EndoNet demonstrates strong potential as a clinical decision support system, particularly due to its high sensitivity and emphasis on reliability and interpretability.
""")

    st.markdown('</div>', unsafe_allow_html=True)
# =========================
# FOOTER
# =========================
st.markdown('<div class="box">Developed by Ebrahim Adel Elkolaly</div>', unsafe_allow_html=True)