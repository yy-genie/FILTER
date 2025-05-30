import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
import requests

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_TAG = "v1.0"  # Release 생성 시 쓰신 태그명
MODEL_URL = f"https://github.com/yy-genie/FILTER/releases/download/{MODEL_TAG}/yjmodel.pth"
MODEL_PATH = "yjmodel.pth"

@st.cache_resource(show_spinner=False)
def load_model_from_release():
    # 1) 로컬에 없으면 다운로드
    if not os.path.exists(MODEL_PATH):
        with st.spinner("모델 다운로드 중…"):
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=1_000_000):
                    f.write(chunk)
    # 2) 로드
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval().to(device)
    return model

# 기존 load_model() 대신
model = load_model_from_release()
st.markdown(
    """
    <div style="display: flex; align-items: baseline; gap: 0.5rem;">
      <h1 style="margin: 0;">FILTER</h1>
      <span style="color: #6c6c6c; font-size:1.25rem;">Image Security</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
      /* 사이드바 전체 배경색 변경 */
      [data-testid="stSidebar"] {
        background-color: #0d1f44;
      }
      /* 사이드바 안의 모든 글자 흰색으로 */
      [data-testid="stSidebar"] * {
        color: white !important;
      }
      /* 드롭다운, 입력창 배경·테두리도 어둡게 (선택사항) */
      [data-testid="stSidebar"] .stSelectbox>div>div {
        background-color: #0d1f44;
        color: white;
      }
      [data-testid="stSidebar"] .stSelectbox>div>div::after {
        border-color: white;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

eps_list    = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05,
               0.06, 0.07, 0.08, 0.09, 0.1]
target_ssim = 0.95
pgd_steps   = 10

@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load('yjmodel.pth', map_location=device))
    model.eval()
    model.to(device)
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                         std=(0.2675, 0.2565, 0.2761))
])

# 함수들
def unnormalize(tensor):
    means = [0.5071, 0.4867, 0.4408]
    stds  = [0.2675, 0.2565, 0.2761]
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(means, stds)],
        std=[1/s for s in stds]
    )
    return inv_normalize(tensor)

def adaptive_pgd_attack(model, input_batch, orig_pred,
                        eps_list=eps_list, k=pgd_steps, target_ssim=target_ssim):
    """  
    input_batch: (1,C,H,W) 정규화된 텐서  
    orig_pred: 원본 예측 라벨(int)  
    return: (adv_tensor, selected_eps, achieved_ssim) or (None,None,None)
    """
    means = [0.5071, 0.4867, 0.4408]
    stds  = [0.2675, 0.2565, 0.2761]
    inv_norm = transforms.Normalize(
        mean=[-m/s for m,s in zip(means,stds)],
        std=[1/s   for s   in stds]
    )
    # 원본 이미지
    orig_img = inv_norm(input_batch[0].cpu()).clamp(0,1).permute(1,2,0).numpy()
    
    for eps in eps_list:
        alpha = eps / k
        x_adv = input_batch.clone().detach().to(device)
        
        # PGD 반복
        for _ in range(k):
            x_adv.requires_grad_()
            out = model(x_adv)
            loss = F.cross_entropy(out, torch.tensor([orig_pred],device=device))
            model.zero_grad(); loss.backward()
            grad_sign = x_adv.grad.sign()
            
            # step & project
            x_adv = x_adv + alpha * grad_sign
            x_adv = torch.max(torch.min(x_adv, input_batch + eps),
                              input_batch - eps)
            # clamp
            for c in range(x_adv.shape[1]):
                lo = (0 - means[c]) / stds[c]
                hi = (1 - means[c]) / stds[c]
                x_adv[0,c].clamp_(lo, hi)
            x_adv = x_adv.detach()
        
        # 성공 여부 확인
        final_pred = model(x_adv).argmax(1).item()
        if final_pred == orig_pred:
            continue
        
        adv_img = inv_norm(x_adv[0].cpu()).clamp(0,1).permute(1,2,0).numpy()
        s = ssim(
            orig_img,
            adv_img,
            channel_axis=2,     
            win_size=3,          
            data_range=1.0
        )
        if s >= target_ssim:
            return x_adv, eps, s
    
    return None, None, None
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "Home"

menu_styles = {
    "container": {"padding": "0!important", "background-color": "#FFFFFF"},
    "nav-link": {
        "font-size": "16px",
        "color": "#0d1f44",            
    },
    "nav-link-selected": {
        "background-color": "#09097D",
        "color": "white",
    },
}

# 상단 메뉴
selected = option_menu(
    menu_title=None,
    options=["Home", "Upload", "Contact"],
    icons=["house", "cloud-upload", "envelope"],
    default_index=["Home", "Upload", "Contact"].index(st.session_state.page),
    orientation="horizontal",
    key="top_menu",
    on_change=lambda key: st.session_state.update(page=st.session_state.top_menu),
    styles=menu_styles
)

# 각 페이지
def show_home():
    st.markdown("## 🏠 Home")
    st.markdown("##### 대충 그림이랑 설명.. 들어갈 자리!!")

def show_upload():
    st.markdown("## ☁️ Upload")
    uploaded_file = st.file_uploader("이미지를 업로드하세요.", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        pil = Image.open(uploaded_file).convert("RGB")
        inp = transform(pil).unsqueeze(0).to(device)
        orig_pred = model(inp).argmax(1).item()

        # 공격 실행
        adv_tensor, used_eps, achieved_ssim = adaptive_pgd_attack(model, inp, orig_pred)

        def tensor_to_np(t: torch.Tensor):
            return (
                unnormalize(t.cpu())
                .clamp(0, 1)
                .permute(1, 2, 0)
                .numpy()
            )

        col1, col2 = st.columns(2)
        with col1:
            orig_np = tensor_to_np(inp[0])
            st.image(
                orig_np,
                caption="원본",
                use_container_width=True
            )
        with col2:
            if adv_tensor is None:
                st.warning("이미지 처리 오류")
            else:
                adv_np = tensor_to_np(adv_tensor[0])
                st.image(
                    adv_np,
                    caption=f"기능 적용 (ε={used_eps:.3f}, SSIM={achieved_ssim:.3f})",
                    use_container_width=True
                )

                # 다운로드
                sub1, sub2, sub3 = st.columns([1,1,1])
                with sub2:
                    buf = io.BytesIO()
                    Image.fromarray((adv_np * 255).astype("uint8"))\
                         .save(buf, format="JPEG")
                    st.download_button(
                        "Download",
                        buf.getvalue(),
                        file_name="result.jpg",
                        mime="image/jpeg"
                    )


def show_contact():
    st.markdown("## ✉️ Contact")
    st.caption("##### 문의 가능 시간 : 09:00~18:00")
    st.markdown("##### Phone : 010-5675-9309")
    st.markdown("##### Email : eu9309@pukyong.ac.kr")


if st.session_state.page == "Home":
    show_home()
elif st.session_state.page == "Upload":
    show_upload()
else:
    show_contact()

# 사이드 바
with st.sidebar:
    st.image("logo.png", width=100)
    st.markdown("# FILTER")
    st.caption("##### Image Security")
    st.markdown("---")

    lang = st.sidebar.selectbox(
    "🌐 language",                  
    ["KR 한국어"],
    index=0,
    key="language"
)

if st.session_state.language.startswith("KR"):
    greeting = "한국어"
