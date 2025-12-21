import sys
from pathlib import Path
import base64

import streamlit as st
from PIL import Image
import torch

# ----------------------------
# Project root (so imports work on Streamlit Cloud)
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Your code is under: pages/model/...
from pages.model.preprocessing_blood import preprocess, CLASS_NAMES
from pages.model.model_blood import MyResNet  # change only if your class name differs

# ----------------------------
# Paths (your current repo layout)
# ----------------------------
BLOOD_WEIGHTS_PATH = ROOT / "pages" / "model" / "model_weights_blood.pth"

# NOTE: you wrote "assests" (typo). I support BOTH variants: pages/assests and pages/assets.
BG_CANDIDATES = [
    ROOT / "pages" / "assests" / "blood.jpg",  # as you wrote
    ROOT / "pages" / "assets" / "blood.jpg",   # common spelling
]

# ----------------------------
# Background helper
# ----------------------------
def set_bg():
    bg_path = next((p for p in BG_CANDIDATES if p.exists()), None)
    if bg_path is None:
        # Optional: uncomment if you want an explicit message
        # st.sidebar.warning("Фон не найден: положи blood.jpg в pages/assets/ или pages/assests/")
        return

    b64 = base64.b64encode(bg_path.read_bytes()).decode("utf-8")

    st.markdown(
        f"""
        <style>
          /* Background image */
          [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpeg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
          }}

          /* Dark overlay for readability */
          [data-testid="stAppViewContainer"]::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.55);
            z-index: 0;
          }}

          /* Keep content above overlay */
          .block-container {{
            position: relative;
            z-index: 1;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_bg()

# ----------------------------
# Modern UI style
# ----------------------------
st.markdown(
    """
    <style>
      .block-container { max-width: 1200px; padding-top: 1.4rem; padding-bottom: 2.2rem; }
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}

      .hero {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 18px 20px;
        background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
        backdrop-filter: blur(6px);
      }
      .h-title { font-size: 32px; font-weight: 820; margin: 0; }
      .h-sub { margin-top: 8px; opacity: .88; line-height: 1.35; }
      .note { font-size: 12px; opacity: .78; margin-top: 10px; }

      .card {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(6px);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Class descriptions (keys must match CLASS_NAMES)
# If your CLASS_NAMES are different (e.g., "Neutrophil" vs "NEUTROPHIL"),
# rename the keys below to match exactly.
# ----------------------------
CLASS_DESCRIPTIONS = {
    "NEUTROPHIL": "Нейтрофилы — клетки врождённого иммунитета; одними из первых приходят к очагу инфекции и уничтожают микробы (в т.ч. фагоцитозом).",
    "MONOCYTE": "Моноциты циркулируют в крови; в тканях могут превращаться в макрофаги/дендритные клетки и участвуют в фагоцитозе и регуляции воспаления.",
    "LYMPHOCYTE": "Лимфоциты — основа адаптивного иммунитета (Т/В-клетки): распознают антигены, координируют ответ, В-клетки участвуют в выработке антител.",
    "EOSINOPHIL": "Эозинофилы важны при паразитарных инфекциях и аллергических реакциях; участвуют в воспалительном ответе.",
}

# ----------------------------
# Model loader (robust for common checkpoint formats)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_blood_model():
    model = MyResNet(num_classes=len(CLASS_NAMES))

    raw = torch.load(str(BLOOD_WEIGHTS_PATH), map_location="cpu")

    # If checkpoint dict -> extract state_dict
    if isinstance(raw, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in raw and isinstance(raw[k], dict):
                raw = raw[k]
                break

    # Remove "module." prefix if saved from DataParallel
    if isinstance(raw, dict):
        raw = {key.replace("module.", "", 1): val for key, val in raw.items()}

    model.load_state_dict(raw, strict=False)


    model.eval()
    return model


def predict_topk(model, pil_img: Image.Image, k: int):
    x = preprocess(pil_img)  # expected: (1, C, H, W)
    with torch.inference_mode():
        logits = model(x)
        prob = torch.softmax(logits, dim=1).squeeze(0)

    k = min(k, prob.numel())
    confs, idxs = torch.topk(prob, k=k)

    top = []
    for c, i in zip(confs.tolist(), idxs.tolist()):
        top.append({"Класс": CLASS_NAMES[i], "Вероятность": float(c)})

    best = top[0]
    return best["Класс"], best["Вероятность"], top


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.title("🩸 Анализ крови")

    k_max = len(CLASS_NAMES)
    top_k = st.slider(
        "Сколько классов показать (распределение вероятностей)",
        min_value=1,
        max_value=k_max,
        value=k_max,
        step=1,
        help="Показываем вероятности по наиболее вероятным классам (в процентах).",
    )

    show_probs = st.checkbox("Показывать таблицу и график", value=True)
    st.divider()
    st.caption("Совет: лучше работают чёткие изображения без сильной компрессии и смаза.")


# ----------------------------
# Hero
# ----------------------------
st.markdown(
    """
    <div class="hero">
      <div class="h-title">🩸 Классификация лейкоцитов по изображению</div>
      <div class="h-sub">
        Загрузите изображение — получите предсказанный класс и распределение вероятностей по классам (в процентах).
      </div>
      <div class="note">
        Дисклеймер: демонстрационный ML-сервис. Не является медицинским заключением.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")


# ----------------------------
# Layout
# ----------------------------
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.subheader("Ввод")
    file = st.file_uploader("Загрузите изображение (JPG/PNG)", type=["jpg", "jpeg", "png"])
    run = st.button("Запустить анализ", type="primary", use_container_width=True)

    img = None
    if file:
        try:
            img = Image.open(file).convert("RGB")
            st.image(img, caption="Загруженное изображение", use_container_width=True)
        except Exception as e:
            st.error(f"Не удалось прочитать изображение: {e}")

with right:
    st.subheader("Результат")

    if not file:
        st.info("Загрузите изображение слева.")
    elif not run:
        st.info("Нажмите «Запустить анализ».")
    elif img is None:
        st.error("Изображение не распознано. Попробуйте другой файл.")
    else:
        model = load_blood_model()

        with st.spinner("Считаю предсказание..."):
            try:
                label, conf, top = predict_topk(model, img, k=top_k)
            except Exception as e:
                st.error(f"Ошибка инференса: {e}")
                top = None

        if top:
            st.success(f"Предсказанный класс: **{label}**")
            st.metric("Уверенность", f"{conf*100:.2f}%")
            st.write("")

            # Help only for predicted Top-K classes
            predicted = [row["Класс"] for row in top]
            with st.expander("Справка по предсказанным классам"):
                for name in predicted:
                    desc = CLASS_DESCRIPTIONS.get(name, "Описание для этого класса не задано.")
                    st.markdown(f"**{name}** — {desc}")

            if show_probs:
                try:
                    import pandas as pd
                    import altair as alt

                    df = pd.DataFrame(top)
                    df["Вероятность, %"] = (df["Вероятность"] * 100).round(2)
                    df = df.drop(columns=["Вероятность"])

                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.write("**Распределение вероятностей по классам**")
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    chart = (
                        alt.Chart(df)
                        .mark_bar()
                        .encode(
                            x=alt.X("Вероятность, %:Q", title="Вероятность, %"),
                            y=alt.Y("Класс:N", sort="-x", title=""),
                            tooltip=["Класс:N", alt.Tooltip("Вероятность, %:Q", format=".2f")],
                        )
                    )
                    st.altair_chart(chart, use_container_width=True)

                    st.download_button(
                        "Скачать вероятности (CSV)",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name="blood_probabilities.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                except Exception:
                    st.write("**Распределение вероятностей по классам**")
                    for row in top:
                        st.write(f"- {row['Класс']}: {row['Вероятность']*100:.2f}%")
