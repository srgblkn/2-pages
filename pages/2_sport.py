import sys
from pathlib import Path
import base64

import streamlit as st
from PIL import Image
import torch

# ----------------------------
# Project root (imports work on Streamlit Cloud)
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Your repo layout: pages/model/...
from pages.model.preprocessing_sport import preprocess, CLASS_NAMES
from pages.model.model_sport import MyResNet  # change only if your class name differs

# ----------------------------
# Paths
# ----------------------------
SPORT_WEIGHTS_PATH = ROOT / "pages" / "model" / "model_weights_sport.pth"

# NOTE: user said "assests" (typo). Support BOTH.
BG_CANDIDATES = [
    ROOT / "pages" / "assests" / "sport.jpg",
    ROOT / "pages" / "assets" / "sport.jpg",
]

# Optional: short hints per class (keys must match CLASS_NAMES exactly)
# If you don't want it, leave the dict empty {}.
CLASS_HINTS = {
    # "football": "Подсказка/описание",
    # "basketball": "Подсказка/описание",
}


# ----------------------------
# Background helper
# ----------------------------
def set_bg():
    bg_path = next((p for p in BG_CANDIDATES if p.exists()), None)
    if bg_path is None:
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

      .pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.04);
        font-size: 12px;
        opacity: .9;
        margin-right: 8px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Model loader (state_dict; robust for common checkpoint formats)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    model = MyResNet(num_classes=len(CLASS_NAMES))

    raw = torch.load(str(SPORT_WEIGHTS_PATH), map_location="cpu")

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
    st.title("🏅 Вид спорта")

    k_max = min(5, len(CLASS_NAMES))
    top_k = st.slider(
        "Сколько вариантов показать (вероятности)",
        min_value=1,
        max_value=k_max,
        value=min(3, k_max),
        step=1,
        help="Показываем вероятности для Top-K наиболее вероятных классов (в процентах).",
    )

    show_probs = st.checkbox("Показывать таблицу и график", value=True)
    show_hints = st.checkbox("Показывать пояснения по предсказанным классам", value=True)

    st.divider()
    st.caption("Совет: лучше работают фото с видимым контекстом (поле/корт/дорожка/форма).")


# ----------------------------
# Hero
# ----------------------------
st.markdown(
    """
    <div class="hero">
      <div class="h-title">🏅 Определение вида спорта по фотографии</div>
      <div class="h-sub">
        Загрузите фото — получите предсказанный спорт и распределение вероятностей по классам (в процентах).
      </div>
      <div class="note">
        Дисклеймер: демонстрационный ML-сервис. Ошибки возможны на нестандартных ракурсах и “смешанных” сценах.
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
    file = st.file_uploader("Загрузите фото (JPG/PNG)", type=["jpg", "jpeg", "png"])
    run = st.button("Распознать", type="primary", use_container_width=True)

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
        st.info("Нажмите «Распознать».")
    elif img is None:
        st.error("Изображение не распознано. Попробуйте другой файл.")
    else:
        model = load_model()

        with st.spinner("Считаю предсказание..."):
            try:
                label, conf, top = predict_topk(model, img, k=top_k)
            except Exception as e:
                st.error(f"Ошибка инференса: {e}")
                top = None

        if top:
            # headline result
            st.success(f"Предсказанный вид спорта: **{label}**")
            st.metric("Уверенность", f"{conf*100:.2f}%")

            # quick “pills” for Top-K
            st.write("")
            pills = " ".join(
                [f"<span class='pill'>{row['Класс']}: {row['Вероятность']*100:.1f}%</span>" for row in top]
            )
            st.markdown(pills, unsafe_allow_html=True)
            st.write("")

            # Optional: show hints only for predicted classes
            if show_hints and CLASS_HINTS:
                predicted = [row["Класс"] for row in top]
                with st.expander("Пояснения по предсказанным классам"):
                    for name in predicted:
                        hint = CLASS_HINTS.get(name)
                        if hint:
                            st.markdown(f"**{name}** — {hint}")

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
                        file_name="sport_probabilities.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                except Exception:
                    st.write("**Распределение вероятностей по классам**")
                    for row in top:
                        st.write(f"- {row['Класс']}: {row['Вероятность']*100:.2f}%")
