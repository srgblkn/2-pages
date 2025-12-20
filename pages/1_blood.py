import streamlit as st
from PIL import Image
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.preprocessing_blood import preprocess, CLASS_NAMES

# Если у тебя в app.py уже есть st.set_page_config(...),
# то здесь НЕ НАДО вызывать set_page_config.

BLOOD_MODEL_PATH = "pages/model/full_model_blood.pth"


# ----------------------------
# Style
# ----------------------------
st.markdown(
    """
    <style>
      .block-container { max-width: 1200px; padding-top: 1.4rem; padding-bottom: 2rem; }
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}

      .card {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 16px 18px;
        background: rgba(255,255,255,0.03);
      }
      .title { font-size: 32px; font-weight: 800; margin: 0; }
      .sub { margin-top: 6px; opacity: .85; }
      .small { font-size: 12px; opacity: .75; margin-top: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Model loader
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_blood_model():
    model = torch.load(BLOOD_MODEL_PATH, map_location="cpu", weights_only=False)
    model.eval()
    return model


def predict_topk(model, pil_img: Image.Image, k: int):
    x = preprocess(pil_img)  # ожидаем (1, C, H, W)
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
    top_k = st.slider("Top-K классов", 2, min(10, len(CLASS_NAMES)), 5, 1)
    show_probs = st.checkbox("Показывать распределение вероятностей", value=True)
    st.divider()
    st.caption("Совет: лучше загружать чёткие изображения с нормальным освещением.")


# ----------------------------
# Hero
# ----------------------------
st.markdown(
    """
    <div class="card">
      <div class="title">🩸 Классификация изображений крови</div>
      <div class="sub">Загрузите изображение — получите предсказанный класс и Top-K вероятностей.</div>
      <div class="small">Дисклеймер: демонстрационный ML-сервис. Не является медицинским заключением.</div>
    </div>
    """,
    unsafe_allow_html=True
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
            st.metric("Уверенность", f"{conf:.2%}")

            if show_probs:
                try:
                    import pandas as pd
                    df = pd.DataFrame(top)
                    df["Вероятность"] = df["Вероятность"].round(6)

                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.write("**Top-K вероятности**")
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    chart_df = df.set_index("Класс")[["Вероятность"]]
                    st.bar_chart(chart_df, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    st.download_button(
                        "Скачать Top-K как CSV",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name="blood_topk.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                except Exception:
                    st.write("**Top-K вероятности**")
                    for row in top:
                        st.write(f"- {row['Класс']}: {row['Вероятность']:.2%}")
