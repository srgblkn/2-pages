import streamlit as st
from PIL import Image
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.preprocessing_sport import preprocess, CLASS_NAMES

# ----------------------------
# Page config (если это файл-страница в pages/, можно убрать и оставить только в app.py)
# Если у тебя уже есть set_page_config в app.py — УДАЛИ этот блок отсюда.
# ----------------------------
# st.set_page_config(page_title="Вид спорта по фото", page_icon="🏅", layout="wide")


# ----------------------------
# Лёгкий стиль (карточки/типографика)
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
# Model loader (кэшируется)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    model = torch.load("model/full_model_sport.pth", map_location="cpu", weights_only=False)
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
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.title("🏅 Вид спорта")
    top_k = st.slider("Top-K классов", 2, min(10, len(CLASS_NAMES)), 5, 1)
    show_probs = st.checkbox("Показывать распределение вероятностей", value=True)
    st.divider()
    st.caption("Подсказка: лучше работают чёткие фото без сильной компрессии.")


# ----------------------------
# Hero
# ----------------------------
st.markdown(
    """
    <div class="card">
      <div class="title">🏅 Определение вида спорта по фотографии</div>
      <div class="sub">Загрузите изображение — получите предсказанный спорт и Top-K вероятностей.</div>
      <div class="small">Дисклеймер: демонстрационный ML-сервис, возможны ошибки на нетипичных фото.</div>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")


# ----------------------------
# Layout: input / output
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
        # модель грузим только при реальном запуске
        model = load_model()

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
                # Таблица + график без жёсткой зависимости от pandas
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
                        file_name="sport_topk.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                except Exception:
                    st.write("**Top-K вероятности**")
                    for row in top:
                        st.write(f"- {row['Класс']}: {row['Вероятность']:.2%}")
