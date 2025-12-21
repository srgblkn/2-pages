import sys
from pathlib import Path

import streamlit as st
from PIL import Image
import torch

# Чтобы импорты работали на Streamlit Cloud независимо от рабочей директории
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# ВАЖНО: у тебя папка model лежит внутри pages/model (по твоей структуре)
from pages.model.preprocessing_blood import preprocess, CLASS_NAMES
from pages.model.model_blood import MyResNet  # если класс называется иначе — поменяй здесь

# Пути к весам (у тебя они лежат в pages/model/)
BLOOD_WEIGHTS_PATH = str(ROOT / "pages" / "model" / "model_weights_blood.pth")

# Описания классов (общие, человеческим языком)
CLASS_DESCRIPTIONS = {
    "NEUTROPHIL": "Нейтрофилы — ключевые клетки врождённого иммунитета: одни из первых приходят к очагу инфекции и уничтожают микробы (в т.ч. через фагоцитоз).",
    "MONOCYTE": "Моноциты циркулируют в крови и при воспалении мигрируют в ткани, где могут превращаться в макрофаги/дендритные клетки; участвуют в фагоцитозе и регуляции воспаления.",
    "LYMPHOCYTE": "Лимфоциты — основа адаптивного иммунитета (Т- и В-клетки): распознают антигены; В-клетки участвуют в выработке антител, Т-клетки координируют ответ и могут уничтожать инфицированные клетки.",
    "EOSINOPHIL": "Эозинофилы важны при паразитарных инфекциях и аллергических реакциях; участвуют в воспалительном ответе.",
}

# ----------------------------
# Page style (современный, но лёгкий)
# ----------------------------
st.markdown(
    """
    <style>
      .block-container { max-width: 1200px; padding-top: 1.4rem; padding-bottom: 2.2rem; }
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}

      .hero {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 18px 20px;
        background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
      }
      .h-title { font-size: 32px; font-weight: 820; margin: 0; }
      .h-sub { margin-top: 8px; opacity: .86; line-height: 1.35; }
      .note { font-size: 12px; opacity: .75; margin-top: 10px; }

      .card {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.03);
      }

      .chip {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.03);
        font-size: 12px;
        opacity: .92;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Model loader (state_dict, чтобы не ломалось из-за pickle)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_blood_model():
    model = MyResNet(num_classes=len(CLASS_NAMES))
    state = torch.load(BLOOD_WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict_topk(model, pil_img: Image.Image, k: int):
    x = preprocess(pil_img)  # (1, C, H, W)
    with torch.inference_mode():
        logits = model(x)
        prob = torch.softmax(logits, dim=1).squeeze(0)  # (num_classes,)

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
        value=k_max,   # логично показать все 4 из 4
        step=1,
        help="Показываем вероятности по наиболее вероятным классам.",
    )

    show_probs = st.checkbox("Показывать таблицу и график", value=True)
    st.divider()
    st.caption("Совет: используйте чёткие изображения (без сильной компрессии и смаза).")

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

    st.write("")
    with st.expander("Справка по классам"):
        for name in CLASS_NAMES:
            desc = CLASS_DESCRIPTIONS.get(name, "Описание для этого класса не задано.")
            st.markdown(f"**{name}** — {desc}")

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
                    # Fallback без pandas/altair
                    st.write("**Распределение вероятностей по классам**")
                    for row in top:
                        st.write(f"- {row['Класс']}: {row['Вероятность']*100:.2f}%")
