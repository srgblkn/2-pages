import base64
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Image Classification Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent

# фон для стартовой страницы: поддерживаем и pages/assests и pages/assets
BG_CANDIDATES = [
    ROOT / "pages" / "assests" / "phon.jpg",  # как у тебя папка написана
    ROOT / "pages" / "assets" / "phon.jpg",   # на всякий случай
    ROOT / "assests" / "phon.jpg",
    ROOT / "assets" / "phon.jpg",
]


def set_bg():
    bg_path = next((p for p in BG_CANDIDATES if p.exists()), None)
    if bg_path is None:
        return

    b64 = base64.b64encode(bg_path.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <style>
          [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpeg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
          }}

          /* тёмная подложка для читаемости */
          [data-testid="stAppViewContainer"]::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.70);
            z-index: 0;
          }}

          .block-container {{
            position: relative;
            z-index: 1;
            max-width: 1200px;
            padding-top: 1.4rem;
            padding-bottom: 2.2rem;
          }}

          #MainMenu {{visibility: hidden;}}
          footer {{visibility: hidden;}}
          header {{visibility: hidden;}}

          .hero {{
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 18px;
            padding: 18px 20px;
            background: rgba(0,0,0,0.35);
            backdrop-filter: blur(8px);
          }}
          .h-title {{ font-size: 34px; font-weight: 820; margin: 0; }}
          .h-sub {{ margin-top: 8px; opacity: .90; line-height: 1.4; }}

          .card {{
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 18px;
            padding: 14px 16px;
            background: rgba(0,0,0,0.30);
            backdrop-filter: blur(8px);
          }}

          [data-testid="stAlert"] {{
            background: rgba(0,0,0,0.35) !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


set_bg()

# ----------------------------
# Sidebar navigation (убираем дубли смайлов: эмодзи только в icon=)
# ----------------------------
with st.sidebar:
    st.title("Навигация")
    st.caption("Выберите модуль сервиса.")
    st.divider()
    st.caption("Страницы:")

    st.page_link("pages/1_blood.py", label="Анализ лейкоцитов", icon="🩸")
    st.page_link("pages/2_sport.py", label="Вида спорта по фото", icon="🏅")

# ----------------------------
# Home content
# ----------------------------
st.markdown(
    """
    <div class="hero">
      <div class="h-title">Демонстрационный сервис классификации изображений</div>
      <div class="h-sub">
        Два независимых модуля: распознавание типа лейкоцита и определение вида спорта по фотографии.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

c1, c2, c3 = st.columns(3, gap="large")

with c1:
    with st.container(border=True):
        st.subheader("🩸 Анализ лейкоцитов")
        st.write(
            "Загрузка изображения → автоматическая классификация → распределение вероятностей по классам."
        )
        st.page_link("pages/1_blood.py", label="Открыть модуль", use_container_width=True)

with c2:
    with st.container(border=True):
        st.subheader("🏅 Вид спорта по фото")
        st.write(
            "Загрузка фотографии → классификация вида спорта → распределение вероятностей по классам."
        )
        st.page_link("pages/2_sport.py", label="Открыть модуль", use_container_width=True)

with c3:
    with st.container(border=True):
        st.subheader("ℹ️ Примечания")
        st.write(
            "Результаты являются демонстрационными и зависят от качества исходного изображения "
            "(ракурс, освещение, шум/сжатие)."
        )
        st.info("Для лучшего результата используйте чёткие изображения с понятным контекстом.", icon="✅")

st.write("")

with st.container(border=True):
    st.subheader("Авторы")
    st.write('Работу выполнили студенты «Эльбруса»: Якунова Елена, Хрипун Евгений и Белькин Сергей.')
