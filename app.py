import streamlit as st

# 1) Конфиг страницы — только здесь
st.set_page_config(
    page_title="ML Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2) Мини-стиль (без отдельных файлов)
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
      .title { font-size: 34px; font-weight: 800; margin: 0; }
      .sub { margin-top: 6px; opacity: .85; }
      .small { font-size: 12px; opacity: .75; margin-top: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

# 3) Sidebar — только навигация/инфо (без моделей)
with st.sidebar:
    st.title("ML Suite")
    st.caption("Два приложения в одном интерфейсе.")
    st.divider()
    st.caption("Страницы:")
    # Эти ссылки заработают, когда ты создашь файлы в папке pages/
    st.page_link("pages/1_blood.py", label="🩸 Анализ крови", icon="🩸")
    st.page_link("pages/2_sport.py", label="🏅 Вид спорта по фото", icon="🏅")

# 4) Главная (landing page)
st.markdown(
    """
    <div class="card">
      <div class="title">🧠 ML Suite</div>
      <div class="sub">
        Единое приложение с двумя независимыми страницами:
        <b>анализ крови</b> и <b>определение вида спорта по фотографии</b>.
      </div>
      <div class="small">
        Примечание: модели намеренно не загружаются на этой странице — только на соответствующих страницах.
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

c1, c2, c3 = st.columns(3, gap="large")
with c1:
    with st.container(border=True):
        st.subheader("🩸 Анализ крови")
        st.write("Загрузка изображения → предсказание класса → Top-K вероятностей.")
        st.page_link("pages/1_blood.py", label="Открыть страницу", use_container_width=True)

with c2:
    with st.container(border=True):
        st.subheader("🏅 Вид спорта по фото")
        st.write("Загрузка фото → определение спорта → Top-K вероятностей.")
        st.page_link("pages/2_sport.py", label="Открыть страницу", use_container_width=True)

with c3:
    with st.container(border=True):
        st.subheader("ℹ️ О проекте")
        st.write("Описание данных, ограничения, дисклеймеры, версия модели.")
        st.info("Добавим позже отдельной вкладкой или страницей.")
