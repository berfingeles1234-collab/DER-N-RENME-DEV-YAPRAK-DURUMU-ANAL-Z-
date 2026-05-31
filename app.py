# -*- coding: utf-8 -*-
"""Streamlit web arayüzü - Bitki Hastalığı Tespiti."""

import streamlit as st
from PIL import Image

from config import MODEL_PATH, RESULTS_DIR
from predict import load_trained_model, predict_image

# Sayfa ayarları
st.set_page_config(
    page_title="Bitki Hastalığı Tespiti",
    page_icon="🌿",
    layout="centered",
)

# Başlık
st.title("🌿 Bitki Hastalığı Tespit Sistemi")
st.markdown(
    """
    Bu uygulama, yüklediğiniz **bitki veya yaprak fotoğrafını** derin öğrenme
    modeli ile analiz eder ve **bitki türünü**, **hastalık durumunu** tahmin eder.
    """
)

# Kenar çubuğu bilgisi
with st.sidebar:
    st.header("ℹ️ Bilgi")
    st.markdown(
        """
        **Nasıl kullanılır?**
        1. Bir yaprak fotoğrafı yükleyin
        2. *Analiz Et* butonuna tıklayın
        3. Sonuçları inceleyin

        **Desteklenen bitkiler:**
        Elma, Domates, Patates, Mısır, Üzüm, Biber ve daha fazlası...
        """
    )

    # Model durumu
    if MODEL_PATH.exists():
        st.success("✅ Model yüklü")
    else:
        st.error("❌ Model bulunamadı")
        st.info("Önce `python train.py` ile modeli eğitin.")

# Fotoğraf yükleme alanı
uploaded_file = st.file_uploader(
    "Bitki veya yaprak fotoğrafı yükleyin",
    type=["jpg", "jpeg", "png", "bmp"],
    help="JPG, PNG veya BMP formatında görsel yükleyebilirsiniz.",
)

if uploaded_file is not None:
    # Yüklenen görseli göster
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Görsel", use_container_width=True)

    # Analiz butonu
    if st.button("🔍 Analiz Et", type="primary", use_container_width=True):
        if not MODEL_PATH.exists():
            st.error("Model dosyası bulunamadı. Lütfen önce modeli eğitin.")
        else:
            with st.spinner("Görsel analiz ediliyor..."):
                try:
                    # Modeli yükle ve tahmin yap
                    model = load_trained_model()
                    result = predict_image(image, model)

                    st.success("Analiz tamamlandı!")

                    # Sonuç kartları
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Bitki Türü", result["bitki_adi"])
                        st.metric("Güven Skoru", f"%{result['guven_skoru']}")

                    with col2:
                        # Durum rengi
                        if result["saglikli"]:
                            st.metric("Durum", "✅ Sağlıklı")
                        else:
                            st.metric("Durum", "⚠️ Hastalıklı")

                        if not result["saglikli"]:
                            st.metric("Hastalık", result["hastalik_adi"])

                    st.divider()

                    # Detaylı bilgiler
                    st.subheader("📋 Detaylı Sonuç")
                    st.write(f"**Tespit Edilen Sınıf:** {result['sinif_adi']}")

                    st.subheader("📖 Hastalık Açıklaması")
                    st.info(result["aciklama"])

                    st.subheader("💡 Öneri")
                    if result["saglikli"]:
                        st.success(result["oneri"])
                    else:
                        st.warning(result["oneri"])

                except Exception as e:
                    st.error(f"Tahmin sırasında hata oluştu: {e}")

else:
    st.info("👆 Başlamak için bir yaprak fotoğrafı yükleyin.")

# Alt bilgi
st.divider()
st.caption(
    "PlantDoc / PlantVillage veri seti ile eğitilmiş MobileNetV2 CNN modeli | "
    "Derin Öğrenme Dersi Projesi"
)

# Eğitim sonuçları varsa göster
graph_path = RESULTS_DIR / "accuracy_loss_graph.png"
if graph_path.exists():
    with st.expander("📊 Model Eğitim Grafikleri"):
        st.image(str(graph_path), caption="Accuracy ve Loss Grafikleri")
