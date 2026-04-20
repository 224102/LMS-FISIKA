import streamlit as st
from ai_engine import siapkan_ai_fisika # Mengambil otak AI dari file sebelah

# 1. Konfigurasi Halaman Web
st.set_page_config(page_title="TanyaFisika AI", page_icon="🍎", layout="centered")
st.title("🤖 Asisten Guru Fisika (Halliday)")
st.caption("Tanyakan materi fisika apa saja berdasarkan buku teks Halliday!")

# 2. Fitur Cache (SANGAT PENTING)
# Agar Streamlit tidak membaca ulang database setiap kali kita mengetik
@st.cache_resource(show_spinner="Menyiapkan memori buku Fisika...")
def load_ai():
    # Sesuaikan dengan lokasi file PDF Halliday Anda
    PATH_PDF = "../data/buku halliday.pdf" 
    return siapkan_ai_fisika(PATH_PDF)

chatbot = load_ai()

# 3. Menyiapkan Memori Obrolan di Layar
if "riwayat_chat" not in st.session_state:
    st.session_state.riwayat_chat = []

# Tampilkan obrolan sebelumnya di layar
for chat in st.session_state.riwayat_chat:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# 4. Kotak Input untuk Siswa
pertanyaan = st.chat_input("Ketik pertanyaan fisika di sini...")

if pertanyaan:
    # Tampilkan pertanyaan siswa di layar
    with st.chat_message("user"):
        st.markdown(pertanyaan)
    # Simpan ke riwayat
    st.session_state.riwayat_chat.append({"role": "user", "content": pertanyaan})

    # Tampilkan jawaban AI
    with st.chat_message("assistant"):
        tempat_loading = st.empty()
        tempat_loading.markdown("⏳ *Bapak AI sedang berpikir...*")
        
        try:
            # Panggil otak Groq
            response = chatbot.invoke({"input": pertanyaan})
            jawaban = response["answer"]
            
            # Tampilkan jawaban asli
            tempat_loading.markdown(jawaban)
            
            # Simpan ke riwayat
            st.session_state.riwayat_chat.append({"role": "assistant", "content": jawaban})
        except Exception as e:
            tempat_loading.error(f"Waduh, terjadi kesalahan: {e}")
