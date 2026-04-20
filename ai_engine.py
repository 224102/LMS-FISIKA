import os
from dotenv import load_dotenv

# Import untuk pengolahan dokumen
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import untuk Vector Database dan Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Import untuk Groq AI
from langchain_groq import ChatGroq

# --- KEMBALI KE CLASSIC ---
# Import untuk Chains
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load API Key dari file .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

def siapkan_ai_fisika(lokasi_pdf):
    """
    Fungsi untuk menyiapkan 'otak' AI. 
    Membaca PDF, memprosesnya, dan menghubungkan ke Groq.
    """
    
    folder_db = "database_fisika"
    
    # Gunakan model embedding gratis dari HuggingFace (dijalankan di CPU lokal)
    print("-> Menginisialisasi model embedding...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # CEK: Apakah database sudah ada?
    if os.path.exists(folder_db) and os.listdir(folder_db):
        print("-> Memori ditemukan. Memuat database dari penyimpanan lokal...")
        vectorstore = Chroma(
            persist_directory=folder_db, 
            embedding_function=embeddings
        )
    else:
        print("-> Memori kosong. Mulai membaca PDF (Ini mungkin memakan waktu untuk file besar)...")
        if not os.path.exists(lokasi_pdf):
            raise FileNotFoundError(f"File PDF tidak ditemukan di: {lokasi_pdf}")
            
        # Load PDF menggunakan PyMuPDF
        loader = PyMuPDFLoader(lokasi_pdf)
        dokumen = loader.load()

        # Potong-potong teks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        potongan_teks = text_splitter.split_documents(dokumen)

        # Simpan ke dalam ChromaDB
        print(f"-> Mengonversi {len(potongan_teks)} potongan teks ke vektor...")
        vectorstore = Chroma.from_documents(
            documents=potongan_teks, 
            embedding=embeddings, 
            persist_directory=folder_db
        )
        print("-> Berhasil! Memori buku fisika telah disimpan di folder 'database_fisika'.")

    # 2. Konfigurasi Otak Groq AI
    print("-> Menghubungkan ke server Groq AI...")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",  
        temperature=0.1, 
        api_key=api_key
    )

    # 3. Membuat Instruksi (Prompt Engineering)
    system_prompt = (
        "Kamu adalah asisten guru Fisika ahli. "
        "Tugasmu adalah menjawab pertanyaan siswa berdasarkan referensi buku yang diberikan.\n\n"
        "ATURAN:\n"
        "1. Gunakan bahasa Indonesia yang edukatif dan mudah dipahami.\n"
        "2. Jika pertanyaan di luar konteks fisika atau tidak ada di referensi buku, "
        "jawablah: 'Mohon maaf, saya hanya asisten khusus mata pelajaran Fisika. Saya tidak bisa menjawab pertanyaan di luar topik tersebut.'\n"
        "3. Jangan menjawab pertanyaan tentang politik, resep makanan, atau gosip.\n\n"
        "REFERENSI BUKU:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 4. Membangun Rantai Pencarian (Retrieval Chain)
    pencari_dokumen = vectorstore.as_retriever(search_kwargs={"k": 5})
    rantai_dokumen = create_stuff_documents_chain(llm, prompt)
    mesin_rag = create_retrieval_chain(pencari_dokumen, rantai_dokumen)
    
    return mesin_rag

# --- AREA UJI COBA ---
if __name__ == "__main__":
    PATH_PDF = "../data/buku halliday.pdf" 

    try:
        chatbot = siapkan_ai_fisika(PATH_PDF)
        print("\n" + "="*30 + "\nAI FISIKA SIAP\n" + "="*30)
        
        while True:
            tanya = input("\nSiswa (ketik 'keluar' untuk berhenti): ")
            if tanya.lower() in ['keluar', 'exit', 'quit']:
                break
            print("AI sedang mencari jawaban di buku...")
            response = chatbot.invoke({"input": tanya})
            print(f"\nJawaban Guru AI:\n{response['answer']}")
            
    except Exception as e:
        print(f"\n[Terjadi Kesalahan]: {e}")