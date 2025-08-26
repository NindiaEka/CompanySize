import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset (pastikan path-nya sesuai)
@st.cache_data

     
    
def load_data():
    df = pd.read_csv("clean_data.csv")  
    return df

def show_eda_page():
    gambar = Image.open('ref_gambar/download.jpg')
    st.image(gambar)

    # Latar belakang
    st.write('# Latar Belakang')
    st.markdown('''
                
                Dalam proses pencarian pekerjaan, pencari kerja tidak hanya mempertimbangkan posisi yang tersedia, tetapi juga kesesuaian dengan budaya perusahaan, struktur organisasi, dan peluang pengembangan karir. Informasi seperti deskripsi perusahaan dan jumlah karyawan memberikan wawasan penting mengenai:
               
                 •	Budaya dan nilai perusahaan: Mengetahui apakah perusahaan memiliki budaya yang mendukung keseimbangan kerja-hidup, inovasi, atau keberagaman.
               
                 •	Ukuran dan struktur organisasi: Memahami apakah perusahaan besar dengan struktur hierarki yang jelas atau perusahaan kecil yang lebih fleksibel dan dinamis.
               
                 •	Peluang karir dan pengembangan: Menilai apakah perusahaan menawarkan jalur karir yang jelas dan peluang untuk berkembang.
                
                Dengan menggunakan teknik Natural Language Processing (NLP), pencari kerja dapat menganalisis deskripsi perusahaan untuk mengekstrak informasi terkait budaya, nilai, dan fokus utama perusahaan, serta mengklasifikasikan perusahaan berdasarkan ukuran dan jumlah karyawan. Hal ini memungkinkan pencari kerja untuk menyesuaikan pilihan mereka dengan perusahaan yang paling sesuai dengan tujuan karir dan nilai pribadi mereka.
                ''')
    
    # Load dataset langsung
    st.write("### Dataset")
    df = pd.read_csv("clean_data.csv", sep=",") 
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    
    # tampilkan dataframe rapi
    st.dataframe(df, use_container_width=True)

    st.subheader(" Eksploratory Data Analysis (EDA)")

    df = load_data()

    st.write("### Distribusi Data pada Masing-Masing Kelas")
    
    # Visualisasi distribusi
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='company_type', palette='Set2')
    plt.title('Distribusi berdasarkan Kategori Perusahaan')
    plt.xlabel('Kategori Perusahaan')
    plt.ylabel('Jumlah Perusahaan')
    st.pyplot(plt)

    st.markdown("""
    ### Distribusi Kategori Perusahaan

    Visualisasi di atas memperlihatkan distribusi dari masing-masing kelas hasil klasifikasi ukuran perusahaan (`company_type`).

    #### Hasil Distribusi:
    - Perusahaan Besar (Large): Kategori terbanyak dalam dataset.
    - Perusahaan Kecil (Small): Kedua terbanyak, menunjukkan banyaknya perusahaan kecil yang terdaftar.
    - Perusahaan Menengah (Medium): Kategori paling sedikit.

    ---

    ### Insight Bisnis:

    - Perusahaan Besar (Large)  
    Dominasi dalam dataset, mencerminkan pengaruh besar di pasar dan kehadiran yang kuat di LinkedIn.

    - Perusahaan Kecil (Small)  
    Banyaknya perusahaan kecil menunjukkan mereka lebih aktif di LinkedIn untuk menjangkau *niche market* atau pelanggan lokal.

    - Perusahaan Menengah (Medium)  
    Jumlah lebih sedikit, menunjukkan mereka berada di tahap peralihan dan mungkin lebih fokus pada operasional internal.

    ---
    """)

    st.header(" Kata - kata yang memiliki frekuensi tertinggi paling sering muncul dari Deskripsi Perusahaan")

    # Gabungkan semua deskripsi jadi satu string
    text_data = " ".join(df['description'].dropna().tolist())

    # Buat WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          stopwords='english', colormap='viridis').generate(text_data)

    # Tampilkan WordCloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Insight
    st.markdown("""
    ### Word Cloud menggambarkan kata-kata yang paling sering muncul dalam suatu kumpulan teks. Semakin besar ukuran kata, semakin sering kata tersebut digunakan.

        Tabel Frekuensi memberikan data numerik terkait seberapa sering kata-kata muncul.
        1. Analisis Deskripsi Perusahaan (Budaya, Nilai, Fokus Utama)
        - Word cloud menunjukkan kata dominan seperti services, solutions, company, business, clients, technology, people, world.  
        - Hal ini menggambarkan bahwa budaya perusahaan berorientasi pada layanan, solusi, dan inovasi teknologi.  
        - Nilai utama yang sering muncul: pelayanan, inovasi, skala global, dan orientasi klien/pelanggan.  
        - Insight: deskripsi perusahaan lebih menonjolkan fungsi layanan dan solusi dibanding aspek internal seperti budaya kerja atau keberlanjutan.

        ---

        2. Wawasan bagi Pencari Kerja
        - Kata clients, people, team, talent menekankan pentingnya hubungan manusia dan kolaborasi.  
        - Bagi pencari kerja, hal ini memberi sinyal bahwa soft skills (misalnya teamwork, komunikasi dengan klien) sama pentingnya dengan hard skills.  
        - Kata technology dan solutions menunjukkan peluang karir yang kuat di bidang inovasi dan teknologi.

        ---
    """)

    st.header(" WordCloud dan Kata Terpopuler per Ukuran Perusahaan")

    # Inisialisasi CountVectorizer
    vectorizer = CountVectorizer(stop_words='english')
    X_all = vectorizer.fit_transform(df['description'].astype(str))

    word_freq_by_size = {}
    for size in ['small', 'medium', 'large']:
        mask = df['company_type'].eq(size)
        X = vectorizer.transform(df.loc[mask, 'description'].astype(str))  # pakai kolom 'description'
        counts = np.asarray(X.sum(axis=0)).ravel()
        vocab  = vectorizer.get_feature_names_out()
        freqs  = {w:int(c) for w,c in zip(vocab, counts) if c>0}
        word_freq_by_size[size] = freqs

    for size in ['small', 'medium', 'large']:
        freqs = word_freq_by_size.get(size, {})
        st.subheader(f" {size.capitalize()} Companies")

        if not freqs:
            st.warning(f"Tidak ada deskripsi untuk kelas: {size}")
            continue

        # WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color='white')\
                    .generate_from_frequencies(freqs)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        # Tabel Top 10 Kata
        word_freq_df = (pd.DataFrame(freqs.items(), columns=['Word','Frequency'])
                          .sort_values('Frequency', ascending=False)
                          .head(10))
        st.markdown("Top 10 Kata Paling Sering Muncul:")
        st.dataframe(word_freq_df)

    st.markdown("""
    ###  Perbandingan Berdasarkan Ukuran Perusahaan

            Small Companies
            - Kata dominan: services, clients, solutions, business, technology, staffing, talent.  
            - Fokus narasi: kedekatan dengan klien, fleksibilitas, tim kecil, dan spesialisasi niche.  
            - Nilai utama: personal, fleksibel, client-oriented.  
            - Cocok untuk: pencari kerja yang ingin pengalaman langsung, peran multi-fungsi, dan lingkungan kerja yang lebih dekat.  

            ---

            Medium Companies
            - Kata dominan: services, solutions, company, business, technology, clients, quality, industry.  
            - Fokus narasi: layanan & solusi tetap dominan, mulai menekankan kualitas, efisiensi, dan kesehatan.  
            - Nilai utama: keseimbangan antara customer focus dan internal process.  
            - Cocok untuk: pencari kerja yang menginginkan stabilitas, sistem lebih terstruktur, namun tetap dinamis.  

            ---

            Large Companies
            - Kata dominan: world, services, company, solutions, health, care, global, research, customers.  
            - Fokus narasi: branding global, inovasi, healthcare, research, dan tanggung jawab sosial.  
            - Nilai utama: skala internasional, inovasi, stabilitas, dampak sosial.  
            - Cocok untuk: pencari kerja yang menginginkan karir jangka panjang, stabilitas, serta exposure internasional.  

            ---

            Hubungan Ukuran Perusahaan & Deskripsi
            - Small companies → narasi lebih personal & spesifik → cenderung menekankan hubungan dengan klien serta fleksibilitas.  
            - Medium companies → narasi mulai menekankan kualitas, efisiensi, serta struktur internal organisasi.  
            - Large companies → narasi lebih global & makro → menekankan skala internasional, inovasi, dan tanggung jawab sosial.  

            ---

            Kesimpulan
            - Deskripsi perusahaan menekankan layanan, solusi, dan teknologi sebagai elemen inti.  
            - Perbedaan narasi terlihat jelas berdasarkan ukuran perusahaan:  
            - Small companies → fleksibel & dekat dengan klien.  
            - Medium companies → seimbang antara orientasi eksternal & internal.  
            - Large companies → global, inovatif, dan berorientasi jangka panjang.  
            - Insight ini dapat membantu pencari kerja menyesuaikan pilihan karir dengan nilai, budaya, dan tujuan profesional yang paling sesuai.
            ---

    """)



