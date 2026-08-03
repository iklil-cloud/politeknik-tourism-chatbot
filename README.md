# Chatbot Sejarah SPM

Prototype chatbot Sejarah SPM Tingkatan 5 berasaskan `RAG + LLM` menggunakan data daripada fail bab dalam `data/chapters/`.

## Susunan data

Letakkan setiap bab sebagai fail `.txt` berasingan di dalam `data/chapters/`, contohnya:

- `data/chapters/BAB 4.txt`
- `data/chapters/BAB 5.txt`

Setiap fail perlu mengandungi tajuk bab seperti `BAB 4 SISTEM PERSEKUTUAN` di dalam kandungan teks.

## Ciri utama

- Soal jawab dalam Bahasa Melayu
- Ringkasan bab
- Rujukan sumber chunk yang digunakan semasa menjawab
- Penilaian awal menggunakan set soalan CSV

## Seni bina

- `Streamlit` untuk antaramuka
- `ChromaDB` untuk stor vektor apabila tersedia
- `Ollama` untuk embedding dan penjanaan jawapan
- `qwen2.5:3b` sebagai model jawapan
- `nomic-embed-text:latest` sebagai model embedding tempatan yang sedia ada

Jika stor `ChromaDB` gagal dibuka dalam folder semasa, aplikasi akan jatuh balik ke stor vektor JSON tempatan supaya prototaip tetap boleh digunakan.

## Jalankan projek

Pastikan Ollama sedang berjalan dan model berikut tersedia:

- `qwen2.5:3b`
- `nomic-embed-text:latest`

Bina indeks:

```bash
python scripts/build_index.py --reset
```

Lancarkan aplikasi:

```bash
streamlit run app/streamlit_app.py
```

Jalankan penilaian:

```bash
python scripts/evaluate.py
```

Jalankan penilaian LLM-as-a-judge:

```bash
python scripts/evaluate.py --judge --skip-bert
```

Pilihan `--judge` menggunakan Ollama untuk menilai jawapan berdasarkan `correctness`, `relevance`, `completeness`, `clarity`, `groundedness` dan `overall`. Gunakan `--judge-model nama-model` jika mahu model penilai yang berbeza.

## Struktur

- `app/` logik aplikasi dan pipeline RAG
- `scripts/` utiliti indexing dan evaluation
- `data/test_questions.csv` set ujian awal
- `results/` laporan penilaian selepas dijana
