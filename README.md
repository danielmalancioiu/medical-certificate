# 🩺 CNAS Medical Certificate OCR App

A lightweight Streamlit app that automatically extracts structured data (series, number, CNP, dates, diagnostic code, etc.) from scanned Romanian **“Certificat de Concediu Medical”** documents using **EasyOCR** and **OpenCV**.

---

## 🚀 Features
- Extracts:
  - Seria și număr certificat
  - Nume pacient
  - CNP
  - Data „De la” și „Până la”
  - Cod diagnostic
- Auto-aligns extraction zones based on text position
- Works with scanned or photographed documents
- Visual ROI preview for debugging

---

## ⚙️ 1. Prerequisites
Make sure you have **Python 3.10+** installed.

### Optional (recommended for Windows):
Install Tesseract OCR (if you ever extend beyond EasyOCR):
> https://github.com/UB-Mannheim/tesseract/wiki

---

## 🧱 2. Clone & Setup

```bash
# Clone or copy project folder
cd medical-certificate

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\activate

# Upgrade pip (important)
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

---

## 🧠 3. Run the App

```bash
python -m streamlit run app.py
```

Then open the link shown in your terminal (usually http://localhost:8501).

---

## 📂 Project Structure

```
medical-certificate/
│
├── app.py               # Streamlit front-end
├── ocr_utils.py         # OCR + preprocessing + field extraction logic
├── requirements.txt     # Dependencies
└── README.md            # This file
```

---

## 🧩 Troubleshooting

### 🔸 Pip install fails with `.deleteme` errors
That’s a Windows permissions issue.  
Fix:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt --no-warn-script-location
```

### 🔸 “No pyvenv.cfg file”
Your virtual environment is broken.  
Fix:
```bash
Remove-Item -Recurse -Force .venv
python -m venv .venv
.venv\Scripts\activate
```

---

## 🧰 Requirements Summary
| Package | Purpose |
|----------|----------|
| `streamlit` | Web UI framework |
| `easyocr` | Text recognition (printed + handwriting) |
| `opencv-python-headless` | Image preprocessing |
| `pillow` | Image handling |
| `numpy` | Array operations |

---

## 🧪 Sample Usage

1. Launch the app.
2. Upload a `.png` or `.jpg` of a CNAS medical certificate.
3. Wait for the preview — green boxes show OCR zones.
4. The structured data appears in JSON format.

---

## 📜 License
MIT License — free to use and modify.

---

👨‍💻 **Developed for academic and internal automation use.**
