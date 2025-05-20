# 🛣️ Riconoscimento Segnali Stradali con AI / Traffic Sign Recognition with AI

## 🇮🇹 Descrizione

Progetto sperimentale di Intelligenza Artificiale applicato al settore **automotive**, focalizzato sul riconoscimento e la classificazione dei **segnali stradali**.  
Il sistema utilizza:

- 🧠 **YOLOv8** per il rilevamento in tempo reale dei segnali
- 🧠 **ResNet50** per la classificazione accurata
- 🗂️ **GTSRB** (German Traffic Sign Recognition Benchmark) come dataset di addestramento

È inclusa una web app interattiva con streaming video che mostra in tempo reale i segnali riconosciuti.

---

## 🇬🇧 Description

Experimental Artificial Intelligence project applied to the **automotive** sector, focused on **traffic sign detection and classification**.  
The system uses:

- 🧠 **YOLOv8** for real-time traffic sign detection  
- 🧠 **ResNet50** for accurate classification  
- 🗂️ **GTSRB** (German Traffic Sign Recognition Benchmark) as training dataset

An interactive web application is included, with video streaming to display detected signs in real-time.

---

## 🛠️ Tecnologie / Technologies

- Python
- HTML
- YOLOv8 (Ultralytics)  
- ResNet50 (torchvision)  
- OpenCV  
- Flask + Socket.IO  
- Bootstrap (frontend)  
- Dataset: GTSRB  

---

## 🚀 Installazione / Installation

1. Clona il repository:

   ```bash
   git clone https://github.com/TUO_USERNAME/NOME_REPOSITORY.git
   cd NOME_REPOSITORY

2. Crea ed attiva un ambiente virtuale (consigliato):
   ```bash
   python3 -m venv env
   source env/bin/activate

4. Installa le dipendenze:
   ultralytics==8.0.20
   torch>=2.0.0
   torchvision>=0.15.0
   opencv-python
   numpy
   flask
   flask-socketio
   eventlet
   gunicorn
   pillow
   matplotlib

5. Esecuzione app:
   ```bash
   python app.py


