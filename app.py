from flask import Flask, request, jsonify, render_template, Response, send_from_directory
import os
import cv2
import json
import logging
import warnings
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch
from PIL import Image
from collections import deque
from flask_socketio import SocketIO, emit
import base64
from io import BytesIO

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)

# Configurazione Flask
app = Flask(__name__, static_folder='uploads')  # Creiamo un'applicazione Flask e gli permettiamo di accedere ai file statici contenuti in uploads, flask è usato per gestire le rotte HTTP, servire pagine e gestire file statici.
# Configurazione Flask-SocketIO
socketio = SocketIO(app, cors_allowed_origins="*") # Per la comunicazione bidirezionale in tempo reale con il frontend

# Cartella per salvare i video caricati
UPLOAD_FOLDER = 'uploads'  # Specifica la cartella di upload
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # crea la cartella se non esiste
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Configura la cartella di upload

# Verifica GPU
device = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" 
                      if torch.backends.mps.is_available() 
                      else "cpu")
print(f"Using device: {device}") 

# Inizializzazione YOLOv8
yolo_model = YOLO("yolov8n.pt").to(device)  # Sostituiamo con il nostro modello se necessario

# Inizializzazione ResNet50
resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
resnet_model.eval()  # Impostiamo Resnet in modalità valutazione

# Trasformazioni per ResNet
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}  # Estensioni di file consentite

def allowed_file(filename):  # Funzione per conotrollare l'estensione del file
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS 

class VideoProcessor:  # Classe per processare i video
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder
        self.last_uploaded_video = None
        os.makedirs(upload_folder, exist_ok=True)

    def save_video(self, video):  # Funzione per salvare il video
        video_path = os.path.join(self.upload_folder, video.filename)
        video.save(video_path)
        self.last_uploaded_video = video.filename
        return video_path

    def get_last_video_path(self):  # Funzione per ottenere il percorso dell'ultimo video
        if not self.last_uploaded_video:
            return None
        video_path = os.path.join(self.upload_folder, self.last_uploaded_video)
        return video_path if os.path.exists(video_path) else None

video_processor = VideoProcessor(UPLOAD_FOLDER)  # Creiamo un'istanza della classe VideoProcessor
detected_signals = set()  # Set per memorizzare i segnali rilevati
recent_signals = deque(maxlen=50)  # Tiene traccia degli ultimi 50 frame

# Pagina principale
@app.route('/')
def index():
    return render_template('cruscotto2.html') # Restituiamo la pagina HTML

@socketio.on('connect')
def handle_connect():
    print("Client connesso")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnesso")

# Controlliamo se un file video è presente nella richiesta
@app.route('/upload_video', methods=['POST'])
def upload_video():  # Funzione per caricare un video
    if 'video' not in request.files:
        return jsonify({"error": "Nessun file video trovato"}), 400

    video = request.files['video']  # Recuperiamo il file video dalla richiesta
    if video.filename == '' or not allowed_file(video.filename):
        return jsonify({"error": "File non valido"}), 400

    video_path = video_processor.save_video(video)  # salviamo il video
    return jsonify({"message": "Video caricato con successo", "video_path": video_path}), 200

# Funzione per restituire il video in streaming
@app.route("/video_feed")
def video_feed():
    video_path = video_processor.get_last_video_path()  # Recuperiamo il percorso del video
    if not video_path:
        return jsonify({"error": "Nessun video caricato"}), 400

    try:
        cap = cv2.VideoCapture(video_path)  # Apriamo il video
        if not cap.isOpened():
            raise IOError("Impossibile aprire il file video")

        def generate_stream():  # Funzione per generare i frame del video
            frame_counter = 0
            skip_frames = 2  # Salta 2 frame su 3
            while cap.isOpened():  # Fino a quando il video è aperto
                ret, frame = cap.read()  # Leggiamo un frame
                if not ret or frame is None:
                    logging.warning("Fine del video o frame non valido")
                    break

                if frame_counter % skip_frames != 0:
                    frame_counter += 1
                    continue

                frame_counter += 1
                logging.info(f"Processamento frame numero: {frame_counter}")

                try:
                    frame = cv2.resize(frame, (640, 360))  # Riduciamo la risoluzione
                    frame_signals = process_frame(frame)

                    # Inviamo i segnali tramite WebSocket
                    socketio.emit('signal_detected', frame_signals)
                    logging.info("Segnali inviati tramite WebSocket")

                    _, buffer = cv2.imencode('.jpg', frame)  # Codifichiamo il frame per lo streaming
                    frame_bytes = buffer.tobytes()  # Convertiamo il frame in byte
                    logging.info("Frame codificato correttamente")

                    # Inviamo i frame al frontend
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    logging.error(f"Errore durante lo streaming del frame: {e}")    

            cap.release()  # Chiudiamo il video  
            logging.info("Video terminato") 
            socketio.emit('video_end', {'message': 'Video terminato'})


        return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame') 

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/ritagli/<filename>')
def serve_image(filename):
    return send_from_directory('ritagli', filename)

def list_ritagli():
    files = os.listdir('ritagli')
    return jsonify(files)

# Funzione per processare il frame e inviare le immagini ritagliate al frontend, 
def process_frame(frame, save_dir="ritagli",  conf_threshold=0.3, tolerance=5): # Salviamo i segnali in "ritagli", confidenza medio/bassa per riconoscere quanti più segnali, tolleranza per eliminare i duplicati
    global recent_signals # Usiamo una coda per tenere traccia dei segnali recenti
    global detected_signals 
    signals_detected = [] 
    results = yolo_model.predict(frame, conf=conf_threshold) 
    detections = results[0].boxes  # Estrarre box, confidenza e classi

    logging.info(f"YOLO rilevazioni: {detections}")
    os.makedirs(save_dir, exist_ok=True)

    # Per ogni rilevazione, ritagliamo il segnale e lo classifichiamo con Resnet
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        if conf < conf_threshold:
            continue

        # Disegniamo il box di rilevamento e la confidenza sul frame
        label = f"{conf:.2f}"
        color = (0, 255, 0)  # Verde per i segnali
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Box
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Testo

        cropped = frame[y1:y2, x1:x2]  # Ritagliamo il segnale
        if cropped.size == 0:
            logging.warning(f"Ritaglio vuoto ignorato: {x1}, {y1}, {x2}, {y2}")
            continue

        try:
            img = Image.fromarray(cropped)  # Convertiamo l'array ricavato in un'immagine
            img_t = transform(img).unsqueeze(0).to(device)  # Applichiamo le trasformazioni e spostiamo sulla GPU
            with torch.no_grad():  # previene il calcolo dei gradienti per risparmiare memoria
                output = resnet_model(img_t)  # Eseguiamo il modello Resnet
                _, predicted = output.max(1)  # Otteniamo la predizione
                class_name = str(predicted.item())  # Otteniamo il nome della classe
                logging.info(f"Classe predetta: {class_name}")

            # Creiamo il nuovo segnale con le coordinate e altre informazioni
            new_signal = {
                "class_name": class_name,
                "confidence": round(conf * 100, 2),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }

            # Verifichiamo se il segnale è simile a uno già rilevato
            if any(is_similar_signal(new_signal, signal, tolerance) for signal in recent_signals):
                logging.info(f"Segnale scartato perché simile: {new_signal}")
                continue  # Saltiamo i segnali simili

            # Aggiungiamo il nuovo segnale alla lista dei segnali recenti
            recent_signals.append(new_signal)

            # Salviamo il frame nella cartella
            save_path = f"{save_dir}/{class_name}_{x1}_{y1}.jpg"
            if not cv2.imwrite(save_path, cropped):
                logging.error(f"Errore nel salvataggio del file: {save_path}")
            else:
                logging.info(f"Ritaglio salvato: {save_path}")   

            # Convertiamo l'immagine in base64
            _, buffer = cv2.imencode('.jpg', cropped)
            signal_image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Memorizziamo il segnale rilevato e la sua confidenza
            signals_detected.append({
                "image": signal_image_base64,  # Aggiungi l'immagine in formato base64
                "class_name": class_name,
                "confidence": round(conf * 100, 2),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

        except Exception as e:
            logging.error(f"Errore durante il processamento del frame: {e}")

    # Mandiamo i segnali rilevati al frontend tramite WebSocket
    for signal in signals_detected:
        if "image" not in signal or "class_name" not in signal or "confidence" not in signal:
            logging.error(f"Dati incompleti per il segnale: {signal}")
            continue
    socketio.emit('signal_detected', signals_detected)        

    return signals_detected  # Restituiamo i risultati

# funzione per verificare segnali simili
def is_similar_signal(new_signal, existing_signal, tolerance=5):
    """
    Verifica se due segnali sono simili basandosi su classe, coordinate e tolleranza.
    """
    return ( # verfichiamo i segnali già identificati con quelli nuovi per evitare duplicati
        new_signal["class_name"] == existing_signal["class_name"] and # confronto classe
        abs(new_signal["x1"] - existing_signal["x1"]) <= tolerance and # confronto ritaglio
        abs(new_signal["y1"] - existing_signal["y1"]) <= tolerance and
        abs(new_signal["x2"] - existing_signal["x2"]) <= tolerance and
        abs(new_signal["y2"] - existing_signal["y2"]) <= tolerance
    )

@socketio.on('request_signals')
def send_detected_signals(): # Funzione per inviare i segnali al frontend
    global detected_signals
    # Deserializza i segnali prima di inviarli
    signals_list = [json.loads(signal) for signal in detected_signals]
    emit('all_signals', signals_list) 

if __name__ == '__main__':
    socketio.run(app, host="127.0.0.1", port=5001, debug=True)



