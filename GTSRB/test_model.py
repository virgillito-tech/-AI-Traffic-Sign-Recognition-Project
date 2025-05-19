import torch
from PIL import Image
from torchvision import transforms, models
from torchvision.models import resnet50, ResNet50_Weights

# Configura il dispositivo (assicurati che sia lo stesso usato in training)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definisci le stesse trasformazioni usate durante l'addestramento
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Evita il flipping orizzontale
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Carica l'architettura del modello
model = resnet50(weights=None)  # Inizializza senza pesi pre-addestrati
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 43)  # Adatta a 43 classi (GTSRB)
model.load_state_dict(torch.load("best_model.pth", map_location=device))  # Carica i pesi salvati
model.to(device)
model.eval()  # Imposta il modello in modalità valutazione

# Carica un'immagine di test (modifica il percorso in base alla tua immagine)
img_path = "/Users/cicci/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Python 3.13/TESI/TESI/GTSRB/120.jpg"
img = Image.open(img_path).convert("RGB")

# Applica le trasformazioni
img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0)  # Aggiunge una dimensione batch

# Effettua la predizione
with torch.no_grad():
    outputs = model(img_tensor.to(device))
    _, predicted = torch.max(outputs, 1)

print(f"Classe predetta (indice): {predicted.item()}")

# (Opzionale) Se hai una mappa delle classi, ad esempio:
class_labels = {
     0:'Limite di velocità (20 km/h)',
            1:'Limite di velocità (30 km/h)', 
            2:'Limite di velocità (50 km/h)', 
            3:'Limite di velocità (60 km/h)', 
            4:'Limite di velocità (70 km/h)', 
            5:'Limite di velocità (80km/h)', 
            6:'Fine del limite di velocità (80 km/h)', 
            7:'Limite di velocità (100 km/h)', 
            8:'Limite di velocità (120 km/h)', 
            9: "Divieto di sorpasso", 
            10:'Non passare i veh oltre 3,5 tonnellate', 
            11: "Diritto di passaggio all'incrocio", 
            12:'Strada prioritaria', 
            13: "Sield", 
            14:'Stop', 
            15:'Nessun veicolo', 
            16:'Veh > 3,5 tonnellate vietato', 
            17:'Nessun ingresso', 
            18:'Attenzione generale', 
            19:'Curva pericolosa a sinistra', 
            20:'Curva pericolosa a destra', 
            21:'Doppia curva', 
            22: 'Strada irregolare', 
            23:'Strada scivolosa', 
            24:'La strada si restringe a destra', 
            25:'Lavori stradali', 
            26: "Segnali di traffico", 
            27:'Pedoni', 
            28: "I bambini attraversano", 
            29:'Attraversamento di biciclette', 
            30:'Attenzione al ghiaccio/neve',
            31:'Animali selvatici che attraversano', 
            32:'Fine velocità + limiti di passaggio', 
            33:'Gira a destra', 
            34:'Gira a sinistra avanti', 
            35:'Solo avanti', 
            36:'Vai dritto o a destra', 
            37:'Vai dritto o a sinistra', 
            38:'Tieni la destra', 
            39:'Mantieni a sinistra', 
            40:'Rotonda obbligatoria', 
            41: 'Fine del non passaggio', 
            42:'Fine senza passaggio veh > 3,5 tonnellate'
}
predicted_label = class_labels.get(predicted.item(), "Sconosciuto")
print(f"Segnale riconosciuto: {predicted_label}")
