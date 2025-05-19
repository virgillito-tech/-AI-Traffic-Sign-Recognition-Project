import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from multiprocessing import freeze_support
from tqdm import tqdm
from torch.amp import GradScaler, autocast

# Monitoraggio iniziale della memoria GPU
print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# Configura il dispositivo: GPU se disponibile, altrimenti CPU/MPS
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Utilizzo dispositivo: {device}")

# **1. Definizione delle Trasformazioni**
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Evita il flipping orizzontale
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# **2. Caricamento del Dataset**
train_dir = r"/Users/cicci/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Python 3.13/TESI/TESI/GTSRB/Train"

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Errore: la cartella {train_dir} non esiste!")

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
print(f"Dataset caricato con successo! Numero di classi: {len(train_dataset.classes)}")

# **3. Divisione Training/Validation (80/20)**
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)

# Verifica il numero di dati caricati
print(f"Train samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")

# **4. Definizione del Modello (ResNet50)**
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 43)  # Adatta l'output a 43 classi di GTSRB
model.to(device)

# Verifica se il modello è sulla GPU
print(f"Model is on GPU: {next(model.parameters()).is_cuda}")

# **5. Definizione dell'Ottimizzatore, Scheduler e Funzione di Perdita**
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Riduce il LR ogni 5 epoche
criterion = nn.CrossEntropyLoss()

# **6. Funzione di Validazione**
def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_loss = val_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# **7. Funzione di Training con Mixed Precision**
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs=40, save_path="best_model.pth"):
    print("Inizio training del modello...")
    best_val_loss = float("inf")
    scaler = GradScaler()  # Per la precisione mista

    for epoch in range(num_epochs):
        print(f"Inizio Epoca {epoch+1}")
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        # Itera sui batch usando la barra di progresso
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            # Verifica che i dati siano sulla GPU
            # print(f"Data on GPU: {inputs.is_cuda}")  # Puoi decommentare per il debug

            optimizer.zero_grad()

            # Usa la precisione mista: esegui la forward pass dentro autocast
            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=train_loss / (i + 1))
            # Puoi stampare il batch loss se necessario:
            # print(f"Batch {i+1} loss: {loss.item()}")

        # Fine epoca: stampa la loss media di addestramento
        epoch_loss = train_loss / len(train_loader)
        print(f"Fine Epoca {epoch+1} - Loss di addestramento: {epoch_loss:.4f}")

        # **Validazione**
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4%}")

        # Salva il miglior modello
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("Miglior modello salvato!")

        scheduler.step()

    print("Training completato!")

# **8. Avvio del Training**
if __name__ == '__main__':
    freeze_support()  # Se necessario su Windows
    train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs=40, save_path="best_model.pth")
