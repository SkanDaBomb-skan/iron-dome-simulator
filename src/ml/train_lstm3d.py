import sys
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Ajouter le dossier src au path pour les imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from simulation.rocket3d import Rocket3D
from ml.predictor3d import RocketLSTM3D

def generer_dataset_lstm_3d(n_traj=5000, n_pts=30, sigma=30, seed=42):
    """
    Génère des séquences [batch, time, features] pour LSTM 3D.
    n_pts = 30 correspond à 3 secondes d'observation (dt=0.1s).
    """
    np.random.seed(seed)
    X, y = [], []
    print(f"Génération de {n_traj} trajectoires d'entraînement...")
    for i in range(n_traj):
        v0 = np.random.uniform(150, 400)
        th = np.random.uniform(30, 75)
        ph = np.random.uniform(-20, 20)
        m = np.random.uniform(50, 200)
        Cd = np.random.uniform(0.2, 0.5)
        x0 = np.random.uniform(-8000, -3000)
        z0 = np.random.uniform(-2000, 2000)
        
        r = Rocket3D(v0=v0, theta_deg=th, phi_deg=ph, masse=m, Cd=Cd, x0=x0, z0=z0)
        tv, xv, yv, zv = r.trajectoire_rk4(dt=0.1)
        
        if len(tv) < n_pts + 5:
            continue
            
        # Simuler le radar (ajouter du bruit)
        to = tv[:n_pts]
        xo = xv[:n_pts] + np.random.normal(0, sigma, n_pts)
        yo = np.maximum(yv[:n_pts] + np.random.normal(0, sigma, n_pts), 0)
        zo = zv[:n_pts] + np.random.normal(0, sigma, n_pts)
        
        # Calculer les vitesses
        dt_r = to[1] - to[0]
        vx = np.gradient(xo, dt_r)
        vy = np.gradient(yo, dt_r)
        vz = np.gradient(zo, dt_r)
        
        # Séquence [n_pts, 6] : x, y, z, vx, vy, vz
        # On normalise les valeurs (km et km/s environ) pour le réseau
        seq = np.column_stack((xo/1000, yo/1000, zo/1000, vx/1000, vy/1000, vz/1000))
        X.append(seq)
        
        # Target : [x_impact, z_impact] en km
        y.append([xv[-1]/1000, zv[-1]/1000])
        
        if (i+1) % 1000 == 0:
            print(f"  {i+1}/{n_traj} trajectoires générées...")
            
    return np.array(X), np.array(y)

def train_and_save_model():
    print("--- DÉBUT DE L'ENTRAÎNEMENT DU MODÈLE LSTM 3D ---")
    # On utilise 20000 trajectoires pour l'entraînement
    # n_pts = 30 (3 secondes d'observation)
    X, y = generer_dataset_lstm_3d(n_traj=20000, n_pts=30, sigma=30)
    
    print(f"Dataset prêt : {X.shape[0]} séquences de {X.shape[1]} points.")
    
    # Split Train/Val (80% / 20%)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil utilisé pour l'entraînement : {device}")
    
    # Conversion en tenseurs
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialisation du modèle
    model = RocketLSTM3D(input_size=6, hidden_size=128, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Entraînement
    epochs = 40  # 40 epochs est suffisant pour 5000 trajectoires
    best_val_loss = float('inf')
    best_model_state = None
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
    print(f"Entraînement terminé en {time.time() - start_time:.1f} secondes.")
    
    # Sauvegarde du modèle
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rocket_lstm3d.pth')
    if best_model_state is not None:
        torch.save(best_model_state, model_path)
    else:
        torch.save(model.state_dict(), model_path)
        
    print(f"Modele LSTM 3D sauvegarde avec succes sous : {model_path}")

if __name__ == '__main__':
    train_and_save_model()
