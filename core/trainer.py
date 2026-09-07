import torch
import torch.nn as nn
import torch.optim as optim
import random
from core.network import DeepPokerNetwork
from core.engine import DeepPokerDSS

class DeepCFRTrainer:
    def __init__(self, brain_network, learning_rate=0.001):
        """
        Arsitektur Pelatih AI (Deep CFR)
        Menggunakan MSELoss untuk Value Network (Regrets) 
        dan KLDivLoss/CrossEntropy untuk Policy Network.
        """
        self.brain = brain_network
        self.optimizer = optim.Adam(self.brain.parameters(), lr=learning_rate)
        
        # Buffer Memori untuk menyimpan pengalaman Self-Play
        self.policy_buffer = []  # Menyimpan: (State Tensor, Target GTO Policy)
        self.value_buffer = []   # Menyimpan: (State Tensor, Target Expected Value)
        
        # Loss functions
        self.value_criterion = nn.MSELoss() # Menghitung jarak error prediksi keuntungan
        
    def simulate_self_play_trajectory(self, num_games=100):
        """
        Fase 1: Pengumpulan Data Mandiri (Self-Play)
        Di versi produksi, ini akan melakukan traversal pohon keputusan PokerKit.
        Untuk scaffolding ini, kita membangun kerangka generator state simulasi.
        """
        print(f"[TRAINER] Memulai simulasi mandiri (Self-Play) sebanyak {num_games} iterasi...")
        engine = DeepPokerDSS()
        
        for _ in range(num_games):
            # 1. Generate state acak (seolah-olah sedang bermain di berbagai skenario)
            state_tensor = torch.rand(1, 106) 
            
            # 2. Logika Regret Matching (Disederhanakan untuk scaffolding)
            # Idealnya di sini kita menghitung node traversal. 
            # Kita simulasikan target probabilitas aksi GTO (harus berjumlah 1.0)
            target_policy = torch.rand(1, 5)
            target_policy = target_policy / target_policy.sum() 
            
            # Target Expected Value (EV) antara -1 (Kalah Total) hingga 1 (Menang Mutlak)
            target_value = torch.FloatTensor([[random.uniform(-1.0, 1.0)]])
            
            # 3. Simpan ke Replay Buffer
            self.policy_buffer.append((state_tensor, target_policy))
            self.value_buffer.append((state_tensor, target_value))
            
    def train_network(self, epochs=5):
        """
        Fase 2: Retraining (Deep Learning Backpropagation)
        Memperbarui bobot Neural Network berdasarkan pengalaman di Replay Buffer.
        """
        if not self.policy_buffer or not self.value_buffer:
            print("[ERROR] Buffer kosong. Jalankan self-play terlebih dahulu.")
            return
            
        print(f"\n[TRAINER] Memulai proses optimasi Neural Network ({epochs} Epochs)...")
        self.brain.train() # Set model ke mode training
        
        for epoch in range(epochs):
            total_policy_loss = 0.0
            total_value_loss = 0.0
            
            # Iterasi ke seluruh memori pengalaman
            for i in range(len(self.policy_buffer)):
                state = self.policy_buffer[i][0]
                target_policy = self.policy_buffer[i][1]
                target_value = self.value_buffer[i][1]
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass (Biarkan AI menebak)
                pred_policy, pred_value = self.brain(state)
                
                # Hitung Error (Loss)
                # Policy loss menggunakan Cross Entropy / selisih probabilitas
                policy_loss = -torch.sum(target_policy * torch.log(pred_policy + 1e-8))
                
                # Value loss menggunakan Mean Squared Error
                value_loss = self.value_criterion(pred_value, target_value)
                
                # Gabungkan error
                total_loss = policy_loss + value_loss
                
                # Backward pass (Backpropagation - Otak AI belajar di sini)
                total_loss.backward()
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                
            print(f"Epoch {epoch+1}/{epochs} | Policy Loss: {total_policy_loss:.4f} | Value Loss: {total_value_loss:.4f}")

    def save_model(self, filepath="data/poker_brain_v1.pth"):
        """Menyimpan arsitektur bobot AI ke folder lokal."""
        torch.save(self.brain.state_dict(), filepath)
        print(f"[SYSTEM] Bobot AI berhasil disimpan ke: {filepath}")
        
# --- UJI COBA INTERNAL TRAINER ---
if __name__ == "__main__":
    from core.network import DeepPokerNetwork
    
    # Inisialisasi Otak
    brain = DeepPokerNetwork()
    
    # Masukkan ke Pelatih (Trainer)
    trainer = DeepCFRTrainer(brain_network=brain, learning_rate=0.005)
    
    # 1. Kumpulkan Data (Membangun "Tabel" secara dinamis)
    trainer.simulate_self_play_trajectory(num_games=100)
    
    # 2. Latih Otak AI
    trainer.train_network(epochs=10)
    
    # 3. Simpan Hasil Latihan
    trainer.save_model()