import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
from core.network import DeepPokerNetwork
from core.engine import DeepPokerDSS
from core.monte_carlo import MonteCarloSimulator

class DeepCFRTrainer:
    def __init__(self, brain_network, learning_rate=0.001):
        self.brain = brain_network
        self.optimizer = optim.Adam(self.brain.parameters(), lr=learning_rate)
        self.value_criterion = nn.MSELoss()
        
        self.policy_buffer = []
        self.value_buffer = []
        
        # Inisialisasi Mesin Evaluasi Matematika
        self.mc_simulator = MonteCarloSimulator()
        
    def simulate_self_play(self, num_games=1000):
        """FASE 1: KUMPULKAN DATA (Membuat dataset GTO berdasarkan Matematika Nyata)"""
        print(f"\n[TRAINER] Mengeksekusi Self-Play & Monte Carlo: Menghasilkan {num_games} putaran simulasi...")
        
        self.policy_buffer.clear()
        self.value_buffer.clear()
        
        # Kita gunakan engine murni kita sendiri untuk memetakan tensornya
        dss = DeepPokerDSS()
        ranks = '23456789TJQKA'
        suits = 'cdhs'
        deck = [r+s for r in ranks for s in suits]
        
        for i in range(num_games):
            # Print progress setiap 100 simulasi agar terminal tidak terlihat hang
            if (i+1) % 100 == 0:
                print(f"Memproses simulasi ke-{i+1}/{num_games}...")

            # 1. Generate Skenario Acak
            random.shuffle(deck)
            hero_hole = [deck[0], deck[1]]
            board_len = random.choice([0, 3, 4, 5])
            board = deck[2:2+board_len]
            
            # 2. HITUNG MATEMATIKA NYATA (MONTE CARLO)
            equity = self.mc_simulator.calculate_equity(hero_hole, board, num_simulations=100)
            real_ev = (equity * 2.0) - 1.0
            
            # 3. Masukkan ke State Engine untuk dijadikan Tensor
            dss.my_hole_cards = hero_hole
            dss.my_board_cards = board
            state_tensor = dss.generate_tensor_state()
            
            # 4. Tentukan Policy GTO (Insting)
            target_policy = torch.zeros(1, 5)
            if equity < 0.30:
                target_policy[0][0] = 0.9 
                target_policy[0][1] = 0.1 
            elif equity > 0.70:
                target_policy[0][2] = 0.3 
                target_policy[0][3] = 0.5 
                target_policy[0][4] = 0.2 
            else:
                target_policy[0][1] = 0.7 
                target_policy[0][2] = 0.3 
                
            target_value = torch.FloatTensor([[real_ev]])
            
            self.policy_buffer.append((state_tensor, target_policy))
            self.value_buffer.append((state_tensor, target_value))

    def train_with_early_stopping(self, max_epochs=10, val_split=0.2, patience=2):
        """FASE 2: TRAINING DENGAN EARLY STOPPING & VALIDATION"""
        total_data = len(self.policy_buffer)
        if total_data == 0: return
            
        val_size = int(total_data * val_split)
        train_size = total_data - val_size
        
        combined = list(zip(self.policy_buffer, self.value_buffer))
        random.shuffle(combined)
        
        train_data = combined[:train_size]
        val_data = combined[train_size:]
        
        print(f"[TRAINER] Data Split: {train_size} Training | {val_size} Validation")
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_weights = None
        
        for epoch in range(max_epochs):
            self.brain.train()
            train_loss = 0.0
            
            for p_buf, v_buf in train_data:
                state, target_policy = p_buf
                target_value = v_buf[1]
                
                self.optimizer.zero_grad()
                pred_policy, pred_value = self.brain(state)
                
                policy_loss = -torch.sum(target_policy * torch.log(pred_policy + 1e-8))
                value_loss = self.value_criterion(pred_value, target_value)
                
                loss = policy_loss + value_loss
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
            avg_train_loss = train_loss / train_size
            
            self.brain.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for p_buf, v_buf in val_data:
                    state, target_policy = p_buf
                    target_value = v_buf[1]
                    
                    pred_policy, pred_value = self.brain(state)
                    p_loss = -torch.sum(target_policy * torch.log(pred_policy + 1e-8))
                    v_loss = self.value_criterion(pred_value, target_value)
                    val_loss += (p_loss + v_loss).item()
                    
            avg_val_loss = val_loss / val_size
            print(f"Epoch {epoch+1}/{max_epochs} | Train Error: {avg_train_loss:.4f} | Validation Error: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_weights = copy.deepcopy(self.brain.state_dict())
            else:
                patience_counter += 1
                print(f" ⚠️ Peringatan: Validation Error naik! (Patience: {patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"\n🛑 EARLY STOPPING DIPICU pada Epoch {epoch+1}! Mencegah Overfitting.")
                    print("Mengembalikan memori otak AI ke kondisi terbaiknya...")
                    break
                    
        if best_model_weights:
            self.brain.load_state_dict(best_model_weights)

    def save_model(self, filepath="data/poker_brain_v1.pth"):
        torch.save(self.brain.state_dict(), filepath)
        print(f"[SYSTEM] Bobot AI (Iterasi Terbaik) berhasil disimpan ke: {filepath}")

# =======================================================
# BLOK EKSEKUSI UTAMA (Yang Tadi Sempat Hilang)
# =======================================================
if __name__ == "__main__":
    brain = DeepPokerNetwork()
    trainer = DeepCFRTrainer(brain_network=brain, learning_rate=0.005)
    
    # Kita turunkan ke 1.000 games per generasi agar tidak memakan waktu berjam-jam
    # karena komputasi Monte Carlo sangat berat.
    NUM_GAMES_PER_GEN = 1000 
    TOTAL_GENERATIONS = 3
    
    print("==================================================")
    print(" MEMULAI LIBRATUS CONTINUOUS TRAINING LOOP (2026)")
    print("==================================================")
    
    for gen in range(TOTAL_GENERATIONS):
        print(f"\n>>> GENERATION {gen+1} <<<")
        trainer.simulate_self_play(num_games=NUM_GAMES_PER_GEN)
        trainer.train_with_early_stopping(max_epochs=10, val_split=0.2, patience=2)
        
    trainer.save_model()