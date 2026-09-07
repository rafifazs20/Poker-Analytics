import torch
import torch.nn.functional as F

class NestedSubgameSolver:
    def __init__(self, brain_network):
        """
        Mesin Pengambil Keputusan Real-Time (Safe Subgame Solver).
        Memadukan Insting Blueprint (AI) dengan Matematika Meja (Pot Odds).
        """
        self.brain = brain_network
        # Pastikan AI dalam mode evaluasi (bukan mode belajar)
        self.brain.eval()

    def calculate_pot_odds(self, call_amount, pot_size):
        """Rumus matematika murni untuk menghitung batas minimal probabilitas menang"""
        if call_amount == 0:
            return 0.0
        return call_amount / (pot_size + call_amount)

    def resolve_subgame(self, state_tensor, call_amount, pot_size):
        """
        Menjalankan Real-Time Solvers:
        1. Meminta AI memprediksi Blueprint & EV.
        2. Mengkalkulasi deviasi musuh (Bluff Detection).
        3. Menghasilkan Output Keputusan Final.
        """
        # 1. Forward Pass ke AI (Tanpa kalkulasi gradien agar secepat kilat)
        with torch.no_grad():
            raw_policy, ev_prediction = self.brain(state_tensor)
            
        raw_policy = raw_policy.squeeze().numpy()
        ev = ev_prediction.item()
        
        # 2. Kalkulasi Pot Odds Matematis
        pot_odds_req = self.calculate_pot_odds(call_amount, pot_size)
        
        # 3. Logika Safe Subgame Solving (Menghukum Bluffing)
        # Jika musuh bet sangat besar (pot odds req tinggi), tapi Value Network kita 
        # mendeteksi EV kita masih positif, sistem mengenali ini sebagai eksploitasi/bluffing.
        
        final_decision = "FOLD"
        bluff_detected = False
        
        # Aturan Dasar GTO
        if ev > pot_odds_req:
            # Jika EV melebihi Pot Odds, Call/Raise adalah langkah rasional
            if raw_policy[4] > 0.2:  # Jika jaringan neural menyarankan overbet
                final_decision = "RAISE 3x (OVERBET)"
            elif raw_policy[2] > raw_policy[3]:
                final_decision = "RAISE 0.5x"
            else:
                final_decision = "CALL"
                
            # Jika musuh menuntut call besar padahal EV kita cuma margin tipis, 
            # musuh terdeteksi mencoba bluffing brutal (Polarized range).
            if pot_odds_req > 0.4 and ev > 0.4:
                bluff_detected = True
                final_decision = "CALL (BLUFF-CATCHER)"
                
        else:
            # Jika matematika pot melarang, buang kartu tanpa emosi
            final_decision = "FOLD"
            
        return {
            "policy_dist": raw_policy,
            "ev": ev,
            "pot_odds": pot_odds_req,
            "bluff_detected": bluff_detected,
            "decision": final_decision
        }