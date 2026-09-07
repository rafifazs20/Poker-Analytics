import torch
import os
from core.engine import DeepPokerDSS
from core.network import DeepPokerNetwork
from core.solver import NestedSubgameSolver

def print_dashboard(dss, solver):
    print("\n" + "="*55)
    print("      LIBRATUS DSS MODERN - REAL-TIME DASHBOARD")
    print("="*55)
    
    # Ambil parameter game dari pokerkit
    pot_size = dss.get_pot_size()
    call_amt = dss.get_call_amount()
    state_tensor = dss.generate_tensor_state().unsqueeze(0)
    
    # Masukkan ke Mesin Solver (Sprint 4)
    analysis = solver.resolve_subgame(state_tensor, call_amt, pot_size)
    probs = analysis["policy_dist"]
    
    # Tampilkan Analisis Subgame
    print(f"[INFO MEJA] Total Pot: ${pot_size} | Harus Call: ${call_amt}")
    print(f"[MATEMATIKA] Pot Odds Requisite : {analysis['pot_odds']*100:.1f}%")
    print(f"[AI PREDIKSI] Expected Value (EV): {analysis['ev']:.4f}")
    
    if analysis["bluff_detected"]:
        print("🚨 [PERINGATAN] TINGKAH LAKU MUSUH TERDETEKSI SEBAGAI BLUFFING! 🚨")
    
    print("-" * 55)
    print("[DISTRIBUSI STRATEGI BLUEPRINT]")
    print(f"FOLD         : {probs[0]*100:.1f}%")
    print(f"CHECK/CALL   : {probs[1]*100:.1f}%")
    print(f"RAISE 0.5x   : {probs[2]*100:.1f}%")
    print(f"RAISE 1x Pot : {probs[3]*100:.1f}%")
    print(f"OVERBET 3x   : {probs[4]*100:.1f}%")
    
    print("-" * 55)
    print(f"👉 KEPUTUSAN MUTLAK : {analysis['decision']}")
    print("="*55 + "\n")

if __name__ == "__main__":
    dss = DeepPokerDSS()
    brain = DeepPokerNetwork()
    
    # Muat bobot otak yang sudah kita latih di Sprint 3!
    model_path = "data/poker_brain_v1.pth"
    if os.path.exists(model_path):
        brain.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"[SYSTEM] Bobot AI '{model_path}' berhasil dimuat!")
    else:
        print("[SYSTEM] Bobot tidak ditemukan, menggunakan otak kosong.")
        
    # Inisialisasi Solver
    solver = NestedSubgameSolver(brain_network=brain)
    
    print("\nSiap bermain! Masukkan parameter dari meja.")
    print("Perintah: 'h <kartu>', 'b <board>', 'bet <jumlah>', 'c' (Call/Check), 'q' (Quit)")
    
    while True:
        user_input = input("DSS> ")
        if user_input.lower() == 'q': break
        
        if dss.parse_quick_input(user_input):
            print_dashboard(dss, solver)