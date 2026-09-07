import torch
from core.engine import DeepPokerDSS
from core.network import DeepPokerNetwork

def print_dashboard(dss, brain):
    print("\n" + "="*50)
    print("      DASBOR ASISTEN AI POKER (REAL-TIME)")
    print("="*50)
    print(f"[INFO MEJA] Total Pot: {dss.state.total_pot} | Harus Call: {dss.state.checking_or_calling_amount}")
    
    # Ambil kondisi meja (Tensor)
    state_tensor = dss.generate_tensor_state().unsqueeze(0) # Tambah dimensi batch
    
    # Masukkan ke dalam Otak AI
    with torch.no_grad(): # Matikan gradient perhitungan (karena ini mode inference/bermain)
        probs, ev = brain(state_tensor)
        
    probs = probs.squeeze().numpy()
    
    print("-" * 50)
    print(f"[PREDIKSI NILAI] Expected Value (EV) Posisi Ini: {ev.item():.4f}")
    print("[DISTRIBUSI GTO (Rekomendasi Aksi)]")
    print(f"👉 FOLD         : {probs[0]*100:.1f}%")
    print(f"👉 CHECK/CALL   : {probs[1]*100:.1f}%")
    print(f"👉 RAISE 0.5x   : {probs[2]*100:.1f}%")
    print(f"👉 RAISE 1x Pot : {probs[3]*100:.1f}%")
    print(f"👉 OVERBET 3x   : {probs[4]*100:.1f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    dss = DeepPokerDSS()
    brain = DeepPokerNetwork()
    
    # Karena belum ditraining di Sprint 3, ini hanya memuat bobot model acak (belum cerdas)
    print("Sistem Siap! (Note: AI belum ditraining, hasil probabilitas masih acak)")
    print("Format input: 'h AhKh' (Hole), 'b 9hTs9s' (Board), 'bet 60', 'c' (Call/Check), 'q' (Quit)")
    
    while True:
        user_input = input("DSS> ")
        if user_input.lower() == 'q': break
        
        # Jika input valid, render ulang dasbor
        if dss.parse_quick_input(user_input):
            print_dashboard(dss, brain)