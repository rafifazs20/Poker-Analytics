import random
import itertools
from pokerkit import StandardHighHand

class MonteCarloSimulator:
    def __init__(self):
        """
        Mesin Probabilitas Murni (Kalkulator Equity)
        Membangun deck standar 52 kartu untuk simulasi cepat.
        """
        self.ranks = '23456789TJQKA'
        self.suits = 'cdhs'
        # Buat daftar string kartu murni: ['2c', '2d', ..., 'As']
        self.full_deck = [r+s for r in self.ranks for s in self.suits]

    def calculate_equity(self, hero_hole, board, num_simulations=1000):
        """
        Simulasi Monte Carlo: Menjalankan N-putaran secara acak untuk mencari probabilitas kemenangan mutlak.
        """
        # Hapus kartu yang sudah terlihat dari deck
        known_cards = set(hero_hole + board)
        remaining_deck = [c for c in self.full_deck if c not in known_cards]
        
        cards_needed_for_board = 5 - len(board)
        
        wins = 0
        ties = 0
        
        for _ in range(num_simulations):
            # 1. Ambil kartu acak untuk sisa meja dan kartu musuh
            sampled_cards = random.sample(remaining_deck, cards_needed_for_board + 2)
            
            simulated_board = board + sampled_cards[:cards_needed_for_board]
            villain_hole = sampled_cards[cards_needed_for_board:]
            
            # Gabungkan menjadi 7 kartu utuh
            hero_7_cards = hero_hole + simulated_board
            villain_7_cards = villain_hole + simulated_board
            
            # 2. EVALUASI MATEMATIKA MUTLAK (Cari kombinasi 5 kartu terbaik dari 7 kartu)
            # Ini akan mencegah error dari PokerKit yang meminta tepat 5 kartu
            hero_score = max(StandardHighHand(" ".join(combo)) for combo in itertools.combinations(hero_7_cards, 5))
            villain_score = max(StandardHighHand(" ".join(combo)) for combo in itertools.combinations(villain_7_cards, 5))
            
            # 3. Adu kekuatan
            if hero_score > villain_score:
                wins += 1
            elif hero_score == villain_score:
                ties += 1
                
        # Hitung probabilitas menang (Equity)
        equity = (wins + (ties / 2.0)) / num_simulations
        return equity

# --- UJI COBA INTERNAL MESIN MATEMATIKA ---
if __name__ == "__main__":
    simulator = MonteCarloSimulator()
    print("Menghitung Monte Carlo Simulation (1.000 Putaran)...")
    hero = ['Ah', 'Kh']
    board = ['9h', 'Ts', '9s']
    equity = simulator.calculate_equity(hero, board, num_simulations=1000)
    print(f"Probabilitas Menang (Equity): {equity * 100:.2f}%")