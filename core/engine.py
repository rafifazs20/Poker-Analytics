import torch
from pokerkit import Automation, NoLimitTexasHoldem

class DeepPokerDSS:
    def __init__(self, bb_amount=20, starting_stack=1000):
        self.bb_amount = bb_amount
        self.starting_stack = starting_stack
        
        # State Game PokerKit
        self.state = NoLimitTexasHoldem.create_state(
            (Automation.ANTE_POSTING, Automation.BET_COLLECTION, Automation.BLIND_OR_STRADDLE_POSTING),
            True, 0, (bb_amount//2, bb_amount), bb_amount, (starting_stack, starting_stack), 2
        )
        
        # FITUR BARU: Memori string murni agar kebal dari perubahan API PokerKit
        self.my_hole_cards = []
        self.my_board_cards = []

    def get_pot_size(self):
        return (self.starting_stack * 2) - sum(self.state.stacks)

    def get_call_amount(self):
        if not self.state.bets: return 0
        return max(self.state.bets) - min(self.state.bets)

    def parse_quick_input(self, command_string):
        parts = command_string.strip().split()
        if not parts: return False
        
        cmd = parts[0].lower()
        try:
            if cmd == 'h':
                self.state.deal_hole(parts[1])
                # Pecah input 'AhKh' menjadi list string murni ['Ah', 'Kh']
                self.my_hole_cards = [parts[1][i:i+2] for i in range(0, len(parts[1]), 2)]
                print(f"[ENGINE] Kartu tangan dibagikan: {parts[1]}")
            elif cmd == 'b':
                self.state.deal_board(parts[1])
                # Pecah input flop '9hTs9s' menjadi ['9h', 'Ts', '9s']
                new_board = [parts[1][i:i+2] for i in range(0, len(parts[1]), 2)]
                self.my_board_cards.extend(new_board)
                print(f"[ENGINE] Kartu meja dibuka: {parts[1]}")
            elif cmd == 'bet':
                amount = int(parts[1])
                self.state.complete_bet_or_raise_to(amount)
                print(f"[ENGINE] Musuh Raise/Bet ke: {amount}")
            elif cmd == 'c':
                self.state.check_or_call()
                print("[ENGINE] Check / Call.")
            else:
                print("[ERROR] Perintah tidak dikenali (Gunakan: h, b, bet, c)")
                return False
            return True
        except Exception as e:
            print(f"[ERROR] Gerakan ilegal (Aturan PokerKit menolak): {e}")
            return False

    def generate_tensor_state(self):
        """Mengubah kondisi meja menjadi PyTorch Tensor 106-dimensi (Murni pakai input lokal)"""
        hole_tensor = torch.zeros(52)
        board_tensor = torch.zeros(52)
        
        ranks = '23456789TJQKA'
        suits = 'cdhs'
        
        # Konversi array string kita sendiri ke indeks tensor (Anti Error)
        for card_str in self.my_hole_cards:
            r_idx = ranks.index(card_str[0].upper())
            s_idx = suits.index(card_str[1].lower())
            hole_tensor[r_idx * 4 + s_idx] = 1.0
            
        for card_str in self.my_board_cards:
            r_idx = ranks.index(card_str[0].upper())
            s_idx = suits.index(card_str[1].lower())
            board_tensor[r_idx * 4 + s_idx] = 1.0

        pot = torch.tensor([self.get_pot_size() / (self.starting_stack * 2)], dtype=torch.float32)
        to_call = torch.tensor([self.get_call_amount() / self.starting_stack], dtype=torch.float32)
        
        return torch.cat([hole_tensor, board_tensor, pot, to_call])