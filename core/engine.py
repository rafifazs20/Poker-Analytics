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

    def parse_quick_input(self, command_string):
        """Memproses input kilat dari pengguna."""
        parts = command_string.strip().split()
        if not parts: return False
        
        cmd = parts[0].lower()
        try:
            if cmd == 'h':
                self.state.deal_hole(parts[1])
                print(f"[ENGINE] Kartu tangan dibagikan: {parts[1]}")
            elif cmd == 'b':
                self.state.deal_board(parts[1])
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
        """Mengubah kondisi meja menjadi PyTorch Tensor 106-dimensi."""
        hole_tensor = torch.zeros(52)
        board_tensor = torch.zeros(52)
        
        if self.state.hole_cards and len(self.state.hole_cards) > 0:
            for card in self.state.hole_cards[0]:
                idx = card.rank.index * 4 + card.suit.index
                hole_tensor[idx] = 1.0
                
        for card in self.state.board_cards:
            idx = card.rank.index * 4 + card.suit.index
            board_tensor[idx] = 1.0

        pot = torch.tensor([self.state.total_pot / (self.starting_stack * 2)], dtype=torch.float32)
        to_call = torch.tensor([self.state.checking_or_calling_amount / self.starting_stack], dtype=torch.float32)
        
        return torch.cat([hole_tensor, board_tensor, pot, to_call])