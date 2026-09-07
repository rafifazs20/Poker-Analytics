import os
import re

class HandHistoryParser:
    def __init__(self, logs_directory="data/logs"):
        """
        Data Pipeline untuk mengekstrak riwayat pertandingan mentah (.txt)
        dan merangkumnya menjadi dataset siap-latih untuk Deep CFR.
        """
        self.logs_directory = logs_directory
        # Buat folder jika belum ada
        if not os.path.exists(self.logs_directory):
            os.makedirs(self.logs_directory)

    def load_raw_logs(self):
        """Mencari semua file log .txt di folder data/logs"""
        log_files = [f for f in os.listdir(self.logs_directory) if f.endswith('.txt')]
        if not log_files:
            print(f"[PIPELINE] Tidak ada file log ditemukan di {self.logs_directory}")
            return []
            
        print(f"[PIPELINE] Menemukan {len(log_files)} file riwayat pertandingan.")
        return log_files

    def parse_coincrypto_format(self, filepath):
        """
        Parser Regex untuk membaca format teks log poker standar (seperti CoinPoker).
        Mengekstrak informasi krusial: Hole Cards, Board, dan Overbets musuh.
        """
        full_path = os.path.join(self.logs_directory, filepath)
        parsed_hands = []
        
        with open(full_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Pemisahan berdasarkan penanda dimulainya hand baru (contoh sederhana)
            hands = content.split("Hand #")
            
            for hand in hands:
                if not hand.strip(): continue
                
                # Ekstraksi menggunakan Regular Expression (Regex)
                # Mencari pola kartu: e.g., [Ah Kh] atau [9h Ts 9s]
                cards_dealt = re.findall(r'\[([2-9TJQKA][cdhs]\s?[2-9TJQKA]?[cdhs]?)\]', hand)
                
                # Mencari pola taruhan gertakan (Raise/Bet besar)
                raises = re.findall(r'raises to (\d+)', hand)
                
                if cards_dealt and raises:
                    parsed_hands.append({
                        "cards_seen": cards_dealt,
                        "max_raise_encountered": max(map(int, raises))
                    })
                    
        return parsed_hands

    def generate_training_dataset(self):
        """
        Fungsi utama pipeline: Mengambil teks mentah -> Parse -> Kembalikan sebagai dataset
        yang bisa dikonsumsi oleh trainer.py di malam hari (Retraining).
        """
        files = self.load_raw_logs()
        total_dataset = []
        
        for file in files:
            print(f"-> Mengekstrak data dari: {file}")
            extracted_data = self.parse_coincrypto_format(file)
            total_dataset.extend(extracted_data)
            
        print(f"[PIPELINE] Berhasil mengekstrak {len(total_dataset)} skenario untuk dipelajari AI.")
        return total_dataset

# --- UJI COBA PIPELINE LOKAL ---
if __name__ == "__main__":
    parser = HandHistoryParser()
    
    # Membuat file log dummy (Simulasi output dari situs poker online)
    dummy_log_path = "data/logs/session_2026_05.txt"
    with open(dummy_log_path, 'w', encoding='utf-8') as f:
        f.write("Hand #12345 - Blinds $10/$20\n")
        f.write("Dealt to Hero [Ah Kh]\n")
        f.write("Villain raises to 2000\n")
        f.write("Hero folds\n\n")
        
        f.write("Hand #12346 - Blinds $10/$20\n")
        f.write("Dealt to Hero [7c 8c]\n")
        f.write("Board [9h Ts 9s]\n")
        f.write("Villain raises to 500\n")
        f.write("Hero calls\n")

    # Jalankan parser
    dataset = parser.generate_training_dataset()
    for data in dataset:
        print(f"Data tersimpan: {data}")