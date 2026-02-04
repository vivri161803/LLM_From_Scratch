import torch
import torch.optim as optim
from model import GPTModel
from trainer import train_model
from utils import TextDataLoader, load_text_data, set_seed, manual_train_val_split

# 1. Configurazione "Large" (relativa al progetto)
config = {
    "vocab_size": 50257,    # GPT-2 tokenizer size
    "context_length": 256,  # Lunghezza contesto (aumentabile a 512 o 1024 se hai VRAM)
    "emb_dim": 384,         # Aumentato (deve essere divisibile per n_heads)
    "n_heads": 6,
    "n_layers": 6,          # Più profondo
    "drop_rate": 0.1,
    "qkv_bias": False,
    "batch_size": 32,       # Riduci se ottieni OutOfMemory
    "max_iters": 2000,      # Numero di step di training
    "eval_interval": 200,
    "learning_rate": 3e-4
}

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    # 2. Caricamento Dati
    # Assicurati di avere un file 'input.txt' (es. un libro o dataset Shakespeare)
    raw_text = load_text_data("input.txt") 
    loader = TextDataLoader(raw_text)
    token_ids = loader.get_data()
    
    # Aggiorna vocab_size reale dal tokenizer
    config["vocab_size"] = loader.vocab_size 

    train_data, val_data = manual_train_val_split(token_ids)

    # 3. Inizializzazione Modello
    model = GPTModel(config)
    model.to(device)
    
    # Stampa numero parametri
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Modello inizializzato con {num_params/1e6:.2f}M parametri.")

    # 4. Training
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-1)
    
    train_hist, val_hist = train_model(
        model, train_data, val_data, optimizer, config, device
    )

    # 5. Salvataggio
    # Salviamo sia i pesi che la configurazione per poterli ricaricare correttamente
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, "gpt_model_large.pth")
    print("Modello salvato in 'gpt_model_large.pth'")

if __name__ == "__main__":
    main()