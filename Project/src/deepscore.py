import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from model import GPTModel
from trainer import estimate_loss # Usiamo estimate_loss invece del loop completo per brevità se preferisci
from utils import TextDataLoader, load_text_data, set_seed, manual_train_val_split, get_batch

# Configurazione Base (più leggera per fare l'esperimento velocemente)
base_config = {
    "vocab_size": 50257,
    "context_length": 64,   # Corto per velocità
    "emb_dim": 128,         # Piccolo per velocità
    "n_heads": 4,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "batch_size": 16,
    "max_iters": 300,       # Pochi step, giusto per vedere il trend iniziale
    "eval_interval": 300    # Valutiamo solo alla fine
}

def train_short_session(config, train_data, val_data, device):
    """Versione semplificata di train_model che restituisce solo la loss finale"""
    model = GPTModel(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # Training Loop rapido
    model.train()
    for i in range(config['max_iters']):
        xb, yb = get_batch(train_data, config['batch_size'], config['context_length'], device)
        logits = model(xb)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    # Stima finale loss
    final_metrics = estimate_loss(model, train_data, val_data, config['batch_size'], config['context_length'], device, eval_iters=20)
    return final_metrics['val']

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dati
    try:
        raw_text = load_text_data("input.txt")
    except FileNotFoundError:
        print("Crea un file 'input.txt' con del testo per lanciare l'esperimento.")
        return

    loader = TextDataLoader(raw_text)
    base_config["vocab_size"] = loader.vocab_size
    train_data, val_data = manual_train_val_split(loader.get_data())

    # Parametri dell'esperimento
    depths = [2, 4, 6, 8] # Variamo il numero di layer
    final_losses = []

    print(f"Inizio esperimento Deep vs Loss su {device}...")
    
    for d in depths:
        print(f"--- Training con Depth = {d} ---")
        # Aggiorna config
        current_config = base_config.copy()
        current_config["n_layers"] = d
        
        # Allena e prendi loss
        loss = train_short_session(current_config, train_data, val_data, device)
        final_losses.append(loss)
        print(f"Depth {d} -> Val Loss: {loss:.4f}")

    # Visualizzazione
    plt.figure(figsize=(8, 5))
    plt.plot(depths, final_losses, marker='o', linestyle='-', color='b')
    plt.title("Performance del Modello al variare della Profondità")
    plt.xlabel("Numero di Layer (n_layers)")
    plt.ylabel("Validation Loss (dopo 300 iters)")
    plt.grid(True)
    plt.xticks(depths)
    
    # Salva il grafico
    plt.savefig("depth_vs_loss.png")
    print("Grafico salvato come 'depth_vs_loss.png'")
    plt.show()

if __name__ == "__main__":
    main()