import os
import torch
import matplotlib.pyplot as plt

# Import dai nostri moduli locali
# from src.utils import set_seed, load_text_data, TextDataLoader, manual_train_val_split, generate_text_simple
# from src.model import GPTModel
# from src.trainer import train_model

# Configurazioni (Simile GPT-2 Small ma scalato per demo veloce)
GPT_DEMO = {
    "vocab_size": 50257,    # TikToken BPE
    "context_length": 256,  # Ridotto per velocità (standard 1024)
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 6,          # Ridotto per velocità (standard 12)
    "drop_rate": 0.1,
    "qkv_bias": False,

    # Parametri Training
    "batch_size": 16,      # Dipende dalla GPU VRAM
    "max_iters": 500,      # Aumentare per convergenza reale
    "eval_interval": 100,
    "learning_rate": 3e-4
}

def main(GPT_CONFIG=GPT_DEMO, input_path = "./data/input.txt"):
    # 1. Setup Environment
    set_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Preparation
    # Assumiamo che il file sia in data/input.txt come da struttura


    # Creazione dummy file se non esiste (per testabilità immediata)
    if not os.path.exists(input_path):
        os.makedirs("data", exist_ok=True)
        print("File non trovato. Creazione file dummy 'The Verdict' (placeholder)...")
        dummy_text = "The verdict was read aloud in the court. " * 1000
        with open(input_path, "w") as f:
            f.write(dummy_text)

    raw_text = load_text_data(input_path)
    loader = TextDataLoader(raw_text)
    tokenized_data = loader.get_data()

    # Manual Split (90% Train, 10% Val)
    train_data, val_data = manual_train_val_split(tokenized_data, train_ratio=0.9)
    print(f"Token totali: {len(tokenized_data)}")
    print(f"Train size: {len(train_data)} | Val size: {len(val_data)}")

    # 3. Model Initialization
    model = GPTModel(GPT_CONFIG)
    model.to(device)

    # Conta parametri
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Modello inizializzato con {num_params/1e6:.2f}M parametri.")

    # 4. Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=GPT_CONFIG['learning_rate'])

    train_hist, val_hist = train_model(
        model, train_data, val_data, optimizer, GPT_CONFIG, device
    )

    # 5. Risultati e Plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_hist, label='Train Loss')
    plt.plot(val_hist, label='Validation Loss')
    plt.xlabel('Eval Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Dynamics')
    plt.savefig('loss_plot.png')
    print("Plot salvato come 'loss_plot.png'")

    # 6. Test Generazione (Inference)
    print("\n--- Generazione Testo Post-Training ---")
    start_context = "The verdict was"
    encoded_start = torch.tensor(loader.tokenizer.encode(start_context)).unsqueeze(0)

    generated_idx = generate_text_simple(
        model,
        encoded_start,
        max_new_tokens=50,
        context_length=GPT_CONFIG['context_length'],
        device=device
    )

    decoded_text = loader.decode(generated_idx[0])
    print(f"Input: {start_context}")
    print(f"Generated: {decoded_text}")

    # 7. Salvataggio Modello
    torch.save(model.state_dict(), "gpt_model_checkpoint.pth")
    print("Modello salvato.")