# src/utils.py
import torch
import tiktoken
import os

def set_seed(seed=42):
    """
    Imposta il seed per garantire la riproducibilità degli esperimenti.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Importante per la determinismo su GPU se disponibile
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_text_data(file_path):
    """Carica il contenuto del file di testo."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Il file {file_path} non è stato trovato.")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def manual_train_val_split(token_ids, train_ratio=0.9):
    """
    Divide il dataset in train e validation SENZA usare librerie esterne.
    Mantiene l'ordine sequenziale per evitare data leakage in serie temporali/testo.
    """
    split_idx = int(len(token_ids) * train_ratio)
    train_data = token_ids[:split_idx]
    val_data = token_ids[split_idx:]
    return train_data, val_data

class TextDataLoader:
    """Gestisce la tokenizzazione e la creazione dei batch."""
    def __init__(self, text, tokenizer_name="gpt2"):
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        # Tokenizzazione intero dataset
        self.encoded_data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        self.vocab_size = self.tokenizer.n_vocab

    def get_data(self):
        return self.encoded_data

    def decode(self, token_ids):
        # Gestione input singolo int o lista/tensore
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return self.tokenizer.decode(token_ids)

def get_batch(data, batch_size, context_length, device):
    """
    Estrae un batch casuale dal tensore dei dati.
    x: sequenza di input
    y: sequenza target (input shiftato di 1 a destra)
    """
    # Genera indici casuali validi
    ix = torch.randint(len(data) - context_length, (batch_size,))

    # Stack delle sequenze
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])

    return x.to(device), y.to(device)

def generate_text_simple(model, idx, max_new_tokens, context_length, device):
    """Funzione di generazione testo per monitorare i progressi."""
    model.eval() # Modalità valutazione
    idx = idx.to(device)
    for _ in range(max_new_tokens):
        # Ritaglia il contesto se supera la lunghezza massima gestita dal modello
        idx_cond = idx[:, -context_length:]

        with torch.no_grad():
            logits = model(idx_cond)

        # Prendi i logits dell'ultimo token
        logits = logits[:, -1, :]

        # Applica softmax per ottenere probabilità (opzionale se usi argmax, qui campioniamo)
        probs = torch.softmax(logits, dim=-1)

        # Seleziona il token con probabilità massima (Greedy decoding per semplicità)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)

        # Appendi il nuovo token alla sequenza
        idx = torch.cat((idx, idx_next), dim=1)

    model.train() # Torna in modalità training
    return idx