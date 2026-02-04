# modello da 50 milioni di parametri circa
import torch
from model import GPTModel
from utils import TextDataLoader, generate_text_simple

def load_model(path, device):
    # Carica il checkpoint
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint['config']
    
    # Ricrea l'architettura
    model = GPTModel(config)
    
    # Carica i pesi addestrati
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, config

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "gpt_model_large.pth"
    
    print("Caricamento modello...")
    try:
        model, config = load_model(model_path, device)
        print("Modello caricato con successo!")
    except FileNotFoundError:
        print(f"Errore: Non trovo il file {model_path}. Esegui prima train_save.py")
        return

    # Inizializza tokenizer per decodifica
    loader = TextDataLoader("") # Stringa vuota, ci serve solo il tokenizer interno

    print("\n--- GPT LIVE DEMO (Digita 'exit' per uscire) ---")
    while True:
        start_text = input("\nPrompt: ")
        if start_text.lower() == 'exit':
            break
            
        # Codifica prompt
        token_ids = loader.tokenizer.encode(start_text)
        token_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0) # Batch dim
        
        # Generazione
        print("Generazione in corso...", end="\r")
        out_ids = generate_text_simple(
            model, 
            token_tensor, 
            max_new_tokens=50, 
            context_length=config['context_length'], 
            device=device
        )
        
        decoded_text = loader.decode(out_ids.squeeze(0))
        print(f"\nRISULTATO:\n{decoded_text}")

if __name__ == "__main__":
    main()