# src/trainer.py
import torch
from tqdm import tqdm  # Libreria esterna per progress bar
# from .utils import get_batch

def estimate_loss(model, train_data, val_data, batch_size, context_length, device, eval_iters=50):
    """Stima la loss su train e val set senza calcolare gradienti."""
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, context_length, device)
            with torch.no_grad():
                logits = model(X)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    Y.view(-1)
                )
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_model(model, train_data, val_data, optimizer, config, device):
    """
    Loop principale di training con monitoraggio tqdm.
    """
    train_losses = []
    val_losses = []

    steps = config['max_iters']
    print(f"Inizio training su dispositivo: {device}")

    # desc: descrizione a sinistra
    # dynamic_ncols: si adatta alla larghezza del terminale
    pbar = tqdm(range(steps), desc="Training GPT", dynamic_ncols=True)

    for iter in pbar:
        # 1. Valutazione periodica
        if iter % config['eval_interval'] == 0:
            losses = estimate_loss(
                model, train_data, val_data,
                config['batch_size'], config['context_length'],
                device
            )

            train_losses.append(losses['train'])
            val_losses.append(losses['val'])

            # Aggiorna la barra con le metriche correnti
            # Questo sostituisce il print() massivo
            pbar.set_postfix({
                "Tr_Loss": f"{losses['train']:.4f}",
                "Val_Loss": f"{losses['val']:.4f}"
            })

        # 2. Training step standard
        xb, yb = get_batch(train_data, config['batch_size'], config['context_length'], device)

        logits = model(xb)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            yb.view(-1)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    pbar.close() # Chiude correttamente la barra alla fine
    print("Training completato.")
    return train_losses, val_losses