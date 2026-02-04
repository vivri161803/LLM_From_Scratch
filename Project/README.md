# Struttura della cartella

Questi sono tutti i file presenti in questa cartella. La cartella `/src` contiene gli snippet di codice che sono stati utilizzati per la stesura della relazione: `CL_Relazione.ipynb`. I dati utilizzati per il training si trovano in `/data`, mentre nella cartella `/slides` troviamo la presentazione, sia in Markdown che in pdf. Lo script `main.py` orchestra tutto il codice qui presente, permettendo di allenare un modello GPT2-like, fornendone la configurazione strutturale. Tutto il codice viene attentamente commentato in `CL_Relazione.ipynb`.

```bash
├── CL_Relazione.ipynb
├── data
│   ├── input.txt
│   └── moby.txt
├── main.py
├── plot
│   ├── Depth_Score.png
│   ├── GPT_BALANCED.png
│   ├── GPT_DEEP.png
│   ├── GPT_DEMO.png
│   ├── GPT_REAL.png
│   └── wall-e.jpg
├── pyproject.toml
├── README.md
├── slides
│   ├── masking.png
│   ├── model.png
│   ├── slides_proj.md
│   └── slides_proj.pdf
├── src
│   ├── deepscore.py
│   ├── gpt_model_large.pth
│   ├── live_demo.py
│   ├── model.py
│   ├── trainer.py
│   ├── trainsave.py
│   └── utils.py
└── uv.lock
```

## 🛠️ Installazione e Requisiti

Il codice richiede Python 3.8+ e le seguenti librerie:

```bash
pip install torch tiktoken tqdm matplotlib
```

## 📖 Come Utilizzare il Progetto

1. Preparazione dei dati
Inserisci il tuo file di testo in data/input.txt. Se il file non esiste, lo script main.py genererà automaticamente un file di esempio per testare la pipeline.

2. Addestramento
Per avviare l'addestramento con la configurazione di default:

```bash
python main.py
```

1. Configurazione Modello
Puoi modificare i parametri nel main.py all'interno del dizionario GPT_CONFIG. Esempio per GPU limitate:

```Python
GPT_CONFIG = {
    "context_length": 256,
    "emb_dim": 768,
    "n_layers": 6,
    "batch_size": 16, # Regolare in base alla VRAM
    "learning_rate": 3e-4
}
```

## 📊 Monitoraggio Risultati

Durante il training, verrà visualizzata una barra di avanzamento (tqdm) con la loss in tempo reale. Al termine, il sistema genererà:

- loss_plot.png: Grafico dell'andamento della loss (Train vs Val).
- gpt_model_checkpoint.pth: I pesi del modello salvati.
- generazione Testo: Una demo di completamento partendo da un prompt predefinito.
