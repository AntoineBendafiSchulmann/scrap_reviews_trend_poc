import os
import re
from transformers import pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "Detail_justificatif_SM.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "Detail_justificatif_SM_with_sentiment.txt")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cmarkea/distilcamembert-base-sentiment"
)

def compute_sentiment_camembert(text):
    result = sentiment_pipeline(text[:512])[0]
    label = result["label"]

    if "5" in label or "4" in label:
        return "POSITIVE"
    elif "1" in label or "2" in label:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERREUR: Le fichier {INPUT_FILE} n'existe pas.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in:
        lines = [line.strip() for line in f_in if line.strip()]

    if not lines:
        print("ERREUR: Le fichier d'entrée est vide ou inexistant.")
        return

    print(f"DEBUG: {len(lines)} lignes lues dans {INPUT_FILE}.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for line in lines:
            sentiment = compute_sentiment_camembert(line)
            f_out.write(f"{line}\t{sentiment}\n")

    print(f"✅ Succès: {len(lines)} lignes analysées et écrites dans {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
