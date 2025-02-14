import os
import re
from transformers import pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
INPUT_FILE = os.path.join(BASE_DIR, "reviews_output.txt")  
OUTPUT_FILE = os.path.join(BASE_DIR, "reviews_with_sentiment_camembert.txt")  

print(f"DEBUG: Chemin d'entrée → {INPUT_FILE}")
print(f"DEBUG: Chemin de sortie → {OUTPUT_FILE}")

sentiment_pipeline = pipeline("sentiment-analysis", model="cmarkea/distilcamembert-base-sentiment")

def compute_sentiment_camembert(text):
    result = sentiment_pipeline(text[:512])[0]
    label = result["label"]

    if "5" in label or "4" in label:
        return "POSITIVE"
    elif "2" in label or "1" in label:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERREUR: Le fichier {INPUT_FILE} n'existe pas.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"DEBUG: {len(lines)} lignes lues dans {INPUT_FILE}")  

    if len(lines) == 0:
        print("ERREUR: `reviews_output.txt` est vide.")
        return

    items = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        line = re.sub(r"\s{2,}", "\t", line)
        p = line.split("\t")

        if len(p) != 7:
            print(f" ERREUR: Ligne mal formatée (cols={len(p)}) → {p}")
            continue

        rid, alias, name, rest_rating, review_id, review_rating, text = p  

        print(f"DEBUG: Avis extrait → {text}")  

        items.append({
            "restaurant_id": rid,
            "alias": alias,
            "name": name,
            "restaurant_rating": rest_rating,
            "review_id": review_id,
            "review_rating": review_rating,
            "text": text
        })

    if len(items) == 0:
        print(" ERREUR: Aucun avis valide trouvé dans `reviews_output.txt`.")
        return

    for it in items:
        it["sentiment"] = compute_sentiment_camembert(it["text"])
        print(f"DEBUG: Avis → {it['text']} | Sentiment détecté → {it['sentiment']}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for it in items:
            f.write(
                f"{it['restaurant_id']}\t"
                f"{it['alias']}\t"
                f"{it['name']}\t"
                f"{it['restaurant_rating']}\t"
                f"{it['review_id']}\t"
                f"{it['review_rating']}\t"
                f"{it['text']}\t"
                f"{it['sentiment']}\n"
            )

    print(f"✅ Succès: {len(items)} avis analysés avec `CamemBERT` et écrits dans `{OUTPUT_FILE}`.")

if __name__ == "__main__":
    main()
