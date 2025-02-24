import os
import re
import torch
import spacy
import pandas as pd
from collections import Counter
from keybert import KeyBERT
from yake import KeywordExtractor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

nlp = spacy.load("fr_core_news_md")

model_name = "plguillou/t5-base-fr-sum-cnndm"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TREND_INPUT_FILE = os.path.join(BASE_DIR, "trustpilot_reviews_with_sentiment_camembert.txt")
TREND_OUTPUT_FILE = os.path.join(BASE_DIR, "trustpilot_sentiment_trends.txt")

kw_extractor = KeywordExtractor(lan="fr", n=5, top=50)
kw_model = KeyBERT("all-mpnet-base-v2")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s']", "", text)
    return text.strip()

def extract_trends(texts, sentiment, top_n=20):
    if not texts:
        return ["Aucune tendance détectée"]

    all_trends = []
    for text in texts:
        yake_keywords = kw_extractor.extract_keywords(text)
        keybert_keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(3, 6),
                                                     stop_words="french", top_n=5)

        all_trends.extend([kw[0] for kw in yake_keywords if len(kw[0].split()) > 4])
        all_trends.extend([kw[0] for kw in keybert_keywords if len(kw[0].split()) > 4])

    word_freq = Counter(all_trends)
    extracted_trends = [
        phrase for phrase, _ in word_freq.most_common(top_n)
        if phrase not in ["bonjour", "rien dire", "merci"]
    ]

    refined_trends = refine_trends(extracted_trends)
    filtered_trends = [
        trend for trend in refined_trends
        if len(trend.split()) > 3
        and not any(word in trend.lower() for word in ["oreille", "bonjour", "rien dire", "merci"])
    ]

    print(f"✅ Tendances filtrées ({sentiment}) :", filtered_trends if filtered_trends else "Aucune tendance détectée")
    return filtered_trends if filtered_trends else ["Aucune tendance détectée"]

def refine_trends(trends):
    if not trends or trends == ["Aucune tendance détectée"]:
        return ["Aucune tendance détectée"]

    refined_trends = []
    seen_trends = set()

    for trend in trends:
        trend = trend.replace("rge vendre de faux panneaux", "arnaque sur les panneaux solaires")
        trend = trend.replace("cofidis et société planète écologique", "cofidis et une entreprise écologique")
        trend = trend.replace("montant mais globalement la conseillère", "avis sur le montant et le conseil")

        doc = nlp(trend)
        if len(doc) > 2 and not any(nlp(existing).similarity(doc) > 0.85 for existing in seen_trends):
            seen_trends.add(trend)
            refined_trends.append(trend)

    print(f"✅ Tendances affinées : {len(refined_trends)} tendances significatives.")
    return refined_trends

def double_pass_summary(text):
    first_pass = summarizer(
        text,
        max_length=80,
        min_length=25,
        do_sample=False
    )[0]["summary_text"].strip()

    second_prompt = f"Transforme ce texte en une seule phrase fluide: {first_pass}"
    second_pass = summarizer(
        second_prompt,
        max_length=60,
        min_length=15,
        do_sample=False
    )[0]["summary_text"].strip()

    if not second_pass.endswith("."):
        second_pass += "."

    return second_pass

def generate_summary(trends, sentiment_type):
    if not trends or trends == ["Aucune tendance détectée"]:
        return "Aucune idée générale détectée."

    text_to_summarize = "; ".join(trends)

    summary = double_pass_summary(text_to_summarize)
    return summary

def format_summary(trends):
    if not trends or trends == ["Aucune tendance détectée"]:
        return "Aucune tendance détectée."
    return "\n".join(f"- {trend}" for trend in trends)

def main():
    print("🛠️ Test rapide du modèle...")
    test_trends = ["réponse rapide", "dossier traité rapidement", "satisfait du service"]
    test_summary = generate_summary(test_trends, "positifs")
    print(f"✅ Test du modèle : {test_summary}")

    print(" Lecture du fichier de tendances...")
    df = pd.read_csv(TREND_INPUT_FILE, sep="\t", header=None, names=["text", "sentiment"])
    print("✅ Fichier chargé avec succès.")

    df["clean_text"] = df["text"].apply(clean_text)
    print(" Nettoyage des avis terminé.")

    pos_reviews = df[df["sentiment"] == "POSITIVE"]["clean_text"].tolist()
    neg_reviews = df[df["sentiment"] == "NEGATIVE"]["clean_text"].tolist()
    neu_reviews = df[df["sentiment"] == "NEUTRAL"]["clean_text"].tolist()

    print(" Début de l'extraction des tendances...")
    pos_trends = extract_trends(pos_reviews, "positif")
    neg_trends = extract_trends(neg_reviews, "négatif")
    neu_trends = extract_trends(neu_reviews, "neutre")

    print(" Affinage des tendances...")
    pos_trends = refine_trends(pos_trends)
    neg_trends = refine_trends(neg_trends)
    neu_trends = refine_trends(neu_trends)

    print(f"🔍 Vérification - Tendances positives envoyées au résumé : {pos_trends}")
    print(f"🔍 Vérification - Tendances négatives envoyées au résumé : {neg_trends}")
    print(f"🔍 Vérification - Tendances neutres envoyées au résumé : {neu_trends}")

    print(" Génération des résumés avec le LLM local...")
    pos_summary = generate_summary(pos_trends, "positifs")
    neg_summary = generate_summary(neg_trends, "négatifs")
    neu_summary = generate_summary(neu_trends, "neutres")

    print("Enregistrement des résultats dans le fichier de sortie...")
    with open(TREND_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("Répartition des sentiments :\n")
        f.write(f"Positifs : {len(pos_reviews)} avis ({len(pos_reviews) / len(df) * 100:.2f}%)\n")
        f.write(f"Négatifs : {len(neg_reviews)} avis ({len(neg_reviews) / len(df) * 100:.2f}%)\n")
        f.write(f"Neutres  : {len(neu_reviews)} avis ({len(neu_reviews) / len(df) * 100:.2f}%)\n\n")

        f.write("**Synthèse des avis positifs :**\n")
        f.write(pos_summary + "\n\n")
        f.write("**Tendances extraites (positif) :**\n")
        f.write(format_summary(pos_trends) + "\n\n")

        f.write("**Synthèse des avis négatifs :**\n")
        f.write(neg_summary + "\n\n")
        f.write("**Tendances extraites (négatif) :**\n")
        f.write(format_summary(neg_trends) + "\n\n")

        f.write("**Synthèse des avis neutres :**\n")
        f.write(neu_summary + "\n\n")
        f.write("**Tendances extraites (neutre) :**\n")
        f.write(format_summary(neu_trends) + "\n\n")

    print(f"✅ Résumé enregistré dans `{TREND_OUTPUT_FILE}`.")

if __name__ == "__main__":
    print(generate_summary(["Service rapide et efficace", "Réponse immédiate"], "positifs"))
    main()
