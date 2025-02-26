import os
import re
import torch
import spacy
import pandas as pd
from collections import Counter
from keybert import KeyBERT
from yake import KeywordExtractor
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
nlp = spacy.load("fr_core_news_md")
kw_extractor = KeywordExtractor(lan="fr", n=3, top=30)
kw_model = KeyBERT("all-mpnet-base-v2")

print(f"Chargement du modèle : {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=120,
    do_sample=False,
    num_beams=4,
    device_map="auto"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TREND_INPUT_FILE = os.path.join(BASE_DIR, "trustpilot_reviews_with_sentiment_camembert.txt")
TREND_OUTPUT_FILE = os.path.join(BASE_DIR, "trustpilot_sentiment_trends.txt")

def clean_text(txt: str) -> str:
    txt = txt.lower()
    txt = re.sub(r"[^\w\s']", "", txt)
    return txt.strip()

def refine_trends(trends):
    if not trends or trends == ["Aucune tendance détectée"]:
        return ["Aucune tendance détectée"]
    refined = []
    seen = set()
    for t in trends:
        t = t.replace("rge vendre de faux panneaux", "arnaque sur les panneaux solaires")
        t = t.replace("cofidis et société planète écologique", "cofidis et une entreprise écologique")
        t = t.replace("montant mais globalement la conseillère", "avis sur le montant et le conseil")
        doc = nlp(t)
        if len(doc) > 2 and not any(nlp(s).similarity(doc) > 0.85 for s in seen):
            seen.add(t)
            refined.append(t)
    return refined

# Renvoie le premier extrait de texte (snippet) contenant la tendance 'tend' 
# dans la liste de reviews, en renvoyant jusqu'à  20  mots  avant et après pour donner le contexte.
#arrete à  la première occurrence trouvée pour une tendance
def find_context(tend: str, reviews: list, window=20) -> str:
    tend_words = tend.lower().split()
    n = len(tend_words)
    for review in reviews:
        words = review.split()
        for i in range(len(words) - n + 1):
            segment_lower = [w.lower() for w in words[i:i+n]]
            if segment_lower == tend_words:
                start = max(0, i - window)
                end = min(len(words), i + n + window)
                snippet = words[start:end]
                return " ".join(snippet)
    return None

def extract_trends(texts, sentiment, top_n=20):
    if not texts:
        return ["Aucune tendance détectée"]
    all_trends = []
    for txt in texts:
        yake_kws = kw_extractor.extract_keywords(txt)
        keybert_kws = kw_model.extract_keywords(
            txt,
            keyphrase_ngram_range=(2, 8),
            stop_words="french",
            top_n=5
        )
        all_trends.extend([k[0] for k in yake_kws if 2 < len(k[0].split()) <= 6])
        all_trends.extend([k[0] for k in keybert_kws if 2 < len(k[0].split()) <= 6])
    freq = Counter(all_trends)
    extracted = [p for p, _ in freq.most_common(top_n) if p not in ["bonjour", "rien dire", "merci"]]
    refined = refine_trends(extracted)
    blacklist = {
        "oreille","bonjour","rien dire","merci","avis","service","produit","client",
        "commande","livraison","qualité","prix","problème","réponse","temps","jour",
        "mois","année","site","achat","article","boutique","contact","expérience",
        "satisfaction","équipe","personnel","support","aide","solution","proposition",
        "demande","information","délai","paiement","facture","remboursement","réclamation",
        "conseil","commentaire","note","évaluation","feedback","fois que je fais appel"
    }
    final_list = []
    for trend in refined:
        nb_words = len(trend.split())
        if 3 <= nb_words <= 8 and not any(bad in trend.lower() for bad in blacklist):
            final_list.append(trend)
    return final_list if final_list else ["Aucune tendance détectée"]

def postprocess_limited_sentences(text: str, max_sentences=3) -> str:
    sentences = text.split('.')
    limited = '.'.join(sentences[:max_sentences]).strip()
    if limited and not limited.endswith('.'):
        limited += '.'
    return limited

def generate_summary_instruct(trends, sentiment_type):
    if not trends or trends == ["Aucune tendance détectée"]:
        return "Aucune idée générale détectée."

    if len(trends) == 1:
        short_text = trends[0]
    elif len(trends) == 2:
        short_text = f"{trends[0]} et {trends[1]}"
    elif len(trends) >= 6:
        short_text = (
            f"plusieurs aspects majeurs, dont {trends[0]}, {trends[1]}, {trends[2]}, "
            f"{trends[3]}, {trends[4]} et {trends[5]}, etc."
        )
    elif len(trends) >= 4:
        short_text = (
            f"plusieurs aspects, comme {trends[0]}, {trends[1]}, {trends[2]} et {trends[3]}, entre autres"
        )
    else:
        short_text = f"plusieurs points récurrents, dont {trends[0]} et {trends[1]} entre autres"

    if sentiment_type == "positifs":
        prompt = (
            f"D'après les retours des clients, les avis positifs font ressortir {short_text}. "
            "Rédige trois phrases maximum en bon français, en reliant les idées avec des connecteurs, "
            "sans énumération brute, et sans mentionner les avis négatifs ni neutres. "
            "Ne termine pas tes phrases de manière incomplète : arrête-toi net après trois phrases. "
            "Évite de répéter plusieurs fois la même expression et corrige toute faute de frappe (par exemple => 'débloquer')."
        )
    elif sentiment_type == "négatifs":
        prompt = (
           f"D'après les retours des clients, les avis négatifs soulignent principalement {short_text}, "
            "notamment la lenteur du déblocage des fonds et le manque de respect. "
            "Rédige trois phrases maximum en bon français, "
            "sans énumération brute, et veille à inclure les problèmes concrets, "
            "sans mentionner les avis positifs ou neutres, et sans finir tes phrases incomplètement. "
            "Évite de répéter plusieurs fois la même expression et corrige toute faute de frappe (par exemple => 'débloquer')."
        )
    else:
        prompt = (
            f"D'après les retours des clients, les avis neutres portent sur {short_text}. "
            "Rédige trois phrases maximum en bon français, en intégrant naturellement ces idées "
            "sans liste ni énumération, et sans mentionner les avis positifs ou négatifs. "
            "Ne termine pas tes phrases de manière incomplète : arrête-toi net après trois phrases. "
            "Évite de répéter plusieurs fois la même expression et corrige toute faute de frappe (par exemple 'déblocer' => 'débloquer')."
        )

    out = text_generator(prompt)
    full_text = out[0]["generated_text"].strip()

    if full_text.startswith(prompt):
        summary = full_text[len(prompt):].strip()
    else:
        summary = full_text

    # Pour éviter que la réponse commence par "Réponse:"
    summary = summary.replace("Réponse:", "").strip()

    summary = postprocess_limited_sentences(summary, max_sentences=3)

    return summary

def main():
    print(" Lecture du fichier de tendances...")
    df = pd.read_csv(TREND_INPUT_FILE, sep="\t", header=None, names=["text","sentiment"])
    print("✅ Fichier chargé avec succès.")
    df["clean_text"] = df["text"].apply(clean_text)
    print(" Nettoyage des avis terminé.")
    pos_reviews = df[df["sentiment"]=="POSITIVE"]["clean_text"].tolist()
    neg_reviews = df[df["sentiment"]=="NEGATIVE"]["clean_text"].tolist()
    neu_reviews = df[df["sentiment"]=="NEUTRAL"]["clean_text"].tolist()
    print(" Début de l'extraction des tendances...")
    pos_trends = extract_trends(pos_reviews, "positif")
    neg_trends = extract_trends(neg_reviews, "négatif")
    neu_trends = extract_trends(neu_reviews, "neutre")
    print(" Génération des résumés ...")
    pos_summary = generate_summary_instruct(pos_trends, "positifs")
    neg_summary = generate_summary_instruct(neg_trends, "négatifs")
    neu_summary = generate_summary_instruct(neu_trends, "neutres")
    total = len(df)
    with open(TREND_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("Répartition des sentiments :\n")
        f.write(f"Positifs : {len(pos_reviews)} avis ({len(pos_reviews)/total*100:.2f}%)\n")
        f.write(f"Négatifs : {len(neg_reviews)} avis ({len(neg_reviews)/total*100:.2f}%)\n")
        f.write(f"Neutres  : {len(neu_reviews)} avis ({len(neu_reviews)/total*100:.2f}%)\n\n")
        f.write("**Synthèse des avis positifs :**\n")
        f.write(pos_summary + "\n\n")
        f.write("**Tendances extraites (positif) :**\n")
        for t in pos_trends:
            snippet = find_context(t, pos_reviews)
            f.write(f"- {t}")
            if snippet:
                f.write(f" (extrait : {snippet})")
            f.write("\n")
        f.write("\n")
        f.write("**Synthèse des avis négatifs :**\n")
        f.write(neg_summary + "\n\n")
        f.write("**Tendances extraites (négatif) :**\n")
        for t in neg_trends:
            snippet = find_context(t, neg_reviews)
            f.write(f"- {t}")
            if snippet:
                f.write(f" (extrait : {snippet})")
            f.write("\n")
        f.write("\n")
        f.write("**Synthèse des avis neutres :**\n")
        f.write(neu_summary + "\n\n")
        f.write("**Tendances extraites (neutre) :**\n")
        for t in neu_trends:
            snippet = find_context(t, neu_reviews)
            f.write(f"- {t}")
            if snippet:
                f.write(f" (extrait : {snippet})")
            f.write("\n")
        f.write("\n")
    print(f"✅ Résumé enregistré dans {TREND_OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
