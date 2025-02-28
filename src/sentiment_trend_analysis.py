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
    max_new_tokens=600,
    do_sample=False,
    num_beams=4,
    device_map="auto",
    repetition_penalty=1.4
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
# dans la liste de reviews, en renvoyant jusqu'à  30  mots  avant et après pour donner le contexte.
#arrete à  la première occurrence trouvée pour une tendance
def find_context(tend: str, reviews: list, window=30) -> str:
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

def postprocess_limited_sentences(text: str, max_sentences: int = 0) -> str:
    if max_sentences <= 0:
        return text.strip()

    sentences = [s.strip() for s in text.split('.') if s.strip()]
    limited = '. '.join(sentences[:max_sentences]).strip()
    if limited and not limited.endswith('.'):
        limited += '.'
    return limited

def remove_incomplete_ending(summary: str) -> str:
    sentences = [s.strip() for s in summary.split('.') if s.strip()]
    if not sentences:
        return summary.strip()

    last_sentence = sentences[-1]
    words = last_sentence.split()

    if len(words) < 5:
        sentences.pop()
    else:
        last_word = words[-1].lower().strip(",;!?'.-")
        if last_word.endswith("é") or last_word.endswith("è") or last_word.endswith("ai"):
            sentences.pop()

    final = '. '.join(sentences).strip()
    if final and not final.endswith('.'):
        final += '.'
    return final


def generate_summary_instruct(trends, sentiment_type, reviews):
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

    snippets = [find_context(trend, reviews) for trend in trends]
    combined_snippets = " ".join([snippet for snippet in snippets if snippet])

    UNDESIRABLE_IN_SNIPPETS = [
        "joue sur",
    ]
    for phrase in UNDESIRABLE_IN_SNIPPETS:
        combined_snippets = combined_snippets.replace(phrase, "")

    if sentiment_type == "positifs":
        prompt = (
            f"On observe que, d'après les retours des clients positifs, les avis positifs portent principalement sur {short_text},"
            f"avec ca je te donne un exemple illustrant ces tendances, évite de juste lister la tendance utilise pour décrire : {combined_snippets} "
            "utilise impérativement cet extrait pour reformuler la tendance de manière précise "
            "(par exemple la phrase: 'Sans frais sont souvent considérés comme inexistants' ne veux rien dire il faut un minimum utiliser le contexte pour que ca ait du sens. "
            "Rédige un seul paragraphe en bon français, sans mentionner les avis négatifs ni neutres. "
            "Contente-toi de décrire les points positifs rapportés dans les retours, de manière purement descriptive ne donne pas de conseils, "
            "n'extrapole pas, pas de mots comme incroyablement, super, garde un ton objectif, impartial. "
            "Privilégie un style narratif fluide, sans énumération brute ni connecteurs répétitifs (ex. 'En outre', 'De plus'). "
            "Ne t’arrête pas en plein mot, termine entièrement chacune des phrases, quand tu commences une idée tu la développes entièrement, "
            "et évite de répéter la même idée. "
            "Ne rédige pas de note ou d'explication sur la façon dont tu as rédigé le texte, "
            "et ne mentionne pas de validation comme ‘Vérifiez si le paragraphe respecte les exigences’ ou ‘Le paragraphe est écrit en bon français.’"
        )
    elif sentiment_type == "négatifs":
        prompt = (
            f"On observe que, d'après les retours des clients négatifs, les avis négatifs portent principalement sur {short_text},"
            f"avec ca je te donne un exemple illustrant ces tendances, évite de juste lister la tendance utilise pour décrire : {combined_snippets} "
            "utilise impérativement cet extrait pour reformuler la tendance de manière précise "
            "(par exemple la phrase: 'Sans frais sont souvent considérés comme inexistants' ne veux rien dire il faut un minimum utiliser le contexte pour que ca ait du sens. "
            "Rédige un seul paragraphe en bon français, sans mentionner les avis positifs ou neutres. "
            "Contente-toi de décrire les points négatifs rapportés dans les retours, de manière purement descriptive, "
            "n'extrapole pas, pas de mots comme incroyablement, super, garde un ton objectif, impartial. "
            "Privilégie un style narratif fluide, sans énumération brute ni connecteurs répétitifs (ex. 'En outre', 'De plus'). "
            "Ne t’arrête pas en plein mot, termine entièrement chacune des phrases, quand tu commences une idée tu la développes entièrement, "
            "et évite de répéter la même idée. "
            "Ne rédige pas de note ou d'explication sur la façon dont tu as rédigé le texte, "
            "et ne mentionne pas de validation comme ‘Vérifiez si le paragraphe respecte les exigences’ ou ‘Le paragraphe est écrit en bon français.’ "
            "N'écris pas que l'entreprise joue sur ses avis négatifs."
        )
    else:
        prompt = (
            f"On observe que, d'après les retours des clients neutres, les avis neutres concernent {short_text}, "
            f"avec ca je te donne un exemple illustrant ces tendances, évite de juste lister la tendance utilise pour décrire : {combined_snippets} "
            "N’écris pas que “joue sur…” ou toute tournure suggérant une volonté de la part de l’entreprise de manipuler les avis."
            "utilise impérativement cet extrait pour reformuler la tendance de manière précise "
            "(par exemple la phrase: 'Sans frais sont souvent considérés comme inexistants' ne veux rien dire il faut un minimum utiliser le contexte pour que ca ait du sens. "
            "Rédige un seul paragraphe en bon français, sans mentionner les avis positifs ou négatifs. "
            "Contente-toi de décrire les points neutres rapportés dans les retours, de manière purement descriptive, "
            "n'extrapole pas, pas de mots comme incroyablement, super, garde un ton objectif, impartial. "
            "Privilégie un style narratif fluide, sans énumération brute ni connecteurs répétitifs (ex. 'En outre', 'De plus'). "
            "Ne t’arrête pas en plein mot, termine entièrement chacune des phrases, quand tu commences une idée tu la développes entièrement, "
            "et évite de répéter la même idée. "
            "Ne rédige pas de note ou d'explication sur la façon dont tu as rédigé le texte, "
            "et ne mentionne pas de validation comme ‘Vérifiez si le paragraphe respecte les exigences’ ou ‘Le paragraphe est écrit en bon français.’"
        )

    out = text_generator(prompt)
    full_text = out[0]["generated_text"].strip()

    if full_text.startswith(prompt):
        summary = full_text[len(prompt):].strip()
    else:
        summary = full_text

    UNDESIRABLE_PHRASES = [
        "réponse:",
        "Réponse:",
        "joue sur",
    ]
    for phr in UNDESIRABLE_PHRASES:
        summary = summary.replace(phr, "")


    summary = re.sub(r"(?i)réponse:", "", summary).strip()

    max_sentences = len(trends) + 4
    summary = postprocess_limited_sentences(summary, max_sentences=max_sentences)
    summary = remove_incomplete_ending(summary)

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
    pos_summary = generate_summary_instruct(pos_trends, "positifs", pos_reviews)
    neg_summary = generate_summary_instruct(neg_trends, "négatifs", neg_reviews)
    neu_summary = generate_summary_instruct(neu_trends, "neutres", neu_reviews)
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
                f.write(f" (exemple de contexte : {snippet})")
            f.write("\n")
        f.write("\n")
        f.write("**Synthèse des avis négatifs :**\n")
        f.write(neg_summary + "\n\n")
        f.write("**Tendances extraites (négatif) :**\n")
        for t in neg_trends:
            snippet = find_context(t, neg_reviews)
            f.write(f"- {t}")
            if snippet:
                f.write(f" (exemple de contexte : {snippet})")
            f.write("\n")
        f.write("\n")
        f.write("**Synthèse des avis neutres :**\n")
        f.write(neu_summary + "\n\n")
        f.write("**Tendances extraites (neutre) :**\n")
        for t in neu_trends:
            snippet = find_context(t, neu_reviews)
            f.write(f"- {t}")
            if snippet:
                f.write(f" (exemple de contexte : {snippet})")
            f.write("\n")
        f.write("\n")
    print(f"✅ Résumé enregistré dans {TREND_OUTPUT_FILE}.")

if __name__ == "__main__":
    main() 