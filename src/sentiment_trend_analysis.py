import os
import re
import torch
import faiss
import spacy
import pandas as pd
import time
import json
from collections import Counter
from keybert import KeyBERT
from yake import KeywordExtractor
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
nlp = spacy.load("fr_core_news_md")
kw_extractor = KeywordExtractor(lan="fr", n=3, top=30)
kw_model = KeyBERT("all-mpnet-base-v2")

encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

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
    max_new_tokens=400,
    do_sample=False,
    num_beams=1,
    device_map="auto",
    repetition_penalty=1.2,
    temperature=0.1,
    top_p=0.7
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_DIR = os.path.join(BASE_DIR, "..", "config")

TREND_INPUT_FILE = os.path.join(BASE_DIR, "trustpilot_reviews_with_sentiment_camembert.txt")
TREND_OUTPUT_FILE = os.path.join(BASE_DIR, "trustpilot_sentiment_trends.txt")

SYNONYMS_FILE  = os.path.join(CONFIG_DIR, "synonyms_map.json")
REPLACE_FILE   = os.path.join(CONFIG_DIR, "replace_map.json")
BLACKLIST_FILE = os.path.join(CONFIG_DIR, "blacklist.json")

def load_config():
    with open(SYNONYMS_FILE, "r", encoding="utf-8") as f:
        synonyms_map = json.load(f)

    with open(REPLACE_FILE, "r", encoding="utf-8") as f:
        replace_map = json.load(f)

    with open(BLACKLIST_FILE, "r", encoding="utf-8") as f:
        blacklist_data = json.load(f)
        blacklist = set(blacklist_data)

    return synonyms_map, replace_map, blacklist


def clean_text(txt: str) -> str:
    txt = txt.lower()
    txt = re.sub(r"[^\w\s']", "", txt)
    return txt.strip()

def is_substantive_enough(phrase: str) -> bool:
    doc = nlp(phrase)
    for token in doc:
        if token.pos_ in ("NOUN", "VERB", "PROPN"):
            return True
    return False

def unify_synonyms(trend: str, synonyms_map: dict) -> str:
    t_lower = trend.lower()
    for canon_label, syn_list in synonyms_map.items():
        for syn in syn_list:
            if syn in t_lower:
                return canon_label
    return trend

def refine_trends(trends, replace_map, synonyms_map):
    if not trends or trends == ["Aucune tendance détectée"]:
        return ["Aucune tendance détectée"]
    refined = []
    seen = set()
    for t in trends:
        for old_str, new_str in replace_map.items():
            t = t.replace(old_str, new_str)

        t = unify_synonyms(t, synonyms_map)
        doc = nlp(t)
        if len(doc) > 2 and not any(nlp(s).similarity(doc) > 0.85 for s in seen):
            seen.add(t)
            refined.append(t)
    return refined

def extract_trends(texts, sentiment, top_n=20):
    if not texts:
        return ["Aucune tendance détectée"]

    synonyms_map, replace_map, blacklist = load_config()
    # print("blacklist chargée :", blacklist)

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
    extracted = [
        p for (p, c) in freq.most_common()
        if c >= 3 and p not in blacklist
    ][:top_n]

    refined = refine_trends(extracted, replace_map, synonyms_map)

    candidate_trends = []
    for t in refined:
        if is_substantive_enough(t):
            candidate_trends.append(t)
    refined = candidate_trends

    final_list = []
    for trend in refined:
        nb_words = len(trend.split())
        if 3 <= nb_words <= 8 and not any(bad in trend.lower() for bad in blacklist):
            final_list.append(trend)

    return final_list if final_list else ["Aucune tendance détectée"]


def build_chunks(texts, min_words=5):
    chunks = []
    for txt in texts:
        paragraphs = txt.split(".")
        for p in paragraphs:
            p = p.strip()
            if len(p.split()) >= min_words:
                chunks.append(p)
    return chunks

def build_vector_store(chunks):
    embeddings = encoder.encode(chunks, convert_to_tensor=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def retrieve_passages_for_trend(trend, index, embeddings, chunks, top_k=3):
    q_emb = encoder.encode([trend], convert_to_tensor=False)
    distances, ids = index.search(q_emb, top_k)
    results = []
    for idx in ids[0]:
        results.append(chunks[idx])
    return results


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


def rag_generate_summary(trends, sentiment_type, index, embeddings, chunks):
    if not trends or trends == ["Aucune tendance détectée"]:
        return "Aucune idée générale détectée."

    short_text = ", ".join(trends)
    all_passages = []
    for t in trends:
        top_passages = retrieve_passages_for_trend(t, index, embeddings, chunks, top_k=3)
        all_passages.extend(top_passages)

    best_passages = all_passages[:10]
    context_block = "\n".join(best_passages)

    if sentiment_type == "positifs":
        prompt = (
            f"Voici plusieurs extraits d'avis positifs sur : {short_text}.\n"
            f"{context_block}\n\n"
            "Rédige un unique paragraphe concis (3 à 5 phrases). "
            "Ne parle pas des avis négatifs ou neutres. "
            "Évite toute formule de politesse ou liste de points. "
            "Reste descriptif et objectif."
        )
    elif sentiment_type == "négatifs":
        prompt = (
            f"Voici plusieurs extraits d'avis négatifs sur : {short_text}.\n"
            f"{context_block}\n\n"
            "Rédige un unique paragraphe concis (3 à 5 phrases). "
            "Ne parle pas des avis positifs ou neutres. "
            "Évite toute formule de politesse ou liste de points. "
            "Reste descriptif et objectif."
        )
    else:
        prompt = (
            f"Voici plusieurs extraits d'avis neutres sur : {short_text}.\n"
            f"{context_block}\n\n"
            "Rédige un unique paragraphe concis (3 à 5 phrases). "
            "Ne parle pas des avis positifs ou négatifs. "
            "Évite toute formule de politesse ou liste de points. "
            "Reste descriptif et objectif."
        )

    out = text_generator(prompt)
    full_text = out[0]["generated_text"].strip()

    if full_text.startswith(prompt):
        summary = full_text[len(prompt):].strip()
    else:
        summary = full_text

    summary = re.sub(r"(?i)si tu veux continuer.*", "", summary)
    summary = re.sub(r"(?i)remplacez le paragraphe.*", "", summary)

    max_sentences = 5
    summary = postprocess_limited_sentences(summary, max_sentences)
    summary = remove_incomplete_ending(summary)
    return summary

def main():
    start_time = time.time()

    df = pd.read_csv(TREND_INPUT_FILE, sep="\t", header=None, names=["text","sentiment"])
    df["clean_text"] = df["text"].apply(clean_text)

    pos_reviews = df[df["sentiment"]=="POSITIVE"]["clean_text"].tolist()
    neg_reviews = df[df["sentiment"]=="NEGATIVE"]["clean_text"].tolist()
    neu_reviews = df[df["sentiment"]=="NEUTRAL"]["clean_text"].tolist()

    total = len(df)

    pos_trends = extract_trends(pos_reviews, "positif")
    neg_trends = extract_trends(neg_reviews, "négatif")
    neu_trends = extract_trends(neu_reviews, "neutre")

    all_texts = pos_reviews + neg_reviews + neu_reviews
    chunks = build_chunks(all_texts, min_words=5)
    index, embeddings = build_vector_store(chunks)

    pos_summary = rag_generate_summary(pos_trends, "positifs", index, embeddings, chunks)
    neg_summary = rag_generate_summary(neg_trends, "négatifs", index, embeddings, chunks)
    neu_summary = rag_generate_summary(neu_trends, "neutres", index, embeddings, chunks)

    with open(TREND_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("Répartition des sentiments :\n")
        f.write(f"Positifs : {len(pos_reviews)} avis ({len(pos_reviews)/total*100:.2f}%)\n")
        f.write(f"Négatifs : {len(neg_reviews)} avis ({len(neg_reviews)/total*100:.2f}%)\n")
        f.write(f"Neutres  : {len(neu_reviews)} avis ({len(neu_reviews)/total*100:.2f}%)\n\n")

        f.write("**Synthèse des avis positifs :**\n")
        f.write(pos_summary + "\n\n")
        f.write("**Tendances extraites (positif) :**\n")
        for t in pos_trends:
            f.write(f"- {t}\n")
        f.write("\n")

        f.write("**Synthèse des avis négatifs :**\n")
        f.write(neg_summary + "\n\n")
        f.write("**Tendances extraites (négatif) :**\n")
        for t in neg_trends:
            f.write(f"- {t}\n")
        f.write("\n")

        f.write("**Synthèse des avis neutres :**\n")
        f.write(neu_summary + "\n\n")
        f.write("**Tendances extraites (neutre) :**\n")
        for t in neu_trends:
            f.write(f"- {t}\n")
        f.write("\n")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"✅ Résumé enregistré dans {TREND_OUTPUT_FILE}.")
    print(f"Temps d'exécution: {elapsed_time:.2f} secondes.")

if __name__ == "__main__":
    main() 