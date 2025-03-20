import os
import re
from collections import Counter
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "Detail_justificatif_SM_with_sentiment.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "Detail_justificatif_SM_summaries.txt")

TOP_K = 10
NB_SENTENCES = 2

CUSTOM_STOPWORDS = [
    "le","la","les","de","des","et","un","une","pour","par","pas","avec",
    "est","a","dans","en","du","d","que","sur","il","elle","au","aux",
    "ce","se","sa","son","ses","l","ou","mais","qu","ai","avis","client",
    "chez","ne","ça","c","m","n","j","encore","c'est","on","dont",
    "suite","ni","2024","2025"
]

def load_data():
    pos_lines, neg_lines, neu_lines = [], [], []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            parts = row.split("\t")
            if len(parts) < 2:
                continue
            text, senti = parts[0], parts[1].upper()
            if senti == "POSITIVE":
                pos_lines.append(text)
            elif senti == "NEGATIVE":
                neg_lines.append(text)
            else:
                neu_lines.append(text)
    return pos_lines, neg_lines, neu_lines

def build_top_words(lines, top_k):
    if not lines:
        return []
    cleaned = []
    for l in lines:
        txt = re.sub(r"\d+", " ", l.lower())    
        txt = re.sub(r"[^\w\s]", " ", txt)  
        cleaned.append(txt)

    vectorizer = TfidfVectorizer(
        stop_words=CUSTOM_STOPWORDS,
        max_features=2000,
        ngram_range=(1,2),
        min_df=2,
        max_df=0.8
    )
    X = vectorizer.fit_transform(cleaned)
    words = vectorizer.get_feature_names_out()
    sums = X.sum(axis=0)
    data = []
    for j, w in enumerate(words):
        data.append((w, sums[0,j]))
    data.sort(key=lambda x: x[1], reverse=True)

    return [t[0] for t in data[:top_k]]

def sumy_extract(lines, nb_sentences):
    if not lines:
        return []
    punct_lines = []
    for l in lines:
        l = l.strip()
        if not l.endswith(('.', '?', '!')):
            l += '.'
        punct_lines.append(l)
    merged = " ".join(punct_lines)
    parser = PlaintextParser.from_string(merged, Tokenizer("french"))
    summarizer = LexRankSummarizer()
    extracted = summarizer(parser.document, nb_sentences)
    return [str(s).strip() for s in extracted]

def main():
    pos_lines, neg_lines, neu_lines = load_data()
    total = len(pos_lines) + len(neg_lines) + len(neu_lines)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"Total : {total}\n")
        f.write(f"Positifs : {len(pos_lines)}\n")
        f.write(f"Négatifs : {len(neg_lines)}\n")
        f.write(f"Neutres  : {len(neu_lines)}\n\n")

        for sentiment_label, subset_lines in [
            ("POSITIVE", pos_lines),
            ("NEGATIVE", neg_lines),
            ("NEUTRE",   neu_lines)
        ]:
            top_words = build_top_words(subset_lines, TOP_K)
            excerpted = sumy_extract(subset_lines, NB_SENTENCES)

            f.write(f"=== Mots-clés {sentiment_label} ===\n")
            if top_words:
                for i, w in enumerate(top_words, start=1):
                    f.write(f"{i}. {w}\n")
            else:
                f.write("(Aucun mot récurrent)\n")
            f.write("\n")

            f.write(f"=== Résumé extrait {sentiment_label} ===\n")
            if excerpted:
                for ex in excerpted:
                    f.write(f"- {ex}\n")
            else:
                f.write("(Aucune phrase)\n")
            f.write("\n")

            nb_lines = len(subset_lines)
            if top_words and excerpted:
                if len(top_words) >= 3 and len(excerpted) >= 2:
                    f.write(
                        f"Synthèse : Sur {nb_lines} avis {sentiment_label}, "
                        f"on retrouve souvent les expressions «{top_words[0]}», "
                        f"«{top_words[1]}» et «{top_words[2]}». "
                        f"Parmi les exemples marquants : «{excerpted[0]}» "
                        f"et «{excerpted[1]}».\n\n"
                    )
                elif len(top_words) >= 3:
                    f.write(
                        f"Synthèse : Sur {nb_lines} avis {sentiment_label}, "
                        f"les expressions dominantes sont «{top_words[0]}», "
                        f"«{top_words[1]}» et «{top_words[2]}». "
                        f"Il n'y a pas assez d'extraits pour préciser davantage.\n\n"
                    )
                else:
                    f.write(
                        f"Synthèse : Sur {nb_lines} avis {sentiment_label}, "
                        f"les données ne permettent pas d'extraire plus de mots-clés ou d'exemples.\n\n"
                    )
            else:
                f.write(
                    f"Synthèse : Les avis {sentiment_label} ne "
                    f"présentent pas de tendances clairement identifiées.\n\n"
                )

    print(f"Terminé. Résultat dans {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
