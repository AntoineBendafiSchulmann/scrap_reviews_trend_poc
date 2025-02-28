# 📊 Analyse de Sentiment en Français et Synthèse des Tendances les plus Récurrentes en local (sans dépendance de services externes)

## 📌 Contexte

Ce projet offre la possibilité de **classer des commentaires** de clients francophones en trois catégories (POSITIVE, NEGATIVE, NEUTRAL) **et** de **générer automatiquement** des **synthèses** décrivant les tendances les plus représentatives pour chaque sentiment, le tout **en local** car ce projet ce destine à la manipulation  de données sensibles d’entreprise sans dépendre de services externes.

- **POSITIVE** 🟢 : Un avis globalement positif  
- **NEGATIVE** 🔴 : Un avis globalement négatif  
- **NEUTRAL** 🟡 : Un avis mitigé/neutre  

L’objectif est de créer un proof of concept (poc) pour l'analyse d'avis automatique en français. Dans l’exemple fourni, il s’agit principalement de critiques de restaurants et d’entreprises, mais la méthodologie peut s’appliquer à d’autres domaines.

Ce projet s'appuie sur plusieurs approches et modèles NLP :

- **TextBlob-fr** : bibliothèque basée sur un dictionnaire de mots-clés (analyse de sentiment de base).
- **CamemBERT** : modèle *transformers* plus apte à gérer des nuances linguistiques complexes.
- **Llama** : grand modèle de langage (LLM) utilisé pour **générer des synthèses** en français, en se fondant sur les tendances détectées dans les avis (positifs, négatifs et neutres).



## ⚙️ Prérequis

Avant de commencer, assurez vous d'avoir installé :
- **Python 3.12+**
- **Un environnement virtuel (`venv`)**

## 🚀 Installation et Exécution

1️⃣ Cloner le projet

```bash
git clone https://github.com/AntoineBendafiSchulmann/scrap_reviews_trend_poc

```

2️⃣ Créer et activer l’environnement virtuel

```bash
source venv/bin/activate 
venv\Scripts\activate 
```

3️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

4️⃣ Lancer l’analyse des avis

```bash
python -m src.sentiment...
```
(Assurez vous que ```trustpilot_reviews.txt``` est bien rempli de n'importe quelle manière avant de lancer le script.)


##  📂 Organisation du projet

```bash
📦 scrap_reviews_trend_poc
 ┣ 📂 src
    ┣ 📜 sentiment[nom du model].py 
    ┣ 📜 sentiment[nom du model].py        # Script  d'analyse de sentiment avec un modèle choisi (TextBlob-fr, CamemBERT)
    ┣ 📜 sentiment_trend_analysis.py  # Script d'analyse de tendances 
    ┃ ┗ 📜 utils.py            # Fonctions auxiliaires
    ┣ 📜 scrape_trustpilot.py  # Script de scraping des avis de trustpilot
┣ 📜 trustpilot_reviews.txt    # Fichier contenant les avis à analyser ici rempli avec le fichier scrape_trustpilot.py  mais on peut le remplir avec n'importe quoi
┣ 📜 reviews_with_sentiment.txt  # Résultats de l'analyse
┣ 📜 requirements.txt       # Liste des dépendances
┗ 📜 README.md             

```

## 📝 Format des fichiers

🔹 Fichier d’entrée (```reviews_output.txt```) (je pense que son format va bouger)

Chaque ligne représente un avis et contient 7 colonnes séparées par des tabulations (```\t```) :

```bash
ID_RESTAURANT  ALIAS  NOM  NOTE_RESTO  ID_REVIEW  NOTE_REVIEW  TEXTE_AVIS
```

Exemple :

```bash
A1b2C3d4E5f6	chez-mamie-louise-lille	Chez Mamie Louise	4.2	00000000-0000-0000-0000-000000000032	0	Gros plus : le coin bibliothèque où on peut feuilleter de vieux magazines en attendant nos plats, c’est sympa !
```

🔹 Fichier de sortie (reviews_with_sentiment.txt)

Ajoute une colonne supplémentaire avec la catégorie de sentiment détecté :

```bash
A1b2C3d4E5f6	chez-mamie-louise-lille	Chez Mamie Louise	4.2	00000000-0000-0000-0000-000000000032	0	Gros plus : le coin bibliothèque où on peut feuilleter de vieux magazines en attendant nos plats, c’est sympa !	POSITIVE
```

## 📌 Exemple d’exécution

Après avoir exécuté la prédiction de sentiment par exemple avec le modèle `textblob_fr` avec la commande suivante :

```bash
python -m src.sentiment_textblob_fr
```
ou utiliser le modèle `CamemBERT` avec la commande suivante :

```bash
python -m src.sentiment_camembert
```

Sortie attendue :

```bash
DEBUG: 124 lignes lues dans reviews_output.txt
DEBUG: Avis → "Ce restaurant est un petit bijou ! La vue sur la mer est à couper le souffle..." | Sentiment détecté → POSITIVE
DEBUG: Avis → "J’espérais mieux après avoir lu les avis. Le serveur était peu souriant..." | Sentiment détecté → NEGATIVE
✅ Succès: 124 avis analysés et écrits dans `reviews_with_sentiment.txt`.
```


## ✅ Ce qui semble bon :

1. Les avis très positifs sont bien détectés → (POSITIVE)

    - "Les produits frais dans l’assiette." → POSITIVE ✅
    - "Les portions sont généreuses, parfaites pour les gros appétits !" → POSITIVE ✅
    - "Le serveur a su nous raconter l’histoire des lieux, rendant l’expérience encore plus authentique." → POSITIVE ✅

2. Les avis clairement négatifs sont bien classés → (NEGATIVE)

    - "Le dessert au citron avait un goût chimique, j’ai été franchement déçue. Dommage car le reste était bon." → NEGATIVE ❌
    - "Je ne comprends pas l’engouement. Les plats étaient tièdes et le vin rouge pas assez aéré." → NEGATIVE ❌
    - "Très déçue par le manque de propreté : la table collait et le sol n’avait pas l’air très net." → NEGATIVE ❌


3. Les avis mitigés sont en NEUTRAL, ce qui peut être encore amélioré :
    - "L’ambiance est cosy, la déco soignée." → NEUTRAL 
    
    (ça pourrait être POSITIVE mais le modèle a un peu du mal avec ce genre de phrases descriptives qui ne sont pas très précises sur le sentiment)
    - "C’est un lieu agréable, mais pas inoubliable." → NEUTRAL ✅
    - "On a attendu un peu, mais le repas était bon." → NEUTRAL ✅ (OK compréhensible l'avis présente une nuance positif/négatif)



## 🔥 Pistes d'amélioration :

#### 📌  Pourquoi ```CamemBERT``` est surrement mieux dans les cas des phrases nuancées :

- Des nuances et contradictions : "Le repas était bon, mais le service trop lent."
- Des expressions complexes : "Je m'attendais à mieux..." → Ce n'est pas 100% négatif.
- Des mots ambigus : "Service correct" peut être neutre ou négatif, selon le ton.

👉 Un modèle ```transformers``` comme ```CamemBERT``` comprend ces subtilités, alors que ```TextBlob-fr``` se base uniquement sur un dictionnaire de mots-clés.

####  📌 Exemple d’erreurs de ```TextBlob-fr```  corrigées par ```CamemBERT``` :

<b>Cas 1 : phrase avec une nuance entre positif et négatif</b>

📌 Phrase :
"Le plat était délicieux, mais le service trop lent."

- ```TextBlob-fr``` : POSITIVE (à cause de "délicieux") ❌
- ```CamemBERT``` : NEUTRAL ✅

<b>Cas 2 : phrase idiomatique (expressions imagées) difficile à détecter</b>

📌 Phrase :
"Franchement, ce n'est pas fameux."

- ```TextBlob-fr``` : NEUTRAL ( surrement parce qu'il ne comprend pas "pas fameux" comme négatif) ❌
- ```CamemBERT``` : NEGATIVE ✅


<b>Cas 3 : phrase  présentant une ambiguïté entre les sentiments positifs et négatifs</b>

📌 Phrase :
"Mon plat de pâtes aux fruits de mer était savoureux. Quelques détails à peaufiner, mais c’est prometteur."

- ```TextBlob-fr``` : NEGATIVE  (sans doute à cause du "mais") ❌
- ```CamemBERT``` : POSITIVE ✅

 Pour les cas, où les avis clients sont  nuancés, longs, et complexes un modèle ```transformers``` plus avancé comme ```CamemBERT``` pourrait mieux gérer ça.


 ## 🎯 Pourquoi se tourner vers le modèle ```CamemBERT```  :

- Globalement, les résultats sont plus cohérents ✅
- Les cas évidents (bon/mauvais) sont bien classés ✅
- Mais les avis un peu nuancés sont parfois mal détectés ⚠
    - ```TextBlob-fr``` reste un modèle basique basé sur des mots-clés et un lexique.

📌 Afin de chercher plus de précision, il a fallu essayer un modèle de NLP plus avancé, comme ```CamemBERT``` parmis les ```transformers```.

Un modèle Transformer comme ```CamemBERT``` comprend les subtilités le modèle s’appuie sur la probabilité des mots dans un contexte, alors que ```TextBlob-fr``` se base uniquement sur <b>un dictionnaire de mots-clés</b>.
 
## 🎯 Exemple d’erreurs du modèle ```CamemBERT``` :

Le modèle a du mal avec certaines structures de phrases. Par exemple, la phrase :

<b>"j'avais besoin rapide d'argent cela m'a dépannée très facilement"</b>

a été classée <b>NEGATIVE</b>, alors qu'en réalité, elle exprime un sentiment positif.

### Pourquoi le modèle fait des erreurs

Le modèle montre ses limites sur des phrases mal formulées ou ambiguës :

1. Syntaxe imparfaite → Mauvaise compréhension

    - Les modèles ```transformers``` sont robustes mais parfois sensibles à des structures de phrase mal construites.
    - La phrase est ici est surrement moyenne  grammaticalement parlant ("besoin rapide d'argent" au lieu de par exemple  "j'avais rapidement besoin d'argent").

2. Présence du mot "argent" et "besoin" → Possiblement mal influencé

    - Dans beaucoup d’avis négatifs, "argent", "besoin", "problème financier" sont souventassociés à des expériences difficiles.
    - Si le modèle a vu beaucoup de phrases du type "J'avais besoin d'argent, impossible d'obtenir un prêt, honteux !" au cours de son entraînement, il risque de mal interpréter cette phrase.

3. Pas de mot-clé explicitement positif dans l'avis

    - Le modèle s’appuie sur la <b>probabilité des mots</b> dans un contexte.
    - "Rapide" et "facilement" sont des mots généralement positifs.
    - Cependant, "besoin rapide d'argent" est perçu par le modèle  comme synonyme d' urgence ou de  détresse, donc associé à un contexte négatif.

4. Contexte et ironie: 

    - Le modèle comprend pas bien le second degré ou l'ironie


## 🔎 Extension : Extraction de tendances et génération de synthèses


En plus de la classification des avis (positifs, négatifs, neutres), ce projet propose un **script** permettant :

1. **D’identifier les tendances principales** (mots ou expressions récurrentes) pour chaque catégorie de sentiment.
2. **De rédiger automatiquement une courte synthèse**, en français, décrivant ces tendances pour chaque polarité (POSITIVE, NEGATIVE, NEUTRE).

### Fichiers concernés

- **`trustpilot_reviews_with_sentiment_camembert.txt`**  
  Fichier d’entrée, contenant des avis déjà labellisés (POSITIVE, NEGATIVE, NEUTRAL) avec le modèle ```CamemBERT``` ( ce choix est expliqué plus haut)
  
- **`trustpilot_sentiment_trends.txt`**  
  Fichier de sortie, dans lequel sont sauvegardés :
  - La répartition des sentiments (nombre d’avis positifs, négatifs et neutres),
  - La synthèse générée pour chaque sentiment,
  - Les tendances extraites, accompagnées d’un exemple de contexte (qui est le contexte de la première appariation d'une des tendances notables extraites) pour chaque tendance pouvant éventuellement être utilisé pour reformuler la tendance, aider le modèle à mieux comprendre le contexte.

### Comment l’exécuter ?

```bash
python sentiment_trend_analysis.py
```
(Assurez-vous d’avoir installé les dépendances requises et d’avoir un environnement Python configuré.)

### Étapes principales du script

1. **Lecture et nettoyage**
   - Charge le fichier `trustpilot_reviews_with_sentiment_camembert.txt`
   - Applique un prétraitement basique (mise en minuscules, suppression de caractères indésirables, etc.)

2. **Extraction de tendances**
   - Utilise KeyBERT et YAKE pour repérer les expressions récurrentes dans chaque catégorie de sentiment (positif, négatif, neutre).
   - Filtre certaines occurrences jugées trop génériques via une “blacklist”.

3. **Recherche de contexte**
   - Pour chaque tendance, le script récupère un court extrait (snippet) où la tendance apparaît dans l’avis, offrant un aperçu concret de la phrase d’origine.

4. **Génération de synthèse**
   - Un modèle Llama (`meta-llama/Llama-3.2-3B-Instruct`) est utilisé pour rédiger, en français, un paragraphe décrivant les grandes lignes relevées dans les avis positifs, négatifs ou neutres.
   - Des consignes spécifiques sont fournies au modèle pour éviter un ton trop subjectif, des formulations incomplètes, ou étranges, sans aucun sens, ainsi qu'éviter les répétition ou les simples énumérations entrecoupées de virgules.

5. **Écriture du résultat**
   - Enregistre la répartition des sentiments, les synthèses, ainsi que la liste des tendances extraites (avec exemples de contexte) dans le fichier `trustpilot_sentiment_trends.txt`.

### Résultat attendu :

Voici un extrait type du fichier de sortie :

```bash
Répartition des sentiments :
Positifs : 300 avis (75.00%)
Négatifs : 50 avis (12.50%)
Neutres  : 50 avis (12.50%)

**Synthèse des avis positifs :**
Les retours des clients soulignent principalement la rapidité du processus de demande et la disponibilité des conseillers...

**Tendances extraites (positif) :**
- rapide et efficace (exemple de contexte : "j'ai choisi de faire confiance à cofidis... le processus a été incroyablement rapide et efficace...")

...
```

### Limitations
-  Certaines formulations originales (p. ex. “Sans frais sont souvent considérés comme inexistants”) peuvent réapparaître si elles ne sont pas filtrées. Il est possible de les réécrire ou supprimer dans la partie “Recherche de contexte” si elles posent problème.
-  L’exécution du modèle Llama requiert des ressources (GPU/CPU) pour un temps de traitement raisonnable.

En conclusion ce script **complète** l’analyse de sentiments en fournissant un **aperçu synthétique** des sujets les plus récurrents, catégorisés par sentiment.

 ## 🔗 Liens 

 lien de la bibliothèque ```TextBlob-fr``` : https://github.com/sloria/textblob-fr

 lien des modèles de ```transformers``` : https://github.com/huggingface/transformers

 lien du modèle  ```CamemBERT``` ( pré-entraîné sur du texte général en français, mais pas spécifique à l'analyse de sentiment): https://huggingface.co/almanach/camembert-base/tree/main

 lien du modèle ```DistilCamemBERT``` (pré-entraîné sur du texte spécifique à l'analyse de sentiment ): https://huggingface.co/cmarkea/distilcamembert-base-sentiment/tree/main

 lien du modèle ```Llama``` (pré-entraîné sur du texte général en français): https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/tree/main