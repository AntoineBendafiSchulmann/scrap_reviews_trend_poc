# 📊 Analyse de Sentiment en Français et Synthèse des Tendances (hors ligne et sans aucun service externe)

Ce projet permet de **classer automatiquement des avis** en français (POSITIVE, NEGATIVE ou NEUTRAL) et de **générer des synthèses rédigées** décrivant les tendances les plus fréquentes pour chaque sentiment, **sans dépendre de services externes** (tout se passe en local).

---

## Contexte

- **POSITIVE** 🟢 : avis globalement positif  
- **NEGATIVE** 🔴 : avis globalement négatif  
- **NEUTRAL** 🟡 : avis plutôt mitigé/neutre  

L’idée est de créer un *Proof of Concept* (poc) pour analyser des avis en français. On se base sur des avis Trustpilot, mais la méthode peut s’appliquer à d’autres domaines.

### Modèles & méthodes

- **CamemBERT** : un modèle *transformers* spécialisé en français, qui comprend mieux la langue et les nuances.  
- **Llama** : un grand modèle de langage (LLM) qui rédige un paragraphe de synthèse pour les avis positifs, négatifs et neutres.  
- **KeyBERT & YAKE** : identifient les mots-clés récurrents (tendances) pour chaque catégorie (POSITIVE, NEGATIVE, NEUTRE).

---

## Prérequis

Avant de commencer, assurez vous d'avoir installé :
- **Python 3.12+**
- **Un environnement virtuel (`venv`)**

## Installation et Exécution

1️⃣ Cloner le projet

```bash
git clone https://github.com/AntoineBendafiSchulmann/scrap_reviews_trend_poc

```

2️⃣ Créer et activer l’environnement virtuel

```bash
# Sous Linux / Mac
source venv/bin/activate 
# Sous Windows
venv\Scripts\activate 
```

3️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

4️⃣ Exécuter les scripts

- Récupérer les avis depuis Trustpilot pour un domaine particulier:

la commande doit entrée doit correspondre à ce format : 

```bash
python scrape_trustpilot.py <site>
```
exemple avec **cofidis.fr**

```bash
python -m src.scrape_trustpilot cofidis.fr
```
Le script va automatiquement parcourir les pages d’avis sur Trustpilot pour ce domaine et enregistrer chaque commentaire dans ```trustpilot_reviews.txt```, il suffit donc de passer le nom de domaine comme argument, le code continue donc la collecte tant qu’il trouve du contenu à extraire pour le domaine indiqué et s’arrête de lui-même dès qu’il n’y a plus aucun avis chargé sur la page suivante.

- Pour classifier les avis :

```bash
python -m src.sentiment_camembert
```
(Assurez vous que le fichier d'avis brut ```trustpilot_reviews.txt``` est bien rempli de n'importe quelle manière avant de lancer le script.)

- Pour analyser les tendances et générer la synthèse :

```bash
python -m src.sentiment_trend_analysis
```
Ce script utilise ```trustpilot_reviews_with_sentiment_camembert.txt```

##  Organisation du projet
```bash
📦 scrap_reviews_trend_poc
 ┣ 📂 src
    ┣ 📜 scrape_trustpilot.py          # Récupère les avis en ligne (ex: cofidis.fr)
    ┣ 📜 sentiment_camembert.py        # Classe chaque avis (POSITIVE, NEGATIVE, NEUTRAL) via CamemBERT
    ┣ 📜 sentiment_trend_analysis.py   # Identifie les tendances & génère la synthèse (via KeyBERT, YAKE, Llama)
 ┣ 📜 trustpilot_reviews.txt                         # Avis bruts
 ┣ 📜 trustpilot_reviews_with_sentiment_camembert.txt # Avis classés par sentiment
 ┣ 📜 trustpilot_sentiment_trends.txt              # Fichier final (répartition, tendances, synthèses)
 ┣ 📜 requirements.txt
 ┗ 📜 README.md
```
## Format des fichiers

- ```trustpilot_reviews.txt``` : Avis bruts, un avis par ligne généré par le fichier ```scrape_trustpilot.py```

Exemple :

```bash
A1b2C3d4E5f6	chez-mamie-louise-lille	Chez Mamie Louise	4.2	00000000-0000-0000-0000-000000000032	0	Gros plus : le coin bibliothèque où on peut feuilleter de vieux magazines en attendant nos plats, c’est sympa !
```

- ```trustpilot_reviews_with_sentiment_camembert.txt``` : Ajoute une colonne supplémentaire avec la catégorie de sentiment détecté : POSITIVE, NEGATIVE ou NEUTRAL.

Exemple :

```bash
A1b2C3d4E5f6	chez-mamie-louise-lille	Chez Mamie Louise	4.2	00000000-0000-0000-0000-000000000032	0	Gros plus : le coin bibliothèque où on peut feuilleter de vieux magazines en attendant nos plats, c’est sympa !	POSITIVE
```

## Comment ça marche ?

1. Scraper les avis

- ```Le script scrape_trustpilot.py``` peut aller sur Trustpilot (par ex. ```cofidis.fr```) et récupérer automatiquement les avis.
- Les avis sont enregistrés dans ```trustpilot_reviews.txt```

2. Classer les avis avec ``` CamemBERT``` 

- Le script ``` sentiment_camembert.py```  lit le fichier brut (``` trustpilot_reviews.txt``` ) et applique ``` CamemBERT```  pour dire si chaque avis est POSITIVE, NEGATIVE ou NEUTRAL.
- Il créé  alors le fichier ```trustpilot_reviews_with_sentiment_camembert.txt``` .

3. Analyser les tendances & générer la synthèse: 

- Le script ```sentiment_trend_analysis.py``` :

1.  Détecte les tendances (mots-clés) avec ```KeyBERT``` et ```YAKE```.
    - extraient des mots-clés de chacun des avis.
    - on cumule tous ces mots-clés dans un ```Counter```, pour chaque sentiment.
    - on ne garde que ceux qui apparaissent ≥3 fois, et qui ne sont pas dans la blacklist.
   
2. Filtre (blacklist) pour supprimer les termes trop génériques 
    
3. RAG (Retrieval-Augmented Generation) via FAISS pour récupérer quelques extraits vraiment pertinents

4. Génère un paragraphe de synthèse avec ```Llama```, qui s’appuie sur ces extraits réels, au lieu d’inventer

- Le résultat se trouve dans ```trustpilot_sentiment_trends.txt```, avec :
    - la répartition des avis (POSITIVE, NEGATIVE, NEUTRAL),
    - les tendances trouvées,
    - la synthèse rédigée pour chaque sentiment.


## RAG (Retrieval-Augmented Generation) : Pourquoi et Comment ?

**But** : Permet d'améliorer la précision des réponses d'un modèle de langage en le forçant à ne pas se baser uniquement sur sa donnée d'entraînement et ainsi éviter que le modèle “invente” des informations. Nous voulons qu’il s’appuie sur **des extraits réels** dans les avis.

Comme dans cet autre projet : https://github.com/AntoineBendafiSchulmann/deaplearning_exploration où grâce à ce processus, les réponses sont plus précises, actualisées et adaptées aux documents fournis spécifiquement sur la tarte aux pommes.

1. **Découper les avis en segments (chunks)**  
   - Par exemple, si vous avez un avis :  
     > "J’ai demandé un prêt chez Cofidis. Le service a été très rapide. L’argent a été débloqué en 2 jours."  
   - On peut le couper en 2 segments :  
     1. "J’ai demandé un prêt chez Cofidis"  
     2. "Le service a été très rapide. L’argent a été débloqué en 2 jours"

2. **Encoder ces segments en vecteurs (embeddings)**  
   - Un modèle comme `sentence-transformers/all-mpnet-base-v2` va transformer chaque phrase/segment en une représentation mathématique (un **vecteur**).  
   - Exemple : le segment `"Le service a été très rapide. L’argent a été débloqué en 2 jours"` sera converti en un tableau de nombres, par ex. `[0.12, -0.45, 0.88, …]`.

3. **Indexation dans FAISS**  
   - On stocke tous ces vecteurs (de tous les avis) dans un **index FAISS**.  
   - FAISS sait **retrouver rapidement** lesquels sont proches ou similaires entre eux.

4. **Recherche des extraits pertinents (Retrieval)**  
   - Quand on veut synthétiser la tendance `"prise en charge"`, on encode `"prise en charge"` en vecteur, par ex. `[0.03, -0.22, 0.54, …]`.  
   - On demande à FAISS : “Quels segments sont les plus proches de ce vecteur ?”  
   - FAISS renvoie 2 ou 3 segments d’avis qui parlent clairement de “prise en charge” ou de sujets proches.

5. **Génération de texte (Augmented Generation)**  
   - On prend ces segments trouvés et on les assemble sous forme d’un “contexte”.  
   - On donne ce contexte + un prompt (instructions) au modèle Llama.  
   - Llama produit alors un **paragraphe** qui décrit la tendance `"prise en charge"` **en se basant** sur le contenu réel des avis retournés par FAISS.

**Grâce à ce système**, le modèle :
- Ne part pas de zéro.  
- Ne se base pas uniquement sur sa “mémoire interne”.  
- Il lit des extraits concrets trouvés via ```FAISS```.  

C’est ce qu’on appelle **Retrieval-Augmented Generation (RAG)** :  
1. **Retrieval** (récupérer des extraits utiles dans ```FAISS```),  
2. **Augmented Generation** (le modèle ```Llama``` s’appuie sur ces extraits pour rédiger une synthèse réaliste).

Résultat : une **synthèse** plus réaliste, **ancrée dans les vrais avis**.


## Paramètres de génération 

on peut par ailleurs utiliser des paramètres de génération servant  à "canaliser la créativité" de ```Llama```

- ```temperature```: rend le texte moins aléatoire, plus concis
- ```top_p```: filtre les mots trop rares et empêche le modèle de partir dans des phrases extravagantes
- ```repetition_penalty```: pénalise la répétition de tokens, pour éviter de se répéter, de radoter


## Exemple de Résultat

```bash
Répartition des sentiments :
Positifs : 19950 avis (92.30%)
Négatifs : 1247 avis (5.77%)
Neutres  : 417 avis (1.93%)

**Synthèse des avis positifs :**
La société propose une prise en charge efficace et rapide pour les demandes financières, avec un traitement expeditif des dossiers qui permet le déblocage des fonds rapidement. Les clients apprécient la qualité de service ...

**Tendances extraites (positif) :**
- prise en charge
- traitement du dossier
- déblocage des fonds


**Synthèse des avis négatifs :**
Les clients ont exprimé leur insatisfaction avec divers aspects de leurs expériences avec Cofidis.De nombreux utilisateurs ont également signalé des problèmes liés aux versements ...

**Tendances extraites (négatif) :**
- fois sans frais
- déblocage des fonds
- manque de respect

**Synthèse des avis neutres :**
Le dossier a été traité avec rapidité, la procédure s est déroulée dans les délais convenus, il y a eu une bonne mise en relation entre le client et le service concerné...

**Tendances extraites (neutre) :**
- traitement du dossier
- accord de principe

```

## Avantages / Limitations

Avantages

- Tout est en local (off-line), pas besoin de services externes.
- CamemBERT gère le français et ses nuances.
- ```Llama``` peut écrire un court paragraphe décrivant les points clés (positifs, négatifs, neutres).


Limitations

- Sur CPU, Llama + FAISS peut être lent pour un large volume d’avis.
- La blacklist et les réglages de ```KeyBERT/YAKE``` peuvent demander un ajustement pour filtrer certains mots-clés “inutiles” ou étranges.
- Si les avis sont très mal écrits, le modèle peut se tromper.

 ## 🔗 Liens 

- Hugging Face transformers https://github.com/huggingface/transformers
- lien du modèle  ```CamemBERT``` ( pré-entraîné sur du texte général en français, mais pas spécifique à l'analyse de sentiment): https://huggingface.co/almanach/camembert-base/tree/main
- ```DistilCamemBERT``` (pré-entraîné sur du texte spécifique à l'analyse de sentiment ): https://huggingface.co/cmarkea/distilcamembert-base-sentiment/tree/main
- ```Llama 3.2 - 3B Instruct``` (pré-entraîné sur du texte général en français): https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/tree/main
- Trouver automatiquement les mots-clés d’un texte en s’appuyant sur des modèles pour identifier des mots-clés en fonction de leur contexte et de leur signification, plutôt que de se baser uniquement sur des statistiques simple ```KeyBERT``` https://github.com/MaartenGr/KeyBERT 
- Repèrer les termes clés d’un texte en se basant sur la fréquence et la position des mots avec ```YAKE``` https://github.com/LIAAD/yake 
- Pour transformer des phrases en vecteurs numérique ```sentence-transformers/all-mpnet-base-v2``` https://huggingface.co/sentence-transformers/all-mpnet-base-v2
- ```FAISS``` stocke des “représentations mathématiques” (vecteurs) et peut rapidement repérer lesquels sont les plus proches, pour retrouver les passages les plus semblables dans un grand ensemble de données. https://github.com/facebookresearch/faiss