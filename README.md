# 📊 Classification des Avis - Analyse de Sentiment en Français

## 📌 Contexte

Ce projet offre la possibilité de **classer des commentaires** de clients francophones en trois catégories :  
- **POSITIVE** 🟢 : Un avis globalement positif  
- **NEGATIVE** 🔴 : Un avis globalement négatif  
- **NEUTRAL** 🟡 : Un avis mitigé/neutre  

L’objectif est de créer un proof of concept (poc) pour l'analyse d'avis automatique en français. Dans cet exemple, j'ai choisi des critiques de restaurants/d'entreprises et leur attribuer automatiquement une **catégorie de sentiment**.

Ce projet s'appuie sur les modèles suivants :

 - **TextBlob-fr**, une bibliothèque NLP  basé sur des mots-clés et un dictionnaire.
 - **CamemBERT** modèle transformers , est plus apte à traiter les phrases nuancées.



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
(Assurez vous que ```reviews_output.txt``` est bien rempli de n'importe quelle manière avant de lancer le script.)


##  📂 Organisation du projet

```bash
📦 scrap_reviews_trend_poc
 ┣ 📂 src
 ┃ ┣ 📜 sentiment[nom du model].py        # Script  d'analyse de sentiment avec un modèle choisi (TextBlob-fr, CamemBERT)
 ┃ ┗ 📜 utils.py            # Fonctions auxiliaires
 ┣ 📜 reviews_output.txt    # Fichier contenant les avis à analyser ici rempli avec de l'API Yelp mais on peut le remplir avec n'importe quoi
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

 ## 🔗 Liens 

 lien de la bibliothèque ```TextBlob-fr``` : https://github.com/sloria/textblob-fr

 lien des modèles de ```transformers``` : https://github.com/huggingface/transformers

 lien du modèle  ```CamemBERT``` ( pré-entraîné sur du texte général en français, mais pas spécifique à l'analyse de sentiment): https://huggingface.co/almanach/camembert-base/tree/main

 lien du modèle ```CamemBERT``` (pré-entraîné sur du texte spécifique à l'analyse de sentiment ): https://huggingface.co/cmarkea/distilcamembert-base-sentiment/tree/main