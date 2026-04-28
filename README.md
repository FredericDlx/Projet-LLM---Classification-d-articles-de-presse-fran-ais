

---

# 📰 Classification d'articles de presse français
> **Master Executive IA & Data Science** > *Auteurs : Xuan PENG et Frédéric DELCROIX*
> 
---

## 📌 Présentation du projet
Ce projet consiste en une étude comparative de modèles de langage (LLM) pré-entraînés pour la **classification automatique d'articles de presse en français**. Nous évaluons la capacité de deux architectures Transformers à catégoriser des textes parmi **13 rubriques thématiques**.

### 🎯 Objectifs principaux
1.  **Benchmark de performance** : Comparer CamemBERT (spécialisé français) et XLM-RoBERTa (multilingue).
2.  **Analyse linguistique** : Évaluer si la spécialisation dans une langue apporte un gain réel face à un modèle universel massif.
3.  **Analyse d'erreurs** : Identifier les confusions entre catégories proches (ex: Monde vs Europe).

---

## 🤖 Modèles étudiés

| Caractéristique | **CamemBERT** | **XLM-RoBERTa** |
| :--- | :--- | :--- |
| **Spécialisation** | 🇫🇷 Français | 🌍 Multilingue (100 langues) |
| **Architecture** | RoBERTa (Base) | RoBERTa (Base/Large) |
| **Paramètres** | ~110M | ~270M |
| **Vocabulaire** | 32K tokens | 250K tokens |
| **Corpus** | OSCAR (138 Go) | CommonCrawl (2.5 To) |

---

## 📊 Dataset et Catégories

Nous utilisons le jeu de données **`diverse_french_news`** (Hugging Face), filtré spécifiquement sur le domaine **"lemonde.fr"** pour garantir une cohérence éditoriale.

### Système de Classification Hiérarchique
En l'absence de labels natifs, un système de classification par **mots-clés pondérés** a été implémenté.
* **Pondération** : Les mots-clés présents dans le **titre** comptent triple (poids 3) par rapport à la **description** (poids 1).
* **Les 13 catégories** : Monde, Europe, Faits-divers, Politique, Société, Environnement, Sport, Culture, Éco/Conso, Santé, Sciences & Tech, Météo, Jeux.

---

## 🛠️ Méthodologie Technique

### Préparation des données
* **Concaténation** : Fusion du titre et de la description pour chaque article.
* **Échantillonnage équilibré** : Utilisation d'un *Balanced Sampling* pour pallier le déséquilibre des classes (certaines rubriques étant naturellement plus fréquentes que d'autres).
* **Tokenization** : Utilisation de `SentencePiece` avec gestion du padding et de la troncation (Max length : 256).

### Hyperparamètres de Fine-tuning
* **Learning Rate** : $3 \times 10^{-5}$
* **Batch Size** : 16
* **Époques** : 3
* **Environnement** : GPU Tesla P100 (16GB)

---

## 📂 Structure du Rapport
1.  **Description des modèles** : Détails techniques de CamemBERT et XLM-R.
2.  **Exploration des données** : Analyse de la distribution initiale et filtrage.
3.  **Système de Labellisation** : Logique de l'algorithme de scoring par mots-clés.
4.  **Préparation & Entraînement** : Pipeline de traitement PyTorch et boucle de fine-tuning.
5.  **Analyse Comparative** : Comparaison des métriques (Accuracy, F1-score) et matrices de confusion.
6.  **Conclusion** : Synthèse des résultats et recommandations.

---

### 💡 Aperçu des données après traitement
```python
# Exemple de distribution après échantillonnage équilibré
# La plupart des catégories sont fixées à 384 échantillons 
# pour garantir une représentativité égale lors de l'apprentissage.
```
