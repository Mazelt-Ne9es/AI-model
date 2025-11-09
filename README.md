# 🚀 Cycle-to-Peak Performance Predictor

## 📖 Vue d'ensemble

Ce projet implémente un système de **prédiction de performance des joueurs de football** basé sur l'intelligence artificielle. Il analyse la charge d'entraînement (données GPS) et les performances techniques (données Wyscout) pour prédire l'état de forme d'un joueur lors de son prochain match.

**Objectif principal** : Fournir un **Score de Préparation au Match** (Match Readiness Score) pour chaque joueur, permettant aux entraîneurs d'optimiser la composition d'équipe et la planification des entraînements.

---

## 🏗️ Architecture du Projet : Les Trois Piliers

### **PILIER 1 : Calcul du Cycle d'Entraînement**

#### Objectif
Cartographier chaque séance d'entraînement en fonction de sa position dans le cycle de préparation au match (J-6, J-5, J-4... J-Day).

#### Méthodologie
1. **Classification des événements** : Chaque jour est classé comme "Match", "Entraînement" ou "Repos"
2. **Calcul des J-jours** : Pour chaque joueur, calcul du nombre de jours jusqu'au prochain match
   - `J-Day` = Jour de match
   - `J-1` = Veille du match
   - `J-2` = 2 jours avant le match
   - etc.

#### Colonnes utilisées
- **GPS Match** : `name`, `team_name`, `date`
- **GPS Training** : `name`, `team_name`, `date`

#### Résultat
- **Fichier généré** : `pillar1_calendar_j_days.csv`
- **Colonnes clés** : `player_name`, `date`, `event_type`, `J_label`, `next_match_date`, `days_to_next_match`

---

### **PILLAR 2 : Ingénierie des Caractéristiques**

#### Objectif
Extraire et fusionner les métriques physiques (GPS) et techniques (Wyscout) pour créer une vue à 360° de chaque joueur.

#### 2.1 Métriques Physiques GPS

**Métriques de Volume** :
- `total_distance_(m)` → Distance totale parcourue
- `total_duration` → Durée totale de l'activité

**Métriques d'Intensité** :
- `hs_distance_(m)` → Distance à haute vitesse (>19.8 km/h)
- `sprint` / `distancia_>_25` → Distance de sprint (>25.2 km/h)
- `distancia_>_20` → Distance au-dessus de 20 km/h

**Charge du Joueur (Accéléromètre)** :
- `total_player_load` → Charge totale du joueur
- `player_load_per_minute` → Charge par minute
- `average_player_load` → Charge moyenne

**Métriques d'Explosivité** :
- `acceleration_b3_efforts_(gen_2)` → Efforts d'accélération élevée
- `deceleration_b3_efforts_(gen_2)` → Efforts de décélération élevée
- `accel_+_decel_efforts` → Total accélérations + décélérations
- `explosive_efforts` → Efforts explosifs

**Puissance Métabolique** :
- `high_metabolic_load_distance_(m)` → Distance à charge métabolique élevée
- `equivalent_distance_(m)` → Distance équivalente

**Fréquence Cardiaque** :
- `avg_heart_rate` → Fréquence cardiaque moyenne
- `maximum_heart_rate` → Fréquence cardiaque maximale
- `heart_rate_exertion` → Effort cardiaque

#### 2.2 Métriques Techniques Wyscout

**Attaque (Joueurs de champ)** :
- `attack_buts` → Buts
- `attack_tirs_total` / `attack_tirs_cadres` → Tirs / Tirs cadrés
- `attack_xg` → Expected Goals (xG)
- `attack_passes_decisives` → Passes décisives
- `attack_dribbles_reussis` / `attack_dribbles_total` → Dribbles réussis / Total
- `attack_duels_offensifs_gagnes` / `attack_duels_offensifs_total` → Duels offensifs gagnés / Total

**Passes** :
- `general_passes_precises` / `general_passes_total` → Passes précises / Total
- `pass_passes_avant_precises` / `pass_passes_avant_total` → Passes vers l'avant
- `pass_passes_tiers3_precises` / `pass_passes_tiers3_total` → Passes dans le dernier tiers

**Défense** :
- `defense_duels_defensifs_gagnes` / `defense_duels_defensifs_total` → Duels défensifs
- `defense_interceptions_total` → Interceptions
- `defense_duels_aeriens_gagnes` / `defense_duels_aeriens_total` → Duels aériens

**Gardiens de but** :
- `goalkeeper_buts_concedes` → Buts encaissés
- `goalkeeper_tirs_contre_total` / `goalkeeper_tirs_contre_cadres` → Tirs contre / cadrés
- `goalkeeper_xcg` → Expected Goals Conceded
- `goalkeeper_sorties_total` → Sorties

#### 2.3 Métriques Dérivées

Calculées automatiquement :
- **Précision des passes (%)** = (Passes précises / Total passes) × 100
- **Succès des dribbles (%)** = (Dribbles réussis / Total dribbles) × 100
- **Duels défensifs gagnés (%)** = (Duels défensifs gagnés / Total) × 100
- **Duels offensifs gagnés (%)** = (Duels offensifs gagnés / Total) × 100

#### Résultat
- **Fichier généré** : `pillar2_master_features.csv`
- **Contenu** : Dataset fusionné avec ~40+ colonnes combinant GPS, Wyscout, J-days et métriques dérivées

---

### **PILLAR 3 : Prédiction de Performance par IA**

#### Objectif
1. Générer un **Score de Performance de Match** historique (1-10)
2. Entraîner un modèle LSTM pour prédire la performance future

#### 3.1 Génération du Score de Performance

**Méthode** :
1. **Standardisation Z-score** : Normalisation de toutes les métriques GPS et Wyscout
2. **PCA (Analyse en Composantes Principales)** : Détermination des poids optimaux pour chaque métrique
3. **Score pondéré** : Combinaison des métriques selon leur importance
4. **Normalisation 1-10** : Mise à l'échelle du score final

**Caractéristiques utilisées pour le rating** :
- **GPS** : distance totale, haute vitesse, sprint, charge joueur, accélérations/décélérations
- **Wyscout** : buts, tirs, xG, passes décisives, dribbles, passes précises, duels, interceptions

#### 3.2 Modèle LSTM (Long Short-Term Memory)

**Architecture** :
```
Input: Séquence de 10 jours avant le match
├─ LSTM Layer 1 (64 unités, ReLU) + Dropout 20%
├─ LSTM Layer 2 (32 unités, ReLU) + Dropout 20%
├─ Dense Layer (16 unités, ReLU)
└─ Output (1 unité) → Score de performance prédit
```

**Entrées du modèle** (pour chaque jour) :
1. Distance totale parcourue
2. Distance à haute vitesse
3. Charge totale du joueur
4. Efforts d'accélération élevée
5. Indicateur jour de match (0/1)
6. Jours jusqu'au prochain match

**Sortie** : Score de performance prédit pour le prochain match (1-10)

#### Résultat
- **Fichier généré** : `lstm_performance_predictor.h5`
- **Modèle entraîné** : Réseau LSTM pour prédiction de performance

---

## 📊 Analyse des Résultats

### Métriques de Performance du Modèle

```
Test Performance:
  Loss (MSE): 0.7868
  MAE: 0.6668
```

### Interprétation

#### ✅ **Points Positifs**

1. **MAE (Mean Absolute Error) = 0.67**
   - Le modèle se trompe en moyenne de **0.67 points** sur une échelle de 1-10
   - Pour un score prédit de 5.0, la vraie valeur est probablement entre 4.33 et 5.67
   - **Précision acceptable** pour un premier modèle

2. **MSE (Mean Squared Error) = 0.79**
   - Pénalise davantage les erreurs importantes
   - Relativement faible, indiquant peu de prédictions très éloignées de la réalité

3. **Cohérence des prédictions**
   - Les prédictions dans les exemples (2.12 - 2.31) sont dans une plage restreinte
   - Le modèle a appris des patterns, mais reste prudent

#### ⚠️ **Points à Améliorer**

1. **Variance des prédictions limitée**
   ```
   Actual: 2.04 | Predicted: 2.12  ✓ (très proche)
   Actual: 2.73 | Predicted: 2.16  ⚠️ (sous-estime)
   Actual: 1.31 | Predicted: 2.16  ⚠️ (surestime)
   Actual: 1.53 | Predicted: 2.31  ⚠️ (surestime)
   Actual: 1.24 | Predicted: 2.14  ⚠️ (surestime fortement)
   ```
   - **Problème** : Le modèle prédit souvent autour de 2.1-2.3, manquant les valeurs extrêmes
   - **Cause probable** : Régression vers la moyenne (dataset limité)

2. **Données d'entraînement limitées**
   - Si peu de matchs avec ratings complets → modèle conservateur
   - Solution : Ajouter plus de données historiques

### 🎯 Verdict Global

**Score : 6.5/10 - Modèle fonctionnel mais perfectible**

| Critère | Score | Commentaire |
|---------|-------|-------------|
| **Précision moyenne** | 7/10 | MAE de 0.67 est correct pour un MVP |
| **Généralisation** | 6/10 | Prédit correctement la tendance, mais pas les extrêmes |
| **Robustesse** | 7/10 | MSE faible = peu d'erreurs catastrophiques |
| **Variance** | 5/10 | Manque de diversité dans les prédictions |

**Recommandations** :

1. ✅ **Utilisable en production** pour des alertes générales (ex: "Joueur en forme moyenne")
2. ⚠️ **Pas encore fiable** pour des décisions critiques (ex: titularisation)
3. 🔄 **Amélioration nécessaire** :
   - Collecter plus de données de matchs
   - Ajouter des features contextuelles (adversaire, compétition, position)
   - Tester d'autres architectures (GRU, Transformer)
   - Augmenter la séquence d'entraînement (15-20 jours au lieu de 10)

---

## 📁 Structure des Fichiers de Sortie

```
data/processed/
├── pillar1_calendar_j_days.csv      # Calendrier avec labels J-jours
├── pillar2_master_features.csv      # Dataset complet fusionné
└── lstm_performance_predictor.h5    # Modèle LSTM entraîné
```

---

## 🚀 Utilisation

### Prédire la performance d'un joueur

```python
import pandas as pd
import numpy as np
from tensorflow import keras

# 1. Charger le modèle
model = keras.models.load_model('data/processed/lstm_performance_predictor.h5')

# 2. Charger les données
df = pd.read_csv('data/processed/pillar2_master_features.csv')

# 3. Filtrer les 10 derniers jours d'un joueur avant son prochain match
player_data = df[df['player_name'] == 'NOM_JOUEUR'].tail(10)

# 4. Préparer la séquence
sequence = player_data[['total_distance', 'high_speed_distance', 
                        'total_player_load', 'high_accel_efforts',
                        'is_match', 'days_to_next_match']].values

# 5. Prédire
prediction = model.predict(sequence.reshape(1, 10, 6))
print(f"Score de préparation prédit : {prediction[0][0]:.2f}/10")
```

---

## 🔧 Améliorations Futures

### Court terme
- [ ] Augmenter l'epochs d'entraînement (50 → 100+)
- [ ] Implémenter Early Stopping pour éviter l'overfitting
- [ ] Ajouter validation croisée (K-fold)
- [ ] Visualiser les courbes d'apprentissage

### Moyen terme
- [ ] Intégrer données contextuelles (adversaire, compétition, météo)
- [ ] Modèles spécifiques par position (défenseurs vs attaquants)
- [ ] Détection d'anomalies (risque blessure)
- [ ] Dashboard interactif (Streamlit/Flask)

### Long terme
- [ ] Système de recommandation d'entraînement personnalisé
- [ ] Prédiction à plusieurs matchs (forme sur 3-5 matchs)
- [ ] Intégration temps réel avec wearables
- [ ] Application mobile pour coaching

---

## 📚 Sources de Données

### Fichiers d'entrée requis
```
hackathon_Data/hackathon/Data After Extraction/CSV/
├── matchs_gps.csv                    # Données GPS des matchs
├── training_gps.csv                  # Données GPS des entraînements
├── wyscout_matchs.csv                # Statistiques de matchs (équipe)
├── wyscout_players_goalkeeper.csv    # Stats Wyscout gardiens
└── wyscout_players_outfield.csv      # Stats Wyscout joueurs de champ
```

---

## 🛠️ Prérequis Techniques

```bash
# Installation des dépendances
pip install pandas numpy scikit-learn tensorflow pathlib
```

**Versions testées** :
- Python 3.10+
- TensorFlow 2.13+
- Pandas 2.0+
- NumPy 1.24+

---

## 👥 Auteur & Contact

**Projet** : Cycle-to-Peak Performance Predictor  
**Date** : Novembre 2025  
**Contexte** : Hackathon Data Science - Football Analytics

---

## 📄 Licence

Ce projet est développé dans le cadre d'un hackathon éducatif.

---

## 🙏 Remerciements

- **Données GPS** : Système de tracking des joueurs
- **Données Wyscout** : Plateforme d'analyse vidéo et statistiques
- **Framework** : TensorFlow pour le deep learning
- **Inspiration** : Méthodologie "Cycle-to-Peak" de périodisation sportive
