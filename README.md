# Iron Dome Simulator

Bienvenue sur le dépôt de notre Projet de Fin d'Année (PFA) réalisé à l'ENSTAB.

L'objectif de ce projet est de simuler le fonctionnement d'un système de défense anti-aérienne complexe, en combinant des modèles physiques réalistes et de l'Intelligence Artificielle.

Le résultat final est un simulateur 3D interactif où une IA détecte des roquettes en vol et lance de manière autonome des missiles pour les intercepter.

## Ce que nous avons développé

Afin de garantir le réalisme de la simulation, le système repose sur des concepts scientifiques et militaires concrets.

* **Physique balistique** : Les trajectoires des roquettes sont calculées en temps réel avec la méthode d'intégration de Runge-Kutta d'ordre 4. Le modèle prend en compte la gravité, la masse et la traînée aérodynamique.
* **Machine Learning** : Le système simule un radar générant des données de position comportant un bruit gaussien. Pour traiter ces données incertaines, nous avons entraîné un réseau de neurones récurrent (LSTM 3D) sur 20 000 trajectoires. Après 3 secondes d'observation, l'IA est capable de prédire le point d'impact final de la menace.
* **Guidage de missile** : Le missile intercepteur utilise l'algorithme de Navigation Proportionnelle (Proportional Navigation). Cela lui permet d'anticiper la trajectoire de la roquette et de l'intercepter de manière optimale, à l'image des vrais systèmes d'interception.
* **Dashboard interactif** : L'interface de Command & Control (C2) est développée en 3D avec Plotly. Elle intègre un suivi en temps réel des cibles, un compte à rebours avant impact (ETA), un système d'évaluation des performances de tir, ainsi qu'un mode auto-défense.

## Architecture du projet

Le code est séparé en deux grandes parties : les notebooks de recherche et le code source métier.

```text
iron-dome-simulator/
│
├── notebooks/                     # Environnement de test et d'exécution
│   ├── phase1_integration.ipynb   # Tests physiques et guidage
│   ├── phase2_prediction_3d.ipynb # Architecture IA et filtres
│   └── phase3_simulation.ipynb    # Dashboard final C2 (Point d'entrée)
│
├── src/                           # Code source métier
│   ├── simulation/                # Modèles physiques (rocket3d.py, interceptor3d.py)
│   ├── ml/                        # Modèle PyTorch et script d'entraînement
│   ├── dashboard/                 # Interface utilisateur
│   └── scoring/                   # Logique d'évaluation des performances
│
└── requirements.txt               # Dépendances requises
```

## Instructions d'installation

Pour lancer le simulateur sur une machine locale, veuillez suivre ces étapes.

1. **Récupérer le code source** :
   ```bash
   git clone <le-lien-de-ce-depot>
   cd iron-dome-simulator
   ```

2. **Installer les dépendances** :
   L'utilisation d'un environnement virtuel est recommandée.
   ```bash
   pip install -r requirements.txt
   ```

3. **Lancer la simulation** :
   Ouvrez Jupyter Notebook et lancez le fichier `notebooks/phase3_simulation.ipynb`. 
   Exécutez l'ensemble des cellules, puis utilisez le bouton "LAUNCH SALVO" pour démarrer la simulation d'attaque.

Il est également possible de relancer l'entraînement complet du modèle d'Intelligence Artificielle en exécutant la commande `python src/ml/train_lstm3d.py`.

---
Projet développé dans le cadre du cursus d'ingénieur à l'ENSTAB.