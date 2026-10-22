# GreenSight Lomé : prédire la consommation énergétique des bâtiments par transfer learning

Étude de machine learning réalisée dans le cadre du programme D-CLIC (OIF, 2024-2025). Elle estime la consommation d'énergie et les émissions de CO₂ des bâtiments non résidentiels de Lomé, une ville où les données de consommation manquent.

## Le problème

Pour cibler la rénovation énergétique, il faut savoir quels bâtiments consomment le plus. À Lomé, il n'existe ni compteurs intelligents ni historique de consommation par bâtiment, et les relevés sur le terrain coûtent cher. On ne peut donc pas entraîner un modèle directement sur des données locales.

## Démarche

1. **Apprentissage sur une ville bien documentée** : les modèles apprennent le lien entre les caractéristiques d'un bâtiment et sa consommation sur le jeu public *2016 Building Energy Benchmarking* de la ville de Seattle.
2. **Préparation des variables** : passage au logarithme des cibles et de la surface pour limiter l'effet des valeurs extrêmes, standardisation, encodage one-hot des types d'usage.
3. **Comparaison et réglage des modèles** : XGBoost est retenu pour l'énergie, LightGBM pour le CO₂.
4. **Adaptation à Lomé** : les prédictions sont transposées au contexte local, avec les types de bâti du recensement RGPH 2022 (banco, semi-moderne, moderne) et le facteur d'émission du réseau électrique togolais.

## Résultats

| Modèle | Données | Résultat |
|---|---|---|
| CO₂ (LightGBM, après réglage) | Seattle, jeu de test | R² 0,996, erreur absolue moyenne en pourcentage de 5,6 % (`models/results_co2_comparison.csv`) |
| CO₂ adapté à Lomé (XGBoost) | jeu synthétique : 375 bâtiments en entraînement, 125 en test | R² 0,98, erreur absolue moyenne en pourcentage de 9,3 % (`models/mvp_lome_metadata.json`) |

**À revérifier.** Sur Seattle, la validation croisée du modèle CO₂ donne un R² moyen négatif (−0,08), très loin du R² de test. Cet écart peut venir d'une fuite d'information entre variables et cible, ou d'un problème dans la validation croisée. Les scores de test sont donc à prendre avec prudence tant que ce point n'est pas éclairci.

Le résultat sur Lomé porte sur des données **synthétiques** : il montre que la chaîne de traitement fonctionne, pas que le modèle est juste sur des bâtiments réels. La validation sur des relevés réels reste à faire.

## Contenu

```
script/
├── EDA_final.ipynb               exploration du jeu de Seattle
├── consomption_prediction.ipynb  modèle de consommation d'énergie
├── Emissions_predictions.ipynb   modèle d'émissions de CO₂
└── adaptation_lomé.ipynb         transposition à Lomé
data/                             données de Seattle, jeu synthétique de Lomé, jeu prêt pour la modélisation
models/                           modèles entraînés, hyperparamètres, métadonnées du modèle Lomé
*.pptx                            présentations du projet
```

L'application de démonstration est dans le dépôt [app-energisight](https://github.com/pierredamien21/app-energisight).

## Stack

Python · pandas · NumPy · SciPy · scikit-learn · XGBoost · LightGBM · Matplotlib · Seaborn
