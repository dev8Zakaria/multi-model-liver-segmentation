# Segmentation vasculaire du parenchyme hépatique

Application web du PFA « Système intelligent pour la caractérisation automatique du parenchyme hépatique » (ENSA Tanger, FMPT). L'utilisateur dépose une image histologique, choisit l'un des cinq modèles entraînés et obtient le masque des veines portes, veines centrales, artères hépatiques et canaux biliaires.

Prototype académique, non validé cliniquement : les résultats ne constituent pas un diagnostic.

## Architecture

```
Navigateur (HTML/CSS/JS)  ──POST /api/predict──▶  FastAPI  ──▶  ONNX Runtime (CPU)
        ◀── aperçu JPEG + carte de classes PNG + statistiques ──┘
```

- `app/static/` : interface (sans framework ni étape de build).
- `app/main.py` : serveur FastAPI, sert l'interface et l'API.
- `app/inference.py` : prétraitement identique à l'entraînement (redimensionnement 256 × 256 par interpolation d'aire, normalisation ImageNet), découpage en tuiles, inférence.
- `app/models.json` : liste des modèles et Dice de validation issus des journaux d'entraînement.
- `models/*.onnx` : les cinq modèles convertis depuis les fichiers `.pth`.
- `tools/export_onnx.py` : conversion `.pth` → ONNX, avec vérification contre PyTorch.
- `tools/evaluate_test_set.py` : Dice par classe du pipeline déployé sur le jeu de test.

Les modèles tournent avec ONNX Runtime plutôt que PyTorch : l'image Docker est bien plus légère et l'inférence sur CPU plus rapide.

## Lancer en local

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m uvicorn app.main:app --port 7860
```

Puis ouvrir http://localhost:7860.

Avec Docker :

```bash
docker build -t pfa-foie .
docker run --rm -p 7860:7860 pfa-foie
```

## Régénérer les modèles ONNX

Nécessite PyTorch (`pip install -r tools/requirements-export.txt`) et les checkpoints d'origine :

```bash
python tools/export_onnx.py --downloads C:/Users/zakar/Downloads
```

## Déployer sur Render

L'application est déployée comme service web Docker sur Render, à partir de ce dépôt GitHub (configuration dans `render.yaml`).

1. Sur render.com : **New > Blueprint**, choisir ce dépôt. Render crée le service `liver-vessel-segmentation` (instance Standard) et construit l'image à partir du `Dockerfile`.
2. Pendant la construction, `deploy/fetch_models.py` télécharge les modèles ONNX depuis GitHub si Render a cloné le dépôt sans Git LFS (vérification par somme SHA-256).
3. Une fois le service en ligne, l'application est disponible à l'adresse `https://<service>.onrender.com`.

Une instance payante est facturée tant qu'elle tourne. Entre deux démonstrations, la **suspendre** (Settings > Suspend Web Service) : un service suspendu n'est pas facturé. La reprendre quelques minutes avant la présentation.

## API

| Méthode | Route | Rôle |
| --- | --- | --- |
| GET | `/api/models` | Classes et modèles disponibles |
| POST | `/api/predict` | Champs `file`, `model`, `mode` (`auto`, `whole`, `tiles`) |
| GET | `/api/health` | État du serveur |
| GET | `/api/docs` | Documentation interactive |
