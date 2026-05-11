# 🏗️ AI Construction Site Safety Monitor — Web Edition v10

Interface web complète pour le monitoring de sécurité sur chantier, basée sur le pipeline dual-model du notebook `safety_monitor_v10_3.ipynb`.

## Structure

```
safetymonitor_web/
├── app.py              ← Serveur Flask (toute la logique du notebook)
├── requirements.txt    ← Dépendances Python
├── run.sh              ← Script de démarrage
├── vew.pt              ← ⚠️ À placer ici (modèle PPE/workers/infra)
├── scafandri.pt        ← ⚠️ À placer ici (modèle scaffold risk)
└── static/
    ├── index.html      ← Landing page (design original)
    └── monitor.html    ← Application d'analyse
```

## Démarrage rapide

1. **Copier les modèles** dans ce dossier :
   ```
   vew.pt
   scafandri.pt
   ```

2. **Lancer le serveur** :
   ```bash
   chmod +x run.sh
   ./run.sh
   ```
   Ou directement :
   ```bash
   pip install -r requirements.txt
   python app.py
   ```

3. **Ouvrir** dans le navigateur :
   - Landing page : http://localhost:5000/
   - Application  : http://localhost:5000/monitor

## Architecture

| Composant | Rôle |
|---|---|
| `app.py` | Serveur Flask + pipeline de détection complet |
| `static/index.html` | Page d'accueil (identique au design HTML fourni) |
| `static/monitor.html` | Interface d'analyse (reproduit l'UI du notebook Gradio) |

## API

`POST /api/analyze`
- Body: `multipart/form-data` avec `image` (fichier), `conf_vew`, `conf_scaf`
- Retourne: JSON avec image annotée (base64), score global, détails par personne, infra

`GET /api/status`
- Vérifie la présence des modèles

## Score (0–100)

```
Score = min(100, PPE_risk[max 50] + Scaffold_risk[max 35] + Crowd[max 15])
```

| Niveau | Score |
|---|---|
| 🟢 SAFE/LOW | 0–20 |
| 🟡 MEDIUM   | 21–50 |
| 🟠 HIGH     | 51–80 |
| 🔴 CRITICAL | 81–100 |
# construction
