# Homography-based 2-DOF readout

## Pipeline

1. **4 DOF fixés** par frame (FMS) : `along_track`, `height`, `pitch`, `roll`.
2. **Prior t−1** : `HE_prior`, `lat_prior` (estimate précédente).
3. Homographie `H_prior` depuis la pose prior → rectification.
4. Lecture 2D : angle de centerline + Y du seuil dans le plan.
5. **Correction** latérale (couplage HE) + mise à jour pose.

## Lecture brute

Dans le plan rectifié par `H_prior` :

| DOF | Proxy 2D | Code |
|-----|----------|------|
| **HE** | Angle centerline vs +X | `measure.heading_error_deg` |
| **Latéral** | Y du milieu du seuil (BL–BR) | `measure.lateral_offset_m` |

⚠️ Le Y brut **n’est pas** le cross-track métrique : il mélange translation et rotation.

## Correction `tan(HE) · along` (warm-start)

Quand la piste est pivotée de `HE` dans le plan, le seuil (à distance
`along_track` le long de la piste) se décale en Y d’environ :

\[
\Delta Y_{spurious} \approx \tan(HE_{2D}) \cdot along\_track
\]

**Formule** (`readout.correct_plane_residuals`) :

```python
lat_corr = lat_2D - tan(radians(he_2D)) * along_track_m
he_corr  = he_2D   # inchangé
```

**Mise à jour séquentielle** (`readout.apply_prior_residual`) — soustraire le résidu :

```python
HE_est  = HE_prior  - he_corr
lat_est = lat_prior - lat_corr
```

Validé sur CYEG 20 (coins GT, prior t−1) : HE MAE ≈ 0.0002°, lat MAE ≈ 3 m,
**pas de drift** sur 30 frames.

### Pourquoi ça ne marche pas avec prior HE=0 / lat=0 ?

Sans warm-start, on lit un **offset absolu** (~−360 m) qui inclut un biais de
perspective. La correction enlève le couplage rotationnel mais pas ce biais.
Il faut un prior proche (t−1 ou FMS grossier).

## Scripts

```bash
# Rectification + overlays (prior zéro)
PYTHONPATH=. python scripts/validate_homography_dof.py --data-root data/LARD -n 8

# Warm-start synthétique (GT + petits deltas)
PYTHONPATH=. python scripts/validate_homography_warmstart.py --airport CYEG --runway 20

# Tracking séquentiel t−1 → t
PYTHONPATH=. python scripts/validate_homography_sequence.py --airport CYEG --runway 20 --init gt
```

Résultats : `data/LARD/results/homography_seq_*.csv` et `*.png`.

## YOLO — prochaine étape (pas encore branché)

Aujourd’hui tout est validé avec **coins GT projetés**. En prod on n’a pas ça.

Avec les masques YOLO (`data/LARD/masks/yolo_runway/`) :

1. **Extraire 4 coins** du masque runway (quad / `minAreaRect` sur le contour).
2. Remplacer `project_corners_local` par ces coins image dans la chaîne séquentielle.
3. Mesurer si le **bruit de segmentation** (forme imparfaite, coins flous) :
   - reste borné frame à frame (~3 m comme GT) ;
   - ou **drift** quand les erreurs de coins s’accumulent via le prior.

C’est le test de robustesse réel : même maths, entrée bruitée. Alternative si les
coins YOLO sont trop instables : OBB dans la vue rectifiée (warp du masque avec
`H_prior`) au lieu des coins image.

## Fichiers

| Module | Rôle |
|--------|------|
| `plane_homography.py` | `H` depuis pose, `pose_with_he_lat`, warp |
| `measure.py` | Lecture brute 2D |
| `readout.py` | Correction `tan(HE)·along` + `apply_prior_residual` |
