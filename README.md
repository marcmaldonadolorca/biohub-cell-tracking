# BioHub Cell Tracking — medalla de bronce con una cabeza de refinado sub-voxel propia

Solución a la competición de Kaggle [BioHub – Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development):
seguir cada célula de un embrión en vídeos de microscopía 3D y reconstruir su linaje. Sobre la pila pública
de detección y enlace añadí una red diminuta que corrige la posición de cada célula detectada. Esa pieza es la que dio
la medalla.

## Resultado

**Bronce: puesto 391 de 4.020 equipos en el leaderboard privado** (corte del bronce: puesto 401).

Mismos envíos, público (29 % de los vídeos ocultos) contra privado (71 %):

| envío | qué cambia | público | **privado** |
| --- | --- | --- | --- |
| pila base, sin cabeza | — | 0,947 | 0,914 |
| pila + cabeza v1 | cabeza entrenada con 24 vídeos | **0,952** | 0,919 |
| pila + flow2 + gapfill + cabeza v1 — **el seleccionado** | | **0,952** | **0,920 → bronce** |
| ídem con cabeza v2 | cabeza entrenada con 64 vídeos | 0,950 | **0,921** (el mejor, no seleccionado) |
| ídem con la cabeza pública de otro participante | la que usaban cientos de equipos | 0,950 | 0,917 |
| ídem con la cabeza v1 a media escala | desplazamiento × 0,5 | 0,949 | 0,919 |

La cabeza vale **+0,006 en el privado** (0,914 → 0,920). Sin ella no había medalla.

## El hallazgo que estructura la solución

**Importaba más dónde están las células que cómo se enlazan.** Probé ocho variantes del umbral de
detección, del enlace y del post-proceso (incluido un prior de flujo que en el banco local ganaba +0,0099); todas se
quedaron en 0,947. Lo único que sacó al envío de ese bloque fue mover cada detección un máximo de 2 µm.

**Para la cabeza, la medición local acertó y el leaderboard público se equivocó.** En 12 vídeos que ninguna de mis
cabezas vio al entrenar, la distancia media detección→célula real baja un 26,65 % con v2, un 25,71 % con v1 y un
10,22 % con la cabeza pública. El privado dio el mismo orden: 0,921 > 0,920 > 0,917. El público, con unos 40 vídeos,
dijo lo contrario, y fue el que usé para elegir los dos envíos finales. Por eso el mejor en privado quedó fuera.

## Arquitectura

- **Entrada:** 224 valores por detección = los 32 canales del UNet en la detección y las diferencias con sus 6 vecinos,
  en la rejilla submuestreada (1, 4, 4), que a 1,625 µm/vóxel es isótropa.
- **Red:** MLP 224 → 32 → 3 (SiLU), 7.299 parámetros. La última capa arranca a cero, así que empieza sin mover nada.
  La salida está acotada a 2 µm (`2·d / (1 + |d|)`).
- **Tres cambios acoplados** en el script de predicción (`patch_predictor()`): refinar justo después de detectar,
  **no** redondear a entero al reescalar, e indexar las features del transformer con interpolación **trilineal**.
  El indexado entero de serie trunca (12,7 → 12) y leería la celda equivocada.
- **Entrenamiento:** capturo las features en vídeos de train y emparejo cada detección con su célula real
  (algoritmo húngaro a 7 µm, el radio de la métrica oficial). Mido **por vídeo** y **por embrión no visto**, con una
  puerta: si no mejora en los vídeos retenidos, la cabeza no se usa. La v1 mejora 6 de 6 vídeos retenidos (−20,5 %)
  y, entrenada en un embrión y medida en el otro, −6,5 % y −13,6 %.

## Qué se midió y no funcionó

- **Prior de flujo vecinal (flow2) y gapfill:** +0,0099 de Jaccard en el banco local (17 de 24 vídeos mejoran), cero
  en el público y en el privado. Los pesos del detector habían visto esos vídeos etiquetados y el banco era optimista.
- **Veto de divisiones más permisivo** (0,25 → 0,15): −0,002 en el público. **TTA con volteo en Z:** −0,006.
- **Divisiones aprendidas con el transformer:** 40-60 falsas por cada verdadera a cualquier umbral.
- **Cabeza «más conservadora»** (desplazamiento a la mitad): peor en público y en privado (0,949 / 0,919).
- **Quitar el validador del pipeline público** para ahorrar tiempo: −0,001, confirmado por tres fuentes.

## Limitaciones y siguientes pasos

- El detector y el modelo de aristas son los públicos (no los reentrené). La cabeza se entrena sobre vídeos que esos
  pesos ya vieron, y así se mide.
- Cada variante se envió una sola vez: diferencias de 0,001 en el público están dentro del ruido.
- Lo siguiente habría sido entrenar la cabeza con los 199 vídeos etiquetados. Otro participante lo publicó después del
  cierre.

## Reproducir

```bash
pip install -r requirements.txt
pytest -q tests/                      # contrato del módulo: 224 entradas, desplazamiento ≤ 2 µm, trilineal = gather en enteros
```

El pipeline completo corre en Kaggle (GPU T4, tope de 12 h): **[notebook público](https://www.kaggle.com/code/marcmaldonado/biohub-sub-voxel-head-0-921-private)**,
que carga la cabeza desde el **[dataset público](https://www.kaggle.com/datasets/marcmaldonado/biohub-coordref-head-public)**
(v1 y v2, con los logs de entrenamiento). Para entrenar una cabeza propia:

```python
import coordref
coordref.patch_predictor("scripts/predict_unet_transformer.py", mode="capture")  # vuelca <vídeo>/<t>.npz
```
```bash
python coordref/train_coordref.py --captures capturas/ --train-dir train/ --out head.pt              # partición por vídeo
python coordref/train_coordref.py --captures capturas/ --train-dir train/ --out head.pt --val-prefix 44b6   # embrión no visto
```

## Créditos

La idea y la arquitectura de la cabeza son de **anvithpothula** (notebook público *biohub-0-953-lb-original*); este
código es una reimplementación independiente, entrenada con más datos. La pila sobre la que va: detector,
DeepCenter y *support pack* de **pilkwang**; *harmonic-fusion v30* de **flexonafft**; `ILP_DIVISION_WEIGHT` 0,6 de
**zhehaoliang**; `SECONDARY_EDGE_FEATURE_TTA_WEIGHT` 1,0 de **sjlee101**; flow2 y gapfill de **thtennant**.

Licencia MIT.
