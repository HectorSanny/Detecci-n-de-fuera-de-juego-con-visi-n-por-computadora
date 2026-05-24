# Automatic Offside Detection using Computer Vision

Proyecto de visión computacional para la detección automática de posibles situaciones de fuera de juego en fútbol utilizando:

- YOLOv8
- Segmentación de jugadores
- Clustering por colores
- Geometría proyectiva
- Punto de fuga
- Tracking temporal
- Detección de toque de balón

---

# Objetivo del Proyecto

El objetivo de este proyecto es construir un pipeline automatizado capaz de:

1. Detectar jugadores en una jugada de fútbol.
2. Separar automáticamente ambos equipos.
3. Determinar posiciones relativas usando geometría proyectiva.
4. Detectar posibles situaciones de offside.
5. Analizar temporalmente la jugada para identificar el momento del pase.

El sistema trabaja sobre videos broadcast reales de fútbol profesional.

---

# Pipeline General

```text
VIDEOS
   ↓
detect_players_pose.py
   ↓
outputs_finales
   ↓
separacion_por_equipo.py
   ↓
outputs_team_finales
   ↓
equipo_atacante.py
   ↓
outputs_offside_results
   ↓
seguimiento_atacantes_decision.py
   ↓
outputs_tracking_ball_touch_eval
```

---

# Estructura del Proyecto

```text
ProyectoVisionData/
│
├── videos/
│
├── src/
│   ├── detect_players_pose.py
│   ├── separacion_por_equipo.py
│   ├── equipo_atacante.py
│   ├── seguimiento_atacantes_decision.py
│   └── run_pipeline.py
│
├── outputs_finales/
├── outputs_team_finales/
├── outputs_offside_results/
├── outputs_tracking_ball_touch_eval/
│
├── requirements.txt
└── README.md
```

---

# Instalación

## 1. Clonar repositorio

```bash
git clone https://github.com/HectorSanny/Detecci-n-de-fuera-de-juego-con-visi-n-por-computadora
cd ProyectoVisionData
```

---

## 2. Crear entorno virtual

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

# Modelos utilizados

## YOLOv8 Detection

Modelo utilizado para:

- detección de jugadores
- detección de balón
- estimación de keypoints

```text
yolov8x.pt
```

---

## YOLOv8 Segmentation

Modelo utilizado para:

- segmentación de jugadores
- extracción del uniforme

```text
yolov8x-seg.pt
```

Los modelos se descargan automáticamente mediante Ultralytics.

---

# Dataset esperado

El primer frame de todos los videos tieen que ser el frame a analizar, todos los videos deben ir dentro de:

```text
videos/
```

Formatos soportados:

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`

---

# Convención de nombres

El nombre del archivo debe indicar dirección de ataque.

## Ataque hacia la derecha

```text
jugada1_der.mp4
```

## Ataque hacia la izquierda

```text
jugada2_izq.mp4
```

Esto permite identificar automáticamente cuál es el punto más adelantado de cada jugador.

---

# Ejecución completa

Desde la carpeta `src`:

```bash
python run_pipeline.py
```

---

# Explicación detallada de cada script

---

# 1. detect_players_pose.py

## Objetivo

Este script realiza:

- detección inicial de jugadores
- filtrado espacial
- filtrado geométrico
- selección del punto más adelantado
- eliminación de ruido
- generación de archivos base

Es la etapa principal de preparación del pipeline.

---

## Flujo del script

```text
Video
   ↓
Extracción del primer frame
   ↓
Detección YOLOv8
   ↓
Máscara verde del campo
   ↓
Filtrado de detecciones
   ↓
Selección de jugadores válidos
   ↓
Cálculo punto avanzado
   ↓
Exportación JSON
```

---

## Funciones principales

---

### process_video()

Función principal del script.

#### Responsabilidades

- abrir video
- extraer frame inicial
- ejecutar YOLOv8
- filtrar detecciones
- calcular puntos avanzados
- generar outputs

#### Input

```python
process_video(video_path, output_folder, model)
```

#### Output

- imagen original
- imagen anotada
- JSON de jugadores

---

### filter_inside_field()

Filtra detecciones fuera del campo.

#### Funcionamiento

Se utiliza una máscara verde para:

- identificar césped
- descartar personas fuera del terreno
- reducir falsos positivos

---

### advanced_point()

Calcula el punto más adelantado del jugador.

#### Ataque derecha

```python
x_adv = max(x_i)
```

#### Ataque izquierda

```python
x_adv = min(x_i)
```

Este punto se usa posteriormente para el análisis de offside.

---

## Outputs generados

```text
outputs_finales/
```

### Archivos

#### Imagen original

```text
play_01_original.jpg
```

#### Imagen procesada

```text
play_players.jpg
```

#### JSON jugadores

```text
play_players.json
```

---

# 2. separacion_por_equipo.py

## Objetivo

Este script separa automáticamente los jugadores por equipo utilizando:

- segmentación
- extracción de color
- clustering

---

## Flujo del script

```text
Jugadores detectados
   ↓
Segmentación YOLOv8-seg
   ↓
Extracción torso
   ↓
Conversión LAB
   ↓
KMeans
   ↓
Clasificación TeamA / TeamB
```

---

## Funciones principales

---

### extract_feature_from_bbox()

Extrae características de color desde la bounding box.

#### Funcionamiento

- recorta torso
- convierte a LAB
- calcula color promedio

Se usa como fallback cuando la máscara falla.

---

### mask_color_feature()

Extrae características usando segmentación.

#### Funcionamiento

- aplica máscara del jugador
- elimina fondo
- reduce influencia del césped
- obtiene color dominante

---

### process_json()

Procesa un JSON generado en la etapa anterior.

#### Responsabilidades

- cargar jugadores
- ejecutar segmentación
- calcular features
- ejecutar clustering
- guardar resultados

---

## Clustering

Se utiliza:

```python
KMeans(n_clusters=2)
```

Los jugadores son clasificados como:

- TeamA
- TeamB

---

## Outputs generados

```text
outputs_team_finales/
```

### Archivos

#### Imagen segmentada

```text
play_teams_seg.jpg
```

#### JSON equipos

```text
play_teamdata.json
```

---

# 3. equipo_atacante.py

## Objetivo

Este script realiza el análisis geométrico para determinar posibles situaciones de offside.

---

## Flujo del script

```text
Selección equipo atacante
   ↓
Selección líneas paralelas
   ↓
Cálculo punto de fuga
   ↓
Líneas de profundidad
   ↓
Comparación jugadores
   ↓
Detección offside y asignación de etiquetas
```

---

## Funciones principales

---

### select_attacking_team()

Permite seleccionar manualmente el equipo atacante.

#### Funcionamiento

El usuario hace click sobre un jugador atacante.

---

### compute_vanishing_point()

Calcula el punto de fuga.

#### Funcionamiento

Se utilizan dos líneas paralelas del campo seleccionadas manualmente.

La intersección estimada define la perspectiva de profundidad.

---

### project_players()

Construye líneas de profundidad para cada jugador.

#### Funcionamiento

Cada jugador se conecta al punto de fuga mediante:

```text
Punto avanzado → Punto de fuga
```

---

### detect_offside()

Determina posibles situaciones de offside.

#### Funcionamiento

- identifica último defensor
- compara profundidad relativa
- etiqueta atacantes

---

## Interacción manual

### Mouse

#### Selección atacante

- click sobre jugador

#### Selección líneas

- 4 clicks:
  - 2 para línea 1
  - 2 para línea 2

---

## Outputs generados

```text
outputs_offside_results/
```

### Archivos

#### Imagen final

```text
play_offside_final.jpg
```

#### JSON final

```text
play_offside_labeled.json
```

---

# 4. seguimiento_atacantes_decision.py

## Objetivo

Este script realiza:

- tracking temporal
- detección de balón
- detección del toque
- decisión final OFFSIDE / ONSIDE
- Video con el seguimiento hasta que se ve un toque
- .json con los resultados

---

## Flujo del script

```text
Video completo
   ↓
Tracking jugadores
   ↓
Detección balón
   ↓
Asociación balón-jugador
   ↓
Detección toque
   ↓
Decisión final
```

---

## Funciones principales

---

### process_play()

Función principal del análisis temporal.

#### Responsabilidades

- cargar video
- detectar balón
- seguir jugadores
- analizar proximidad
- generar decisión final

---

### detect_ball_touch()

Detecta el posible instante de toque.

#### Funcionamiento

Se calcula:

```text
distancia(ball, jugador)
```

El jugador más cercano al balón es considerado posible ejecutor del pase.

---

### compute_metrics()

Calcula métricas finales.

#### Métricas

- Accuracy
- Precision
- Recall
- F1 Score

---

## Decisión final

El sistema produce:

```text
OFFSIDE
```

o

```text
ONSIDE
```

---

## Outputs generados

```text
outputs_tracking_ball_touch_eval/
```

### Archivos

#### Video final

```text
play_tracking.mp4
```

#### Resumen JSON

```text
play_summary.json
```

#### Evaluación global

```text
ALL_TOUCH_SUMMARY.json
```

---

# run_pipeline.py

## Objetivo

Ejecutar automáticamente todo el pipeline.

---

## Orden de ejecución

```python
run_detection()
run_team_separation()
run_offside_detection()
run_tracking()
```

---

## Uso

```bash
python run_pipeline.py
```

---

# Ejemplo completo de ejecución

## Paso 1

Agregar videos:

```text
videos/
```

---

## Paso 2

Ejecutar:

```bash
python run_pipeline.py
```

---

## Paso 3

El pipeline generará automáticamente:

- detecciones
- separación por equipos
- análisis offside
- tracking
- decisión final

---

# Resultados Esperados

El sistema es capaz de:

- detectar jugadores
- separar equipos automáticamente
- estimar profundidad relativa
- detectar posibles offsides
- analizar temporalmente el pase

---

# Tecnologías utilizadas

- Python
- OpenCV
- YOLOv8
- Ultralytics
- NumPy
- Scikit-Learn

---

# Posibles mejoras futuras

- Automatización del punto de fuga
- Tracking avanzado
- Homografía automática
- Procesamiento en tiempo real
- Integración VAR
- Deploy web

---

# Autor

Héctor Sierra

Matemáticas Aplicadas y Ciencias de la Computación
# Licencia

MIT License
