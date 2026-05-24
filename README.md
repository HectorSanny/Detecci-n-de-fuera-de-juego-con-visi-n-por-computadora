# Automatic Offside Detection using Computer Vision

Proyecto de visión computacional para detección automática de offside en fútbol utilizando:

- YOLOv8
- Segmentación de jugadores
- Clustering por colores de uniforme
- Geometría proyectiva
- Punto de fuga
- Tracking temporal
- Detección de toque de balón

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
venv\\Scripts\\activate
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

## Detección

- `yolov8x.pt`

## Segmentación

- `yolov8x-seg.pt`

Los modelos se descargan automáticamente mediante Ultralytics.

---

# Dataset esperado

El primer frame de los videos debe ser el frame a analizar para hacer la detección de los jugadores y cada video debe ir dentro de:

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

El nombre del video debe indicar dirección de ataque.

## Ejemplos

### Ataque hacia la derecha

```text
jugada1_der.mp4
```

### Ataque hacia la izquierda

```text
jugada2_izq.mp4
```

---

# Ejecución completa

Desde la carpeta `src`:

```bash
python run_pipeline.py
```

---

# Etapas del Pipeline

---

## 1. Detección de jugadores

Script:

```text
detect_players_pose.py
```

### Funcionalidades

- Detección de personas con YOLOv8
- Filtrado espacial
- Filtrado por geometría
- Unión multi-frame
- Eliminación manual de árbitros/arqueros
- Generación de puntos avanzados

### Outputs

```text
outputs_finales/
```

Archivos generados:

- Imagen original
- Imagen final anotada
- JSON de jugadores

---

## 2. Separación por equipos

Script:

```text
separacion_por_equipo.py
```

### Funcionalidades

- Segmentación de jugadores
- Extracción de color LAB
- Clustering KMeans
- Clasificación TeamA / TeamB

### Outputs

```text
outputs_team_finales/
```

Archivos generados:

- Imagen segmentada
- JSON con equipos

---

## 3. Detección de offside

Script:

```text
equipo_atacante.py
```

### Funcionalidades

- Selección manual del equipo atacante
- Selección manual de líneas de profundidad
- Estimación del punto de fuga
- Proyección geométrica
- Decisión offside/onside

### Outputs

```text
outputs_offside_results/
```

Archivos generados:

- Imagen before
- Imagen after
- Imagen final etiquetada
- JSON final

---

## 4. Tracking y decisión final

Script:

```text
seguimiento_atacantes_decision.py
```

### Funcionalidades

- Tracking temporal de jugadores
- Detección de balón
- Asociación jugador-balón
- Confirmación de toque
- Decisión final OFFSIDE / ONSIDE
- Evaluación automática

### Outputs

```text
outputs_tracking_ball_touch_eval/
```

Archivos generados:

- Video trackeado
- JSON resumen
- Evaluación global

---

# Evaluación

El sistema calcula automáticamente:

- Accuracy
- Precision
- Recall
- F1 Score

---

# Controles Manuales

## Filtrado de jugadores

### Mouse

- Click → eliminar/restaurar jugador

### Teclado

- ENTER → confirmar
- ESC → cancelar

---

## Selección de equipo atacante

- Click sobre jugador atacante

---

## Selección de líneas de profundidad

- 4 clicks:
  - 2 para línea 1
  - 2 para línea 2

---

# Tecnologías utilizadas

- Python
- OpenCV
- YOLOv8
- Ultralytics
- NumPy
- Scikit-learn

---

# Posibles mejoras futuras

- Homografía automática
- Detección automática del punto de fuga
- Tracking con DeepSORT / ByteTrack
- Integración VAR en tiempo real
- Interfaz gráfica
- Deploy web
- Entrenamiento especializado en fútbol

---

# Autor

Héctor Sierra

Matemáticas Aplicadas y Ciencias de la Computación
