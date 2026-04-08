---
title: Mano LSC Translator
emoji: 🤟
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🤟 Mano — Traductor de Lengua de Señas Colombiana

**Mano** traduce el deletreo manual de la Lengua de Señas Colombiana (LSC) a texto en tiempo real, directamente desde tu navegador.

Apunta tu cámara a una mano haciendo una seña y Mano la reconoce, construye palabras letra por letra y (próximamente) usa IA para corregir la ortografía en palabras reales del español.

👉 **Pruébalo en vivo:** [davidrfb.github.io/Mano](https://davidrfb.github.io/Mano/)

---

## El Problema

Colombia tiene más de 500.000 personas sordas o con dificultades auditivas que usan LSC (Lengua de Señas Colombiana). Las barreras de comunicación entre la comunidad sorda y la oyente siguen siendo uno de los mayores obstáculos para la inclusión en educación, salud y la vida cotidiana.

Las herramientas de traducción existentes son escasas, costosas o están diseñadas para ASL (lengua de señas americana) — no para LSC. La LSC tiene su propio alfabeto, gramática y variaciones regionales que los modelos genéricos no manejan bien.

## La Solución

Mano es un pipeline completo que:

1. **Detecta los puntos de la mano** usando MediaPipe directamente en el navegador (el video nunca sale de tu dispositivo)
2. **Extrae características** de 21 puntos de referencia — posiciones, ángulos de los dedos y distancias clave
3. **Clasifica la letra** usando una red neuronal ligera alojada en Hugging Face Spaces
4. **Construye palabras** con un sistema de captura por estabilidad (una letra debe mantenerse firme para registrarse)
5. **Corrige ortografía** (próximamente) usando un LLM para convertir secuencias ruidosas de letras en palabras reales del español

Todo funciona solo con una webcam — sin hardware especial, sin instalar apps, sin registro.

---

## Cómo Funciona

```
Navegador (tu dispositivo)         Nube (HF Spaces)
┌──────────────────────┐           ┌──────────────────┐
│  Cámara              │           │  Servidor FastAPI │
│  ↓                   │           │                   │
│  MediaPipe Hands     │  landmarks│  Normalizar +     │
│  (21 puntos)     ────┼──── JSON ─┼→ extraer features │
│                      │           │  ↓                │
│  Dibujar landmarks   │  letra +  │  Red neuronal     │
│  Mostrar predicción←─┼── JSON ──┼← (MLP / BiGRU)    │
│  Construir palabra   │ confianza │                   │
└──────────────────────┘           └──────────────────┘
```

**Privacidad:** El video nunca sale de tu dispositivo. Solo se envían al servidor las 21 coordenadas (x, y, z) de los puntos de referencia.

---

## Cobertura del Alfabeto

Mano reconoce las **27 letras** del alfabeto LSC:

- **22 letras estáticas** (pose fija de la mano): A–I, K–R, T–Y
- **5 letras dinámicas** (requieren movimiento): H, J, S, Z, Ñ

El mejor modelo actual alcanza **97% de precisión en test** para letras estáticas usando el modo de características `xy_angles_distances`.

---

## Estructura del Proyecto

```
Mano/
├── api/main.py              # Servidor de predicción (FastAPI)
├── docs/index.html           # Frontend web (GitHub Pages)
├── scripts/                  # Captura de datos, entrenamiento, evaluación
├── src/
│   ├── preprocessing/        # Normalización y features de landmarks
│   ├── models/               # Modelos MLP (estático) y RNN (dinámico)
│   ├── training/             # Loop de entrenamiento y métricas
│   └── llm/                  # Corrección de palabras con LLM
├── data/                     # Datasets de landmarks (versionados con DVC)
├── models/                   # Checkpoints y experimentos MLflow
└── blog/                     # Blog de progreso del proyecto
```

Ver [STRUCTURE.md](STRUCTURE.md) para el desglose completo.

---

## Inicio Rápido

### Ejecutar el demo localmente

```bash
# 1. Crear entorno
micromamba create -n Mano python=3.11
micromamba activate Mano

# 2. Instalar dependencias
micromamba install pytorch torchvision -c pytorch -c conda-forge
pip install -r requirements.txt

# 3. Ejecutar la API localmente
uvicorn api.main:app --reload --port 8000

# 4. Abrir docs/index.html en tu navegador
#    (cambia la URL de la API en configuración a http://localhost:8000)
```

### Entrenar un modelo

```bash
# Capturar datos de entrenamiento
python scripts/01_capture_static.py --mode landmarks

# Entrenar
python scripts/04_train.py --model static --features xy_angles_distances --epochs 100

# Evaluar
python scripts/05_evaluate.py --checkpoint models/checkpoints/<run>/best.pth
```

### Desplegar en Hugging Face Spaces

```bash
docker build -t mano-api .
docker run -p 7860:7860 mano-api
```

---

## Stack Tecnológico

| Componente | Tecnología |
|-----------|-----------|
| Detección de mano | MediaPipe Hands (en el navegador) |
| Modelo ML | PyTorch (MLP para estáticas, BiGRU para dinámicas) |
| API | FastAPI + Uvicorn |
| Hosting | Hugging Face Spaces (Docker) |
| Frontend | HTML/JS vanilla (GitHub Pages) |
| Seguimiento de experimentos | MLflow |
| Versionado de datos | DVC (Google Drive) |

---

## Hoja de Ruta

- [x] Reconocimiento de letras estáticas (97% precisión)
- [x] Demo en tiempo real en el navegador
- [x] Desplegar API en Hugging Face Spaces
- [ ] Reconocimiento de letras dinámicas en producción
- [ ] Corrección de palabras con LLM en la web
- [ ] Reconocimiento a nivel de palabras completas
- [ ] UI optimizada para móviles

---

## Licencia

Este proyecto es para fines educativos y de investigación.

---

Hecho con 🇨🇴 para la comunidad sorda colombiana.
