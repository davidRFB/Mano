# Roadmap

Estado actual del proyecto y direcciones futuras.

**Versión actual:** 0.5.1 | **Última actualización:** 2026-04-08

---

## Estado Actual

### Funcionando
- Modelo estático de letras: **97% precisión** en test (MLP, 66 features)
- API desplegada en Hugging Face Spaces (FastAPI + Docker)
- Frontend web en GitHub Pages con detección en tiempo real
- Pipeline de captura, entrenamiento y evaluación completo
- Landmarks de MediaPipe procesados en el navegador (video nunca sale del dispositivo)

### Limitaciones Conocidas
- Dataset pequeño: 214 muestras en 27 clases (~8 por clase)
- NN (Ñ) tiene solo 2 muestras — el modelo no generaliza
- Confusión entre letras similares: N/M/P, X/F, R/V/Z
- Letras dinámicas (J, H, Z, Ñ, S) no están en producción
- Sin tests unitarios

---

## Fase 1: Mejorar el Modelo Actual

**Objetivo:** Subir de 97% a 99%+ con datos más robustos.

- [ ] **Más datos para letras débiles**
  - NN (Ñ): 2 → 30+ muestras
  - N, M, P: capturar con ángulos de mano variados
  - X, F: capturar diferencias sutiles de flexión
  - Meta: 30+ muestras por clase

- [ ] **Evaluar con datos nuevos**
  - Capturar datos con otras personas (no solo el creador)
  - Probar con diferentes fondos, iluminación, distancias
  - Medir si el modelo generaliza o memoriza

- [ ] **Tests unitarios**
  - `tests/test_preprocessing.py` — normalización, extracción de features
  - `tests/test_models.py` — forward pass, shapes correctas
  - `tests/test_api.py` — endpoints, validación de input

---

## Fase 2: Letras Dinámicas en Producción

**Objetivo:** Soporte completo del alfabeto LSC, incluyendo letras con movimiento.

- [ ] **Modelo de secuencias (BiGRU)** para J, H, Z, Ñ, S
  - Ya existe la arquitectura en `src/models/dynamic.py`
  - Ya existe captura de secuencias en `scripts/02_capture_dynamic.py`
  - Falta: entrenar, evaluar, y desplegar

- [ ] **Modelo híbrido estático+dinámico**
  - Router que decide si usar MLP (estático) o BiGRU (dinámico)
  - La API ya soporta ambos formatos (single frame vs sequence)

- [ ] **Frontend: buffer de frames**
  - Enviar secuencias de landmarks para letras dinámicas
  - Detectar automáticamente cuándo hay movimiento

---

## Fase 3: Corrección con LLM

**Objetivo:** Convertir secuencias ruidosas de letras en palabras reales del español.

- [ ] **Conectar `src/llm/corrector.py` al frontend**
  - Backend ya existe (Groq/Ollama)
  - Agregar endpoint `/correct` al API
  - Botón "AI Correct" en el frontend (actualmente deshabilitado)

- [ ] **Mejorar corrección con contexto**
  - Usar confianza del modelo como señal (letras con baja confianza son más flexibles)
  - Autocomplete con diccionario español antes de usar LLM

---

## Fase 4: Reconocimiento de Palabras

**Objetivo:** Ir más allá del deletreo — reconocer señas completas de palabras.

- [ ] **Migrar pipeline de palabras a la arquitectura nueva**
  - Existe en `src/cv_model/words_*.py` (legacy)
  - Portar a `src/preprocessing/` + `src/models/`
  - Dataset: 1482 videos, 1251 palabras (INSOR + YouTube)

- [ ] **Soporte multi-mano**
  - Muchas señas de palabras usan ambas manos
  - MediaPipe Holistic: 51 landmarks (pose + 2 manos)

- [ ] **Segmentación continua de gestos**
  - Detectar inicio/fin de señas automáticamente
  - Sin depender de que el usuario marque los límites

---

## Fase 5: Pulir para Usuarios Reales

**Objetivo:** Que cualquier persona pueda usarlo sin instrucciones.

- [ ] **UI móvil optimizada**
  - Mejorar rendimiento en dispositivos de gama baja
  - Progressive Web App (PWA) — instalar como app
  - Modo offline para la parte de MediaPipe

- [ ] **Onboarding**
  - Tutorial interactivo para nuevos usuarios
  - Guía visual de las señas del alfabeto LSC

- [ ] **Accesibilidad**
  - Feedback háptico cuando se captura una letra
  - Modo alto contraste
  - Soporte para lectores de pantalla

---

## Ideas Exploratorias

No están en el plan inmediato, pero vale la pena investigar:

- **Fine-tuning del modelo con datos del usuario** — personalizar a la mano de cada persona
- **Modelo edge (ONNX/TFLite)** — correr el modelo directamente en el navegador, sin API
- **Señas regionales** — LSC tiene variaciones por ciudad (Bogotá vs Medellín vs Cali)
- **Traducción bidireccional** — texto/voz → animación de señas

---

## Deuda Técnica

Cosas que no bloquean pero deberían resolverse:

- [ ] `src/cv_model/` — archivar o mover a rama separada (16MB de notebooks legacy)
- [ ] `weights_only=False` en `torch.load` — riesgo de seguridad si el modelo fuera comprometido
- [ ] DVC remote en Google Drive personal — considerar almacenamiento más accesible
- [ ] GEMINI.md — revisar si sigue siendo relevante o se puede eliminar

---

## Contribuir

Si quieres ayudar:
1. **Datos**: Lo más valioso es capturar más señas con `scripts/01_capture_static.py`
2. **Tests**: Agregar tests para los módulos core
3. **UI**: Mejorar la experiencia móvil en `docs/index.html`
4. **Investigar**: Probar nuevas arquitecturas o features para las letras problemáticas
