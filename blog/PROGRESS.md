# ML Progress: Sign Language Recognition

## Approaches Compared

### 1. CNN-Based Model (MobileNetV2/V3)
- **Input**: Raw RGB images (224x224)
- **Architecture**: Pretrained MobileNet with custom classifier head
- **Result**: Did not perform as well as expected

**Why it struggled:**
- High variance in hand position, distance to camera, lighting
- The model tries to learn everything: hand shape, background, skin tone, etc.
- Requires large amounts of diverse data to generalize
- Preprocessing choices (crop vs resize) significantly affected results

### 2. Landmark-Based Model (MLP)
- **Input**: MediaPipe hand landmarks (21 points x,y coordinates + joint angles + distances)
- **Architecture**: Simple 2-layer MLP (66 features → 128 → 64 → classes)
- **Result**: Works better and is significantly faster

**Why it works better:**
- MediaPipe already solves hand detection and normalization
- Landmarks are invariant to lighting, skin tone, background
- Much smaller input space (66 features vs 224x224x3 = 150K pixels)
- Faster inference: landmark extraction + MLP is lightweight

## Feature Engineering Journey

### xy_angles (56 features) — First attempt
- 42 XY coordinates + 14 finger joint angles
- 77.8% val / 78.6% test accuracy
- Problem: R/V/Z confusion (all involve index+middle finger extended)

### xy_angles_distances (66 features) — Current best
- Added 10 key distances (fingertip spreads, palm width, wrist distances)
- **96.9% val / 97.0% test accuracy**
- Distance features help separate similar finger configurations (R vs V vs Z)
- Key insight: fingertip-to-fingertip distances capture finger crossing/separation that XY alone misses

## Solved Challenges

### Dynamic letters in static model (J)
- J is a dynamic letter (draw a J shape with pinky)
- Frame 0 of J looks identical to I (both: fist with pinky up)
- **Solution**: `--static-frame-override j:18` — train J using frame 18 (end of motion)
- The final hand position of J is distinctive enough for static classification

### Ñ display in OpenCV
- OpenCV `putText` only supports ASCII — "Ñ" showed as "??"
- **Solution**: Display as "NN" in all CV windows, keep "Ñ" in terminal output

### Hand distance to camera
- Not an issue: normalization (center on wrist, scale by max XY distance) already handles this
- Z-axis dropped entirely (unreliable from single camera)
- Hand close or far produces same normalized features

## Current Challenges

### Similar Signs
Some letter pairs still confuse the model:
- **R/V/Z** — all involve index+middle fingers (improved with distances)
- **N/M/P** — similar closed-fist shapes
- **X/F** — similar bent-finger shape

**Potential solutions:**
- More training data (currently ~6-15 samples per class)
- Sequence models (BiGRU) for dynamic letters
- Attention mechanisms for subtle differences

### Data Size
- 214 total samples across 27 classes (~8 average per class)
- Some classes have only 2 samples (NN)
- Model memorizes more than generalizes at this scale
- Target: 30+ samples per class with varied hand angles

## Architecture Summary

```
Raw Video
    │
    ├──► [MediaPipe] ──► Landmarks (21x3) ──► [Normalize] ──► [Features 66d] ──► [MLP] ──► Prediction
    │                                              │                  │
    │                                     Center on wrist      xy + angles
    │                                     Scale by hand size   + 10 distances
    │
    └──► [Resize 224] ──► [MobileNet] ──► Prediction
                                              ↑
                                    Deprecated: needs more data
```

## Model History

| Run | Model | Features | Val Acc | Test Acc | Notes |
|-----|-------|----------|---------|----------|-------|
| `c6d5eb3a` | static | xy_angles (56) | 70.0% | 81.8% | 69 samples, first attempt |
| `bd9eea30` | static | xy_angles (56) | 77.8% | 78.6% | 183 samples, j:18 override |
| `81b912ec` | static | xy_angles_distances (66) | **96.9%** | **97.0%** | 214 samples, current best |

## Deployment

- **API**: FastAPI on Hugging Face Spaces (Docker)
- **Endpoint**: `POST /predict` accepts single frame or sequence
- **Model size**: 236 KB (.pth with config)
- **No MLflow at runtime** — model loads directly from checkpoint

## Key Takeaways

1. **Simpler is better** (for now): The MLP on landmarks outperforms CNN on raw images with limited data
2. **Let specialized models do the heavy lifting**: MediaPipe handles hand detection/tracking, we focus on classification
3. **Speed matters**: Real-time demo needs fast inference; MLP is ideal
4. **Feature engineering matters**: Adding 10 distance features jumped accuracy from 78% to 97%
5. **Data quality > quantity**: For similar signs, better features help more than more of the same data

## Next Steps

- [x] Collect more landmark data to improve class coverage
- [x] Investigate which letter pairs are most confused (confusion matrix)
- [x] Explore normalization techniques for hand-camera distance invariance
- [x] Deploy API to Hugging Face Spaces
- [x] Web frontend: switch to landmark-based JSON prediction
- [x] Mobile polish: camera flip, minimal layout, fixed-height word display

### Short-term: Improve failing letters
- [ ] Fix NN (Ñ) — only 2 samples, model can't generalize; collect 15+ more
- [ ] Improve N/M/P confusion — similar closed-fist shapes, need more varied angles
- [ ] Improve X/F confusion — subtle bent-finger differences
- [ ] Collect 30+ samples per class for robust generalization

### Medium-term: Motion & dynamic letters
- [ ] Experiment with sequence models (BiGRU) for dynamic letters (J, H, Z, Ñ, S)
- [ ] Build dynamic letter capture pipeline (multi-frame sequences)
- [ ] Hybrid static+dynamic model — static MLP for most letters, sequence model for motion letters

### Long-term: Words & sentences
- [ ] Train word-level recognition with holistic landmarks (pose + both hands)
- [ ] Add multi-hand support for two-handed signs
- [ ] LLM-powered word correction from letter sequences
- [ ] Continuous gesture segmentation (no manual letter boundaries)
