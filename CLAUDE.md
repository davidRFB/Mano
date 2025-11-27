# CLAUDE.md - AI Assistant Guidelines

## Project Context
MANO: Colombian Sign Language translator using CV model + LLM pipeline.

## Core Principles
- **Ask, don't assume**: When requirements are ambiguous, ask clarifying questions
- **Concise over verbose**: Short, clear explanations. No unnecessary elaboration
- **Show, don't tell**: Provide working code examples, not theoretical explanations
- **Incremental changes**: Small, testable modifications over large refactors. DO NOT MAKE MASSIVE CHANGES WITHOUT PRIOR DISCUSSION
- **Follow existing patterns**: Match coding style and structure already present in the codebase

---

## Coding Standards

### Python Style
- **Formatter**: Black (line length: 88)
- **Linter**: Flake8
- **Type hints**: Required for all functions
- **Docstrings**: Google style for public functions only
- **Imports**: Absolute imports, grouped (stdlib → third-party → local)

```python
# Good
def predict_gesture(image: np.ndarray, model: torch.nn.Module) -> dict[str, float]:
    """Predict gesture from image."""
    pass

# Bad
def predict_gesture(image, model):
    '''
    This function takes an image and a model and returns predictions...
    '''
    pass
```

### Project Structure Awareness
- **Always check STRUCTURE.md** before creating new files 
- **Update CHANGELOG.md** when adding features or making significant changes. Make sure they are finished to update. Wait for user input explicitly saying the feature is finished.
- **Follow existing patterns** - if similar code exists, match its style

### Testing
- Test file naming: `test_*.py`
- One test file per module
- Use pytest fixtures for setup
- Aim for >80% coverage

### API Design
- RESTful conventions
- Pydantic models for request/response
- Explicit error handling with appropriate HTTP status codes
- Include request validation

```python
# Good
@app.post("/predict", response_model=PredictionResponse)
async def predict_gesture(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    # ...
```

### Docker
- Multi-stage builds when beneficial
- Minimal base images (slim/alpine variants)
- .dockerignore to exclude unnecessary files
- Health checks in Dockerfile

### Git Workflow
- Branch naming: `feature/`, `fix/`, `refactor/`
- Commits: Imperative mood, concise ("Add prediction endpoint" not "Added prediction endpoint")
- **Update CHANGELOG.md** before merging to main

---

## File Organization

### Key Reference Files
- **STRUCTURE.md**: Complete file structure and descriptions
- **CHANGELOG.md**: Feature additions, changes, and fixes
- **README.md**: User-facing documentation

### Before Creating New Files
1. Check STRUCTURE.md to see if file/location already exists
2. Follow existing naming patterns
3. Update STRUCTURE.md only when a feature is finished. A conversation can go for a while before a feature is finished.  Wait for user input explicitly saying the feature is finished.

---

## Development Workflow

### When Adding Features
1. **Understand context**: Read relevant sections of STRUCTURE.md
2. **Plan changes**: Ask questions if unclear
3. **Implement incrementally**: One logical unit at a time
4. **Test immediately**: Don't accumulate untested code
5. **Update documentation**: CHANGELOG.md + STRUCTURE.md if needed

### When Debugging
1. **Reproduce first**: Ensure you understand the problem
2. **Minimal changes**: Fix the issue, don't refactor unnecessarily
3. **Add test**: Prevent regression

### When Refactoring
1. **Justify need**: Explain why refactoring is necessary
2. **Preserve behavior**: Tests should still pass
3. **One thing at a time**: Don't mix refactoring with features

---

## MLOps Conventions

### Experiment Tracking (MLflow)
- Log all hyperparameters
- Log metrics every epoch
- Save model artifacts with version tags
- Use descriptive run names: `mobilenetv2_lr0.001_aug_heavy`

### Model Versioning (DVC)
- Track all datasets in `data/`
- Track model checkpoints in `models/`
- Commit `.dvc` files to git
- Use semantic versioning for model releases

### Model Files
- Naming: `{model_arch}_v{version}_{metric}.pth`
- Example: `mobilenetv2_v1.2_acc0.87.pth`
- Include metadata JSON with training config

---

## Cloud & Deployment

### Environment Variables
- Never hardcode secrets
- Use `.env` files locally (gitignored)
- Use cloud secret managers in production
- Document all required env vars in README

### Docker Images
- Tag with semantic versions: `v1.0.0`, `v1.0.1`
- Also tag `latest` for convenience
- Include git commit SHA in image label

### Logging
- Use structured logging (JSON format)
- Log levels: DEBUG (dev), INFO (prod), ERROR (always)
- Include trace IDs for request tracking

---

## Communication Style

### When Responding to Requests
- **Confirm understanding** if request is complex
- **Propose approach** before implementing large changes
- **Highlight tradeoffs** when multiple solutions exist
- **Ask for priorities** when requirements conflict

### Code Comments
- **Why, not what**: Explain reasoning, not obvious behavior
- **Warnings**: Flag gotchas, performance considerations, dependencies
- **TODOs**: Include context and priority

```python
# Good
# Using quantization here reduces inference time by 40% but 
# slightly decreases accuracy (0.87 → 0.85)
quantized_model = torch.quantization.quantize_dynamic(...)

# Bad
# Quantize the model
quantized_model = torch.quantization.quantize_dynamic(...)
```

---

## Anti-Patterns to Avoid

### Code
- ❌ Overly complex abstractions for simple tasks
- ❌ Premature optimization
- ❌ Copy-pasting code instead of refactoring to shared function
- ❌ Ignoring errors silently (`except: pass`)
- ❌ Magic numbers without explanation

### Communication
- ❌ Long explanations when code example is clearer
- ❌ Implementing without confirming approach first
- ❌ Assuming requirements without asking
- ❌ Making breaking changes without discussion

### Project Management
- ❌ Forgetting to update CHANGELOG.md
- ❌ Creating files without checking STRUCTURE.md
- ❌ Mixing multiple unrelated changes in one commit
- ❌ Skipping tests "to save time"

---

## Quick Reference

### Starting a Feature
```bash
# 1. Check structure
cat STRUCTURE.md | grep -A 5 "relevant_section"

# 2. Create branch
git checkout -b feature/your-feature

# 3. Implement & test
pytest tests/test_new_feature.py

# 4. Update docs
# Edit CHANGELOG.md
# Edit STRUCTURE.md if files added

# 5. Commit
git commit -m "Add feature: brief description"
```

### Helpful Commands
```bash
# Format code
black src/

# Lint
flake8 src/

# Type check
mypy src/

# Run tests with coverage
pytest --cov=src tests/

# Check what DVC tracks
dvc status

# View MLflow runs
mlflow ui
```

---

## Questions to Ask

When context is unclear, ask:
- "What's the expected behavior if X fails?"
- "Should this be optimized for speed or accuracy?"
- "Do you want me to update tests/docs as well?"
- "Is this blocking other work, or can we iterate?"
- "Where should this fit in the existing structure?"

---

## Version
Last updated: 2025-11-27
Review this file periodically as project evolves.