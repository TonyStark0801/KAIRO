---
library_name: mlx
pipeline_tag: automatic-speech-recognition
---

# whisper-large-v3-turbo
This model was converted to MLX format from [`large-v3-turbo`]().

## Use with mlx
```bash
pip install mlx-whisper
```

```python
import mlx_whisper

result = mlx_whisper.transcribe(
    "FILE_NAME",
    path_or_hf_repo=mlx-community/whisper-large-v3-turbo,
)
```
