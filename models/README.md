# Local Models

Store local embedding and reranking models here.

This directory is intentionally ignored by git except for this README and the
local `.gitignore`.

Initial target model:

- `BAAI/bge-m3`
- Local path: `models/bge-m3/`
- Hugging Face: <https://huggingface.co/BAAI/bge-m3>

Download:

```bash
huggingface-cli download BAAI/bge-m3 \
  --local-dir models/bge-m3 \
  --local-dir-use-symlinks False
```
