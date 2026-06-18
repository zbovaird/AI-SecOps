# Local Runtime And Configuration

- OS: macOS Apple Silicon
- Python: 3.13 (`.python-version`)
- Ollama: `llama3:latest` at `http://localhost:11434`
- Wolfram Engine: 14.3.0 cask, kernel at `/Applications/Wolfram Engine.app/Contents/Resources/Wolfram Player.app/Contents/MacOS/WolframKernel`

Manual Ollama serve (if brew services fails):

```bash
OLLAMA_FLASH_ATTENTION='1' OLLAMA_KV_CACHE_TYPE='q8_0' ollama serve
```

Verification:

```bash
.venv/bin/python -m pytest -q
wolframscript -code '$Version'
ollama list
```
