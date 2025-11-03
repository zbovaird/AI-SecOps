# Accessing the Gradio Interface

## ✅ Correct URL

**Use HTTP (not HTTPS):**

```
http://localhost:7860
```

## ❌ Common Error

If you see `ERR_SSL_PROTOCOL_ERROR` or "site can't provide a secure connection", you're likely using:

```
https://localhost:7860  ← This won't work!
```

Gradio runs on **HTTP** by default for local development. There's no SSL certificate, so HTTPS won't work.

## Quick Fix

1. **Make sure you're using HTTP:**
   ```
   http://localhost:7860
   ```

2. **If your browser auto-redirects to HTTPS:**
   - Type `http://` explicitly in the address bar
   - Or try: `http://127.0.0.1:7860`

3. **If browser shows security warning:**
   - Click "Advanced" → "Proceed to localhost" (Chrome)
   - Or "Accept the Risk" (Firefox)

## Verify Server is Running

```bash
# Check if server is running
ps aux | grep pyrit_gradio_app

# Test connection
curl http://localhost:7860
```

If curl returns HTML, the server is working correctly.

## Restart Server if Needed

If you need to restart:

```bash
# Stop server
pkill -f pyrit_gradio_app.py

# Start again
cd "/Users/zachbovaird/Documents/GitHub/AI SecOps"
conda activate aisecops
python pyrit_gradio_app.py
```

## Alternative: Use IP Address

If `localhost` doesn't work, try:
```
http://127.0.0.1:7860
```

