# Colab Extension GPU Connection Troubleshooting

## Issue: Selecting GPU (L4) but nothing happens

### Common Causes & Solutions

### 1. **Authentication Not Completed** ⚠️ MOST COMMON

**Problem**: You need to authenticate with Google before selecting GPU.

**Solution**:
1. **First, authenticate**:
   - Click "Select Kernel" (top right)
   - Choose "Colab" → **"New Colab Server"** (NOT "GPU" yet)
   - A browser window should open asking you to sign in with Google
   - Complete the Google authentication
   - Wait for the connection to establish (you'll see "Connected" status)

2. **Then select GPU**:
   - Once connected, click "Select Kernel" again
   - Choose "Colab" → "GPU" → "L4" (or T4/V100/A100)
   - The runtime should restart with GPU

**Why this happens**: The extension needs an active Colab session before it can change the runtime type.

---

### 2. **Extension Needs Restart**

**Solution**:
1. Close Cursor/VS Code completely
2. Reopen Cursor/VS Code
3. Open your notebook
4. Try connecting again

---

### 3. **Check Extension Status**

**Solution**:
1. Open Command Palette (`Cmd+Shift+P` on Mac, `Ctrl+Shift+P` on Windows/Linux)
2. Type: `Colab: Show Output`
3. Check for error messages
4. Also try: `Colab: Show Status`

---

### 4. **Manual Connection via Colab Website**

**Alternative Solution** (if extension keeps failing):
1. Go to https://colab.research.google.com/
2. Upload your notebook (`latent_space_redteaming_complete.ipynb`)
3. In Colab: **Runtime** → **Change runtime type**
4. Set **Hardware accelerator** to **GPU**
5. Select **GPU type** (T4, L4, V100, or A100)
6. Click **Save**
7. The runtime will restart with GPU

Then you can:
- Run cells directly in Colab, OR
- Connect the extension to the existing Colab session

---

### 5. **Check Network/Firewall**

**Problem**: Corporate firewall or network blocking Colab connections.

**Solution**:
- Try from a different network (home vs work)
- Check if `colab.research.google.com` is accessible
- Try VPN if on restricted network

---

### 6. **Extension Version Issue**

**Solution**:
1. Open Extensions (`Cmd+Shift+X`)
2. Find "Colab" extension
3. Check if there's an update available
4. Update if available
5. Reload window (`Cmd+R` or `Ctrl+R`)

---

### 7. **Clear Extension Cache**

**Solution**:
1. Close Cursor/VS Code
2. Delete extension cache (location varies by OS):
   - **Mac**: `~/Library/Application Support/Cursor/User/workspaceStorage/`
   - **Windows**: `%APPDATA%\Cursor\User\workspaceStorage\`
   - **Linux**: `~/.config/Cursor/User/workspaceStorage/`
3. Reopen and try again

---

### 8. **Verify GPU Selection Process**

**Correct Order**:
1. ✅ Click "Select Kernel" (top right of notebook)
2. ✅ Choose "Colab" → **"New Colab Server"** (first time)
3. ✅ Authenticate with Google (browser popup)
4. ✅ Wait for "Connected" status
5. ✅ Click "Select Kernel" again
6. ✅ Choose "Colab" → "GPU" → "L4"
7. ✅ Runtime restarts with GPU

**Wrong Order** (won't work):
- ❌ Selecting GPU before authenticating
- ❌ Selecting GPU when not connected to Colab

---

### 9. **Check for Error Messages**

**Look for**:
- Status bar (bottom) showing connection status
- Output panel (`View` → `Output` → Select "Colab" from dropdown)
- Notification popups (top right)

**Common errors**:
- "Authentication failed" → Re-authenticate
- "Connection timeout" → Check network
- "GPU not available" → Try different GPU type (T4 instead of L4)

---

### 10. **Alternative: Use Colab Directly**

If extension continues to fail, use Colab website:

1. **Upload notebook**:
   - Go to https://colab.research.google.com/
   - File → Upload notebook
   - Select `latent_space_redteaming_complete.ipynb`

2. **Set GPU**:
   - Runtime → Change runtime type
   - Hardware accelerator: GPU
   - GPU type: T4 (or L4 if available)
   - Save

3. **Run cells**:
   - Runtime → Run all
   - Or run cells individually

4. **Download results**:
   - Files → Download
   - Or mount Google Drive for persistence

---

## Quick Diagnostic Steps

Run these in order:

1. **Check extension installed**:
   - Extensions → Search "Colab" → Should see "Colab" extension installed

2. **Check authentication**:
   - Command Palette → `Colab: Sign In`
   - Complete Google sign-in

3. **Check connection**:
   - Command Palette → `Colab: Show Status`
   - Should show "Connected" or connection details

4. **Try basic connection first**:
   - Select Kernel → Colab → New Colab Server
   - Wait for connection
   - Then try GPU selection

---

## Still Not Working?

**Last resort**:
1. Uninstall Colab extension
2. Restart Cursor/VS Code
3. Reinstall Colab extension
4. Try again

**Or use Colab website directly** - it's more reliable for GPU access anyway.

---

## Verification: Check GPU is Active

Once connected, run this in a notebook cell:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
```

Expected output:
```
CUDA available: True
GPU: Tesla T4 (or L4/V100/A100)
CUDA Version: 11.x or 12.x
```

If you see `CUDA available: False`, GPU is not active.

---

## Notes

- **L4 availability**: L4 GPUs may not always be available. Try T4 as fallback.
- **Free tier limits**: Free Colab accounts have limited GPU hours per day.
- **Runtime timeout**: Colab runtimes disconnect after ~90 minutes of inactivity.

