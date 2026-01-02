# Running the Safe Network Test

## Issue
Python scripts are being killed by the system (likely resource limits or security restrictions).

## Manual Execution Options

### Option 1: Run Directly in Terminal

Open a terminal and run:

```bash
cd "/Users/zachbovaird/Documents/GitHub/AI SecOps/redteam_kit"
python3 examples/minimal_safe_test.py
```

### Option 2: Run with Python Interactive Mode

```bash
cd "/Users/zachbovaird/Documents/GitHub/AI SecOps/redteam_kit"
python3 -i examples/minimal_safe_test.py
```

### Option 3: Use Python Script Directly

```bash
cd "/Users/zachbovaird/Documents/GitHub/AI SecOps/redteam_kit"
chmod +x examples/minimal_safe_test.py
./examples/minimal_safe_test.py
```

### Option 4: Run Step by Step (Python Shell)

```python
import sys
sys.path.insert(0, '/Users/zachbovaird/Documents/GitHub/AI SecOps/redteam_kit')
exec(open('examples/minimal_safe_test.py').read())
```

## What the Script Does

1. **Detects your network range** (e.g., 192.168.1.0/24)
2. **Scans first 10 IPs** (safe, limited range)
3. **Pings each IP** to find active hosts
4. **Scans common ports** (22, 80, 443) - read-only
5. **Generates reports** in `reports/` directory

## Expected Output

```
======================================================================
SAFE LOCAL NETWORK TEST (Minimal)
======================================================================

[+] Hostname: your-hostname
[+] Local IP: 192.168.1.X
[+] Network: 192.168.1.0/24
[+] Scanning first 10 IPs only

[1/4] Testing ping...
[+] Localhost: alive

[2/4] Discovering hosts...
  192.168.1.1... ✓ Found (router.local)
  192.168.1.2... ✗
  ...

[3/4] Scanning common ports...
[4/4] Generating report...

======================================================================
RESULTS
======================================================================
Hosts Discovered: 3

Discovered Hosts:
  192.168.1.1        router.local            Ports: 80, 443
  192.168.1.100      server01.local          Ports: 22, 80, 443
  192.168.1.101      workstation01.local     Ports: 22

[+] Report saved: reports/safe_test_TIMESTAMP.md
[+] JSON saved: reports/safe_test_TIMESTAMP.json
[+] Test completed safely - no disruption
======================================================================
```

## Troubleshooting

If Python keeps getting killed:

1. **Check system limits:**
   ```bash
   ulimit -a
   ```

2. **Try different Python:**
   ```bash
   /usr/bin/python3 examples/minimal_safe_test.py
   ```

3. **Check Activity Monitor** for resource usage

4. **Run from different directory:**
   ```bash
   cd ~
   python3 "/Users/zachbovaird/Documents/GitHub/AI SecOps/redteam_kit/examples/minimal_safe_test.py"
   ```

5. **Check macOS security settings:**
   - System Preferences > Security & Privacy
   - Allow Python if blocked

## Alternative: Use Shell Script

If Python continues to fail, I can create a bash script version that uses native tools (ping, nmap, etc.) instead of Python.

## Reports Generated

- `reports/safe_test_TIMESTAMP.md` - Markdown report
- `reports/safe_test_TIMESTAMP.json` - JSON report

Both reports will be created in the `reports/` directory with timestamps.

