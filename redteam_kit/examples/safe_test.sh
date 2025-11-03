#!/bin/bash
# Safe Local Network Test (Shell Script Version)
# Uses native tools to avoid Python execution issues

set -e

echo "======================================================================"
echo "SAFE LOCAL NETWORK TEST (Shell Script)"
echo "======================================================================"
echo ""
echo "[*] Safe operations only - no disruption"
echo ""

# Get network info
if [[ "$OSTYPE" == "darwin"* ]]; then
    LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "127.0.0.1")
    HOSTNAME=$(hostname)
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    LOCAL_IP=$(hostname -I | awk '{print $1}')
    HOSTNAME=$(hostname)
else
    LOCAL_IP="127.0.0.1"
    HOSTNAME=$(hostname)
fi

NETWORK_PREFIX=$(echo $LOCAL_IP | cut -d'.' -f1-3)

echo "[+] Hostname: $HOSTNAME"
echo "[+] Local IP: $LOCAL_IP"
echo "[+] Network: $NETWORK_PREFIX.0/24"
echo "[+] Scanning first 10 IPs only"
echo ""

# Create reports directory
REPORT_DIR="reports"
mkdir -p "$REPORT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MD_REPORT="$REPORT_DIR/safe_test_${TIMESTAMP}.md"
JSON_REPORT="$REPORT_DIR/safe_test_${TIMESTAMP}.json"

# Test localhost
echo "[1/4] Testing ping..."
if ping -c 1 -W 1 127.0.0.1 > /dev/null 2>&1; then
    echo "[+] Localhost: alive"
else
    echo "[+] Localhost: not responding"
fi
echo ""

# Discover hosts
echo "[2/4] Discovering hosts..."
DISCOVERED=()

for i in {1..10}; do
    IP="${NETWORK_PREFIX}.${i}"
    echo -n "  ${IP}... "
    
    if ping -c 1 -W 1 "$IP" > /dev/null 2>&1; then
        # Try to get hostname
        HOSTNAME_IP=$(getent hosts "$IP" 2>/dev/null | awk '{print $2}' | head -1)
        if [ -z "$HOSTNAME_IP" ]; then
            HOSTNAME_IP="unknown"
        fi
        
        DISCOVERED+=("$IP|$HOSTNAME_IP")
        echo "✓ Found ($HOSTNAME_IP)"
    else
        echo "✗"
    fi
    
    sleep 0.5
done

HOST_COUNT=${#DISCOVERED[@]}
echo ""
echo "[+] Found $HOST_COUNT active host(s)"
echo ""

# Scan ports (limited)
echo "[3/4] Scanning common ports (read-only)..."
SAFE_PORTS="22 80 443"

for host_entry in "${DISCOVERED[@]}"; do
    IP=$(echo "$host_entry" | cut -d'|' -f1)
    HOSTNAME_IP=$(echo "$host_entry" | cut -d'|' -f2)
    echo -n "  ${IP}... "
    
    OPEN_PORTS=()
    for port in $SAFE_PORTS; do
        if timeout 0.3 bash -c "echo > /dev/tcp/$IP/$port" 2>/dev/null; then
            OPEN_PORTS+=("$port")
        fi
        sleep 0.2
    done
    
    PORT_COUNT=${#OPEN_PORTS[@]}
    if [ $PORT_COUNT -gt 0 ]; then
        PORT_STR=$(IFS=','; echo "${OPEN_PORTS[*]}")
        echo "✓ $PORT_COUNT port(s) open ($PORT_STR)"
        # Update discovered entry with ports
        DISCOVERED=("${DISCOVERED[@]/$host_entry/$host_entry|$PORT_STR}")
    else
        echo "✓ no open ports"
        DISCOVERED=("${DISCOVERED[@]/$host_entry/$host_entry|none}")
    fi
done

echo ""

# Generate Markdown report
echo "[4/4] Generating report..."

cat > "$MD_REPORT" << EOF
# Safe Local Network Test Report

**Generated:** $(date '+%Y-%m-%d %H:%M:%S')

## Summary

- **Network:** ${NETWORK_PREFIX}.0/24
- **Hosts Discovered:** $HOST_COUNT
- **Test Type:** Safe, non-disruptive

## Discovered Hosts

| IP Address | Hostname | Open Ports |
|------------|----------|------------|
EOF

for host_entry in "${DISCOVERED[@]}"; do
    IP=$(echo "$host_entry" | cut -d'|' -f1)
    HOSTNAME_IP=$(echo "$host_entry" | cut -d'|' -f2)
    PORTS=$(echo "$host_entry" | cut -d'|' -f3)
    echo "| $IP | $HOSTNAME_IP | $PORTS |" >> "$MD_REPORT"
done

cat >> "$MD_REPORT" << EOF

## Findings

**Info:** Discovered $HOST_COUNT active host(s) on local network

---

*Safe test completed - no services disrupted*
EOF

# Generate JSON report
echo "{" > "$JSON_REPORT"
echo "  \"metadata\": {" >> "$JSON_REPORT"
echo "    \"network\": \"${NETWORK_PREFIX}.0/24\"," >> "$JSON_REPORT"
echo "    \"local_ip\": \"$LOCAL_IP\"," >> "$JSON_REPORT"
echo "    \"timestamp\": \"$TIMESTAMP\"," >> "$JSON_REPORT"
echo "    \"test_type\": \"safe_non_disruptive\"" >> "$JSON_REPORT"
echo "  }," >> "$JSON_REPORT"
echo "  \"hosts\": [" >> "$JSON_REPORT"

FIRST=true
for host_entry in "${DISCOVERED[@]}"; do
    IP=$(echo "$host_entry" | cut -d'|' -f1)
    HOSTNAME_IP=$(echo "$host_entry" | cut -d'|' -f2)
    PORTS=$(echo "$host_entry" | cut -d'|' -f3)
    
    if [ "$FIRST" = true ]; then
        FIRST=false
    else
        echo "," >> "$JSON_REPORT"
    fi
    
    echo "    {" >> "$JSON_REPORT"
    echo "      \"ip\": \"$IP\"," >> "$JSON_REPORT"
    echo "      \"hostname\": \"$HOSTNAME_IP\"," >> "$JSON_REPORT"
    echo "      \"open_ports\": \"$PORTS\"" >> "$JSON_REPORT"
    echo -n "    }" >> "$JSON_REPORT"
done

echo "" >> "$JSON_REPORT"
echo "  ]," >> "$JSON_REPORT"
echo "  \"summary\": {" >> "$JSON_REPORT"
echo "    \"hosts_discovered\": $HOST_COUNT" >> "$JSON_REPORT"
echo "  }" >> "$JSON_REPORT"
echo "}" >> "$JSON_REPORT"

# Display results
echo "======================================================================"
echo "RESULTS"
echo "======================================================================"
echo "Hosts Discovered: $HOST_COUNT"
echo ""

if [ $HOST_COUNT -gt 0 ]; then
    echo "Discovered Hosts:"
    for host_entry in "${DISCOVERED[@]}"; do
        IP=$(echo "$host_entry" | cut -d'|' -f1)
        HOSTNAME_IP=$(echo "$host_entry" | cut -d'|' -f2)
        PORTS=$(echo "$host_entry" | cut -d'|' -f3)
        printf "  %-18s %-25s Ports: %s\n" "$IP" "$HOSTNAME_IP" "$PORTS"
    done
fi

echo ""
echo "======================================================================"
echo "[+] Report saved: $MD_REPORT"
echo "[+] JSON saved: $JSON_REPORT"
echo "[+] Test completed safely - no disruption"
echo "======================================================================"
