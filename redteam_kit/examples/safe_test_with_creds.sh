#!/bin/bash
# Safe Local Network Test with Credential Harvesting (Shell Script)
# Uses native tools and Python only for credential harvesting

set -e

echo "======================================================================"
echo "SAFE LOCAL NETWORK TEST WITH CREDENTIAL HARVESTING"
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
echo ""

# Create reports directory
REPORT_DIR="reports"
mkdir -p "$REPORT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MD_REPORT="$REPORT_DIR/safe_engagement_report_${TIMESTAMP}.md"
JSON_REPORT="$REPORT_DIR/safe_engagement_report_${TIMESTAMP}.json"

START_TIME=$(date +%s)

# Step 1: Network Discovery
echo "[1/3] Network Discovery..."
echo "      Scanning first 10 IPs...\n"

DISCOVERED=()

for i in {1..10}; do
    IP="${NETWORK_PREFIX}.${i}"
    echo -n "  ${IP}... "
    
    if ping -c 1 -W 1 "$IP" > /dev/null 2>&1; then
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
echo "[+] Found $HOST_COUNT active host(s)\n"

# Step 2: Credential Harvesting
echo "[2/3] Credential Harvesting..."
echo "      Searching for credentials...\n"

# Pure shell-based credential harvesting (no Python needed)
CREDS=()
SOURCES=()

# Check environment variables
while IFS='=' read -r key value; do
    if [ ${#value} -gt 10 ]; then
        CREDS+=("type=environment_variable|source=environment|name=$key|value=$value")
        SOURCES+=("environment")
    fi
done < <(env | grep -iE "(API|SECRET|PASSWORD|TOKEN|KEY)")

# Check common config files
for file in ~/.aws/credentials ~/.aws/config ~/.env; do
    if [ -f "$file" ] && [ -r "$file" ]; then
        while IFS= read -r line; do
            CREDS+=("type=config_file|source=$file|value=$line")
            SOURCES+=("$file")
        done < <(grep -iE "(api[_-]?key|password|secret|token|bearer)" "$file" 2>/dev/null)
    fi
done

CRED_COUNT=${#CREDS[@]}

if [ $CRED_COUNT -gt 0 ]; then
    echo "[+] Credentials found: $CRED_COUNT"
    
    # Get unique sources
    UNIQUE_SOURCES=$(printf '%s\n' "${SOURCES[@]}" | sort -u | tr '\n' ',' | sed 's/,$//')
    echo "[+] Sources: $UNIQUE_SOURCES"
    
    # Parse credential types
    CRED_TYPES=""
    for cred in "${CREDS[@]}"; do
        IFS='|' read -r type source name value <<< "$cred"
        CRED_TYPES="$CRED_TYPES $type"
    done
    CRED_TYPES=$(echo "$CRED_TYPES" | tr ' ' '\n' | sort -u | tr '\n' ',' | sed 's/,$//')
    echo "[+] Types: $CRED_TYPES"
else
    echo "[+] No credentials found in accessible locations"
fi

echo ""

# Step 3: Generate Report
echo "[3/3] Generating Report...\n"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Generate Markdown report
cat > "$MD_REPORT" << EOF
# Safe Local Network Red Team Engagement Report

**Generated:** $(date '+%Y-%m-%d %H:%M:%S')
**Duration:** ${DURATION}s

## Executive Summary

- **Network:** ${NETWORK_PREFIX}.0/24
- **Hosts Discovered:** $HOST_COUNT
- **Credentials Found:** $CRED_COUNT
- **Test Type:** Safe, non-disruptive

## Network Discovery Results

### Discovered Hosts

| IP Address | Hostname |
|------------|----------|
EOF

for host_entry in "${DISCOVERED[@]}"; do
    IP=$(echo "$host_entry" | cut -d'|' -f1)
    HOSTNAME_IP=$(echo "$host_entry" | cut -d'|' -f2)
    echo "| $IP | $HOSTNAME_IP |" >> "$MD_REPORT"
done

cat >> "$MD_REPORT" << EOF

## Credential Harvesting Results

EOF

if [ $CRED_COUNT -gt 0 ]; then
    cat >> "$MD_REPORT" << EOF
### 🔐 Credentials Discovered

**Total Credentials Found:** $CRED_COUNT

**Sources:** $(printf '%s\n' "${SOURCES[@]}" | sort -u | tr '\n' ',' | sed 's/,$//')

### Detailed Credential Information

> **⚠️ WARNING:** The following credentials were discovered during this engagement.
> **IMMEDIATE ACTION REQUIRED:** Rotate all credentials immediately.

EOF

    # Format credentials in Markdown
    i=1
    for cred in "${CREDS[@]}"; do
        IFS='|' read -r type source name value <<< "$cred"
        echo "#### Credential $i: $(echo $type | tr '[:lower:]' '[:upper:]')" >> "$MD_REPORT"
        echo "" >> "$MD_REPORT"
        echo "- **Type:** \`$type\`" >> "$MD_REPORT"
        if [ -n "$name" ] && [ "$name" != "none" ]; then
            echo "- **Variable Name:** \`$name\`" >> "$MD_REPORT"
        fi
        echo "- **Source:** \`$source\`" >> "$MD_REPORT"
        echo "- **Value:** \`$value\`" >> "$MD_REPORT"
        echo "" >> "$MD_REPORT"
        i=$((i + 1))
    done
    
    cat >> "$MD_REPORT" << EOF
### ⚠️ CRITICAL: Credential Security Recommendations

**IMMEDIATE ACTION REQUIRED:** Credentials were discovered during this engagement.

1. **Rotate all discovered credentials immediately**
   - Change passwords, API keys, tokens, and secrets
   - Invalidate existing sessions

2. **Review credential storage locations:**
EOF

    printf '%s\n' "${SOURCES[@]}" | sort -u | while read -r source; do
        echo "   - \`$source\`" >> "$MD_REPORT"
    done
    
    cat >> "$MD_REPORT" << EOF

3. **Implement credential management best practices:**
   - Use environment variables or secure vaults (AWS Secrets Manager, HashiCorp Vault)
   - Never commit credentials to version control
   - Use least-privilege access principles
   - Regularly audit credential storage
   - Implement credential rotation policies
   - Use strong, unique credentials
   - Monitor for credential exposure

EOF
else
    cat >> "$MD_REPORT" << EOF
**No credentials found in accessible locations.**

This could mean:
- Credentials are properly secured in protected locations
- Credentials are stored in secure vaults
- No accessible credential files were found

EOF
fi

cat >> "$MD_REPORT" << EOF

## Findings

**Info:** Discovered $HOST_COUNT active host(s) on local network

EOF

if [ "$CRED_COUNT" -gt 0 ]; then
    cat >> "$MD_REPORT" << EOF
**High:** Credentials discovered - immediate rotation required

EOF
fi

cat >> "$MD_REPORT" << EOF
## Recommendations

- Monitor network for unauthorized devices
- Review exposed services and restrict access where possible
- Implement network segmentation
- Enable logging and monitoring
- Regular security assessments

---

*Safe test completed - no services disrupted*
*Report generated at: $(date '+%Y-%m-%d %H:%M:%S')*
EOF

# Generate JSON report
cat > "$JSON_REPORT" << EOF
{
  "metadata": {
    "network": "${NETWORK_PREFIX}.0/24",
    "local_ip": "$LOCAL_IP",
    "timestamp": "$TIMESTAMP",
    "duration": $DURATION,
    "test_type": "safe_non_disruptive"
  },
  "hosts": [
EOF

FIRST=true
for host_entry in "${DISCOVERED[@]}"; do
    IP=$(echo "$host_entry" | cut -d'|' -f1)
    HOSTNAME_IP=$(echo "$host_entry" | cut -d'|' -f2)
    
    if [ "$FIRST" = true ]; then
        FIRST=false
    else
        echo "," >> "$JSON_REPORT"
    fi
    
    echo "    {" >> "$JSON_REPORT"
    echo "      \"ip\": \"$IP\"," >> "$JSON_REPORT"
    echo "      \"hostname\": \"$HOSTNAME_IP\"" >> "$JSON_REPORT"
    echo -n "    }" >> "$JSON_REPORT"
done

# Add credentials to JSON
if [ $CRED_COUNT -gt 0 ]; then
    echo "," >> "$JSON_REPORT"
    echo "  \"credentials\": [" >> "$JSON_REPORT"
    
    FIRST=true
    for cred in "${CREDS[@]}"; do
        IFS='|' read -r type source name value <<< "$cred"
        
        if [ "$FIRST" = true ]; then
            FIRST=false
        else
            echo "," >> "$JSON_REPORT"
        fi
        
        echo "    {" >> "$JSON_REPORT"
        echo "      \"type\": \"$type\"," >> "$JSON_REPORT"
        echo "      \"source\": \"$source\"," >> "$JSON_REPORT"
        if [ -n "$name" ] && [ "$name" != "none" ]; then
            echo "      \"name\": \"$name\"," >> "$JSON_REPORT"
        fi
        echo "      \"value\": \"$value\"" >> "$JSON_REPORT"
        echo -n "    }" >> "$JSON_REPORT"
    done
    
    echo "" >> "$JSON_REPORT"
    echo "  ]" >> "$JSON_REPORT"
else
    echo "," >> "$JSON_REPORT"
    echo "  \"credentials\": []" >> "$JSON_REPORT"
fi

cat >> "$JSON_REPORT" << EOF
  ],
  "summary": {
    "hosts_discovered": $HOST_COUNT,
    "credentials_found": $CRED_COUNT
  }
}
EOF

# No cleanup needed (pure shell)

# Display results
echo "======================================================================"
echo "ENGAGEMENT SUMMARY"
echo "======================================================================"
echo "Targets: $HOST_COUNT host(s)"
echo "Duration: ${DURATION}s"
echo "Credentials Found: $CRED_COUNT"
echo ""
echo "Findings by Severity:"
if [ "$CRED_COUNT" -gt 0 ]; then
    echo "  High: 1 (Credentials Discovered)"
fi
echo "  Info: 1 (Network Hosts Discovered)"
echo ""
echo "======================================================================"
echo "[+] Markdown Report: $MD_REPORT"
echo "[+] JSON Report: $JSON_REPORT"
echo ""
echo "[*] All operations completed safely - no services disrupted"
echo "======================================================================"

