#!/bin/bash
# =============================================================================
# Traot — VPS Security Hardening (Hostinger Ubuntu)
# =============================================================================
# Run this ONCE after setup_vps.sh to lock down your server.
#
# Usage (on the VPS as root):
#   bash scripts/secure_vps.sh
#
# What this does:
#   1. System updates + auto security patches
#   2. Firewall (UFW) — only SSH + outbound allowed
#   3. SSH hardening — disable password auth, root password login
#   4. Fail2Ban — auto-ban brute force attackers
#   5. Disable unused services
#   6. File permissions for the bot
#   7. Swap file (prevents OOM kills)
# =============================================================================

set -e

echo ""
echo "=========================================="
echo "  Traot — VPS Security Hardening"
echo "=========================================="
echo ""

# ------------------------------------------------------------------
# Step 1: System updates + unattended upgrades
# ------------------------------------------------------------------
echo "[1/7] Updating system and enabling auto security patches..."
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq unattended-upgrades > /dev/null 2>&1

# Enable automatic security updates
cat > /etc/apt/apt.conf.d/20auto-upgrades << 'APTEOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
APTEOF
echo "      Auto security updates enabled."

# ------------------------------------------------------------------
# Step 2: Firewall (UFW)
# ------------------------------------------------------------------
echo "[2/7] Configuring firewall (UFW)..."
apt-get install -y -qq ufw > /dev/null 2>&1

# Reset to defaults
ufw --force reset > /dev/null 2>&1

# Default policies: deny incoming, allow outgoing
ufw default deny incoming > /dev/null 2>&1
ufw default allow outgoing > /dev/null 2>&1

# Allow SSH (port 22)
ufw allow 22/tcp > /dev/null 2>&1

# Allow Streamlit dashboard (optional — only if you want remote access)
# Uncomment the next line if you want dashboard access from outside:
# ufw allow 8501/tcp > /dev/null 2>&1

# Enable firewall
ufw --force enable > /dev/null 2>&1
echo "      Firewall active: SSH(22) allowed, all other incoming blocked."
echo "      Outbound allowed (needed for Binance/TwelveData API calls)."

# ------------------------------------------------------------------
# Step 3: SSH hardening
# ------------------------------------------------------------------
echo "[3/7] Hardening SSH..."
SSHD_CONFIG="/etc/ssh/sshd_config"

# Backup original config
cp "$SSHD_CONFIG" "${SSHD_CONFIG}.bak.$(date +%s)"

# Apply hardening settings (only if not already set)
apply_ssh_setting() {
    local key="$1"
    local value="$2"
    if grep -q "^${key}" "$SSHD_CONFIG"; then
        sed -i "s/^${key}.*/${key} ${value}/" "$SSHD_CONFIG"
    else
        echo "${key} ${value}" >> "$SSHD_CONFIG"
    fi
}

# Disable root password login (key-only access)
apply_ssh_setting "PermitRootLogin" "prohibit-password"

# Disable password authentication entirely (SSH keys only)
# IMPORTANT: Make sure your SSH key works BEFORE enabling this!
# Uncomment the next line ONLY after confirming SSH key login works:
# apply_ssh_setting "PasswordAuthentication" "no"

# Disable empty passwords
apply_ssh_setting "PermitEmptyPasswords" "no"

# Limit login attempts
apply_ssh_setting "MaxAuthTries" "3"

# Disable X11 forwarding (not needed)
apply_ssh_setting "X11Forwarding" "no"

# Disable TCP forwarding (not needed)
apply_ssh_setting "AllowTcpForwarding" "no"

# Session timeout: disconnect idle sessions after 10 minutes
apply_ssh_setting "ClientAliveInterval" "300"
apply_ssh_setting "ClientAliveCountMax" "2"

# Restart SSH to apply
systemctl restart sshd
echo "      SSH hardened: root password login disabled, max 3 attempts, idle timeout."

# ------------------------------------------------------------------
# Step 4: Fail2Ban (auto-ban brute force)
# ------------------------------------------------------------------
echo "[4/7] Installing Fail2Ban (brute force protection)..."
apt-get install -y -qq fail2ban > /dev/null 2>&1

# Create local config (survives package updates)
cat > /etc/fail2ban/jail.local << 'F2BEOF'
[DEFAULT]
# Ban for 1 hour after 5 failed attempts in 10 minutes
bantime = 3600
findtime = 600
maxretry = 5
# Use systemd backend
backend = systemd

[sshd]
enabled = true
port = ssh
filter = sshd
maxretry = 3
bantime = 7200
F2BEOF

systemctl enable fail2ban > /dev/null 2>&1
systemctl restart fail2ban
echo "      Fail2Ban active: 3 failed SSH attempts = 2 hour ban."

# ------------------------------------------------------------------
# Step 5: Disable unused services
# ------------------------------------------------------------------
echo "[5/7] Disabling unused services..."
# Disable services that aren't needed for a trading bot
for svc in cups avahi-daemon bluetooth; do
    if systemctl is-active --quiet "$svc" 2>/dev/null; then
        systemctl stop "$svc" > /dev/null 2>&1
        systemctl disable "$svc" > /dev/null 2>&1
        echo "      Disabled: $svc"
    fi
done
echo "      Done."

# ------------------------------------------------------------------
# Step 6: File permissions
# ------------------------------------------------------------------
echo "[6/7] Setting file permissions..."
APP_DIR="/root/ai-trade-bot"
if [ -d "$APP_DIR" ]; then
    # .env should only be readable by root
    chmod 600 "$APP_DIR/.env" 2>/dev/null || true

    # Scripts should be executable but not world-writable
    chmod 700 "$APP_DIR/scripts/"*.sh 2>/dev/null || true

    # Database should only be accessible by root
    chmod 600 "$APP_DIR/data/trading.db" 2>/dev/null || true

    # Models directory
    chmod 700 "$APP_DIR/models/" 2>/dev/null || true

    echo "      .env=600, scripts=700, db=600, models=700"
else
    echo "      App directory not found — run setup_vps.sh first."
fi

# ------------------------------------------------------------------
# Step 7: Swap file (prevents OOM kills during training)
# ------------------------------------------------------------------
echo "[7/7] Setting up swap space..."
if [ ! -f /swapfile ]; then
    # Create 2GB swap
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile > /dev/null 2>&1
    swapon /swapfile

    # Make persistent across reboots
    if ! grep -q "/swapfile" /etc/fstab; then
        echo "/swapfile none swap sw 0 0" >> /etc/fstab
    fi

    # Optimize swap behavior
    sysctl vm.swappiness=10 > /dev/null 2>&1
    echo "vm.swappiness=10" >> /etc/sysctl.conf

    echo "      2GB swap created (prevents OOM during model training)."
else
    echo "      Swap already exists."
fi

# ------------------------------------------------------------------
# Step 8: Anti-Mining / Anti-Malware Protection
# ------------------------------------------------------------------
echo "[8/10] Installing anti-mining protection..."

# Block known mining pool domains via /etc/hosts
MINING_BLOCKED=0
MINING_DOMAINS=(
    "pool.minergate.com"
    "xmr.pool.minergate.com"
    "stratum.antpool.com"
    "xmr-eu1.nanopool.org"
    "xmr-eu2.nanopool.org"
    "xmr-us-east1.nanopool.org"
    "xmr-us-west1.nanopool.org"
    "xmr-asia1.nanopool.org"
    "pool.supportxmr.com"
    "mine.c3pool.com"
    "xmr.2miners.com"
    "pool.hashvault.pro"
    "gulf.moneroocean.stream"
    "pool.minexmr.com"
    "monerohash.com"
    "stratum+tcp://xmr.pool.minergate.com"
    "coinhive.com"
    "authedmine.com"
    "crypto-loot.com"
    "coin-hive.com"
    "ppxxmr.com"
    "minergate.com"
    "miningrigrentals.com"
    "nicehash.com"
    "unmineable.com"
    "herominers.com"
    "kryptex.com"
    "pool.woolypooly.com"
)

for domain in "${MINING_DOMAINS[@]}"; do
    if ! grep -q "$domain" /etc/hosts 2>/dev/null; then
        echo "0.0.0.0 $domain" >> /etc/hosts
        MINING_BLOCKED=$((MINING_BLOCKED + 1))
    fi
done
echo "      Blocked ${MINING_BLOCKED} mining pool domains in /etc/hosts."

# Block common mining ports via UFW (outbound)
for port in 3333 4444 5555 7777 8888 9999 14433 14444 45560 45700; do
    ufw deny out "$port" > /dev/null 2>&1
done
echo "      Blocked mining stratum ports (3333,4444,5555,7777,8888,9999,etc) outbound."

# Kill any existing mining processes (common miner names)
MINERS_KILLED=0
for miner_name in xmrig xmr-stak minerd cpuminer cgminer bfgminer ethminer nbminer t-rex gminer lolminer; do
    if pgrep -x "$miner_name" > /dev/null 2>&1; then
        pkill -9 "$miner_name" 2>/dev/null
        MINERS_KILLED=$((MINERS_KILLED + 1))
        echo "      KILLED running miner: $miner_name"
    fi
done
if [ "$MINERS_KILLED" -eq 0 ]; then
    echo "      No mining processes found (clean system)."
fi

# Remove common miner binaries if they exist
for miner_path in /tmp/xmrig /tmp/.xmrig /var/tmp/xmrig /dev/shm/xmrig /tmp/kdevtmpfsi /tmp/kinsing; do
    if [ -f "$miner_path" ] || [ -d "$miner_path" ]; then
        rm -rf "$miner_path"
        echo "      REMOVED miner binary: $miner_path"
    fi
done

echo "      Anti-mining protection active."

# ------------------------------------------------------------------
# Step 9: Rootkit detection + integrity monitoring
# ------------------------------------------------------------------
echo "[9/10] Installing rootkit detection..."
apt-get install -y -qq rkhunter chkrootkit > /dev/null 2>&1

# Update rkhunter database
rkhunter --update > /dev/null 2>&1 || true
rkhunter --propupd > /dev/null 2>&1 || true
echo "      rkhunter + chkrootkit installed."
echo "      Run manually: rkhunter --check --sk"
echo "      Run manually: chkrootkit"

# ------------------------------------------------------------------
# Step 10: CPU monitoring cron job (detect mining)
# ------------------------------------------------------------------
echo "[10/10] Setting up CPU monitoring watchdog..."

cat > /root/mining_watchdog.sh << 'WATCHEOF'
#!/bin/bash
# Mining Watchdog — kills processes using >80% CPU for extended periods
# Runs every 5 minutes via cron

LOG="/var/log/mining_watchdog.log"
THRESHOLD=80
WHITELIST="python3|python|pip|apt|dpkg|journalctl|systemd"

# Find processes using more than threshold CPU (exclude whitelisted)
HIGH_CPU=$(ps aux --sort=-%cpu | awk -v threshold="$THRESHOLD" 'NR>1 && $3>threshold {print $2, $11, $3}' | grep -vE "$WHITELIST" | head -5)

if [ -n "$HIGH_CPU" ]; then
    echo "$(date): HIGH CPU detected:" >> "$LOG"
    echo "$HIGH_CPU" >> "$LOG"

    # Check if process name matches known miners
    echo "$HIGH_CPU" | while read pid cmd cpu; do
        cmd_lower=$(echo "$cmd" | tr '[:upper:]' '[:lower:]')
        case "$cmd_lower" in
            *xmrig*|*minerd*|*cpuminer*|*cgminer*|*kdevtmpfsi*|*kinsing*|*xmr*|*stratum*)
                echo "$(date): KILLING miner process: PID=$pid CMD=$cmd CPU=$cpu%" >> "$LOG"
                kill -9 "$pid" 2>/dev/null
                ;;
            *)
                echo "$(date): WARNING: Unknown high CPU process: PID=$pid CMD=$cmd CPU=$cpu% (not killed)" >> "$LOG"
                ;;
        esac
    done
fi
WATCHEOF

chmod 700 /root/mining_watchdog.sh

# Install cron job (every 5 minutes)
CRON_LINE="*/5 * * * * /root/mining_watchdog.sh"
if ! crontab -l 2>/dev/null | grep -q "mining_watchdog"; then
    (crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -
    echo "      CPU watchdog cron installed (checks every 5 minutes)."
else
    echo "      CPU watchdog cron already installed."
fi
echo "      Auto-kills known miners using >80% CPU."
echo "      Log: /var/log/mining_watchdog.log"

echo ""
echo "=========================================="
echo "  Security Hardening Complete!"
echo "=========================================="
echo ""
echo "  Summary:"
echo "  [DONE] Auto security updates enabled"
echo "  [DONE] Firewall: only SSH(22) open, all other incoming blocked"
echo "  [DONE] SSH: root password login disabled, max 3 attempts"
echo "  [DONE] Fail2Ban: 3 failed SSH = 2 hour IP ban"
echo "  [DONE] Unused services disabled"
echo "  [DONE] File permissions locked down (.env, db, models)"
echo "  [DONE] 2GB swap prevents out-of-memory crashes"
echo "  [DONE] Mining pool domains blocked (28+ domains)"
echo "  [DONE] Mining stratum ports blocked outbound"
echo "  [DONE] Rootkit detection installed (rkhunter + chkrootkit)"
echo "  [DONE] CPU watchdog: auto-kills miners every 5 minutes"
echo ""
echo "  IMPORTANT remaining steps:"
echo "  1. Test SSH login still works (open new terminal, try ssh root@IP)"
echo "  2. If SSH key login works, uncomment PasswordAuthentication=no"
echo "     in /etc/ssh/sshd_config for maximum security"
echo "  3. Consider changing SSH port (optional, edit ufw + sshd_config)"
echo ""
echo "  Security commands:"
echo "    fail2ban-client status sshd       # Check banned IPs"
echo "    ufw status verbose                # Firewall rules"
echo "    rkhunter --check --sk             # Rootkit scan"
echo "    chkrootkit                        # Second rootkit scan"
echo "    cat /var/log/mining_watchdog.log  # Mining detection log"
echo ""
