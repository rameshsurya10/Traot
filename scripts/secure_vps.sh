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
echo ""
echo "  IMPORTANT remaining steps:"
echo "  1. Test SSH login still works (open new terminal, try ssh root@IP)"
echo "  2. If SSH key login works, uncomment PasswordAuthentication=no"
echo "     in /etc/ssh/sshd_config for maximum security"
echo "  3. Consider changing SSH port (optional, edit ufw + sshd_config)"
echo ""
echo "  To check Fail2Ban status:"
echo "    fail2ban-client status sshd"
echo ""
echo "  To check firewall status:"
echo "    ufw status verbose"
echo ""
