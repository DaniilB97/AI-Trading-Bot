#!/bin/bash

# Whale Signal Orchestrator - Deployment Script for Vultr
# Автоматическое развертывание системы на Ubuntu 22.04

set -e

echo "🐋 Whale Signal Orchestrator - Vultr Deployment"
echo "==============================================="

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функция логирования
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Проверка root прав
if [[ $EUID -eq 0 ]]; then
   error "Этот скрипт не должен запускаться от root"
   exit 1
fi

# Создание пользователя для приложения
setup_user() {
    log "Настройка пользователя whale..."
    
    if ! id "whale" &>/dev/null; then
        sudo useradd -m -s /bin/bash whale
        sudo usermod -aG sudo whale
        log "Пользователь whale создан"
    else
        log "Пользователь whale уже существует"
    fi
}

# Обновление системы
update_system() {
    log "Обновление системы..."
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y curl wget git htop nano ufw python3-pip python3-venv nginx certbot python3-certbot-nginx
}

# Настройка firewall
setup_firewall() {
    log "Настройка firewall..."
    sudo ufw default deny incoming
    sudo ufw default allow outgoing
    sudo ufw allow ssh
    sudo ufw allow 80
    sudo ufw allow 443
    sudo ufw allow 8000  # Для разработки
    sudo ufw --force enable
}

# Установка Docker (опционально)
install_docker() {
    log "Установка Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker whale
    sudo systemctl enable docker
    sudo systemctl start docker
    rm get-docker.sh
}

# Клонирование проекта
clone_project() {
    log "Клонирование проекта..."
    
    APP_DIR="/home/whale/whale-signal-orchestrator"
    
    if [ -d "$APP_DIR" ]; then
        warn "Директория проекта уже существует, обновляем..."
        cd $APP_DIR
        git pull
    else
        sudo -u whale git clone https://github.com/your-repo/whale-signal-orchestrator.git $APP_DIR
        sudo chown -R whale:whale $APP_DIR
    fi
    
    cd $APP_DIR
}

# Настройка Python окружения
setup_python() {
    log "Настройка Python окружения..."
    
    sudo -u whale python3 -m venv venv
    sudo -u whale bash -c "source venv/bin/activate && pip install --upgrade pip"
    sudo -u whale bash -c "source venv/bin/activate && pip install -r requirements.txt"
}

# Создание конфигурационных файлов
setup_config() {
    log "Создание конфигурационных файлов..."
    
    # Создаем .env файл
    if [ ! -f .env ]; then
        cat > .env << EOF
# Ethereum API Keys
ETHERSCAN_API_KEY=your_etherscan_api_key_here
INFURA_API_KEY=your_infura_api_key_here
ALCHEMY_API_KEY=your_alchemy_api_key_here

# Solana RPC
QUICKNODE_SOLANA_URL=your_quicknode_solana_url_here

# Trading (Optional)
PRIVATE_KEY=your_private_key_here

# Additional APIs
SOLSCAN_API_KEY=your_solscan_api_key_here
EOF
        warn "⚠️  Не забудьте настроить API ключи в .env файле!"
}

# Запуск с проверкой аргументов
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Использование: $0 [домен]"
    echo "Пример: $0 whale-signals.yourdomain.com"
    exit 0
fi

main $1Отредактируйте файл .env с вашими API ключами"
    fi
    
    # Создаем необходимые директории
    sudo -u whale mkdir -p config data/historical logs data/paper_trading
    
    # Создаем базовые конфиг файлы
    if [ ! -f config/dex_wallets.json ]; then
        cat > config/dex_wallets.json << EOF
[
  {
    "address": "J29AYczWMaUY61cHmhdFdhZnpk5mATmqN2GRCddFnHKi",
    "name": "Test Trader #1",
    "category": "meme_trader",
    "track_since": "$(date -Iseconds)",
    "notes": "Test wallet for development",
    "estimated_win_rate": 0.75
  }
]
EOF
    fi
    
    if [ ! -f config/wallets.json ]; then
        echo "[]" > config/wallets.json
    fi
}

# Создание systemd сервисов
setup_systemd() {
    log "Создание systemd сервисов..."
    
    # Web Server Service
    sudo tee /etc/systemd/system/whale-web.service > /dev/null << EOF
[Unit]
Description=Whale Signal Web Server
After=network.target

[Service]
Type=simple
User=whale
WorkingDirectory=/home/whale/whale-signal-orchestrator
Environment=PATH=/home/whale/whale-signal-orchestrator/venv/bin
ExecStart=/home/whale/whale-signal-orchestrator/venv/bin/python web_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # CLI Orchestrator Service (опционально)
    sudo tee /etc/systemd/system/whale-cli.service > /dev/null << EOF
[Unit]
Description=Whale Signal CLI Orchestrator
After=network.target

[Service]
Type=simple
User=whale
WorkingDirectory=/home/whale/whale-signal-orchestrator
Environment=PATH=/home/whale/whale-signal-orchestrator/venv/bin
ExecStart=/home/whale/whale-signal-orchestrator/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable whale-web.service
}

# Настройка Nginx
setup_nginx() {
    log "Настройка Nginx..."
    
    DOMAIN=${1:-"your-domain.com"}
    
    sudo tee /etc/nginx/sites-available/whale-signal << EOF
server {
    listen 80;
    server_name $DOMAIN;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

    sudo ln -sf /etc/nginx/sites-available/whale-signal /etc/nginx/sites-enabled/
    sudo rm -f /etc/nginx/sites-enabled/default
    sudo nginx -t && sudo systemctl reload nginx
}

# Создание скрипта мониторинга
setup_monitoring() {
    log "Создание скрипта мониторинга..."
    
    cat > monitor.sh << 'EOF'
#!/bin/bash

# Мониторинг сервисов Whale Signal

check_service() {
    if systemctl is-active --quiet $1; then
        echo "✅ $1 работает"
    else
        echo "❌ $1 не работает"
        sudo systemctl restart $1
        echo "🔄 $1 перезапущен"
    fi
}

echo "🐋 Whale Signal Monitoring - $(date)"
echo "=================================="

check_service whale-web.service
check_service nginx

# Проверка использования ресурсов
echo ""
echo "📊 Использование ресурсов:"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')"
echo "RAM: $(free -h | awk '/^Mem/ {print $3 "/" $2}')"
echo "Disk: $(df -h / | awk 'NR==2 {print $5}')"

# Проверка логов
echo ""
echo "📝 Последние логи:"
tail -n 5 /home/whale/whale-signal-orchestrator/logs/web_server_$(date +%Y%m%d).log 2>/dev/null || echo "Нет логов"
EOF

    chmod +x monitor.sh
}

# SSL сертификат
setup_ssl() {
    local domain=$1
    if [ -n "$domain" ] && [ "$domain" != "your-domain.com" ]; then
        log "Получение SSL сертификата для $domain..."
        sudo certbot --nginx -d $domain --non-interactive --agree-tos --email admin@$domain
    else
        warn "Пропуск настройки SSL (укажите домен: ./deploy.sh your-domain.com)"
    fi
}

# Главная функция
main() {
    echo "🚀 Начинаем развертывание..."
    
    setup_user
    update_system
    setup_firewall
    install_docker
    clone_project
    setup_python
    setup_config
    setup_systemd
    setup_nginx $1
    setup_monitoring
    setup_ssl $1
    
    log "Запуск сервисов..."
    sudo systemctl start whale-web.service
    sudo systemctl start nginx
    
    echo ""
    echo "🎉 Развертывание завершено!"
    echo "=================================="
    echo "• Web интерфейс: http://$(curl -s ifconfig.me):8000"
    echo "• Логи: journalctl -u whale-web.service -f"
    echo "• Статус: systemctl status whale-web.service"
    echo "• Мониторинг: ./monitor.sh"
    echo ""
    echo "📝 Следующие шаги:"
    echo "1. Отредактируйте .env файл с вашими API ключами"
    echo "2. Добавьте кошельки в config/dex_wallets.json"
    echo "3. Перезапустите сервис: sudo systemctl restart whale-web.service"
    echo ""
    warn "⚠️