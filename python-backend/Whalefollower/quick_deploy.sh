#!/bin/bash

# Whale Signal Orchestrator - Quick Deploy Script
# Быстрое развертывание на любом сервере с Docker

set -e

echo "🐋 Whale Signal Orchestrator - Quick Deploy"
echo "==========================================="

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Проверка Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        error "Docker не установлен!"
        echo "Установите Docker: curl -fsSL https://get.docker.com | sh"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose не установлен!"
        echo "Установите Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    log "Docker готов к использованию"
}

# Создание .env файла
create_env() {
    if [ ! -f .env ]; then
        log "Создание .env файла..."
        cat > .env << 'EOF'
# =================================
# Whale Signal Orchestrator Config
# =================================

# Ethereum API Keys (обязательно)
ETHERSCAN_API_KEY=your_etherscan_api_key_here
INFURA_API_KEY=your_infura_api_key_here
ALCHEMY_API_KEY=your_alchemy_api_key_here

# Solana RPC (обязательно для DEX мониторинга)
QUICKNODE_SOLANA_URL=your_quicknode_solana_url_here

# Trading (опционально, для real trading)
PRIVATE_KEY=your_private_key_here

# Additional APIs (опционально)
SOLSCAN_API_KEY=your_solscan_api_key_here

# Web Server
WEB_HOST=0.0.0.0
WEB_PORT=8000

# Logging
LOG_LEVEL=INFO
EOF
        warn "⚠️  Отредактируйте .env файл с вашими API ключами!"
        warn "⚠️  Минимум нужны: ETHERSCAN_API_KEY и QUICKNODE_SOLANA_URL"
    else
        log ".env файл уже существует"
    fi
}

# Создание директорий
create_dirs() {
    log "Создание необходимых директорий..."
    mkdir -p {config,data/{historical,paper_trading},logs,ssl}
    
    # Создание базовых конфигов
    if [ ! -f config/dex_wallets.json ]; then
        cat > config/dex_wallets.json << 'EOF'
[
  {
    "address": "J29AYczWMaUY61cHmhdFdhZnpk5mATmqN2GRCddFnHKi",
    "name": "Test Meme Trader",
    "category": "meme_trader",
    "track_since": "2025-01-27T00:00:00",
    "notes": "Тестовый кошелек для демонстрации",
    "estimated_win_rate": 0.75,
    "specialization": ["meme_coins"]
  }
]
EOF
        log "Создан базовый config/dex_wallets.json"
    fi
    
    if [ ! -f config/wallets.json ]; then
        echo "[]" > config/wallets.json
        log "Создан пустой config/wallets.json"
    fi
}

# Создание SSL сертификатов (self-signed для разработки)
create_ssl() {
    if [ ! -f ssl/cert.pem ]; then
        log "Создание самоподписанного SSL сертификата..."
        openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes \
            -subj "/C=US/ST=CA/L=SF/O=Whale/CN=localhost"
        log "SSL сертификат создан (самоподписанный)"
    fi
}

# Сборка и запуск
deploy() {
    log "Сборка Docker образов..."
    docker-compose build
    
    log "Запуск сервисов..."
    docker-compose up -d
    
    # Ожидание запуска
    log "Ожидание запуска сервисов..."
    sleep 10
    
    # Проверка статуса
    if docker-compose ps | grep -q "Up"; then
        log "✅ Сервисы запущены успешно!"
    else
        error "❌ Ошибка запуска сервисов"
        docker-compose logs
        exit 1
    fi
}

# Отображение информации о доступе
show_info() {
    local SERVER_IP=$(curl -s ifconfig.me 2>/dev/null || echo "localhost")
    
    echo ""
    echo "🎉 Развертывание завершено!"
    echo "=========================="
    echo ""
    echo "📍 Доступ к приложению:"
    echo "   HTTP:  http://${SERVER_IP}"
    echo "   HTTPS: https://${SERVER_IP} (самоподписанный сертификат)"
    echo "   Порт:  8000 (прямой доступ)"
    echo ""
    echo "📊 Мониторинг:"
    echo "   Grafana: http://${SERVER_IP}:3000 (admin/admin)"
    echo "   Логи:    docker-compose logs -f whale-web"
    echo "   Статус:  docker-compose ps"
    echo ""
    echo "🔧 Управление:"
    echo "   Остановка:     docker-compose stop"
    echo "   Перезапуск:    docker-compose restart"
    echo "   Обновление:    git pull && docker-compose up -d --build"
    echo "   Удаление:      docker-compose down -v"
    echo ""
    echo "⚙️  Конфигурация:"
    echo "   API ключи:     .env"
    echo "   ETH кошельки:  config/wallets.json"
    echo "   SOL кошельки:  config/dex_wallets.json"
    echo "   Логи:          logs/"
    echo "   Данные:        data/"
    echo ""
    
    if [ -f .env ] && grep -q "your_.*_here" .env; then
        warn "⚠️  ВНИМАНИЕ: Не забудьте настроить API ключи в .env файле!"
        warn "⚠️  После настройки выполните: docker-compose restart whale-web"
    fi
}

# Проверка обновлений
check_updates() {
    log "Проверка обновлений..."
    if git status &>/dev/null; then
        if git fetch && [ "$(git rev-list HEAD...origin/main --count)" != "0" ]; then
            warn "Доступны обновления! Выполните: git pull && docker-compose up -d --build"
        else
            log "Используется последняя версия"
        fi
    fi
}

# Главная функция
main() {
    case "$1" in
        "stop")
            log "Остановка сервисов..."
            docker-compose stop
            ;;
        "restart")
            log "Перезапуск сервисов..."
            docker-compose restart
            ;;
        "logs")
            docker-compose logs -f whale-web
            ;;
        "status")
            docker-compose ps
            ;;
        "update")
            log "Обновление приложения..."
            git pull
            docker-compose down
            docker-compose up -d --build
            show_info
            ;;
        "clean")
            warn "Удаление всех данных..."
            read -p "Вы уверены? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                docker-compose down -v
                docker system prune -f
                log "Очистка завершена"
            fi
            ;;
        "backup")
            log "Создание резервной копии..."
            tar -czf "whale-backup-$(date +%Y%m%d-%H%M%S).tar.gz" config/ data/ .env
            log "Резервная копия создана"
            ;;
        "--help"|"-h"|"help")
            echo "Использование: $0 [команда]"
            echo ""
            echo "Команды:"
            echo "  (пусто)   Полное развертывание"
            echo "  stop      Остановка сервисов"
            echo "  restart   Перезапуск сервисов"
            echo "  logs      Просмотр логов"
            echo "  status    Статус сервисов"
            echo "  update    Обновление приложения"
            echo "  clean     Полная очистка"
            echo "  backup    Создание резервной копии"
            echo "  help      Эта справка"
            ;;
        *)
            # Полное развертывание
            check_docker
            create_env
            create_dirs
            create_ssl
            deploy
            check_updates
            show_info
            ;;
    esac
}

main "$@"