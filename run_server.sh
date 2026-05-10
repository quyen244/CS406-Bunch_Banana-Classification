#!/bin/bash

# --- CẤU HÌNH ĐƯỜNG DẪN ---
LOG_FILE="D:/Projects/Assignment/LLM-AI-Assistant-Projects/BananaClassification/logs/server_startup.log"
PROJECT_DIR="D:/Projects/Assignment/LLM-AI-Assistant-Projects/BananaClassification"
TUNNEL_NAME="server-nha-lam"

# Tạo thư mục log nếu chưa có
mkdir -p "D:/Projects/Assignment/LLM-AI-Assistant-Projects/BananaClassification/logs"

# Hàm ghi log có kèm thời gian
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Bắt đầu quá trình
log_message "=== BẮT ĐẦU QUÁ TRÌNH KHỞI ĐỘNG HỆ THỐNG ==="

# Chờ mạng ổn định
log_message "Đang chờ 10 giây để hệ thống ổn định mạng..."
sleep 10

# 1. Chạy Cloudflared Tunnel
log_message "Đang khởi chạy Cloudflared Tunnel: $TUNNEL_NAME..."
if cloudflared tunnel run "$TUNNEL_NAME" > /dev/null 2>&1 & then
    log_message "Lệnh khởi chạy Cloudflared đã được gửi vào background."
else
    log_message "LỖI: Không thể khởi chạy Cloudflared."
fi

# Chờ tunnel thiết lập kết nối
sleep 5

# 2. Chạy Docker Compose
log_message "Đang di chuyển đến thư mục dự án: $PROJECT_DIR"
if cd "$PROJECT_DIR"; then
    log_message "Đang chạy docker-compose up -d..."
    if docker-compose up -d >> "$LOG_FILE" 2>&1; then
        log_message "Docker Compose đã khởi chạy thành công các container."
    else
        log_message "LỖI: Docker Compose thất bại. Kiểm tra log chi tiết phía trên."
    fi
else
    log_message "LỖI: Không tìm thấy đường dẫn thư mục dự án."
fi

log_message "=== TOÀN BỘ QUY TRÌNH HOÀN TẤT ==="
echo "-------------------------------------------" >> "$LOG_FILE"