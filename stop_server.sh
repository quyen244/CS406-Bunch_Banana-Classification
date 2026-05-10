#!/bin/bash

LOG_FILE="D:/Projects/Assignment/LLM-AI-Assistant-Projects/BananaClassification/logs/server_stop.log"
PROJECT_DIR="D:/Projects/Assignment/LLM-AI-Assistant-Projects/BananaClassification"

# Tạo thư mục log nếu chưa có
mkdir -p "D:/Projects/Assignment/LLM-AI-Assistant-Projects/BananaClassification/logs"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- ĐANG TẮT HỆ THỐNG ---" >> "$LOG_FILE"

# 1. Tắt Docker
echo "Đang dừng Docker containers..."
cd "$PROJECT_DIR"
docker-compose down >> "$LOG_FILE" 2>&1

# 2. Tắt Cloudflared
echo "Đang dừng Cloudflared Tunnel..."
taskkill //F //IM cloudflared.exe >> "$LOG_FILE" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- ĐÃ TẮT TOÀN BỘ ---" >> "$LOG_FILE"
echo "Done!"