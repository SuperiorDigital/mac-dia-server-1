#!/bin/bash

# 项目根目录
PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
VENV_DIR="$PROJECT_DIR/.venv"
LOG_FILE="$PROJECT_DIR/server.log"
PID_FILE="$PROJECT_DIR/server.pid"

start_server() {
    source "$VENV_DIR/bin/activate"
    if [ "$1" == "-d" ]; then
        # 后台启动
        nohup uv run start.py > "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        echo "服务已在后台启动，日志输出到 $LOG_FILE，PID: $(cat $PID_FILE)"
    else
        # 前台启动
        uv run start.py
    fi
}

stop_server() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 $PID > /dev/null 2>&1; then
            kill $PID
            echo "服务已停止 (PID: $PID)"
        else
            echo "未找到正在运行的服务进程 (PID: $PID)，可能已退出。"
        fi
        rm -f "$PID_FILE"
    else
        echo "未找到PID文件，服务可能未启动或已手动关闭。"
    fi
}

case "$1" in
    start)
        start_server $2
        ;;
    stop)
        stop_server
        ;;
    *)
        echo "用法: $0 {start [-d]|stop}"
        echo "  start    前台启动服务"
        echo "  start -d 后台启动服务，日志写入 $LOG_FILE"
        echo "  stop     停止服务"
        exit 1
        ;;
esac
