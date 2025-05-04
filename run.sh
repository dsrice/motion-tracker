#!/bin/bash

# 色の定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}関節トラッキングシステム - Docker環境${NC}"
echo "========================================"

# 必要なディレクトリを作成
mkdir -p tmp output

# 入力ビデオの確認
VIDEO_COUNT=$(ls -1 tmp/*.{mp4,mov,avi} 2>/dev/null | wc -l)

if [ "$VIDEO_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}警告: tmpディレクトリにビデオファイルがありません。${NC}"
    echo "処理するビデオファイル（.mp4, .mov, .avi）をtmpディレクトリに配置してください。"
    exit 1
fi

echo -e "${GREEN}処理可能なビデオファイル:${NC}"
ls -1 tmp/*.{mp4,mov,avi} 2>/dev/null | nl

echo ""

# GPU対応チェック
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPUが検出されました。GPU版を使用します。${NC}"
    USE_GPU=1
else
    echo -e "${YELLOW}NVIDIA GPUが検出されませんでした。CPU版を使用します。${NC}"
    USE_GPU=0
fi

echo ""
echo -e "${BLUE}実行モードを選択してください:${NC}"
echo "1) バッチ処理（すべてのビデオを自動処理）"
echo "2) 単一ビデオ処理（ビデオを選択して処理）"
echo "3) 対話モード（Dockerコンテナにアクセスして手動実行）"
read -p "選択 (1-3): " CHOICE

case $CHOICE in
    1)
        # バッチ処理
        echo -e "${GREEN}すべてのビデオを処理します...${NC}"
        SERVICE="joint-tracker"
        if [ "$USE_GPU" -eq 1 ]; then
            SERVICE="joint-tracker-gpu"
        fi
        
        docker-compose run --rm $SERVICE python3 joint_tracker.py --batch
        ;;
    2)
        # 単一ビデオ処理
        echo -e "${GREEN}処理するビデオを選択してください:${NC}"
        ls -1 tmp/*.{mp4,mov,avi} 2>/dev/null | nl
        read -p "ビデオ番号: " VIDEO_NUM
        
        VIDEO_FILE=$(ls -1 tmp/*.{mp4,mov,avi} 2>/dev/null | sed -n "${VIDEO_NUM}p")
        if [ -z "$VIDEO_FILE" ]; then
            echo -e "${RED}エラー: 無効なビデオ番号です。${NC}"
            exit 1
        fi
        
        echo -e "${GREEN}ディスプレイ表示を有効にしますか? (y/n)${NC}"
        read -p "選択: " DISPLAY_CHOICE
        DISPLAY_OPT=""
        if [[ "$DISPLAY_CHOICE" == "y" || "$DISPLAY_CHOICE" == "Y" ]]; then
            DISPLAY_OPT="-d"
            # X11接続を許可
            xhost +local:docker > /dev/null 2>&1
        fi
        
        SERVICE="joint-tracker"
        if [ "$USE_GPU" -eq 1 ]; then
            SERVICE="joint-tracker-gpu"
        fi
        
        docker-compose run --rm $SERVICE python3 joint_tracker.py "$VIDEO_FILE" $DISPLAY_OPT
        ;;
    3)
        # 対話モード
        echo -e "${GREEN}Dockerコンテナを起動します...${NC}"
        SERVICE="joint-tracker"
        if [ "$USE_GPU" -eq 1 ]; then
            SERVICE="joint-tracker-gpu"
        fi
        
        echo -e "${YELLOW}コンテナ内でコマンドを実行するには:${NC}"
        echo "python3 joint_tracker.py tmp/ビデオファイル.mp4 -d"
        echo ""
        echo -e "${YELLOW}コンテナを終了するには:${NC}"
        echo "exit"
        echo ""
        
        # X11接続を許可
        xhost +local:docker > /dev/null 2>&1
        
        docker-compose run --rm $SERVICE
        ;;
    *)
        echo -e "${RED}エラー: 無効な選択です。${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}処理が完了しました！${NC}"
echo "結果は 'output' ディレクトリに保存されています。"
