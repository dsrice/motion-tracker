#!/bin/bash

# 依存関係のインストール
echo "mediapipeと必要なライブラリをインストールしています..."
pip install mediapipe opencv-python numpy

# スクリプトに実行権限を付与
echo "スクリプトに実行権限を付与します..."
chmod +x joint_tracker.py

# 必要なディレクトリを作成
echo "必要なディレクトリを作成しています..."
mkdir -p tmp output

echo "セットアップが完了しました！"
echo ""
echo "使用方法:"
echo "1. 処理したいMOVファイルを tmp ディレクトリに配置します"
echo "2. 以下のコマンドで実行します:"
echo "   ./joint_tracker.py tmp/動画ファイル名.mov -d"
echo ""
echo "コマンドラインオプション:"
echo "  -o, --output     出力ビデオファイルのパス"
echo "  -d, --display    処理中のビデオをウィンドウに表示する"
echo "  -t, --trace      軌跡を表示するフレーム数（デフォルト: 30）"
echo "  --help           ヘルプメッセージを表示"
