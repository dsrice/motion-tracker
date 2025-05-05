#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
関節トラッキングスクリプト
MOVファイルなどのビデオから人体の関節を検出し、その動きをトレースします。
"""

import cv2
import numpy as np
import os
import argparse
import time
from datetime import datetime
from collections import deque
import mediapipe as mp


class PoseTracker:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, trace_length=30, 
                 body_part='all', show_traces=True):
        """
        関節トラッキングクラスの初期化

        Parameters:
        -----------
        min_detection_confidence : float
            検出の最小信頼度
        min_tracking_confidence : float
            トラッキングの最小信頼度
        trace_length : int
            軌跡を記録するフレーム数
        body_part : str
            トラッキングする身体部位 ('upper', 'lower', 'all')
        show_traces : bool
            関節の軌跡を表示するかどうか
        """
        # MediaPipeのポーズ検出モジュールを初期化
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)

        # 関節ごとの軌跡を保存するための辞書
        self.trace_length = trace_length
        self.joint_traces = {}
        self.body_part = body_part  # トラッキングする身体部位
        self.show_traces = show_traces  # 軌跡表示フラグ

        # 上半身の関節
        self.upper_body_joints = {
            '左手首': self.mp_pose.PoseLandmark.LEFT_WRIST,
            '右手首': self.mp_pose.PoseLandmark.RIGHT_WRIST,
            '左肘': self.mp_pose.PoseLandmark.LEFT_ELBOW,
            '右肘': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            '左肩': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            '右肩': self.mp_pose.PoseLandmark.RIGHT_SHOULDER
        }

        # 下半身の関節
        self.lower_body_joints = {
            '左膝': self.mp_pose.PoseLandmark.LEFT_KNEE,
            '右膝': self.mp_pose.PoseLandmark.RIGHT_KNEE,
            '左足首': self.mp_pose.PoseLandmark.LEFT_ANKLE,
            '右足首': self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            '左腰': self.mp_pose.PoseLandmark.LEFT_HIP,
            '右腰': self.mp_pose.PoseLandmark.RIGHT_HIP
        }
        
        # 選択されたトラッキング部位に基づいて主要な関節を設定
        if self.body_part == 'upper':
            self.key_joints = self.upper_body_joints
        elif self.body_part == 'lower':
            self.key_joints = self.lower_body_joints
        else:  # 'all'
            self.key_joints = {**self.upper_body_joints, **self.lower_body_joints}

        # 各関節の軌跡の色
        self.joint_colors = {
            '左手首': (255, 0, 0),      # 赤
            '右手首': (0, 0, 255),      # 青
            '左肘': (255, 0, 255),      # マゼンタ
            '右肘': (0, 255, 255),      # シアン
            '左肩': (255, 128, 0),      # オレンジ
            '右肩': (0, 128, 255),      # ライトブルー
            '左膝': (255, 255, 0),      # 黄
            '右膝': (0, 255, 0),        # 緑
            '左足首': (128, 0, 128),    # 紫
            '右足首': (0, 128, 128),    # ティール
            '左腰': (128, 64, 0),       # 茶色
            '右腰': (0, 64, 128)        # 紺
        }

        # 各関節の軌跡を初期化
        for joint_name in self.key_joints.keys():
            self.joint_traces[joint_name] = deque(maxlen=self.trace_length)

    def detect_pose(self, frame):
        """
        フレーム内の姿勢を検出する

        Parameters:
        -----------
        frame : ndarray
            分析するビデオフレーム

        Returns:
        --------
        results : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
            検出された姿勢のランドマーク
        """
        # 画像をRGBに変換（MediaPipeはRGBを使用）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 姿勢を検出（画像の書き込みを無効化）
        results = self.pose.process(frame_rgb)

        return results

    def draw_pose(self, frame, results):
        """
        検出された姿勢を描画する

        Parameters:
        -----------
        frame : ndarray
            描画するフレーム
        results : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
            検出された姿勢のランドマーク

        Returns:
        --------
        frame : ndarray
            姿勢が描画されたフレーム
        pose_detected : bool
            姿勢が検出されたかどうか
        """
        frame_copy = frame.copy()
        pose_detected = False

        if results.pose_landmarks:
            pose_detected = True

            # 姿勢のランドマークを描画
            self.mp_drawing.draw_landmarks(
                frame_copy,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

            # 主要な関節の位置を保存
            h, w, _ = frame.shape
            for joint_name, joint_idx in self.key_joints.items():
                landmark = results.pose_landmarks.landmark[joint_idx]
                # 正規化された座標をピクセル座標に変換
                x, y = int(landmark.x * w), int(landmark.y * h)
                self.joint_traces[joint_name].append((x, y))

            # 関節の軌跡を描画
            frame_copy = self.draw_traces(frame_copy)

        return frame_copy, pose_detected

    def draw_traces(self, frame):
        """
        関節の軌跡を描画する

        Parameters:
        -----------
        frame : ndarray
            描画するフレーム

        Returns:
        --------
        frame : ndarray
            軌跡が描画されたフレーム
        """
        # 軌跡表示が無効の場合は何もせずに返す
        if not self.show_traces:
            return frame
            
        for joint_name, trace in self.joint_traces.items():
            color = self.joint_colors[joint_name]
            # トレース内の点を接続して軌跡を描画
            for i in range(1, len(trace)):
                if trace[i - 1] is None or trace[i] is None:
                    continue
                # 線の太さは軌跡の終端に向かって太くする
                thickness = int(np.sqrt(self.trace_length / float(i + 1)) * 2.0)
                cv2.line(frame, trace[i - 1], trace[i], color, thickness)

            # 軌跡の最新の位置に関節名を表示
            if len(trace) > 0 and trace[-1] is not None:
                cv2.putText(frame, joint_name,
                            (trace[-1][0] + 10, trace[-1][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return frame

    def reset_traces(self):
        """
        すべての関節の軌跡をリセットする
        """
        for joint_name in self.key_joints.keys():
            self.joint_traces[joint_name].clear()


def process_video(input_path, output_path=None, display_video=False, trace_length=30, min_detection_confidence=0.7,
                  min_tracking_confidence=0.5, body_part='all', show_traces=True):
    """
    ビデオを処理し、人体の関節をトラッキングする

    Parameters:
    -----------
    input_path : str
        入力ビデオのパス
    output_path : str, optional
        出力ビデオのパス
    display_video : bool, optional
        処理中のビデオをウィンドウに表示するか
    trace_length : int, optional
        軌跡を記録するフレーム数
    min_detection_confidence : float, optional
        検出の最小信頼度
    min_tracking_confidence : float, optional
        トラッキングの最小信頼度
    body_part : str, optional
        トラッキングする身体部位 ('upper', 'lower', 'all')
    show_traces : bool, optional
        関節の軌跡を表示するかどうか
    """
    # ビデオキャプチャを開く
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"エラー: ビデオ '{input_path}' を開けませんでした")
        return

    # ビデオのプロパティを取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"ビデオサイズ: {width}x{height}, FPS: {fps:.2f}, フレーム数: {frame_count}")

    # 出力ビデオを設定
    writer = None
    if output_path:
        # 出力ディレクトリが存在しない場合は作成
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 関節トラッカーを初期化
    tracker = PoseTracker(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        trace_length=trace_length,
        body_part=body_part,
        show_traces=show_traces
    )

    frame_idx = 0
    pose_frames = 0

    # 処理時間の測定
    start_time = time.time()
    last_log_time = start_time

    # もしウィンドウ表示が有効なら、ウィンドウを作成
    if display_video:
        cv2.namedWindow("関節トラッキング", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("関節トラッキング", 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # 姿勢検出
        results = tracker.detect_pose(frame)

        # 姿勢と軌跡を描画
        result_frame, pose_detected = tracker.draw_pose(frame, results)

        if pose_detected:
            pose_frames += 1

            # 現在の時刻を表示
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(result_frame, f"姿勢検出: {timestamp}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 身体部位と軌跡表示の状態を表示
            body_part_text = {
                'all': '全身',
                'upper': '上半身',
                'lower': '下半身'
            }.get(body_part, '全身')
            
            trace_text = '軌跡表示: 有効' if show_traces else '軌跡表示: 無効'
            cv2.putText(result_frame, f"トラッキング部位: {body_part_text}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            cv2.putText(result_frame, trace_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        
        # 進捗表示
        progress = (frame_idx / frame_count) * 100
        cv2.putText(result_frame, f"処理中: {progress:.1f}%", (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 出力ビデオに書き込み
        if writer:
            writer.write(result_frame)

        # ウィンドウに表示
        if display_video:
            cv2.imshow("関節トラッキング", result_frame)
            # ESCキーで処理を中断
            if cv2.waitKey(1) == 27:
                break

        # 進捗ログを1秒ごとまたは1%ごとに表示
        current_time = time.time()
        if current_time - last_log_time >= 1.0 or frame_idx % max(1, frame_count // 100) == 0:
            elapsed_time = current_time - start_time
            fps_processed = frame_idx / elapsed_time if elapsed_time > 0 else 0
            estimated_total = elapsed_time / (frame_idx / frame_count) if frame_idx > 0 else 0
            remaining_time = max(0, estimated_total - elapsed_time)

            print(f"処理中: {progress:.1f}% 完了 ({fps_processed:.1f} fps, 残り約{remaining_time:.1f}秒)", end='\r')
            last_log_time = current_time

    # リソースの解放
    cap.release()
    if writer:
        writer.release()
    if display_video:
        cv2.destroyAllWindows()

    # 結果を表示
    total_time = time.time() - start_time
    pose_percentage = (pose_frames / frame_idx) * 100 if frame_idx > 0 else 0
    fps_processed = frame_idx / total_time if total_time > 0 else 0

    print(f"\n処理完了: {frame_idx}フレーム処理、{pose_frames}フレームで姿勢検出 ({pose_percentage:.2f}%)")
    print(f"処理時間: {total_time:.2f}秒（{fps_processed:.2f} fps）")

    if output_path and os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"出力ビデオを保存しました: {output_path} ({file_size_mb:.2f} MB)")

    return pose_frames, frame_idx


def main():
    """
    メイン関数 - コマンドライン引数を解析して処理を実行
    """
    parser = argparse.ArgumentParser(description="MOVファイルから関節の動きをトレースするツール")

    parser.add_argument("input", help="入力ビデオファイルのパス")
    parser.add_argument("-o", "--output", help="出力ビデオファイルのパス（指定しない場合は出力なし）")
    parser.add_argument("-d", "--display", action="store_true", help="処理中のビデオをウィンドウに表示する")
    parser.add_argument("-t", "--trace", type=int, default=30, help="関節の軌跡を表示するフレーム数（デフォルト: 30）")
    parser.add_argument("--detection-confidence", type=float, default=0.7, help="検出の最小信頼度（デフォルト: 0.7）")
    parser.add_argument("--tracking-confidence", type=float, default=0.5,
                        help="トラッキングの最小信頼度（デフォルト: 0.5）")
    parser.add_argument("-p", "--body-part", choices=['upper', 'lower', 'all'], default='all',
                        help="トラッキングする身体部位 (upper=上半身, lower=下半身, all=全身) （デフォルト: all）")
    parser.add_argument("--no-traces", action="store_true", help="関節の軌跡を表示しない")

    args = parser.parse_args()

    # 入力ファイルの存在を確認
    if not os.path.exists(args.input):
        print(f"エラー: 入力ファイル '{args.input}' が見つかりません")
        return

    # もし出力パスが指定されていない場合、入力ファイル名をベースに出力パスを作成
    if not args.output:
        # 入力ファイルの名前とパスを取得
        input_dir = os.path.dirname(args.input)
        input_filename = os.path.basename(args.input)
        base_name, ext = os.path.splitext(input_filename)

        # 出力ディレクトリを確認
        output_dir = os.path.join(input_dir, "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 出力ファイル名を生成
        args.output = os.path.join(output_dir, f"{base_name}_traced.mp4")

    print(f"入力ビデオ: {args.input}")
    print(f"出力ビデオ: {args.output}")
    print(f"トレース長さ: {args.trace}フレーム")
    print(f"検出信頼度: {args.detection_confidence}")
    print(f"トラッキング信頼度: {args.tracking_confidence}")
    print(f"ウィンドウ表示: {'有効' if args.display else '無効'}")
    print(f"トラッキング部位: {args.body_part}")
    print(f"軌跡表示: {'無効' if args.no_traces else '有効'}")
    
    # ビデオ処理を実行
    process_video(
        input_path=args.input,
        output_path=args.output,
        display_video=args.display,
        trace_length=args.trace,
        min_detection_confidence=args.detection_confidence,
        min_tracking_confidence=args.tracking_confidence,
        body_part=args.body_part,
        show_traces=not args.no_traces
    )


if __name__ == "__main__":
    main()

