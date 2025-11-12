#!/usr/bin/env python3
"""
plot_winrate.py

指定したゲーム数（スライディングウィンドウ）または累積での勝率推移をプロットして保存します。

使い方例:
  python BlackJack/plot_winrate.py --log BlackJack/play_log.csv --window 100 --type sliding --output winrate.png
  python BlackJack/plot_winrate.py --log selected_log.csv --type cumulative --output cumulative.png

デフォルトでは `result` カラムが 'win' の行を勝ちとして扱います。
"""
import argparse
import csv
import os
import sys
from typing import List

import numpy as np
import matplotlib.pyplot as plt


def read_results(csv_path: str, result_col: str = "result", skip_unsettled: bool = True) -> np.ndarray:
    """CSV から result 列を読み、勝ちを1、それ以外を0として返す。

    skip_unsettled=True の場合は 'unsettled' や 'RETRY' を持つ行を除外する。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ログファイルが見つかりません: {csv_path}")

    values: List[float] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV にヘッダがありません。")
        if result_col not in reader.fieldnames:
            raise ValueError(f"CSV に result カラムが見つかりません。カラム候補: {reader.fieldnames}")

        for row in reader:
            # まず result_col を取り出し、None 安全かつ文字列化して扱う
            raw = row.get(result_col)
            res = str(raw).strip() if raw is not None else ""

            # 'unsettled' や 'RETRY' はスキップ（大文字小文字無視）
            if res and skip_unsettled and res.lower() in ("unsettled", "retry"):
                continue

            # 数値として解釈できるなら、0 より大きいとき勝ち(1)、それ以外 0
            if res:
                try:
                    num = float(res)
                    values.append(1.0 if num > 0 else 0.0)
                    continue
                except ValueError:
                    # 数値でなければ文字として解釈（'win' を勝ちとする）
                    values.append(1.0 if res.lower() == "win" else 0.0)
                    continue

            # res が空（もしくは欠損）の場合は 'reward' 列にフォールバック
            rew_raw = row.get("reward")
            if rew_raw is not None:
                rew = str(rew_raw).strip()
                if rew:
                    try:
                        num = float(rew)
                        values.append(1.0 if num > 0 else 0.0)
                        continue
                    except ValueError:
                        pass

            # どちらも解釈できない場合は無視
            continue

    return np.array(values, dtype=float)


def sliding_win_rate(arr: np.ndarray, window: int) -> (np.ndarray, np.ndarray):
    n = arr.size
    if n == 0:
        return np.array([]), np.array([])
    if window <= 0:
        raise ValueError("window は 1 以上の整数を指定してください")
    if window == 1:
        x = np.arange(1, n + 1)
        return x, arr.copy()

    # 畳み込みでスライディング平均（'valid' は長さ n-window+1）
    conv = np.convolve(arr, np.ones(window, dtype=float), mode="valid")
    rates = conv / float(window)
    x = np.arange(window, n + 1)
    return x, rates


def cumulative_win_rate(arr: np.ndarray) -> (np.ndarray, np.ndarray):
    n = arr.size
    if n == 0:
        return np.array([]), np.array([])
    cum = np.cumsum(arr)
    rates = cum / np.arange(1, n + 1)
    x = np.arange(1, n + 1)
    return x, rates


def plot_and_save(x: np.ndarray, y: np.ndarray, out_path: str, title: str = "Win rate"):
    if x.size == 0:
        raise ValueError("プロットするデータがありません（フィルタで行が除外されすぎている可能性があります）。")

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label="win rate", color="#1f77b4")
    plt.fill_between(x, 0, y, color="#aec7e8", alpha=0.3)
    plt.ylim(0, 1)
    plt.xlabel("ゲーム数")
    plt.ylabel("勝率")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"保存しました: {out_path}")


def main():
    p = argparse.ArgumentParser(description="指定したゲーム数での勝率推移をプロットします")
    p.add_argument("--log", type=str, default=os.path.join(os.path.dirname(__file__), "play_log.csv"),
                   help="読み込むログCSVファイル（デフォルト: BlackJack/play_log.csv）")
    p.add_argument("--window", type=int, default=100, help="スライディングウィンドウのサイズ（type=sliding 時）")
    p.add_argument("--type", type=str, choices=("sliding", "cumulative"), default="sliding",
                   help="sliding: 指定ウィンドウでの移動平均 / cumulative: 累積勝率")
    p.add_argument("--output", type=str, default="winrate.png", help="出力画像ファイル名")
    p.add_argument("--skip-unsettled", action="store_true", help="'unsettled' や 'RETRY' をログから除外する")
    p.add_argument("--result-col", type=str, default="result", help="勝敗を表すカラム名（デフォルト 'result'）")

    args = p.parse_args()

    try:
        arr = read_results(args.log, result_col=args.result_col, skip_unsettled=args.skip_unsettled)
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(2)

    if args.type == "cumulative":
        x, y = cumulative_win_rate(arr)
        title = f"Cumulative win rate ({os.path.basename(args.log)})"
    else:
        if arr.size < args.window:
            print(f"警告: データ点数({arr.size})が window({args.window}) 未満です。累積プロットにフォールバックします。")
            x, y = cumulative_win_rate(arr)
            title = f"Cumulative win rate (fallback, {os.path.basename(args.log)})"
        else:
            x, y = sliding_win_rate(arr, args.window)
            title = f"Sliding win rate (window={args.window}) - {os.path.basename(args.log)}"

    try:
        plot_and_save(x, y, args.output, title=title)
    except Exception as e:
        print(f"プロット中にエラー: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
