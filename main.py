import sys

import argparse
import numpy as np
import os
import shutil

from sanken import create_test_image    # テスト用の入力画像の生成
from sanken import split_image          # 入力画像ファイルをパッチ画像に分割
from sanken import result_util          # PatchCore出力の加工用ユーティリティ関数群

def main():
    # 引数から入力ファイル名を取得
    parser = argparse.ArgumentParser(description="Process input image file.")
    parser.add_argument("input_file", type=str, help="Path to the input image file")
    args = parser.parse_args()

    input_file_name: str = args.input_file    # 入力画像ファイル名 (ex. "input_image.png")

    #===========================================================
    # テスト用の入力画像を生成
    input_w = 800
    input_h = 600

    create_test_image.create_test_image(input_file_name, input_w, input_h)

    #===========================================================
    # 入力画像をパッチ画像に分割
    patch_h = 160   # パッチの高さ
    patch_w = 160   # パッチの幅
    pad = 32        # パッチの周囲の余分の幅

    temp_dir_name, patch_count = split_image.split_image(input_file_name, patch_h, patch_w, pad)

    #===========================================================
    # 最終結果の出力先となる、入力画像と同じサイズのdouble配列を生成
    final_result_arr1 = np.zeros((input_w, input_h), dtype=np.float64)  # パターン１用
    final_result_arr2 = np.zeros((input_w, input_h), dtype=np.float64)  # パターン２用

    #===========================================================
    # PatchCoreの結果を模擬する (1, 1, 14, 14) の配列を作成 (中身はランダム値)
    dummy_result_arr14x14 = np.random.rand(1, 1, 14, 14).astype(np.float64)

    slice2d = dummy_result_arr14x14[0, 0, :, :]
    np.savetxt('dummy_result_arr14x14.csv', slice2d, delimiter=',')

    #===========================================================
    # 【パターン１】
    # (1, 1, 14, 14)の配列の中央部(10, 10)の領域を抽出して16倍して(160, 160)の配列を取得する
    dummy_result_arr160x160 = result_util.crop_and_scale_nn(dummy_result_arr14x14)
    np.savetxt('dummy_result_arr160x160.csv', dummy_result_arr160x160, delimiter=',')

    # 160x160のパッチを、最終結果の配列の指定の場所に貼り付け
    y=160
    x=160
    result_util.paste_patch(final_result_arr1, dummy_result_arr160x160, y, x)
    np.savetxt('final_result_arr1.csv', final_result_arr1, delimiter=',')

    #===========================================================
    # 【パターン２】
    # (1, 1, 14, 14)の配列を縦横16倍して(224, 224)の配列を作成
    dummy_result_arr224x224 = result_util.upscale_nearest_repeat(dummy_result_arr14x14, 16)
    np.savetxt('dummy_result_arr224x224.csv', dummy_result_arr224x224, delimiter=',')

    # (224, 224)のパッチの中央部の(160, 160)の領域を切り出して、最終結果の配列の指定の場所に貼り付け
    #y=160
    #x=160
    result_util.paste_center_patch_2d_inplace(final_result_arr2, dummy_result_arr224x224, y, x, pad=pad)
    np.savetxt('final_result_arr2.csv', final_result_arr2, delimiter=',')

    #===========================================================

    # 最終結果の配列をtiffファイルに保存
    base, _ = os.path.splitext(input_file_name) # 入力ファイル名の拡張子を除外
    result_file_name = base + "_result.tiff"    # 最終結果のファイル名を決定

    result_util.save_file(result_file_name, final_result_arr2)

    # テンポラリフォルダ(パッチ画像の保存先フォルダ)の削除
    shutil.rmtree(f".\\{temp_dir_name}")

# 実行コマンド例 : python .\main.py input_image.png
if __name__ == "__main__":
    main()