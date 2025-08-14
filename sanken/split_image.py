import cv2
import numpy as np
import os
import re
import tempfile

from typing import Tuple

def create_temp_dir(prefix: str="patches_") -> Tuple[str, str]:
    """
    カレントにテンポラリフォルダ (パッチ画像の出力先) を作成

    Args:
        prefix:     作成するテンポラリフォルダの接頭語

    Returns:
        temp_dir_name (str): 作成したテンポラリフォルダ名
        good_dir (int):      テンポラリフォルダ下に作成した 'good' フォルダのパス (これがパッチ画像の出力先)
    """

    # カレントにパッチ画像の出力先テンポラリフォルダを作成
    temp_dir: str = tempfile.mkdtemp(prefix=prefix, dir=".")

    # テンポラリフォルダの「フォルダ名のみ」取得
    temp_dir_name: str = os.path.basename(temp_dir)  # フォルダ名のみ取得

    # テンポラリフォルダ下に 'good' サブフォルダ作成
    good_dir: str = os.path.join(temp_dir, "good")
    os.makedirs(good_dir, exist_ok=True)

    return (temp_dir_name, good_dir)


def get_padded_patch(img: np.ndarray, row: int, col: int, patch_h: int, patch_w: int, pad: int = 32) -> np.ndarray:
    """
    元画像 img から (row, col) を左上座標として patch_h x patch_w の領域を切り出し、
    さらに周囲に pad ピクセルの枠を付けて (patch_h + 2*pad) x (patch_w + 2*pad) サイズの画像を返す。

    枠部分が画像外の場合は黒で埋める。
    
    Args:
        img (np.ndarray): 元画像 (H, W, C)
        row (int):        切り出し開始の行
        col (int):        切り出し開始の列
        patch_h (int):    切り出し領域の高さ
        patch_w (int):    切り出し領域の幅
        pad (int):        周囲の枠幅（ピクセル）
    
    Returns:
        np.ndarray: パディング付きパッチ画像 (patch_h + 2*pad, patch_w + 2*pad, C)
    """
    h, w, c = img.shape
    out_h, out_w = patch_h + 2 * pad, patch_w + 2 * pad
    
    # 出力パッチ用の黒画像を作成
    patch_img = np.zeros((out_h, out_w, c), dtype=img.dtype)
    
    # 元画像から切り出す範囲（枠込みで計算）
    src_left = col - pad
    src_top = row - pad
    src_right = col + patch_w + pad
    src_bottom = row + patch_h + pad
    
    # 出力パッチのどの範囲に元画像のデータを埋め込むか
    dst_left = max(0, -src_left)    # src_left が負ならパディング側でずらす
    dst_top = max(0, -src_top)
    
    # 元画像の切り出し範囲（画像内に制限）
    src_left_clip = max(0, src_left)
    src_top_clip = max(0, src_top)
    src_right_clip = min(w, src_right)
    src_bottom_clip = min(h, src_bottom)
    
    # コピーする幅・高さ
    copy_w = src_right_clip - src_left_clip
    copy_h = src_bottom_clip - src_top_clip
    
    # 完全に外にはみ出してる場合は黒画像を返す
    if copy_h <= 0 or copy_w <= 0:
        return patch_img
    
    # 元画像からパッチ画像にコピー
    patch_img[dst_top:dst_top+copy_h, dst_left:dst_left+copy_w, :] = img[src_top_clip:src_bottom_clip, src_left_clip:src_right_clip, :]
    
    return patch_img


def split_image(input_image_path, patch_h: int=160, patch_w: int=160, pad: int=32) -> Tuple[str, int]:
    """
    画像を指定サイズのパッチに分割し、テンポラリフォルダに保存する。
    
    Args:
        input_image_path (str): 入力画像ファイルパス
        patch_h (int):          パッチの高さ
        patch_w (int):          パッチの幅
    
    Returns:
        temp_dir (str):     パッチ画像の保存先テンポラリフォルダのパス
        patch_count (int):  保存したパッチ数
    """

    # テンポラリフォルダの作成
    temp_dir_name, good_dir = create_temp_dir()
    
    # 画像読み込み
    img = cv2.imread(input_image_path)
    if img is None:
        raise FileNotFoundError(f"入力画像ファイルの読込みに失敗: {input_image_path}")
    
    h, w, _ = img.shape
    patch_count = 0
    
    # 画像の分割 & 保存
    for row in range(0, h, patch_h):
        for col in range(0, w, patch_w):
            # 周囲に枠をつけてパッチ画像を切り出し
            patch = get_padded_patch(img, row, col, patch_h, patch_w, pad)

            # 切り出したパッチ画像を 'good' フォルダ下に保存
            filename = f"patch_{row}_{col}.png"
            cv2.imwrite(os.path.join(good_dir, filename), patch)

            patch_count += 1

    return temp_dir_name, patch_count


def parse_patch_filename(filename: str) -> Tuple[int, int]:
    """
    パッチ画像ファイル名 'patch_{row}_{col}.png' から row, col を取り出す

    Args:
        filename (str): ファイル名（パスでなくファイル名のみ）

    Returns:
        Tuple[int, int]: (row, col) の座標
    
    Raises:
        ValueError: フォーマットが違う場合に発生
    """

    pattern = r"patch_(\d+)_(\d+)\.png$"
    match = re.match(pattern, filename)

    if not match:
        raise ValueError(f"ファイル名の形式が不正です: {filename}")

    row = int(match.group(1))
    col = int(match.group(2))

    return row, col