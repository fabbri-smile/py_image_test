import numpy as np
import tifffile     # 要 conda install tifffile

def upscale_nearest_repeat(arr: np.ndarray, scale: int) -> np.ndarray:
    """
    (1,1,H,W) の配列を補間なしでブロック拡大し、(H*scale, W*scale) の2次元配列を返す関数。
    
    Args:
        arr (np.ndarray):   入力配列 (1, 1, H, W)
        scale (int):        拡大倍率（縦横同じ）
    
    Returns:
        np.ndarray:     拡大後の2次元配列 (H*scale, W*scale)
    
    Raises:
        ValueError:     入力形状や scale の値が不正な場合
    """

    # 引数 scale が正の整数であることをチェック
    if not isinstance(scale, int) or scale <= 0:
        raise ValueError("scale は正の整数である必要があります。")
    
    # 入力配列のサイズをチェック
    if arr.ndim != 4 or arr.shape[0] != 1 or arr.shape[1] != 1:
        raise ValueError("入力配列は (1, 1, H, W) の形状である必要があります。")
    
    n, c, h, w = arr.shape
    
    out = np.repeat(np.repeat(arr, scale, axis=2), scale, axis=3)
    
    # (1, 1, H*scale, W*scale) -> (H*scale, W*scale)
    out_2d = out[0, 0]
    return out_2d


def paste_patch(result_img: np.ndarray, patch_img: np.ndarray, top_left_y: int, top_left_x: int) -> None:
    """
    result_img (H,W) に patch_img (160,160) を top_left_y, top_left_x の位置に上書き貼り付け（はみ出し考慮）。
    
    Args:
        result_img (np.ndarray):    貼り付け先の配列 (H,W)
        patch_img (np.ndarray):     貼り付けるパッチの配列 (160,160)
        top_left_y (int):           result_img 貼り付け開始の y 座標（行）
        top_left_x (int):           result_img 貼り付け開始の x 座標（列）
    
    Returns:
        None
    """

    # 貼り付ける画像のサイズ(160x160)を取得
    crop_h, crop_w = patch_img.shape

    # 貼り付け範囲クリップ（はみ出し対応）
    result_h, result_w = result_img.shape

    paste_start_y = max(top_left_y, 0)
    paste_start_x = max(top_left_x, 0)
    paste_end_y = min(top_left_y + crop_h, result_h)
    paste_end_x = min(top_left_x + crop_w, result_w)

    paste_h = paste_end_y - paste_start_y
    paste_w = paste_end_x - paste_start_x

    # patch_img 側のコピー範囲
    patch_start_y = max(0, -top_left_y)
    patch_start_x = max(0, -top_left_x)
    patch_end_y = patch_start_y + paste_h
    patch_end_x = patch_start_x + paste_w

    # 上書き貼り付け
    result_img[paste_start_y:paste_end_y, paste_start_x:paste_end_x] = patch_img[patch_start_y:patch_end_y, patch_start_x:patch_end_x]


def paste_center_patch_2d_inplace(result_img: np.ndarray, patch_img: np.ndarray, top_left_y: int, top_left_x: int, pad: int = 32) -> None:
    """
    result_img (H,W) に patch_img (224,224) の中央160x160部分のみを top_left_y, top_left_x の位置に上書き貼り付け（はみ出し考慮）。
    パディング部分は含まれない。
    
    Args:
        result_img (np.ndarray):    貼り付け先の配列 (H,W)
        patch_img (np.ndarray):     貼り付けるパッチの配列 (224,224)
        top_left_y (int):           result_img 貼り付け開始の y 座標（行）
        top_left_x (int):           result_img 貼り付け開始の x 座標（列）
        pad (int):                  パディング幅（既定32）
    
    Returns:
        None
    """

    # 中央160x160領域の座標計算
    patch_h, patch_w = patch_img.shape

    crop_start_y = pad
    crop_start_x = pad
    crop_h = patch_h - 2*pad
    crop_w = patch_w - 2*pad

    # 中央の160x160部分を切り出し
    patch_center = patch_img[crop_start_y:crop_start_y+crop_h, crop_start_x:crop_start_x+crop_w]

    # 切り出した160x160の部分を貼り付け先の配列に貼り付け
    paste_patch(result_img, patch_center, top_left_y, top_left_x)


def crop_and_scale_nn(arr: np.ndarray, crop_h: int = 10, crop_w: int = 10, scale: int = 16) -> np.ndarray:
    """
    入力:
        arr: NumPy ndarray      切り出し元の配列 (1, 1, 14, 14) を想定
        crop_h, crop_w:         中央から切り出す高さと幅（ピクセル単位）
        scale:                  拡大倍率（補間なしの nearest-neighbor）

    出力:
        形状が (1, 1, crop_h*scale, crop_w*scale) の ndarray を返す
    """

    # 入力配列の形状取得
    _, _, H, W = arr.shape

    # 中央から切り出す開始位置
    start_h = H // 2 - crop_h // 2
    start_w = W // 2 - crop_w // 2

    # (1, 1, 14, 14)の中央部(10, 10)を切り出し
    center = arr[0, 0, start_h:start_h + crop_h, start_w:start_w + crop_w]

    # 単純複製（nearest-neighbor）で縦横それぞれ scale 倍に拡大
    scaled = np.repeat(np.repeat(center, scale, axis=0), scale, axis=1)

    return scaled


def save_file(filename: str, arr_to_save: np.ndarray):
    """
    NumPy の float64（double）配列を TIFF ファイルに保存する関数。

    Args:
        filename (str):                 保存先のファイル名（例："output.tif"）
        arr_to_save (numpy.ndarray):    保存したいの配列
    """

    tifffile.imwrite(filename, arr_to_save)
