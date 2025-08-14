import numpy as np
from PIL import Image

# SMPTEバーのRGBカラー（近似値）
bars = [
    (255, 255, 255),  # white
    (255, 255, 0),    # yellow
    (0, 255, 255),    # cyan
    (0, 255, 0),      # green
    (255, 0, 255),    # magenta
    (255, 0, 0),      # red
    (0, 0, 255)       # blue
]

# カラーテストパターン画像の生成
def create_test_image(image_file_name: str='smpte_bars.png', width: int=4000, height: int=3000):

    bar_width = width // len(bars)

    img = np.zeros((height, width, 3), dtype=np.uint8)

    '''
    # 上部2/3に色バーを描画
    for i, color in enumerate(bars):
        img[:int(height * 2/3), i*bar_width:(i+1)*bar_width] = color
    '''
    for i, color in enumerate(bars):
        img[:height, i*bar_width:(i+1)*bar_width] = color

    # 画像を保存
    image = Image.fromarray(img)
    image.save(image_file_name)
