import os

from PIL import Image


def crop_image(image_path, top_border, bottom_border, left_border, right_border, output_path):
    """
    裁剪圖片檔，去掉頂部、底部、左側和右側邊界。

    :param image_path: 輸入圖片文件的路徑
    :param top_border: 要裁掉的頂部像素數
    :param bottom_border: 要裁掉的底部像素數
    :param left_border: 要裁掉的左側像素數
    :param right_border: 要裁掉的右側像素數
    :param output_path: 裁剪後圖片保存的路徑
    :return: 無（直接保存裁剪後的圖像）
    """
    # 打開圖片
    image = Image.open(image_path)

    # 獲取圖片的寬度和高度
    width, height = image.size

    # 計算新圖像的裁剪區域
    left = left_border
    top = top_border
    right = width - right_border
    bottom = height - bottom_border

    # 裁剪圖片
    cropped_image = image.crop((left, top, right, bottom))

    # 保存裁剪後的圖片
    cropped_image.save(output_path)
    print(f"裁剪後的圖片已保存到 {output_path}")

TO_BE_CROPPED_LIST = [
    '~/Pictures/Screenshots/lbf_diff_sight/1.png',
    '~/Pictures/Screenshots/lbf_diff_sight/3.png',
    '~/Pictures/Screenshots/lbf_diff_sight/6.png',
    '~/Pictures/Screenshots/lbf_diff_sight/999.png',
    '~/Pictures/Screenshots/lbf_diff_sight/all_player_same_color.png',
]

# Loop each path and crop it and show it
for FILE_PATH in TO_BE_CROPPED_LIST:
    # 設置裁剪邊界
    TOP_BORDER = 39
    BOTTOM_BORDER = 2
    LEFT_BORDER = 0
    RIGHT_BORDER = 2

    # Get ~'s real path auto
    FILE_PATH = os.path.expanduser(FILE_PATH)
    print(f"FILE_PATH={FILE_PATH}")


    # 輸出文件的路徑
    OUTPUT_PATH = FILE_PATH.replace('.png', '_processed.png')

    # 裁剪圖片
    crop_image(FILE_PATH, TOP_BORDER, BOTTOM_BORDER, LEFT_BORDER, RIGHT_BORDER, OUTPUT_PATH)

    # # 顯示裁剪後的圖片
    # Image.open(OUTPUT_PATH).show()
