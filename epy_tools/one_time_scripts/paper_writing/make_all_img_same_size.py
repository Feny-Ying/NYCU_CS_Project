# import os.path
#
# from PIL import Image, ImageOps
#
# ROOT = '/media/ppo/workspace/epymarl_research/exp_plot/figures/'
#
# FILE_NAMES = [
#     'plot-SR@LBF.drawio.png',
#     'plot-SR@RWARE.drawio.png',
#     'smac_example.png',
#     'plot-SR@SMAC.png'
# ]
#
# # 讀取圖像
# images = [Image.open(os.path.join(ROOT, filename)) for filename in FILE_NAMES]
#
# # 找出所有圖像中最大的寬度和高度
# max_width = max(image.size[0] for image in images)
# max_height = max(image.size[1] for image in images)
#
# # 將每張圖片統一調整為最大寬度和高度，保持比例，填充白色
# resized_images = []
# for img in images:
#     # 設置目標畫布大小為最大寬高
#     new_img = ImageOps.pad(img, (max_width, max_height), color="white")
#     resized_images.append(new_img)
#
# # 保存調整後的圖像
# output_filenames = [os.path.join(ROOT, f"resized_{filename}") for filename in FILE_NAMES]
#
# for img, filename in zip(resized_images, output_filenames):
#     img.save(filename)
#
# print("Images have been resized and saved.")

import os
from PIL import Image, ImageOps

ROOT = '/media/ppo/workspace/epymarl_research/exp_plot/figures/'

FILE_NAMES = [
    'plot-SR@LBF.png',
    'plot-SR@RWARE.png',
    'smac_example.png',
    'plot-SR@SMAC.png'
]

# 讀取圖像
images = [Image.open(os.path.join(ROOT, filename)) for filename in FILE_NAMES]

# 找出最小的高度和最大的寬度
min_height = min(image.size[1] for image in images)
max_width = max(image.size[0] for image in images)

# 將每張圖片按最小高度等比例縮放，並左右填充
resized_images = []
for img in images:
    # 等比例縮放圖片，使高度等於最小高度
    img_resized = ImageOps.contain(img, (img.size[0], min_height))  # 等比例縮放
    # 填充到最大寬度
    new_img = ImageOps.pad(img_resized, (max_width, min_height), color="white", centering=(0.5, 0.5))  # 左右填充
    resized_images.append(new_img)

# 保存調整後的圖像
output_filenames = [os.path.join(ROOT, f"resized_{filename}") for filename in FILE_NAMES]

for img, filename in zip(resized_images, output_filenames):
    img.save(filename)

print("Images have been resized and saved.")
