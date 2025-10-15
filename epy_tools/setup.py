# 現在在 repo root 的 ./epy_tools/ 底下
# 這個檔案是用來安裝 epy_tools 的
# 幫我寫好的 setup.py 檔案
import setuptools


setuptools.setup(
    name="epy_tools",
    version="0.0.1",
    author="LiaoWC",
    packages=setuptools.find_packages(),
)

