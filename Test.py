import pandas as pd

# import pyprind
import time
import logging
# import tensorflow as tf

# 进度条指标
# for progress in pyprind.prog_bar(20):
#     time.sleep(1)
#
# # 百分比指标
# for progress in pyprind.prog_percent(range(20)):
#     time.sleep(1)

# bar = pyprind.ProgBar(100)
# for i in range(100):
#     time.sleep(0.5)
#     bar.update()

import re

string = '我很很 很很 很很  好'
string = re.sub(r"\s{2,}", " ", string)

print(string)