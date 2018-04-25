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

str = 'id="4123274" url="https://zh.wikipedia.org/wiki?curid=4123274" title="弗兰克·皮尔森"> '
match = re.match('<doc', str)
print(match.span())
