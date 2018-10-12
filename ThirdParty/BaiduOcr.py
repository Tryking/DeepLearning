import time

from aip import AipOcr

""" 你的 APPID AK SK """
APP_ID = '11778419'
API_KEY = 'R9WWCzzQS4e1G7GjDuAhV0ye'
SECRET_KEY = 't2OIvfdhnmYGpGYkL5VmLX4WfB08KRGd '

client = AipOcr(APP_ID, API_KEY, SECRET_KEY)


def get_file_content(filePath):
    """ 读取图片 """
    with open(filePath, 'rb') as fp:
        return fp.read()


image = get_file_content('test.jpg')

""" 调用通用文字识别, 图片参数为本地图片 """
client.basicGeneral(image)

""" 如果有可选参数 """
options = {}
options["language_type"] = "CHN_ENG"
options["detect_direction"] = "true"
options["detect_language"] = "true"
options["probability"] = "true"

""" 带参数调用通用文字识别, 图片参数为本地图片 """
start = time.time()
general = client.basicGeneral(image, options)
cost = round(time.time() - start, ndigits=1)
print(general)
print('\ncost: %s s' % cost)

# url = "https//www.x.com/sample.jpg"
#
# """ 调用通用文字识别, 图片参数为远程url图片 """
# client.basicGeneralUrl(url)
#
# """ 如果有可选参数 """
# options = {}
# options["language_type"] = "CHN_ENG"
# options["detect_direction"] = "true"
# options["detect_language"] = "true"
# options["probability"] = "true"
#
# """ 带参数调用通用文字识别, 图片参数为远程url图片 """
# client.basicGeneralUrl(url, options)
