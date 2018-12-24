from sklearn.preprocessing import LabelBinarizer
import flashtext

cities = ['猫', '狗', '熊猫', '熊猫']
encoder = LabelBinarizer()
city_labels = encoder.fit_transform(cities)
print(city_labels)
# output：
# [[0 0 1]
#  [0 1 0]
#  [1 0 0]]

processor = flashtext.KeywordProcessor()
processor.get_all_keywords()
print(processor)
