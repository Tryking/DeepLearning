#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
import os
import re
from pyltp import Segmentor


# 清理 wiki 数据的 <doc> 标签
class CleanData(object):
    def __init__(self):
        self.pattern = '[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、：~@#￥%……&*（）]+'

    @staticmethod
    def remove_unuse_label(file_path):
        try:
            new_file_path = file_path + '_remove_label_handler'
            with open(file=new_file_path, mode='w+') as new_f:
                with open(file=file_path) as f:
                    lines = f.readlines()
                    pass_step = False
                    for line in lines:
                        content = line.strip()
                        if pass_step is True:
                            pass_step = False
                            pass
                        else:
                            if re.match('<doc', content) is not None:
                                # 它的下一行也是不需要的，跳过
                                pass_step = True
                            elif re.match('Category', content) is not None:
                                pass
                            elif re.match('</doc>', content) is not None:
                                pass
                            elif content is '':
                                pass
                            else:
                                new_f.write(content)
                                new_f.write('\n')
        except Exception as e:
            logging.error(str(e))

    @staticmethod
    def split_word(file_path):
        """
            分词
            :return:
        """
        new_file_path = file_path + '_split_word_handler'
        try:
            LTP_DATA_DIR = '/home/ReincarnationEyes/test/hit-scir/ltp_data_v3.4.0'  # ltp模型目录的路径
            cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
            segmentor = Segmentor()  # 初始化实例
            segmentor.load(cws_model_path)  # 加载模型
            with open(file=new_file_path, mode='w') as new_f:
                with open(file=file_path) as f:
                    lines = f.readlines()
                    for line in lines:
                        content = line.strip()
                        words = segmentor.segment(content)
                        new_f.write('\t'.join(words))
                        new_f.write('\n')
            segmentor.release()  # 释放模型
        except Exception as e:
            logging.error(str(e))

    @staticmethod
    def remove_punctuation(file_path):
        """
         移除标点符号等特殊符号
        :return:
        """
        try:
            new_file_path = file_path + '_remove_punctuation_handler'
            with open(file=new_file_path, mode='w') as new_f:
                with open(file=file_path) as f:
                    lines = f.readlines()
                    for line in lines:
                        content = line.strip()
                        content = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、：~@#￥%……&*（）]+|[A-Za-z0-9]+", "", content)
                        new_f.write(content)
                        new_f.write('\n')

            pass
        except Exception as e:
            logging.error(str(e))

    def main(self):
        try:
            # self.remove_unuse_label(file_path='/home/ReincarnationEyes/test/wiki_data/extracted/AA/wiki_00_chs')
            # self.split_word(file_path='/home/ReincarnationEyes/test/wiki_data/extracted/AA/all_remove_label.txt')
            self.remove_punctuation(file_path='/home/ReincarnationEyes/test/wiki_data/extracted/AA/all_remove_label.txt')
        except Exception as e:
            logging.error(str(e))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')
    clean_data = CleanData()
    clean_data.main()
