# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:52:22 2017

@author: xps13
"""

import os
import xml.etree.ElementTree as ET
import logging

logger = logging.getLogger()


def renumber(input_path):
    f = open('test.txt', 'w')  # 若是'wb'就表示写二进制文件
    for root, dirs, files in os.walk(input_path):
        for filename in files:
            if filename.split('.')[-1].lower() in ("xml"):
                file_full_name = root + os.path.sep + filename
                tree = ET.parse(filename)
                tree_root = tree.getroot()
                for size in tree_root.iter('size'):
                    width = size.find("width").text
                    height = size.find("height").text
                for name in tree_root.iter('object'):
                    for child_name in name.iter('bndbox'):
                        a = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                            filename, width, height
                            , child_name[0].text, child_name[1].text
                            , child_name[2].text, child_name[3].text
                            , int(width) - int(child_name[0].text), int(height) - int(child_name[1].text)
                            , int(width) - int(child_name[2].text), int(height) - int(child_name[3].text))

                        f.write(a)
    f.close()


# renumber('.\\')
def write_xml(tree, out_path):
    """
    将xml文件写出
    tree: xml树
    out_path: 写出路径
    """
    tree.write(out_path, encoding="utf-8", xml_declaration=False)


def is_match(node, key, value):
    """
    判断某个节点是否包含所有传入参数属性
    node: 节点
    kv_map: 属性及属性值组成的map
    """
    for child_node in node.getchildren():
        print(child_node.tag, child_node.text)
        if child_node.tag == key and child_node.text == value:
            return True
    return False

    # for key in kv_map:
    #     print(kv_map.get(key), node.get(key))
    #     if node.get(key) != kv_map.get(key):
    #         return False
    #     return True


def judge_is_useful_file(filename):
    """
    判断是否为合法文件，如果为合法的，则对不合理的地方进行修改，如果不合理，则删除文件
    :param filename: 传入文件全路径，便于删除
    :return:
    """
    tree = ET.parse(filename)
    tree_root = tree.getroot()
    children = tree_root.getchildren()
    for child in children:
        if child.tag == 'object' and not is_match(child, key='name', value='person'):
            tree_root.remove(child)

    need_save = False
    children = tree_root.getchildren()
    for child in children:
        if child.tag == 'object' and is_match(child, key='name', value='person'):
            need_save = True

    if need_save:
        write_xml(tree, filename)
        logger.debug('重写：%s' % filename)
    else:
        os.remove(filename)
        logger.error('删除：%s' % filename)


def main(file_dir):
    """
    列出所有的xml文件并进行处理
    file_dir: 要处理的文件夹
    :return:
    """
    files = os.listdir(file_dir)
    for file in files:
        if file.rsplit('.')[-1] == 'xml':
            judge_is_useful_file(filename=file)


main(file_dir=os.getcwd())
