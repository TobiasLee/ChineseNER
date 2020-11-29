#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/17 15:18


import os
from collections import defaultdict
import random



def get_random(start, end):
    assert start <= end - 1
    return random.randint(start, end - 1)


class Ner:
    def __init__(self, ner_dir_name: str, ignore_tag_list: list,
                 data_augment_tag_list: list,
                 augment_size: int = 3,
                 seed: int = 0,
                 dedup: bool = True):
        random.seed(seed)
        self.ignore_tag_list = ignore_tag_list
        self.size = augment_size
        self.data_augment_tag_list = data_augment_tag_list
        self.tag_map = self.__get_all_tag_map(ner_dir_name, dedup=dedup)

    def __get_random_ner(self, tag: str):
        assert tag in self.tag_map
        max_size = len(self.tag_map[tag])
        assert max_size > 1
        select_idx = get_random(0, max_size)
        new_sene = self.tag_map[tag][select_idx]
        return new_sene

    def __get_all_tag_map(self, dir_name: str, dedup=True):
        '''
        get all named entities from train.txt and dev.txt，except those in the ignore_tag_list
        :param dir_name:
        :return:
        '''
        tag_map = defaultdict(list)
        for name in os.listdir(dir_name):
            # if name not in ['train.txt', 'dev.txt']:
            if name not in ['train.txt']:
                continue
            file_path = os.path.join(dir_name, name)
            data_iter = self.__get_file_data_iter(file_path)
            for char_tag in data_iter:
                t_tag, t_span = char_tag[0], char_tag[1]
                if t_tag in self.ignore_tag_list:
                    continue
                tag_map[t_tag].append(t_span)
        if dedup:
            for tag, nes in tag_map.items():
                tag_map[tag] = list(set(nes))
        return tag_map

    def __get_file_data_iter(self, file_path: str, aug_phase=False):
        with open(file_path, 'r', encoding='utf-8') as r_f:
            pre_tag = ''
            span = ''
            for line in r_f:
                if line == '\n':  # TODO
                    yield [pre_tag, span]
                    pre_tag = ''
                    span = ''
                    if aug_phase:
                        yield 'SF'  # sentence finished
                    continue
                t_char, t_label = line.replace('\n', '').split(' ')
                tp_tag = 'O'
                if 'O' != t_label:
                    tp_tag = t_label.split('-')[1]
                if pre_tag == '':
                    pre_tag = tp_tag
                    span += t_char
                elif pre_tag == tp_tag:
                    span += t_char
                else:
                    yield [pre_tag, span]
                    pre_tag = tp_tag
                    span = t_char
            if span != '':
                yield [pre_tag, span]

    def __data_augment_one(self, org_data):
        new_data = []
        for di in org_data:
            t_tag, t_span = di[0], di[1]
            if t_tag in self.data_augment_tag_list and t_tag in self.tag_map:
                rdm_select_ner = self.__get_random_ner(t_tag)
                new_data.append([t_tag, rdm_select_ner])
            else:
                new_data.append([t_tag, t_span])
        return new_data

    def __data_augment(self, org_data, size=3):
        '''
        对原始数据做增强
        :param org_data:
        :param size: 增强/最多/数量
        :return:
        '''

        new_data = []
        org_sent = ''.join([di[1] for di in org_data])
        for i in range(size):
            o_new_data = self.__data_augment_one(org_data)
            new_sent = ''.join([di[1] for di in o_new_data])
            if org_sent != new_sent:
                new_data.append(o_new_data)
        return new_data

    def __paser_ner(self, ner_data):
        # 数据还原成NER数组，字数组，标签数组
        sentence_arr = []
        label_arr = []
        for i in range(len(ner_data)):
            # if len(ner_data[i][1]) == 1 and ner_data[i][0] != 'O':  # for S
            #     label_arr.append('S-' + ner_data[i][0])
            #     sentence_arr.append(ner_data[i][1])
            #     continue
            for j in range(len(ner_data[i][1])):
                if ner_data[i][0] == 'O':
                    label_arr.append(ner_data[i][0])
                else:
                    if j == 0:
                        label_arr.append('B-' + ner_data[i][0])
                    else:
                        label_arr.append('I-' + ner_data[i][0])
                sentence_arr.append(ner_data[i][1][j])
        return sentence_arr, label_arr

    def augment(self, file_name) -> tuple:
        '''
        对文件做增强，输出文件路径，返回size个增强好的数据对 [sentence_arr, label_arr]
        :param file_name:
        :return:
        '''
        data_iter = self.__get_file_data_iter(file_name, aug_phase=True)
        org_data = []
        tag_span_pairs = []
        for tag_span_pair in data_iter:
            if tag_span_pair == 'SF' and tag_span_pairs is not []:
                org_data.append(tag_span_pairs)
                tag_span_pairs = []
            else:
                tag_span_pairs.append(tag_span_pair)
        if tag_span_pairs is not []:
            org_data.append(tag_span_pairs)

        new_datas = [self.__data_augment(data, self.size) for data in org_data]
        new_datas = sum(new_datas, []) + org_data
        random.shuffle(new_datas)  # TODO

        total_samples = []
        total_sample_tags = []
        for sample in new_datas:
            tokens, token_tags = self.__paser_ner(sample)
            total_samples.append(tokens)
            total_sample_tags.append(token_tags)
        return total_samples, total_sample_tags