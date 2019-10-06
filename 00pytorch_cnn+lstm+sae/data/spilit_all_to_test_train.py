import json
import random
import numpy as np


def get_list(num, dataset):
    list = []
    for k, v in dataset.items():
        if v == num:
            list.append(k)
    if num == 0:
        data_sample = random.sample(list, 10000)
    if num == 1:
        data_sample = list*20
    if num == 2:
        data_sample = list*200
    if num == 3:
        data_sample = list*10
    if num == 4:
        data_sample = list*500
    random.shuffle(data_all)
    return data_sample


if __name__ == '__main__':
    f = open('/home/dell/zy/00pytorch_cnn+lstm+sae/data/path_to_tagNew.json', 'r')
    dataset = json.load(f)
    data_all = []
    train = []
    test = []
    for i in range(5):
        data_all = get_list(i, dataset)
        length = len(data_all)
        for item in range(int(0.8*length)):
            train.append([data_all[item], i])
        for item in range(int(0.8*length), length):
            test.append([data_all[item], i])
    f = open('train_data_json', 'w')
    json.dump(train, f, indent=2)

    ff = open('test_data_json', 'w')
    json.dump(test, ff, indent=2)

