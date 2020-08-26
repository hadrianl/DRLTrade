#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/9 0009 11:01
# @Author  : Hadrianl 
# @File    : utils


from mongoengine import Document, StringField, FloatField, IntField, DateTimeField
import json
from pathlib import Path
from tensorboardX import SummaryWriter

Writer = SummaryWriter('run/ppo')
Writer_test = SummaryWriter('run/test')
CURDIR = Path.cwd()

def load_json_settings(filename: str):
    """
    Load data from json file in temp path.
    """
    filepath = CURDIR.joinpath(filename)

    if filepath.exists():
        with open(filepath, mode='r') as f:
            data = json.load(f)
        return data
    else:
        save_json_settings(filename, {})
        return {}

def save_json_settings(filename: str, data: dict):
    """
    Save data into json file in temp path.
    """
    filepath = CURDIR.joinpath(filename)
    with open(filepath, mode='w+') as f:
        json.dump(data, f, indent=4)


class HKMarketDataBaseDocument(Document):
    code = StringField(required=True)
    datetime = DateTimeField(required=True, unique_with='code')
    open = FloatField()
    high = FloatField()
    low = FloatField()
    close = FloatField()
    volume = IntField()
    trade_date = DateTimeField()

    meta = {'db_alias': 'HKFuture', 'abstract': True}