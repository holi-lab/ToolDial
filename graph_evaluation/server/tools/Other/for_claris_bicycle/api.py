import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ping(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "疎通確認用のAPIです。"
    
    """
    url = f"https://for-claris-bicycle.p.rapidapi.com/v1/ping"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "for-claris-bicycle.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transport_node(word: str, datum: str='wgs84', type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "駅名検索：キーワードや住所など、条件に該当する駅・空港などの一覧を取得します。"
    
    """
    url = f"https://for-claris-bicycle.p.rapidapi.com/v1/transport_node"
    querystring = {'word': word, }
    if datum:
        querystring['datum'] = datum
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "for-claris-bicycle.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def address(datum: str='wgs84', address_filter: str=None, word: str='代々木', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "住所検索：キーワードを指定して緯度経度等の住所情報を取得します。"
    
    """
    url = f"https://for-claris-bicycle.p.rapidapi.com/v1/address"
    querystring = {}
    if datum:
        querystring['datum'] = datum
    if address_filter:
        querystring['address_filter'] = address_filter
    if word:
        querystring['word'] = word
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "for-claris-bicycle.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def route_bicycle(goal: str, start: str, goal_time: str=None, via: str=None, options: str=None, via_type: str=None, condition: str=None, datum: str='wgs84', start_time: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ルート検索（自転車）：自転車を移動手段として、２地点間のルートを探索します。経由地の指定も可能です。"
    
    """
    url = f"https://for-claris-bicycle.p.rapidapi.com/v1/route_bicycle"
    querystring = {'goal': goal, 'start': start, }
    if goal_time:
        querystring['goal_time'] = goal_time
    if via:
        querystring['via'] = via
    if options:
        querystring['options'] = options
    if via_type:
        querystring['via_type'] = via_type
    if condition:
        querystring['condition'] = condition
    if datum:
        querystring['datum'] = datum
    if start_time:
        querystring['start_time'] = start_time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "for-claris-bicycle.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shape_bicycle(goal: str, start: str, goal_time: str=None, start_time: str=None, via_type: str=None, datum: str='wgs84', condition: str=None, via: str=None, options: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ルート形状取得（自転車）：自転車を移動手段としてルートを探索し、その結果を形状で取得します。"
    
    """
    url = f"https://for-claris-bicycle.p.rapidapi.com/v1/shape_bicycle"
    querystring = {'goal': goal, 'start': start, }
    if goal_time:
        querystring['goal_time'] = goal_time
    if start_time:
        querystring['start_time'] = start_time
    if via_type:
        querystring['via_type'] = via_type
    if datum:
        querystring['datum'] = datum
    if condition:
        querystring['condition'] = condition
    if via:
        querystring['via'] = via
    if options:
        querystring['options'] = options
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "for-claris-bicycle.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

