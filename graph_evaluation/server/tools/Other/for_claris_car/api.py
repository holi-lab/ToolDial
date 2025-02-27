import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ping(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "疎通確認用のAPIです。"
    
    """
    url = f"https://for-claris-car.p.rapidapi.com/v1/ping"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "for-claris-car.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transport_node(word: str, type: str=None, datum: str='wgs84', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "駅名検索：キーワードや住所など、条件に該当する駅・空港などの一覧を取得します。"
    
    """
    url = f"https://for-claris-car.p.rapidapi.com/v1/transport_node"
    querystring = {'word': word, }
    if type:
        querystring['type'] = type
    if datum:
        querystring['datum'] = datum
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "for-claris-car.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def address(address_filter: str=None, word: str='代々木', datum: str='wgs84', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "住所検索：キーワードを指定して緯度経度等の住所情報を取得します。"
    
    """
    url = f"https://for-claris-car.p.rapidapi.com/v1/address"
    querystring = {}
    if address_filter:
        querystring['address_filter'] = address_filter
    if word:
        querystring['word'] = word
    if datum:
        querystring['datum'] = datum
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "for-claris-car.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shape_car(start: str, goal: str, via_type: str=None, goal_time: str=None, start_time: str=None, via: str=None, condition: str=None, no: int=None, datum: str='wgs84', options: str=None, order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ルート形状取得(車）：自動車の最適ルートを探索し、その結果を形状で取得します。"
    
    """
    url = f"https://for-claris-car.p.rapidapi.com/v1/shape_car"
    querystring = {'start': start, 'goal': goal, }
    if via_type:
        querystring['via_type'] = via_type
    if goal_time:
        querystring['goal_time'] = goal_time
    if start_time:
        querystring['start_time'] = start_time
    if via:
        querystring['via'] = via
    if condition:
        querystring['condition'] = condition
    if no:
        querystring['no'] = no
    if datum:
        querystring['datum'] = datum
    if options:
        querystring['options'] = options
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "for-claris-car.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def route_car(start: str, goal: str, via_type: str=None, via: str=None, goal_time: str=None, start_time: str=None, order: str=None, condition: str=None, options: str=None, datum: str='wgs84', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ルート検索（車）：自動車を移動手段として、2地点間のルートを探索します。複数経由地を巡回するルート検索や、多対1検索も行えます。"
    
    """
    url = f"https://for-claris-car.p.rapidapi.com/v1/route_car"
    querystring = {'start': start, 'goal': goal, }
    if via_type:
        querystring['via_type'] = via_type
    if via:
        querystring['via'] = via
    if goal_time:
        querystring['goal_time'] = goal_time
    if start_time:
        querystring['start_time'] = start_time
    if order:
        querystring['order'] = order
    if condition:
        querystring['condition'] = condition
    if options:
        querystring['options'] = options
    if datum:
        querystring['datum'] = datum
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "for-claris-car.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

