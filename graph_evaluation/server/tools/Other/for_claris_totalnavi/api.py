import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ping(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "疎通確認用のAPIです。"
    
    """
    url = f"https://for-claris-totalnavi.p.rapidapi.com/v1/ping"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "for-claris-totalnavi.p.rapidapi.com"
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
    url = f"https://for-claris-totalnavi.p.rapidapi.com/v1/transport_node"
    querystring = {'word': word, }
    if datum:
        querystring['datum'] = datum
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "for-claris-totalnavi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def route_transit(start: str, goal: str, via: str=None, limit: int=5, datum: str='wgs84', start_time: str='2020-08-19T10:00:00', via_type: str=None, use_car: str=None, unuse: str=None, order: str=None, goal_time: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ルート検索（トータルナビ）：電車（航空路線、徒歩含む）を移動手段として、２地点間のルートを探索します。経由地の指定も可能です。"
    
    """
    url = f"https://for-claris-totalnavi.p.rapidapi.com/v1/route_transit"
    querystring = {'start': start, 'goal': goal, }
    if via:
        querystring['via'] = via
    if limit:
        querystring['limit'] = limit
    if datum:
        querystring['datum'] = datum
    if start_time:
        querystring['start_time'] = start_time
    if via_type:
        querystring['via_type'] = via_type
    if use_car:
        querystring['use_car'] = use_car
    if unuse:
        querystring['unuse'] = unuse
    if order:
        querystring['order'] = order
    if goal_time:
        querystring['goal_time'] = goal_time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "for-claris-totalnavi.p.rapidapi.com"
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
    url = f"https://for-claris-totalnavi.p.rapidapi.com/v1/address"
    querystring = {}
    if address_filter:
        querystring['address_filter'] = address_filter
    if word:
        querystring['word'] = word
    if datum:
        querystring['datum'] = datum
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "for-claris-totalnavi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shape_transit(goal: str, start: str, goal_time: str=None, no: int=None, options: str=None, via: str=None, datum: str='wgs84', via_type: str=None, unuse: str=None, use_car: str=None, start_time: str='2020-08-19T10:00:00', limit: int=5, order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ルート形状取得（トータルナビ）：電車（航空路線、徒歩含む）を移動手段としてルートを探索し、その結果を形状で取得します。"
    
    """
    url = f"https://for-claris-totalnavi.p.rapidapi.com/v1/shape_transit"
    querystring = {'goal': goal, 'start': start, }
    if goal_time:
        querystring['goal_time'] = goal_time
    if no:
        querystring['no'] = no
    if options:
        querystring['options'] = options
    if via:
        querystring['via'] = via
    if datum:
        querystring['datum'] = datum
    if via_type:
        querystring['via_type'] = via_type
    if unuse:
        querystring['unuse'] = unuse
    if use_car:
        querystring['use_car'] = use_car
    if start_time:
        querystring['start_time'] = start_time
    if limit:
        querystring['limit'] = limit
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "for-claris-totalnavi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

