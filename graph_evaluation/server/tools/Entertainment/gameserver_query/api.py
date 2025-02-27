import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def game_server_query(host: str, type: str, token: str=None, port: int=None, port_query: int=None, login: str=None, password: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can find additional Informations and a List of the supported Games under https://market.mashape.com/millyvanilly/gameserver-query/overview"
    host: The Domain name or IP Address of the Game Server
        type: The Game Type of the Game Server
        token: REST user token, only required for Terraria
        port: Server Port, only required if not default
        port_query: The Query Port, only required if not default
        login: Only required for Nadeo Games like ShootMania, TrackMania ...
        password: Only required for Nadeo Games like ShootMania, TrackMania ...
        
    """
    url = f"https://gameserverquery.p.rapidapi.com/serverquery"
    querystring = {'host': host, 'type': type, }
    if token:
        querystring['token'] = token
    if port:
        querystring['port'] = port
    if port_query:
        querystring['port_query'] = port_query
    if login:
        querystring['login'] = login
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gameserverquery.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

