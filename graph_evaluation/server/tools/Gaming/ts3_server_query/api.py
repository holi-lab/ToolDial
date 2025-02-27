import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_clientlist(ip: str, username: str, query_port: str, server_port: str, password: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all online clients"
    ip: ip from your TS3 server
        username: default: serveradmin
        query_port: default: 10011
        server_port: default: 9987
        password: default: no password
        
    """
    url = f"https://ts3-server-query.p.rapidapi.com/index.php"
    querystring = {'ip': ip, 'username': username, 'query_port': query_port, 'server_port': server_port, }
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ts3-server-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

