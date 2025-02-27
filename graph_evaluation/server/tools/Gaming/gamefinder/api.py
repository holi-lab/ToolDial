import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def server_info_copy(ip: str, port: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve eco server information"
    ip: IP Address or URL of an eco server
        port: Web Port of the specified server.
        
    """
    url = f"https://gamefinder.p.rapidapi.com/rust/serverinfo"
    querystring = {'ip': ip, 'port': port, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gamefinder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def server_info(ip: str, webport: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve eco server information"
    ip: IP Address or URL of an eco server
        webport: Web Port of the specified server.
        
    """
    url = f"https://gamefinder.p.rapidapi.com/eco/serverinfo"
    querystring = {'ip': ip, 'webport': webport, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gamefinder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def factorio_servers_online(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all online factorio servers."
    
    """
    url = f"https://gamefinder.p.rapidapi.com/factorio/games"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gamefinder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def factorio_server_details(gameid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details about a specific server."
    
    """
    url = f"https://gamefinder.p.rapidapi.com/factorio/gamedetails"
    querystring = {'gameid': gameid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gamefinder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

