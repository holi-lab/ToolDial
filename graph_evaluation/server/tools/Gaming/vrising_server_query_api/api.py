import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def discord_185_225_233_110_30015(port: str, ip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the discord link found in the server description."
    
    """
    url = f"https://vrising-server-query-api.p.rapidapi.com/discord/{ip}/{port}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vrising-server-query-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keydetails_185_225_233_110_30015(port: str, ip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the key details from server - change `/IP/PORT`"
    
    """
    url = f"https://vrising-server-query-api.p.rapidapi.com/keyDetails/{ip}/{port}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vrising-server-query-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playercount_185_225_233_110_30015(port: str, ip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the player count from server - change `/IP/PORT`"
    
    """
    url = f"https://vrising-server-query-api.p.rapidapi.com/playerCount/{ip}/{port}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vrising-server-query-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def connectip_185_225_233_110_30015(ip: str, port: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the connect IP from server - change `/IP/PORT`"
    
    """
    url = f"https://vrising-server-query-api.p.rapidapi.com/connectIP/{ip}/{port}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vrising-server-query-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def description_185_225_233_110_30015(ip: str, port: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the full description from server - change `/IP/PORT`"
    
    """
    url = f"https://vrising-server-query-api.p.rapidapi.com/description/{ip}/{port}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vrising-server-query-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specificplayer_185_225_233_110_30015_billy(name: str, port: int, ip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a specific player currently online on server - change `/IP/PORT`"
    
    """
    url = f"https://vrising-server-query-api.p.rapidapi.com/specificPlayer/{ip}/{port}/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vrising-server-query-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alldetails_185_225_233_110_30015(ip: str, port: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all details from server - change `/IP/PORT`"
    
    """
    url = f"https://vrising-server-query-api.p.rapidapi.com/allDetails/{ip}/{port}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vrising-server-query-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_185_225_233_110_30015(ip: str, port: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all players currently online from server - change `/IP/PORT`"
    
    """
    url = f"https://vrising-server-query-api.p.rapidapi.com/players/{ip}/{port}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vrising-server-query-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

