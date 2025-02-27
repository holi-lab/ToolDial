import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_user_build_info(champid: str, buildid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides information from selected build based on champid and buildid params. Info includes:
		-pathmain
		-pathsecondary
		-mainrunes
		-secondaryrunes
		-bonusrunes
		-items"
    
    """
    url = f"https://mobafire-lol-builds.p.rapidapi.com/api/builds/{champid}/{buildid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mobafire-lol-builds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_champion_suggested_runes(champid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns suggested runes for selected champion based on high win percentage rune combo."
    
    """
    url = f"https://mobafire-lol-builds.p.rapidapi.com/api/champions/{champid}/runes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mobafire-lol-builds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_builds_for_champion(champid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of most popular builds for selected champion. The "buildid" is extracted from the url and is used to get that specific builds info in "GET User Build Info""
    
    """
    url = f"https://mobafire-lol-builds.p.rapidapi.com/api/builds/{champid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mobafire-lol-builds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_champion_suggested_items(champid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns top percentage items for selected champion."
    
    """
    url = f"https://mobafire-lol-builds.p.rapidapi.com/api/champions/{champid}/items"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mobafire-lol-builds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_champion_stats(champid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns selected champion stats
		Includes: name, champid,  winP,  banP,  pickP,  and url"
    
    """
    url = f"https://mobafire-lol-builds.p.rapidapi.com/api/champions/{champid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mobafire-lol-builds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_champions_counters(champid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get easiest matchups for selected champion based on "counter rate" from counterstats.net"
    
    """
    url = f"https://mobafire-lol-builds.p.rapidapi.com/api/champions/{champid}/counters"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mobafire-lol-builds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_champions_that_counter_my_champion(champid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the hardest matchups for your selected champion based on "counter rate" from counterstats.net"
    
    """
    url = f"https://mobafire-lol-builds.p.rapidapi.com/api/champions/{champid}/counteredby"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mobafire-lol-builds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_champion_id_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of champion names along with their champid.
		-Organized by winP"
    
    """
    url = f"https://mobafire-lol-builds.p.rapidapi.com/api/champions/id-list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mobafire-lol-builds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_champion_stats(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns champions organized by win percentage.
		Includes: name, champid,  winP,  banP,  pickP,  and url"
    
    """
    url = f"https://mobafire-lol-builds.p.rapidapi.com/api/champions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mobafire-lol-builds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

