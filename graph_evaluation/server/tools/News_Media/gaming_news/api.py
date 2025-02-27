import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_individual_publisher_news(publisherid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets News about a specific Gaming Publisher.
		
		Available Publishers (36):
		- rockstar (Rockstar Games)
		- ubisoft
		- valve
		- activision
		- blizzard
		- bethesda
		- ea (Electronic Arts)
		- taketwo (Take-Two)
		- bandai
		- epic (Epic Games)
		- nintendo
		- capcom
		- squareenix (Square Enix)
		- playstation
		- microsoft
		- xbox
		- riot
		- 2k
		- amazon (Amazon Games)
		- cdprojektred (CD Projekt RED)
		- codemasters
		- crytek
		- gameloft
		- konami
		- mojang
		- supercell
		- tencent
		- origin
		- respawn
		- niantic
		- naughtydog (Naughty Dog)
		- treyarch
		- frontier
		- raven
		- infinityward
		- sledgehammer"
    
    """
    url = f"https://gaming-news1.p.rapidapi.com/news/{publisherid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gaming-news1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_news(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all news about gaming publisher from all over the world."
    
    """
    url = f"https://gaming-news1.p.rapidapi.com/news"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gaming-news1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

