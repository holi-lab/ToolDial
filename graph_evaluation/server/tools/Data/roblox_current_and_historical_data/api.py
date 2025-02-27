import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def isuserdeveloperwithgamelist(userid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Acts the same as IsUserDeveloper, however this endpoint is not compatible with batched user requests. You can only request data for one user at a time with this endpoint. The major difference is that this also returns a full list of all the games under the player's account and groups that RTrack found, up to a limit of 15 (ordered by total visit count)"
    
    """
    url = f"https://roblox-current-and-historical-data.p.rapidapi.com/IsUserDeveloperWithGameList"
    querystring = {'UserId': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "roblox-current-and-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_game_current_concurrent_rank(placeid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the game (given by placeid parameter) rank across the whole of the Roblox platform, ordered by the number of people currently playing the game."
    
    """
    url = f"https://roblox-current-and-historical-data.p.rapidapi.com/GetNowConcurrentRank"
    querystring = {'placeid': placeid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "roblox-current-and-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_yesterday_s_game_concurrent_rank(placeid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the concurrent rank for this game 24 hours ago, allowing for comparison between the two stats."
    
    """
    url = f"https://roblox-current-and-historical-data.p.rapidapi.com/GetYesterdayConcurrentRank"
    querystring = {'placeid': placeid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "roblox-current-and-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_game_details(rootplaceid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets details about a certain Roblox game, updated live. Response includes the game's name, the creator's name, the number of people playing, likes, dislikes and visits."
    
    """
    url = f"https://roblox-current-and-historical-data.p.rapidapi.com/getgamedetails"
    querystring = {'rootplaceid': rootplaceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "roblox-current-and-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

