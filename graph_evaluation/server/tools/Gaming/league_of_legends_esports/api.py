import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_tournaments_for_league(leagueid: str='101097443346691685', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint can gives you all of the tournaments for league id. You can get all league id from the Get Leagues endpoint . If you want you can get all tournaments data without filtering leagues."
    
    """
    url = f"https://league-of-legends-esports.p.rapidapi.com/tournaments"
    querystring = {}
    if leagueid:
        querystring['leagueId'] = leagueid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_vods(tournamentid: str='107458367237283414', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "In this api method you can get all previous event , match game information. You can search by tournaments id and you can get this value from the Get Tournaments For League endpoint."
    
    """
    url = f"https://league-of-legends-esports.p.rapidapi.com/vods"
    querystring = {}
    if tournamentid:
        querystring['tournamentId'] = tournamentid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_game(gameid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "In this api method you can search all of the game data with game id. You can get this id from the Get Vods api method."
    
    """
    url = f"https://league-of-legends-esports.p.rapidapi.com/game/{gameid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_live(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "In this method you can get all live events score"
    
    """
    url = f"https://league-of-legends-esports.p.rapidapi.com/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_schedule(leagueid: str='98767991299243165%2C99332500638116286%2C98767991302996019', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "In this method you can get all schedule for the leagues. You can filter multiple leagueId with join %  seperator."
    
    """
    url = f"https://league-of-legends-esports.p.rapidapi.com/schedule"
    querystring = {}
    if leagueid:
        querystring['leagueId'] = leagueid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_teams_and_players(name: str=None, is_id: str='lng-esports', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can get all teams and player this api methods also you can filtered by team id or name."
    
    """
    url = f"https://league-of-legends-esports.p.rapidapi.com/teams"
    querystring = {}
    if name:
        querystring['name'] = name
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_statics(period: str, tier: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives you champion statics. This statics can filtered time period and league."
    tier: You can filtered league name for example ;
Iron, Bronze, Silver, Gold, Platinum, Diamond, Master, Grandmaster, Challenger
        
    """
    url = f"https://league-of-legends-esports.p.rapidapi.com/statics"
    querystring = {'period': period, }
    if tier:
        querystring['tier'] = tier
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_standings(tournamentid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "In this api method you can get all standings for tournaments. You can get tournaments id Get Tournaments From League method."
    
    """
    url = f"https://league-of-legends-esports.p.rapidapi.com/standings/{tournamentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_event_detail(matchid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "In this api method you can get all of the event detail from the match or event id (exactly same those ids.) you can get this id from the Get Vods or Get Match api methods."
    
    """
    url = f"https://league-of-legends-esports.p.rapidapi.com/events/{matchid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_leagues(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This api method can gives you all of the leagues and region information"
    
    """
    url = f"https://league-of-legends-esports.p.rapidapi.com/leagues"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

