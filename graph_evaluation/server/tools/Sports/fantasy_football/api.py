import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def weekly_proj_by_position_nonppr(week: str, position: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the projected points for each fantasy football player for the specified week and position (nonPPR). Positions: QB, RB, WR, TE, K, DST"
    
    """
    url = f"https://fantasy-football2.p.rapidapi.com/weekly/projections/nonpprPosition/{week}/{position}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-football2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weekly_proj_by_position_ppr(position: str, week: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the projected points for each fantasy football player for the specified week and position (nonPPR). Positions: QB, RB, WR, TE, K, DST"
    
    """
    url = f"https://fantasy-football2.p.rapidapi.com/weekly/projections/pprPosition/{week}/{position}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-football2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weekly_rank_by_position_nonppr(position: str, week: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the  rankings for players for the specified week and position (nonPPR). Positions: QB, RB, WR, TE, K, DST"
    
    """
    url = f"https://fantasy-football2.p.rapidapi.com/weekly/rankings/nonpprPosition/{week}/{position}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-football2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weekly_rank_by_position_ppr(week: str, position: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the weekly rankings of players for the specified week and position (PPR). Positions: QB, RB, WR, TE, K, DST"
    
    """
    url = f"https://fantasy-football2.p.rapidapi.com/weekly/rankings/pprPosition/{week}/{position}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-football2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weekly_projections_by_player_nonppr(playername: str, week: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns projected points for specified week and specified player (nonPPR). Use hyphenated name and capital first letters (i.e. Patrick-Mahomes)"
    
    """
    url = f"https://fantasy-football2.p.rapidapi.com/weekly/projections/nonpprPlayer/{week}/{playername}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-football2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weekly_projections_by_player_ppr(playername: str, week: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns projected points for specified week and specified player (PPR). Use hyphenated name and capital first letters (i.e. Patrick-Mahomes)"
    
    """
    url = f"https://fantasy-football2.p.rapidapi.com/weekly/projections/pprPlayer/{week}/{playername}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-football2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_projections_by_player_nonppr(playername: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the remaining projected points for a specific player (nonPPR). Use hyphenated name and capital first letters (i.e. Tom-Brady)"
    
    """
    url = f"https://fantasy-football2.p.rapidapi.com/season/projections/nonppr/player/{playername}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-football2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_projections_by_player_ppr(playername: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the remaining projected points for a specific player (PPR). Use hyphenated name and capital first letters (i.e. Tom-Brady)"
    
    """
    url = f"https://fantasy-football2.p.rapidapi.com/season/projections/ppr/player/{playername}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-football2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_rankings_by_position_ppr(position: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the rankings of fantasy football players at specified position based on rest of season projections (PPR)."
    
    """
    url = f"https://fantasy-football2.p.rapidapi.com/season/rankings/ppr/{position}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-football2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_projections_by_position_ppr(position: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the remaining projected points for all players at specified position (PPR). Positions: QB, RB, WR, TE, K, DST"
    
    """
    url = f"https://fantasy-football2.p.rapidapi.com/season/projections/ppr/{position}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-football2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_projections_all_nonppr(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the remaining season projected points for each fantasy football player (nonPPR)."
    
    """
    url = f"https://fantasy-football2.p.rapidapi.com/season/projections/nonppr"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-football2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weekly_projections_all_ppr(week: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the projected points for all fantasy football players for the specified week (PPR)."
    
    """
    url = f"https://fantasy-football2.p.rapidapi.com/weekly/projections/ppr/{week}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-football2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weekly_projections_all_nonppr(week: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the projected points for all fantasy football players for the specified week (nonPPR)."
    
    """
    url = f"https://fantasy-football2.p.rapidapi.com/weekly/projections/nonppr/{week}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-football2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_rankings_by_position_nonppr(position: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the rankings of fantasy football players at specified position based on rest of season projections (nonPPR)."
    
    """
    url = f"https://fantasy-football2.p.rapidapi.com/season/rankings/nonppr/{position}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-football2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_projections_by_position_nonppr(position: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the remaining projected points for all players at specified position (nonPPR). Positions: QB, RB, WR, TE, K, DST"
    
    """
    url = f"https://fantasy-football2.p.rapidapi.com/season/projections/nonppr/{position}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-football2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_projections_all_ppr(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the remaining season projected points for each fantasy football player (PPR)."
    
    """
    url = f"https://fantasy-football2.p.rapidapi.com/season/projections/ppr"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-football2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

