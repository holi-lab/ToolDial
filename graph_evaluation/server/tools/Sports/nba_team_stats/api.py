import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def team_stats(leagueyear: str=None, team: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides team stats for the current league year. Optionally, supply a `leagueYear` or `team` parameter to get results for those options."
    
    """
    url = f"https://nba-team-stats.p.rapidapi.com/teamStats"
    querystring = {}
    if leagueyear:
        querystring['leagueYear'] = leagueyear
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-team-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

