import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def schedule(team: str='GSW', date: str='31-01-2022', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the stats of past NBA games and schedule for upcoming ones"
    team: Team Tri Code (LAL, GSW, LAC, etc)
        date: DD-MM-YYYY
        
    """
    url = f"https://nba-schedule.p.rapidapi.com/schedule"
    querystring = {}
    if team:
        querystring['team'] = team
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-schedule.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

