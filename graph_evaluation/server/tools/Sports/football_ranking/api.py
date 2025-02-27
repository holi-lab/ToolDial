import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def footballrank(league: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can call the API to get the ranking of your favorite league, choosing from [ITA for Italian Serie A, GER for German Bundesliga, SPA for Spanish La Liga, FRA1 for French Ligue 1, EUR for European Championship Finals, EURW for European World Cup Qualifiers, CHMP for Champions League, UEFA for UEFA Europa League, ENGP for English Premier League, DUTCH for Dutch Eredivisie SCOT for Scottish Premiership]
		and entering the value in query under the league field."
    league: ITA-A
        
    """
    url = f"https://football-ranking.p.rapidapi.com/competitionRank"
    querystring = {'league': league, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-ranking.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

