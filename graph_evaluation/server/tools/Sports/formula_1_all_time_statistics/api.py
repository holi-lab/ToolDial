import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_statistics_in_specified_category_from_specified_season(filter: str, category: str, season: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint can return all race results, fastest lap awards, final (or current) championship standings. You can also specify a certain team or driver to see certain informations about them."
    
    """
    url = f"https://formula-1-all-time-statistics.p.rapidapi.com/{season}/{category}/{filter}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-all-time-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

