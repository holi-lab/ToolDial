import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_women_s_rankings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back data from the PDGA United States Tour including player name, rank, rating, picture & PDGA profile url."
    
    """
    url = f"https://current-pdga-united-states-tour-rankings.p.rapidapi.com/pdga-rankings-women"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "current-pdga-united-states-tour-rankings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_men_s_rankings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back data from the PDGA United States Tour including player name, rank, rating, picture & PDGA profile url."
    
    """
    url = f"https://current-pdga-united-states-tour-rankings.p.rapidapi.com/pdga-rankings-men"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "current-pdga-united-states-tour-rankings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

