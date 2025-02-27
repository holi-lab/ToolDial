import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def credit_score_by_name_city_and_or_state(where: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Look up credit scores for private companies that do not have a ticker symbol. Almost 8 million companies available. Try searching by city and state first, then look for a name match."
    
    """
    url = f"https://business-credit-scores.p.rapidapi.com/classes/PrivateBusinessScores"
    querystring = {'where': where, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "business-credit-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def credit_score_by_ticker(where: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide a ticker and receive a score: 1- 10. 
		1 is the best and 10 is the worst."
    
    """
    url = f"https://business-credit-scores.p.rapidapi.com/classes/creditScores"
    querystring = {'where': where, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "business-credit-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def distinct_tickers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of tickers for which credit scores exist."
    
    """
    url = f"https://business-credit-scores.p.rapidapi.com/aggregate/creditScores"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "business-credit-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

