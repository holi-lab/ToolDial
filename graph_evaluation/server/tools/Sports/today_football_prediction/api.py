import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def vip_prediction_scores(page: str='1', date: str='2022-08-13', league: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This **VIP** endpoint returns match score predictions with average goal stats. Same query strings and pagination functionality is supported with **/predictions/list**
		
		* Use **/predictions/list** endpoint  to get  market bet predictions."
    
    """
    url = f"https://today-football-prediction.p.rapidapi.com/predictions/scores"
    querystring = {}
    if page:
        querystring['page'] = page
    if date:
        querystring['date'] = date
    if league:
        querystring['league'] = league
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "today-football-prediction.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_predictions(league: str=None, page: str='1', market: str=None, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns **Daily Football Predictions** with pagination support. 
		Search/filter can be used by **date**, **league**, **market-type**.
		
		* Use **predictions/{matchId}/details** endpoint to see more information about any particular match prediction.
		* Use **predictions/scores** endpoint to get score prediction and average goals."
    league: Example: 1
        page: Example: 1
        market: Example: 1X2, OU25, bts

        date: Example: 2022-08-13
        
    """
    url = f"https://today-football-prediction.p.rapidapi.com/predictions/list"
    querystring = {}
    if league:
        querystring['league'] = league
    if page:
        querystring['page'] = page
    if market:
        querystring['market'] = market
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "today-football-prediction.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vip_featured_predictions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns **Daily  Featured Prediction** results which have higher probability and better odds than others. More efficient selections.
		
		Compare its performance from **stats/performance** endpoint."
    
    """
    url = f"https://today-football-prediction.p.rapidapi.com/predictions/featured"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "today-football-prediction.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prediction_details(match_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Shows all details about the match, its prediction and the prediction results."
    
    """
    url = f"https://today-football-prediction.p.rapidapi.com/predictions/{match_id}/details"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "today-football-prediction.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stats_performance(date: str='2022-08-13', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns statistics of AI prediction module with selected odds, prediction probabilities and profit-loss calculation. 
		
		**Profit-Loss**: calculated by considering only single bet is played from 1 unit amount. If the bet is won, gets prediction odd as positive number else -1."
    
    """
    url = f"https://today-football-prediction.p.rapidapi.com/stats/performance"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "today-football-prediction.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

