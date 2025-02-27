import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_today_s_win_draw_double_chance_predictions(event_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""Get Today's Predictions" is a powerful feature offered by the predictX API that allows developers to retrieve accurate predictions for events happening on the current day. This feature provides users with valuable insights and helps them make informed decisions about upcoming matches.
		
		 These predictions are generated using advanced algorithms and statistical models that take into account various factors such as team performance, player statistics, historical data, and other relevant parameters."
    
    """
    url = f"https://predictx.p.rapidapi.com/todaypredictions"
    querystring = {'event_date': event_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "predictx.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_next_win_draw_double_chance_predictions(event_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""Get Next Predictions" is a powerful feature offered by the predictX API that allows developers to retrieve accurate predictions for upcoming events happening on the current day. This feature provides users with valuable insights and helps them make informed decisions about upcoming matches.
		
		 These predictions are generated using advanced algorithms and statistical models that take into account various factors such as team performance, player statistics, historical data, and other relevant parameters."
    
    """
    url = f"https://predictx.p.rapidapi.com/nextpredictions"
    querystring = {'event_date': event_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "predictx.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_today_s_goals_predictions(event_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""Get Today's Goals Predictions" is a powerful feature offered by the predictX API that allows developers to retrieve accurate predictions for events happening on the current day. This feature provides users with valuable insights and helps them make informed decisions about upcoming matches."
    
    """
    url = f"https://predictx.p.rapidapi.com/todaymatchgoals"
    querystring = {'event_date': event_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "predictx.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tomorrow_s_goals_predictions(event_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""Get Tomorrow's Goals Predictions" is a powerful feature offered by the predictX API that allows developers to retrieve accurate predictions for events happening on the current day. This feature provides users with valuable insights and helps them make informed decisions about upcoming matches."
    
    """
    url = f"https://predictx.p.rapidapi.com/tomorrowmatchgoals"
    querystring = {'event_date': event_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "predictx.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

