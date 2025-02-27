import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_popular_football_matches_predictions_insights_updated_every_day_at_10_00_14_00_gmt(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Keep a pulse on the most anticipated football matches with this endpoint. Updated daily between 10:00-14:00 GMT, it delivers timely and comprehensive predictions and insights for popular football matches. Empower your betting decisions with data-rich forecasts and crucial insights, ensuring you're always prepared for the game ahead. Stay informed and enhance your betting strategy with this indispensable resource."
    
    """
    url = f"https://football-betting-prediction-api.p.rapidapi.com/popular_predictions/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-betting-prediction-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_predictions_for_today_updated_every_day_at_00_00_05_00_gmt(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint is specifically designed to retrieve today's football match predictions. By leveraging real-time data and sophisticated predictive algorithms, it provides the latest forecasts for match outcomes happening on the current day.
		
		The 'Get Predictions for Today' endpoint delivers essential data such as predicted winner, expected goals, potential scorers, and other valuable betting information. This immediate, data-driven insight enables bettors to make informed decisions and strategies for today's matches.
		
		Easy to integrate and use, this endpoint serves as a crucial tool for any football betting platform, enhancing user engagement and betting success rate by providing accurate and timely predictions for today's football matches."
    
    """
    url = f"https://football-betting-prediction-api.p.rapidapi.com/predictions_rapidapi/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-betting-prediction-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tickets_for_today_updated_every_day_at_00_00_05_00_gmt(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Take the 3 (Gold, Silver or Bronze) tickets and be a winner."
    
    """
    url = f"https://football-betting-prediction-api.p.rapidapi.com/tickets/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-betting-prediction-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

