import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def planet_currently_visible(long: str, lat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Planet currently visible"
    
    """
    url = f"https://ouranos3.p.rapidapi.com/planetvisible"
    querystring = {'long': long, 'lat': lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ouranos3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def predict_feature_forecast_7_day(long: str=None, lat: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Predict Forecast returns 
		- Binary predict value, 1 if it’s a good night to observe and 0 if it’s not. This value is calculated according to the forecast for the night.
		- Rating, score out of 5.
		- Tips for astronomers based on the forecast."
    
    """
    url = f"https://ouranos3.p.rapidapi.com/predict7day"
    querystring = {}
    if long:
        querystring['long'] = long
    if lat:
        querystring['lat'] = lat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ouranos3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def predict_feature_forecast_1_day(lat: str, long: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Predict Forecast returns 
		- Binary predict value, 1 if it’s a good night to observe and 0 if it’s not. This value is calculated according to the forecast for the night.
		- Rating, score out of 5.
		- Tips for astronomers based on the forecast."
    
    """
    url = f"https://ouranos3.p.rapidapi.com/predict1day"
    querystring = {'lat': lat, 'long': long, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ouranos3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_7_day_forecast(lat: str, long: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "7 Day Forecast return seeing value in arc seconds and transparency on a scale of 1 for the next 7 day every 3 hours"
    
    """
    url = f"https://ouranos3.p.rapidapi.com/7day"
    querystring = {'lat': lat, 'long': long, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ouranos3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_3_day_forecast(long: str, lat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "3 Day Forecast return seeing value in arc seconds and transparency on a scale of 1 for the next 3 day every 3 hours"
    
    """
    url = f"https://ouranos3.p.rapidapi.com/3day"
    querystring = {'long': long, 'lat': lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ouranos3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def moon_information(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Moon Information return moon phase and illumination."
    
    """
    url = f"https://ouranos3.p.rapidapi.com/moon"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ouranos3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_48_hour_forecast(long: str, lat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hourly Forecast return seeing value in arc seconds and transparency on a scale of 1 for the next 48 hours."
    
    """
    url = f"https://ouranos3.p.rapidapi.com/hourly"
    querystring = {'long': long, 'lat': lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ouranos3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

