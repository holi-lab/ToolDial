import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def billboard_global_excl_us(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Billboard Global Excl. US. Available from SEPTEMBER 19, 2020"
    date: Set the date for which you want to get a chart.
        
    """
    url = f"https://billboard2.p.rapidapi.com/billboard_global_200_excl_us"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hot_100(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Hot 100 chart information"
    date: Set the date for which you want to get a chart.
        
    """
    url = f"https://billboard2.p.rapidapi.com/hot_100"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artist_100(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Artist 100 chart information."
    date: Set the date for which you want to get a chart.
        
    """
    url = f"https://billboard2.p.rapidapi.com/artist_100"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def billboard_200(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Billboard 100 chart information."
    date: Set the date for which you want to get a chart.
        
    """
    url = f"https://billboard2.p.rapidapi.com/billboard_200"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def billboard_global_200(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide Billboard Global 200 chart information. Available from SEPTEMBER 19, 2020"
    date: Set the date for which you want to get a chart.
        
    """
    url = f"https://billboard2.p.rapidapi.com/billboard_global_200"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

