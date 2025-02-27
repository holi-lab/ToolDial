import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def latest_10_drawings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest, and last winning 10 Fantasy 5 number sets."
    
    """
    url = f"https://fantasy-5.p.rapidapi.com/latest10"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def drawings_between_dates(date1: str, date2: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Fantasy 5 winning number sets between two specified dates where the first number supplied is the beginning date and the second number supplied is the end date."
    date1: Must be a date field in the form of MM-DD-YYYY. Please no slashes \\\"\\\\\\\".
        date2: Must be a date field in the form of MM-DD-YYYY. Please no slashes \\\"\\\\\\\".
        
    """
    url = f"https://fantasy-5.p.rapidapi.com/BetweenDates/{date1}/{date2}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def drawing_by_date(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the winning Fantasy 5 number set from a specific drawing date."
    date: Must be a date field in the form of MM-DD-YYYY. Please no slashes \\\"\\\\\\\".
        
    """
    url = f"https://fantasy-5.p.rapidapi.com/{date}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_quickpick_for_play(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a randomized Quick Pick for the Fantasy 5. This is a random set of playable Fantasy 5 numbers."
    
    """
    url = f"https://fantasy-5.p.rapidapi.com/QuickPick"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_drawing(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest, and last winning Fantasy 5 winning number set."
    
    """
    url = f"https://fantasy-5.p.rapidapi.com/latest"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def general_statistics(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint pulls basic statistics of the Fantasy 5 numbers. For each ball (denoted firstNumber, secondNumber, thirdNumber, fourthNumber, fifthNumber) this endpoint returns the mean, median, mode, min, max, standard deviation, variance, skewness, and kurtosis. After which, the endpoint returns the number of occurrences of each number over the life of game."
    
    """
    url = f"https://fantasy-5.p.rapidapi.com/stats"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_fantasy_5(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all winning drawings in Fantasy 5"
    
    """
    url = f"https://fantasy-5.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

