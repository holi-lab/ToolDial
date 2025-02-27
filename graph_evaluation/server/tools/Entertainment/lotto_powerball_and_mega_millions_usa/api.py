import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def between_two_dates(lotto: str, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to search for all records for a given `lotto` between two `dates`."
    
    """
    url = f"https://lotto-powerball-and-mega-millions-usa.p.rapidapi.com/{lotto}/between/{is_from}/{to}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lotto-powerball-and-mega-millions-usa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specific_drawing(date: str, lotto: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint for a given `lotto` and `date` in `MM-YY-YYYY` format to search for the drawing record."
    
    """
    url = f"https://lotto-powerball-and-mega-millions-usa.p.rapidapi.com/{lotto}/date/{date}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lotto-powerball-and-mega-millions-usa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cold(lotto: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to get the coldest numbers for a given `lotto`. The coldest numbers can also be described as the bottom 5 numbers that have been drawn. This request returns a JSON array of objects with the ball number and counts."
    
    """
    url = f"https://lotto-powerball-and-mega-millions-usa.p.rapidapi.com/{lotto}/cold"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lotto-powerball-and-mega-millions-usa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hot(lotto: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to get the hottest numbers for a given `lotto`. The hottest numbers can also be described as the top 5 numbers that have been drawn. This request returns a JSON array of objects with the ball number and counts."
    
    """
    url = f"https://lotto-powerball-and-mega-millions-usa.p.rapidapi.com/{lotto}/hot"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lotto-powerball-and-mega-millions-usa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def counts(lotto: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to receive a JSON object with each ball in the given `lotto` as the key, and the number of times it has been drawn as the value."
    
    """
    url = f"https://lotto-powerball-and-mega-millions-usa.p.rapidapi.com/{lotto}/counts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lotto-powerball-and-mega-millions-usa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest(lotto: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to get the latest drawing record for a given `lotto`."
    
    """
    url = f"https://lotto-powerball-and-mega-millions-usa.p.rapidapi.com/{lotto}/latest"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lotto-powerball-and-mega-millions-usa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all(lotto: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to grab all drawing records for a given `lotto`."
    
    """
    url = f"https://lotto-powerball-and-mega-millions-usa.p.rapidapi.com/{lotto}/all"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lotto-powerball-and-mega-millions-usa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

