import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getrandomholiday(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An endpoint of an API that returns random Polish unusual holiday would provide information on the lesser-known or unique holiday that is being celebrated on a random date in Poland. This endpoint would likely be accessible via a GET request and would return data in a structured format, such as JSON."
    
    """
    url = f"https://polish-unusual-holidays.p.rapidapi.com/random"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "polish-unusual-holidays.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettodaysholiday(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An endpoint of an API that returns today's Polish unusual holiday would provide information on the lesser-known or unique holiday that is being celebrated on the current date in Poland. This endpoint would likely be accessible via a GET request and would return data in a structured format, such as JSON."
    
    """
    url = f"https://polish-unusual-holidays.p.rapidapi.com/today"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "polish-unusual-holidays.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallholidays(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An endpoint of an API that returns all the Polish unusual holidays would provide a comprehensive list of lesser-known or unique holidays celebrated in Poland. This endpoint would likely be accessible via a GET request and would return data in a structured format, such as JSON."
    
    """
    url = f"https://polish-unusual-holidays.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "polish-unusual-holidays.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

