import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getmostrecentday(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches the most recent Trinidad & Tobago Covid 19 statistics.  It returns data for that specific day alone."
    
    """
    url = f"https://trinidad-covid-19-statistics.p.rapidapi.com/latestday"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trinidad-covid-19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstatsbyday(yearid: str, monthid: str, dayid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches Trinidad & Tobago Covid 19 statistics by  day. For a valid response, the months are input as "01" or "12" and not the actual month name such as "January" or "December". Th e days are entered in the similar format as months ie "01" -"31" and not actual days such as "Thursday" or "Monday","
    
    """
    url = f"https://trinidad-covid-19-statistics.p.rapidapi.com/year/{yearid}/month/{monthid}/day/{dayid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trinidad-covid-19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstatsbyyear(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint delivers Trinidad & Tobago Covid 19 statistics by year. For a valid response, the beginning year is 2020 and the latest year is the current year."
    
    """
    url = f"https://trinidad-covid-19-statistics.p.rapidapi.com/year/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trinidad-covid-19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstatsbymonth(yearid: str, monthid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches Trinidad & Tobago Covid 19 statistics by month. For a valid response, the months are input as "01" or "12" and not the actual month name such as "January" or "December"."
    
    """
    url = f"https://trinidad-covid-19-statistics.p.rapidapi.com/year/{yearid}/month/{monthid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trinidad-covid-19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

