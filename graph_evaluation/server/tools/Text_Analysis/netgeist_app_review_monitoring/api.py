import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ping(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Check if server is still alive**
		
		Args:
		* _name_: Your name
		
		Returns:
		* A greeting"
    
    """
    url = f"https://netgeist-app-review-monitoring.p.rapidapi.com/ping"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netgeist-app-review-monitoring.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def app_metadata(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Get meta data of the app**
		
		Args:
		* _None_
		
		Returns:
		* Metadata of required app"
    
    """
    url = f"https://netgeist-app-review-monitoring.p.rapidapi.com/app/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netgeist-app-review-monitoring.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def app_topics(datefrom: str, dateto: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Get topics of an app sorted by count of reviews**
		
		Args:
		* _dateFrom_: from which date to filter (format: yyyy-MM-ddTHH:mm:ss)
		* _dateTo_: to which date to filter (format: yyyy-MM-ddTHH:mm:ss)
		
		Returns:
		* Topics of an app sorted by count of reviews"
    
    """
    url = f"https://netgeist-app-review-monitoring.p.rapidapi.com/app/{is_id}/topics"
    querystring = {'dateFrom': datefrom, 'dateTo': dateto, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netgeist-app-review-monitoring.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def app_review_metrics(is_id: str, dateto: str, datefrom: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Get daily aggregated metrics of app reviews**
		
		Args:
		* _dateFrom_: from which date to filter (format: yyyy-MM-ddTHH:mm:ss)
		* _dateTo_: to which date to filter (format: yyyy-MM-ddTHH:mm:ss)
		
		Returns:
		* Daily aggregated metrics of app reviews"
    
    """
    url = f"https://netgeist-app-review-monitoring.p.rapidapi.com/app/{is_id}/review-metrics"
    querystring = {'dateTo': dateto, 'dateFrom': datefrom, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netgeist-app-review-monitoring.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_apps(appsubstring: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Get apps that are similar to a given app-name**
		
		Args:
		* _appSubstring_: Search apps by name with a given string
		
		Returns:
		* Apps that contain appSubstring"
    
    """
    url = f"https://netgeist-app-review-monitoring.p.rapidapi.com/query-apps"
    querystring = {'appSubstring': appsubstring, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netgeist-app-review-monitoring.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def app_reviews(datefrom: str, is_id: str, dateto: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Get reviews of a given app sorted by @timestamp**
		
		Args:
		* _dateFrom_: from which date to filter (format: yyyy-MM-ddTHH:mm:ss)
		* _dateTo_: to which date to filter (format: yyyy-MM-ddTHH:mm:ss)
		
		Returns:
		* Reviews of a given app sorted by @timestamp"
    
    """
    url = f"https://netgeist-app-review-monitoring.p.rapidapi.com/app/{is_id}/reviews"
    querystring = {'dateFrom': datefrom, 'dateTo': dateto, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netgeist-app-review-monitoring.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

