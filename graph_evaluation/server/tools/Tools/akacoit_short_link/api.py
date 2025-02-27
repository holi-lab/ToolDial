import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def deletelink(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# /v1/link/delete/:id
		
		Deletes your link, given his ID. Doesn't remove the analytics.
		
		| **Parameter** | **Values** |
		| --- | --- |
		| ID | *String:* ID of your link |
		| full** | *Boolean:* Deletes all data connected to the link (analytics included) |
		
		**TBD"
    
    """
    url = f"https://akacoit-short-link.p.rapidapi.com/v1/link/delete/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "akacoit-short-link.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlinksuser(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# /v1/link/list
		
		Retrieves all your links data.
		
		| **Parameter** | **Values** |
		| --- | --- |
		| deep: **Optional** | *Number*: Can be [1,2]; is how many level of sublinks to show, if any. Default: 1. |
		| page: **Optional** | *Number*: To select which page of results to retrieve, if more than 1. Default: 1. |"
    
    """
    url = f"https://akacoit-short-link.p.rapidapi.com/v1/link/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "akacoit-short-link.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlink(is_id: str, deep: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# /v1/link/get/:id
		
		Retrieves the data about your link, given his id name.
		
		| **Parameter** | **Values** |
		| --- | --- |
		| id | *String:* id of your link |
		| deep: **Optional** | *Number*: Can be [1,2]; is how many level of sublinks to show, if any. Default: 1. |
		| page: **Optional** | *Number*: To select which page of results to retrieve, if more than 1. Default: 1. |"
    deep: Level of deepness of sublinks
        
    """
    url = f"https://akacoit-short-link.p.rapidapi.com/v1/link/get/{is_id}"
    querystring = {'deep': deep, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "akacoit-short-link.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getanalyticscount(is_id: str, download: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# /**v1/analytics/count/:id**
		
		Retrieves analytics of your link, given his ID, split by hour sampled every quarter.  
		Available only for payment plans.
		
		| **Parameter** | **Values** |
		| --- | --- |
		| ID | *String:* ID of your link |"
    download: If true downloads a copy of the results as csv
        
    """
    url = f"https://akacoit-short-link.p.rapidapi.com/v1/analytics/count/{is_id}"
    querystring = {'download': download, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "akacoit-short-link.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

