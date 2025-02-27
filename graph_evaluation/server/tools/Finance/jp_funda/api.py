import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getlistbyedinetcode(edinet_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### Return all securities report data of the company specified with edinet code.
		Note: If the company doesn't have a consolidated subsidiary, all of consolidated management indicators become null.　But you can obtain data instead of consolidated management indicators from non consolidated management indicators"
    
    """
    url = f"https://jp-funda.p.rapidapi.com/edinet_code/list/{edinet_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jp-funda.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlistbysecuritiescode(securities_code: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### Return all of securities report data of the company specified with securities code.
		Note: If  the company doesn't  have a consolidated subsidiary,  all of consolidated management indicators become null.　But you can obtain data instead of consolidated management indicators from non consolidated management indicators"
    
    """
    url = f"https://jp-funda.p.rapidapi.com/securities_code/list/{securities_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jp-funda.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlatestbysecuritiescode(securities_code: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### Return latest securities report data of the company specified with securities code.
		Note: If  the company doesn't  have a consolidated subsidiary,  all of consolidated management indicators become null.　But you can obtain data instead of consolidated management indicators from non consolidated management indicators"
    
    """
    url = f"https://jp-funda.p.rapidapi.com/securities_code/{securities_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jp-funda.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlatestbyedinetcode(edinet_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### Return latest securities report data of the company specified with edinet code.
		Note: If  the company doesn't  have a consolidated subsidiary,  all of consolidated management indicators become null.　But you can obtain data instead of consolidated management indicators from non consolidated management indicators"
    
    """
    url = f"https://jp-funda.p.rapidapi.com/edinet_code/{edinet_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jp-funda.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdisclosedtoday(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### Return securities report data disclosed today
		Note: If the company doesn't have a consolidated subsidiary, all of consolidated management indicators become null.　But you can obtain data instead of consolidated management indicators from non consolidated management indicators"
    
    """
    url = f"https://jp-funda.p.rapidapi.com/today/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jp-funda.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdisclosedyesterday(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### Return securities report data disclosed yesterday
		Note: If the company doesn't have a consolidated subsidiary, all of consolidated management indicators become null.　But you can obtain data instead of consolidated management indicators from non consolidated management indicators"
    
    """
    url = f"https://jp-funda.p.rapidapi.com/yesterday/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jp-funda.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdisclosedweek(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### Return securities report data disclosed within this week include today.
		Note: If the company doesn't have a consolidated subsidiary, all of consolidated management indicators become null.　But you can obtain data instead of consolidated management indicators from non consolidated management indicators"
    
    """
    url = f"https://jp-funda.p.rapidapi.com/week/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jp-funda.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdiscloseddaterange(start_date: str, end_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### Return securities report data Specified by the date range.
		Note: This endpoint requires the 2 query parameters, start_date and end_date
		Note: max date range is 31 days. if you want longer date range data, need to requesting API twice or more."
    
    """
    url = f"https://jp-funda.p.rapidapi.com/date_range/?start_date&end_date"
    querystring = {'start_date': start_date, 'end_date': end_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jp-funda.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

