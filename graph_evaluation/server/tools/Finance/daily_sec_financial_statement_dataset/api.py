import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def premium_daily(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoints delivers the daily zip files of filed 10-q and 10-k reports. The files are published daily (resp. once or twice a week during the beta phase).
		
		The structure of the data is similar as in the SEC financial statement dataset (https://www.sec.gov/files/aqfs.pdf), with the exception  that not TAG file is present and that the SUB file only contains a subset of the attributes.
		
		Please not that the https-url has to end with a "/": 
		https://daily-sec-financial-statement-dataset.p.rapidapi.com/premium/day/2023-07-07**/**"
    
    """
    url = f"https://daily-sec-financial-statement-dataset.p.rapidapi.com/premium/day/{date}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daily-sec-financial-statement-dataset.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basic_daily(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoints delivers the daily zip files of filed 10-q and 10-k reports. With the basic subcription, you only have access to daily zip files older than 21 days.
		
		The structure of the data is similar as in the SEC financial statement dataset (https://www.sec.gov/files/aqfs.pdf), with the exception  that not TAG file is present and that the SUB file only contains a subset of the attributes.
		
		Please not that the https-url has to end with a "/": 
		https://daily-sec-financial-statement-dataset.p.rapidapi.com/basic/day/2023-07-07**/**"
    
    """
    url = f"https://daily-sec-financial-statement-dataset.p.rapidapi.com/basic/day/{date}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daily-sec-financial-statement-dataset.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def content(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint delivers the a json that shows which dates are available with which subscription. The format is as follows:
		
		```
		{
		  "daily": [
		  {  
		    "date": "2022-01-03",
		    "file": "20220103.zip",
		    "subscription": "basic"
		  },
		...
		  {  
		    "date": "2022-12-02",
		    "file": "20221202.zip",
		    "subscription": "premium"
		  },
		
		```
		Entries marked with "subscription: premium" are only accessible with the premium plan. All other entries are available with basic and premium plan"
    
    """
    url = f"https://daily-sec-financial-statement-dataset.p.rapidapi.com/content/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daily-sec-financial-statement-dataset.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def heartbeat(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A simple heartbeat that returns the time of the server to check if the connection to the server is working."
    
    """
    url = f"https://daily-sec-financial-statement-dataset.p.rapidapi.com/heartbeat/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daily-sec-financial-statement-dataset.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

