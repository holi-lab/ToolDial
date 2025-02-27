import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_job_stories(limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this api you can get the detailed information of job stories on hackernews."
    limit: Limit can't be more than 200
        
    """
    url = f"https://go-hackernews.p.rapidapi.com/api/v1/jobstories/{limit}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "go-hackernews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_show_stories(limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this api you can get the detailed information of show stories on hackernews."
    limit: Limit can't be more than 200
        
    """
    url = f"https://go-hackernews.p.rapidapi.com/api/v1/showstories/{limit}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "go-hackernews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_ask_stories(limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this api you can get the detailed information of ask stories on hackernews."
    limit: Limit can't be more than 200.
        
    """
    url = f"https://go-hackernews.p.rapidapi.com/api/v1/askstories/{limit}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "go-hackernews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_best_stories(limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this api you can get the detailed information of best stories on hackernews."
    limit: Limit can't be more than 500
        
    """
    url = f"https://go-hackernews.p.rapidapi.com/api/v1/beststories/{limit}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "go-hackernews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_new_stories(limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using new stories you can get hackernews new posts."
    limit: Limit is required and your limit can't be more than 500
        
    """
    url = f"https://go-hackernews.p.rapidapi.com/api/v1/newstories/{limit}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "go-hackernews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_stories(limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this api you can get the detailed information of top stories on hackernews."
    limit: The default limit set by official api of hackernews is 500 so you can't pass limit more than 500.
        
    """
    url = f"https://go-hackernews.p.rapidapi.com/api/v1/topstories/{limit}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "go-hackernews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

