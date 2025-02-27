import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def date_range(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show Date Range from date todate"
    
    """
    url = f"https://samepoint.p.rapidapi.com/q/obama/fromdate/2013-5-1/todate/2013-6-1"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "samepoint.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def positive_senitment(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endoint to return only positve responses"
    
    """
    url = f"https://samepoint.p.rapidapi.com/q/president/sentiment/positve"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "samepoint.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_count_of_a_given_query(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Count is an output format that only returns the total count of results from the query. Use this for pagination bounds."
    
    """
    url = f"https://samepoint.p.rapidapi.com/q/obama/output/count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "samepoint.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pagination(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pagination: Start Offset =0 Max returned = 25  Pagination: The API outputs 25 Records at a time. In the response header, there is a numfound datapoint. Use numfound to control your bounds when you are looping through the data.  Start / Offset Switch/ Pagination start=0   • /q/president/format/json/start/0  • /q/president/format/json/start/25  • /q/president/format/json/start/50"
    
    """
    url = f"https://samepoint.p.rapidapi.com/q/obama/start/0"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "samepoint.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def show_abridged_content_summary(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show just a summary where the keywords are."
    
    """
    url = f"https://samepoint.p.rapidapi.com/q/obama/abridged/true"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "samepoint.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def negative_sentiment(sentiment: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    sentiment: Return Negative Sentiment
        
    """
    url = f"https://samepoint.p.rapidapi.com/q/obama/sentiment/negative"
    querystring = {'sentiment': sentiment, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "samepoint.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

