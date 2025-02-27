import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def previous_posts_with_date_filter(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "1. Your URL should look like this if you are filtering the posted data by date: -
		
		https://catktorapi.onrender.com/previousPosts/{DATE}
		
		2. Replace **{DATE}** with the date you want to retrieve data from.
		
		>  As an example, retriving the data from the date October 26, 2022 looks like this:-
		
		https://catktorapi.onrender.com/previousPosts/26OCTOBER2022"
    
    """
    url = f"https://cat-api5.p.rapidapi.com/previousPosts/{date}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cat-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def previous_posts(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the previously posted data by the bot in JSON format."
    
    """
    url = f"https://cat-api5.p.rapidapi.com/previousPosts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cat-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

