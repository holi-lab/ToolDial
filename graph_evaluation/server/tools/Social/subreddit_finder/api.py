import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def subreddit_finder(topic: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Subreddit finder api:
		Given a topic, return a list of subreddits.
		User friendly interactive demo: 
		[https://segue.co/subreddit-finder/](https://segue.co/subreddit-finder/)"
    
    """
    url = f"https://subreddit-finder.p.rapidapi.com/"
    querystring = {'topic': topic, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "subreddit-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

