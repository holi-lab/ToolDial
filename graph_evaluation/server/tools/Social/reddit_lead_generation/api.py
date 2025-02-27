import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reddit_lead_generation(topic: str, subredditnames: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reddit Lead Generation by topic and subreddits"
    
    """
    url = f"https://reddit-lead-generation.p.rapidapi.com/"
    querystring = {'topic': topic, 'subredditNames': subredditnames, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddit-lead-generation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

