import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_the_latest_technologies_list_and_details(latest_projects: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This end points all the latest project techonoliges how much they are being use all over the world and what do they actually do."
    
    """
    url = f"https://all-technology-projects.p.rapidapi.com/latest-projects"
    querystring = {}
    if latest_projects:
        querystring['latest-projects'] = latest_projects
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-technology-projects.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

