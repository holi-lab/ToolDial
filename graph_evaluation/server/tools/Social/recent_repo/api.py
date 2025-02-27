import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def github_profile_repo(user: str, repo: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetches the repositories of a particular profile.
		
		Query parameter repo is optional whereas user is required."
    
    """
    url = f"https://recent-repo.p.rapidapi.com/repos"
    querystring = {'user': user, }
    if repo:
        querystring['repo'] = repo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "recent-repo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

