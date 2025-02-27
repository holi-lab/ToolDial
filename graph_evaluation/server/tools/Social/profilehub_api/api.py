import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def user_data(numuser: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves user data for a specific user ID. This endpoint returns information such as the user's first name, last name, username, email address, posts made by the user, images posted by the user, avatar image URL, comments made by the user, and posts liked by the user."
    
    """
    url = f"https://profilehub-api.p.rapidapi.com/api/users/{numuser}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "profilehub-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

