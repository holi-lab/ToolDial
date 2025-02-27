import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def famous_popular_fictional_characters_from_kenya_kenyan_descent(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Most Famous & Popular Fictional Characters from Kenya Kenyan Descent"
    
    """
    url = f"https://famous-popular-fictional-characters-from-kenya-kenyan-descent.p.rapidapi.com/bO7QaS/famouspopularfictionalcharacterskenyakenyandescent"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "famous-popular-fictional-characters-from-kenya-kenyan-descent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

