import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bible_advise_question(question: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API gives access to specific bible verses, advise on life-situations based on bible verses, anything about the bible. You just have to put a string in description of what it is you want about/from the bible"
    
    """
    url = f"https://bible-verses-api-bible-advise-sermons.p.rapidapi.com/bible_advise/{question}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bible-verses-api-bible-advise-sermons.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

