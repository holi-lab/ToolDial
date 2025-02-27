import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def music_lovers(music_lovers: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "recibe datos"
    music_lovers: Red Musical para Amateurs
        
    """
    url = f"https://69bd0c7193msh3c4abb39db3a82fp18c336jsn8470910ae9f0.p.rapidapi.com/radio"
    querystring = {'music_lovers': music_lovers, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "69bd0c7193msh3c4abb39db3a82fp18c336jsn8470910ae9f0.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

