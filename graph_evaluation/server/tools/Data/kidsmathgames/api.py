import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fun_math_games_for_kids(ligiaozimek: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Math games are a great opportunity to teach kids math skills at an early age. The acquaintance of kids with mathematics should take place in an exciting and colorful atmosphere that can interest anyone, even the most restless child."
    
    """
    url = f"https://kidsmathgames.p.rapidapi.com/"
    querystring = {}
    if ligiaozimek:
        querystring['LigiaOzimek'] = ligiaozimek
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kidsmathgames.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

