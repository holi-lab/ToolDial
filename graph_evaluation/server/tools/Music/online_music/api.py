import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def online_music(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Слушать бесплатно онлайн на online-music.su. Лучшие песни русских ретро групп и исполнителей. Руccкая ретро поп-рок-шансон музыка 1970х-2000х, лучшая забугорная музыка ретро эпохи."
    
    """
    url = f"https://online-music.p.rapidapi.com/online-music.su"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-music.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

