import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def video_games_with_deadliest_outbreak_settings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Video Games with the Deadliest Outbreak Settings"
    
    """
    url = f"https://video-games-with-deadliest-outbreak-settings.p.rapidapi.com/ydWrS4/videogameswithdeadliestoutbreaksettings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "video-games-with-deadliest-outbreak-settings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

