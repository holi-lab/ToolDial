import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def online_casino_malaysia(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Malaysia Online Casino](https://onlinemalaysiacasino.com/
		) is the top rated casino in Malaysia! Here you  can play hundreds of Online Malaysia Casino games. All the Slots, sportsbook and Live Casino table games offered on the website are of high quality and the payout ratio is also very high. Get Register with us and enjoy gambling with real money!"
    
    """
    url = f"https://top-online-live-casino-malaysia-games.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "top-online-live-casino-malaysia-games.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

