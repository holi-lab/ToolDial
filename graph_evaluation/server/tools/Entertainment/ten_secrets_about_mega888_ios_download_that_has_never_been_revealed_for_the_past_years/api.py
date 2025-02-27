import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def agen_mega_singapore(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ingapore Mega888 we bring you all the latest slot games including 5 Fortune, 7 Crazy, Aladdin Wishes, Sea World, Roulette, God of Wealth, Enchanted Garden, Highway Kings, Boyking’s Treasure, Captains Treasure, Panther Moon, Mystery Dragon etc. With wide range of promotions and bonuses players get a chance to spin the reel for free and win big jackpots."
    
    """
    url = f"https://ten-secrets-about-mega888-ios-download-that-has-never-been-revealed-for-the-past-years.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ten-secrets-about-mega888-ios-download-that-has-never-been-revealed-for-the-past-years.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

