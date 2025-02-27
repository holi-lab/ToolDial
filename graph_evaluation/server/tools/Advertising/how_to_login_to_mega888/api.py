import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def mega888_malaysia(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Reputed Mega888 Malaysia Agent is one of the most popular online casinos to make money while playing games. It is very easy to use the platform and play Mega888 Slot Game. At Agen Mega888 website you’ll experience so much fun and entertainment that is why it’s nearly impossible to leave. With a wide range of product lines we also provide 24/7 customer support for Malaysian Players. Don’t think too much and join us now to receive huge welcome bonuses and cashback rebates on your losses!"
    
    """
    url = f"https://how-to-login-to-mega888.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "how-to-login-to-mega888.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

