import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def postcode(postcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Use UK valid post codes.**
		Simply to use this API you just need to enter the postCode parameter and enter a valid UK post code in capitals and with a space between the 2 parts. For example if the post code you’re looking for is wa41db which is St Mary’s Court in Warrington, you need to enter WA4 1DB with the space between 2 parts. For example “wa41db” would be “WA4 1DB”. 
		**If you enter it any other way you will get an error or not found.**"
    
    """
    url = f"https://uk-street-name-lookup.p.rapidapi.com/"
    querystring = {'postCode': postcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-street-name-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

