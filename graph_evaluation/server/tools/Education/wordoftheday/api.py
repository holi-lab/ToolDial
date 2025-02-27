import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_wordoftheday(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/words/{date} // yyyy-MM-dd
		```
		{
		    "date": "2020-03-28",
		    "word": "lampblack",
		    "description": "The fine impalpable soot obtained from the smoke of carbonaceous substances which have been only partly burnt, as in the flame of a smoking lamp. It consists of finely divided carbon, with sometimes a very small proportion of various impurities. It is used as an ingredient of printers' ink, and various black pigments and cements."
		}
		```"
    
    """
    url = f"https://wordoftheday2.p.rapidapi.com/words/{date}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordoftheday2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

