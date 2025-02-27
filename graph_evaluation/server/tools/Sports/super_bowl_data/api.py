import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def allsuperbowldata(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "1) All winning and losing teams from 1967 until 2023
		2) All event venues, cities and states where games occurred from 1967 until 2023
		3) All winning and losing team coaches from 1967 until 2023
		4) All winning and losing teams' scores from 1967 until 2023"
    
    """
    url = f"https://super-bowl-data.p.rapidapi.com/AllSuperBowlData"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "super-bowl-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

