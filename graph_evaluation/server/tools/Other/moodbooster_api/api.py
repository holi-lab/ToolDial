import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def moodbooster_api(file: str='https://frenzvalios.tk/moodbooster/secure/ULAUKf3PAUpxN6f9FegfK6JptXFwUFrSWy4h52EKW35QsqVzKaYY9KLW2GNN9U4D9fZ8T/ALCvdUHaFxAuquDMVpxAMR8W7AfbnnkFtx5gKUKATWN3R6VAw8H5gJYbaLtNMNeG58DB2/243tK5LEBtANjsxF3n2Z4G4b7UKzvctedvADnQ2DNz3BXWCXX3nNehgLFfLQmLyx3CGja/14255-4413-15793.jpg', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "random girls xdd"
    
    """
    url = f"https://moodbooster-api.p.rapidapi.com/moodbooster-api.php"
    querystring = {}
    if file:
        querystring['file'] = file
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moodbooster-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

