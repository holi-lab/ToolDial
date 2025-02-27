import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fix_pay_simulation(months: str, principal: str, rate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fix Pay Simulation. Parameters are /principal/rate/months where principal is the amount of loan, rate is the interest and months the number of monthly payments. The output will be a json containing the monthly ammount to be paid and an array with the beginning balance, interest, capital (principal) and ending balance"
    
    """
    url = f"https://loansimulator1.p.rapidapi.com/simulation/fix_pay/{principal}/{rate}/{months}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loansimulator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

