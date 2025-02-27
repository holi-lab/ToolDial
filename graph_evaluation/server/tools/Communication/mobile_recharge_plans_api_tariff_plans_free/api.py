import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tariff_plans_api(tariff_plan_api: str, ctrlkey: str, demo: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Just send the operator and circle and GET ALL PLANS according to circle and operator our api"
    tariff_plan_api: Hello First you Get the API from our site nixinfo.in  for CTRLKEY.
        ctrlkey: Enter nixinfo CTRLKEY
        demo: [{"Detail":"140 Local SMS. Maximum 100 SMS per Day","Amount":"13","Validity":"5 Days"},{"Detail":"200 Local and STD SMS. Maximum 100 SMS per Day","Amount":"26","Validity":"14 days"},{"Detail":"340 Local and STD SMS. Maximum 100 SMS per Day","Amount":"34","Validity":"28 days"},{"Detail":"450 Local and STD SMS. Maximum 100 SMS per Day","Amount":"42","Validity":"28 days"},{"Detail":"700 Local and STD SMS. Maximum 100 SMS per Day","Amount":"62","Validity":"28 days"},{"Detail":"1000 Local and STD SMS. Maximum 100 SMS per Day","Amount":"88","Validity":"28 days"},{"Detail":"2400 Local and STD SMS. Maximum 100 SMS per Day","Amount":"218","Validity":"28 days"}]
        
    """
    url = f"https://nixinfo-tariff-plan-code-v1.p.rapidapi.com/Recharge Plan api"
    querystring = {'tariff-plan-api': tariff_plan_api, 'ctrlkey': ctrlkey, 'demo': demo, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nixinfo-tariff-plan-code-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

