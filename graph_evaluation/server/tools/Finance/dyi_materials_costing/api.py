import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def test_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use to Test connection"
    
    """
    url = f"https://dyi-materials-costing.p.rapidapi.com/testAPI"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dyi-materials-costing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def diy_materials_costing(costof: str='Wooden Chair', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parameter:  Kite
		
		[Example Result]
		Materials Needed:
		-Kite frame (wooden or plastic dowel rods) - $3-5
		-Kite string - $3-5
		-Kite fabric (ripstop nylon, polyester, or silk) - $10-20
		-Kite tail (ribbon, plastic strips, or fabric) - $1-5
		-Kite bridle (string or cord) - $3-5
		-Kite handles (plastic, wood, or foam) - $3-5
		-kite line (cotton, polyester, or kevlar) - $5-15
		
		Total estimated cost: $30-65"
    
    """
    url = f"https://dyi-materials-costing.p.rapidapi.com/Query"
    querystring = {}
    if costof:
        querystring['costof'] = costof
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dyi-materials-costing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

