import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def free_vbucks_generator_how_to_get_free_vbucks(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Free Vbucks Generator is the best way to generate unlimited free fortnite vbucks, It is the oldest and 100% working method for getting free Vbucks. By using this generator you can generate 4000+ free VBUCKS 👉👉👉 https://sptool.xyz/ How to Generate Free Fortnite vbucks 1. Go above the given link 2.type your username.3. Seletect the platform 4. Now select how much Vbucks  you want like 400 to 10000+ 3. Now wait for 5 minutes to complete generate prosses and complete and survey free vbucks  generator free fortnite vbucks hack free vbucks codes free vbucks no human verification free vbucks website free vbucks app free fortnite vbucks for kids free vbucks codes 2022"
    
    """
    url = f"https://free-vbucks-generator-how-to-get-free-vbucks.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "free-vbucks-generator-how-to-get-free-vbucks.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

