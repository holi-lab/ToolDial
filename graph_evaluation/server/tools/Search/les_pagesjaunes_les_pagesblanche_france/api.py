import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def pagesblanches(pbpage: int, pbkeyword: str, pblocation: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extraire les données du Pages Blanches scraping email et téléphone"
    
    """
    url = f"https://les-pagesjaunes-les-pagesblanche-france.p.rapidapi.com/"
    querystring = {'pbpage': pbpage, 'pbkeyword': pbkeyword, 'pblocation': pblocation, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "les-pagesjaunes-les-pagesblanche-france.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pagesjaunes(pjpage: int, pjlocation: str, pjkeyword: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extraire les données du pages jaunes email et téléphone"
    
    """
    url = f"https://les-pagesjaunes-les-pagesblanche-france.p.rapidapi.com/"
    querystring = {'pjpage': pjpage, 'pjlocation': pjlocation, 'pjkeyword': pjkeyword, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "les-pagesjaunes-les-pagesblanche-france.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

