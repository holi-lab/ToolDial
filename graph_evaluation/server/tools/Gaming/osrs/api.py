import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def personal_hiscores_s_skills_section(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A Get Request to retrieve OSRS's personal hiscores's skills section from source URL: https://secure.runescape.com/m=hiscore_oldschool/overall"
    
    """
    url = f"https://osrs2.p.rapidapi.com/personal-hiscores/skills/{username}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "osrs2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

