import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def samyutam(name_aaron_gender_male_description_the_name_aaron_is_male: str='Aaron', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "peter name is male most"
    name_aaron_gender_male_description_the_name_aaron_is_male: Aaron Name is mostly male
        
    """
    url = f"https://rahulhimt89-nitians-v1.p.rapidapi.com//curl --get --include "https://montanaflynn-gender-guesser.p.mashape.com/?name=puneeta" \   -H "X-Mashape-Key: hDsKMXyFJDmshYAKeml52DLavT90p1z3W2fjsnPtyjQVXfI6Dc" \   -H "Accept: application/json""
    querystring = {}
    if name_aaron_gender_male_description_the_name_aaron_is_male:
        querystring['name-aaron-gender-male-description-the-name-aaron-is-male'] = name_aaron_gender_male_description_the_name_aaron_is_male
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rahulhimt89-nitians-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

