import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def samyutam(name_rahul_gender_male_probability_1_00_count_5_country_id_india_language_id_ind: str, name_chris_gender_male_description_the_name_chris_is_mostly_male: str='Rahul', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rahul Name is Almost male"
    name_chris_gender_male_description_the_name_chris_is_mostly_male: Rahul
        
    """
    url = f"https://rahulsamyutam-samyutam-eduction-v1.p.rapidapi.com// {name_rahul_gender_male_probability_1_00_count_5_country_id_india_language_id_ind}"
    querystring = {}
    if name_chris_gender_male_description_the_name_chris_is_mostly_male:
        querystring['name-chris-gender-male-description-the-name-chris-is-mostly-male'] = name_chris_gender_male_description_the_name_chris_is_mostly_male
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rahulsamyutam-samyutam-eduction-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

