import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def malnutrition(growth_ref: str, weight_in_kg: int, height_in_cm: int, sex_f_or_m: str, dateofbirth: str, dateofmeasure: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Malnutrition score and anthropometric indicators"
    growth_ref: Growth references, value are who or flemish
        weight_in_kg: Weight in kg Body weight Measured in kg http://s.details.loinc.org/LOINC/3141-9.html?sections=Comprehensive
        height_in_cm: Body height Measured in cm http://r.details.loinc.org/LOINC/3137-7.html?sections=Comprehensive
        sex_f_or_m: Sex is F or M http://hl7.org/fhir/valueset-administrative-gender.html
        dateofbirth: Birth date as YYYY-MM-DD
        dateofmeasure: Mesure date as YYYY-MM-DD
        
    """
    url = f"https://emmanuelchevallier-malnutrition-v1.p.rapidapi.com/malnutrition"
    querystring = {'growth_ref': growth_ref, 'weight_in_kg': weight_in_kg, 'height_in_cm': height_in_cm, 'sex_F_or_M': sex_f_or_m, 'dateofbirth': dateofbirth, }
    if dateofmeasure:
        querystring['dateofmeasure'] = dateofmeasure
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emmanuelchevallier-malnutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

