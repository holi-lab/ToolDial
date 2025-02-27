import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate(ethnicity: str='european', gender: str='female', age: str='20s', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a face given optional parameters: **gender**, **age**, and **ethnicity**. Ethnicity is an approximation of the corresponding ethnic/racial phenotype for a given option."
    ethnicity: For ethnicity, the following options are available: *european*, *african*, *west_asian*, *south_asian*, *east_asian*, *southeast_asian*, and *latin_american*. If not specified, the ethnic phenotype of the face will be random.
        gender: For gender, the following options are available: *male* and *female*. If not specified, the gender of the face can be either male or female.
        age: For age, the following options are available: *10s*, *20s*, *30s*, *40s*, *50s*, and *60s*. For more fine-grained control, it is also possible to input a specific *numeric* value for age. If age is not specified, then the age of the face will be random.
        
    """
    url = f"https://face-studio.p.rapidapi.com/generate"
    querystring = {}
    if ethnicity:
        querystring['ethnicity'] = ethnicity
    if gender:
        querystring['gender'] = gender
    if age:
        querystring['age'] = age
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "face-studio.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

