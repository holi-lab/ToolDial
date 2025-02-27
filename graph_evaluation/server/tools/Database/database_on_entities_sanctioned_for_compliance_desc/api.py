import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def sanction_data(name_operator: str=None, type: str=None, drop_fields: str=None, source: str=None, date: str=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name_operator: The SQL LIKE operator that can be used in combination with `name`. Possible values for this parameter are `CONTAINS`, `STARTSWITH`, `ENDSWITH`. The value passed is not case sensitive.
        type: The type of sanctioned entry. Possible values are `Individual`, `Entity`.
        drop_fields: Any field(s) that you would like to omit from the returning dataset. If you would like to omit multiple fields, separate them with a comma. The possible values for this parameter are any of the fields found in the "Fields and Descriptions of CSV Data" in the "About" tab.
        source: The source of the sanction data. The possible values for this parameter are `Argentina`, `Australia`, `Canada`, `European Union`, `Ukraine`, `United Kingdom`, `United Nations`, and `United States of America`. The value passed is not case sensitive.
        date: The date that the sanction data file was generated in the format `YYYY-MM-DD`. The lowest possible value is 2022-12-20 and the highest possible value is the date of when the request is being made.
        name: The name of an individual/entity. If the `name_operator` is not passed as a parameter in conjunction with the `name` parameter, the value passed to `name` will be searched through the DESC database for an exact match. The value passed is not case sensitive.
        
    """
    url = f"https://database-on-entities-sanctioned-for-compliance-desc.p.rapidapi.com/sanction-data"
    querystring = {}
    if name_operator:
        querystring['name_operator'] = name_operator
    if type:
        querystring['type'] = type
    if drop_fields:
        querystring['drop_fields'] = drop_fields
    if source:
        querystring['source'] = source
    if date:
        querystring['date'] = date
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "database-on-entities-sanctioned-for-compliance-desc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

