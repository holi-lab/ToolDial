import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def diacritizator(q: str, lang: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Adds diacritical marks to the text (provided with `q` query parameter) for the language given in `lang` path parameter."
    lang: Currently supported values for the `lang` path parameter are:

- `cs` (Czech)
- `pl` (Polish)
        
    """
    url = f"https://linguatools-diacritizator.p.rapidapi.com/diacritizator/{lang}"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linguatools-diacritizator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

