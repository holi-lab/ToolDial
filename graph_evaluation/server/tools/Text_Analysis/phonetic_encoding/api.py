import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def encode(alg: str, str: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint accepts the  string to encode (max 100 characters) and the name of the algorithm to use.
		The available algorithms are:
		
		- SOUNDEX,
		- CAVERPHONE,
		- CAVERPHONE2,
		- COLOGNE_PHONETIC,
		- DAITCH_MOKOTOFF_SOUDEX,
		- METAPHONE,
		- METAPHONE2,
		- METAPHONE3,
		- NYSIIS,
		- REFINED_SOUNDEX,
		- MATCH_RATING_APPROACH,
		- BEIDER_MORSE"
    
    """
    url = f"https://phonetic-encoding.p.rapidapi.com/encode"
    querystring = {'alg': alg, 'str': str, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phonetic-encoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

