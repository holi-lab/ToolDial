import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def languages(api_version: str, accept_language: str=None, x_clienttraceid: str=None, scope: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the set of languages currently supported by other operations of the Translator Text API."
    api_version: Version of the API requested by the client. Value must be **3.0**.
        accept_language: The language to use for user interface strings. Some of the fields in the response are names of languages or names of regions. Use this parameter to define the language in which these names are returned. The language is specified by providing a well-formed BCP 47 language tag. For instance, use the value `fr` to request names in French or use the value `zh-Hant` to request names in Chinese Traditional. Names are provided in the English language when a target language is not specified or when localization is not available.
        x_clienttraceid: A client-generated GUID to uniquely identify the request. Note that you can omit this header if you include the trace ID in the query string using a query parameter named ClientTraceId.
        scope: A comma-separated list of names defining the group of languages to return. Allowed group names are- `translation`, `transliteration` and `dictionary`. If no scope is given, then all groups are returned, which is equivalent to passing `scope=translation,transliteration,dictionary`. To decide which set of supported languages is appropriate for your scenario, see the description of the response object.
        
    """
    url = f"https://microsoft-translator-text.p.rapidapi.com/languages"
    querystring = {'api-version': api_version, }
    if accept_language:
        querystring['Accept-Language'] = accept_language
    if x_clienttraceid:
        querystring['X-ClientTraceId'] = x_clienttraceid
    if scope:
        querystring['scope'] = scope
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "microsoft-translator-text.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

