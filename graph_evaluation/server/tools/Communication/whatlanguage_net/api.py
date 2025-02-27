import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def detect_language_of_text_string_s_or_url_s(q: str='Is this English or Dutch?', encoding: str='UTF-8', format: str='json or xml', prettyprint: str='true or false', url: str='http://www.facebook.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Detect the language of one or more text strings"
    q: Text from which you want to identify the language. (Please see README - either the q or the url parameter need to be specified)
        encoding: Encoding used to URL encode the text from the q parameter.  If you do not specify an encoding parameter, we will look at the charset of your request. If that is not supplied we will assume you URL encoded your text from the q parameter in UTF-8.
        format: Format of response. Available formats are: json or xml
        prettyprint: Returns a human readable response (pretty printed) with indentations and line breaks when set to true. Available values are: true or false
        url: URL from which you want to identify the language. URL can start with http://, https:// or ftp:// . (Please see README - either the q or the url parameter need to be specified)
        
    """
    url = f"https://whatlanguage.p.rapidapi.com/detect"
    querystring = {}
    if q:
        querystring['q'] = q
    if encoding:
        querystring['encoding'] = encoding
    if format:
        querystring['format'] = format
    if prettyprint:
        querystring['prettyprint'] = prettyprint
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatlanguage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

