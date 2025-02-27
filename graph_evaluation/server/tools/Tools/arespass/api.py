import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def about(outputformat: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    outputformat: **The format of the returned metadata.**

Allowed values are *json*, *xml* and *yaml*.

The default value is *xml*.

        
    """
    url = f"https://arespass.p.rapidapi.com/about"
    querystring = {}
    if outputformat:
        querystring['outputFormat'] = outputformat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "arespass.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def about(outputformat: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    outputformat: **The format of the returned metadata.**

Allowed values are *json*, *xml* and *yaml*.

The default value is *xml*.

        
    """
    url = f"https://arespass.p.rapidapi.com/about"
    querystring = {}
    if outputformat:
        querystring['outputFormat'] = outputformat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "arespass.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ec(password: str, penalty: int=None, outputformat: str=None, reqid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    password: **The password to be analyzed.**

Minimum length is 4 characters; maximum length is 128 characters.

Beware that certain characters like '&#35;', '&#61;' or '&#63;' must be properly encoded.

For more information about this issue, please refer to RFC 3986 ("*Uniform Resource Identifier (URI): Generic Syntax*"), sections 2.1, 2.2 and 2.4.

        penalty: **The penalty applied to each character that is part of a word, number sequence, alphabet sequence, etc.**

The penalty is a float number in the range [0, 1]. Full penalty, 0; no penalty, 1.

The character used as decimal separator is always '&#46;'. Hence, a parameter value like *0,33* would be illegal.

The default value is *0.25*.

        outputformat: **The format of the returned analysis.**

Allowed values are *json*, *xml* and *yaml*.

The default value is *xml*.

        reqid: **An identifier for this request.**

The request identifier is a string that must match the regular expression */(?i)^[a-z0-9]{8,16}$/*.

This identifier is echoed in the returned response. Its value has no effect on the password analysis.

If this parameter is unset, a randomly generated identifier will be automatically assigned to this request.

        
    """
    url = f"https://arespass.p.rapidapi.com/ec"
    querystring = {'password': password, }
    if penalty:
        querystring['penalty'] = penalty
    if outputformat:
        querystring['outputFormat'] = outputformat
    if reqid:
        querystring['reqId'] = reqid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "arespass.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ec(password: str, penalty: int=None, outputformat: str=None, reqid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    password: **The password to be analyzed.**

Minimum length is 4 characters; maximum length is 128 characters.

Beware that certain characters like '&#35;', '&#61;' or '&#63;' must be properly encoded.

For more information about this issue, please refer to RFC 3986 ("*Uniform Resource Identifier (URI): Generic Syntax*"), sections 2.1, 2.2 and 2.4.

        penalty: **The penalty applied to each character that is part of a word, number sequence, alphabet sequence, etc.**

The penalty is a float number in the range [0, 1]. Full penalty, 0; no penalty, 1.

The character used as decimal separator is always '&#46;'. Hence, a parameter value like *0,33* would be illegal.

The default value is *0.25*.

        outputformat: **The format of the returned analysis.**

Allowed values are *json*, *xml* and *yaml*.

The default value is *xml*.

        reqid: **An identifier for this request.**

The request identifier is a string that must match the regular expression */(?i)^[a-z0-9]{8,16}$/*.

This identifier is echoed in the returned response. Its value has no effect on the password analysis.

If this parameter is unset, a randomly generated identifier will be automatically assigned to this request.

        
    """
    url = f"https://arespass.p.rapidapi.com/ec"
    querystring = {'password': password, }
    if penalty:
        querystring['penalty'] = penalty
    if outputformat:
        querystring['outputFormat'] = outputformat
    if reqid:
        querystring['reqId'] = reqid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "arespass.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

