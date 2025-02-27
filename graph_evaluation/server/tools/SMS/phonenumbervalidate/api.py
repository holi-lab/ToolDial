import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def validatetest(number: str, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Validate any phone number from any country. For testing purposes only."
    number: 1_ Option: local format (without prefix):
Example: 94887799

2_ Option: format E 164:
Recommended: starting with symbol + followed by country prefix and number, without blank spaces
Remark: the + symbol in a url is escaped for the text %2B leaving %2B59894887799
Example: +59894887799 

1_ Opción: formato local (sin prefijo):
Ejemplo: 94887799

2_ Opción: formato E 164:
Recomendado: comenzando con símbolo + seguido de prefijo país y número, sin espacios en blanco
Observación: el símbolo + en una url se escapea para el texto %2B quedando %2B59894887799
Ejemplo: +59894887799
        country: Format: ISO 3166-1 alpha-2 code
Remark: in capital letters.
Optional if the number parameter starts with the + symbol 

Formato: ISO 3166-1 alpha-2 code
Observación: en mayúsculas.
Opcional si el parametro number comienza con el símbolo +
        
    """
    url = f"https://phonenumbervalidate.p.rapidapi.com/ts_PhoneNumberValidateTest.jsp"
    querystring = {'number': number, }
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phonenumbervalidate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def validate(number: str, country: str='UY', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Validate any phone number from any country."
    number: 1_ Option: local format (without prefix):
Example: 94887799

2_ Option: format E 164:
Recommended: starting with symbol + followed by country prefix and number, without blank spaces
Remark: the + symbol in a url is escaped for the text %2B leaving %2B59894887799
Example: +59894887799 

----

1_ Opción: formato local (sin prefijo):
Ejemplo: 94887799

2_ Opción: formato E 164:
Recomendado: comenzando con símbolo + seguido de prefijo país y número, sin espacios en blanco
Observación: el símbolo + en una url se escapea para el texto %2B quedando %2B59894887799
Ejemplo: +59894887799

        country: Format: ISO 3166-1 alpha-2 code
Remark: in capital letters.
Optional if the number parameter starts with the + symbol 

Formato: ISO 3166-1 alpha-2 code
Observación: en mayúsculas.
Opcional si el parametro number comienza con el símbolo +
        
    """
    url = f"https://phonenumbervalidate.p.rapidapi.com/ts_PhoneNumberValidate.jsp"
    querystring = {'number': number, }
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phonenumbervalidate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

