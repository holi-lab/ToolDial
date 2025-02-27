import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def locale(locale: str='zh_CN', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sets a specific locale and language
		
		**Locale	Name**
		
		- af_ZA	Afrikaans
		- ar	Arabic
		- az	Azerbaijani
		- cz	Czech
		- de	German
		- de_AT	German (Austria)
		- de_CH	German (Switzerland)
		- el	Greek
		- en	English
		- en_AU	English (Australia)
		- en_AU_ocker	English (Australia Ocker)
		- en_BORK	English (Bork)
		- en_CA	English (Canada)
		- en_GB	English (Great Britain)
		- en_GH	English (Ghana)
		- en_IE	English (Ireland)
		- en_IND	English (India)
		- en_NG	Nigeria (English)
		- en_US	English (United States)
		- en_ZA	English (South Africa)
		- es	Spanish
		- es_MX	Spanish (Mexico)
		- fa	Farsi
		- fi	Finnish
		- fr	French
		- fr_BE	Français (Belgique)
		- fr_CA	French (Canada)
		- fr_CH	French (Switzerland)
		- ge	Georgian
		- he	Hebrew
		- hr	Hrvatski
		- hu	Hungarian
		- hy	Armenian
		- id_ID	Indonesia
		- it	Italian
		- ja	Japanese
		- ko	Korean
		- lv	Latvian
		- mk	Macedonian
		- nb_NO	Norwegian
		- ne	Nepalese
		- nl	Dutch
		vnl_BE	Dutch (Belgium)
		- pl	Polish
		- pt_BR	Portuguese (Brazil)
		- pt_PT	Portuguese (Portugal)
		- ro	Romanian
		- ru	Russian
		- sk	Slovakian
		- sv	Swedish
		- tr	Turkish
		- uk	Ukrainian
		- ur	Urdu
		- vi	Vietnamese
		- zh_CN	Chinese
		- zh_TW	Chinese (Taiwan)
		- zu_ZA	Zulu (South Africa)"
    
    """
    url = f"https://user-generator-and-random-profiles.p.rapidapi.com/"
    querystring = {}
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "user-generator-and-random-profiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

