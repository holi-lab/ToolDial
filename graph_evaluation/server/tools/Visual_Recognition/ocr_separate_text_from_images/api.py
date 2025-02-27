import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ocr(language: str, url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is the main endpoint for using ocr API.
		
		The "url" parameter requires an image or PDF file url. The PDF file must not be long.
		
		The "language" parameter requires some specific language codes:
		
		- Arabic - ara
		- Bulgarian- bul
		- ChineseSimplified- chs
		- ChineseTraditional- cht
		- Croatian- hrv
		- Czech- cze
		- Danish- dan
		- Dutch- dut
		- English- eng
		- Finnish- fin
		- French- fre
		- German- ger
		- Greek- gre
		- Hungarian- hun
		- Italian- ita
		- Japanese- jpn
		- Korean- kor
		- Polish- pol
		- Portuguese- por
		- Russian- rus
		- Slovenian- slv
		- Spanish- spa
		- Swedish- swe
		- Turkish- tur"
    
    """
    url = f"https://ocr-separate-text-from-images.p.rapidapi.com/parse/imageurl/"
    querystring = {'language': language, 'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ocr-separate-text-from-images.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

