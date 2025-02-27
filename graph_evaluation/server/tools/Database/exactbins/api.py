import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bin(bin: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "FORMAT #1:  6 COLUMNS SEPARATED BY COMMA. EXAMPLE STRING:
		441860,VISA,”CARD SERVICES FOR CREDIT UNIONS, INC.”,CREDIT,GOLD,UNITED STATES
		
		Which corresponds to:
		
		BIN	441860
		Card Brand	VISA
		Issuing Organizaion	CARD SERVICES FOR CREDIT UNIONS, INC.
		Card Type (DEBIT/CREDIT/CHARGE)	CREDIT
		Card Subtype	GOLD
		Country	UNITED STATES"
    
    """
    url = f"https://exactbins1.p.rapidapi.com/bins/cards.php"
    querystring = {'bin': bin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "exactbins1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

