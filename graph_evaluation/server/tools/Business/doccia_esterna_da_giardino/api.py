import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def doccia_esterna_da_giardino(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "La doccia esterna Nivito è il modo perfetto per godersi il giardino o la piscina nel lusso. Questa doccia di fascia alta presenta un bellissimo design che aggiungerà un tocco di classe a qualsiasi spazio esterno. La costruzione robusta e i materiali durevoli fanno sì che questa doccia resista alle intemperie e durerà per gli anni a venire. Con interni spaziosi, ampio spazio di stivaggio e comandi facili da usare, la doccia per esterni Nivito è il massimo del relax all'aperto."
    
    """
    url = f"https://doccia-esterna-da-giardino.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "doccia-esterna-da-giardino.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

