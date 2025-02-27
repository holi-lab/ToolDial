import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_nft_details(assetid: int, network: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the details of an NFT by its assetId"
    assetid: ID of the asset/nft
        network: The name of the asset/nft
        
    """
    url = f"https://create-algorand-nfts.p.rapidapi.com/nft"
    querystring = {'assetId': assetid, 'network': network, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "create-algorand-nfts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

