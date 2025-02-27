import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def nftsearch(filter: str, offset: int, chain: str, q: str, from_block: int=None, to_block: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get NFTs that match a given metadata search in all widely used networks"
    filter: - To look into the entire metadata set the value to '*global*'. 
- To have a better response time you can look into a specific field like '*name*'

        offset: Pagination index. Default is set to 0
        chain: The chain to query:
- eth
- bsc
- avalanche
- polygon
- 0x1
- kovan
- avalanche testnet
- bsc testnet
- mumbai
- ropsten
- 0x3
- 0x4
- goerli
        q: Term to look for.
        
    """
    url = f"https://nft-explorer.p.rapidapi.com/search"
    querystring = {'filter': filter, 'offset': offset, 'chain': chain, 'q': q, }
    if from_block:
        querystring['from_block'] = from_block
    if to_block:
        querystring['to_block'] = to_block
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nft-explorer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

