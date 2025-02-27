import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getnfts_byaddress(owner: str, pagekey: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets all NFTs currently owned by a given address. Unlimited, results display in 100s."
    owner: Input the address for NFT owner and scrape their NFTs.

        pagekey: To pull subsequent requests. Note: it expires after 10 minutes
        
    """
    url = f"https://nfts-by-address.p.rapidapi.com/getNFTs/"
    querystring = {'owner': owner, }
    if pagekey:
        querystring['pageKey'] = pagekey
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nfts-by-address.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getnfts_metadata(contractaddress: str, tokenid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get NFTs metadata, attributes and enclosed media."
    contractaddress: Address of NFT contract. Example for Bored APES: 0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb

        tokenid: Id for NFT
        
    """
    url = f"https://nfts-by-address.p.rapidapi.com/getNFTMetadata"
    querystring = {'contractAddress': contractaddress, 'tokenId': tokenid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nfts-by-address.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

