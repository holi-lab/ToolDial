import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def hotfileindex(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "HotFileIndex is the #1 Agency Marketplace For LifeTime Deal, Best Selling JvZoo, WarriorPlus products and its OTOs(One-Time Offers) for cheap.
		#hotfileindex 
		Address 26 Dobson St Ascot, Canberra, New South Wales, 2601, Australia
		Hotline 0987.101.697
		Email: support@hotfileindex.com
		Website https://hotfileindex.com 
		https://www.hotfileindex.com/product/content-gorilla-otos/ 
		https://www.hotfileindex.com/product/cloudfunnels-pro-plan-ltd/ 
		https://www.hotfileindex.com/product/pr-rage-2-0-otos/ 
		https://www.hotfileindex.com/product/marketing-master-io-ultra-grand-master/ 
		https://www.facebook.com/HotFileIndexcom/ 
		https://k289gitlab1.citrin.ch/-/snippets/4578
		https://www.cnccode.com/user/hotfileindex
		https://mastodon.online/@hotfileindex
		https://amara.org/en/profiles/profile/HCvF1RFILk-c4ssR2u2kVFiqGOJ7cjbDtF8cwYuKeDE/
		https://www.mifare.net/support/forum/users/hotfileindex/
		http://onlineboxing.net/jforum/user/edit/144897.page
		https://godotengine.org/qa/user/hotfileindex
		https://app.roll20.net/users/9933181/hotfileindex
		https://woorise.com/hotfileindex
		http://gitlab.aic.ru/snippets/4500
		http://www.asmetalwork.1gb.ua/forum/user/edit/57190.page
		https://beacons.page/hotfileindex
		https://band.us/band/86188502
		https://www.weddingbee.com/members/hotfileindex/
		https://bookme.name/hotfileindex
		https://able2know.org/user/hotfileindex/
		https://medium.com/@hotfileindexcom
		https://medium.com/@hotfileindexcom/hotfileindex-c9aa7f9a6e70
		https://hypixel.net/members/hotfileindex.5761508/#about
		https://www.bitchute.com/channel/nnG0wgrSEuTI/"
    
    """
    url = f"https://hotfileindex.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotfileindex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

