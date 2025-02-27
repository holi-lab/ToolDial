import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def lazada_item_search(region: str, q: str, locale: str=None, sort: str=None, page: int=1, startprice: str=None, endprice: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Complete Item Search Endpoint. Supported sites from Singapore, Vietnam, Thailand, Indonesia, Philippines, and Malaysia.
		
		If you need more filtering options, send a request."
    region: Supported Sites:
- Philippines **`PH`**:  [www.lazada.com.ph](https://www.lazada.com.ph)
- Singapore **`SG`**: [www.lazada.sg](https://www.lazada.sg)
- Indonesia **`ID`**: [www.lazada.co.id](https://www.lazada.co.id)
- Malaysia **`MY`**: [www.lazada.com.my](https://www.lazada.com.my)
- Thailand **`TH`**: [www.lazada.co.th](https://www.lazada.co.th)
- Vietnam **`VN`**: [www.lazada.vn](https://www.lazada.vn)
        locale: Available options for each **region** parameter:
- Philippines **`PH`**:  `en_US` 
- Singapore **`SG`**: `en_SG`; `zn_CN`
- Indonesia **`ID`**: `en_US`; `id_ID`
- Malaysia **`MY`**: `en_MY`  ; `zn_CN`  ; ` ms_MY`
- Thailand **`TH`**: `en_US`; `th_TH`
- Vietnam **`VN`**: `en_SG`; `vi_VN`
        
    """
    url = f"https://lazada-datahub.p.rapidapi.com/item_search"
    querystring = {'region': region, 'q': q, }
    if locale:
        querystring['locale'] = locale
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    if startprice:
        querystring['startPrice'] = startprice
    if endprice:
        querystring['endPrice'] = endprice
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lazada-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lazada_item_detail_paused(region: str, itemid: str, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Complete Item Detail. Supported sites from Singapore, Vietnam, Thailand, Indonesia, Philippines, and Malaysia."
    region: Supported Sites:
- Philippines **`PH`**:  [www.lazada.com.ph](https://www.lazada.com.ph)
- Singapore **`SG`**: [www.lazada.sg](https://www.lazada.sg)
- Indonesia **`ID`**: [www.lazada.co.id](https://www.lazada.co.id)
- Malaysia **`MY`**: [www.lazada.com.my](https://www.lazada.com.my)
- Thailand **`TH`**: [www.lazada.co.th](https://www.lazada.co.th)
- Vietnam **`VN`**: [www.lazada.vn](https://www.lazada.vn)
        locale: Available options for each **region** parameter:
- Philippines **`PH`**:  `en_US` 
- Singapore **`SG`**: `en_SG`; `zh_CN`
- Indonesia **`ID`**: `en_US`; `id_ID`
- Malaysia **`MY`**: `en_MY`  ; `zh_CN`  ; ` ms_MY`
- Thailand **`TH`**: `en_US`; `th_TH`
- Vietnam **`VN`**: `en_SG`; `vi_VN`
        
    """
    url = f"https://lazada-datahub.p.rapidapi.com/item_detail"
    querystring = {'region': region, 'itemId': itemid, }
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lazada-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lazada_item_detail_2_working(region: str, itemid: str, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Complete Item Detail. Supported sites from Singapore, Vietnam, Thailand, Indonesia, Philippines, and Malaysia."
    region: Supported Sites:
- Philippines **`PH`**:  [www.lazada.com.ph](https://www.lazada.com.ph)
- Singapore **`SG`**: [www.lazada.sg](https://www.lazada.sg)
- Indonesia **`ID`**: [www.lazada.co.id](https://www.lazada.co.id)
- Malaysia **`MY`**: [www.lazada.com.my](https://www.lazada.com.my)
- Thailand **`TH`**: [www.lazada.co.th](https://www.lazada.co.th)
- Vietnam **`VN`**: [www.lazada.vn](https://www.lazada.vn)
        locale: Available options for each **region** parameter:
- Philippines **`PH`**:  `en_US` 
- Singapore **`SG`**: `en_SG`; `zn_CN`
- Indonesia **`ID`**: `en_US`; `id_ID`
- Malaysia **`MY`**: `en_MY`  ; `zn_CN`  ; ` ms_MY`
- Thailand **`TH`**: `en_US`; `th_TH`
- Vietnam **`VN`**: `en_SG`; `vi_VN`
        
    """
    url = f"https://lazada-datahub.p.rapidapi.com/item_detail_2"
    querystring = {'region': region, 'itemId': itemid, }
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lazada-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lazada_item_search_by_image(imgurl: str, region: str, locale: str=None, catid: str=None, imgregion: str=None, sort: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Complete Item Search Endpoint. Supported sites from Singapore, Vietnam, Thailand, Indonesia, Philippines, and Malaysia.
		
		If you need more filtering options, send a request."
    region: Supported Sites:
- Philippines **`PH`**:  [www.lazada.com.ph](https://www.lazada.com.ph)
- Singapore **`SG`**: [www.lazada.sg](https://www.lazada.sg)
- Indonesia **`ID`**: [www.lazada.co.id](https://www.lazada.co.id)
- Malaysia **`MY`**: [www.lazada.com.my](https://www.lazada.com.my)
- Thailand **`TH`**: [www.lazada.co.th](https://www.lazada.co.th)
- Vietnam **`VN`**: [www.lazada.vn](https://www.lazada.vn)
        locale: Available options for each **region** parameter:
- Philippines **`PH`**:  `en_US` 
- Singapore **`SG`**: `en_SG`; `zn_CN`
- Indonesia **`ID`**: `en_US`; `id_ID`
- Malaysia **`MY`**: `en_MY`  ; `zn_CN`  ; ` ms_MY`
- Thailand **`TH`**: `en_US`; `th_TH`
- Vietnam **`VN`**: `en_SG`; `vi_VN`
        imgregion: Image Region is the search focus position from the given image. Eg.: 120,45,190,150

![https://i.ibb.co/QmphdGT/img-Region-jpg.png](https://i.ibb.co/QmphdGT/img-Region-jpg.png)
        
    """
    url = f"https://lazada-datahub.p.rapidapi.com/item_image_search"
    querystring = {'imgUrl': imgurl, 'region': region, }
    if locale:
        querystring['locale'] = locale
    if catid:
        querystring['catId'] = catid
    if imgregion:
        querystring['imgRegion'] = imgregion
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lazada-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

