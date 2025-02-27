import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def datum_conversion(coord: str, after_datum: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "緯度経度の測地系(日本測地系/世界測地系)を変換します。"
    coord: Latitude and longitude before conversion.
Support millisecond and degree indication.
        after_datum: Geodetic system after conversion.
(wgs84: World Geodetic System (default), tokyo: Old Japan Geodetic System)
        
    """
    url = f"https://navitime-geocoding.p.rapidapi.com/datum_conversion"
    querystring = {'coord': coord, 'after_datum': after_datum, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "navitime-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def address_inclusion(code: str, coord: str, datum: str='wgs84', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "緯度経度が住所コードの範囲内に含まれているかを判定します。"
    code: Address Code.
Can be specified multiple times, separated by a period.
        coord: Latitude and longitude.
        datum: Geodetic system of latitude and longitude.
(wgs84: World Geodetic System (default), tokyo: Old Japan Geodetic System)
        
    """
    url = f"https://navitime-geocoding.p.rapidapi.com/address_inclusion"
    querystring = {'code': code, 'coord': coord, }
    if datum:
        querystring['datum'] = datum
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "navitime-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def address_reverse_geocoding(coord: str, datum: str='wgs84', coord_unit: str='degree', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "逆ジオコーディング
		緯度経度を指定して住所情報を取得します。"
    coord: Latitude and longitude.
        datum: Geodetic system of latitude and longitude.
(wgs84: World Geodetic System (default), tokyo: Old Japan Geodetic System)
        coord_unit: The unit of latitude and longitude included in the output data.
(degree: decimal system of degrees (default), millisec: milliseconds)
        
    """
    url = f"https://navitime-geocoding.p.rapidapi.com/address/reverse_geocoding"
    querystring = {'coord': coord, }
    if datum:
        querystring['datum'] = datum
    if coord_unit:
        querystring['coord_unit'] = coord_unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "navitime-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def address(coord_unit: str='degree', datum: str='wgs84', kana_row: str=None, limit: int=10, level_from: int=None, level_to: int=None, word: str='代々木', code: str=None, sort: str='code_asc', offset: int=0, address_filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return address information such as a postal code or latitude/longitude from free keyword of request parameters."
    coord_unit: The unit of latitude and longitude included in the output data.
(degree: decimal system of degrees (default), millisec: milliseconds)
        datum: Geodetic system of latitude and longitude.
(wgs84: World Geodetic System (default), tokyo: Old Japan Geodetic System)
        kana_row: Output result filter (consonants).
Only addresses in the response that begin with the letter on the specified line will be returned.

Available only when 'code' is specified.
        limit: Limits the number of response data. (Minimum value: 1, Maximum value: 100, Default value: 10)
        level_from: Minimum address level.
(Minimum value: 1, Maximum value: 7)
Narrow down to only addresses up to a specified address level
　1: Prefecture
　2: City, ward, town or village
　3: Large sections
　4: Small sections
　5: Block
　6: Land number
　7: Branch number
        level_to: Maximum address level.
(Minimum value: 1, Maximum value: 7)
Narrow down to only addresses up to a specified address level
　1: Prefecture
　2: City, ward, town or village
　3: Large sections
　4: Small sections
　5: Block
　6: Land number
　7: Branch number
        word: Search word.

You cannot be used in combination with 'code'.
Please be sure to specify either 'word' or 'code'.
        code: Address code.

You cannot be used in combination with 'word'.
Please be sure to specify either 'word' or 'code'.
        sort: Sorting order of addresses.
(lexical: lexical order, level_asc: address level ascending order, code_asc: address code ascending order (default))
        offset: Skips a given number of first data.
(Minimum value: 0, Maximum value: 2000, Default value: 0)
        address_filter: Address filter.

You can use this when you want to get a specific address code.
If you prefix the address code with a '-', you can exclude the relevant address from the response.

Available only when 'word' is specified.
        
    """
    url = f"https://navitime-geocoding.p.rapidapi.com/address"
    querystring = {}
    if coord_unit:
        querystring['coord_unit'] = coord_unit
    if datum:
        querystring['datum'] = datum
    if kana_row:
        querystring['kana_row'] = kana_row
    if limit:
        querystring['limit'] = limit
    if level_from:
        querystring['level_from'] = level_from
    if level_to:
        querystring['level_to'] = level_to
    if word:
        querystring['word'] = word
    if code:
        querystring['code'] = code
    if sort:
        querystring['sort'] = sort
    if offset:
        querystring['offset'] = offset
    if address_filter:
        querystring['address_filter'] = address_filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "navitime-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def address_code(code: str, coord_unit: str='degree', datum: str='wgs84', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return address information from address code of request parameters."
    code: Address Code.
Can be specified multiple times, separated by a period.
        coord_unit: The unit of latitude and longitude included in the output data.
(degree: decimal system of degrees (default), millisec: milliseconds)
        datum: Geodetic system of latitude and longitude.
(wgs84: World Geodetic System (default), tokyo: Old Japan Geodetic System)
        
    """
    url = f"https://navitime-geocoding.p.rapidapi.com/address/code"
    querystring = {'code': code, }
    if coord_unit:
        querystring['coord_unit'] = coord_unit
    if datum:
        querystring['datum'] = datum
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "navitime-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def address_postal_code(postal_code: str, datum: str='wgs84', offset: int=0, coord_unit: str='degree', limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return address information from postal code of request parameters."
    postal_code: Postal code string (Minimum: 3 digits, Maximum: 7 digits)
        datum: Geodetic system of latitude and longitude.
(wgs84: World Geodetic System (default), tokyo: Old Japan Geodetic System)
        offset: Skips a given number of first data.
(Minimum value: 0, Maximum value: 2000, Default value: 0)
        coord_unit: The unit of latitude and longitude included in the output data.
(degree: decimal system of degrees (default), millisec: milliseconds)
        limit: Limits the number of response data. (Minimum value: 1, Maximum value: 100, Default value: 10)
        
    """
    url = f"https://navitime-geocoding.p.rapidapi.com/address/postal_code"
    querystring = {'postal_code': postal_code, }
    if datum:
        querystring['datum'] = datum
    if offset:
        querystring['offset'] = offset
    if coord_unit:
        querystring['coord_unit'] = coord_unit
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "navitime-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def address_autocomplete(word: str, datum: str='wgs84', coord_unit: str='degree', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return address predictions from the specified keyword of request parameters."
    word: Address string (Minimum: 2 characters, Maximum: 50 characters).

If a string of 51 or more characters is entered, the 51st and subsequent characters will be deleted and searched.
Arabic and Chinese numerals are searched as they are.
        datum: Geodetic system of latitude and longitude.
(wgs84: World Geodetic System (default), tokyo: Old Japan Geodetic System)
        coord_unit: The unit of latitude and longitude included in the output data.
(degree: decimal system of degrees (default), millisec: milliseconds)
        
    """
    url = f"https://navitime-geocoding.p.rapidapi.com/address/autocomplete"
    querystring = {'word': word, }
    if datum:
        querystring['datum'] = datum
    if coord_unit:
        querystring['coord_unit'] = coord_unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "navitime-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

