import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_personal_reading_report(input: str, options: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Modules available for personal reading report are listed below.
		
		|#|Module Name|Code|Options|Credit|Description|
		|---|---|---|---|---|---|
		|1|Birth Details| `birth-details` |NA|1000|Write Description|
		|2|Rasi & Navamsa Chart|`chart`|`chart_style`|1000|Write Description|
		|3|Planet Positions|`planet-position`|NA|1000|Write Description|
		|4|Mangal Dosha Report|`mangal-dosha`|`chart_style`|1000|Write Description|
		|5|Yoga Details|`yoga-details`|NA|1000|Write Description|
		|6|Kaalsarp Dosha Report|`kaal-sarp-dosha`|`chart_style`|1000|Write Description|
		|7|Sade Sati Report|`sade-sati`|`chart_style`|1000|Write Description|
		|8|Shodasavarga Charts|`shodasavarga-chart`|`chart_style`|1000|Write Description|
		|9|Vimsottari Dasha & Sub Periods|`dasa-periods`|`antardasha`<br>`pratyantardasha`|1000|Write Description|
		|10|Papa Dosha|`papa-dosha`|`chart_style`|1000|Write Description|
		
		Module specific options
		
		|#|Option|Allowed Value|Default|Description|
		|---|---|---|---|---|
		|1|`chart_style`| <ul><li>`north-indian`</li><li>`south-indian`</li><li>`east-indian`</li></ul>|None|Mandatory parameter. Style for included charts.|
		|2|`antardasha`|<ul><li>`all`</li><li>`current_mahadasha`</li><li>`none`</li></ul>|`all`|Antardasha periods to include in report.|
		|3|`pratyantardasha`|<ul><li>`all`</li><li>`current_mahadasha`</li><li>`current_antardasha`</li><li>`none`</li></ul>|`all`|Pratyantardasha periods to include in report.|"
    
    """
    url = f"https://astrology4.p.rapidapi.com/report/personal-reading/instant"
    querystring = {'input': input, 'options': options, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_kundli_matching_advanced(ayanamsa: int, boy_dob: str, boy_coordinates: str, girl_coordinates: str, girl_dob: str, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed kundli matching"
    ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        boy_dob: Boy date and time of birth should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        boy_coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        girl_coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        girl_dob: Girl date and time of birth should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        la: Choose language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/kundli-matching/advanced"
    querystring = {'ayanamsa': ayanamsa, 'boy_dob': boy_dob, 'boy_coordinates': boy_coordinates, 'girl_coordinates': girl_coordinates, 'girl_dob': girl_dob, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_papasamyam_check(girl_dob: str, boy_dob: str, girl_coordinates: str, ayanamsa: int, boy_coordinates: str, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Papasamyam Marriage compatability"
    girl_dob: Girl date and time of birth should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        boy_dob: Boy date and time of birth should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        girl_coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        boy_coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        la: Choose language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/papasamyam-check"
    querystring = {'girl_dob': girl_dob, 'boy_dob': boy_dob, 'girl_coordinates': girl_coordinates, 'ayanamsa': ayanamsa, 'boy_coordinates': boy_coordinates, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_porutham(girl_dob: str, ayanamsa: int, boy_coordinates: str, girl_coordinates: str, system: str, boy_dob: str, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get kerala and tamil nadu style marriage match making"
    girl_dob: Girl date and time of birth should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        boy_coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        girl_coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        system: Choose a system for match making. For Tamil Nadu style use `tamil` and for Kerala style use `kerala`
        boy_dob: Boy date and time of birth should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        la: Choose language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/porutham"
    querystring = {'girl_dob': girl_dob, 'ayanamsa': ayanamsa, 'boy_coordinates': boy_coordinates, 'girl_coordinates': girl_coordinates, 'system': system, 'boy_dob': boy_dob, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_nakshatra_porutham_advanced(girl_nakshatra_pada: int, boy_nakshatra: int, girl_nakshatra: int, boy_nakshatra_pada: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed nakshatra porutham"
    girl_nakshatra_pada: Nakshatra pada of girl.
        boy_nakshatra: Nakshatra id of boy.
        girl_nakshatra: Nakshatra id of girl.
        boy_nakshatra_pada: Nakshatra pada of boy.
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/nakshatra-porutham/advanced"
    querystring = {'girl_nakshatra_pada': girl_nakshatra_pada, 'boy_nakshatra': boy_nakshatra, 'girl_nakshatra': girl_nakshatra, 'boy_nakshatra_pada': boy_nakshatra_pada, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_kundli_matching(ayanamsa: int, boy_coordinates: str, girl_dob: str, boy_dob: str, girl_coordinates: str, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get kundli matching or kundli milan or guna milan based on ashta kuta system, calculating compatibility out of 36 points."
    ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        boy_coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        girl_dob: Girl date and time of birth should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        boy_dob: Boy date and time of birth should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        girl_coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        la: Choose language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/kundli-matching"
    querystring = {'ayanamsa': ayanamsa, 'boy_coordinates': boy_coordinates, 'girl_dob': girl_dob, 'boy_dob': boy_dob, 'girl_coordinates': girl_coordinates, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_horoscope_daily(datetime: str, sign: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        sign: Choose the zodiac sign
        
    """
    url = f"https://astrology4.p.rapidapi.com/horoscope/daily"
    querystring = {'datetime': datetime, 'sign': sign, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_thirumana_porutham(girl_nakshatra_pada: int, girl_nakshatra: int, boy_nakshatra: int, boy_nakshatra_pada: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tamil nadu style nakshatra and rasi porutham for marriage matching"
    girl_nakshatra_pada: Nakshatra pada of girl.
        girl_nakshatra: Nakshatra id of girl.
        boy_nakshatra: Nakshatra id of boy.
        boy_nakshatra_pada: Nakshatra pada of boy.
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/thirumana-porutham"
    querystring = {'girl_nakshatra_pada': girl_nakshatra_pada, 'girl_nakshatra': girl_nakshatra, 'boy_nakshatra': boy_nakshatra, 'boy_nakshatra_pada': boy_nakshatra_pada, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_thirumana_porutham_advanced(girl_nakshatra_pada: int, boy_nakshatra_pada: int, boy_nakshatra: int, girl_nakshatra: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed thirumana porutham details"
    girl_nakshatra_pada: Nakshatra pada of girl.
        boy_nakshatra_pada: Nakshatra pada of boy.
        boy_nakshatra: Nakshatra id of boy.
        girl_nakshatra: Nakshatra id of girl.
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/thirumana-porutham/advanced"
    querystring = {'girl_nakshatra_pada': girl_nakshatra_pada, 'boy_nakshatra_pada': boy_nakshatra_pada, 'boy_nakshatra': boy_nakshatra, 'girl_nakshatra': girl_nakshatra, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_compatibility_reading_report(input: str, options: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Modules available for personal reading report are given below
		
		|#|Module Name|Code|Options|Credit|Description|
		|---|---|---|---|---|---|
		|1|Kundali Matching|`kundli-matching`|NA|2000|Write Description|
		|2|Kerala Porutham|`porutham-kerala`|NA|2000|Write Description|
		|3|Tamil Porutham|`porutham-tamil`|NA|2000|Write Description|
		|4|Birth Details|`birth-details`|`chart_style`|2000|Write Description|
		|5|Mangal Dosha|`mangal-dosha`|NA|2000|Write Description|
		
		Option Values
		
		|#|Option|Value|Mandatory|Description|
		|---|---|---|---|---|
		|1|`chart_style`|One of `north-indian` `south-indian` `east-indian`|Mandatory, No Default Value|Write Description|"
    
    """
    url = f"https://astrology4.p.rapidapi.com/report/compatibility-reading/instant"
    querystring = {'input': input, 'options': options, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_nakshatra_porutham(boy_nakshatra: int, boy_nakshatra_pada: int, girl_nakshatra_pada: int, girl_nakshatra: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Kerala style nakshatra and rasi porutham for marriage matching"
    boy_nakshatra: Nakshatra id of boy.
        boy_nakshatra_pada: Nakshatra pada of boy.
        girl_nakshatra_pada: Nakshatra pada of girl.
        girl_nakshatra: Nakshatra id of girl.
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/nakshatra-porutham"
    querystring = {'boy_nakshatra': boy_nakshatra, 'boy_nakshatra_pada': boy_nakshatra_pada, 'girl_nakshatra_pada': girl_nakshatra_pada, 'girl_nakshatra': girl_nakshatra, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_porutham_advanced(boy_dob: str, boy_coordinates: str, girl_coordinates: str, girl_dob: str, ayanamsa: int, system: str, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed porutham"
    boy_dob: Boy date and time of birth should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        boy_coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        girl_coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        girl_dob: Girl date and time of birth should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        system: Choose a system for match making. For Tamil Nadu style use `tamil` and for Kerala style use `kerala`
        la: Choose language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/porutham/advanced"
    querystring = {'boy_dob': boy_dob, 'boy_coordinates': boy_coordinates, 'girl_coordinates': girl_coordinates, 'girl_dob': girl_dob, 'ayanamsa': ayanamsa, 'system': system, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_hora(ayanamsa: int, coordinates: str, datetime: str, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        la: Choose Language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/hora"
    querystring = {'ayanamsa': ayanamsa, 'coordinates': coordinates, 'datetime': datetime, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_panchang_advanced(ayanamsa: int, coordinates: str, datetime: str, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed panchang"
    ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        la: Choose Language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/panchang/advanced"
    querystring = {'ayanamsa': ayanamsa, 'coordinates': coordinates, 'datetime': datetime, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_inauspicious_period(ayanamsa: int, coordinates: str, datetime: str, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get inauspicious timings like Rahu Kaal, Yamaganda Kaal, Gulika Kaal, Dur Muhurat and Varjyam"
    ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        la: Choose Language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/inauspicious-period"
    querystring = {'ayanamsa': ayanamsa, 'coordinates': coordinates, 'datetime': datetime, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_chandra_bala(datetime: str, coordinates: str, ayanamsa: int, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        la: Choose Language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/chandra-bala"
    querystring = {'datetime': datetime, 'coordinates': coordinates, 'ayanamsa': ayanamsa, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_anandadi_yoga(datetime: str, ayanamsa: int, coordinates: str, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        la: Choose language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/anandadi-yoga"
    querystring = {'datetime': datetime, 'ayanamsa': ayanamsa, 'coordinates': coordinates, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_astrology_solstice(datetime: str, ayanamsa: int, coordinates: str, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        la: Choose Language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/solstice"
    querystring = {'datetime': datetime, 'ayanamsa': ayanamsa, 'coordinates': coordinates, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_ritu(ayanamsa: int, datetime: str, coordinates: str, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        la: Choose Language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/ritu"
    querystring = {'ayanamsa': ayanamsa, 'datetime': datetime, 'coordinates': coordinates, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_auspicious_yoga(datetime: str, ayanamsa: int, coordinates: str, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        la: Choose Language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/auspicious-yoga"
    querystring = {'datetime': datetime, 'ayanamsa': ayanamsa, 'coordinates': coordinates, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_panchang(coordinates: str, datetime: str, ayanamsa: int, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get daily panchang details"
    coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        la: Choose Language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/panchang"
    querystring = {'coordinates': coordinates, 'datetime': datetime, 'ayanamsa': ayanamsa, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tara_bala(datetime: str, coordinates: str, ayanamsa: int, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        la: Choose Language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/tara-bala"
    querystring = {'datetime': datetime, 'coordinates': coordinates, 'ayanamsa': ayanamsa, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_auspicious_period(ayanamsa: int, coordinates: str, datetime: str, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get auspicious timings like Abhijit Muhurat, Amrit Kaal and Brahma Muhurat"
    ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        la: Choose Language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/auspicious-period"
    querystring = {'ayanamsa': ayanamsa, 'coordinates': coordinates, 'datetime': datetime, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_choghadiya(ayanamsa: int, coordinates: str, datetime: str, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get daily choghadiya timings"
    ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        la: Choose language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/choghadiya"
    querystring = {'ayanamsa': ayanamsa, 'coordinates': coordinates, 'datetime': datetime, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_disha_shool(ayanamsa: int, datetime: str, coordinates: str, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        la: Choose Language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/disha-shool"
    querystring = {'ayanamsa': ayanamsa, 'datetime': datetime, 'coordinates': coordinates, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_calendar(date: str, calendar: str, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "|#|Calendar|Value|Supported Languages|
		|---|---|---|---|
		|1|Tamil|`tamil`|`en` `ta`|
		|2|Malayalam|`malayalam`|`en` `ml`|
		|3|Shaka Samwat|`shaka-samvat`|`en` `ml` `hi` `ta` `te`|
		|4|Vikram Samwat|`vikram-samvat`|`en` `ml` `hi` `ta` `te`|
		|5|Amanta|`amanta`|`en` `ml` `hi` `ta` `te`|
		|6|Purnimanta|`purnimanta`|`en` `ml` `hi` `ta` `te`|
		|7|Hijri|`hijri`|`en`|
		|8|Gujarati|`gujarati`|`en` `gu`|
		|9|Bengali|`bengali`|`en` `bn`|
		|10|Lunar|`lunar`|`en` `ml` `hi` `ta` `te`|
		"
    date: Date should be in ISO 8601 extended format YYYY-MM-DD eg: 2004-02-12.
        calendar: Choose the calendar type
        la: Choose language
        
    """
    url = f"https://astrology4.p.rapidapi.com/calendar"
    querystring = {'date': date, 'calendar': calendar, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_universal_year_number(datetime: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/numerology/universal-year-number"
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/universal-year-number"
    querystring = {'datetime': datetime, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_capstone_number(first_name: str, last_name: str, middle_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    first_name: Enter your first name
        last_name: Enter last Name
        middle_name: Enter Middle Name
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/capstone-number"
    querystring = {'first_name': first_name, 'last_name': last_name, }
    if middle_name:
        querystring['middle_name'] = middle_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_period_cycle_number(datetime: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Period Cycle Number"
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/period-cycle-number"
    querystring = {'datetime': datetime, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_transit_cycle_number(first_name: str, last_name: str, middle_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/numerology/transit-cycle-number"
    first_name: Enter your first name
        last_name: Enter last Name
        middle_name: Enter Middle Name
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/transit-cycle-number"
    querystring = {'first_name': first_name, 'last_name': last_name, }
    if middle_name:
        querystring['middle_name'] = middle_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_expression_number(last_name: str, first_name: str, middle_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Numerology Expression Number "
    last_name: Enter last Name
        first_name: Enter your first name
        middle_name: Enter Middle Name
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/expression-number"
    querystring = {'last_name': last_name, 'first_name': first_name, }
    if middle_name:
        querystring['middle_name'] = middle_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_karmic_debt_number(first_name: str, datetime: str, last_name: str, middle_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "karmic-debt-number"
    first_name: Enter your first name
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        last_name: Enter last Name
        middle_name: Enter Middle Name
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/karmic-debt-number"
    querystring = {'first_name': first_name, 'datetime': datetime, 'last_name': last_name, }
    if middle_name:
        querystring['middle_name'] = middle_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_universal_month_number(datetime: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/numerology/universal-month-number"
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/universal-month-number"
    querystring = {'datetime': datetime, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_hidden_passion(last_name: str, first_name: str, middle_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "hidden passion"
    last_name: Enter last Name
        first_name: Enter your first name
        middle_name: Enter Middle Name
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/hidden-passion-number"
    querystring = {'last_name': last_name, 'first_name': first_name, }
    if middle_name:
        querystring['middle_name'] = middle_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_universal_day_number(datetime: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/numerology/universal-day-number"
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/universal-day-number"
    querystring = {'datetime': datetime, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_personal_month_number(datetime: str, reference_year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/numerology/personal-month-number"
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        reference_year: Enter the years in int (eg: 2022)
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/personal-month-number"
    querystring = {'datetime': datetime, 'reference_year': reference_year, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_attainment_number(last_name: str, datetime: str, first_name: str, middle_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Attainment Number"
    last_name: Enter last Name
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        first_name: Enter your first name
        middle_name: Enter Middle Name
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/attainment-number"
    querystring = {'last_name': last_name, 'datetime': datetime, 'first_name': first_name, }
    if middle_name:
        querystring['middle_name'] = middle_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_destiny_number(first_name: str, last_name: str, middle_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "destiny number"
    first_name: Enter your first name
        last_name: Enter last Name
        middle_name: Enter Middle Name
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/destiny-number"
    querystring = {'first_name': first_name, 'last_name': last_name, }
    if middle_name:
        querystring['middle_name'] = middle_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_inclusion_table(last_name: str, first_name: str, middle_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Inclusion Number"
    last_name: Enter last Name
        first_name: Enter your first name
        middle_name: Enter Middle Name
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/inclusion-table-number"
    querystring = {'last_name': last_name, 'first_name': first_name, }
    if middle_name:
        querystring['middle_name'] = middle_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_chaldean_lifepath_number(datetime: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/numerology/chaldean/lifepath-number"
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/chaldean/life-path-number"
    querystring = {'datetime': datetime, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_chaldean_whole_name_number(first_name: str, last_name: str, middle_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/numerology/chaldean-whole-name-number"
    first_name: Enter your first name
        last_name: Enter last Name
        middle_name: Enter Middle Name
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/chaldean/whole-name-number"
    querystring = {'first_name': first_name, 'last_name': last_name, }
    if middle_name:
        querystring['middle_name'] = middle_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_chaldean_identity_inital_code_number(first_name: str, last_name: str, middle_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/numerology/chaldean/identity-inital-code-number"
    first_name: Enter your first name
        last_name: Enter last Name
        middle_name: Enter Middle Name
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/chaldean/identity-initial-code-number"
    querystring = {'first_name': first_name, 'last_name': last_name, }
    if middle_name:
        querystring['middle_name'] = middle_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_chaldean_birth_number(datetime: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/numerology/chaldean/birth-number"
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/chaldean/birth-number"
    querystring = {'datetime': datetime, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_chaldean_daily_name_number(first_name: str, last_name: str, middle_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/numerology/chaldean/daily-name-number"
    first_name: Enter your first name
        last_name: Enter last Name
        middle_name: Enter Middle Name
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/chaldean/daily-name-number"
    querystring = {'first_name': first_name, 'last_name': last_name, }
    if middle_name:
        querystring['middle_name'] = middle_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_planet_position(ayanamsa: int, coordinates: str, datetime: str, la: str='en', planets: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get planet position
		
		Parameter `planets` is optional and by default the api returns all planets excluding `URANUS`, `NEPTUNE` and `PLUTO`.
		
		List of Planets and their ID's
		
		|Planet Name|Planet ID|
		|---|---|
		|SUN|0|
		|MOON|1|
		|MERCURY|2|
		|VENUS|3|
		|MARS|4|
		|JUPITER|5|
		|SATURN|6|
		|URANUS|7|
		|NEPTUNE|8|
		|PLUTO|9|
		|ASCENDANT|100|
		|RAHU|101|
		|KETU|102|"
    ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        la: Choose Language
        planets: (Optional) Coma separted list of planet ids. eg: `0,1,100,102`
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/planet-position"
    querystring = {'ayanamsa': ayanamsa, 'coordinates': coordinates, 'datetime': datetime, }
    if la:
        querystring['la'] = la
    if planets:
        querystring['planets'] = planets
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_kundli(datetime: str, coordinates: str, ayanamsa: int, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full horoscope or kundli of a person."
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        la: Choose language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/kundli"
    querystring = {'datetime': datetime, 'coordinates': coordinates, 'ayanamsa': ayanamsa, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_mangal_dosha_advanced(datetime: str, ayanamsa: int, coordinates: str, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed mangal dosha interpretation"
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        la: Choose language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/mangal-dosha/advanced"
    querystring = {'datetime': datetime, 'ayanamsa': ayanamsa, 'coordinates': coordinates, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_yoga(coordinates: str, datetime: str, ayanamsa: int, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        la: Choose language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/yoga"
    querystring = {'coordinates': coordinates, 'datetime': datetime, 'ayanamsa': ayanamsa, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_dasha_periods(datetime: str, ayanamsa: int, coordinates: str, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        la: Choose language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/dasha-periods"
    querystring = {'datetime': datetime, 'ayanamsa': ayanamsa, 'coordinates': coordinates, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_mangal_dosha(datetime: str, ayanamsa: int, coordinates: str, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check a persons kundli has mangal dosha."
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        la: Choose language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/mangal-dosha"
    querystring = {'datetime': datetime, 'ayanamsa': ayanamsa, 'coordinates': coordinates, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sade_sati(ayanamsa: int, coordinates: str, datetime: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sade sati report"
    ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/sade-sati"
    querystring = {'ayanamsa': ayanamsa, 'coordinates': coordinates, 'datetime': datetime, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_birth_chart(chart_type: str, format: str, chart_style: str, datetime: str, coordinates: str, ayanamsa: int, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get planet positions of different charts in kundli"
    chart_type: Choose type of chart to get planet position of the chart
        format: Choose your chart output format. Only `svg` is supported at present.
        chart_style: Choose type of chart style
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        la: Choose Language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/chart"
    querystring = {'chart_type': chart_type, 'format': format, 'chart_style': chart_style, 'datetime': datetime, 'coordinates': coordinates, 'ayanamsa': ayanamsa, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_kundli_advanced(datetime: str, coordinates: str, ayanamsa: int, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed kundli with dasha periods and mangal dosha details"
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        la: Choose language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/kundli/advanced"
    querystring = {'datetime': datetime, 'coordinates': coordinates, 'ayanamsa': ayanamsa, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_papasamyam(ayanamsa: int, coordinates: str, datetime: str, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the papasamyam details of a person"
    ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        la: Choose language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/papasamyam"
    querystring = {'ayanamsa': ayanamsa, 'coordinates': coordinates, 'datetime': datetime, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_kaal_sarp_dosha(coordinates: str, datetime: str, ayanamsa: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check if a person has kaal sarp dosha."
    coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/kaal-sarp-dosha"
    querystring = {'coordinates': coordinates, 'datetime': datetime, 'ayanamsa': ayanamsa, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sade_sati_advanced(ayanamsa: int, coordinates: str, datetime: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed sade sati intrepretation and transit reports"
    ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/sade-sati/advanced"
    querystring = {'ayanamsa': ayanamsa, 'coordinates': coordinates, 'datetime': datetime, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_birth_details(coordinates: str, ayanamsa: int, datetime: str, la: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get nakshatra and birth details
		
		To know more about nakshatra and corresponding name of nakshatras in Malayalam, Tamil, Telugu and Hindi along with their meanings. [Click Here To Read](https://www.prokerala.com/astrology/nakshatra/)"
    coordinates: Location is accepted in the form of latitude and longitude eg: 10.214747,78.097626.
        ayanamsa: The value for ayanamsa parameter is `1` for Lahiri, `3` for Raman and `5` for KP astrology.
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        la: Choose language
        
    """
    url = f"https://astrology4.p.rapidapi.com/astrology/birth-details"
    querystring = {'coordinates': coordinates, 'ayanamsa': ayanamsa, 'datetime': datetime, }
    if la:
        querystring['la'] = la
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_maturity_number(last_name: str, datetime: str, first_name: str, middle_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Maturity Number"
    last_name: Enter last Name
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        first_name: Enter your first name
        middle_name: Enter Middle Name
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/maturity-number"
    querystring = {'last_name': last_name, 'datetime': datetime, 'first_name': first_name, }
    if middle_name:
        querystring['middle_name'] = middle_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_planes_of_expression_number(first_name: str, last_name: str, middle_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "planes-of-expression-number"
    first_name: Enter your first name
        last_name: Enter last Name
        middle_name: Enter Middle Name
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/planes-of-expression-number"
    querystring = {'first_name': first_name, 'last_name': last_name, }
    if middle_name:
        querystring['middle_name'] = middle_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_subconscious_self_number(last_name: str, first_name: str, middle_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/numerology/subconscious-self-number"
    last_name: Enter last Name
        first_name: Enter your first name
        middle_name: Enter Middle Name
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/subconscious-self-number"
    querystring = {'last_name': last_name, 'first_name': first_name, }
    if middle_name:
        querystring['middle_name'] = middle_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_soul_urge_number(last_name: str, first_name: str, additional_vowel: bool=None, middle_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/numerology/soul-urge-number"
    last_name: Enter last Name
        first_name: Enter your first name
        additional_vowel: if additional vowel (Y,W are additional vowel) needed true (default is false) 
        middle_name: Enter Middle Name
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/soul-urge-number"
    querystring = {'last_name': last_name, 'first_name': first_name, }
    if additional_vowel:
        querystring['additional_vowel'] = additional_vowel
    if middle_name:
        querystring['middle_name'] = middle_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_karmic_lesson_number(last_name: str, first_name: str, middle_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "karmic-lesson-number"
    last_name: Enter last Name
        first_name: Enter your first name
        middle_name: Enter Middle Name
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/karmic-lesson-number"
    querystring = {'last_name': last_name, 'first_name': first_name, }
    if middle_name:
        querystring['middle_name'] = middle_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_cornerstone_number(first_name: str, last_name: str, middle_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Corner Stone Number"
    first_name: Enter your first name
        last_name: Enter last Name
        middle_name: Enter Middle Name
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/cornerstone-number"
    querystring = {'first_name': first_name, 'last_name': last_name, }
    if middle_name:
        querystring['middle_name'] = middle_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_birthmonth_number(datetime: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/birth-month-number"
    querystring = {'datetime': datetime, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_rational_thought_number(last_name: str, first_name: str, datetime: str, middle_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/numerology/rational-thought-number"
    last_name: Enter last Name
        first_name: Enter your first name
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        middle_name: Enter Middle Name
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/rational-thought-number"
    querystring = {'last_name': last_name, 'first_name': first_name, 'datetime': datetime, }
    if middle_name:
        querystring['middle_name'] = middle_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_inner_dream_number(last_name: str, first_name: str, middle_name: str=None, additional_vowel: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Inner Dream Number"
    last_name: Enter last Name
        first_name: Enter your first name
        middle_name: Enter Middle Name
        additional_vowel: if additional vowel (Y,W are additional vowel) needed true (default is false) 
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/inner-dream-number"
    querystring = {'last_name': last_name, 'first_name': first_name, }
    if middle_name:
        querystring['middle_name'] = middle_name
    if additional_vowel:
        querystring['additional_vowel'] = additional_vowel
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_balance_number(last_name: str, first_name: str, middle_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "balance Number"
    last_name: Enter last Name
        first_name: Enter your first name
        middle_name: Enter Middle Name
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/balance-number"
    querystring = {'last_name': last_name, 'first_name': first_name, }
    if middle_name:
        querystring['middle_name'] = middle_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_pinnacle_number(datetime: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/numerology/pinnacle-number"
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/pinnacle-number"
    querystring = {'datetime': datetime, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_bith(datetime: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Challenge Number"
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/challenge-number"
    querystring = {'datetime': datetime, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_personality_number(first_name: str, last_name: str, middle_name: str=None, additional_vowel: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/numerology/personality-number"
    first_name: Enter your first name
        last_name: Enter last Name
        middle_name: Enter Middle Name
        additional_vowel: if additional vowel (Y,W are additional vowel) needed true (default is false) 
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/personality-number"
    querystring = {'first_name': first_name, 'last_name': last_name, }
    if middle_name:
        querystring['middle_name'] = middle_name
    if additional_vowel:
        querystring['additional_vowel'] = additional_vowel
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_personal_year_number(reference_year: int, datetime: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/numerology/personal-year-number"
    reference_year: Enter the years in int (eg: 2022)
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/personal-year-number"
    querystring = {'reference_year': reference_year, 'datetime': datetime, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_essence_number(last_name: str, first_name: str, datetime: str, reference_year: int, middle_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Essence Number"
    last_name: Enter last Name
        first_name: Enter your first name
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        reference_year: Enter the years in int (eg: 2022)
        middle_name: Enter Middle Name
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/essence-number"
    querystring = {'last_name': last_name, 'first_name': first_name, 'datetime': datetime, 'reference_year': reference_year, }
    if middle_name:
        querystring['middle_name'] = middle_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_birth_year_number(datetime: str, last_name: str, first_name: str, middle_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        last_name: Enter last Name
        first_name: Enter your first name
        middle_name: Enter Middle Name
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/bridge-number"
    querystring = {'datetime': datetime, 'last_name': last_name, 'first_name': first_name, }
    if middle_name:
        querystring['middle_name'] = middle_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_life_path_number(datetime: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Life Path Number"
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/life-path-number"
    querystring = {'datetime': datetime, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology(datetime: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/birthday-number"
    querystring = {'datetime': datetime, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_numerology_personal_day_number(reference_year: int, datetime: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/numerology/personal-day-number"
    reference_year: Enter the years in int (eg: 2022)
        datetime: Date and time should be in ISO 8601 format YYYY-MM-DDTHH:MM:SSZ eg:2004-02-12T15:19:21+05:30.
        
    """
    url = f"https://astrology4.p.rapidapi.com/numerology/personal-day-number"
    querystring = {'reference_year': reference_year, 'datetime': datetime, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astrology4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

