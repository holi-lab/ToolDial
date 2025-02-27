import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def aircraft_registration_lookup_us(air_worth_date: str='19681202', serial_number: str='28-11211', year_mfr: str='1968', mfr_mdl_code: str='7102810', street: str='14744 W RIVER*', county: str='097', type_registrant: str='7', country: str='US', n_number: str='6722*', name: str='AMERICAN*', type_engine: str='1', unique_id: str='600021', certification: str='1N', last_action_date: str='20150225', city: str='LINCOLNSHIRE', zip_code: str='600692202', eng_mfr_mdl: str='41532', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Aircraft Registration Lookup.  Returns max 20 matching entries for all keys selected.  Adding a wildcard * character at the end of a value will return all matching values and continue looking for other matching keys.  NOTE: These values are treat as an "or" (inclusive) not an "and" operation in this release."
    air_worth_date: Filter by AIR_WORTH_DATE
        serial_number: Filter by SERIAL_NUMBER
        year_mfr: Filter by YEAR_MFR
        mfr_mdl_code: Filter by MFR_MDL_CODE
        street: Filter by STREET
        county: Filter by COUNTY
        type_registrant: Filter by TYPE_REGISTRANT
        country: Filter by COUNTRY
        n_number: Filter by N-NUMBER
        name: Filter by Name
        type_engine: Filter by TYPE_ENGINE
        unique_id: Filter by UNIQUE_ID
        certification: Filter by CERTIFICATION
        last_action_date: Filter by LAST_ACTION_DATE
        city: Filter by CITY
        zip_code: Filter by ZIP_CODE
        eng_mfr_mdl: Filter by ENG_MFR_MDL
        
    """
    url = f"https://us-aircraft-registration-information.p.rapidapi.com/flights/v1/master"
    querystring = {}
    if air_worth_date:
        querystring['AIR_WORTH_DATE'] = air_worth_date
    if serial_number:
        querystring['SERIAL_NUMBER'] = serial_number
    if year_mfr:
        querystring['YEAR_MFR'] = year_mfr
    if mfr_mdl_code:
        querystring['MFR_MDL_CODE'] = mfr_mdl_code
    if street:
        querystring['STREET'] = street
    if county:
        querystring['COUNTY'] = county
    if type_registrant:
        querystring['TYPE_REGISTRANT'] = type_registrant
    if country:
        querystring['COUNTRY'] = country
    if n_number:
        querystring['N-NUMBER'] = n_number
    if name:
        querystring['NAME'] = name
    if type_engine:
        querystring['TYPE_ENGINE'] = type_engine
    if unique_id:
        querystring['UNIQUE_ID'] = unique_id
    if certification:
        querystring['CERTIFICATION'] = certification
    if last_action_date:
        querystring['LAST_ACTION_DATE'] = last_action_date
    if city:
        querystring['CITY'] = city
    if zip_code:
        querystring['ZIP_CODE'] = zip_code
    if eng_mfr_mdl:
        querystring['ENG_MFR_MDL'] = eng_mfr_mdl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-aircraft-registration-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

