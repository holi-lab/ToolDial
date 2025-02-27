import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def school_detail(sekolah_id_enkrip: str='5DB43D89E6BFB0F76FBC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "School Detail by sekolah_id_enkrip field that get by other endpoint (see List of School)."
    
    """
    url = f"https://indonesia-school-list.p.rapidapi.com/sekolah_detail_by_kode_id"
    querystring = {}
    if sekolah_id_enkrip:
        querystring['sekolah_id_enkrip'] = sekolah_id_enkrip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesia-school-list.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_school_in_sub_disctrict_kecamatan(kode_wilayah_kecamatan_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of School in Sub-Disctrict (Kecamatan), this endpoint has school NPSN, and sekolah_id_enkrip field to lookup school detail in other endpoint.  See Example Result"
    
    """
    url = f"https://indonesia-school-list.p.rapidapi.com/sekolah_by_kode_kecamatan"
    querystring = {'kode_wilayah_kecamatan_id': kode_wilayah_kecamatan_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesia-school-list.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def number_of_schools_in_indonesia_sub_district_kecamatan(kode_wilayah_kabupaten_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Number of Schools in Indonesia Sub-District (Kecamatan), also return kode_wilayah that used in other endpoint. (See example result and find kode_wilayah  field)"
    
    """
    url = f"https://indonesia-school-list.p.rapidapi.com/sekolah_kecamatan_by_kode_kabupaten"
    querystring = {'kode_wilayah_kabupaten_id': kode_wilayah_kabupaten_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesia-school-list.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def number_of_schools_in_indonesian_city_district_kabupaten_kota(kode_wilayah_propinsi_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Number of Schools in Indonesian City / District (Kabupaten / Kota), also return kode_wilayah that used in other endpoints (See Example response)."
    
    """
    url = f"https://indonesia-school-list.p.rapidapi.com/sekolah_kabupaten_by_kode_propinsi"
    querystring = {'kode_wilayah_propinsi_id': kode_wilayah_propinsi_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesia-school-list.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def number_of_schools_in_the_province_of_indonesia(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Number of Schools in the Province of Indonesia, also return region code (kode_wilayah) that used in other endpoints"
    
    """
    url = f"https://indonesia-school-list.p.rapidapi.com/sekolah_nasional"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesia-school-list.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

