import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def latest_top_15_earthquake_felt_by_local(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Latest Top 15 Earthquake (felt by local)"
    
    """
    url = f"https://indonesia-latest-weather-and-earthquake.p.rapidapi.com/feelbylocal_top15_earthquake"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesia-latest-weather-and-earthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_top_15_earthquake(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Latest Top 15 Earthquake"
    
    """
    url = f"https://indonesia-latest-weather-and-earthquake.p.rapidapi.com/latest_top15_earthquake"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesia-latest-weather-and-earthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_earthquake(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Latest Earthquake"
    
    """
    url = f"https://indonesia-latest-weather-and-earthquake.p.rapidapi.com/latest_earthquake"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesia-latest-weather-and-earthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weather_cities_in_indonesia_filtered_by_province_code(kode_bps: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Weather Cities in Indonesia filtered by Province (Code)
		
		Here is the list of Povince Code (kode_bps)
		- 11	Provinsi Aceh
		- 51	Provinsi Bali
		- 19	Provinsi Bangka Belitung
		- 36	Provinsi Banten
		- 17	Provinsi Bengkulu
		- 34	Provinsi DI Yogyakarta
		- 31	Provinsi DKI Jakarta
		- 75	Provinsi Gorontalo
		- 15	Provinsi Jambi
		- 32	Provinsi Jawa Barat
		- 33	Provinsi Jawa Tengah
		- 35	Provinsi Jawa Timur
		- 61	Provinsi Kalimantan Barat
		- 63	Provinsi Kalimantan Selatan
		- 62	Provinsi Kalimantan Tengah
		- 64	Provinsi Kalimantan Timur
		- 65	Provinsi Kalimantan Utara
		- 21	Provinsi Kepulauan Riau
		- 18	Provinsi Lampung
		- 81	Provinsi Maluku
		- 82	Provinsi Maluku Utara
		- 52	Provinsi Nusa Tenggara Barat
		- 53	Provinsi Nusa Tenggara Timur
		- 94	Provinsi Papua
		- 91	Provinsi Papua Barat
		- 14	Provinsi Riau
		- 76	Provinsi Sulawesi Barat
		- 73	Provinsi Sulawesi Selatan
		- 72	Provinsi Sulawesi Tengah
		- 74	Provinsi Sulawesi Tenggara
		- 71	Provinsi Sulawesi Utara
		- 13	Provinsi Sumatera Barat
		- 16	Provinsi Sumatera Selatan
		- 12	Provinsi Sumatera Utara"
    
    """
    url = f"https://indonesia-latest-weather-and-earthquake.p.rapidapi.com/weather_indonesia_by_province"
    querystring = {'kode_bps': kode_bps, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesia-latest-weather-and-earthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_cities_weather_in_indonesia(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Top Cities Weather in Indonesia"
    
    """
    url = f"https://indonesia-latest-weather-and-earthquake.p.rapidapi.com/weather_indonesia"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesia-latest-weather-and-earthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

