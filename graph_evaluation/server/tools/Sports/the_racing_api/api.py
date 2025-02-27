import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def racecard_horse_results(horse_id: str, type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', region: str='[
  "gb",
  "ire",
  "aus"
]', limit: int=25, age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', course: str='[
  "crs_1234"
]', start_date: str='2022-01-01', sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', skip: int=0, min_distance_y: int=1500, end_date: str='2022-02-01', race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', max_distance_y: int=2200, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get results for a horse on today or tomorrows racecards (PRO PLAN)"
    type: Query results by race type
        going: Query results by going 
        region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        age_band: Query results by age band
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        start_date: Query results from date with format YYYY-MM-DD
        sex_restriction: Query results by sex restriction
        min_distance_y: Query results by minimum race distance (yards) 
        end_date: Query results to date with format YYYY-MM-DD
        race_class: Query results by class
        max_distance_y: Query results by maximum race distance (yards) 
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/racecards/{horse_id}/results"
    querystring = {}
    if type:
        querystring['type'] = type
    if going:
        querystring['going'] = going
    if region:
        querystring['region'] = region
    if limit:
        querystring['limit'] = limit
    if age_band:
        querystring['age_band'] = age_band
    if course:
        querystring['course'] = course
    if start_date:
        querystring['start_date'] = start_date
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if skip:
        querystring['skip'] = skip
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if end_date:
        querystring['end_date'] = end_date
    if race_class:
        querystring['race_class'] = race_class
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_racecards(region_codes: str='["gb", "ire"]', day: str='today', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get today and tomorrows racecards (PRO PLAN)"
    region_codes: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.
        day: Query racecards by day: today, tomorrow
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/racecards"
    querystring = {}
    if region_codes:
        querystring['region_codes'] = region_codes
    if day:
        querystring['day'] = day
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_racecards_basic(region_codes: str='["gb", "ire"]', day: str='today', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get today and tomorrows basic racecards (BASIC PLAN)"
    
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/racecards/free"
    querystring = {}
    if region_codes:
        querystring['region_codes'] = region_codes
    if day:
        querystring['day'] = day
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def todays_results(skip: int=0, limit: int=25, region: str='["gb", "ire"]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Today's results (PRO PLAN)"
    
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/results/today"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if limit:
        querystring['limit'] = limit
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def damsire_results(damsire_id: str, min_distance_y: int=1500, type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', max_distance_y: int=2200, limit: int=25, region: str='[
  "gb",
  "ire",
  "aus"
]', sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', course: str='[
  "crs_1234"
]', start_date: str='2022-01-01', end_date: str='2022-02-01', age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', skip: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Full results for damsire grandoffspring (MEGA PLAN). 
		
		 For damsire results and statistics, use the horses endpoints, replacing the 'dsi_' damsire id prefix with 'hrs_'."
    min_distance_y: Query results by minimum race distance (yards) 
        type: Query results by race type
        max_distance_y: Query results by maximum race distance (yards) 
        region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        sex_restriction: Query results by sex restriction
        race_class: Query results by class
        going: Query results by going 
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        start_date: Query results from date with format YYYY-MM-DD
        end_date: Query results to date with format YYYY-MM-DD
        age_band: Query results by age band
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/damsires/{damsire_id}/results"
    querystring = {}
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if type:
        querystring['type'] = type
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if limit:
        querystring['limit'] = limit
    if region:
        querystring['region'] = region
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if race_class:
        querystring['race_class'] = race_class
    if going:
        querystring['going'] = going
    if course:
        querystring['course'] = course
    if start_date:
        querystring['start_date'] = start_date
    if end_date:
        querystring['end_date'] = end_date
    if age_band:
        querystring['age_band'] = age_band
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sire_class_analysis(sire_id: str, type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', min_distance_y: int=1500, going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', start_date: str='2022-01-01', region: str='[
  "gb",
  "ire",
  "aus"
]', course: str='[
  "crs_1234"
]', sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', max_distance_y: int=2200, end_date: str='2022-02-01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Offspring class statistics for sire (ULTRA PLAN). 
		
		 For sire results and statistics, use the horses endpoints, replacing the 'sir_' sire id prefix with 'hrs_'."
    type: Query results by race type
        min_distance_y: Query results by minimum race distance (yards) 
        going: Query results by going 
        start_date: Query results from date with format YYYY-MM-DD
        region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        sex_restriction: Query results by sex restriction
        race_class: Query results by class
        age_band: Query results by age band
        max_distance_y: Query results by maximum race distance (yards) 
        end_date: Query results to date with format YYYY-MM-DD
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/sires/{sire_id}/analysis/classes"
    querystring = {}
    if type:
        querystring['type'] = type
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if going:
        querystring['going'] = going
    if start_date:
        querystring['start_date'] = start_date
    if region:
        querystring['region'] = region
    if course:
        querystring['course'] = course
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if race_class:
        querystring['race_class'] = race_class
    if age_band:
        querystring['age_band'] = age_band
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if end_date:
        querystring['end_date'] = end_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trainer_horse_age_analysis(trainer_id: str, race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', region: str='[
  "gb",
  "ire",
  "aus"
]', going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', start_date: str='2022-01-01', course: str='[
  "crs_1234"
]', sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', max_distance_y: int=2200, type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', end_date: str='2022-02-01', min_distance_y: int=1500, age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Horse age statistics for a trainer (ULTRA PLAN)"
    race_class: Query results by class
        region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        going: Query results by going 
        start_date: Query results from date with format YYYY-MM-DD
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        sex_restriction: Query results by sex restriction
        max_distance_y: Query results by maximum race distance (yards) 
        type: Query results by race type
        end_date: Query results to date with format YYYY-MM-DD
        min_distance_y: Query results by minimum race distance (yards) 
        age_band: Query results by age band
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/trainers/{trainer_id}/analysis/horse-age"
    querystring = {}
    if race_class:
        querystring['race_class'] = race_class
    if region:
        querystring['region'] = region
    if going:
        querystring['going'] = going
    if start_date:
        querystring['start_date'] = start_date
    if course:
        querystring['course'] = course
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if type:
        querystring['type'] = type
    if end_date:
        querystring['end_date'] = end_date
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if age_band:
        querystring['age_band'] = age_band
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def horse_distance_time_analysis(horse_id: str, age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', start_date: str='2022-01-01', min_distance_y: int=1500, race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', end_date: str='2022-02-01', region: str='[
  "gb",
  "ire",
  "aus"
]', type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', max_distance_y: int=2200, course: str='[
  "crs_1234"
]', going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Distance statistics for a horse, with times and going (ULTRA PLAN)"
    age_band: Query results by age band
        start_date: Query results from date with format YYYY-MM-DD
        min_distance_y: Query results by minimum race distance (yards) 
        race_class: Query results by class
        sex_restriction: Query results by sex restriction
        end_date: Query results to date with format YYYY-MM-DD
        region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        type: Query results by race type
        max_distance_y: Query results by maximum race distance (yards) 
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        going: Query results by going 
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/horses/{horse_id}/analysis/distance-times"
    querystring = {}
    if age_band:
        querystring['age_band'] = age_band
    if start_date:
        querystring['start_date'] = start_date
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if race_class:
        querystring['race_class'] = race_class
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if end_date:
        querystring['end_date'] = end_date
    if region:
        querystring['region'] = region
    if type:
        querystring['type'] = type
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if course:
        querystring['course'] = course
    if going:
        querystring['going'] = going
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def damsire_class_analysis(damsire_id: str, race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', min_distance_y: int=1500, region: str='[
  "gb",
  "ire",
  "aus"
]', end_date: str='2022-02-01', going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', course: str='[
  "crs_1234"
]', sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', max_distance_y: int=2200, start_date: str='2022-01-01', age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Grandoffspring class statistics for damsire (ULTRA PLAN). 
		
		For damsire results and statistics, use the horses endpoints, replacing the 'dsi_' damsire id prefix with 'hrs_'."
    race_class: Query results by class
        min_distance_y: Query results by minimum race distance (yards) 
        region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        end_date: Query results to date with format YYYY-MM-DD
        going: Query results by going 
        type: Query results by race type
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        sex_restriction: Query results by sex restriction
        max_distance_y: Query results by maximum race distance (yards) 
        start_date: Query results from date with format YYYY-MM-DD
        age_band: Query results by age band
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/damsires/{damsire_id}/analysis/classes"
    querystring = {}
    if race_class:
        querystring['race_class'] = race_class
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if region:
        querystring['region'] = region
    if end_date:
        querystring['end_date'] = end_date
    if going:
        querystring['going'] = going
    if type:
        querystring['type'] = type
    if course:
        querystring['course'] = course
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if start_date:
        querystring['start_date'] = start_date
    if age_band:
        querystring['age_band'] = age_band
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_damsires(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search damsires by name (ULTRA PLAN)"
    
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/damsires/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def damsire_distance_analysis(damsire_id: str, race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', min_distance_y: int=1500, going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', max_distance_y: int=2200, end_date: str='2022-02-01', type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', region: str='[
  "gb",
  "ire",
  "aus"
]', age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', course: str='[
  "crs_1234"
]', start_date: str='2022-01-01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Grandoffspring distance statistics for damsire (ULTRA PLAN). 
		
		For damsire results and statistics, use the horses endpoints, replacing the 'dsi_' damsire id prefix with 'hrs_'."
    race_class: Query results by class
        min_distance_y: Query results by minimum race distance (yards) 
        going: Query results by going 
        sex_restriction: Query results by sex restriction
        max_distance_y: Query results by maximum race distance (yards) 
        end_date: Query results to date with format YYYY-MM-DD
        type: Query results by race type
        region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        age_band: Query results by age band
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        start_date: Query results from date with format YYYY-MM-DD
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/damsires/{damsire_id}/analysis/distances"
    querystring = {}
    if race_class:
        querystring['race_class'] = race_class
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if going:
        querystring['going'] = going
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if end_date:
        querystring['end_date'] = end_date
    if type:
        querystring['type'] = type
    if region:
        querystring['region'] = region
    if age_band:
        querystring['age_band'] = age_band
    if course:
        querystring['course'] = course
    if start_date:
        querystring['start_date'] = start_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dam_class_analysis(dam_id: str, region: str='[
  "gb",
  "ire",
  "aus"
]', sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', min_distance_y: int=1500, age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', course: str='[
  "crs_1234"
]', going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', end_date: str='2022-02-01', max_distance_y: int=2200, start_date: str='2022-01-01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Offspring class statistics for dam (ULTRA PLAN). 
		
		For dam results and statistics, use the horses endpoints, replacing the 'dam_' dam id prefix with 'hrs_'."
    region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        sex_restriction: Query results by sex restriction
        min_distance_y: Query results by minimum race distance (yards) 
        age_band: Query results by age band
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        going: Query results by going 
        type: Query results by race type
        race_class: Query results by class
        end_date: Query results to date with format YYYY-MM-DD
        max_distance_y: Query results by maximum race distance (yards) 
        start_date: Query results from date with format YYYY-MM-DD
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/dams/{dam_id}/analysis/classes"
    querystring = {}
    if region:
        querystring['region'] = region
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if age_band:
        querystring['age_band'] = age_band
    if course:
        querystring['course'] = course
    if going:
        querystring['going'] = going
    if type:
        querystring['type'] = type
    if race_class:
        querystring['race_class'] = race_class
    if end_date:
        querystring['end_date'] = end_date
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if start_date:
        querystring['start_date'] = start_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sire_distance_analysis(sire_id: str, start_date: str='2022-01-01', age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', course: str='[
  "crs_1234"
]', max_distance_y: int=2200, sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', region: str='[
  "gb",
  "ire",
  "aus"
]', type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', end_date: str='2022-02-01', race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', min_distance_y: int=1500, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Offspring distance statistics for sire (ULTRA PLAN). 
		
		For sire results and statistics, use the horses endpoints, replacing the 'sir_' sire id prefix with 'hrs_'."
    start_date: Query results from date with format YYYY-MM-DD
        age_band: Query results by age band
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        max_distance_y: Query results by maximum race distance (yards) 
        sex_restriction: Query results by sex restriction
        going: Query results by going 
        region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        type: Query results by race type
        end_date: Query results to date with format YYYY-MM-DD
        race_class: Query results by class
        min_distance_y: Query results by minimum race distance (yards) 
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/sires/{sire_id}/analysis/distances"
    querystring = {}
    if start_date:
        querystring['start_date'] = start_date
    if age_band:
        querystring['age_band'] = age_band
    if course:
        querystring['course'] = course
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if going:
        querystring['going'] = going
    if region:
        querystring['region'] = region
    if type:
        querystring['type'] = type
    if end_date:
        querystring['end_date'] = end_date
    if race_class:
        querystring['race_class'] = race_class
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def owner_trainer_analysis(owner_id: str, max_distance_y: int=2200, course: str='[
  "crs_1234"
]', min_distance_y: int=1500, region: str='[
  "gb",
  "ire",
  "aus"
]', going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', start_date: str='2022-01-01', sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', end_date: str='2022-02-01', type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Trainer statistics for an owner (ULTRA PLAN)"
    max_distance_y: Query results by maximum race distance (yards) 
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        min_distance_y: Query results by minimum race distance (yards) 
        region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        going: Query results by going 
        start_date: Query results from date with format YYYY-MM-DD
        sex_restriction: Query results by sex restriction
        age_band: Query results by age band
        race_class: Query results by class
        end_date: Query results to date with format YYYY-MM-DD
        type: Query results by race type
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/owners/{owner_id}/analysis/trainers"
    querystring = {}
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if course:
        querystring['course'] = course
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if region:
        querystring['region'] = region
    if going:
        querystring['going'] = going
    if start_date:
        querystring['start_date'] = start_date
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if age_band:
        querystring['age_band'] = age_band
    if race_class:
        querystring['race_class'] = race_class
    if end_date:
        querystring['end_date'] = end_date
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def owner_jockey_analysis(owner_id: str, end_date: str='2022-02-01', course: str='[
  "crs_1234"
]', race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', max_distance_y: int=2200, region: str='[
  "gb",
  "ire",
  "aus"
]', age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', start_date: str='2022-01-01', min_distance_y: int=1500, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Jockey statistics for an owner (ULTRA PLAN)"
    end_date: Query results to date with format YYYY-MM-DD
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        race_class: Query results by class
        going: Query results by going 
        sex_restriction: Query results by sex restriction
        type: Query results by race type
        max_distance_y: Query results by maximum race distance (yards) 
        region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        age_band: Query results by age band
        start_date: Query results from date with format YYYY-MM-DD
        min_distance_y: Query results by minimum race distance (yards) 
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/owners/{owner_id}/analysis/jockeys"
    querystring = {}
    if end_date:
        querystring['end_date'] = end_date
    if course:
        querystring['course'] = course
    if race_class:
        querystring['race_class'] = race_class
    if going:
        querystring['going'] = going
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if type:
        querystring['type'] = type
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if region:
        querystring['region'] = region
    if age_band:
        querystring['age_band'] = age_band
    if start_date:
        querystring['start_date'] = start_date
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_owners(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search owners by name (ULTRA PLAN)"
    
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/owners/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trainer_course_analysis(trainer_id: str, region: str='[
  "gb",
  "ire",
  "aus"
]', end_date: str='2022-02-01', type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', min_distance_y: int=1500, sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', course: str='[
  "crs_1234"
]', start_date: str='2022-01-01', max_distance_y: int=2200, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Course statistics for a trainer (ULTRA PLAN)"
    region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        end_date: Query results to date with format YYYY-MM-DD
        type: Query results by race type
        going: Query results by going 
        age_band: Query results by age band
        min_distance_y: Query results by minimum race distance (yards) 
        sex_restriction: Query results by sex restriction
        race_class: Query results by class
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        start_date: Query results from date with format YYYY-MM-DD
        max_distance_y: Query results by maximum race distance (yards) 
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/trainers/{trainer_id}/analysis/courses"
    querystring = {}
    if region:
        querystring['region'] = region
    if end_date:
        querystring['end_date'] = end_date
    if type:
        querystring['type'] = type
    if going:
        querystring['going'] = going
    if age_band:
        querystring['age_band'] = age_band
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if race_class:
        querystring['race_class'] = race_class
    if course:
        querystring['course'] = course
    if start_date:
        querystring['start_date'] = start_date
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_horses(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search horses by name (ULTRA PLAN)"
    
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/horses/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_courses(region_codes: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List courses (PRO PLAN)"
    region_codes:  Filter courses by region codes. Get the full list using the 'List regions' endpoint.
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/courses"
    querystring = {}
    if region_codes:
        querystring['region_codes'] = region_codes
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_regions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List regions (BASIC PLAN)"
    
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/courses/regions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def horse_results(horse_id: str, min_distance_y: int=1500, start_date: str='2022-01-01', max_distance_y: int=2200, limit: int=25, sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', region: str='[
  "gb",
  "ire",
  "aus"
]', course: str='[
  "crs_1234"
]', end_date: str='2022-02-01', going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', skip: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Full results for a horse (MEGA PLAN)"
    min_distance_y: Query results by minimum race distance (yards) 
        start_date: Query results from date with format YYYY-MM-DD
        max_distance_y: Query results by maximum race distance (yards) 
        sex_restriction: Query results by sex restriction
        region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        end_date: Query results to date with format YYYY-MM-DD
        going: Query results by going 
        race_class: Query results by class
        type: Query results by race type
        age_band: Query results by age band
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/horses/{horse_id}/results"
    querystring = {}
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if start_date:
        querystring['start_date'] = start_date
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if limit:
        querystring['limit'] = limit
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if region:
        querystring['region'] = region
    if course:
        querystring['course'] = course
    if end_date:
        querystring['end_date'] = end_date
    if going:
        querystring['going'] = going
    if race_class:
        querystring['race_class'] = race_class
    if type:
        querystring['type'] = type
    if age_band:
        querystring['age_band'] = age_band
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def owner_distance_analysis(owner_id: str, region: str='[
  "gb",
  "ire",
  "aus"
]', end_date: str='2022-02-01', type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', max_distance_y: int=2200, going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', min_distance_y: int=1500, start_date: str='2022-01-01', age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', course: str='[
  "crs_1234"
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Distance statistics for an owner (ULTRA PLAN)"
    region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        end_date: Query results to date with format YYYY-MM-DD
        type: Query results by race type
        sex_restriction: Query results by sex restriction
        max_distance_y: Query results by maximum race distance (yards) 
        going: Query results by going 
        race_class: Query results by class
        min_distance_y: Query results by minimum race distance (yards) 
        start_date: Query results from date with format YYYY-MM-DD
        age_band: Query results by age band
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/owners/{owner_id}/analysis/distances"
    querystring = {}
    if region:
        querystring['region'] = region
    if end_date:
        querystring['end_date'] = end_date
    if type:
        querystring['type'] = type
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if going:
        querystring['going'] = going
    if race_class:
        querystring['race_class'] = race_class
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if start_date:
        querystring['start_date'] = start_date
    if age_band:
        querystring['age_band'] = age_band
    if course:
        querystring['course'] = course
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_sires(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search sires by name (ULTRA PLAN)"
    
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/sires/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dam_distance_analysis(dam_id: str, sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', course: str='[
  "crs_1234"
]', max_distance_y: int=2200, going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', end_date: str='2022-02-01', type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', start_date: str='2022-01-01', region: str='[
  "gb",
  "ire",
  "aus"
]', race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', min_distance_y: int=1500, age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Offspring distance statistics for dam (ULTRA PLAN). 
		
		 For dam results and statistics, use the horses endpoints, replacing the 'dam_' dam id prefix with 'hrs_'."
    sex_restriction: Query results by sex restriction
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        max_distance_y: Query results by maximum race distance (yards) 
        going: Query results by going 
        end_date: Query results to date with format YYYY-MM-DD
        type: Query results by race type
        start_date: Query results from date with format YYYY-MM-DD
        region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        race_class: Query results by class
        min_distance_y: Query results by minimum race distance (yards) 
        age_band: Query results by age band
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/dams/{dam_id}/analysis/distances"
    querystring = {}
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if course:
        querystring['course'] = course
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if going:
        querystring['going'] = going
    if end_date:
        querystring['end_date'] = end_date
    if type:
        querystring['type'] = type
    if start_date:
        querystring['start_date'] = start_date
    if region:
        querystring['region'] = region
    if race_class:
        querystring['race_class'] = race_class
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if age_band:
        querystring['age_band'] = age_band
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_jockeys(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search jockeys by name (ULTRA PLAN)"
    
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/jockeys/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def owner_course_analysis(owner_id: str, max_distance_y: int=2200, start_date: str='2022-01-01', going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', min_distance_y: int=1500, region: str='[
  "gb",
  "ire",
  "aus"
]', type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', course: str='[
  "crs_1234"
]', race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', end_date: str='2022-02-01', age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Course statistics for an owner (ULTRA PLAN)"
    max_distance_y: Query results by maximum race distance (yards) 
        start_date: Query results from date with format YYYY-MM-DD
        going: Query results by going 
        min_distance_y: Query results by minimum race distance (yards) 
        region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        type: Query results by race type
        sex_restriction: Query results by sex restriction
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        race_class: Query results by class
        end_date: Query results to date with format YYYY-MM-DD
        age_band: Query results by age band
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/owners/{owner_id}/analysis/courses"
    querystring = {}
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if start_date:
        querystring['start_date'] = start_date
    if going:
        querystring['going'] = going
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if region:
        querystring['region'] = region
    if type:
        querystring['type'] = type
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if course:
        querystring['course'] = course
    if race_class:
        querystring['race_class'] = race_class
    if end_date:
        querystring['end_date'] = end_date
    if age_band:
        querystring['age_band'] = age_band
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def owner_results(owner_id: str, end_date: str='2022-02-01', race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', min_distance_y: int=1500, limit: int=25, sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', region: str='[
  "gb",
  "ire",
  "aus"
]', going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', max_distance_y: int=2200, start_date: str='2022-01-01', type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', course: str='[
  "crs_1234"
]', skip: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Full results for an owner (MEGA PLAN)"
    end_date: Query results to date with format YYYY-MM-DD
        race_class: Query results by class
        min_distance_y: Query results by minimum race distance (yards) 
        sex_restriction: Query results by sex restriction
        region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        going: Query results by going 
        age_band: Query results by age band
        max_distance_y: Query results by maximum race distance (yards) 
        start_date: Query results from date with format YYYY-MM-DD
        type: Query results by race type
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/owners/{owner_id}/results"
    querystring = {}
    if end_date:
        querystring['end_date'] = end_date
    if race_class:
        querystring['race_class'] = race_class
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if limit:
        querystring['limit'] = limit
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if region:
        querystring['region'] = region
    if going:
        querystring['going'] = going
    if age_band:
        querystring['age_band'] = age_band
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if start_date:
        querystring['start_date'] = start_date
    if type:
        querystring['type'] = type
    if course:
        querystring['course'] = course
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dam_results(dam_id: str, skip: int=0, sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', limit: int=25, region: str='[
  "gb",
  "ire",
  "aus"
]', course: str='[
  "crs_1234"
]', end_date: str='2022-02-01', race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', min_distance_y: int=1500, start_date: str='2022-01-01', max_distance_y: int=2200, going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Full results for dam offspring (MEGA PLAN). 
		
		For dam results and statistics, use the horses endpoints, replacing the 'dam_' dam id prefix with 'hrs_'."
    sex_restriction: Query results by sex restriction
        type: Query results by race type
        age_band: Query results by age band
        region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        end_date: Query results to date with format YYYY-MM-DD
        race_class: Query results by class
        min_distance_y: Query results by minimum race distance (yards) 
        start_date: Query results from date with format YYYY-MM-DD
        max_distance_y: Query results by maximum race distance (yards) 
        going: Query results by going 
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/dams/{dam_id}/results"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if type:
        querystring['type'] = type
    if age_band:
        querystring['age_band'] = age_band
    if limit:
        querystring['limit'] = limit
    if region:
        querystring['region'] = region
    if course:
        querystring['course'] = course
    if end_date:
        querystring['end_date'] = end_date
    if race_class:
        querystring['race_class'] = race_class
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if start_date:
        querystring['start_date'] = start_date
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if going:
        querystring['going'] = going
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trainer_distance_analysis(trainer_id: str, region: str='[
  "gb",
  "ire",
  "aus"
]', end_date: str='2022-02-01', min_distance_y: int=1500, type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', course: str='[
  "crs_1234"
]', start_date: str='2022-01-01', max_distance_y: int=2200, sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Distance statistics for a trainer (ULTRA PLAN)"
    region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        end_date: Query results to date with format YYYY-MM-DD
        min_distance_y: Query results by minimum race distance (yards) 
        type: Query results by race type
        going: Query results by going 
        age_band: Query results by age band
        race_class: Query results by class
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        start_date: Query results from date with format YYYY-MM-DD
        max_distance_y: Query results by maximum race distance (yards) 
        sex_restriction: Query results by sex restriction
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/trainers/{trainer_id}/analysis/distances"
    querystring = {}
    if region:
        querystring['region'] = region
    if end_date:
        querystring['end_date'] = end_date
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if type:
        querystring['type'] = type
    if going:
        querystring['going'] = going
    if age_band:
        querystring['age_band'] = age_band
    if race_class:
        querystring['race_class'] = race_class
    if course:
        querystring['course'] = course
    if start_date:
        querystring['start_date'] = start_date
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_dams(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search dams by name (ULTRA PLAN)"
    
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/dams/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trainer_results(trainer_id: str, age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', region: str='[
  "gb",
  "ire",
  "aus"
]', start_date: str='2022-01-01', min_distance_y: int=1500, limit: int=25, skip: int=0, race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', end_date: str='2022-02-01', course: str='[
  "crs_1234"
]', max_distance_y: int=2200, type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Full results for a trainer (MEGA PLAN)"
    age_band: Query results by age band
        sex_restriction: Query results by sex restriction
        region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        start_date: Query results from date with format YYYY-MM-DD
        min_distance_y: Query results by minimum race distance (yards) 
        race_class: Query results by class
        going: Query results by going 
        end_date: Query results to date with format YYYY-MM-DD
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        max_distance_y: Query results by maximum race distance (yards) 
        type: Query results by race type
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/trainers/{trainer_id}/results"
    querystring = {}
    if age_band:
        querystring['age_band'] = age_band
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if region:
        querystring['region'] = region
    if start_date:
        querystring['start_date'] = start_date
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if limit:
        querystring['limit'] = limit
    if skip:
        querystring['skip'] = skip
    if race_class:
        querystring['race_class'] = race_class
    if going:
        querystring['going'] = going
    if end_date:
        querystring['end_date'] = end_date
    if course:
        querystring['course'] = course
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_results(age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', end_date: str='2022-02-01', sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', max_distance_y: int=2200, min_distance_y: int=1500, type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', skip: int=0, region: str='[
  "gb",
  "ire",
  "aus"
]', going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', course: str='[
  "crs_1234"
]', race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', start_date: str='2022-01-01', limit: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Full results, up to 12 months in the past (ULTRA PLAN)"
    age_band: Query results by age band
        end_date: Query results to date with format YYYY-MM-DD
        sex_restriction: Query results by sex restriction
        max_distance_y: Query results by maximum race distance (yards) 
        min_distance_y: Query results by minimum race distance (yards) 
        type: Query results by race type
        region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        going: Query results by going 
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        race_class: Query results by class
        start_date: Query results from date with format YYYY-MM-DD
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/results"
    querystring = {}
    if age_band:
        querystring['age_band'] = age_band
    if end_date:
        querystring['end_date'] = end_date
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if type:
        querystring['type'] = type
    if skip:
        querystring['skip'] = skip
    if region:
        querystring['region'] = region
    if going:
        querystring['going'] = going
    if course:
        querystring['course'] = course
    if race_class:
        querystring['race_class'] = race_class
    if start_date:
        querystring['start_date'] = start_date
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trainer_jockey_analysis(trainer_id: str, race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', min_distance_y: int=1500, age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', start_date: str='2022-01-01', end_date: str='2022-02-01', sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', region: str='[
  "gb",
  "ire",
  "aus"
]', type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', course: str='[
  "crs_1234"
]', max_distance_y: int=2200, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Jockey statistics for a trainer (ULTRA PLAN)"
    race_class: Query results by class
        going: Query results by going 
        min_distance_y: Query results by minimum race distance (yards) 
        age_band: Query results by age band
        start_date: Query results from date with format YYYY-MM-DD
        end_date: Query results to date with format YYYY-MM-DD
        sex_restriction: Query results by sex restriction
        region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        type: Query results by race type
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        max_distance_y: Query results by maximum race distance (yards) 
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/trainers/{trainer_id}/analysis/jockeys"
    querystring = {}
    if race_class:
        querystring['race_class'] = race_class
    if going:
        querystring['going'] = going
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if age_band:
        querystring['age_band'] = age_band
    if start_date:
        querystring['start_date'] = start_date
    if end_date:
        querystring['end_date'] = end_date
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if region:
        querystring['region'] = region
    if type:
        querystring['type'] = type
    if course:
        querystring['course'] = course
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sire_results(sire_id: str, race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', skip: int=0, sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', end_date: str='2022-02-01', limit: int=25, going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', course: str='[
  "crs_1234"
]', min_distance_y: int=1500, type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', max_distance_y: int=2200, region: str='[
  "gb",
  "ire",
  "aus"
]', start_date: str='2022-01-01', age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Full results for sire offspring (MEGA PLAN). 
		
		For sire results and statistics, use the horses endpoints, replacing the 'sir_' sire id prefix with 'hrs_'."
    race_class: Query results by class
        sex_restriction: Query results by sex restriction
        end_date: Query results to date with format YYYY-MM-DD
        going: Query results by going 
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        min_distance_y: Query results by minimum race distance (yards) 
        type: Query results by race type
        max_distance_y: Query results by maximum race distance (yards) 
        region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        start_date: Query results from date with format YYYY-MM-DD
        age_band: Query results by age band
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/sires/{sire_id}/results"
    querystring = {}
    if race_class:
        querystring['race_class'] = race_class
    if skip:
        querystring['skip'] = skip
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if end_date:
        querystring['end_date'] = end_date
    if limit:
        querystring['limit'] = limit
    if going:
        querystring['going'] = going
    if course:
        querystring['course'] = course
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if type:
        querystring['type'] = type
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if region:
        querystring['region'] = region
    if start_date:
        querystring['start_date'] = start_date
    if age_band:
        querystring['age_band'] = age_band
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trainer_owner_analysis(trainer_id: str, end_date: str='2022-02-01', going: str='[
  "fast",
  "firm",
  "good",
  "good_to_firm",
  "good_to_soft",
  "good_to_yielding",
  "hard",
  "heavy",
  "holding",
  "muddy",
  "sloppy",
  "slow",
  "soft",
  "soft_to_heavy",
  "standard",
  "standard_to_fast",
  "standard_to_slow",
  "very_soft",
  "yielding",
  "yielding_to_soft"
]', type: str='[
  "chase",
  "flat",
  "hurdle",
  "nh_flat"
]', min_distance_y: int=1500, age_band: str='[
  "10yo+",
  "2-3yo",
  "2yo",
  "2yo+",
  "3-4yo",
  "3-5yo",
  "3-6yo",
  "3yo",
  "3yo+",
  "4-5yo",
  "4-6yo",
  "4-7yo",
  "4-8yo",
  "4yo",
  "4yo+",
  "5-6yo",
  "5-7yo",
  "5-8yo",
  "5yo",
  "5yo+",
  "6-7yo",
  "6yo",
  "6yo+",
  "7yo+",
  "8yo+",
  "9yo+"
]', sex_restriction: str='[
  "c&f",
  "c&g",
  "f",
  "f&m",
  "m",
  "m&g"
]', course: str='[
  "crs_1234"
]', max_distance_y: int=2200, region: str='[
  "gb",
  "ire",
  "aus"
]', start_date: str='2022-01-01', race_class: str='[
  "class_1",
  "class_2",
  "class_3",
  "class_4",
  "class_5",
  "class_6",
  "class_7"
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Owner statistics for a trainer (ULTRA PLAN)"
    end_date: Query results to date with format YYYY-MM-DD
        going: Query results by going 
        type: Query results by race type
        min_distance_y: Query results by minimum race distance (yards) 
        age_band: Query results by age band
        sex_restriction: Query results by sex restriction
        course: Query results by course ids. Get the full list using the 'List courses' endpoint.
        max_distance_y: Query results by maximum race distance (yards) 
        region: Filter racecards by region codes. Get the full list using the 'List regions' endpoint.<br>Note: If the course query parameter is specified, this will be ignored.
        start_date: Query results from date with format YYYY-MM-DD
        race_class: Query results by class
        
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/trainers/{trainer_id}/analysis/owners"
    querystring = {}
    if end_date:
        querystring['end_date'] = end_date
    if going:
        querystring['going'] = going
    if type:
        querystring['type'] = type
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if age_band:
        querystring['age_band'] = age_band
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if course:
        querystring['course'] = course
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if region:
        querystring['region'] = region
    if start_date:
        querystring['start_date'] = start_date
    if race_class:
        querystring['race_class'] = race_class
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_trainers(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search trainers by name (ULTRA PLAN)"
    
    """
    url = f"https://the-racing-api1.p.rapidapi.com/v1/trainers/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-racing-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

