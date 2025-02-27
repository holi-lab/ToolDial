import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def owner_trainer_analysis_v1_owners_owner_id_analysis_trainers_get(owner_id: str, type: str='[]', region: str='[]', end_date: str=None, start_date: str=None, race_class: str='[]', sex_restriction: str='[]', age_band: str='[]', min_distance_y: int=None, going: str='[]', course: str='[]', max_distance_y: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Trainer statistics for an owner (STANDARD PLAN)"
    type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        min_distance_y: <p>Query results by minimum race distance (yards)</p>
        going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        max_distance_y: <p>Query results by maximum race distance (yards)</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/owners/{owner_id}/analysis/trainers"
    querystring = {}
    if type:
        querystring['type'] = type
    if region:
        querystring['region'] = region
    if end_date:
        querystring['end_date'] = end_date
    if start_date:
        querystring['start_date'] = start_date
    if race_class:
        querystring['race_class'] = race_class
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if age_band:
        querystring['age_band'] = age_band
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if going:
        querystring['going'] = going
    if course:
        querystring['course'] = course
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def owner_distance_analysis_v1_owners_owner_id_analysis_distances_get(owner_id: str, race_class: str='[]', going: str='[]', max_distance_y: int=None, course: str='[]', age_band: str='[]', type: str='[]', region: str='[]', end_date: str=None, min_distance_y: int=None, start_date: str=None, sex_restriction: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Distance statistics for an owner (STANDARD PLAN)"
    race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        max_distance_y: <p>Query results by maximum race distance (yards)</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        min_distance_y: <p>Query results by minimum race distance (yards)</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/owners/{owner_id}/analysis/distances"
    querystring = {}
    if race_class:
        querystring['race_class'] = race_class
    if going:
        querystring['going'] = going
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if course:
        querystring['course'] = course
    if age_band:
        querystring['age_band'] = age_band
    if type:
        querystring['type'] = type
    if region:
        querystring['region'] = region
    if end_date:
        querystring['end_date'] = end_date
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if start_date:
        querystring['start_date'] = start_date
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def owner_jockey_analysis_v1_owners_owner_id_analysis_jockeys_get(owner_id: str, min_distance_y: int=None, start_date: str=None, age_band: str='[]', region: str='[]', going: str='[]', type: str='[]', sex_restriction: str='[]', course: str='[]', race_class: str='[]', end_date: str=None, max_distance_y: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Jockey statistics for an owner (STANDARD PLAN)"
    min_distance_y: <p>Query results by minimum race distance (yards)</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        max_distance_y: <p>Query results by maximum race distance (yards)</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/owners/{owner_id}/analysis/jockeys"
    querystring = {}
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if start_date:
        querystring['start_date'] = start_date
    if age_band:
        querystring['age_band'] = age_band
    if region:
        querystring['region'] = region
    if going:
        querystring['going'] = going
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
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def owner_results_v1_owners_owner_id_results_get(owner_id: str, end_date: str=None, race_class: str='[]', course: str='[]', region: str='[]', min_distance_y: int=None, max_distance_y: int=None, going: str='[]', sex_restriction: str='[]', type: str='[]', age_band: str='[]', start_date: str=None, skip: int=0, limit: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Full historic results for an owner (PRO PLAN)"
    end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        min_distance_y: <p>Query results by minimum race distance (yards)</p>
        max_distance_y: <p>Query results by maximum race distance (yards)</p>
        going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/owners/{owner_id}/results"
    querystring = {}
    if end_date:
        querystring['end_date'] = end_date
    if race_class:
        querystring['race_class'] = race_class
    if course:
        querystring['course'] = course
    if region:
        querystring['region'] = region
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if going:
        querystring['going'] = going
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if type:
        querystring['type'] = type
    if age_band:
        querystring['age_band'] = age_band
    if start_date:
        querystring['start_date'] = start_date
    if skip:
        querystring['skip'] = skip
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def owner_course_analysis_v1_owners_owner_id_analysis_courses_get(owner_id: str, race_class: str='[]', max_distance_y: int=None, sex_restriction: str='[]', age_band: str='[]', min_distance_y: int=None, course: str='[]', region: str='[]', going: str='[]', type: str='[]', end_date: str=None, start_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Course statistics for an owner (STANDARD PLAN)"
    race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        max_distance_y: <p>Query results by maximum race distance (yards)</p>
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        min_distance_y: <p>Query results by minimum race distance (yards)</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/owners/{owner_id}/analysis/courses"
    querystring = {}
    if race_class:
        querystring['race_class'] = race_class
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if age_band:
        querystring['age_band'] = age_band
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if course:
        querystring['course'] = course
    if region:
        querystring['region'] = region
    if going:
        querystring['going'] = going
    if type:
        querystring['type'] = type
    if end_date:
        querystring['end_date'] = end_date
    if start_date:
        querystring['start_date'] = start_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_owners_v1_owners_search_get(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search owners by name (STANDARD PLAN)"
    
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/owners/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_jockeys_v1_jockeys_search_get(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search jockeys by name (STANDARD PLAN)"
    
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/jockeys/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def todays_results_v1_results_today_get(skip: int=0, limit: int=25, region: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Today's results (BASIC PLAN)"
    region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/results/today"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if limit:
        querystring['limit'] = limit
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_results_v1_results_get(course: str='[]', region: str='[]', sex_restriction: str='[]', race_class: str='[]', max_distance_y: int=None, limit: int=25, skip: int=0, age_band: str='[]', type: str='[]', start_date: str=None, min_distance_y: int=None, end_date: str=None, going: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All historic results, up to 12 months in the past (STANDARD PLAN)"
    course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        max_distance_y: <p>Query results by maximum race distance (yards)</p>
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        min_distance_y: <p>Query results by minimum race distance (yards)</p>
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/results"
    querystring = {}
    if course:
        querystring['course'] = course
    if region:
        querystring['region'] = region
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if race_class:
        querystring['race_class'] = race_class
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if limit:
        querystring['limit'] = limit
    if skip:
        querystring['skip'] = skip
    if age_band:
        querystring['age_band'] = age_band
    if type:
        querystring['type'] = type
    if start_date:
        querystring['start_date'] = start_date
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if end_date:
        querystring['end_date'] = end_date
    if going:
        querystring['going'] = going
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dam_class_analysis_v1_dams_dam_id_analysis_classes_get(dam_id: str, max_distance_y: int=None, start_date: str=None, min_distance_y: int=None, going: str='[]', course: str='[]', region: str='[]', end_date: str=None, age_band: str='[]', sex_restriction: str='[]', type: str='[]', race_class: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Offspring class statistics for dam (STANDARD PLAN).</p><p>For dam results and statistics, use the <a href='https://api.theracingapi.com/documentation#tag/Horses'>horses endpoints</a>, replacing the 'dam_' dam id prefix with 'hrs_'.</p>"
    max_distance_y: <p>Query results by maximum race distance (yards)</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        min_distance_y: <p>Query results by minimum race distance (yards)</p>
        going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/dams/{dam_id}/analysis/classes"
    querystring = {}
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if start_date:
        querystring['start_date'] = start_date
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if going:
        querystring['going'] = going
    if course:
        querystring['course'] = course
    if region:
        querystring['region'] = region
    if end_date:
        querystring['end_date'] = end_date
    if age_band:
        querystring['age_band'] = age_band
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if type:
        querystring['type'] = type
    if race_class:
        querystring['race_class'] = race_class
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dam_results_v1_dams_dam_id_results_get(dam_id: str, sex_restriction: str='[]', going: str='[]', course: str='[]', type: str='[]', age_band: str='[]', min_distance_y: int=None, region: str='[]', end_date: str=None, max_distance_y: int=None, skip: int=0, limit: int=25, start_date: str=None, race_class: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Full historic results for dam offspring (PRO PLAN).</p><p>For dam results and statistics, use the <a href='https://api.theracingapi.com/documentation#tag/Horses'>horses endpoints</a>, replacing the 'dam_' dam id prefix with 'hrs_'.</p>"
    sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        min_distance_y: <p>Query results by minimum race distance (yards)</p>
        region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        max_distance_y: <p>Query results by maximum race distance (yards)</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/dams/{dam_id}/results"
    querystring = {}
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if going:
        querystring['going'] = going
    if course:
        querystring['course'] = course
    if type:
        querystring['type'] = type
    if age_band:
        querystring['age_band'] = age_band
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if region:
        querystring['region'] = region
    if end_date:
        querystring['end_date'] = end_date
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if skip:
        querystring['skip'] = skip
    if limit:
        querystring['limit'] = limit
    if start_date:
        querystring['start_date'] = start_date
    if race_class:
        querystring['race_class'] = race_class
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_dams_v1_dams_search_get(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search dams by name (STANDARD PLAN)"
    
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/dams/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dam_distance_analysis_v1_dams_dam_id_analysis_distances_get(dam_id: str, going: str='[]', region: str='[]', course: str='[]', max_distance_y: int=None, sex_restriction: str='[]', age_band: str='[]', type: str='[]', end_date: str=None, min_distance_y: int=None, race_class: str='[]', start_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Offspring distance statistics for dam (STANDARD PLAN). .</p><p>For dam results and statistics, use the <a href='https://api.theracingapi.com/documentation#tag/Horses'>horses endpoints</a>, replacing the 'dam_' dam id prefix with 'hrs_'.</p>"
    going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        max_distance_y: <p>Query results by maximum race distance (yards)</p>
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        min_distance_y: <p>Query results by minimum race distance (yards)</p>
        race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/dams/{dam_id}/analysis/distances"
    querystring = {}
    if going:
        querystring['going'] = going
    if region:
        querystring['region'] = region
    if course:
        querystring['course'] = course
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if age_band:
        querystring['age_band'] = age_band
    if type:
        querystring['type'] = type
    if end_date:
        querystring['end_date'] = end_date
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if race_class:
        querystring['race_class'] = race_class
    if start_date:
        querystring['start_date'] = start_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def horse_distance_time_analysis_v1_horses_horse_id_analysis_distance_times_get(horse_id: str, region: str='[]', age_band: str='[]', going: str='[]', course: str='[]', end_date: str=None, min_distance_y: int=None, sex_restriction: str='[]', max_distance_y: int=None, type: str='[]', start_date: str=None, race_class: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Distance statistics for a horse, with times and going (STANDARD PLAN)"
    region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        min_distance_y: <p>Query results by minimum race distance (yards)</p>
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        max_distance_y: <p>Query results by maximum race distance (yards)</p>
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/horses/{horse_id}/analysis/distance-times"
    querystring = {}
    if region:
        querystring['region'] = region
    if age_band:
        querystring['age_band'] = age_band
    if going:
        querystring['going'] = going
    if course:
        querystring['course'] = course
    if end_date:
        querystring['end_date'] = end_date
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if type:
        querystring['type'] = type
    if start_date:
        querystring['start_date'] = start_date
    if race_class:
        querystring['race_class'] = race_class
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def horse_results_v1_horses_horse_id_results_get(horse_id: str, going: str='[]', sex_restriction: str='[]', region: str='[]', race_class: str='[]', course: str='[]', age_band: str='[]', limit: int=25, max_distance_y: int=None, type: str='[]', start_date: str=None, min_distance_y: int=None, end_date: str=None, skip: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Full historic results for a horse (PRO PLAN)"
    going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        max_distance_y: <p>Query results by maximum race distance (yards)</p>
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        min_distance_y: <p>Query results by minimum race distance (yards)</p>
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/horses/{horse_id}/results"
    querystring = {}
    if going:
        querystring['going'] = going
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if region:
        querystring['region'] = region
    if race_class:
        querystring['race_class'] = race_class
    if course:
        querystring['course'] = course
    if age_band:
        querystring['age_band'] = age_band
    if limit:
        querystring['limit'] = limit
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if type:
        querystring['type'] = type
    if start_date:
        querystring['start_date'] = start_date
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if end_date:
        querystring['end_date'] = end_date
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_horses_v1_horses_search_get(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search horses by name (STANDARD PLAN)"
    
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/horses/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_racecards_beta_v1_racecards_beta_get(region_codes: str='[]', day: str='today', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get today and tomorrows advanced racecards (ODDS BETA)"
    region_codes: Filter racecards by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.
        day: Query racecards by day:<br> today, tomorrow
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/racecards/beta"
    querystring = {}
    if region_codes:
        querystring['region_codes'] = region_codes
    if day:
        querystring['day'] = day
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_racecards_basic_v1_racecards_get(day: str='today', region_codes: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get today and tomorrows advanced racecards (BASIC PLAN)"
    day: Query racecards by day:<br> today, tomorrow
        region_codes: Filter racecards by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/racecards"
    querystring = {}
    if day:
        querystring['day'] = day
    if region_codes:
        querystring['region_codes'] = region_codes
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_racecards_free_v1_racecards_basic_get(region_codes: str='[]', day: str='today', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get today and tomorrows basic racecards (FREE PLAN)"
    region_codes: Filter racecards by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.
        day: Query racecards by day:<br> today, tomorrow
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/racecards/basic"
    querystring = {}
    if region_codes:
        querystring['region_codes'] = region_codes
    if day:
        querystring['day'] = day
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def racecard_horse_results_v1_racecards_horse_id_results_get(horse_id: str, region: str='[]', end_date: str=None, course: str='[]', min_distance_y: int=None, race_class: str='[]', going: str='[]', max_distance_y: int=None, age_band: str='[]', type: str='[]', sex_restriction: str='[]', skip: int=0, start_date: str=None, limit: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full historic results for a horse on today or tomorrows racecards (BASIC PLAN)"
    region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        min_distance_y: <p>Query results by minimum race distance (yards)</p>
        race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        max_distance_y: <p>Query results by maximum race distance (yards)</p>
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/racecards/{horse_id}/results"
    querystring = {}
    if region:
        querystring['region'] = region
    if end_date:
        querystring['end_date'] = end_date
    if course:
        querystring['course'] = course
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if race_class:
        querystring['race_class'] = race_class
    if going:
        querystring['going'] = going
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if age_band:
        querystring['age_band'] = age_band
    if type:
        querystring['type'] = type
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if skip:
        querystring['skip'] = skip
    if start_date:
        querystring['start_date'] = start_date
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_racecards_standard_v1_racecards_standard_get(region_codes: str='[]', day: str='today', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get today and tomorrows advanced racecards, including runner odds for UK & IRE racing. (STANDARD PLAN)"
    region_codes: Filter racecards by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.
        day: Query racecards by day:<br> today, tomorrow
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/racecards/standard"
    querystring = {}
    if region_codes:
        querystring['region_codes'] = region_codes
    if day:
        querystring['day'] = day
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_damsires_v1_damsires_search_get(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search damsires by name (STANDARD PLAN)"
    
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/damsires/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_courses_v1_courses_get(region_codes: str="['gb', 'ire']", toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List courses (BASIC PLAN)"
    region_codes: Filter courses by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/courses"
    querystring = {}
    if region_codes:
        querystring['region_codes'] = region_codes
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def damsire_class_analysis_v1_damsires_damsire_id_analysis_classes_get(damsire_id: str, race_class: str='[]', sex_restriction: str='[]', course: str='[]', type: str='[]', age_band: str='[]', going: str='[]', end_date: str=None, region: str='[]', min_distance_y: int=None, max_distance_y: int=None, start_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Grandoffspring class statistics for damsire (STANDARD PLAN).</p><p>For damsire results and statistics, use the <a href='https://api.theracingapi.com/documentation#tag/Horses'>horses endpoints</a>, replacing the 'dsi_' damsire id prefix with 'hrs_'.</p>"
    race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        min_distance_y: <p>Query results by minimum race distance (yards)</p>
        max_distance_y: <p>Query results by maximum race distance (yards)</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/damsires/{damsire_id}/analysis/classes"
    querystring = {}
    if race_class:
        querystring['race_class'] = race_class
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if course:
        querystring['course'] = course
    if type:
        querystring['type'] = type
    if age_band:
        querystring['age_band'] = age_band
    if going:
        querystring['going'] = going
    if end_date:
        querystring['end_date'] = end_date
    if region:
        querystring['region'] = region
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if start_date:
        querystring['start_date'] = start_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_racecards_big_races_v1_racecards_big_races_get(region_codes: str='[]', end_date: str=None, start_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get future big race's racecards, such as Cheltenham, The Grand National and The Derby (STANDARD PLAN)"
    region_codes: Filter racecards by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.
        end_date: <p>Query racecards to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        start_date: <p>Query racecards from date with format YYYY-MM-DD (e.g. 2020-01-01)</p><p>Date must be today or greater, defaults to today.</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/racecards/big-races"
    querystring = {}
    if region_codes:
        querystring['region_codes'] = region_codes
    if end_date:
        querystring['end_date'] = end_date
    if start_date:
        querystring['start_date'] = start_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def damsire_results_v1_damsires_damsire_id_results_get(damsire_id: str, going: str='[]', min_distance_y: int=None, end_date: str=None, limit: int=25, region: str='[]', course: str='[]', skip: int=0, type: str='[]', sex_restriction: str='[]', race_class: str='[]', age_band: str='[]', max_distance_y: int=None, start_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Full historic results for damsire grandoffspring (PRO PLAN). .</p><p>For damsire results and statistics, use the <a href='https://api.theracingapi.com/documentation#tag/Horses'>horses endpoints</a>, replacing the 'dsi_' damsire id prefix with 'hrs_'.</p>"
    going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        min_distance_y: <p>Query results by minimum race distance (yards)</p>
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        max_distance_y: <p>Query results by maximum race distance (yards)</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/damsires/{damsire_id}/results"
    querystring = {}
    if going:
        querystring['going'] = going
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if end_date:
        querystring['end_date'] = end_date
    if limit:
        querystring['limit'] = limit
    if region:
        querystring['region'] = region
    if course:
        querystring['course'] = course
    if skip:
        querystring['skip'] = skip
    if type:
        querystring['type'] = type
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if race_class:
        querystring['race_class'] = race_class
    if age_band:
        querystring['age_band'] = age_band
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if start_date:
        querystring['start_date'] = start_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_regions_v1_courses_regions_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List regions (FREE PLAN)"
    
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/courses/regions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_racecards_pro_v1_racecards_pro_get(region_codes: str='[]', date: str='2023-04-16', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get past and future advanced racecards by date, including runner odds for UK & IRE racing. (PRO PLAN)<br> (Past racecards available from 2023-01-23. Future racecards available up to 1 week in advance.)"
    region_codes: Filter racecards by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.
        date: Query racecards by date with format YYYY-MM-DD (e.g 2023-04-05)
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/racecards/pro"
    querystring = {}
    if region_codes:
        querystring['region_codes'] = region_codes
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def damsire_distance_analysis_v1_damsires_damsire_id_analysis_distances_get(damsire_id: str, max_distance_y: int=None, age_band: str='[]', race_class: str='[]', region: str='[]', going: str='[]', course: str='[]', min_distance_y: int=None, end_date: str=None, type: str='[]', start_date: str=None, sex_restriction: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Grandoffspring distance statistics for damsire (STANDARD PLAN).</p><p>For damsire results and statistics, use the <a href='https://api.theracingapi.com/documentation#tag/Horses'>horses endpoints</a>, replacing the 'dsi_' damsire id prefix with 'hrs_'.</p>"
    max_distance_y: <p>Query results by maximum race distance (yards)</p>
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        min_distance_y: <p>Query results by minimum race distance (yards)</p>
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/damsires/{damsire_id}/analysis/distances"
    querystring = {}
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if age_band:
        querystring['age_band'] = age_band
    if race_class:
        querystring['race_class'] = race_class
    if region:
        querystring['region'] = region
    if going:
        querystring['going'] = going
    if course:
        querystring['course'] = course
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if end_date:
        querystring['end_date'] = end_date
    if type:
        querystring['type'] = type
    if start_date:
        querystring['start_date'] = start_date
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trainer_owner_analysis_v1_trainers_trainer_id_analysis_owners_get(trainer_id: str, min_distance_y: int=None, region: str='[]', start_date: str=None, going: str='[]', course: str='[]', sex_restriction: str='[]', race_class: str='[]', type: str='[]', age_band: str='[]', end_date: str=None, max_distance_y: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Owner statistics for a trainer (STANDARD PLAN)"
    min_distance_y: <p>Query results by minimum race distance (yards)</p>
        region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        max_distance_y: <p>Query results by maximum race distance (yards)</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/trainers/{trainer_id}/analysis/owners"
    querystring = {}
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if region:
        querystring['region'] = region
    if start_date:
        querystring['start_date'] = start_date
    if going:
        querystring['going'] = going
    if course:
        querystring['course'] = course
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if race_class:
        querystring['race_class'] = race_class
    if type:
        querystring['type'] = type
    if age_band:
        querystring['age_band'] = age_band
    if end_date:
        querystring['end_date'] = end_date
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trainer_course_analysis_v1_trainers_trainer_id_analysis_courses_get(trainer_id: str, going: str='[]', region: str='[]', min_distance_y: int=None, type: str='[]', race_class: str='[]', end_date: str=None, max_distance_y: int=None, age_band: str='[]', sex_restriction: str='[]', start_date: str=None, course: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Course statistics for a trainer (STANDARD PLAN)"
    going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        min_distance_y: <p>Query results by minimum race distance (yards)</p>
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        max_distance_y: <p>Query results by maximum race distance (yards)</p>
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/trainers/{trainer_id}/analysis/courses"
    querystring = {}
    if going:
        querystring['going'] = going
    if region:
        querystring['region'] = region
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if type:
        querystring['type'] = type
    if race_class:
        querystring['race_class'] = race_class
    if end_date:
        querystring['end_date'] = end_date
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if age_band:
        querystring['age_band'] = age_band
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if start_date:
        querystring['start_date'] = start_date
    if course:
        querystring['course'] = course
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trainer_results_v1_trainers_trainer_id_results_get(trainer_id: str, race_class: str='[]', going: str='[]', start_date: str=None, age_band: str='[]', region: str='[]', course: str='[]', type: str='[]', min_distance_y: int=None, sex_restriction: str='[]', limit: int=25, skip: int=0, max_distance_y: int=None, end_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Full historic results for a trainer (PRO PLAN)"
    race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        min_distance_y: <p>Query results by minimum race distance (yards)</p>
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        max_distance_y: <p>Query results by maximum race distance (yards)</p>
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/trainers/{trainer_id}/results"
    querystring = {}
    if race_class:
        querystring['race_class'] = race_class
    if going:
        querystring['going'] = going
    if start_date:
        querystring['start_date'] = start_date
    if age_band:
        querystring['age_band'] = age_band
    if region:
        querystring['region'] = region
    if course:
        querystring['course'] = course
    if type:
        querystring['type'] = type
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if limit:
        querystring['limit'] = limit
    if skip:
        querystring['skip'] = skip
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if end_date:
        querystring['end_date'] = end_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trainer_distance_analysis_v1_trainers_trainer_id_analysis_distances_get(trainer_id: str, going: str='[]', max_distance_y: int=None, region: str='[]', sex_restriction: str='[]', age_band: str='[]', course: str='[]', min_distance_y: int=None, type: str='[]', end_date: str=None, start_date: str=None, race_class: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Distance statistics for a trainer (STANDARD PLAN)"
    going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        max_distance_y: <p>Query results by maximum race distance (yards)</p>
        region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        min_distance_y: <p>Query results by minimum race distance (yards)</p>
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/trainers/{trainer_id}/analysis/distances"
    querystring = {}
    if going:
        querystring['going'] = going
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if region:
        querystring['region'] = region
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if age_band:
        querystring['age_band'] = age_band
    if course:
        querystring['course'] = course
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if type:
        querystring['type'] = type
    if end_date:
        querystring['end_date'] = end_date
    if start_date:
        querystring['start_date'] = start_date
    if race_class:
        querystring['race_class'] = race_class
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trainer_horse_age_analysis_v1_trainers_trainer_id_analysis_horse_age_get(trainer_id: str, min_distance_y: int=None, type: str='[]', region: str='[]', end_date: str=None, going: str='[]', course: str='[]', sex_restriction: str='[]', age_band: str='[]', max_distance_y: int=None, race_class: str='[]', start_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Horse age statistics for a trainer (STANDARD PLAN)"
    min_distance_y: <p>Query results by minimum race distance (yards)</p>
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        max_distance_y: <p>Query results by maximum race distance (yards)</p>
        race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/trainers/{trainer_id}/analysis/horse-age"
    querystring = {}
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if type:
        querystring['type'] = type
    if region:
        querystring['region'] = region
    if end_date:
        querystring['end_date'] = end_date
    if going:
        querystring['going'] = going
    if course:
        querystring['course'] = course
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if age_band:
        querystring['age_band'] = age_band
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if race_class:
        querystring['race_class'] = race_class
    if start_date:
        querystring['start_date'] = start_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_trainers_v1_trainers_search_get(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search trainers by name (STANDARD PLAN)"
    
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/trainers/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trainer_jockey_analysis_v1_trainers_trainer_id_analysis_jockeys_get(trainer_id: str, max_distance_y: int=None, going: str='[]', race_class: str='[]', sex_restriction: str='[]', type: str='[]', region: str='[]', course: str='[]', end_date: str=None, age_band: str='[]', start_date: str=None, min_distance_y: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Jockey statistics for a trainer (STANDARD PLAN)"
    max_distance_y: <p>Query results by maximum race distance (yards)</p>
        going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        min_distance_y: <p>Query results by minimum race distance (yards)</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/trainers/{trainer_id}/analysis/jockeys"
    querystring = {}
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if going:
        querystring['going'] = going
    if race_class:
        querystring['race_class'] = race_class
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if type:
        querystring['type'] = type
    if region:
        querystring['region'] = region
    if course:
        querystring['course'] = course
    if end_date:
        querystring['end_date'] = end_date
    if age_band:
        querystring['age_band'] = age_band
    if start_date:
        querystring['start_date'] = start_date
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sire_class_analysis_v1_sires_sire_id_analysis_classes_get(sire_id: str, max_distance_y: int=None, end_date: str=None, race_class: str='[]', min_distance_y: int=None, type: str='[]', sex_restriction: str='[]', start_date: str=None, age_band: str='[]', course: str='[]', going: str='[]', region: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Offspring class statistics for sire (STANDARD PLAN).</p><p>For sire results and statistics, use the <a href='https://api.theracingapi.com/documentation#tag/Horses'>horses endpoints</a>, replacing the 'sir_' sire id prefix with 'hrs_'.</p>"
    max_distance_y: <p>Query results by maximum race distance (yards)</p>
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        min_distance_y: <p>Query results by minimum race distance (yards)</p>
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/sires/{sire_id}/analysis/classes"
    querystring = {}
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if end_date:
        querystring['end_date'] = end_date
    if race_class:
        querystring['race_class'] = race_class
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if type:
        querystring['type'] = type
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if start_date:
        querystring['start_date'] = start_date
    if age_band:
        querystring['age_band'] = age_band
    if course:
        querystring['course'] = course
    if going:
        querystring['going'] = going
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sire_results_v1_sires_sire_id_results_get(sire_id: str, region: str='[]', skip: int=0, age_band: str='[]', race_class: str='[]', course: str='[]', min_distance_y: int=None, max_distance_y: int=None, start_date: str=None, limit: int=25, type: str='[]', going: str='[]', end_date: str=None, sex_restriction: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Full historic results for sire offspring (PRO PLAN).</p><p>For sire results and statistics, use the <a href='https://api.theracingapi.com/documentation#tag/Horses'>horses endpoints</a>, replacing the 'sir_' sire id prefix with 'hrs_'.</p>"
    region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        min_distance_y: <p>Query results by minimum race distance (yards)</p>
        max_distance_y: <p>Query results by maximum race distance (yards)</p>
        start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/sires/{sire_id}/results"
    querystring = {}
    if region:
        querystring['region'] = region
    if skip:
        querystring['skip'] = skip
    if age_band:
        querystring['age_band'] = age_band
    if race_class:
        querystring['race_class'] = race_class
    if course:
        querystring['course'] = course
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if start_date:
        querystring['start_date'] = start_date
    if limit:
        querystring['limit'] = limit
    if type:
        querystring['type'] = type
    if going:
        querystring['going'] = going
    if end_date:
        querystring['end_date'] = end_date
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sire_distance_analysis_v1_sires_sire_id_analysis_distances_get(sire_id: str, start_date: str=None, end_date: str=None, course: str='[]', region: str='[]', age_band: str='[]', type: str='[]', max_distance_y: int=None, going: str='[]', min_distance_y: int=None, sex_restriction: str='[]', race_class: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Offspring distance statistics for sire (STANDARD PLAN).</p><p>For sire results and statistics, use the <a href='https://api.theracingapi.com/documentation#tag/Horses'>horses endpoints</a>, replacing the 'sir_' sire id prefix with 'hrs_'.</p>"
    start_date: <p>Query results from date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        end_date: <p>Query results to date with format YYYY-MM-DD (e.g. 2020-01-01)</p>
        course: Query results by course ids. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_courses_v1_courses_get'>here</a>.
        region: <p>Query results by region codes. Get the full list <a href='https://api.theracingapi.com/documentation#tag/Courses/operation/list_regions_v1_courses_regions_get'>here</a>.</p><p>Note: If the course query parameter is specified, this will be ignored.</p>
        age_band: <p>Query results by age band:</p><p>10yo+, 2-3yo, 2yo, 2yo+, 3-4yo, 3-5yo, 3-6yo, 3yo, 3yo+, 4-5yo, 4-6yo, 4-7yo, 4-8yo, 4yo, 4yo+, 5-6yo, 5-7yo, 5-8yo, 5yo, 5yo+, 6-7yo, 6yo, 6yo+, 7yo+, 8yo+, 9yo+</p>
        type: <p>Query results by race type:</p><p>chase, flat, hurdle, nh_flat</p>
        max_distance_y: <p>Query results by maximum race distance (yards)</p>
        going: <p>Query results by going:</p><p>fast, firm, good, good_to_firm, good_to_soft, good_to_yielding, hard, heavy, holding, muddy, sloppy, slow, soft, soft_to_heavy, standard, standard_to_fast, standard_to_slow, very_soft, yielding, yielding_to_soft</p>
        min_distance_y: <p>Query results by minimum race distance (yards)</p>
        sex_restriction: <p>Query results by sex restriction:</p><p>c&f, c&g, f, f&m, m, m&g</p>
        race_class: <p>Query results by class:</p><p>class_1, class_2, class_3, class_4, class_5, class_6, class_7</p>
        
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/sires/{sire_id}/analysis/distances"
    querystring = {}
    if start_date:
        querystring['start_date'] = start_date
    if end_date:
        querystring['end_date'] = end_date
    if course:
        querystring['course'] = course
    if region:
        querystring['region'] = region
    if age_band:
        querystring['age_band'] = age_band
    if type:
        querystring['type'] = type
    if max_distance_y:
        querystring['max_distance_y'] = max_distance_y
    if going:
        querystring['going'] = going
    if min_distance_y:
        querystring['min_distance_y'] = min_distance_y
    if sex_restriction:
        querystring['sex_restriction'] = sex_restriction
    if race_class:
        querystring['race_class'] = race_class
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_sires_v1_sires_search_get(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search sires by name (STANDARD PLAN)"
    
    """
    url = f"https://horse-racing-ai.p.rapidapi.com/v1/sires/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

