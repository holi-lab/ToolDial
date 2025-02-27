import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def test_query_metadata_valid_parameters_get(offset: int=0, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://my-schools-api.p.rapidapi.com/metadata/valid_parameters"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-schools-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def test_query_metadata_valid_parameter_values_get(limit: int=10, param: str=None, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://my-schools-api.p.rapidapi.com/metadata/valid_parameter_values"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if param:
        querystring['param'] = param
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-schools-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def test_query_search_get(top_sea_quarter_perc: int=None, rolled_reporting_description: str=None, full_time_equivalent_non_teaching_staff: int=None, campus_type: str=None, latitude: int=None, full_time_equivalent_teaching_staff: int=None, icsea_percentile: int=None, geolocation: str=None, acara_sml_id: int=None, school_name: str=None, statistical_area_2_name: str=None, offset: int=0, statistical_area_1: int=None, commonwealth_electoral_division_name: str=None, suburb: str=None, school_type: str=None, school_age_id: str=None, statistical_area_4: int=None, school_url: str=None, boys_enrolments: int=None, meshblock: int=None, bottom_sea_quarter_perc: int=None, year_range: str=None, school_sector: str=None, lower_middle_sea_quarter_perc: int=None, non_teaching_staff: int=None, language_background_other_than_english_yes_perc: int=None, fuzzy: bool=None, state_electoral_division_name: str=None, postcode: int=None, governing_body_url: str=None, longitude: int=None, language_background_other_than_english_no_perc: int=None, commonwealth_electoral_division: int=None, state: str=None, statistical_area_3_name: str=None, statistical_area_2: int=None, abs_remoteness_area_name: str=None, indigenous_enrolments_perc: int=None, language_background_other_than_english_not_stated_perc: int=None, location_age_id: str=None, teaching_staff: int=None, upper_middle_sea_quarter_perc: int=None, governing_body: str=None, limit: int=10, icsea: int=None, girls_enrolments: int=None, local_government_area: int=None, calendar_year: int=None, is_id: int=None, full_time_equivalent_enrolments: int=None, state_electoral_division: int=None, statistical_area_4_name: str=None, local_government_area_name: str=None, total_enrolments: int=None, abs_remoteness_area: int=None, statistical_area_3: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://my-schools-api.p.rapidapi.com/search"
    querystring = {}
    if top_sea_quarter_perc:
        querystring['top_sea_quarter_perc'] = top_sea_quarter_perc
    if rolled_reporting_description:
        querystring['rolled_reporting_description'] = rolled_reporting_description
    if full_time_equivalent_non_teaching_staff:
        querystring['full_time_equivalent_non_teaching_staff'] = full_time_equivalent_non_teaching_staff
    if campus_type:
        querystring['campus_type'] = campus_type
    if latitude:
        querystring['latitude'] = latitude
    if full_time_equivalent_teaching_staff:
        querystring['full_time_equivalent_teaching_staff'] = full_time_equivalent_teaching_staff
    if icsea_percentile:
        querystring['icsea_percentile'] = icsea_percentile
    if geolocation:
        querystring['geolocation'] = geolocation
    if acara_sml_id:
        querystring['acara_sml_id'] = acara_sml_id
    if school_name:
        querystring['school_name'] = school_name
    if statistical_area_2_name:
        querystring['statistical_area_2_name'] = statistical_area_2_name
    if offset:
        querystring['offset'] = offset
    if statistical_area_1:
        querystring['statistical_area_1'] = statistical_area_1
    if commonwealth_electoral_division_name:
        querystring['commonwealth_electoral_division_name'] = commonwealth_electoral_division_name
    if suburb:
        querystring['suburb'] = suburb
    if school_type:
        querystring['school_type'] = school_type
    if school_age_id:
        querystring['school_age_id'] = school_age_id
    if statistical_area_4:
        querystring['statistical_area_4'] = statistical_area_4
    if school_url:
        querystring['school_url'] = school_url
    if boys_enrolments:
        querystring['boys_enrolments'] = boys_enrolments
    if meshblock:
        querystring['meshblock'] = meshblock
    if bottom_sea_quarter_perc:
        querystring['bottom_sea_quarter_perc'] = bottom_sea_quarter_perc
    if year_range:
        querystring['year_range'] = year_range
    if school_sector:
        querystring['school_sector'] = school_sector
    if lower_middle_sea_quarter_perc:
        querystring['lower_middle_sea_quarter_perc'] = lower_middle_sea_quarter_perc
    if non_teaching_staff:
        querystring['non_teaching_staff'] = non_teaching_staff
    if language_background_other_than_english_yes_perc:
        querystring['language_background_other_than_english_yes_perc'] = language_background_other_than_english_yes_perc
    if fuzzy:
        querystring['fuzzy'] = fuzzy
    if state_electoral_division_name:
        querystring['state_electoral_division_name'] = state_electoral_division_name
    if postcode:
        querystring['postcode'] = postcode
    if governing_body_url:
        querystring['governing_body_url'] = governing_body_url
    if longitude:
        querystring['longitude'] = longitude
    if language_background_other_than_english_no_perc:
        querystring['language_background_other_than_english_no_perc'] = language_background_other_than_english_no_perc
    if commonwealth_electoral_division:
        querystring['commonwealth_electoral_division'] = commonwealth_electoral_division
    if state:
        querystring['state'] = state
    if statistical_area_3_name:
        querystring['statistical_area_3_name'] = statistical_area_3_name
    if statistical_area_2:
        querystring['statistical_area_2'] = statistical_area_2
    if abs_remoteness_area_name:
        querystring['abs_remoteness_area_name'] = abs_remoteness_area_name
    if indigenous_enrolments_perc:
        querystring['indigenous_enrolments_perc'] = indigenous_enrolments_perc
    if language_background_other_than_english_not_stated_perc:
        querystring['language_background_other_than_english_not_stated_perc'] = language_background_other_than_english_not_stated_perc
    if location_age_id:
        querystring['location_age_id'] = location_age_id
    if teaching_staff:
        querystring['teaching_staff'] = teaching_staff
    if upper_middle_sea_quarter_perc:
        querystring['upper_middle_sea_quarter_perc'] = upper_middle_sea_quarter_perc
    if governing_body:
        querystring['governing_body'] = governing_body
    if limit:
        querystring['limit'] = limit
    if icsea:
        querystring['icsea'] = icsea
    if girls_enrolments:
        querystring['girls_enrolments'] = girls_enrolments
    if local_government_area:
        querystring['local_government_area'] = local_government_area
    if calendar_year:
        querystring['calendar_year'] = calendar_year
    if is_id:
        querystring['id'] = is_id
    if full_time_equivalent_enrolments:
        querystring['full_time_equivalent_enrolments'] = full_time_equivalent_enrolments
    if state_electoral_division:
        querystring['state_electoral_division'] = state_electoral_division
    if statistical_area_4_name:
        querystring['statistical_area_4_name'] = statistical_area_4_name
    if local_government_area_name:
        querystring['local_government_area_name'] = local_government_area_name
    if total_enrolments:
        querystring['total_enrolments'] = total_enrolments
    if abs_remoteness_area:
        querystring['abs_remoteness_area'] = abs_remoteness_area
    if statistical_area_3:
        querystring['statistical_area_3'] = statistical_area_3
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-schools-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

