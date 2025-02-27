import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def opera_list(genre_name_hy_in: str=None, genre_name_ru_in: str=None, genre_name_en_in: str=None, year_in: str=None, published_at_gt: str=None, catalogue_id: str=None, genre_name_en: str=None, published_at: str=None, published_at_gte: str=None, genre_name_hy: str=None, year: str=None, catalogue_id_in: str=None, genre_name_ru: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    genre_name_hy_in: Multiple values may be separated by commas.
        genre_name_ru_in: Multiple values may be separated by commas.
        genre_name_en_in: Multiple values may be separated by commas.
        year_in: Multiple values may be separated by commas.
        catalogue_id_in: Multiple values may be separated by commas.
        
    """
    url = f"https://testing-options.p.rapidapi.com/opera/"
    querystring = {}
    if genre_name_hy_in:
        querystring['genre__name_hy__in'] = genre_name_hy_in
    if genre_name_ru_in:
        querystring['genre__name_ru__in'] = genre_name_ru_in
    if genre_name_en_in:
        querystring['genre__name_en__in'] = genre_name_en_in
    if year_in:
        querystring['year__in'] = year_in
    if published_at_gt:
        querystring['published_at__gt'] = published_at_gt
    if catalogue_id:
        querystring['catalogue_id'] = catalogue_id
    if genre_name_en:
        querystring['genre__name_en'] = genre_name_en
    if published_at:
        querystring['published_at'] = published_at
    if published_at_gte:
        querystring['published_at__gte'] = published_at_gte
    if genre_name_hy:
        querystring['genre__name_hy'] = genre_name_hy
    if year:
        querystring['year'] = year
    if catalogue_id_in:
        querystring['catalogue_id__in'] = catalogue_id_in
    if genre_name_ru:
        querystring['genre__name_ru'] = genre_name_ru
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing-options.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

