import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_anime_detail(is_id: str, fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will allow a user to get the data of a specific anime. It will take the anime's mal_id as a parameter and return the anime's data in JSON format. The data returned will include the anime's title, aired episodes, genre, and a brief synopsis."
    id: Insert the anime `mal_id`
        fields: Get the fields that you want.
Disponible fields:  

- * (to all data)
- id,
- title, 
- main_picture,
- start_date,
- end_data,
- synopsis,
- mean,
- rank,
- popularity,
- `num_list_users,`
- `num_scoring_users`,
- status,
- genres,
- num_episodes,
- source,
- studios,
- volume,
- chapter,
- light_novel,
- media_type,
- mal_id,
- broadcast,
- statistics,
- related_manga,
- related_anime,
- season,
- year,
- title_english,
- title_japanese,
- synonyms
        
    """
    url = f"https://animes5.p.rapidapi.com/anime/{is_id}"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animes5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_animes(year_greater: int=None, media_type: str=None, studio: str=None, year_less: int=None, nsfw: str=None, status: str=None, limit: int=None, q: str=None, genre: str=None, sort: bool=None, offset: int=None, season: str=None, fields: str=None, year_equal: int=None, source: str=None, order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows for the retrieval of data about anime. The query parameters include q (the search query), limit (the number of results per page), offset (the number of results to skip), ascending (sort order), order (sort order), `media_type `(anime type), studio (studio name), source (source material), status (`currently_airing `or `finished_airing`), genre, season, `year_equal `(equal to a year specified), `year_less `(less than a year specified), `year_greater `(greater than a year specified), language, and nsfw (not safe for work content)."
    year_greater: Find the animes with the year greater than the given one

        media_type: The media type of the anime.
They are:

- music,
- tv,
- ona,
- ova,
- movie,
- special
        studio: Filter the animes by the studio
        year_less: Find the animes with the year less than the given one

        nsfw: Activate the nsfw content. If you don't want it, don't use it
        status: Filter the animes by the actual status of the anime.

- currently_airing,
- finished_airing,
- ` not_yet_aired`
        limit: Limit the number of data recieved
        q: Search the anime title in english or japanase.
        genre: Filter the animes by the genre
        sort: Especify the order if asc or desc.
        offset: Recieve the data after the number of the offset.
        season: Filter the animes by their season
        fields: Get the fields that you want.
Disponible fields:  

- * (to all data)
- id,
- title, 
- main_picture,
- start_date,
- end_data,
- synopsis,
- mean,
- rank,
- popularity,
- `num_list_users,`
- `num_scoring_users`,
- status,
- genres,
- num_episodes,
- source,
- studios,
- volume,
- chapter,
- light_novel,
- media_type,
- mal_id,
- broadcast,
- statistics,
- related_manga,
- related_anime,
- season,
- year,
- title_english,
- title_japanese,
- synonyms
        year_equal: Filter the anime by the year release
        source: Filter the anime by the source.
They are:

- manga,
- visual_novel,
- novel,
- `4_koma_manga`,
- book,
- card_game,
- game,
- light_novel,
- mixed_media,
- music,
- novel,
- original,
- web_manga,
- web_novel,
- visual_novel,
- picture_book,
- other
        order: Order the data with the field tha you want
        
    """
    url = f"https://animes5.p.rapidapi.com/"
    querystring = {}
    if year_greater:
        querystring['year_greater'] = year_greater
    if media_type:
        querystring['media_type'] = media_type
    if studio:
        querystring['studio'] = studio
    if year_less:
        querystring['year_less'] = year_less
    if nsfw:
        querystring['nsfw'] = nsfw
    if status:
        querystring['status'] = status
    if limit:
        querystring['limit'] = limit
    if q:
        querystring['q'] = q
    if genre:
        querystring['genre'] = genre
    if sort:
        querystring['sort'] = sort
    if offset:
        querystring['offset'] = offset
    if season:
        querystring['season'] = season
    if fields:
        querystring['fields'] = fields
    if year_equal:
        querystring['year_equal'] = year_equal
    if source:
        querystring['source'] = source
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animes5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

