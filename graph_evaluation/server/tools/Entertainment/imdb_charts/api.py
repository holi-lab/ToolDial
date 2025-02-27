import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def chart_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an IMDB Chart by ID
		
		The API endpoint expects an `id` query parameter and the JSON response format is different based on it. We currently support these chart ids:
		* `boxoffice`: Top Box Office (US)
		* `moviemeter`: Most Popular Movies
		* `top`: IMDb Top 250 Movies
		* `top-english-movies`: Top Rated English Movies
		* `tvmeter`: Most Popular TV Shows
		* `toptv`: Top Rated TV Shows
		* `top-rated-indian-movies`: Top Rated Indian Movies
		* `bottom`: Lowest Rated Movies
		* `top-action`: (Top Rated Action Movies
		* `top-adventure`: Top Rated Adventure Movies
		* `top-animation`: Top Rated Animation Movies
		* `top-biography`: Top Rated Biography Movies
		* `top-comedy`: Top Rated Comedy Movies
		* `top-crime`: Top Rated Crime Movies
		* `top-drama`: Top Rated Drama Movies
		* `top-family`: Top Rated Family Movies
		* `top-fantasy`: Top Rated Fantasy Movies
		* `top-film_noir`: Top Rated Film-Noir Movies
		* `top-history`: Top Rated History Movies
		* `top-horror`: Top Rated Horror Movies
		* `top-music`: Top Rated Music Movies
		* `top-musical`: Top Rated Musical Movies
		* `top-mystery`: Top Rated Mystery Movies
		* `top-romance`: Top Rated Romance Movies
		* `top-sci_fi`: Top Rated Sci-Fi Movies
		* `top-sport`: Top Rated Sport Movies
		* `top-thriller`: Top Rated Thriller Movies
		* `top-war`: Top Rated War Movies
		* `top-western`: Top Rated Western Movies"
    
    """
    url = f"https://imdb-charts.p.rapidapi.com/charts"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imdb-charts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

