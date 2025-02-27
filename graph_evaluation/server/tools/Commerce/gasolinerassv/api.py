import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def precios_de_venta_gasolineras(gasolineras_por_pagina: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Devuelve el listado de gasolineras, precios de venta y ubicación, el endpoint devolverá los resultados de forma paginada. Un dato a tomar en cuenta es que existen dos query params obligatorios.
		
		- pagina: El numero de pagina.
		- gasolineras_por_pagina: El numero de gasolineras para cada pagina, debe ser entre 5-100 y un multiplo de 5
		
		El endpoint devolverá también la cantidad de páginas disponibles en base a la cantidad de gasolineras por pagina que el usuario coloque."
    
    """
    url = f"https://gasolinerassv.p.rapidapi.com/"
    querystring = {'gasolineras_por_pagina': gasolineras_por_pagina, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gasolinerassv.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hist_rico_de_precios_de_venta(pagina: str, gasolineras_por_pagina: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Devuelve el listado histórico de gasolineras, precios de venta y ubicación, el endpoint devolverá los resultados de forma paginada. Un dato a tomar en cuenta es que existen dos query params obligatorios.
		
		- pagina: El numero de pagina.
		- gasolineras_por_pagina: El numero de gasolineras para cada pagina, debe ser entre 5-1000 y un multiplo de 5
		
		El endpoint devolverá también la cantidad de páginas disponibles en base a la cantidad de gasolineras por pagina que el usuario coloque."
    
    """
    url = f"https://gasolinerassv.p.rapidapi.com/registros_historicos"
    querystring = {'pagina': pagina, 'gasolineras_por_pagina': gasolineras_por_pagina, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gasolinerassv.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

