import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_rfc(apellido_materno: str, apellido_paterno: str, fecha: str, nombre: str, solo_homoclave: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Regresa el RFC con homoclave, o sólo la homoclave."
    apellido_materno: El apellido materno de la persona física
        apellido_paterno: El apellido paterno de la persona física
        fecha: La fecha de nacimiento de la persona física en formato 
aaaa-mm-dd ó yyyy-mm-dd
        nombre: El nombre de la persona física
        solo_homoclave: Si sólo quiere calcular la homoclave, o todo el rfc

1 => Calcúla sólo la homoclave => LS4
0 => Regresa todo el rfc => HEGJ880317LS4
        
    """
    url = f"https://jfhe88-rfc-generator-mexico.p.rapidapi.com/rest1/rfc/get"
    querystring = {'apellido_materno': apellido_materno, 'apellido_paterno': apellido_paterno, 'fecha': fecha, 'nombre': nombre, }
    if solo_homoclave:
        querystring['solo_homoclave'] = solo_homoclave
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jfhe88-rfc-generator-mexico.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

