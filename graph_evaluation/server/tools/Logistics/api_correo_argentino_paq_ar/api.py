import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def precio(pesopaquetekg: int, entrega: str, iva: str, altocm: int, anchocm: int, profundidadcm: int, tipo: str, codigopostaldestino: str, codigopostalorigen: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API MiCorreo ePaq - Correo Argentino
		Precio & Sucursales
		
		Token:"Solicitar mas info por Email"
		Contacto:"info@micorreoar.com"
		Endpoint: "https://demo.micorreoar.com/precio"
		Endpoint: "https://demo.micorreoar.com/precios"
		Endpoint: "https://demo.micorreoar.com/sucursales""
    entrega: D = Domicilio
S = Sucursal
        iva: 1 = Precio con iva
0 = Precio sin iva
        tipo: CP = Envio Clasico
EP = Envio Expreso
        
    """
    url = f"https://api-correo-argentino-paq-ar.p.rapidapi.com/precio/{codigopostalorigen}/{codigopostaldestino}/{iva}/{tipo}/{entrega}/{pesopaquetekg}/{altocm}/{anchocm}/{profundidadcm}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-correo-argentino-paq-ar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

