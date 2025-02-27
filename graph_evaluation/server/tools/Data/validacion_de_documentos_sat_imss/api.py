import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def decifradatosenbase64(base64data: str, documentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Decifra los datos que estan en base64 de la respuesta de la validacion de documento"
    
    """
    url = f"https://validacion-de-documentos-sat-imss1.p.rapidapi.com/evalidaDocumentosApi/api/Documents/decodeData"
    querystring = {'base64Data': base64data, 'documentId': documentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "validacion-de-documentos-sat-imss1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listadetiposdedocumentosquetienenvalidaci_n(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lista de documentos disponibles para validar"
    
    """
    url = f"https://validacion-de-documentos-sat-imss1.p.rapidapi.com/evalidaDocumentosApi/api/documents"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "validacion-de-documentos-sat-imss1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

