import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gera_vis_o_da_rua(cnpj: str, fov: int=None, height: int=None, width: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gera Visão da Rua"
    
    """
    url = f"https://consulta-cnpj-tempo-real.p.rapidapi.com/office/{cnpj}/street"
    querystring = {}
    if fov:
        querystring['fov'] = fov
    if height:
        querystring['height'] = height
    if width:
        querystring['width'] = width
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-cnpj-tempo-real.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gera_mapa_dos_arredores(cnpj: str, height: int=None, zoom: int=None, width: int=None, scale: int=None, type: str='hybrid', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gera Mapa dos Arredores"
    
    """
    url = f"https://consulta-cnpj-tempo-real.p.rapidapi.com/office/{cnpj}/map"
    querystring = {}
    if height:
        querystring['height'] = height
    if zoom:
        querystring['zoom'] = zoom
    if width:
        querystring['width'] = width
    if scale:
        querystring['scale'] = scale
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-cnpj-tempo-real.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gera_comprovante_rfb(taxid: str, pages: str='REGISTRATION,MEMBERS', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gera Comprovante da Receita Federal em PDF"
    
    """
    url = f"https://consulta-cnpj-tempo-real.p.rapidapi.com/rfb/certificate"
    querystring = {'taxId': taxid, }
    if pages:
        querystring['pages'] = pages
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-cnpj-tempo-real.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def consulta_cep(cep: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Consulta CEP"
    
    """
    url = f"https://consulta-cnpj-tempo-real.p.rapidapi.com/zip/{cep}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-cnpj-tempo-real.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def consulta_estabelecimento(cnpj: str, maxage: int=30, simpleshistory: bool=None, registrations: str='BR', simples: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Consulta CNPJ unificada a Receita Federal, Simples Nacional e Cadastro de Contribuintes"
    registrations: UFs separadas por vírgula para adicionar informações do Cadastro de Contribuintes, utilize 'BR' para considerar todas.
        
    """
    url = f"https://consulta-cnpj-tempo-real.p.rapidapi.com/office/{cnpj}"
    querystring = {}
    if maxage:
        querystring['maxAge'] = maxage
    if simpleshistory:
        querystring['simplesHistory'] = simpleshistory
    if registrations:
        querystring['registrations'] = registrations
    if simples:
        querystring['simples'] = simples
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-cnpj-tempo-real.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def consulta_cadastro_de_contribuintes(taxid: str, maxage: int=30, states: str='BR', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Consulta as Inscrições Estaduais no Cadastro Centralizado de Contribuintes"
    states: UFs separadas por vírgula para adicionar informações do Cadastro de Contribuintes, utilize 'BR' para considerar todas.
        
    """
    url = f"https://consulta-cnpj-tempo-real.p.rapidapi.com/ccc"
    querystring = {'taxId': taxid, }
    if maxage:
        querystring['maxAge'] = maxage
    if states:
        querystring['states'] = states
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-cnpj-tempo-real.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gera_comprovante_simples(taxid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gera Comprovante em PDF do Simples Nacional"
    
    """
    url = f"https://consulta-cnpj-tempo-real.p.rapidapi.com/simples/certificate"
    querystring = {'taxId': taxid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-cnpj-tempo-real.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def consulta_simples_nacional(taxid: str, maxage: int=30, history: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Consulta Simples Nacional"
    
    """
    url = f"https://consulta-cnpj-tempo-real.p.rapidapi.com/simples"
    querystring = {'taxId': taxid, }
    if maxage:
        querystring['maxAge'] = maxage
    if history:
        querystring['history'] = history
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-cnpj-tempo-real.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def consulta_receita_federal(taxid: str, maxage: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Consulta Receita Federal"
    
    """
    url = f"https://consulta-cnpj-tempo-real.p.rapidapi.com/rfb"
    querystring = {'taxId': taxid, }
    if maxage:
        querystring['maxAge'] = maxage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-cnpj-tempo-real.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def consulta_pessoa(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Consulta Pessoa"
    
    """
    url = f"https://consulta-cnpj-tempo-real.p.rapidapi.com/person/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-cnpj-tempo-real.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def consulta_empresa(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Consulta Empresa"
    id: 8 primeiros dígitos do CNPJ
        
    """
    url = f"https://consulta-cnpj-tempo-real.p.rapidapi.com/company/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-cnpj-tempo-real.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def consulta_unificada(cnpj: str, company_max_age: int=30, simples_max_age: int=30, sintegra_max_age: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retorna todos os dados presentes no Comprovante de Inscrição do Cadastro Nacional da Pessoa Jurídica junto a Receita Federal"
    cnpj: CNPJ da Empresa
        company_max_age: Idade máxima dos dados relativos da Receita Federal
        simples_max_age: Ativa e define a idade máxima dos dados relativos ao Simples Nacional
        sintegra_max_age: Ativa e define a idade máxima dos dados relativos ao Cadastro de Contribuintes
        
    """
    url = f"https://consulta-cnpj-tempo-real.p.rapidapi.com/companies/{cnpj}"
    querystring = {}
    if company_max_age:
        querystring['company_max_age'] = company_max_age
    if simples_max_age:
        querystring['simples_max_age'] = simples_max_age
    if sintegra_max_age:
        querystring['sintegra_max_age'] = sintegra_max_age
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-cnpj-tempo-real.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

