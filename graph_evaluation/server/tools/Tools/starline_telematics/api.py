import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def userinfov2(user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Получение списка устройств, принадлежащих пользователю или устройств, доступ к которым предоставлен пользователю другими пользователями. Ответ содержит полное состояние устройства для обновление текущего состояния устройств. Для получения данных устройства необходимо передать идентификатор пользователя user_id и cookie, полученный при авторизации пользователя. Данный метод устарел, желательно использовать /json/v3/user/{user_id}/data. В качестве примера можно использовать Python-скрипт get_user_info.py из репозитория <a href="https://gitlab.com/starline/openapi">gitlab.com/starline/openapi</a>. В результате выполнения скрипт выводит в консоль ответ на запрос. Пример выполнения запроса с помощью curl: <br><br><code>curl "https://developer.starline.ru/json/v2/user/1116/user_info" --cookie 'slnet=780D719C7F54465FB281B4DD65498168'</code>
		"
    user_id: Идентификатор пользователя в SLNet
        
    """
    url = f"https://starline-telematics.p.rapidapi.com/json/v2/user/{user_id}/user_info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starline-telematics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def libraryevents(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "В ответе на запрос приходят все существующие события с полным описанием. Пример выполнения запроса с помощью curl: <br><br><code>curl "https://developer.starline.ru/json/v3/library/events"</code>
		"
    
    """
    url = f"https://starline-telematics.p.rapidapi.com/json/v3/library/events"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starline-telematics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def userinfo(user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Получение списка устройств, принадлежащих пользователю или устройств, доступ к которым предоставлен пользователю другими пользователями. Ответ содержит полное состояние устройств. Данный метод устарел, желательно использовать /json/v3/user/{user_id}/data. В качестве примера можно использовать Python-скрипт get_user_info.py из репозитория <a href="https://gitlab.com/starline/openapi">gitlab.com/starline/openapi</a>. В результате выполнения скрипт выводит в консоль ответ на запрос. Пример выполнения запроса с помощью curl: <br><br><code>curl "https://developer.starline.ru/json/user/4568857/user_info" --cookie 'slnet=780D719C7F54465FB281B4DD65498168'</code>
		"
    user_id: Идентификатор пользователя в SLNet
        
    """
    url = f"https://starline-telematics.p.rapidapi.com/json/user/{user_id}/user_info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starline-telematics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def deviceobdparams(device_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Запрос данных, полученных от автомобиля и хранящихся в кеше. Любой из возвращаемых параметров (fuel, errors, mileage) может быть null. Это значит, что либо он еще не считан системой с автомобиля, либо данное ТС не поддерживается установленной версией CAN библиотеки, либо данные невозможно считать через CAN. Пример выполнения запроса с помощью curl: <br><br><code>curl "https://developer.starline.ru/json/v1/device/4568857/obd_params" --cookie 'slnet=15CFA19BFD3E8884C716FFA4E39AF866'</code>
		"
    device_id: Идентификатор устройства в SLNet
        
    """
    url = f"https://starline-telematics.p.rapidapi.com/json/device/{device_id}/obd_params"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starline-telematics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def libraryeventsid(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "В ответе на запрос приходит событие с полным описанием. Пример выполнения запроса с помощью curl: <br><br><code>curl "https://developer.starline.ru/json/v3/library/events/307"</code>
		"
    id: Идентификатор события
        
    """
    url = f"https://starline-telematics.p.rapidapi.com/json/v3/library/events/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starline-telematics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def userdevices(user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ответ на запрос содержит список устройств пользователя, а также данные о каждом устройстве. Данный метод устарел, желательно использовать /json/v3/user/{user_id}/data. Пример выполнения запроса с помощью curl: <br><br><code>curl "https://developer.starline.ru/json/v1/user/1116/devices" --cookie 'slnet=780D719C7F54465FB281B4DD65498168'</code>
		"
    user_id: Идентификатор пользователя в SLNet
        
    """
    url = f"https://starline-telematics.p.rapidapi.com/json/v1/user/{user_id}/devices"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starline-telematics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def deviceobderrors(device_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Запрос данных об ошибках OBD, полученных от автомобиля и хранящихся в кеше. Пример выполнения запроса с помощью curl: <br><br><code>curl "https://developer.starline.ru/json/v1/device/4568857/obd_errors" --cookie 'slnet=780D719C7F54465FB281B4DD65498168'</code>
		"
    device_id: Идентификатор устройства в SLNet
        
    """
    url = f"https://starline-telematics.p.rapidapi.com/json/device/{device_id}/obd_errors"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starline-telematics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def device(device_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Получение суммарного отчета о состоянии устройства. Данный метод устарел, желательно использовать /json/v3/device/{device_id}/data. Пример выполнения запроса с помощью curl: <br><br><code>curl "https://developer.starline.ru/json/v2/device/356306052111516" --cookie 'slnet=780D719C7F54465FB281B4DD65498168'</code>
		"
    device_id: Идентификатор устройства в SLNet
        
    """
    url = f"https://starline-telematics.p.rapidapi.com/json/v2/device/{device_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starline-telematics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def digest_authentication(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Токен авторизации предварительно необходимо получить на сервере StarLineID. Полученный в результате успешного выполнения команды cookie необходимо использовать в методах WebAPI для прохождения авторизации. Для того чтобы сервер понимал, что аутентификацию следует выполнять digest методом, клиент должен передать в запросе заголовок 'DigestAuth:true'. username – slid-token, password – идентификатор пользователя на StarLineID сервере (slid user_id). Пример выполнения запроса с помощью curl: <br><br><code>curl "https://developer.starline.ru/json/v3/device/357803045497365/data" -v --digest -u '418db3fe902abe4eb4a6d92ad37ab7a7_276388:276388' -H 'DigestAuth:true'</code>
		"
    
    """
    url = f"https://starline-telematics.p.rapidapi.com/any_api_method"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starline-telematics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def userdata(user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Получение списка устройств, принадлежащих пользователю или устройств, доступ к которым предоставлен пользователю другими пользователями. Ответ содержит полное состояние устройства для обновление текущего состояния устройств. Для получения данных устройства необходимо передать идентификатор пользователя user_id и cookie, полученный при авторизации пользователя. Пример выполнения запроса с помощью curl: <br><br><code>curl "https://developer.starline.ru/json/v3/user/1116/data" --cookie 'slnet=780D719C7F54465FB281B4DD65498168'</code>
		"
    user_id: Идентификатор пользователя в SLNet
        
    """
    url = f"https://starline-telematics.p.rapidapi.com/json/v3/user/{user_id}/data"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starline-telematics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def usermobiledevices(user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Запрос на получение информации о мобильных устройствах пользователя. Пример выполнения запроса с помощью curl: <br><br><code>curl "https://developer.starline.ru/json/v1/user/1116/mobile_devices" --cookie 'slnet=780D719C7F54465FB281B4DD65498168'</code>
		"
    user_id: Идентификатор пользователя в SLNet
        
    """
    url = f"https://starline-telematics.p.rapidapi.com/json/v1/user/{user_id}/mobile_devices"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starline-telematics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def devicedata(device_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ответ содержит полное состояние устройства для обновления текущего состояния устройств. Для получения данных устройства необходимо передать идентификатор устройства device_id и cookie, полученный при авторизации пользователя. Пример выполнения запроса с помощью curl: <br><br><code>curl "https://developer.starline.ru/json/v3/device/357976061290482/data" --cookie 'slnet=15CFA19BFD3E8884C716FFA4E39AF866'</code>
		"
    device_id: Идентификатор устройства в SLNet
        
    """
    url = f"https://starline-telematics.p.rapidapi.com/json/v3/device/{device_id}/data"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starline-telematics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def deviceposition(device_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Запрос данных, о текущем (последнем зафиксированном) местоположении устройства. В случае, если в базе данных нет информации о местоположении устройства, блок position будет возвращен пустым. Данный метод устарел, желательно использовать /json/v3/device/{device_id}/data. Пример выполнения запроса с помощью curl: <br><br><code>curl "https://developer.starline.ru/json/v1/device/4568857/position" --cookie 'slnet=15CFA19BFD3E8884C716FFA4E39AF866'</code>
		"
    device_id: Идентификатор устройства в SLNet
        
    """
    url = f"https://starline-telematics.p.rapidapi.com/json/v1/device/{device_id}/position"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starline-telematics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def devicestate(device_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Получение текущего состояния устройства или последнего известного состояния устройства. Данный метод устарел, желательно использовать /json/v3/device/{device_id}/data. Пример выполнения запроса с помощью curl: <br><br><code>curl "https://developer.starline.ru/json/v2/device/356306052111516/state" --cookie 'slnet=15CFA19BFD3E8884C716FFA4E39AF866'</code>
		"
    device_id: Идентификатор устройства в SLNet
        
    """
    url = f"https://starline-telematics.p.rapidapi.com/json/v2/device/{device_id}/state"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starline-telematics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def controls_library(device_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Метод используется для получения всех возможных для данного типа устройств кнопок управления. Возвращается список кнопок с текстовым описанием их предназначения. Текстовое описание выводится на языке, выбранном в личном кабинете или переданном в параметрах данной сессии. Пример выполнения запроса с помощью curl: <br><br><code>curl "https://developer.starline.ru/json/device/45657/ctrls_library" --cookie 'slnet=15CFA19BFD3E8884C716FFA4E39AF866'</code>
		"
    device_id: Идентификатор устройства в SLNet
        
    """
    url = f"https://starline-telematics.p.rapidapi.com/json/device/{device_id}/ctrls_library"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starline-telematics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def deviceinfoget(device_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Метод для получения состояния устройства. Данный метод устарел, желательно использовать /json/v3/device/{device_id}/data. Пример выполнения запроса с помощью curl: <br><br><code>curl "https://developer.starline.ru/json/v1/device/4568857/info" --cookie 'slnet=15CFA19BFD3E8884C716FFA4E39AF866'</code>
		"
    device_id: Идентификатор устройства в SLNet
        
    """
    url = f"https://starline-telematics.p.rapidapi.com/json/v1/device/{device_id}/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starline-telematics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def deviceasyncget(device_id: int, cmd_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Метод предназначен для получения статуса выполнения команды, которая была отправлена ранее с помощью метода /json/v2/device/{device_id}/async. Для запроса необходимо передать идентификатор устройства device_id, идентификатор команды cmd_id и cookie, полученный при авторизации пользователя. Пример выполнения запроса с помощью curl: <br><br><code>curl "https://developer.starline.ru/json/v2/device/9854082/async/AEVXNX0QLGMQRLVOW66HYAB6OYIWO6WW" --cookie 'slnet=15CFA19BFD3E8884C716FFA4E39AF866'</code>
		"
    device_id: Идентификатор устройства в SLNet
        cmd_id: Идентификатор команды управления
        
    """
    url = f"https://starline-telematics.p.rapidapi.com/json/v2/device/{device_id}/async/{cmd_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starline-telematics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getappcode(secret: str, appid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Получение кода приложения для дальнейшего получения токена. Срок годности кода приложения – 1 час. Идентификатор приложения и пароль можно получить в личном кабинете в разделе "Разработчикам" на <a href="https://my.starline.ru">my.starline.ru</a>. В качестве примера можно использовать Python-скрипт get_app_code.py из репозитория <a href="https://gitlab.com/starline/openapi">gitlab.com/starline/openapi</a>. В результате выполнения скрипт выводит в консоль код приложения. Пример выполнения запроса с помощью curl: <br><br><code>curl "https://id.starline.ru/apiV3/application/getCode?appId=2&secret=d1e0c6cbd6c93999e5072cee4deed9c9" -i</code>
		"
    secret: MD5 от пароля приложения
        appid: Идентификатор приложения
        
    """
    url = f"https://starline-telematics.p.rapidapi.com/apiV3/application/getCode"
    querystring = {'secret': secret, 'appId': appid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starline-telematics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getapptoken(secret: str, appid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Получение токена приложения для дальнейшей авторизации. Время жизни токена приложения – 4 часа. Идентификатор приложения и пароль можно получить в личном кабинете в разделе "Разработчикам" на <a href="https://my.starline.ru">my.starline.ru</a>. В качестве примера можно использовать Python-скрипт get_app_token.py из репозитория <a href="https://gitlab.com/starline/openapi">gitlab.com/starline/openapi</a>. В результате выполнения скрипт выводит в консоль токен приложения. Пример выполнения запроса с помощью curl: <br><br><code>curl "https://id.starline.ru/apiV3/application/getToken?appId=2&secret=2646398c5a7bb70b7f4eadca7c4f62cb" -i</code>
		"
    secret: MD5 от результата конкатенации пароля приложения и кода, полученного через /application/getCode
        appid: Идентификатор приложения
        
    """
    url = f"https://starline-telematics.p.rapidapi.com/apiV3/application/getToken"
    querystring = {'secret': secret, 'appId': appid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starline-telematics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def settingssettings(device_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Получение текущего списка настроек или последнего известного списка настроек устройств 3-5 поколения. Для получения данных устройства необходимо передать идентификатор устройства id и cookie, полученный при авторизации пользователя. Пример выполнения запроса с помощью curl: <br><br><code>curl "https://developer.starline.ru/json/v3/device/4568857/settings" --cookie 'slnet=780D719C7F54465FB281B4DD65498168'</code>
		"
    device_id: Идентификатор устройства в SLNet
        
    """
    url = f"https://starline-telematics.p.rapidapi.com/json/v3/device/{device_id}/settings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starline-telematics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

