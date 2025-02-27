import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def testing_endpoint(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "testing endpoint"
    
    """
    url = f"https://20211230-testing-upload-swagger.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "20211230-testing-upload-swagger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def open_api_v1_0_indoor_air_quality_iot(devicename: str, co2range: int, dataowner: str, devicetype: str, accuracy: str, temperaturerange: int, model: str, humidityrange: int, weight: int, batterylift: int, dataexpiry: str, datacreated: str, productlifecycle: int, location: str, devicedimension: str, devicesubtype: str, manufacture: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Apply for this API to access the steps for indoor air quality information about IoT device - v2"
    devicename: IoT device name
        co2range: Define the minimum and maximun ppm for IAQ device
        dataowner: Data owner information, can be name or origization
        devicetype: Reference the data directory - Device Type tab
        accuracy: Three dimension of the IoT device
        temperaturerange: Define the minimum and maximun temperature for IAQ device
        model: Model number of the IoT device
        humidityrange: Define the minimum and maximun humidity for IAQ device
        weight: Weight of the IoT device
        batterylift: Battery life as a month
        dataexpiry: Data life cycle information of the data expiry datetime. <i>Format</i>: "yyyy-MM-dd_HH:mm:ss"
        datacreated: Data life cycle information of the data created datetime. <i>Format</i>: "yyyy-MM-dd_HH:mm:ss"
        productlifecycle: Product life cycle as a month
        location: Reference the data directory - Location tab for input format
        devicedimension: Three dimension of the IoT device
        devicesubtype: Reference the data directory - Device Type tab
        manufacture: IoT device date of manufacture
        
    """
    url = f"https://20211230-testing-upload-swagger.p.rapidapi.com/open-api/v1.0/indoor-air-quality/iot/"
    querystring = {'DeviceName': devicename, 'CO2Range': co2range, 'DataOwner': dataowner, 'DeviceType': devicetype, 'Accuracy': accuracy, 'TemperatureRange': temperaturerange, 'Model': model, 'HumidityRange': humidityrange, 'Weight': weight, 'BatteryLift': batterylift, 'DataExpiry': dataexpiry, 'DataCreated': datacreated, 'ProductLifeCycle': productlifecycle, 'Location': location, 'DeviceDimension': devicedimension, 'DeviceSubtype': devicesubtype, }
    if manufacture:
        querystring['Manufacture'] = manufacture
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "20211230-testing-upload-swagger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

