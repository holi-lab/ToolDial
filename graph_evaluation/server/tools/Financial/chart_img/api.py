import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def advanced_chart(symbol: str='BINANCE:BTCUSDT', studies: str='MACD', format: str='png', theme: str='dark', interval: str='1d', width: int=800, height: int=600, timezone: str='Etc/UTC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint captures TradingView advanced real-time chart image."
    symbol: `The field must be valid tradingview symbol.`

Example: `BINANCE:BTCUSDT`, `BINANCE:BTCUSDTPERP`
        studies: `The field support multiple values.`

| Indicator                             | Study    | Default                       |
| ------------------------------------- | -------- | ----------------------------- |
| Accumulation Distribution             | ACCD     | -                             |
| Advance Decline Ratio                 | ADR      | ADR:9                         |
| Aroon                                 | AROON    | AROON:14                      |
| Average True Range                    | ATR      | ATR:14,RMA                    |
| Awesome Oscillator                    | AO       | -                             |
| Bollinger Bands                       | BB       | BB:20,close,2                 |
| Bollinger Bands %B                    | BBR      | BBR:20,close,2                |
| Bollinger Bands Width                 | BBW      | BBW:20,close,2                |
| Chaikin Money Flow                    | CMF      | CMF:20                        |
| Chaikin Oscillator                    | CO       | CO:3,10                       |
| Chande Momentum Oscillator            | CMO      | CMO:9,close                   |
| Choppiness Index                      | CHOP     | CHOP:14                       |
| Commodity Channel Index               | CCI      | CCI:20,close                  |
| Connors RSI                           | CRSI     | CRSI:3,2,100                  |
| Correlation Coefficient               | CC       | CC:BINANCE:BTCUSDT,close,20   |
| Detrended Price Oscillator            | DPO      | DPO:21,false                  |
| Directional Movement Index            | DMI      | DMI:14,14                     |
| Donchian Channels                     | DONCH    | DONCH:20                      |
| Double EMA                            | DEMA     | DEMA:9                        |
| Ease Of Movement                      | EOM      | EOM:14,10000                  |
| Elders Force Index                    | EFI      | EFI:13                        |
| Exponential Moving Average            | EMA      | EMA:9,close                   |
| Envelope                              | ENV      | ENV:20,10,close,false         |
| Fisher Transform                      | FISHER   | FISHER:9                      |
| Historical Volatility                 | HV       | HV:10                         |
| Hull Moving Average                   | HMA      | HMA:9,close                   |
| Ichimoku Cloud                        | IC       | IC:9,26,52,26                 |
| Keltner Channels                      | KC       | KC:20,1,close,true range      |
| Know Sure Thing                       | KST      | KST:10,15,20,30,10,10,10,15,9 |
| Linear Regression                     | LR       | LR:2,-2,100,close             |
| Moving Average Convergence Divergence | MACD     | MACD:12,26,close,9            |
| Momentum                              | MOM      | MOM:10,close                  |
| Money Flow Index                      | MFI      | MFI:14                        |
| Moon Phases                           | MP       | -                             |
| Moving Average                        | MA       | MA:9,close                    |
| On Balance Volume                     | OBV      | -                             |
| Pivot Points High Low                 | PPHL     | PPHL:10,10                    |
| Pivot Points Standard                 | PPS      | PPS:Traditional,Auto,15       |
| Price Oscillator                      | PPO      | PPO:10,21,close               |
| Price Volume Trend                    | PVT      | -                             |
| Rate of Change                        | ROC      | ROC:9,close                   |
| Relative Strength Index               | RSI      | RSI:14,close                  |
| Vigor Index                           | RVGI     | RVGI:10                       |
| Volatility Index                      | RVI      | RVI:10                        |
| Parabolic SAR                         | SAR      | SAR:0.02,0.02,0.02            |
| SMI Ergodic Indicator                 | SMII     | SMII:20,5,5                   |
| SMI Ergodic Oscillator                | SMIO     | SMIO:20,5,5                   |
| Stochastic                            | STOCH    | STOCH:14,3,3                  |
| Stochastic RSI                        | STOCHRSI | STOCHRSI:3,3,14,14,close      |
| Triple EMA                            | TEMA     | TEMA:9                        |
| Triple Exponential Average            | TRIX     | TRIX:18                       |
| Ultimate Oscillator                   | UO       | UO:7,14,28                    |
| Volatility Stop                       | VSTOP    | VSTOP:20,close,2              |
| Volume                                | VOL      | VOL:20                        |
| Volume Weighted Average               | VWAP     | VWAP:Session,hlc3             |
| Volume Weighted Moving Average        | VWMA     | VWMA:20,close                 |
| Weighted Moving Average               | WMA      | WMA:9,close                   |
| Williams Alligator                    | WA       | WA:13,8,5                     |
| William %R                            | WR       | WR:14,close                   |
| Williams Fractal                      | WF       | -                             |
| Zig Zag                               | ZZ       | -                             |
        format: `The field must be valid format.`

Support: `png`, `jpeg`
        theme: `The field must be valid color theme.`

Support: `light`, `dark`
        interval: `The field must be valid interval.`

Support: `1m`,`3m`,`5m`,`15m`,`30m`,`45m`,`1h`,`2h`,`3h`,`4h`,`1d`,`1w`
        width: `The field must be within the valid range.`

Minimum: `320`

| Plan  | Max width |
| ----- | --------- |
| BASIC | 800 |
        height: `The field must be within the valid range.`

Minimum: `240`

| Plan  | Max height |
| ----- | --------- |
| BASIC | 600 |
        timezone: `The field must be valid timezone.`

| Timezone                       | Description                 |
| ------------------------------ | --------------------------- |
| Etc/UTC                        | UTC                         |
| Pacific/Honolulu               | (UTC-10) Honolulu           |
| America/Juneau                 | (UTC-9) Juneau              |
| America/Los_Angeles            | (UTC-8) Los Angeles         |
| America/Vancouver              | (UTC-8) Vancouver           |
| US/Mountain                    | (UTC-7) Denver              |
| America/Phoenix                | (UTC-7) Phoenix             |
| America/Chicago                | (UTC-6) Chicago             |
| America/Mexico_City            | (UTC-6) Mexico City         |
| America/El_Salvador            | (UTC-6) San Salvador        |
| America/Bogota                 | (UTC-5) Bogota              |
| America/Lima                   | (UTC-5) Lima                |
| America/New_York               | (UTC-5) New York            |
| America/Toronto                | (UTC-5) Toronto             |
| America/Caracas                | (UTC-4) Caracas             |
| America/Argentina/Buenos_Aires | (UTC-3) Buenos Aires        |
| America/Santiago               | (UTC-3) Santiago            |
| America/Sao_Paulo              | (UTC-3) Sao Paulo           |
| Europe/Dublin                  | (UTC) Dublin                |
| Europe/Lisbon                  | (UTC) Lisbon                |
| Europe/London                  | (UTC) London                |
| Atlantic/Reykjavik             | (UTC) Reykjavik             |
| Europe/Amsterdam               | (UTC+1) Amsterdam           |
| Europe/Belgrade                | (UTC+1) Belgrade            |
| Europe/Berlin                  | (UTC+1) Berlin              |
| Europe/Brussels                | (UTC+1) Brussels            |
| Europe/Copenhagen              | (UTC+1) Copenhagen          |
| Africa/Lagos                   | (UTC+1) Lagos               |
| Europe/Luxembourg              | (UTC+1) Luxembourg          |
| Europe/Madrid                  | (UTC+1) Madrid              |
| Europe/Malta                   | (UTC+1) Malta               |
| Europe/Oslo                    | (UTC+1) Oslo                |
| Europe/Paris                   | (UTC+1) Paris               |
| Europe/Rome                    | (UTC+1) Rome                |
| Europe/Stockholm               | (UTC+1) Stockholm           |
| Europe/Warsaw                  | (UTC+1) Warsaw              |
| Europe/Zurich                  | (UTC+1) Zurich              |
| Europe/Athens                  | (UTC+2) Athens              |
| Africa/Cairo                   | (UTC+2) Cairo               |
| Europe/Helsinki                | (UTC+2) Helsinki            |
| Asia/Jerusalem                 | (UTC+2) Jerusalem           |
| Africa/Johannesburg            | (UTC+2) Johannesburg        |
| Europe/Riga                    | (UTC+2) Riga                |
| Europe/Tallinn                 | (UTC+2) Tallinn             |
| Europe/Vilnius                 | (UTC+2) Vilnius             |
| Asia/Bahrain                   | (UTC+3) Bahrain             |
| Europe/Istanbul                | (UTC+3) Istanbul            |
| Asia/Kuwait                    | (UTC+3) Kuwait              |
| Europe/Moscow                  | (UTC+3) Moscow              |
| Asia/Qatar                     | (UTC+3) Qatar               |
| Asia/Riyadh                    | (UTC+3) Riyadh              |
| Asia/Tehran                    | (UTC+3:30) Tehran           |
| Asia/Dubai                     | (UTC+4) Dubai               |
| Asia/Muscat                    | (UTC+4) Muscat              |
| Asia/Ashkhabad                 | (UTC+5) Ashgabat            |
| Asia/Kolkata                   | (UTC+5:30) Kolkata          |
| Asia/Almaty                    | (UTC+6) Almaty              |
| Asia/Bangkok                   | (UTC+7) Bangkok             |
| Asia/Ho_Chi_Minh               | (UTC+7) Ho Chi Minh         |
| Asia/Jakarta                   | (UTC+7) Jakarta             |
| Asia/Chongqing                 | (UTC+8) Chongqing           |
| Asia/Hong_Kong                 | (UTC+8) Hong Kong           |
| Australia/Perth                | (UTC+8) Perth               |
| Asia/Shanghai                  | (UTC+8) Shanghai            |
| Asia/Singapore                 | (UTC+8) Singapore           |
| Asia/Taipei                    | (UTC+8) Taipei              |
| Asia/Seoul                     | (UTC+9) Seoul               |
| Asia/Tokyo                     | (UTC+9) Tokyo               |
| Australia/Brisbane             | (UTC+10) Brisbane           |
| Australia/Adelaide             | (UTC+10:30) Adelaide        |
| Australia/Sydney               | (UTC+11) Sydney             |
| Pacific/Norfolk                | (UTC+12) Norfolk Island     |
| Pacific/Auckland               | (UTC+13) New Zealand        |
| Pacific/Fakaofo                | (UTC+13) Tokelau            |
| Pacific/Chatham                | (UTC+13:45) Chatham Islands |
        
    """
    url = f"https://chart-img.p.rapidapi.com/advanced-chart"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    if studies:
        querystring['studies'] = studies
    if format:
        querystring['format'] = format
    if theme:
        querystring['theme'] = theme
    if interval:
        querystring['interval'] = interval
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chart-img.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mini_chart(width: int=800, format: str='png', symbol: str='BINANCE:BTCUSDT', interval: str='1M', theme: str='dark', height: int=400, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint captures TradingView mini chart image."
    width: `The field must be within the valid range.`

Minimum: `320`

| Plan  | Max width |
| ----- | --------- |
| BASIC | 800 |
        format: `The field must be valid format.`

Support: `png`, `jpeg`
        symbol: `The field must be valid tradingview symbol.`

Example: `BINANCE:BTCUSDT`, `BINANCE:BTCUSDTPERP`
        interval: `The field must be valid interval.`

Support: `1d`,`1M`, `3M`, `1Y`, `5Y`, `all`
        theme: `The field must be valid color theme.`

Support: `light`, `dark`
        height: `The field must be within the valid range.`

Minimum: `220`

| Plan  | Max height |
| ----- | --------- |
| BASIC | 600 |
        
    """
    url = f"https://chart-img.p.rapidapi.com/mini-chart"
    querystring = {}
    if width:
        querystring['width'] = width
    if format:
        querystring['format'] = format
    if symbol:
        querystring['symbol'] = symbol
    if interval:
        querystring['interval'] = interval
    if theme:
        querystring['theme'] = theme
    if height:
        querystring['height'] = height
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chart-img.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

