import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def auto_complete_deprecated(text: str, type: str=None, lang: str='en', exchange: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get suggestion by term or phrase"
    text: Any term or phrase that you are familiar with
        type: One of the following : stock|futures|forex|index|bond|economic|bitcoin|crypto. Separated by comma for multiple options. Ex : bitcoin,crypto,stock,...
        lang: The language code
        exchange: The value of \"value\" field returned in .../exchanges/list endpoint
        
    """
    url = f"https://trading-view.p.rapidapi.com/auto-complete"
    querystring = {'text': text, }
    if type:
        querystring['type'] = type
    if lang:
        querystring['lang'] = lang
    if exchange:
        querystring['exchange'] = exchange
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trading-view.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list(locale: str='en', symbol: str=None, country: str='us', per_page: int=20, category: str='base', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List latest news with options and filters"
    locale: The language code
        symbol: List ideas posts related to specified symbol. Ex : NASDAQ:AAPL
        country: The country code, only functionable if the market parameter is stock.
        per_page: The number of items per response, for paging purpose
        category: Leave empty or one of the following : base|stock|cryto|forex|index|futures|bond|economic
        page: The page index, for paging purpose
        
    """
    url = f"https://trading-view.p.rapidapi.com/news/list"
    querystring = {}
    if locale:
        querystring['locale'] = locale
    if symbol:
        querystring['symbol'] = symbol
    if country:
        querystring['country'] = country
    if per_page:
        querystring['per_page'] = per_page
    if category:
        querystring['category'] = category
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trading-view.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_movers(exchange: str, name: str='volume_gainers', locale: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get movers in different exchanges and different categories"
    exchange: One of the following : US|ADX|AMEX|ATHEX|EURONEXT&#95;BRU|ASX|BAHRAIN|BCBA|BCS|BELEX|BET|BER|BIST|BME|BMFBOVESPA|BMV|BSE|BSSE|BVC|BVL|BVB|BVCV|BX|CSE|DFM|DUS|OMXCOP|OMXTSE|OMXHEX|EGX|EURONEXT|EURONEXT&#95;PAR|EURONEXT&#95;AMS|EURONEXT&#95;LIS|FWB|GPW|HAN|HKEX|HNX|HOSE|IDX|JSE|LSE|LSIN|MIL|MOEX|MYX|MUN|NAG|NASDAQ|NEO|NEWCONNECT|NGM|NSE|NSENG|NYSE|NZX|KRX|OTC|OMXICE|OMXRSE|OMXSTO|OMXVSE|LUXSE|OSL|PSE|QSE|SGX|SIX|SWB|SZSE|SSE|SET|TADAWUL|TASE|TPEX|TSE|TSX|TSXV|TWSE|UPCOM|XETR
        name: One of the following : volume&#95;gainers|percent&#95;change&#95;gainers|percent&#95;change&#95;loosers|percent&#95;range&#95;gainers|percent&#95;range&#95;loosers|gap&#95;gainers|gap&#95;loosers|percent&#95;gap&#95;gainers|percent&#95;gap&#95;loosers
        locale: The language code
        
    """
    url = f"https://trading-view.p.rapidapi.com/market/get-movers"
    querystring = {'exchange': exchange, }
    if name:
        querystring['name'] = name
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trading-view.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stocks_get_financials(symbol: str, columns: str, screenername: str='america', lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get financials information related to a ticker or symbol"
    symbol: The format is "exchange:symbol". Ex : NASDAQ:TSLA
        columns: One of the following (Separated by comma for multiple options) : ADR,ADX,ADX-DI,ADX-DI|1,ADX-DI|120,ADX-DI|15,ADX-DI|1M,ADX-DI|1W,ADX-DI|240,ADX-DI|30,ADX-DI|5,ADX-DI|60,ADX+DI,ADX|1,ADX|120,ADX|15,ADX|1M,ADX|1W,ADX|240,ADX|30,ADX|5,ADX|60,after&#95;tax&#95;margin,AO,AO|1,AO|120,AO|15,AO|1M,AO|1W,AO|240,AO|30,AO|5,AO|60,Aroon.Down,Aroon.Up,ATR,average&#95;volume,average&#95;volume&#95;10d&#95;calc,average&#95;volume&#95;30d&#95;calc,average&#95;volume&#95;60d&#95;calc,average&#95;volume&#95;90d&#95;calc,base&#95;currency&#95;logoid,basic&#95;elements,basic&#95;eps&#95;net&#95;income,BB.lower,BB.upper,BBPower,BBPower|1,BBPower|120,BBPower|15,BBPower|1M,BBPower|1W,BBPower|240,BBPower|30,BBPower|5,BBPower|60,beta&#95;1&#95;year,beta&#95;3&#95;year,beta&#95;5&#95;year,CCI20,CCI20|1,CCI20|120,CCI20|15,CCI20|1M,CCI20|1W,CCI20|240,CCI20|30,CCI20|5,CCI20|60,ChaikinMoneyFlow,change,change&#95;abs,change&#95;abs|1,change&#95;abs|15,change&#95;abs|1M,change&#95;abs|1W,change&#95;abs|240,change&#95;abs|5,change&#95;abs|60,change&#95;from&#95;open,change&#95;from&#95;open&#95;abs,change|1,change|15,change|1M,change|1W,change|240,change|5,change|60,close,close|1,close|120,close|15,close|1M,close|1W,close|240,close|30,close|5,close|60,country,currency&#95;logoid,current&#95;ratio,debt&#95;to&#95;assets,debt&#95;to&#95;equity,debt&#95;to&#95;equity&#95;fq,description,dividend&#95;yield&#95;recent,dividends&#95;paid,dividends&#95;per&#95;share&#95;fq,dividends&#95;yield,DonchCh20.Lower,DonchCh20.Upper,dps&#95;common&#95;stock&#95;prim&#95;issue&#95;fy,earnings&#95;per&#95;share&#95;basic&#95;ttm,earnings&#95;per&#95;share&#95;diluted&#95;ttm,earnings&#95;per&#95;share&#95;forecast&#95;next&#95;fq,earnings&#95;per&#95;share&#95;fq,earnings&#95;release&#95;date,earnings&#95;release&#95;next&#95;date,ebitda,elements,EMA10,EMA10|1,EMA10|120,EMA10|15,EMA10|1M,EMA10|1W,EMA10|240,EMA10|30,EMA10|5,EMA10|60,EMA100,EMA100|1,EMA100|120,EMA100|15,EMA100|1M,EMA100|1W,EMA100|240,EMA100|30,EMA100|5,EMA100|60,EMA20,EMA20|1,EMA20|120,EMA20|15,EMA20|1M,EMA20|1W,EMA20|240,EMA20|30,EMA20|5,EMA20|60,EMA200,EMA200|1,EMA200|120,EMA200|15,EMA200|1M,EMA200|1W,EMA200|240,EMA200|30,EMA200|5,EMA200|60,EMA30,EMA30|1,EMA30|120,EMA30|15,EMA30|1M,EMA30|1W,EMA30|240,EMA30|30,EMA30|5,EMA30|60,EMA5,EMA50,EMA50|1,EMA50|120,EMA50|15,EMA50|1M,EMA50|1W,EMA50|240,EMA50|30,EMA50|5,EMA50|60,enterprise&#95;value&#95;ebitda&#95;ttm,enterprise&#95;value&#95;fq,eps&#95;surprise&#95;fq,eps&#95;surprise&#95;percent&#95;fq,exchange,float&#95;shares&#95;outstanding,free&#95;cash&#95;flow,gap,goodwill,gross&#95;margin,gross&#95;profit,gross&#95;profit&#95;fq,high,High.1M,High.3M,High.6M,High.All,HullMA9,HullMA9|1,HullMA9|120,HullMA9|15,HullMA9|1M,HullMA9|1W,HullMA9|240,HullMA9|30,HullMA9|5,HullMA9|60,Ichimoku.BLine,Ichimoku.BLine|1,Ichimoku.BLine|120,Ichimoku.BLine|15,Ichimoku.BLine|1M,Ichimoku.BLine|1W,Ichimoku.BLine|240,Ichimoku.BLine|30,Ichimoku.BLine|5,Ichimoku.BLine|60,Ichimoku.CLine,Ichimoku.Lead1,Ichimoku.Lead2,industry,is&#95;primary,KltChnl.lower,KltChnl.upper,last&#95;annual&#95;eps,last&#95;annual&#95;revenue,logoid,long&#95;term&#95;capital,low,Low.1M,Low.3M,Low.6M,Low.All,MACD.macd,MACD.macd|1,MACD.macd|120,MACD.macd|15,MACD.macd|1M,MACD.macd|1W,MACD.macd|240,MACD.macd|30,MACD.macd|5,MACD.macd|60,MACD.signal,MACD.signal|1,MACD.signal|120,MACD.signal|15,MACD.signal|1M,MACD.signal|1W,MACD.signal|240,MACD.signal|30,MACD.signal|5,MACD.signal|60,market&#95;cap&#95;basic,market&#95;cap&#95;calc,Mom,Mom|1,Mom|120,Mom|15,Mom|1M,Mom|1W,Mom|240,Mom|30,Mom|5,Mom|60,MoneyFlow,name,net&#95;debt,net&#95;income,number&#95;of&#95;employees,number&#95;of&#95;shareholders,open,operating&#95;margin,P.SAR,Perf.1M,Perf.3M,Perf.6M,Perf.W,Perf.Y,Perf.YTD,Pivot.M.Camarilla.Middle,Pivot.M.Camarilla.Middle|1,Pivot.M.Camarilla.Middle|120,Pivot.M.Camarilla.Middle|15,Pivot.M.Camarilla.Middle|1M,Pivot.M.Camarilla.Middle|1W,Pivot.M.Camarilla.Middle|240,Pivot.M.Camarilla.Middle|30,Pivot.M.Camarilla.Middle|5,Pivot.M.Camarilla.Middle|60,Pivot.M.Camarilla.R1,Pivot.M.Camarilla.R1|1,Pivot.M.Camarilla.R1|120,Pivot.M.Camarilla.R1|15,Pivot.M.Camarilla.R1|1M,Pivot.M.Camarilla.R1|1W,Pivot.M.Camarilla.R1|240,Pivot.M.Camarilla.R1|30,Pivot.M.Camarilla.R1|5,Pivot.M.Camarilla.R1|60,Pivot.M.Camarilla.R2,Pivot.M.Camarilla.R2|1,Pivot.M.Camarilla.R2|120,Pivot.M.Camarilla.R2|15,Pivot.M.Camarilla.R2|1M,Pivot.M.Camarilla.R2|1W,Pivot.M.Camarilla.R2|240,Pivot.M.Camarilla.R2|30,Pivot.M.Camarilla.R2|5,Pivot.M.Camarilla.R2|60,Pivot.M.Camarilla.R3,Pivot.M.Camarilla.R3|1,Pivot.M.Camarilla.R3|120,Pivot.M.Camarilla.R3|15,Pivot.M.Camarilla.R3|1M,Pivot.M.Camarilla.R3|1W,Pivot.M.Camarilla.R3|240,Pivot.M.Camarilla.R3|30,Pivot.M.Camarilla.R3|5,Pivot.M.Camarilla.R3|60,Pivot.M.Camarilla.S1,Pivot.M.Camarilla.S1|1,Pivot.M.Camarilla.S1|120,Pivot.M.Camarilla.S1|15,Pivot.M.Camarilla.S1|1M,Pivot.M.Camarilla.S1|1W,Pivot.M.Camarilla.S1|240,Pivot.M.Camarilla.S1|30,Pivot.M.Camarilla.S1|5,Pivot.M.Camarilla.S1|60,Pivot.M.Camarilla.S2,Pivot.M.Camarilla.S2|1,Pivot.M.Camarilla.S2|120,Pivot.M.Camarilla.S2|15,Pivot.M.Camarilla.S2|1M,Pivot.M.Camarilla.S2|1W,Pivot.M.Camarilla.S2|240,Pivot.M.Camarilla.S2|30,Pivot.M.Camarilla.S2|5,Pivot.M.Camarilla.S2|60,Pivot.M.Camarilla.S3,Pivot.M.Camarilla.S3|1,Pivot.M.Camarilla.S3|120,Pivot.M.Camarilla.S3|15,Pivot.M.Camarilla.S3|1M,Pivot.M.Camarilla.S3|1W,Pivot.M.Camarilla.S3|240,Pivot.M.Camarilla.S3|30,Pivot.M.Camarilla.S3|5,Pivot.M.Camarilla.S3|60,Pivot.M.Classic.Middle,Pivot.M.Classic.Middle|1,Pivot.M.Classic.Middle|120,Pivot.M.Classic.Middle|15,Pivot.M.Classic.Middle|1M,Pivot.M.Classic.Middle|1W,Pivot.M.Classic.Middle|240,Pivot.M.Classic.Middle|30,Pivot.M.Classic.Middle|5,Pivot.M.Classic.Middle|60,Pivot.M.Classic.R1,Pivot.M.Classic.R1|1,Pivot.M.Classic.R1|120,Pivot.M.Classic.R1|15,Pivot.M.Classic.R1|1M,Pivot.M.Classic.R1|1W,Pivot.M.Classic.R1|240,Pivot.M.Classic.R1|30,Pivot.M.Classic.R1|5,Pivot.M.Classic.R1|60,Pivot.M.Classic.R2,Pivot.M.Classic.R2|1,Pivot.M.Classic.R2|120,Pivot.M.Classic.R2|15,Pivot.M.Classic.R2|1M,Pivot.M.Classic.R2|1W,Pivot.M.Classic.R2|240,Pivot.M.Classic.R2|30,Pivot.M.Classic.R2|5,Pivot.M.Classic.R2|60,Pivot.M.Classic.R3,Pivot.M.Classic.R3|1,Pivot.M.Classic.R3|120,Pivot.M.Classic.R3|15,Pivot.M.Classic.R3|1M,Pivot.M.Classic.R3|1W,Pivot.M.Classic.R3|240,Pivot.M.Classic.R3|30,Pivot.M.Classic.R3|5,Pivot.M.Classic.R3|60,Pivot.M.Classic.S1,Pivot.M.Classic.S1|1,Pivot.M.Classic.S1|120,Pivot.M.Classic.S1|15,Pivot.M.Classic.S1|1M,Pivot.M.Classic.S1|1W,Pivot.M.Classic.S1|240,Pivot.M.Classic.S1|30,Pivot.M.Classic.S1|5,Pivot.M.Classic.S1|60,Pivot.M.Classic.S2,Pivot.M.Classic.S2|1,Pivot.M.Classic.S2|120,Pivot.M.Classic.S2|15,Pivot.M.Classic.S2|1M,Pivot.M.Classic.S2|1W,Pivot.M.Classic.S2|240,Pivot.M.Classic.S2|30,Pivot.M.Classic.S2|5,Pivot.M.Classic.S2|60,Pivot.M.Classic.S3,Pivot.M.Classic.S3|1,Pivot.M.Classic.S3|120,Pivot.M.Classic.S3|15,Pivot.M.Classic.S3|1M,Pivot.M.Classic.S3|1W,Pivot.M.Classic.S3|240,Pivot.M.Classic.S3|30,Pivot.M.Classic.S3|5,Pivot.M.Classic.S3|60,Pivot.M.Demark.Middle,Pivot.M.Demark.Middle|1,Pivot.M.Demark.Middle|120,Pivot.M.Demark.Middle|15,Pivot.M.Demark.Middle|1M,Pivot.M.Demark.Middle|1W,Pivot.M.Demark.Middle|240,Pivot.M.Demark.Middle|30,Pivot.M.Demark.Middle|5,Pivot.M.Demark.Middle|60,Pivot.M.Demark.R1,Pivot.M.Demark.R1|1,Pivot.M.Demark.R1|120,Pivot.M.Demark.R1|15,Pivot.M.Demark.R1|1M,Pivot.M.Demark.R1|1W,Pivot.M.Demark.R1|240,Pivot.M.Demark.R1|30,Pivot.M.Demark.R1|5,Pivot.M.Demark.R1|60,Pivot.M.Demark.S1,Pivot.M.Demark.S1|1,Pivot.M.Demark.S1|120,Pivot.M.Demark.S1|15,Pivot.M.Demark.S1|1M,Pivot.M.Demark.S1|1W,Pivot.M.Demark.S1|240,Pivot.M.Demark.S1|30,Pivot.M.Demark.S1|5,Pivot.M.Demark.S1|60,Pivot.M.Fibonacci.Middle,Pivot.M.Fibonacci.Middle|1,Pivot.M.Fibonacci.Middle|120,Pivot.M.Fibonacci.Middle|15,Pivot.M.Fibonacci.Middle|1M,Pivot.M.Fibonacci.Middle|1W,Pivot.M.Fibonacci.Middle|240,Pivot.M.Fibonacci.Middle|30,Pivot.M.Fibonacci.Middle|5,Pivot.M.Fibonacci.Middle|60,Pivot.M.Fibonacci.R1,Pivot.M.Fibonacci.R1|1,Pivot.M.Fibonacci.R1|120,Pivot.M.Fibonacci.R1|15,Pivot.M.Fibonacci.R1|1M,Pivot.M.Fibonacci.R1|1W,Pivot.M.Fibonacci.R1|240,Pivot.M.Fibonacci.R1|30,Pivot.M.Fibonacci.R1|5,Pivot.M.Fibonacci.R1|60,Pivot.M.Fibonacci.R2,Pivot.M.Fibonacci.R2|1,Pivot.M.Fibonacci.R2|120,Pivot.M.Fibonacci.R2|15,Pivot.M.Fibonacci.R2|1M,Pivot.M.Fibonacci.R2|1W,Pivot.M.Fibonacci.R2|240,Pivot.M.Fibonacci.R2|30,Pivot.M.Fibonacci.R2|5,Pivot.M.Fibonacci.R2|60,Pivot.M.Fibonacci.R3,Pivot.M.Fibonacci.R3|1,Pivot.M.Fibonacci.R3|120,Pivot.M.Fibonacci.R3|15,Pivot.M.Fibonacci.R3|1M,Pivot.M.Fibonacci.R3|1W,Pivot.M.Fibonacci.R3|240,Pivot.M.Fibonacci.R3|30,Pivot.M.Fibonacci.R3|5,Pivot.M.Fibonacci.R3|60,Pivot.M.Fibonacci.S1,Pivot.M.Fibonacci.S1|1,Pivot.M.Fibonacci.S1|120,Pivot.M.Fibonacci.S1|15,Pivot.M.Fibonacci.S1|1M,Pivot.M.Fibonacci.S1|1W,Pivot.M.Fibonacci.S1|240,Pivot.M.Fibonacci.S1|30,Pivot.M.Fibonacci.S1|5,Pivot.M.Fibonacci.S1|60,Pivot.M.Fibonacci.S2,Pivot.M.Fibonacci.S2|1,Pivot.M.Fibonacci.S2|120,Pivot.M.Fibonacci.S2|15,Pivot.M.Fibonacci.S2|1M,Pivot.M.Fibonacci.S2|1W,Pivot.M.Fibonacci.S2|240,Pivot.M.Fibonacci.S2|30,Pivot.M.Fibonacci.S2|5,Pivot.M.Fibonacci.S2|60,Pivot.M.Fibonacci.S3,Pivot.M.Fibonacci.S3|1,Pivot.M.Fibonacci.S3|120,Pivot.M.Fibonacci.S3|15,Pivot.M.Fibonacci.S3|1M,Pivot.M.Fibonacci.S3|1W,Pivot.M.Fibonacci.S3|240,Pivot.M.Fibonacci.S3|30,Pivot.M.Fibonacci.S3|5,Pivot.M.Fibonacci.S3|60,Pivot.M.Woodie.Middle,Pivot.M.Woodie.Middle|1,Pivot.M.Woodie.Middle|120,Pivot.M.Woodie.Middle|15,Pivot.M.Woodie.Middle|1M,Pivot.M.Woodie.Middle|1W,Pivot.M.Woodie.Middle|240,Pivot.M.Woodie.Middle|30,Pivot.M.Woodie.Middle|5,Pivot.M.Woodie.Middle|60,Pivot.M.Woodie.R1,Pivot.M.Woodie.R1|1,Pivot.M.Woodie.R1|120,Pivot.M.Woodie.R1|15,Pivot.M.Woodie.R1|1M,Pivot.M.Woodie.R1|1W,Pivot.M.Woodie.R1|240,Pivot.M.Woodie.R1|30,Pivot.M.Woodie.R1|5,Pivot.M.Woodie.R1|60,Pivot.M.Woodie.R2,Pivot.M.Woodie.R2|1,Pivot.M.Woodie.R2|120,Pivot.M.Woodie.R2|15,Pivot.M.Woodie.R2|1M,Pivot.M.Woodie.R2|1W,Pivot.M.Woodie.R2|240,Pivot.M.Woodie.R2|30,Pivot.M.Woodie.R2|5,Pivot.M.Woodie.R2|60,Pivot.M.Woodie.R3,Pivot.M.Woodie.R3|1,Pivot.M.Woodie.R3|120,Pivot.M.Woodie.R3|15,Pivot.M.Woodie.R3|1M,Pivot.M.Woodie.R3|1W,Pivot.M.Woodie.R3|240,Pivot.M.Woodie.R3|30,Pivot.M.Woodie.R3|5,Pivot.M.Woodie.R3|60,Pivot.M.Woodie.S1,Pivot.M.Woodie.S1|1,Pivot.M.Woodie.S1|120,Pivot.M.Woodie.S1|15,Pivot.M.Woodie.S1|1M,Pivot.M.Woodie.S1|1W,Pivot.M.Woodie.S1|240,Pivot.M.Woodie.S1|30,Pivot.M.Woodie.S1|5,Pivot.M.Woodie.S1|60,Pivot.M.Woodie.S2,Pivot.M.Woodie.S2|1,Pivot.M.Woodie.S2|120,Pivot.M.Woodie.S2|15,Pivot.M.Woodie.S2|1M,Pivot.M.Woodie.S2|1W,Pivot.M.Woodie.S2|240,Pivot.M.Woodie.S2|30,Pivot.M.Woodie.S2|5,Pivot.M.Woodie.S2|60,Pivot.M.Woodie.S3,Pivot.M.Woodie.S3|1,Pivot.M.Woodie.S3|120,Pivot.M.Woodie.S3|15,Pivot.M.Woodie.S3|1M,Pivot.M.Woodie.S3|1W,Pivot.M.Woodie.S3|240,Pivot.M.Woodie.S3|30,Pivot.M.Woodie.S3|5,Pivot.M.Woodie.S3|60,post&#95;change,postmarket&#95;change,postmarket&#95;change&#95;abs,postmarket&#95;close,postmarket&#95;high,postmarket&#95;low,postmarket&#95;open,postmarket&#95;volume,pre&#95;change,pre&#95;tax&#95;margin,preferred&#95;dividends,premarket&#95;change,premarket&#95;change&#95;abs,premarket&#95;change&#95;from&#95;open,premarket&#95;change&#95;from&#95;open&#95;abs,premarket&#95;close,premarket&#95;gap,premarket&#95;high,premarket&#95;low,premarket&#95;open,premarket&#95;volume,price&#95;52&#95;week&#95;high,price&#95;52&#95;week&#95;low,price&#95;book&#95;fq,price&#95;book&#95;ratio,price&#95;earnings&#95;to&#95;growth&#95;ttm,price&#95;earnings&#95;ttm,price&#95;free&#95;cash&#95;flow&#95;ttm,price&#95;revenue&#95;ttm,price&#95;sales,price&#95;sales&#95;ratio,quick&#95;ratio,Rec.BBPower,Rec.BBPower|1,Rec.BBPower|120,Rec.BBPower|15,Rec.BBPower|1M,Rec.BBPower|1W,Rec.BBPower|240,Rec.BBPower|30,Rec.BBPower|5,Rec.BBPower|60,Rec.HullMA9,Rec.HullMA9|1,Rec.HullMA9|120,Rec.HullMA9|15,Rec.HullMA9|1M,Rec.HullMA9|1W,Rec.HullMA9|240,Rec.HullMA9|30,Rec.HullMA9|5,Rec.HullMA9|60,Rec.Ichimoku,Rec.Ichimoku|1,Rec.Ichimoku|120,Rec.Ichimoku|15,Rec.Ichimoku|1M,Rec.Ichimoku|1W,Rec.Ichimoku|240,Rec.Ichimoku|30,Rec.Ichimoku|5,Rec.Ichimoku|60,Rec.Stoch.RSI,Rec.Stoch.RSI|1,Rec.Stoch.RSI|120,Rec.Stoch.RSI|15,Rec.Stoch.RSI|1M,Rec.Stoch.RSI|1W,Rec.Stoch.RSI|240,Rec.Stoch.RSI|30,Rec.Stoch.RSI|5,Rec.Stoch.RSI|60,Rec.UO,Rec.UO|1,Rec.UO|120,Rec.UO|15,Rec.UO|1M,Rec.UO|1W,Rec.UO|240,Rec.UO|30,Rec.UO|5,Rec.UO|60,Rec.VWMA,Rec.VWMA|1,Rec.VWMA|120,Rec.VWMA|15,Rec.VWMA|1M,Rec.VWMA|1W,Rec.VWMA|240,Rec.VWMA|30,Rec.VWMA|5,Rec.VWMA|60,Rec.WR,Rec.WR|1,Rec.WR|120,Rec.WR|15,Rec.WR|1M,Rec.WR|1W,Rec.WR|240,Rec.WR|30,Rec.WR|5,Rec.WR|60,Recommend.All,Recommend.All|1,Recommend.All|120,Recommend.All|15,Recommend.All|1M,Recommend.All|1W,Recommend.All|240,Recommend.All|30,Recommend.All|5,Recommend.All|60,Recommend.MA,Recommend.MA|1,Recommend.MA|120,Recommend.MA|15,Recommend.MA|1M,Recommend.MA|1W,Recommend.MA|240,Recommend.MA|30,Recommend.MA|5,Recommend.MA|60,Recommend.Other,Recommend.Other|1,Recommend.Other|120,Recommend.Other|15,Recommend.Other|1M,Recommend.Other|1W,Recommend.Other|240,Recommend.Other|30,Recommend.Other|5,Recommend.Other|60,relative&#95;volume,relative&#95;volume&#95;10d&#95;calc,relative&#95;volume&#95;intraday|5,return&#95;of&#95;invested&#95;capital&#95;percent&#95;ttm,return&#95;on&#95;assets,return&#95;on&#95;equity,return&#95;on&#95;invested&#95;capital,revenue&#95;per&#95;employee,ROC,RSI,RSI|1,RSI|120,RSI|15,RSI|1M,RSI|1W,RSI|240,RSI|30,RSI|5,RSI|60,RSI7,sector,SMA10,SMA10|1,SMA10|120,SMA10|15,SMA10|1M,SMA10|1W,SMA10|240,SMA10|30,SMA10|5,SMA10|60,SMA100,SMA100|1,SMA100|120,SMA100|15,SMA100|1M,SMA100|1W,SMA100|240,SMA100|30,SMA100|5,SMA100|60,SMA20,SMA20|1,SMA20|120,SMA20|15,SMA20|1M,SMA20|1W,SMA20|240,SMA20|30,SMA20|5,SMA20|60,SMA200,SMA200|1,SMA200|120,SMA200|15,SMA200|1M,SMA200|1W,SMA200|240,SMA200|30,SMA200|5,SMA200|60,SMA30,SMA30|1,SMA30|120,SMA30|15,SMA30|1M,SMA30|1W,SMA30|240,SMA30|30,SMA30|5,SMA30|60,SMA5,SMA50,SMA50|1,SMA50|120,SMA50|15,SMA50|1M,SMA50|1W,SMA50|240,SMA50|30,SMA50|5,SMA50|60,Stoch.D,Stoch.D|1,Stoch.D|120,Stoch.D|15,Stoch.D|1M,Stoch.D|1W,Stoch.D|240,Stoch.D|30,Stoch.D|5,Stoch.D|60,Stoch.K,Stoch.K|1,Stoch.K|120,Stoch.K|15,Stoch.K|1M,Stoch.K|1W,Stoch.K|240,Stoch.K|30,Stoch.K|5,Stoch.K|60,Stoch.RSI.D,Stoch.RSI.K,Stoch.RSI.K|1,Stoch.RSI.K|120,Stoch.RSI.K|15,Stoch.RSI.K|1M,Stoch.RSI.K|1W,Stoch.RSI.K|240,Stoch.RSI.K|30,Stoch.RSI.K|5,Stoch.RSI.K|60,submarket,total&#95;assets,total&#95;capital,total&#95;current&#95;assets,total&#95;debt,total&#95;liabilities&#95;fq,total&#95;liabilities&#95;fy,total&#95;revenue,total&#95;shares&#95;outstanding,total&#95;shares&#95;outstanding&#95;fundamental,type,UO,UO|1,UO|120,UO|15,UO|1M,UO|1W,UO|240,UO|30,UO|5,UO|60,update&#95;mode,Value.Traded,Volatility.D,Volatility.M,Volatility.W,volume,VWAP,VWMA,VWMA|1,VWMA|120,VWMA|15,VWMA|1M,VWMA|1W,VWMA|240,VWMA|30,VWMA|5,VWMA|60,W.R,W.R|1,W.R|120,W.R|15,W.R|1M,W.R|1W,W.R|240,W.R|30,W.R|5,W.R|60
        screenername: The value of screenerName field returned in .../countries/list endpoint
        lang: The language code
        
    """
    url = f"https://trading-view.p.rapidapi.com/stocks/get-financials"
    querystring = {'symbol': symbol, 'columns': columns, }
    if screenername:
        querystring['screenerName'] = screenername
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trading-view.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchanges_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all available and supported exchanges"
    
    """
    url = f"https://trading-view.p.rapidapi.com/exchanges/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trading-view.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all available supported countries"
    
    """
    url = f"https://trading-view.p.rapidapi.com/countries/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trading-view.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_auto_complete(text: str, start: str='0', search_type: str=None, lang: str='en', exchange: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get suggestion by term or phrase"
    text: Any term or phrase that you are familiar with
        start: The offset of records to ignore for paging purpose.
        search_type: One of the following : stock|futures|forex|index|bond|economic|bitcoin|crypto. Separated by comma for multiple options. Ex : bitcoin,crypto,stock,...
        lang: The language code
        exchange: The value of \"value\" field returned in .../exchanges/list endpoint
        
    """
    url = f"https://trading-view.p.rapidapi.com/v2/auto-complete"
    querystring = {'text': text, }
    if start:
        querystring['start'] = start
    if search_type:
        querystring['search_type'] = search_type
    if lang:
        querystring['lang'] = lang
    if exchange:
        querystring['exchange'] = exchange
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trading-view.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calendars_get_economic_calendar(is_from: str, to: str, minimportance: int=None, lang: str='en', countries: str='US,EU,JP,AU,DE,GB,CA,FR,IT,NZ,ES,MX,CH,TR,ZA', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get economic calendar by specific date range and countries"
    from: The date format is yyyy-MM-dd. Ex : 2022-08-21
        to: The date format is yyyy-MM-dd. Ex : 2022-09-05
        minimportance: One of the following : -1|0|1
        lang: The language code
        countries: One of the following : US|EU|JP|AU|DE|GB|CA|FR|IT|NZ|ES|MX|CH|TR|ZA. Separated by comma for multiple options
        
    """
    url = f"https://trading-view.p.rapidapi.com/calendars/get-economic-calendar"
    querystring = {'from': is_from, 'to': to, }
    if minimportance:
        querystring['minImportance'] = minimportance
    if lang:
        querystring['lang'] = lang
    if countries:
        querystring['countries'] = countries
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trading-view.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ideas_list(locale: str='en', category: str=None, stock_country: str='us', page: int=1, per_page: int=20, market: str=None, symbol: str=None, sort: str='latest_popular', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List ideas post with options and filters"
    locale: The language code
        category: Leave empty or one of the following : picked|newest|popular
        stock_country: The country code, only functionable if the market parameter is stock.
        page: The page index, for paging purpose
        per_page: The number of items per response, for paging purpose
        market: Leave empty or one of the following : bond|futures|index|forex|bitcoin|stock
        symbol: List ideas posts related to specified symbol. Ex : NASDAQ:AAPL
        sort: One of the following : latest&#95;popular|recent|picked&#95;time
        
    """
    url = f"https://trading-view.p.rapidapi.com/ideas/list"
    querystring = {}
    if locale:
        querystring['locale'] = locale
    if category:
        querystring['category'] = category
    if stock_country:
        querystring['stock_country'] = stock_country
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    if market:
        querystring['market'] = market
    if symbol:
        querystring['symbol'] = symbol
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trading-view.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ideas_get_view_count(uuid: str, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get view count related to an ideal post"
    uuid: The value of image_url field returned in .../ideas/list endpoint
        lang: The language code
        
    """
    url = f"https://trading-view.p.rapidapi.com/ideas/get-view-count"
    querystring = {'uuid': uuid, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trading-view.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ideas_get_replies(is_id: str, uuid: str, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get replies relating to an ideal post and comment"
    uuid: The value of image_url field returned in .../ideas/list endpoint
        lang: The language code
        
    """
    url = f"https://trading-view.p.rapidapi.com/ideas/get-replies"
    querystring = {'id': is_id, 'uuid': uuid, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trading-view.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ideas_get_comments(uuid: str, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comments relating to an ideal post"
    uuid: The value of image_url field returned in .../ideas/list endpoint
        lang: The language code
        
    """
    url = f"https://trading-view.p.rapidapi.com/ideas/get-comments"
    querystring = {'uuid': uuid, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trading-view.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ideas_detail(uuid: str, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed ideal post"
    uuid: The value of image_url field returned in .../ideas/list endpoint
        lang: The language code
        
    """
    url = f"https://trading-view.p.rapidapi.com/ideas/detail"
    querystring = {'uuid': uuid, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trading-view.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calendars_get_earning_calendar(is_from: int, to: int, lang: str='en', screenername: str='america', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get earning calendar by specific date range and exchange"
    from: The epoch timestamp in seconds. Ex : 1661187600
        to: The epoch timestamp in seconds. Ex : 1661619600
        lang: The language code
        screenername: The value of screenerName returned in .../countries/list endpoint
        
    """
    url = f"https://trading-view.p.rapidapi.com/calendars/get-earning-calendar"
    querystring = {'from': is_from, 'to': to, }
    if lang:
        querystring['lang'] = lang
    if screenername:
        querystring['screenerName'] = screenername
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trading-view.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

