"""
ClaudeQuantTrading Backend - Quant-based Market Analysis API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import httpx
import numpy as np
from datetime import datetime, timedelta
import os
from groq import Groq

app = FastAPI(title="ClaudeQuantTrading API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global context storage
analysis_context: Dict[str, Any] = {}

# Models
class AnalyzeRequest(BaseModel):
    asset: str
    timeframe: str

class ChatRequest(BaseModel):
    question: str

class IndicatorResult(BaseModel):
    name: str
    value: Any
    status: str
    interpretation: str

class AgentReasoning(BaseModel):
    agent: str
    reasoning: str
    conclusion: str

class AnalyzeResponse(BaseModel):
    final_bias: str
    confidence: float
    summary_reasoning: str
    indicators: List[IndicatorResult]
    agent_reasoning: List[AgentReasoning]
    price_series: List[Dict[str, Any]]

class ChatResponse(BaseModel):
    answer: str

# CoinGecko mapping
COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "SOL": "solana",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "DOT": "polkadot",
    "AVAX": "avalanche-2",
    "MATIC": "matic-network"
}

TIMEFRAME_DAYS = {
    "1h": 1,
    "4h": 1,
    "1d": 30,
    "7d": 90
}

# Technical Indicator Calculations
def calculate_sma(prices: List[float], period: int) -> List[float]:
    """Calculate Simple Moving Average"""
    sma = []
    for i in range(len(prices)):
        if i < period - 1:
            sma.append(None)
        else:
            sma.append(np.mean(prices[i - period + 1:i + 1]))
    return sma

def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return 50.0
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)

def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
    """Calculate MACD"""
    if len(prices) < slow:
        return {"macd": 0, "signal": 0, "histogram": 0, "direction": "neutral"}
    
    prices_arr = np.array(prices)
    
    # EMA calculation
    def ema(data, period):
        alpha = 2 / (period + 1)
        ema_vals = [data[0]]
        for i in range(1, len(data)):
            ema_vals.append(alpha * data[i] + (1 - alpha) * ema_vals[-1])
        return np.array(ema_vals)
    
    ema_fast = ema(prices_arr, fast)
    ema_slow = ema(prices_arr, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    direction = "bullish" if histogram[-1] > 0 else "bearish"
    
    return {
        "macd": round(float(macd_line[-1]), 4),
        "signal": round(float(signal_line[-1]), 4),
        "histogram": round(float(histogram[-1]), 4),
        "direction": direction
    }

def calculate_stochastic(highs: List[float], lows: List[float], closes: List[float], 
                         k_period: int = 14, d_period: int = 3) -> Dict:
    """Calculate Stochastic Oscillator"""
    if len(closes) < k_period:
        return {"k": 50, "d": 50, "condition": "neutral"}
    
    k_values = []
    for i in range(k_period - 1, len(closes)):
        highest_high = max(highs[i - k_period + 1:i + 1])
        lowest_low = min(lows[i - k_period + 1:i + 1])
        
        if highest_high == lowest_low:
            k_values.append(50)
        else:
            k = ((closes[i] - lowest_low) / (highest_high - lowest_low)) * 100
            k_values.append(k)
    
    k = k_values[-1] if k_values else 50
    d = np.mean(k_values[-d_period:]) if len(k_values) >= d_period else k
    
    if k > 80:
        condition = "overbought"
    elif k < 20:
        condition = "oversold"
    else:
        condition = "neutral"
    
    return {"k": round(k, 2), "d": round(d, 2), "condition": condition}

async def fetch_ohlcv(asset: str, days: int) -> Dict:
    """Fetch OHLCV data from CoinGecko"""
    coin_id = COINGECKO_IDS.get(asset.upper())
    if not coin_id:
        raise HTTPException(status_code=400, detail=f"Unknown asset: {asset}")
    
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch market data")
        
        data = response.json()
        
    prices = [p[1] for p in data.get("prices", [])]
    timestamps = [p[0] for p in data.get("prices", [])]
    
    # Generate synthetic OHLC from price data
    ohlcv = []
    for i, (ts, price) in enumerate(zip(timestamps, prices)):
        variation = price * 0.005
        ohlcv.append({
            "timestamp": ts,
            "open": price - variation if i % 2 == 0 else price + variation,
            "high": price + variation * 2,
            "low": price - variation * 2,
            "close": price
        })
    
    return {
        "prices": prices,
        "timestamps": timestamps,
        "ohlcv": ohlcv,
        "highs": [o["high"] for o in ohlcv],
        "lows": [o["low"] for o in ohlcv],
        "closes": [o["close"] for o in ohlcv]
    }

def get_groq_client():
    """Get Groq client"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

async def run_agent(client, agent_name: str, prompt: str) -> str:
    """Run single agent with LLM"""
    if not client:
        # Fallback without LLM
        return f"Analysis based on quantitative indicators."
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": f"You are {agent_name}, a professional crypto market analyst. Provide concise analysis in 2-3 sentences."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Agent analysis unavailable: {str(e)}"

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Main analysis endpoint"""
    global analysis_context
    
    asset = request.asset.upper()
    timeframe = request.timeframe
    days = TIMEFRAME_DAYS.get(timeframe, 30)
    
    # Fetch market data
    market_data = await fetch_ohlcv(asset, days)
    prices = market_data["prices"]
    
    # Calculate indicators mathematically
    rsi = calculate_rsi(prices)
    macd = calculate_macd(prices)
    stoch = calculate_stochastic(
        market_data["highs"], 
        market_data["lows"], 
        market_data["closes"]
    )
    sma20 = calculate_sma(prices, 20)
    sma50 = calculate_sma(prices, 50)
    
    current_price = prices[-1]
    sma20_val = sma20[-1] if sma20[-1] else current_price
    sma50_val = sma50[-1] if sma50[-1] else current_price
    
    # Prepare indicators result
    indicators = [
        IndicatorResult(
            name="RSI (14)",
            value=rsi,
            status="overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral",
            interpretation=f"RSI at {rsi} indicates {'overbought conditions' if rsi > 70 else 'oversold conditions' if rsi < 30 else 'neutral momentum'}"
        ),
        IndicatorResult(
            name="MACD (12,26,9)",
            value=macd,
            status=macd["direction"],
            interpretation=f"MACD histogram at {macd['histogram']} shows {macd['direction']} momentum"
        ),
        IndicatorResult(
            name="Stochastic (14,3,3)",
            value={"k": stoch["k"], "d": stoch["d"]},
            status=stoch["condition"],
            interpretation=f"Stochastic %K at {stoch['k']} is {stoch['condition']}"
        ),
        IndicatorResult(
            name="Moving Averages",
            value={"sma20": round(sma20_val, 2), "sma50": round(sma50_val, 2), "price": round(current_price, 2)},
            status="bullish" if current_price > sma20_val > sma50_val else "bearish" if current_price < sma20_val < sma50_val else "mixed",
            interpretation=f"Price {'above' if current_price > sma20_val else 'below'} SMA20, {'above' if current_price > sma50_val else 'below'} SMA50"
        )
    ]
    
    # Run agent system
    client = get_groq_client()
    agent_reasoning = []
    
    # Indicator Agent
    indicator_prompt = f"""
    Analyze these indicators for {asset}:
    - RSI: {rsi} ({'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'})
    - MACD: histogram {macd['histogram']} ({macd['direction']})
    - Stochastic: %K {stoch['k']} ({stoch['condition']})
    What do these indicators suggest about current momentum?
    """
    indicator_analysis = await run_agent(client, "Indicator Agent", indicator_prompt)
    agent_reasoning.append(AgentReasoning(
        agent="Indicator Agent",
        reasoning=indicator_analysis,
        conclusion="bullish" if rsi < 50 and macd["direction"] == "bullish" else "bearish" if rsi > 50 and macd["direction"] == "bearish" else "neutral"
    ))
    
    # Trend Agent
    trend_prompt = f"""
    Analyze trend for {asset}:
    - Current Price: ${current_price:.2f}
    - SMA 20: ${sma20_val:.2f}
    - SMA 50: ${sma50_val:.2f}
    - Price vs SMA20: {'above' if current_price > sma20_val else 'below'}
    - Price vs SMA50: {'above' if current_price > sma50_val else 'below'}
    What is the current trend direction?
    """
    trend_analysis = await run_agent(client, "Trend Agent", trend_prompt)
    trend_bias = "bullish" if current_price > sma20_val and current_price > sma50_val else "bearish" if current_price < sma20_val and current_price < sma50_val else "neutral"
    agent_reasoning.append(AgentReasoning(
        agent="Trend Agent",
        reasoning=trend_analysis,
        conclusion=trend_bias
    ))
    
    # Pattern Agent
    price_change = ((prices[-1] - prices[0]) / prices[0]) * 100
    pattern_prompt = f"""
    Analyze price pattern for {asset}:
    - Price change over period: {price_change:.2f}%
    - Current trend: {trend_bias}
    - RSI momentum: {rsi}
    Identify any notable patterns or formations.
    """
    pattern_analysis = await run_agent(client, "Pattern Agent", pattern_prompt)
    agent_reasoning.append(AgentReasoning(
        agent="Pattern Agent",
        reasoning=pattern_analysis,
        conclusion="continuation" if abs(price_change) > 5 else "consolidation"
    ))
    
    # Decision Agent
    decision_prompt = f"""
    Final decision for {asset}:
    - Indicator Agent: {agent_reasoning[0].conclusion}
    - Trend Agent: {agent_reasoning[1].conclusion}
    - Pattern Agent: {agent_reasoning[2].conclusion}
    - RSI: {rsi}, MACD: {macd['direction']}, Stoch: {stoch['condition']}
    Provide final market bias and confidence level.
    """
    decision_analysis = await run_agent(client, "Decision Agent", decision_prompt)
    
    # Calculate final bias and confidence
    bullish_count = sum(1 for a in agent_reasoning if a.conclusion in ["bullish", "continuation"])
    bearish_count = sum(1 for a in agent_reasoning if a.conclusion in ["bearish"])
    
    if bullish_count > bearish_count:
        final_bias = "BULLISH"
        confidence = min(70 + (bullish_count * 10), 90)
    elif bearish_count > bullish_count:
        final_bias = "BEARISH"
        confidence = min(70 + (bearish_count * 10), 90)
    else:
        final_bias = "NEUTRAL"
        confidence = 50
    
    agent_reasoning.append(AgentReasoning(
        agent="Decision Agent",
        reasoning=decision_analysis,
        conclusion=final_bias.lower()
    ))
    
    # Prepare price series for chart
    price_series = []
    for i, (ts, price) in enumerate(zip(market_data["timestamps"], prices)):
        price_series.append({
            "timestamp": ts,
            "price": round(price, 2),
            "sma20": round(sma20[i], 2) if sma20[i] else None,
            "sma50": round(sma50[i], 2) if sma50[i] else None
        })
    
    # Store context for chat
    analysis_context = {
        "asset": asset,
        "timeframe": timeframe,
        "indicators": [i.model_dump() for i in indicators],
        "agent_reasoning": [a.model_dump() for a in agent_reasoning],
        "final_bias": final_bias,
        "confidence": confidence,
        "current_price": current_price,
        "timestamp": datetime.now().isoformat()
    }
    
    return AnalyzeResponse(
        final_bias=final_bias,
        confidence=confidence,
        summary_reasoning=decision_analysis,
        indicators=indicators,
        agent_reasoning=agent_reasoning,
        price_series=price_series[-100:]  # Last 100 data points
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint - only active after analysis"""
    global analysis_context
    
    if not analysis_context:
        raise HTTPException(status_code=400, detail="No analysis context available. Please run analysis first.")
    
    client = get_groq_client()
    if not client:
        return ChatResponse(answer="Chat unavailable. Please configure GROQ_API_KEY.")
    
    context_str = f"""
    Previous Analysis Context:
    - Asset: {analysis_context['asset']}
    - Timeframe: {analysis_context['timeframe']}
    - Final Bias: {analysis_context['final_bias']}
    - Confidence: {analysis_context['confidence']}%
    - Current Price: ${analysis_context['current_price']:.2f}
    - Indicators: {analysis_context['indicators']}
    - Agent Analysis: {analysis_context['agent_reasoning']}
    """
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": f"You are a crypto analyst assistant. Use ONLY the provided analysis context to answer questions. Do NOT recalculate indicators or fetch new data. Context: {context_str}"},
                {"role": "user", "content": request.question}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return ChatResponse(answer=response.choices[0].message.content)
    except Exception as e:
        return ChatResponse(answer=f"Error processing question: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
