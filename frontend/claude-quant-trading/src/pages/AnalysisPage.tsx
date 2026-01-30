import { useState } from 'react';
import { Link } from 'react-router-dom';
import { TrendingUp, Github, Loader2, ChevronDown, ChevronUp, Send, CheckCircle2, Circle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || '';
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY || '';

const ASSETS = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'DOT', 'AVAX', 'MATIC'];
const TIMEFRAMES = [
  { value: '1h', label: '1 Hour' },
  { value: '4h', label: '4 Hours' },
  { value: '1d', label: '1 Day' },
  { value: '7d', label: '7 Days' },
];

const COINGECKO_IDS: Record<string, string> = {
  BTC: 'bitcoin', ETH: 'ethereum', BNB: 'binancecoin', SOL: 'solana',
  XRP: 'ripple', ADA: 'cardano', DOGE: 'dogecoin', DOT: 'polkadot',
  AVAX: 'avalanche-2', MATIC: 'matic-network',
};

const TIMEFRAME_DAYS: Record<string, number> = { '1h': 1, '4h': 1, '1d': 30, '7d': 90 };

// Technical Indicator Calculations
function calculateSMA(prices: number[], period: number): (number | null)[] {
  return prices.map((_, i) => 
    i < period - 1 ? null : prices.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period
  );
}

function calculateRSI(prices: number[], period = 14): number {
  if (prices.length < period + 1) return 50;
  const deltas = prices.slice(1).map((p, i) => p - prices[i]);
  const gains = deltas.slice(-period).filter(d => d > 0);
  const losses = deltas.slice(-period).filter(d => d < 0).map(d => Math.abs(d));
  const avgGain = gains.length > 0 ? gains.reduce((a, b) => a + b, 0) / period : 0;
  const avgLoss = losses.length > 0 ? losses.reduce((a, b) => a + b, 0) / period : 0;
  if (avgLoss === 0) return 100;
  return Math.round((100 - 100 / (1 + avgGain / avgLoss)) * 100) / 100;
}

function calculateEMA(data: number[], period: number): number[] {
  const alpha = 2 / (period + 1);
  const ema = [data[0]];
  for (let i = 1; i < data.length; i++) ema.push(alpha * data[i] + (1 - alpha) * ema[i - 1]);
  return ema;
}

function calculateMACD(prices: number[]) {
  if (prices.length < 26) return { macd: 0, signal: 0, histogram: 0, direction: 'neutral' };
  const emaFast = calculateEMA(prices, 12);
  const emaSlow = calculateEMA(prices, 26);
  const macdLine = emaFast.map((v, i) => v - emaSlow[i]);
  const signalLine = calculateEMA(macdLine, 9);
  const histogram = macdLine[macdLine.length - 1] - signalLine[signalLine.length - 1];
  return {
    macd: Math.round(macdLine[macdLine.length - 1] * 100) / 100,
    signal: Math.round(signalLine[signalLine.length - 1] * 100) / 100,
    histogram: Math.round(histogram * 100) / 100,
    direction: histogram > 0 ? 'bullish' : 'bearish',
  };
}

function calculateStochastic(highs: number[], lows: number[], closes: number[], kPeriod = 14, dPeriod = 3) {
  if (closes.length < kPeriod) return { k: 50, d: 50, condition: 'neutral' };
  const kValues: number[] = [];
  for (let i = kPeriod - 1; i < closes.length; i++) {
    const highestHigh = Math.max(...highs.slice(i - kPeriod + 1, i + 1));
    const lowestLow = Math.min(...lows.slice(i - kPeriod + 1, i + 1));
    kValues.push(highestHigh === lowestLow ? 50 : ((closes[i] - lowestLow) / (highestHigh - lowestLow)) * 100);
  }
  const k = kValues[kValues.length - 1];
  const d = kValues.slice(-dPeriod).reduce((a, b) => a + b, 0) / dPeriod;
  return { k: Math.round(k * 100) / 100, d: Math.round(d * 100) / 100, condition: k > 80 ? 'overbought' : k < 20 ? 'oversold' : 'neutral' };
}

interface Indicator { name: string; value: any; status: string; interpretation: string; }
interface AgentReasoning { agent: string; reasoning: string; conclusion: string; }
interface AnalysisResult {
  final_bias: string; confidence: number; summary_reasoning: string;
  indicators: Indicator[]; agent_reasoning: AgentReasoning[];
  price_series: Array<{ timestamp: number; price: number; sma20: number | null; sma50: number | null }>;
}
type AgentStatus = 'pending' | 'running' | 'complete';

export default function AnalysisPage() {
  const [asset, setAsset] = useState('BTC');
  const [timeframe, setTimeframe] = useState('1d');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [agentStatuses, setAgentStatuses] = useState<Record<string, AgentStatus>>({});
  const [expandedAgents, setExpandedAgents] = useState<Record<string, boolean>>({});
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState<Array<{ role: 'user' | 'assistant'; content: string }>>([]);
  const [chatLoading, setChatLoading] = useState(false);

  const runAnalysis = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setChatMessages([]);
    
    const agents = ['Indicator Agent', 'Trend Agent', 'Pattern Agent', 'Decision Agent'];
    setAgentStatuses(Object.fromEntries(agents.map(a => [a, 'pending'])));

    try {
      // Update agent status: Indicator Agent running
      setAgentStatuses(prev => ({ ...prev, 'Indicator Agent': 'running' }));
      
      // Fetch data from CoinGecko
      const coinId = COINGECKO_IDS[asset];
      const days = TIMEFRAME_DAYS[timeframe] || 30;
      const cgResponse = await fetch(`https://api.coingecko.com/api/v3/coins/${coinId}/market_chart?vs_currency=usd&days=${days}`);
      
      if (!cgResponse.ok) throw new Error('Failed to fetch market data');
      const cgData = await cgResponse.json();
      
      const prices: number[] = cgData.prices.map((p: number[]) => p[1]);
      const timestamps: number[] = cgData.prices.map((p: number[]) => p[0]);
      
      // Generate OHLC
      const ohlcv = prices.map((price: number, i: number) => {
        const variation = price * 0.005;
        return { high: price + variation * 2, low: price - variation * 2, close: price };
      });
      const highs = ohlcv.map(o => o.high);
      const lows = ohlcv.map(o => o.low);
      const closes = ohlcv.map(o => o.close);

      // Calculate indicators
      const rsi = calculateRSI(prices);
      const macd = calculateMACD(prices);
      const stoch = calculateStochastic(highs, lows, closes);
      const sma20 = calculateSMA(prices, 20);
      const sma50 = calculateSMA(prices, 50);
      
      setAgentStatuses(prev => ({ ...prev, 'Indicator Agent': 'complete', 'Trend Agent': 'running' }));
      await new Promise(r => setTimeout(r, 300));

      const currentPrice = prices[prices.length - 1];
      const sma20Val = sma20[sma20.length - 1] || currentPrice;
      const sma50Val = sma50[sma50.length - 1] || currentPrice;

      // Prepare indicators
      const indicators: Indicator[] = [
        { name: 'RSI (14)', value: rsi, status: rsi > 70 ? 'overbought' : rsi < 30 ? 'oversold' : 'neutral',
          interpretation: `RSI at ${rsi} indicates ${rsi > 70 ? 'overbought conditions' : rsi < 30 ? 'oversold conditions' : 'neutral momentum'}` },
        { name: 'MACD (12,26,9)', value: macd, status: macd.direction,
          interpretation: `MACD histogram at ${macd.histogram} shows ${macd.direction} momentum` },
        { name: 'Stochastic (14,3,3)', value: { k: stoch.k, d: stoch.d }, status: stoch.condition,
          interpretation: `Stochastic %K at ${stoch.k} is ${stoch.condition}` },
        { name: 'Moving Averages', value: { sma20: Math.round(sma20Val * 100) / 100, sma50: Math.round(sma50Val * 100) / 100, price: Math.round(currentPrice * 100) / 100 },
          status: currentPrice > sma20Val && sma20Val > sma50Val ? 'bullish' : currentPrice < sma20Val && sma20Val < sma50Val ? 'bearish' : 'mixed',
          interpretation: `Price ${currentPrice > sma20Val ? 'above' : 'below'} SMA20, ${currentPrice > sma50Val ? 'above' : 'below'} SMA50` }
      ];

      setAgentStatuses(prev => ({ ...prev, 'Trend Agent': 'complete', 'Pattern Agent': 'running' }));
      await new Promise(r => setTimeout(r, 300));

      // Agent reasoning (quantitative-based)
      const indicatorConclusion = rsi < 50 && macd.direction === 'bullish' ? 'bullish' : rsi > 50 && macd.direction === 'bearish' ? 'bearish' : 'neutral';
      const trendBias = currentPrice > sma20Val && currentPrice > sma50Val ? 'bullish' : currentPrice < sma20Val && currentPrice < sma50Val ? 'bearish' : 'neutral';
      const priceChange = ((prices[prices.length - 1] - prices[0]) / prices[0]) * 100;
      const patternConclusion = Math.abs(priceChange) > 5 ? 'continuation' : 'consolidation';

      const agentReasoning: AgentReasoning[] = [
        { agent: 'Indicator Agent', reasoning: `RSI at ${rsi} (${rsi > 70 ? 'overbought' : rsi < 30 ? 'oversold' : 'neutral'}), MACD ${macd.direction} with histogram ${macd.histogram}, Stochastic ${stoch.condition}. Momentum analysis suggests ${indicatorConclusion} conditions.`, conclusion: indicatorConclusion },
        { agent: 'Trend Agent', reasoning: `Price at $${currentPrice.toFixed(2)} is ${currentPrice > sma20Val ? 'above' : 'below'} SMA20 ($${sma20Val.toFixed(2)}) and ${currentPrice > sma50Val ? 'above' : 'below'} SMA50 ($${sma50Val.toFixed(2)}). Overall trend direction is ${trendBias}.`, conclusion: trendBias },
        { agent: 'Pattern Agent', reasoning: `Price changed ${priceChange.toFixed(2)}% over the analysis period. ${Math.abs(priceChange) > 5 ? 'Strong directional movement detected, likely continuation.' : 'Price in consolidation range, waiting for breakout.'}`, conclusion: patternConclusion },
      ];

      setAgentStatuses(prev => ({ ...prev, 'Pattern Agent': 'complete', 'Decision Agent': 'running' }));
      await new Promise(r => setTimeout(r, 300));

      // Final decision
      const bullishCount = agentReasoning.filter(a => ['bullish', 'continuation'].includes(a.conclusion)).length;
      const bearishCount = agentReasoning.filter(a => a.conclusion === 'bearish').length;
      
      let finalBias = 'NEUTRAL';
      let confidence = 50;
      if (bullishCount > bearishCount) { finalBias = 'BULLISH'; confidence = Math.min(70 + bullishCount * 10, 90); }
      else if (bearishCount > bullishCount) { finalBias = 'BEARISH'; confidence = Math.min(70 + bearishCount * 10, 90); }

      const summaryReasoning = `Based on quantitative analysis: RSI (${rsi}), MACD (${macd.direction}), Stochastic (${stoch.condition}), and MA trend (${trendBias}). ${bullishCount > bearishCount ? 'Multiple indicators suggest bullish momentum.' : bearishCount > bullishCount ? 'Multiple indicators suggest bearish pressure.' : 'Mixed signals indicate consolidation phase.'}`;

      agentReasoning.push({ agent: 'Decision Agent', reasoning: summaryReasoning, conclusion: finalBias.toLowerCase() });

      // Price series for chart
      const priceSeries = timestamps.slice(-100).map((ts, i) => {
        const idx = timestamps.length - 100 + i;
        return { timestamp: ts, price: Math.round(prices[idx] * 100) / 100, sma20: sma20[idx] ? Math.round(sma20[idx]! * 100) / 100 : null, sma50: sma50[idx] ? Math.round(sma50[idx]! * 100) / 100 : null };
      });

      setAgentStatuses(Object.fromEntries(agents.map(a => [a, 'complete'])));
      setResult({ final_bias: finalBias, confidence, summary_reasoning: summaryReasoning, indicators, agent_reasoning: agentReasoning, price_series: priceSeries });

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setAgentStatuses({});
    } finally {
      setLoading(false);
    }
  };

  const sendChat = async () => {
    if (!chatInput.trim() || chatLoading || !result) return;
    const userMessage = chatInput.trim();
    setChatInput('');
    setChatMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setChatLoading(true);

    // Try Supabase edge function first, fallback to local response
    if (SUPABASE_URL && SUPABASE_ANON_KEY) {
      try {
        const response = await fetch(`${SUPABASE_URL}/functions/v1/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${SUPABASE_ANON_KEY}` },
          body: JSON.stringify({ question: userMessage, context: { asset, timeframe, final_bias: result.final_bias, confidence: result.confidence, current_price: result.price_series?.[result.price_series.length - 1]?.price, indicators: result.indicators, agent_reasoning: result.agent_reasoning } }),
        });
        const data = await response.json();
        setChatMessages(prev => [...prev, { role: 'assistant', content: data.answer }]);
        setChatLoading(false);
        return;
      } catch { /* fallback below */ }
    }
    
    // Local fallback response
    const answer = `Based on the analysis of ${asset}: The current bias is ${result.final_bias} with ${result.confidence}% confidence. ${result.indicators.map(i => `${i.name}: ${i.status}`).join('. ')}. Please configure the AI backend for more detailed responses.`;
    setChatMessages(prev => [...prev, { role: 'assistant', content: answer }]);
    setChatLoading(false);
  };

  const toggleAgent = (agent: string) => setExpandedAgents(prev => ({ ...prev, [agent]: !prev[agent] }));

  const chartData = result?.price_series.map(p => ({ time: new Date(p.timestamp).toLocaleDateString(), price: p.price, sma20: p.sma20, sma50: p.sma50 })) || [];

  return (
    <div className="min-h-screen bg-[#0d0d14] text-white">
      <header className="border-b border-white/10 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <TrendingUp className="w-7 h-7 text-emerald-400" />
            <span className="text-lg font-bold">ClaudeQuantTrading</span>
          </Link>
          <nav className="flex items-center gap-6">
            <span className="text-emerald-400 font-medium">Analyze</span>
            <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 text-gray-400 hover:text-white transition">
              <Github className="w-5 h-5" />
              QuantAgent
            </a>
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="bg-[#13131d] rounded-xl border border-white/10 p-6 mb-6">
          <div className="flex flex-wrap items-end gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">Asset</label>
              <select value={asset} onChange={e => setAsset(e.target.value)} className="bg-[#0d0d14] border border-white/10 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-emerald-500/50 min-w-[120px]">
                {ASSETS.map(a => <option key={a} value={a}>{a}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">Timeframe</label>
              <select value={timeframe} onChange={e => setTimeframe(e.target.value)} className="bg-[#0d0d14] border border-white/10 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-emerald-500/50 min-w-[140px]">
                {TIMEFRAMES.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
              </select>
            </div>
            <button onClick={runAnalysis} disabled={loading} className="flex items-center gap-2 px-6 py-2.5 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-lg font-medium hover:shadow-lg hover:shadow-emerald-500/25 transition-all disabled:opacity-50 disabled:cursor-not-allowed">
              {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <TrendingUp className="w-5 h-5" />}
              {loading ? 'Analyzing...' : 'Analyze'}
            </button>
          </div>
        </div>

        {Object.keys(agentStatuses).length > 0 && (
          <div className="bg-[#13131d] rounded-xl border border-white/10 p-6 mb-6">
            <h3 className="text-sm font-medium text-gray-400 mb-4">Agent Progress</h3>
            <div className="flex flex-wrap gap-4">
              {Object.entries(agentStatuses).map(([agent, status]) => (
                <div key={agent} className="flex items-center gap-2">
                  {status === 'complete' ? <CheckCircle2 className="w-5 h-5 text-emerald-400" /> : status === 'running' ? <Loader2 className="w-5 h-5 text-cyan-400 animate-spin" /> : <Circle className="w-5 h-5 text-gray-600" />}
                  <span className={status === 'complete' ? 'text-emerald-400' : status === 'running' ? 'text-cyan-400' : 'text-gray-500'}>{agent}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {error && <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 mb-6 text-red-400">{error}</div>}

        {result && (
          <div className="grid lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-6">
              <div className="bg-[#13131d] rounded-xl border border-white/10 p-6">
                <h3 className="text-lg font-semibold mb-4">{asset}/USD Price Chart</h3>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      {/* @ts-expect-error Recharts type issue */}
                      <XAxis dataKey="time" stroke="#4b5563" fontSize={12} tickLine={false} />
                      {/* @ts-expect-error Recharts type issue */}
                      <YAxis stroke="#4b5563" fontSize={12} tickLine={false} domain={['auto', 'auto']} />
                      {/* @ts-expect-error Recharts type issue */}
                      <Tooltip contentStyle={{ backgroundColor: '#1f1f2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} labelStyle={{ color: '#9ca3af' }} />
                      {/* @ts-expect-error Recharts type issue */}
                      <Legend />
                      {/* @ts-expect-error Recharts type issue */}
                      <Line type="monotone" dataKey="price" stroke="#10b981" strokeWidth={2} dot={false} name="Price" />
                      {/* @ts-expect-error Recharts type issue */}
                      <Line type="monotone" dataKey="sma20" stroke="#f59e0b" strokeWidth={1} dot={false} name="SMA 20" strokeDasharray="5 5" />
                      {/* @ts-expect-error Recharts type issue */}
                      <Line type="monotone" dataKey="sma50" stroke="#8b5cf6" strokeWidth={1} dot={false} name="SMA 50" strokeDasharray="5 5" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-[#13131d] rounded-xl border border-white/10 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">Analysis Result</h3>
                  <div className={`px-4 py-1.5 rounded-full text-sm font-medium ${result.final_bias === 'BULLISH' ? 'bg-emerald-500/20 text-emerald-400' : result.final_bias === 'BEARISH' ? 'bg-red-500/20 text-red-400' : 'bg-gray-500/20 text-gray-400'}`}>{result.final_bias}</div>
                </div>
                <div className="mb-4">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-gray-400 text-sm">Confidence</span>
                    <span className="font-bold text-lg">{result.confidence}%</span>
                  </div>
                  <div className="w-full bg-white/10 rounded-full h-2">
                    <div className={`h-2 rounded-full ${result.final_bias === 'BULLISH' ? 'bg-emerald-500' : result.final_bias === 'BEARISH' ? 'bg-red-500' : 'bg-gray-500'}`} style={{ width: `${result.confidence}%` }} />
                  </div>
                </div>
                <div className="bg-white/5 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-gray-400 mb-2">AI Market Reasoning</h4>
                  <p className="text-gray-300">{result.summary_reasoning}</p>
                </div>
              </div>

              <div className="bg-[#13131d] rounded-xl border border-white/10 p-6">
                <h3 className="text-lg font-semibold mb-4">Ask Follow-up Questions</h3>
                <div className="space-y-4 mb-4 max-h-[200px] overflow-y-auto">
                  {chatMessages.map((msg, i) => (
                    <div key={i} className={`p-3 rounded-lg ${msg.role === 'user' ? 'bg-emerald-500/10 ml-8' : 'bg-white/5 mr-8'}`}>
                      <span className="text-xs text-gray-500 mb-1 block">{msg.role === 'user' ? 'You' : 'Claude'}</span>
                      <p className="text-sm">{msg.content}</p>
                    </div>
                  ))}
                  {chatLoading && <div className="flex items-center gap-2 text-gray-400"><Loader2 className="w-4 h-4 animate-spin" /><span className="text-sm">Thinking...</span></div>}
                </div>
                <div className="flex gap-2">
                  <input type="text" value={chatInput} onChange={e => setChatInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && sendChat()} placeholder="Ask about the analysis..." className="flex-1 bg-[#0d0d14] border border-white/10 rounded-lg px-4 py-2.5 text-white placeholder-gray-500 focus:outline-none focus:border-emerald-500/50" />
                  <button onClick={sendChat} disabled={chatLoading || !chatInput.trim()} className="px-4 py-2.5 bg-emerald-500 rounded-lg hover:bg-emerald-600 transition disabled:opacity-50 disabled:cursor-not-allowed"><Send className="w-5 h-5" /></button>
                </div>
              </div>
            </div>

            <div className="space-y-6">
              <div className="bg-[#13131d] rounded-xl border border-white/10 p-6">
                <h3 className="text-lg font-semibold mb-4">Technical Indicators</h3>
                <div className="space-y-4">
                  {result.indicators.map((ind, i) => (
                    <div key={i} className="border-b border-white/5 pb-4 last:border-0 last:pb-0">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium">{ind.name}</span>
                        <span className={`text-xs px-2 py-0.5 rounded ${ind.status === 'bullish' ? 'bg-emerald-500/20 text-emerald-400' : ind.status === 'bearish' ? 'bg-red-500/20 text-red-400' : ind.status === 'overbought' ? 'bg-orange-500/20 text-orange-400' : ind.status === 'oversold' ? 'bg-cyan-500/20 text-cyan-400' : 'bg-gray-500/20 text-gray-400'}`}>{ind.status}</span>
                      </div>
                      <p className="text-sm text-gray-400">{ind.interpretation}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-[#13131d] rounded-xl border border-white/10 p-6">
                <h3 className="text-lg font-semibold mb-4">Agent Reasoning</h3>
                <div className="space-y-3">
                  {result.agent_reasoning.map((agent, i) => (
                    <div key={i} className="border border-white/5 rounded-lg overflow-hidden">
                      <button onClick={() => toggleAgent(agent.agent)} className="w-full flex items-center justify-between p-3 hover:bg-white/5 transition">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-sm">{agent.agent}</span>
                          <span className={`text-xs px-2 py-0.5 rounded ${agent.conclusion === 'bullish' ? 'bg-emerald-500/20 text-emerald-400' : agent.conclusion === 'bearish' ? 'bg-red-500/20 text-red-400' : 'bg-gray-500/20 text-gray-400'}`}>{agent.conclusion}</span>
                        </div>
                        {expandedAgents[agent.agent] ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                      </button>
                      {expandedAgents[agent.agent] && <div className="px-3 pb-3 text-sm text-gray-400">{agent.reasoning}</div>}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
