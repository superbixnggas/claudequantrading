import { Link } from 'react-router-dom';
import { TrendingUp, BarChart3, Brain, Zap } from 'lucide-react';

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[#0d0d14] text-white overflow-hidden relative">
      {/* Background gradient effects */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-gradient-to-r from-emerald-500/5 to-cyan-500/5 rounded-full blur-3xl" />
      </div>
      
      {/* Grid pattern overlay */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:50px_50px]" />

      {/* Navigation */}
      <nav className="relative z-10 flex items-center justify-between px-8 py-6 max-w-7xl mx-auto">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-8 h-8 text-emerald-400" />
          <span className="text-xl font-bold">ClaudeQuantTrading</span>
        </div>
        <div className="flex items-center gap-6">
          <Link to="/analyze" className="text-gray-400 hover:text-white transition">Analyze</Link>
          <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-white transition">GitHub</a>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="relative z-10 flex flex-col items-center justify-center px-8 pt-20 pb-32 max-w-5xl mx-auto text-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20 mb-8">
          <Zap className="w-4 h-4 text-emerald-400" />
          <span className="text-sm text-emerald-400">Powered by AI Agent System</span>
        </div>
        
        <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
          <span className="bg-gradient-to-r from-white via-emerald-200 to-cyan-200 bg-clip-text text-transparent">
            ClaudeQuantTrading
          </span>
        </h1>
        
        <p className="text-xl md:text-2xl text-gray-400 mb-12 max-w-2xl">
          Quant-based market analysis with AI reasoning. 
          Multi-agent system that interprets technical indicators 
          and provides actionable insights.
        </p>

        <Link 
          to="/analyze"
          className="group relative inline-flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-xl font-semibold text-lg hover:shadow-lg hover:shadow-emerald-500/25 transition-all duration-300"
        >
          <Brain className="w-5 h-5" />
          Start with Claude Agent
          <span className="absolute inset-0 rounded-xl bg-gradient-to-r from-emerald-400 to-cyan-400 opacity-0 group-hover:opacity-20 transition-opacity" />
        </Link>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-3 gap-6 mt-24 w-full">
          <FeatureCard 
            icon={<BarChart3 className="w-6 h-6" />}
            title="Real Indicators"
            description="RSI, MACD, Stochastic, Moving Averages calculated mathematically from live data"
          />
          <FeatureCard 
            icon={<Brain className="w-6 h-6" />}
            title="Multi-Agent System"
            description="Four specialized agents analyze and reason about market conditions"
          />
          <FeatureCard 
            icon={<Zap className="w-6 h-6" />}
            title="Interactive Chat"
            description="Ask follow-up questions about the analysis with context-aware responses"
          />
        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/5 py-8 px-8">
        <div className="max-w-7xl mx-auto flex items-center justify-between text-gray-500 text-sm">
          <span>Powered by QuantAgent | LLM backend flexible</span>
          <span>ClaudeQuantTrading 2026</span>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <div className="p-6 rounded-2xl bg-white/5 border border-white/10 hover:border-emerald-500/30 transition-colors">
      <div className="w-12 h-12 rounded-xl bg-emerald-500/10 flex items-center justify-center text-emerald-400 mb-4">
        {icon}
      </div>
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <p className="text-gray-400 text-sm">{description}</p>
    </div>
  );
}
