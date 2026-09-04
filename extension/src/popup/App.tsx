/**
 * Kavach — Extension Popup (React + TypeScript)
 * =====================================================
 * Main popup component rendered when the extension icon is clicked.
 * Features: email scan, risk gauge, XAI explanation, recommendations,
 *           drag-and-drop file upload, clipboard paste, history.
 */

import React, { useCallback, useEffect, useRef, useState } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// Type definitions
// ─────────────────────────────────────────────────────────────────────────────
type RiskLevel = "safe" | "low" | "medium" | "high" | "critical";
type Classification = "LEGITIMATE" | "SPAM" | "PHISHING";

interface RiskBreakdown {
  bert_contribution: number;
  url_contribution: number;
  header_contribution: number;
  sender_contribution: number;
  keyword_contribution: number;
}

interface Recommendation {
  severity: "critical" | "high" | "medium" | "low" | "info";
  category: string;
  message: string;
  action: string;
}

interface TokenImportance {
  token: string;
  importance: number;
  layer: string;
}

interface Explanation {
  method: string;
  natural_language_summary: string;
  top_positive_tokens: [string, number][];
  top_negative_tokens: [string, number][];
}

interface URLAnalysis {
  urls_found: number;
  malicious_url_count: number;
  newly_registered_domain_count: number;
  typosquat_count: number;
  aggregate_risk: number;
  summary: string;
}

interface HeaderAnalysis {
  spf_pass: boolean;
  dkim_pass: boolean;
  dmarc_pass: boolean;
  sender_trust_score: number;
  risk_score: number;
  flags: string[];
}

interface AnalysisResult {
  email_hash: string;
  subject: string;
  sender: string;
  classification: Classification;
  risk_score: number;
  risk_level: RiskLevel;
  confidence: number;
  flagged_keywords: string[];
  spam_patterns_found: string[];
  recommendations: Recommendation[];
  risk_breakdown: RiskBreakdown;
  url_analysis: URLAnalysis;
  header_analysis: HeaderAnalysis;
  explanation?: Explanation;
  bert_probabilities: Record<Classification, number>;
  bert_inference_ms: number;
  total_latency_ms: number;
  fromCache?: boolean;
  error?: string;
}

type TabId = "scan" | "result" | "history" | "settings";

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
const RISK_COLORS: Record<RiskLevel, { bg: string; text: string; border: string; gauge: string }> = {
  safe:     { bg: "#f0fdf4", text: "#166534", border: "#86efac", gauge: "#22c55e" },
  low:      { bg: "#f0fdf4", text: "#15803d", border: "#4ade80", gauge: "#4ade80" },
  medium:   { bg: "#fefce8", text: "#854d0e", border: "#fbbf24", gauge: "#f59e0b" },
  high:     { bg: "#fff1f2", text: "#9f1239", border: "#fda4af", gauge: "#ef4444" },
  critical: { bg: "#fef2f2", text: "#7f1d1d", border: "#f87171", gauge: "#dc2626" },
};

const SEVERITY_ICONS: Record<string, string> = {
  critical: "🚨",
  high:     "⚠️",
  medium:   "🔶",
  low:      "🔵",
  info:     "ℹ️",
};

const CLASSIFICATION_ICONS: Record<Classification, string> = {
  LEGITIMATE: "✅",
  SPAM:       "📧",
  PHISHING:   "🎣",
};

// ─────────────────────────────────────────────────────────────────────────────
// Utility components
// ─────────────────────────────────────────────────────────────────────────────
const Badge: React.FC<{
  label: string;
  variant: "green" | "amber" | "red" | "blue" | "gray";
}> = ({ label, variant }) => {
  const colors = {
    green: "bg-green-100 text-green-800 border-green-200",
    amber: "bg-amber-100 text-amber-800 border-amber-200",
    red:   "bg-red-100   text-red-800   border-red-200",
    blue:  "bg-blue-100  text-blue-800  border-blue-200",
    gray:  "bg-gray-100  text-gray-700  border-gray-200",
  };
  return (
    <span className={`inline-flex items-center px-2 py-0.5 text-xs font-medium border rounded ${colors[variant]}`}>
      {label}
    </span>
  );
};

const AuthCheck: React.FC<{ pass: boolean; label: string }> = ({ pass, label }) => (
  <div className="flex items-center justify-between py-1.5 border-b border-gray-100 last:border-0">
    <span className="text-xs text-gray-500 font-mono">{label}</span>
    <span className={`text-xs font-semibold font-mono ${pass ? "text-green-600" : "text-red-500"}`}>
      {pass ? "✓ PASS" : "✗ FAIL"}
    </span>
  </div>
);

const SignalBar: React.FC<{
  label: string;
  value: number;
  max: number;
  color: string;
}> = ({ label, value, max, color }) => {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="flex items-center gap-2 py-1">
      <span className="text-xs text-gray-400 font-mono w-20 flex-shrink-0">{label}</span>
      <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className="text-xs font-mono text-gray-500 w-8 text-right">{value.toFixed(0)}</span>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Risk Gauge SVG
// ─────────────────────────────────────────────────────────────────────────────
const RiskGauge: React.FC<{ score: number; level: RiskLevel }> = ({ score, level }) => {
  const colors = RISK_COLORS[level];
  const CIRC = 220;
  const fill = CIRC - (CIRC * Math.min(score, 100)) / 100 * 0.95;

  return (
    <div className="flex flex-col items-center">
      <svg width="160" height="90" viewBox="0 0 160 90" aria-label={`Risk score: ${score} out of 100`}>
        {/* Background arc */}
        <path
          d="M10 80 A70 70 0 0 1 150 80"
          fill="none"
          stroke="#f1f5f9"
          strokeWidth="14"
          strokeLinecap="round"
        />
        {/* Score arc */}
        <path
          d="M10 80 A70 70 0 0 1 150 80"
          fill="none"
          stroke={colors.gauge}
          strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={CIRC}
          strokeDashoffset={fill}
          style={{ transition: "stroke-dashoffset 1s ease, stroke 0.4s" }}
        />
        {/* Score number */}
        <text
          x="80" y="62"
          textAnchor="middle"
          fontSize="30"
          fontWeight="800"
          fontFamily="system-ui, sans-serif"
          fill={colors.gauge}
        >
          {score}
        </text>
        {/* Label */}
        <text
          x="80" y="78"
          textAnchor="middle"
          fontSize="9"
          fontFamily="monospace"
          fill="#94a3b8"
          letterSpacing="1"
        >
          {level.toUpperCase()} RISK
        </text>
      </svg>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────────────────────────────────────
export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabId>("scan");
  const [emailText, setEmailText] = useState("");
  const [isScanning, setIsScanning] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [apiStatus, setApiStatus] = useState<"online" | "offline" | "checking">("checking");
  const [history, setHistory] = useState<AnalysisResult[]>([]);
  const [showExplain, setShowExplain] = useState(false);
  const [expandedReco, setExpandedReco] = useState<number | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // ── Check API status on mount ────────────────────────────────────────────
  const checkStatus = useCallback(async () => {
    try {
      const res = await chrome.runtime.sendMessage({ type: "GET_API_STATUS" });
      if (res?.status === "healthy") {
        setApiStatus("online");
        return;
      }
    } catch {}

    try {
      const direct = await fetch("http://localhost:8000/health");
      if (direct.ok) {
        setApiStatus("online");
        return;
      }
    } catch {}

    setApiStatus("offline");
  }, []);

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 15000);


    // Load cached last result
    if (chrome?.storage?.session) {
      chrome.storage.session.get("lastResult").then((data) => {
        if (data?.lastResult) {
          setResult(data.lastResult);
          setActiveTab("result");
        }
      }).catch(() => {});
    }

    // Load history
    if (chrome?.storage?.local) {
      chrome.storage.local.get("analysisHistory").then((data) => {
        setHistory(data?.analysisHistory || []);
      }).catch(() => {});
    }

    return () => clearInterval(interval);
  }, [checkStatus]);

  // ── Scan handler ──────────────────────────────────────────────────────────
  const handleScan = useCallback(async () => {
    const text = emailText.trim();
    if (!text || isScanning) return;

    setIsScanning(true);
    try {
      let res: AnalysisResult;

      // Try chrome.runtime.sendMessage first, fall back to direct fetch
      try {
        res = await chrome.runtime.sendMessage({
          type: "ANALYZE_EMAIL",
          payload: { content: text, subject: "", sender: "" },
        });
      } catch {
        // Fallback: call the API directly if service worker is unavailable
        const resp = await fetch("http://localhost:8000/api/v1/analyze/text", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content: text, subject: "", sender: "" }),
        });
        if (!resp.ok) throw new Error(`API error: ${resp.status}`);
        res = await resp.json();
      }

      if (!res || typeof res !== "object") {
        alert("Received an invalid response from the server.");
        return;
      }

      if (res.error) {
        alert(`Analysis failed: ${res.error}`);
        return;
      }

      // Normalize: fill in defaults for any missing fields so rendering never crashes
      const safe: AnalysisResult = {
        email_hash: res.email_hash ?? "",
        subject: res.subject ?? "",
        sender: res.sender ?? "",
        classification: res.classification ?? "LEGITIMATE",
        risk_score: res.risk_score ?? 0,
        risk_level: res.risk_level ?? "safe",
        confidence: res.confidence ?? 0,
        flagged_keywords: Array.isArray(res.flagged_keywords) ? res.flagged_keywords : [],
        spam_patterns_found: Array.isArray(res.spam_patterns_found) ? res.spam_patterns_found : [],
        recommendations: Array.isArray(res.recommendations) ? res.recommendations : [],
        risk_breakdown: {
          bert_contribution: res.risk_breakdown?.bert_contribution ?? 0,
          url_contribution: res.risk_breakdown?.url_contribution ?? 0,
          header_contribution: res.risk_breakdown?.header_contribution ?? 0,
          sender_contribution: res.risk_breakdown?.sender_contribution ?? 0,
          keyword_contribution: res.risk_breakdown?.keyword_contribution ?? 0,
        },
        url_analysis: res.url_analysis ?? { urls_found: 0, malicious_url_count: 0, newly_registered_domain_count: 0, typosquat_count: 0, aggregate_risk: 0, summary: "" },
        header_analysis: res.header_analysis ?? { spf_pass: false, dkim_pass: false, dmarc_pass: false, sender_trust_score: 0, risk_score: 0, flags: [] },
        explanation: res.explanation ?? undefined,
        bert_probabilities: res.bert_probabilities ?? { LEGITIMATE: 0, SPAM: 0, PHISHING: 0 },
        bert_inference_ms: res.bert_inference_ms ?? 0,
        total_latency_ms: res.total_latency_ms ?? 0,
        fromCache: res.fromCache,
        error: res.error,
      };

      setResult(safe);
      setActiveTab("result");
      if (chrome?.storage?.session) {
        chrome.storage.session.set({ lastResult: safe }).catch(() => {});
      }

      // Update history (last 20)
      setHistory(prev => {
        const updated = [safe, ...prev.filter(h => h.email_hash !== safe.email_hash)].slice(0, 20);
        if (chrome?.storage?.local) {
          chrome.storage.local.set({ analysisHistory: updated }).catch(() => {});
        }
        return updated;
      });
    } catch (err) {
      alert("Connection error. Please check your Kavach API connection.");
    } finally {
      setIsScanning(false);
    }
  }, [emailText, isScanning]);

  // ── Clipboard paste ───────────────────────────────────────────────────────
  const handlePasteFromClipboard = useCallback(async () => {
    try {
      const text = await navigator.clipboard.readText();
      setEmailText(text);
      textareaRef.current?.focus();
    } catch {
      alert("Could not read clipboard. Please paste manually with Ctrl+V.");
    }
  }, []);

  // ── Drag and drop ─────────────────────────────────────────────────────────
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (!file) return;

    if (!file.name.match(/\.(eml|msg|txt)$/i)) {
      alert("Only .eml, .msg, and .txt files are supported.");
      return;
    }

    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result as string;
      setEmailText(text);
    };
    reader.readAsText(file);
  }, []);

  // ── Keyboard shortcut ─────────────────────────────────────────────────────
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") handleScan();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [handleScan]);

  const riskColors = result ? RISK_COLORS[result.risk_level] : RISK_COLORS.safe;

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div className="w-96 min-h-64 bg-white flex flex-col font-sans text-gray-800 shadow-2xl">

      {/* ── Topbar ───────────────────────────────────────────────────────── */}
      <header className="flex items-center justify-between px-4 py-2.5 bg-slate-900 text-white">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 bg-blue-600 rounded-lg flex items-center justify-center text-sm">🛡️</div>
          <div>
            <div className="text-sm font-bold tracking-tight">Kavach</div>
            <div className="text-xs text-slate-400 font-mono">Email Security · BERT + XAI</div>
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          <div className={`w-2 h-2 rounded-full ${apiStatus === "online" ? "bg-green-400 animate-pulse" : "bg-red-400"}`} />
          <span className="text-xs font-mono text-slate-300">
            {apiStatus === "checking" ? "…" : apiStatus}
          </span>
        </div>
      </header>

      {/* ── Tab Bar ──────────────────────────────────────────────────────── */}
      <nav className="flex border-b border-gray-200 bg-gray-50">
        {(["scan", "result", "history", "settings"] as TabId[]).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`flex-1 py-2 text-xs font-semibold capitalize transition-colors
              ${activeTab === tab
                ? "text-blue-600 border-b-2 border-blue-600 bg-white"
                : "text-gray-400 hover:text-gray-600"}`}
          >
            {tab}
            {tab === "result" && result && (
              <span className={`ml-1 inline-block w-2 h-2 rounded-full ${
                result.risk_level === "safe" ? "bg-green-400" :
                result.risk_level === "critical" ? "bg-red-500" : "bg-amber-400"
              }`} />
            )}
          </button>
        ))}
      </nav>

      {/* ── Scan Tab ─────────────────────────────────────────────────────── */}
      {activeTab === "scan" && (
        <div className="p-4 flex flex-col gap-3">
          <div className="flex gap-2">
            <button
              onClick={handlePasteFromClipboard}
              className="flex-1 text-xs py-1.5 px-3 border border-gray-200 rounded-lg
                hover:border-blue-300 hover:text-blue-600 transition-colors font-medium"
            >
              📋 Paste from Clipboard
            </button>
            <button
              onClick={() => setEmailText("")}
              className="text-xs py-1.5 px-3 border border-gray-200 rounded-lg
                hover:border-gray-300 text-gray-400 hover:text-gray-600 transition-colors"
            >
              Clear
            </button>
          </div>

          <textarea
            ref={textareaRef}
            value={emailText}
            onChange={(e) => setEmailText(e.target.value)}
            placeholder="Paste email content here…&#10;&#10;Subject, headers, and body are all analyzed."
            rows={7}
            className="w-full text-xs font-mono border border-gray-200 rounded-lg p-3
              focus:outline-none focus:ring-2 focus:ring-blue-300 focus:border-blue-300
              resize-none text-gray-700 placeholder-gray-300 leading-relaxed"
          />

          {/* Drag and drop zone */}
          <div
            onDrop={handleDrop}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            className={`border-2 border-dashed rounded-lg py-3 text-center cursor-pointer
              transition-colors text-xs
              ${isDragging
                ? "border-blue-400 bg-blue-50 text-blue-600"
                : "border-gray-200 text-gray-400 hover:border-gray-300"}`}
          >
            📎 Drop .eml, .msg, or .txt file here
          </div>

          <button
            onClick={handleScan}
            disabled={!emailText.trim() || isScanning || apiStatus === "offline"}
            className="w-full py-2.5 bg-blue-600 text-white text-sm font-bold rounded-lg
              hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed
              transition-colors flex items-center justify-center gap-2"
          >
            {isScanning ? (
              <>
                <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Analyzing…
              </>
            ) : (
              "🛡️ Analyze Email  ⌘↵"
            )}
          </button>

          {apiStatus === "offline" && (
            <p className="text-xs text-red-500 text-center">
              API offline — check your connection settings.
            </p>
          )}
        </div>
      )}

      {/* ── Result Tab ───────────────────────────────────────────────────── */}
      {activeTab === "result" && result && !result.error && (
        <div className="flex flex-col overflow-y-auto max-h-[520px]">

          {/* Risk header */}
          <div
            className="px-4 pt-4 pb-3 text-center"
            style={{ background: `linear-gradient(135deg, ${riskColors.bg}, white)` }}
          >
            <RiskGauge score={result.risk_score} level={result.risk_level} />
            <div className="flex items-center justify-center gap-2 mt-1">
              <span className="text-lg">{CLASSIFICATION_ICONS[result.classification]}</span>
              <span className="text-sm font-bold" style={{ color: riskColors.text }}>
                {result.classification}
              </span>
              <Badge
                label={`${(result.confidence * 100).toFixed(0)}% confidence`}
                variant={result.risk_level === "safe" ? "green" : result.risk_level === "critical" ? "red" : "amber"}
              />
            </div>
            {result.fromCache && (
              <p className="text-xs text-gray-400 mt-1 font-mono">cached result</p>
            )}
          </div>

          <div className="px-4 py-3 space-y-4">

            {/* Risk breakdown signals */}
            <section>
              <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-2 font-mono">
                Risk Signal Breakdown
              </h3>
              <SignalBar label="BERT Model"   value={result.risk_breakdown.bert_contribution}    max={35} color="#ef4444" />
              <SignalBar label="URL Risk"     value={result.risk_breakdown.url_contribution}     max={25} color="#f59e0b" />
              <SignalBar label="Headers"      value={result.risk_breakdown.header_contribution}  max={20} color="#3b82f6" />
              <SignalBar label="Sender"       value={result.risk_breakdown.sender_contribution}  max={10} color="#8b5cf6" />
              <SignalBar label="Keywords"     value={result.risk_breakdown.keyword_contribution} max={10} color="#06b6d4" />
            </section>

            {/* Auth checks */}
            {result.header_analysis && (
              <section>
                <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-2 font-mono">
                  Email Authentication
                </h3>
                <div className="bg-gray-50 rounded-lg px-3 py-1">
                  <AuthCheck pass={!!result.header_analysis.spf_pass}  label="SPF" />
                  <AuthCheck pass={!!result.header_analysis.dkim_pass} label="DKIM" />
                  <AuthCheck pass={!!result.header_analysis.dmarc_pass} label="DMARC" />
                </div>
              </section>
            )}

            {/* URL analysis */}
            {Boolean(result.url_analysis && result.url_analysis.urls_found > 0) && (
              <section>
                <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-2 font-mono">
                  URL Analysis ({result.url_analysis?.urls_found || 0} found)
                </h3>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    { label: "Malicious", val: result.url_analysis?.malicious_url_count || 0,            bad: true },
                    { label: "New Domains", val: result.url_analysis?.newly_registered_domain_count || 0, bad: true },
                    { label: "Typosquats",  val: result.url_analysis?.typosquat_count || 0,               bad: true },
                    { label: "Agg. Risk",   val: `${((result.url_analysis?.aggregate_risk || 0) * 100).toFixed(0)}%`, bad: (result.url_analysis?.aggregate_risk || 0) > 0.3 },
                  ].map(({ label, val, bad }) => (
                    <div key={label} className={`px-3 py-2 rounded-lg text-center border ${
                      bad && Number(val) > 0 ? "bg-red-50 border-red-200" : "bg-gray-50 border-gray-200"
                    }`}>
                      <div className={`text-lg font-bold ${bad && Number(val) > 0 ? "text-red-600" : "text-gray-600"}`}>{val}</div>
                      <div className="text-xs text-gray-400 font-mono">{label}</div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* BERT probabilities */}
            {result.bert_probabilities && (
              <section>
                <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-2 font-mono">
                  Model Probabilities
                </h3>
                {(["LEGITIMATE", "SPAM", "PHISHING"] as Classification[]).map((cls) => {
                  const prob = result.bert_probabilities?.[cls] ?? 0;
                  const colors = { LEGITIMATE: "#22c55e", SPAM: "#f59e0b", PHISHING: "#ef4444" };
                  return (
                    <div key={cls} className="flex items-center gap-2 py-1">
                      <span className="text-xs font-mono text-gray-400 w-20">{cls}</span>
                    <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-700"
                        style={{ width: `${prob * 100}%`, background: colors[cls] }}
                      />
                    </div>
                    <span className="text-xs font-mono text-gray-500 w-10 text-right">
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                );
              })}
            </section>
          )}

            {/* Flagged keywords */}
            {Array.isArray(result.flagged_keywords) && result.flagged_keywords.length > 0 && (
              <section>
                <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-2 font-mono">
                  Flagged Keywords
                </h3>
                <div className="flex flex-wrap gap-1">
                  {result.flagged_keywords.map((kw) => (
                    <span key={kw} className="px-2 py-0.5 bg-red-50 border border-red-200 rounded text-xs font-mono text-red-700">
                      {kw}
                    </span>
                  ))}
                </div>
              </section>
            )}

            {/* XAI Explanation */}
            {result.explanation && (
              <section>
                <button
                  onClick={() => setShowExplain((v) => !v)}
                  className="flex items-center justify-between w-full text-xs font-bold
                    text-gray-400 uppercase tracking-widest mb-2 font-mono hover:text-gray-600"
                >
                  <span>AI Explanation (SHAP)</span>
                  <span>{showExplain ? "▲" : "▼"}</span>
                </button>
                {showExplain && (
                  <div className="space-y-2">
                    <p className="text-xs text-gray-600 leading-relaxed bg-blue-50 rounded-lg p-3 border border-blue-100">
                      {result.explanation.natural_language_summary}
                    </p>
                    <div className="text-xs font-mono text-gray-400 mb-1">
                      Method: {result.explanation.method}
                    </div>
                    {result.explanation.top_positive_tokens.length > 0 && (
                      <div>
                        <div className="text-xs text-gray-400 mb-1">Top risk tokens:</div>
                        <div className="flex flex-wrap gap-1">
                          {result.explanation.top_positive_tokens.map(([tok, val]) => (
                            <span key={tok}
                              className="px-1.5 py-0.5 bg-red-100 text-red-700 rounded text-xs font-mono border border-red-200"
                              title={`SHAP value: ${val.toFixed(4)}`}
                            >
                              ▲ {tok}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </section>
            )}

            {/* Recommendations */}
            <section>
              <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-2 font-mono">
                Recommendations
              </h3>
              <div className="space-y-2">
                {(result.recommendations ?? []).map((rec, i) => (
                  <div
                    key={i}
                    className="rounded-lg border overflow-hidden cursor-pointer"
                    style={{
                      borderColor: rec.severity === "critical" ? "#fca5a5" :
                                   rec.severity === "high"     ? "#fdba74" :
                                   rec.severity === "info"     ? "#bfdbfe" : "#fde68a",
                      background:  rec.severity === "critical" ? "#fef2f2" :
                                   rec.severity === "high"     ? "#fff7ed" :
                                   rec.severity === "info"     ? "#eff6ff" : "#fffbeb",
                    }}
                    onClick={() => setExpandedReco(expandedReco === i ? null : i)}
                  >
                    <div className="flex items-start gap-2 px-3 py-2">
                      <span className="text-base flex-shrink-0">{SEVERITY_ICONS[rec.severity] ?? "🔵"}</span>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-0.5">
                          <span className="text-xs font-bold text-gray-600">{rec.category}</span>
                          <Badge
                            label={rec.severity}
                            variant={
                              rec.severity === "critical" || rec.severity === "high" ? "red" :
                              rec.severity === "medium" ? "amber" :
                              rec.severity === "info"   ? "blue"  : "gray"
                            }
                          />
                        </div>
                        <p className="text-xs text-gray-600 leading-relaxed">{rec.message}</p>
                        {expandedReco === i && (
                          <p className="text-xs text-gray-700 font-semibold mt-1 pt-1 border-t border-gray-200">
                            → {rec.action}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>

            {/* Performance footer */}
            <div className="flex items-center justify-between pt-2 border-t border-gray-100">
              <span className="text-xs font-mono text-gray-300">
                {(result.total_latency_ms ?? 0).toFixed(1)}ms · BERT {(result.bert_inference_ms ?? 0).toFixed(1)}ms
              </span>
              <button
                onClick={() => { setEmailText(""); setActiveTab("scan"); }}
                className="text-xs text-blue-500 hover:text-blue-700 font-medium"
              >
                Scan another →
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── History Tab ──────────────────────────────────────────────────── */}
      {activeTab === "history" && (
        <div className="p-4 flex flex-col gap-2 overflow-y-auto max-h-96">
          {history.length === 0 ? (
            <p className="text-xs text-gray-400 text-center py-8">No scans yet.</p>
          ) : (
            history.map((h, i) => (
              <button
                key={h.email_hash + i}
                onClick={() => { setResult(h); setActiveTab("result"); }}
                className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200 hover:border-gray-300 text-left transition-colors"
              >
                <span className="text-lg">{CLASSIFICATION_ICONS[h.classification]}</span>
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-semibold text-gray-700 truncate">
                    {h.subject || h.email_hash}
                  </div>
                  <div className="text-xs text-gray-400 font-mono">
                    Risk: {h.risk_score} · {h.classification}
                  </div>
                </div>
                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold text-white flex-shrink-0"
                  style={{ background: RISK_COLORS[h.risk_level].gauge }}
                >
                  {h.risk_score}
                </div>
              </button>
            ))
          )}
        </div>
      )}

      {/* ── Settings Tab ─────────────────────────────────────────────────── */}
      {activeTab === "settings" && (
        <div className="p-4 space-y-4">
          <div>
            <label className="block text-xs font-bold text-gray-500 uppercase tracking-widest mb-1 font-mono">
              API Endpoint
            </label>
            <input
              type="url"
              defaultValue="https://kavach-api.yourdomain.com"
              className="w-full text-xs border border-gray-200 rounded-lg px-3 py-2 font-mono
                focus:outline-none focus:ring-2 focus:ring-blue-300"
            />
          </div>
          <div>
            <label className="block text-xs font-bold text-gray-500 uppercase tracking-widest mb-1 font-mono">
              Alert Threshold
            </label>
            <select
              defaultValue="60"
              className="w-full text-xs border border-gray-200 rounded-lg px-3 py-2
              focus:outline-none focus:ring-2 focus:ring-blue-300">
              <option value="40">Medium (40+)</option>
              <option value="60">High (60+)</option>
              <option value="80">Critical only (80+)</option>
            </select>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-600">Auto-scan open emails</span>
            <div className="w-10 h-5 bg-blue-500 rounded-full relative cursor-pointer">
              <div className="absolute right-0.5 top-0.5 w-4 h-4 bg-white rounded-full shadow" />
            </div>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-600">Show inline risk badge</span>
            <div className="w-10 h-5 bg-blue-500 rounded-full relative cursor-pointer">
              <div className="absolute right-0.5 top-0.5 w-4 h-4 bg-white rounded-full shadow" />
            </div>
          </div>
          <div className="pt-2 border-t border-gray-100">
            <p className="text-xs text-gray-400 font-mono text-center">
              Kavach v1.0.0 · BERT + XAI
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
