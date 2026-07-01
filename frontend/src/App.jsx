import { useState, useRef } from 'react';
import {
  GitPullRequest, FileCode, Bug, Lock, Zap, PenLine,
  CheckCircle2, Loader2, Play, AlertCircle, Clock, FileDiff
} from 'lucide-react';
import './App.css';

const API_BASE = 'http://localhost:8000';

const CATEGORY_META = {
  bug:         { icon: Bug,          color: '#F85149', label: 'Bug'         },
  security:    { icon: Lock,         color: '#A371F7', label: 'Security'    },
  performance: { icon: Zap,          color: '#D29922', label: 'Performance' },
  style:       { icon: PenLine,      color: '#58A6FF', label: 'Style'       },
  ok:          { icon: CheckCircle2, color: '#3FB950', label: 'Clean'       },
};

function StampIcon({ category }) {
  const meta = CATEGORY_META[category] || CATEGORY_META.ok;
  const Icon = meta.icon;
  return (
    <span className="stamp" style={{ '--stamp-color': meta.color }}>
      <Icon size={12} strokeWidth={2.5} />
      {meta.label}
    </span>
  );
}

function ConfidenceBar({ value }) {
  const pct = Math.round(value * 100);
  return (
    <div className="confidence-bar" title={`${pct}% confidence`}>
      <div className="confidence-fill" style={{ width: `${pct}%` }} />
      <span className="confidence-label">{pct}%</span>
    </div>
  );
}

function Finding({ finding }) {
  const meta = CATEGORY_META[finding.category] || CATEGORY_META.ok;
  return (
    <div className="finding" style={{ '--accent': meta.color }}>
      <div className="finding-header">
        <StampIcon category={finding.category} />
        <ConfidenceBar value={finding.confidence} />
      </div>
      <p className="finding-feedback">
        {finding.feedback
            .replace(/```[a-z]*\n?/g, '')
            .replace(/```/g, '')
            .replace(/`([^`]+)`/g, '$1')
            .trim()}
        </p>
      <pre className="finding-code"><code>{finding.chunk}</code></pre>
    </div>
  );
}

function FileBlock({ file }) {
  const actionable = file.findings.filter(f => f.category !== 'ok');
  return (
    <div className="file-block">
      <div className="file-header">
        <FileDiff size={14} />
        <span className="file-name">{file.filename}</span>
        <span className="file-stats">
          <span className="add">+{file.additions}</span>
          <span className="del">−{file.deletions}</span>
        </span>
        <span className="file-lang">{file.language}</span>
      </div>
      {actionable.length === 0 ? (
        <div className="file-clean">
          <CheckCircle2 size={13} /> No issues found
        </div>
      ) : (
        <div className="findings-list">
          {actionable.map((f, i) => <Finding key={i} finding={f} />)}
        </div>
      )}
    </div>
  );
}

function SummaryBar({ result }) {
  const { category_counts, total_chunks, total_issues, elapsed_seconds } = result;
  return (
    <div className="summary-bar">
      <div className="summary-stats">
        <span><strong>{total_chunks}</strong> chunks</span>
        <span className="dot">·</span>
        <span><strong>{total_issues}</strong> issues</span>
        <span className="dot">·</span>
        <span className="elapsed"><Clock size={11} /> {elapsed_seconds}s</span>
      </div>
      <div className="summary-pills">
        {Object.entries(category_counts)
          .filter(([cat, count]) => cat !== 'ok' && count > 0)
          .map(([cat, count]) => {
            const meta = CATEGORY_META[cat];
            const Icon = meta.icon;
            return (
              <span key={cat} className="pill" style={{ '--pill-color': meta.color }}>
                <Icon size={11} /> {count}
              </span>
            );
          })}
      </div>
    </div>
  );
}

export default function App() {
  const [mode, setMode]               = useState('pr');
  const [prUrl, setPrUrl]             = useState('');
  const [code, setCode]               = useState('');
  const [filename, setFilename]       = useState('snippet.py');
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState(null);
  const [result, setResult]           = useState(null);
  const [loadingStage, setLoadingStage] = useState('');
  const stageTimer = useRef(null);

  const STAGES = [
    'Fetching source…',
    'Chunking diff…',
    'Running CodeBERT…',
    'Generating feedback…',
  ];

  const runReview = async () => {
    setError(null);
    setResult(null);
    setLoading(true);

    let idx = 0;
    setLoadingStage(STAGES[0]);
    stageTimer.current = setInterval(() => {
      idx = (idx + 1) % STAGES.length;
      setLoadingStage(STAGES[idx]);
    }, 3500);

    try {
      const endpoint = mode === 'pr' ? '/review/pr' : '/review/code';
      const body = mode === 'pr' ? { pr_url: prUrl } : { code, filename };

      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Something went wrong');
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      clearInterval(stageTimer.current);
      setLoading(false);
    }
  };

  const canSubmit = mode === 'pr' ? prUrl.trim().length > 0 : code.trim().length > 0;

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <GitPullRequest size={19} strokeWidth={2.5} />
          <div>
            <h1>Code Review Assistant</h1>
            <p>CodeBERT · Llama 3.2 · FastAPI</p>
          </div>
        </div>

        <div className="mode-toggle">
          <button className={mode === 'pr' ? 'active' : ''} onClick={() => setMode('pr')}>
            <GitPullRequest size={13} /> PR URL
          </button>
          <button className={mode === 'code' ? 'active' : ''} onClick={() => setMode('code')}>
            <FileCode size={13} /> Paste Code
          </button>
        </div>

        {mode === 'pr' ? (
          <div className="input-group">
            <label>GitHub Pull Request URL</label>
            <input
              type="text"
              placeholder="https://github.com/owner/repo/pull/123"
              value={prUrl}
              onChange={e => setPrUrl(e.target.value)}
            />
          </div>
        ) : (
          <>
            <div className="input-group">
              <label>Filename</label>
              <input
                type="text"
                placeholder="snippet.py"
                value={filename}
                onChange={e => setFilename(e.target.value)}
              />
            </div>
            <div className="input-group grow">
              <label>Code</label>
              <textarea
                placeholder="Paste your code here…"
                value={code}
                onChange={e => setCode(e.target.value)}
                spellCheck={false}
              />
            </div>
          </>
        )}

        <button className="submit-btn" disabled={!canSubmit || loading} onClick={runReview}>
          {loading
            ? <><Loader2 size={14} className="spin" /> Reviewing…</>
            : <><Play size={14} /> Run Review</>}
        </button>

        {error && (
          <div className="error-box">
            <AlertCircle size={13} />
            <span>{error}</span>
          </div>
        )}

        <div className="sidebar-footer">
          Fine-tuned CodeBERT classifies each diff chunk into bug / security /
          performance / style / clean. A local Llama 3.2 model then generates
          a natural-language review comment for each flagged chunk.
        </div>
      </aside>

      <main className="main">
        {!result && !loading && (
          <div className="empty-state">
            <FileDiff size={38} strokeWidth={1.5} />
            <h2>No review yet</h2>
            <p>Paste a GitHub PR URL or some code on the left and hit Run Review.</p>
          </div>
        )}

        {loading && (
          <div className="loading-state">
            <Loader2 size={26} className="spin" />
            <p className="loading-stage">{loadingStage}</p>
            <p className="loading-hint">This can take a moment — Llama runs locally.</p>
          </div>
        )}

        {result && (
          <>
            <SummaryBar result={result} />
            {result.files.map((file, i) => <FileBlock key={i} file={file} />)}
          </>
        )}
      </main>
    </div>
  );
}