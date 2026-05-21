/* ================================================================
   markowitz.js — Client-side walk-forward Markowitz portfolio engine
   ================================================================
   Sections:
     1. Utility / math helpers
     2. Matrix helpers
     3. EWMA covariance + mean
     4. James-Stein shrinkage
     5. Markowitz QP (projected gradient ascent)
     6. Data fetching (Yahoo Finance + allorigins proxy)
     7. Walk-forward backtest
     8. Statistics
     9. Chart / heatmap rendering
    10. UI wiring (asset list, sliders, run button)
   ================================================================ */

'use strict';

/* ================================================================
   1. Utility / math helpers
   ================================================================ */

/** Sample standard deviation of an array */
function stdDev(arr) {
  const n = arr.length;
  if (n < 2) return 0;
  const mean = arr.reduce((s, x) => s + x, 0) / n;
  const variance = arr.reduce((s, x) => s + (x - mean) ** 2, 0) / (n - 1);
  return Math.sqrt(variance);
}

function rankArray(arr) {
  const sorted = [...arr].map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const ranks = new Array(arr.length);
  sorted.forEach(({ i }, rank) => { ranks[i] = rank + 1; });
  return ranks;
}

function pearsonCorr(x, y) {
  const n = x.length;
  if (n < 2) return NaN;
  const mx = x.reduce((s, v) => s + v, 0) / n;
  const my = y.reduce((s, v) => s + v, 0) / n;
  let num = 0, vx = 0, vy = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx, dy = y[i] - my;
    num += dx * dy; vx += dx * dx; vy += dy * dy;
  }
  const den = Math.sqrt(vx * vy);
  return den < 1e-12 ? NaN : num / den;
}

function spearmanCorr(x, y) {
  return pearsonCorr(rankArray(x), rankArray(y));
}

/** Clip a value to [lo, hi] */
function clip(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

/** Format a number as a percentage string, e.g. 0.123 → "12.3%" */
function fmtPct(x, decimals = 1) {
  return (x * 100).toFixed(decimals) + '%';
}

/** Format a ratio to fixed decimals */
function fmtNum(x, decimals = 2) {
  return x.toFixed(decimals);
}

/** Today's date as YYYY-MM-DD */
function todayISO() {
  return new Date().toISOString().slice(0, 10);
}

/* ================================================================
   2. Matrix helpers
   ================================================================ */

/**
 * Matrix × vector: A[N][N] × x[N] → result[N]
 */
function matVecMul(A, x) {
  const N = x.length;
  const result = new Array(N).fill(0);
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (let j = 0; j < N; j++) s += A[i][j] * x[j];
    result[i] = s;
  }
  return result;
}

/**
 * Outer product: a[N], b[N] → M[N][N]  where M[i][j] = a[i]*b[j]
 */
function outerProduct(a, b) {
  const N = a.length;
  const M = [];
  for (let i = 0; i < N; i++) {
    M[i] = new Array(N);
    for (let j = 0; j < N; j++) M[i][j] = a[i] * b[j];
  }
  return M;
}

/**
 * Max row L1 norm of a matrix (used for gradient step size)
 */
function rowNormMax(A) {
  let max = 0;
  for (let i = 0; i < A.length; i++) {
    let s = 0;
    for (let j = 0; j < A[i].length; j++) s += Math.abs(A[i][j]);
    if (s > max) max = s;
  }
  return max;
}

/**
 * Create an N×N identity-scaled matrix (diagonal eps)
 */
function diagMatrix(N, eps) {
  const M = [];
  for (let i = 0; i < N; i++) {
    M[i] = new Array(N).fill(0);
    M[i][i] = eps;
  }
  return M;
}

/* ================================================================
   3. EWMA covariance + mean
   ================================================================ */

/**
 * Compute exponential weights for T observations with given half-life.
 * w[t] ∝ λ^(T-1-t),  λ = exp(-ln2 / halflife).
 * Returns normalised weights array of length T.
 */
function ewmaWeights(T, halflifeDays) {
  const lambda = Math.exp(-Math.LN2 / halflifeDays);
  const w = new Array(T);
  for (let t = 0; t < T; t++) w[t] = Math.pow(lambda, T - 1 - t);
  const sum = w.reduce((s, x) => s + x, 0);
  return w.map(x => x / sum);
}

/**
 * EWMA covariance matrix.
 * @param {number[][]} returnRows  – shape [T][N]
 * @param {number}     halflifeDays
 * @returns {number[][]} N×N covariance matrix (annualised ×252, regularised)
 */
function ewmaCov(returnRows, halflifeDays) {
  const T = returnRows.length;
  const N = returnRows[0].length;
  const w = ewmaWeights(T, halflifeDays);

  // Weighted mean
  const mu = new Array(N).fill(0);
  for (let t = 0; t < T; t++)
    for (let j = 0; j < N; j++)
      mu[j] += w[t] * returnRows[t][j];

  // Weighted outer product of demeaned returns
  const cov = diagMatrix(N, 1e-6); // regularisation
  for (let t = 0; t < T; t++) {
    const dev = returnRows[t].map((r, j) => r - mu[j]);
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++)
        cov[i][j] += w[t] * dev[i] * dev[j];
  }

  // Annualise
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++)
      cov[i][j] *= 252;

  return cov;
}

/**
 * EWMA mean vector.
 * @param {number[][]} returnRows – shape [T][N]
 * @param {number}     halflifeDays
 * @returns {number[]} length-N annualised mean vector
 */
function ewmaMean(returnRows, halflifeDays) {
  const T = returnRows.length;
  const N = returnRows[0].length;
  const w = ewmaWeights(T, halflifeDays);

  const mu = new Array(N).fill(0);
  for (let t = 0; t < T; t++)
    for (let j = 0; j < N; j++)
      mu[j] += w[t] * returnRows[t][j];

  // Annualise (daily log-returns × 252)
  return mu.map(m => m * 252);
}

/* ================================================================
   4. James-Stein shrinkage
   ================================================================ */

/**
 * Cross-sectional shrinkage of mu toward equal-weighted mean.
 * @param {number[]} mu     – raw expected return vector
 * @param {number}   alpha  – shrinkage intensity ∈ [0,1]
 * @returns {number[]} shrunk mu
 */
function shrinkMu(mu, alpha) {
  const muBar = mu.reduce((s, x) => s + x, 0) / mu.length;
  return mu.map(m => (1 - alpha) * m + alpha * muBar);
}

/* ================================================================
   5. Markowitz QP — projected gradient ascent
   ================================================================ */

/**
 * Project v onto the probability simplex with per-asset box constraint [0, maxWt].
 * Uses binary search on the dual variable θ (Condat 2016).
 * @param {number[]} v
 * @param {number}   maxWt – max weight per asset ∈ (0,1]
 * @returns {number[]} projected weight vector summing to 1
 */
function projectSimplexBox(v, maxWt) {
  const N = v.length;
  // Find θ via bisection such that sum(clip(v_i - θ, 0, maxWt)) = 1
  let lo = Math.min(...v) - 1;
  let hi = Math.max(...v);

  const F = theta => v.reduce((s, vi) => s + clip(vi - theta, 0, maxWt), 0);

  // 100 bisection iterations
  for (let iter = 0; iter < 100; iter++) {
    const mid = (lo + hi) / 2;
    if (F(mid) > 1) lo = mid;
    else hi = mid;
  }

  const theta = (lo + hi) / 2;
  const w = v.map(vi => clip(vi - theta, 0, maxWt));

  // Renormalise to fix floating-point drift
  const total = w.reduce((s, x) => s + x, 0);
  if (total < 1e-12) {
    // Fall back to equal weight
    return new Array(N).fill(1 / N);
  }
  return w.map(x => x / total);
}

/**
 * Solve Markowitz mean-variance optimisation via projected gradient ascent.
 *
 *   max  (μ − rf)ᵀw − γ·wᵀΣw
 *   s.t. Σwᵢ = 1,  0 ≤ wᵢ ≤ maxWt
 *
 * @param {number[]}   muAnn    – annualised expected returns, length N
 * @param {number[][]} sigmaAnn – annualised covariance, N×N
 * @param {number}     gamma    – risk aversion coefficient
 * @param {number}     maxWt    – max weight per asset ∈ (0,1]
 * @param {number}     rfRate   – annual risk-free rate (default 4.5%)
 * @param {number}     maxIter  – max gradient iterations
 * @returns {number[]} optimal weight vector
 */
function solveMarkowitz(muAnn, sigmaAnn, gamma, maxWt, rfRate = 0.045, maxIter = 3000) {
  const N = muAnn.length;
  if (N === 0) return [];
  if (N === 1) return [1.0];

  // Excess returns
  const mu_ex = muAnn.map(m => m - rfRate);

  // Gradient step size: 1 / (2γ·||Σ||_∞ + ε)
  const step = 1 / (2 * gamma * rowNormMax(sigmaAnn) + 1e-8);

  // Initialise at equal weight
  let w = new Array(N).fill(1 / N);

  for (let iter = 0; iter < maxIter; iter++) {
    // Gradient of objective: (mu_ex) - 2γΣw
    const sigma_w = matVecMul(sigmaAnn, w);
    const grad = mu_ex.map((m, i) => m - 2 * gamma * sigma_w[i]);

    // Gradient ascent step
    const w_new_raw = w.map((wi, i) => wi + step * grad[i]);

    // Project onto simplex ∩ box
    const w_new = projectSimplexBox(w_new_raw, maxWt);

    // Convergence check
    let maxDelta = 0;
    for (let i = 0; i < N; i++) {
      const d = Math.abs(w_new[i] - w[i]);
      if (d > maxDelta) maxDelta = d;
    }
    w = w_new;
    if (maxDelta < 1e-9) break;
  }

  return w;
}

/* ================================================================
   6. Data fetching — two-layer cache → Cloudflare Worker proxy
   ================================================================
   Layer 1 (browser): localStorage — instant, per-user, survives refresh.
   Layer 2 (shared):  Cloudflare Worker + KV — shared across all visitors,
                      API keys live server-side only, never exposed here.

   Cache keys: `mz_price:{symbol}:{start}:{end}`
   TTL: pure-historical data (end < today) → cached 30 days in KV;
        recent data                         → 6 hours in KV.
   ================================================================ */

// After deploying the worker (`cd website/worker && wrangler deploy`)
// replace this with your worker URL, e.g.:
//   https://markowitz-data.YOUR_SUBDOMAIN.workers.dev
const WORKER_URL    = 'https://markowitz-data.theprices.workers.dev';
const FETCH_TIMEOUT = 25000;

/* ── Browser-side localStorage cache ─────────────────────────── */

const LC_PREFIX = 'mz_price:';
const LC_TTL_RECENT_MS = 6 * 60 * 60 * 1000; // 6 hours

function lcKey(symbol, start, end) {
  return `${LC_PREFIX}${symbol}:${start}:${end}`;
}

function lcGet(symbol, start, end) {
  try {
    const raw = localStorage.getItem(lcKey(symbol, start, end));
    if (!raw) return null;
    const { data, cachedAt, historical } = JSON.parse(raw);
    if (historical) return data; // never expire
    if (Date.now() - cachedAt < LC_TTL_RECENT_MS) return data;
    localStorage.removeItem(lcKey(symbol, start, end));
    return null;
  } catch (_) { return null; }
}

function lcSet(symbol, start, end, priceMap) {
  try {
    const today = new Date().toISOString().slice(0, 10);
    localStorage.setItem(lcKey(symbol, start, end), JSON.stringify({
      data:      priceMap,
      cachedAt:  Date.now(),
      historical: end < today,
    }));
  } catch (_) { /* localStorage full — silently skip */ }
}

/* ── HTTP helper ──────────────────────────────────────────────── */

async function fetchWithTimeout(url, ms = FETCH_TIMEOUT) {
  const ctrl  = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ms);
  try {
    const resp = await fetch(url, { signal: ctrl.signal });
    clearTimeout(timer);
    return resp;
  } catch (err) {
    clearTimeout(timer);
    throw err;
  }
}

/**
 * Fetch adjusted daily closes for a batch of symbols.
 * Checks localStorage first; uncached symbols go to the Worker (which
 * checks Cloudflare KV, then Alpaca if still a miss).
 *
 * @param {string[]} symbols
 * @param {string}   startDate – "YYYY-MM-DD"
 * @param {string}   endDate   – "YYYY-MM-DD"
 * @param {Function} onProgress(msg)
 * @returns {Object} { symbol: { "YYYY-MM-DD": price, … }, … }
 */
async function fetchAllPrices(symbols, startDate, endDate, onProgress) {
  const results = {};
  const toFetch = []; // symbols not in browser cache

  // ── Layer 1: browser localStorage ─────────────────────────
  for (const sym of symbols) {
    const cached = lcGet(sym, startDate, endDate);
    if (cached) {
      results[sym] = cached;
    } else {
      toFetch.push(sym);
    }
  }

  if (results && Object.keys(results).length) {
    onProgress(`${Object.keys(results).length} symbol(s) loaded from cache.`);
  }

  if (toFetch.length === 0) {
    onProgress('All data served from cache.');
    return results;
  }

  // ── Layer 2: Worker (KV + Alpaca) — batch in groups of 20 ─
  const BATCH = 20;
  let fetched = 0;

  for (let i = 0; i < toFetch.length; i += BATCH) {
    const batch  = toFetch.slice(i, i + BATCH);
    const params = new URLSearchParams({
      symbols: batch.join(','),
      start:   startDate,
      end:     endDate,
    });

    const resp = await fetchWithTimeout(`${WORKER_URL}/bars?${params}`);
    if (!resp.ok) {
      const body = await resp.text().catch(() => '');
      throw new Error(`Data worker HTTP ${resp.status}: ${body.slice(0, 120)}`);
    }

    const data = await resp.json();
    for (const sym of batch) {
      const map = data[sym] || {};
      results[sym] = map;
      lcSet(sym, startDate, endDate, map); // write to browser cache
    }

    fetched += batch.length;
    onProgress(`Fetched ${fetched}/${toFetch.length} new symbol(s)…`);
  }

  const total = symbols.length;
  const failed = symbols.filter(s => Object.keys(results[s] || {}).length === 0).length;
  if (failed === total) {
    throw new Error('All fetches failed. Check your internet connection.');
  }

  return results;
}

/* ================================================================
   7. Walk-forward backtest
   ================================================================ */

/**
 * Align price maps to a common set of dates.
 * Only keeps dates where ALL symbols have a non-null price.
 * Returns sorted array of date strings.
 */
function alignDates(priceData, symbols) {
  // Intersection of all date keys
  let dates = null;
  for (const sym of symbols) {
    const symDates = new Set(Object.keys(priceData[sym]));
    if (dates === null) {
      dates = symDates;
    } else {
      for (const d of dates) {
        if (!symDates.has(d)) dates.delete(d);
      }
    }
  }
  if (!dates || dates.size === 0) return [];
  return Array.from(dates).sort();
}

/**
 * Compute log returns from aligned price array.
 * @param {number[]} prices – length T
 * @returns {number[]} log returns of length T-1
 */
function logReturns(prices) {
  const ret = [];
  for (let t = 1; t < prices.length; t++) {
    ret.push(Math.log(prices[t] / prices[t - 1]));
  }
  return ret;
}

/**
 * Determine rebalance dates from a sorted list of trading dates.
 * Monthly: first date of each calendar month.
 * Quarterly: first date of Jan, Apr, Jul, Oct.
 */
function getRebalanceDates(dates, freq) {
  const quarterMonths = new Set([0, 3, 6, 9]); // 0-indexed
  const seen = new Set();
  const rebals = [];
  for (const d of dates) {
    const dt = new Date(d);
    const key = freq === 'quarterly'
      ? `${dt.getFullYear()}-${Math.floor(dt.getMonth() / 3)}`
      : `${dt.getFullYear()}-${dt.getMonth()}`;

    if (freq === 'quarterly' && !quarterMonths.has(dt.getMonth())) continue;

    if (!seen.has(key)) {
      seen.add(key);
      rebals.push(d);
    }
  }
  return rebals;
}

/**
 * Main walk-forward backtest.
 *
 * @param {Object} params
 * @param {string[]} params.symbols
 * @param {string}   params.startDate
 * @param {string}   params.endDate
 * @param {number}   params.gamma
 * @param {number}   params.maxWt        – fraction, e.g. 0.20
 * @param {number}   params.shrinkage    – fraction, e.g. 0.80
 * @param {string}   params.rebalFreq    – "monthly" | "quarterly"
 * @param {number}   [params.covHalflife=21]
 * @param {number}   [params.muHalflife=252]
 * @param {number}   [params.rfRate=0.045]
 * @param {Function} params.onProgress(msg)
 *
 * @returns {Object} { pvPortfolio, pvSpy, pvEW, wtsHistory, dates, symbols }
 */
async function runBacktest(params) {
  const {
    symbols,
    startDate,
    endDate,
    gamma,
    maxWt,
    shrinkage,
    rebalFreq,
    covHalflife = 21,
    muHalflife  = 252,
    rfRate      = 0.045,
    onProgress,
  } = params;

  const allSymbols = [...symbols, 'SPY'];

  // ── Step 1: Fetch all prices ─────────────────────────────────
  onProgress('Fetching price data…');
  const priceData = await fetchAllPrices(allSymbols, startDate, endDate, onProgress);

  // ── Step 2: Align dates ──────────────────────────────────────
  onProgress('Aligning dates and computing returns…');
  let validSymbols = symbols.filter(s => Object.keys(priceData[s]).length > 5);
  if (validSymbols.length < 2) {
    throw new Error('Not enough valid symbols with price data. Try different assets or a wider date range.');
  }

  // Auto-drop the most-restrictive symbol (latest start date) until we have
  // at least 60 overlapping trading days across all remaining assets.
  const droppedSymbols = [];
  let dates = alignDates(priceData, [...validSymbols, 'SPY']);
  while (dates.length < 60 && validSymbols.length > 1) {
    let worstSym = null, latestFirst = '';
    for (const s of validSymbols) {
      const first = Object.keys(priceData[s]).sort()[0] ?? '';
      if (first > latestFirst) { latestFirst = first; worstSym = s; }
    }
    if (!worstSym) break;
    droppedSymbols.push({ sym: worstSym, first: latestFirst });
    validSymbols = validSymbols.filter(s => s !== worstSym);
    dates = alignDates(priceData, [...validSymbols, 'SPY']);
  }
  if (dates.length < 60) {
    throw new Error(
      `Only ${dates.length} overlapping trading days found even after removing ` +
      `short-history assets. Please widen your date range.`
    );
  }

  // Build price matrices: priceMatrix[t][sym_idx]
  const T = dates.length;
  const N = validSymbols.length;

  const portPrices = []; // [T][N] for portfolio symbols
  const spyPrices  = []; // [T]

  for (let t = 0; t < T; t++) {
    const d = dates[t];
    portPrices.push(validSymbols.map(s => priceData[s][d]));
    spyPrices.push(priceData['SPY'][d]);
  }

  // Log returns: [T-1][N+1]
  const portRets = []; // [T-1][N]
  const spyRets  = []; // [T-1]

  for (let t = 1; t < T; t++) {
    portRets.push(validSymbols.map((_, j) => Math.log(portPrices[t][j] / portPrices[t - 1][j])));
    spyRets.push(Math.log(spyPrices[t] / spyPrices[t - 1]));
  }

  const retDates = dates.slice(1); // dates aligned with returns

  // ── Step 3: Walk-forward loop ────────────────────────────────
  onProgress('Running walk-forward optimisation…');
  const rebalDates = getRebalanceDates(retDates, rebalFreq);

  let currentWeights = new Array(N).fill(1 / N); // start equal-weight
  let ewWeights      = new Array(N).fill(1 / N);

  // Portfolio value series (starts at 1.0)
  const pvPortfolio = [1.0];
  const predMuLog   = []; // { date, tIdx, mu[], vol[] } at each rebalance
  const pvSpy       = [1.0];
  const pvEW        = [1.0];

  // Weights history: { date, weights: number[] }[]
  const wtsHistory = [];

  // Build a Set for O(1) rebalance-date lookup
  const rebalSet = new Set(rebalDates);

  // Keep a rebalance-date index for EW
  let ewRebalSet = new Set(rebalDates);

  for (let t = 0; t < retDates.length; t++) {
    const d = retDates[t];

    // ── Rebalance on this date ────────────────────────────────
    if (rebalSet.has(d)) {
      // All history up to (and including) this index
      const histRets = portRets.slice(0, t + 1);

      if (histRets.length >= Math.min(covHalflife * 3, 60)) {
        try {
          const sigmaAnn = ewmaCov(histRets, covHalflife);
          const muRaw    = ewmaMean(histRets, muHalflife);
          const muEff    = shrinkMu(muRaw, shrinkage);
          currentWeights = solveMarkowitz(muEff, sigmaAnn, gamma, maxWt, rfRate);
          predMuLog.push({
            date: d, tIdx: t,
            mu:  [...muEff],
            vol: validSymbols.map((_, j) => Math.sqrt(Math.max(0, sigmaAnn[j][j]))),
          });
        } catch (_) {
          // Keep previous weights if optimisation fails
        }
      }

      wtsHistory.push({ date: d, weights: [...currentWeights] });

      // Equal-weight: rebalance to 1/N on same schedule
      ewWeights = new Array(N).fill(1 / N);
    }

    // ── Mark-to-market ────────────────────────────────────────
    const dayRets = portRets[t];
    const spyRet  = spyRets[t];

    const portRet = currentWeights.reduce((s, w, i) => s + w * dayRets[i], 0);
    const ewRet   = ewWeights.reduce((s, w, i) => s + w * dayRets[i], 0);

    pvPortfolio.push(pvPortfolio[pvPortfolio.length - 1] * Math.exp(portRet));
    pvSpy.push(pvSpy[pvSpy.length - 1] * Math.exp(spyRet));
    pvEW.push(pvEW[pvEW.length - 1] * Math.exp(ewRet));
  }

  onProgress('Backtest complete.');

  // Compute realized forward μ and vol for each recorded rebalance period
  const realMuLog = predMuLog.map((entry, i) => {
    const startT = entry.tIdx;
    const endT   = (i + 1 < predMuLog.length) ? predMuLog[i + 1].tIdx : portRets.length;
    const nDays  = endT - startT;
    if (nDays <= 0) return { date: entry.date, mu: new Array(N).fill(NaN), vol: new Array(N).fill(NaN) };
    const mu = validSymbols.map((_, j) => {
      let cum = 0;
      for (let k = startT; k < endT; k++) cum += portRets[k][j];
      return cum * (252 / nDays);
    });
    const vol = validSymbols.map((_, j) => {
      const slice = portRets.slice(startT, endT).map(r => r[j]);
      return stdDev(slice) * Math.sqrt(252);
    });
    return { date: entry.date, mu, vol };
  });

  return {
    pvPortfolio,
    pvSpy,
    pvEW,
    wtsHistory,
    dates: [dates[0], ...retDates],
    symbols: validSymbols,
    predMuLog,
    realMuLog,
    droppedSymbols,
  };
}

/* ================================================================
   8. Statistics
   ================================================================ */

/**
 * Compute annualised performance statistics from a portfolio value series.
 * @param {number[]} pvSeries – normalised portfolio values
 * @param {number}   rfRate   – annual risk-free rate
 * @returns {{ annReturn, annVol, sharpe, maxDD }}
 */
function computeStats(pvSeries, rfRate = 0.045) {
  const nDays = pvSeries.length - 1;
  if (nDays < 2) return { annReturn: 0, annVol: 0, sharpe: 0, maxDD: 0 };

  const dailyRets = [];
  for (let t = 1; t < pvSeries.length; t++) {
    dailyRets.push(Math.log(pvSeries[t] / pvSeries[t - 1]));
  }

  const annReturn = Math.pow(pvSeries[nDays] / pvSeries[0], 252 / nDays) - 1;
  const annVol    = stdDev(dailyRets) * Math.sqrt(252);
  const sharpe    = annVol > 0 ? (annReturn - rfRate) / annVol : 0;

  // Maximum drawdown
  let maxDD   = 0;
  let running = pvSeries[0];
  for (let t = 1; t < pvSeries.length; t++) {
    if (pvSeries[t] > running) running = pvSeries[t];
    const dd = (pvSeries[t] - running) / running;
    if (dd < maxDD) maxDD = dd;
  }

  return { annReturn, annVol, sharpe, maxDD };
}

/* ================================================================
   8b. Forecast accuracy
   ================================================================ */

function computeForecastAccuracy(predMuLog, realMuLog, symbols) {
  const nPeriods = Math.min(predMuLog.length, realMuLog.length) - 1; // exclude last (no forward realized)
  if (nPeriods < 3) return null;

  const pred = predMuLog.slice(0, nPeriods);
  const real = realMuLog.slice(0, nPeriods);

  // Spearman IC per rebalance date (cross-sectional rank correlation)
  const icSeries = pred.map((p, i) => {
    const x = p.mu, y = real[i].mu;
    if (x.some(v => !isFinite(v)) || y.some(v => !isFinite(v))) return null;
    return { date: p.date, ic: spearmanCorr(x, y) };
  }).filter(Boolean);

  const meanIC = icSeries.length > 0
    ? icSeries.reduce((s, x) => s + x.ic, 0) / icSeries.length
    : NaN;

  // Per-asset accuracy stats
  const perAsset = symbols.map((sym, j) => {
    const mu_pairs  = pred.map((p, i) => [p.mu[j],  real[i].mu[j]])
                         .filter(([p, r]) => isFinite(p) && isFinite(r));
    const vol_pairs = pred.map((p, i) => [p.vol[j], real[i].vol[j]])
                         .filter(([p, r]) => isFinite(p) && isFinite(r) && r > 0);
    const n = mu_pairs.length;
    if (n === 0) return { sym, bias: NaN, mae: NaN, hitRate: NaN, r: NaN, volRatio: NaN };
    const bias    = mu_pairs.reduce((s, [p, r]) => s + (p - r), 0) / n;
    const mae     = mu_pairs.reduce((s, [p, r]) => s + Math.abs(p - r), 0) / n;
    const hitRate = mu_pairs.filter(([p, r]) => (p >= 0) === (r >= 0)).length / n;
    const r       = pearsonCorr(mu_pairs.map(x => x[0]), mu_pairs.map(x => x[1]));
    const volRatio = vol_pairs.length > 0
      ? vol_pairs.reduce((s, [p, rv]) => s + p / rv, 0) / vol_pairs.length
      : NaN;
    return { sym, bias, mae, hitRate, r, volRatio, n };
  });

  // All (pred, real) scatter points for μ plot
  const scatterPoints = symbols.flatMap((sym, j) =>
    pred.map((p, i) => ({ x: p.mu[j], y: real[i].mu[j], sym }))
        .filter(pt => isFinite(pt.x) && isFinite(pt.y))
  );

  return { icSeries, perAsset, scatterPoints, meanIC };
}

/* ================================================================
   9. Chart / heatmap rendering
   ================================================================ */

let pvChartInstance = null; // track Chart.js instance for destroy/recreate

/**
 * Render the portfolio value chart using Chart.js.
 * @param {{ labels: string[], portfolio: number[], spy: number[], ew: number[] }} chartData
 */
function renderPvChart(chartData) {
  const canvas = document.getElementById('pv-chart');
  const ctx    = canvas.getContext('2d');

  // Destroy previous instance if exists
  if (pvChartInstance) {
    pvChartInstance.destroy();
    pvChartInstance = null;
  }

  // Thin date labels (show ~8)
  const stride  = Math.max(1, Math.floor(chartData.labels.length / 8));
  const labels  = chartData.labels.map((d, i) =>
    i % stride === 0 ? d.slice(0, 7) : '' // "YYYY-MM"
  );

  pvChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Portfolio',
          data: chartData.portfolio,
          borderColor: '#2563eb',
          backgroundColor: 'rgba(37,99,235,.07)',
          fill: true,
          tension: 0.2,
          pointRadius: 0,
          borderWidth: 2,
        },
        {
          label: 'SPY',
          data: chartData.spy,
          borderColor: '#0f172a',
          fill: false,
          tension: 0.2,
          pointRadius: 0,
          borderWidth: 1.5,
        },
        {
          label: '1/N',
          data: chartData.ew,
          borderColor: '#94a3b8',
          borderDash: [5, 4],
          fill: false,
          tension: 0.2,
          pointRadius: 0,
          borderWidth: 1.5,
        },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            font: { family: 'Inter', size: 11 },
            boxWidth: 20,
            padding: 14,
          },
        },
        tooltip: {
          callbacks: {
            label: ctx =>
              ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(3)}`,
          },
        },
      },
      scales: {
        x: {
          ticks: {
            font: { family: 'Inter', size: 10 },
            maxRotation: 0,
            autoSkip: false,
          },
          grid: { color: '#f1f5f9' },
        },
        y: {
          ticks: {
            font: { family: 'Inter', size: 10 },
            callback: v => v.toFixed(2),
          },
          grid: { color: '#f1f5f9' },
        },
      },
    },
  });
}

/**
 * Map a weight fraction [0, 1] to an RGB colour using navy→white→amber diverging scale.
 * fraction = weight / maxWt   (clamped 0–1)
 */
function weightColor(fraction) {
  fraction = Math.max(0, Math.min(1, fraction));
  // Navy  #1e3a8a → White #ffffff → Amber #f59e0b
  // We split at fraction=0 (navy) → 0.5 (white) → 1 (amber)
  if (fraction < 0.5) {
    const t = fraction * 2; // 0→1
    const r = Math.round(30  + (255 - 30)  * t);
    const g = Math.round(58  + (255 - 58)  * t);
    const b = Math.round(138 + (255 - 138) * t);
    return `rgb(${r},${g},${b})`;
  } else {
    const t = (fraction - 0.5) * 2; // 0→1
    const r = Math.round(255 + (245 - 255) * t);
    const g = Math.round(255 + (158 - 255) * t);
    const b = Math.round(255 + (11  - 255) * t);
    return `rgb(${r},${g},${b})`;
  }
}

/**
 * Render the portfolio weights heatmap on a canvas element.
 * @param {HTMLCanvasElement} canvas
 * @param {{ date: string, weights: number[] }[]} wtsHistory
 * @param {string[]} symbols
 * @param {number}   maxWt – maximum weight fraction
 */
function renderWeightsHeatmap(canvas, wtsHistory, symbols, maxWt) {
  const ctx = canvas.getContext('2d');

  // Edge-case guards
  if (!wtsHistory || wtsHistory.length === 0) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#475569';
    ctx.font = '12px Inter, sans-serif';
    ctx.fillText('No rebalance data.', 10, 30);
    return;
  }

  const N_all = symbols.length;
  const T     = wtsHistory.length;

  // Sort assets by average weight descending, take top 12
  const avgWeights = symbols.map((_, j) =>
    wtsHistory.reduce((s, h) => s + (h.weights[j] || 0), 0) / T
  );
  const sortedIdx = avgWeights
    .map((w, i) => ({ w, i }))
    .sort((a, b) => b.w - a.w)
    .slice(0, 12)
    .map(x => x.i);

  const topSymbols = sortedIdx.map(i => symbols[i]);
  const N = topSymbols.length;

  if (N === 0) return;

  // Canvas sizing
  const labelW   = 50;   // left margin for ticker labels
  const bottomH  = 22;   // bottom margin for date labels
  const topPad   = 6;
  const drawW    = canvas.width - labelW;
  const drawH    = canvas.height - bottomH - topPad;
  const cellW    = drawW / T;
  const cellH    = drawH / N;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw cells
  for (let col = 0; col < T; col++) {
    for (let row = 0; row < N; row++) {
      const symIdx  = sortedIdx[row];
      const w       = wtsHistory[col].weights[symIdx] || 0;
      const frac    = maxWt > 0 ? w / maxWt : 0;

      ctx.fillStyle = weightColor(frac);
      const x = labelW + col * cellW;
      const y = topPad + row * cellH;
      ctx.fillRect(x, y, cellW, cellH);

      // Thin border
      ctx.strokeStyle = 'rgba(255,255,255,0.25)';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(x, y, cellW, cellH);
    }
  }

  // Asset ticker labels (left side)
  ctx.fillStyle = '#0f172a';
  ctx.font = '8px Inter, system-ui, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let row = 0; row < N; row++) {
    const y = topPad + row * cellH + cellH / 2;
    ctx.fillText(topSymbols[row], labelW - 3, y);
  }

  // Date labels (bottom, ~5 evenly spaced)
  const nLabels = Math.min(5, T);
  ctx.fillStyle = '#475569';
  ctx.font = '8px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let k = 0; k < nLabels; k++) {
    const col = Math.round((k / (nLabels - 1 || 1)) * (T - 1));
    const x   = labelW + col * cellW + cellW / 2;
    const d   = wtsHistory[col].date.slice(0, 7); // "YYYY-MM"
    ctx.fillText(d, x, topPad + drawH + 3);
  }

  // Update HTML legend labels below canvas
  const midEl = document.getElementById('hl-mid');
  const hiEl  = document.getElementById('hl-hi');
  if (midEl) midEl.textContent = fmtPct(maxWt / 2, 0);
  if (hiEl)  hiEl.textContent  = fmtPct(maxWt, 0);
}

/** Render current allocation as CSS progress bars */
function renderLiveWeights(wtsHistory, symbols, maxWt) {
  const panel   = document.getElementById('live-weights-body');
  const dateLbl = document.getElementById('live-weights-date');
  if (!panel || wtsHistory.length === 0) return;

  const latest = wtsHistory[wtsHistory.length - 1];
  if (dateLbl) dateLbl.textContent = 'As of ' + latest.date.slice(0, 10);

  const entries = symbols
    .map((sym, j) => ({ sym, w: latest.weights[j] || 0 }))
    .filter(e => e.w > 0.005)
    .sort((a, b) => b.w - a.w);

  panel.innerHTML = entries.map(({ sym, w }) => {
    const pct  = (w * 100).toFixed(1);
    const barW = Math.min(100, (w / maxWt) * 100).toFixed(1);
    return `<div class="lw-row">
      <span class="lw-sym">${sym}</span>
      <div class="lw-bar-wrap"><div class="lw-bar" style="width:${barW}%"></div></div>
      <span class="lw-pct">${pct}%</span>
    </div>`;
  }).join('');
}

/** Render forecast accuracy stats table */
function renderAccuracyTable(perAsset) {
  const tbody = document.getElementById('acc-tbody');
  if (!tbody) return;
  tbody.innerHTML = perAsset.map(({ sym, bias, mae, hitRate, r, volRatio }) => {
    const biasCls = isFinite(bias) ? (bias > 0 ? 'acc-pos' : 'acc-neg') : '';
    const rCls    = isFinite(r)    ? (r    > 0 ? 'acc-pos' : 'acc-neg') : '';
    return `<tr>
      <td class="acc-sym">${sym}</td>
      <td class="${biasCls}">${isFinite(bias) ? fmtPct(bias) : '—'}</td>
      <td>${isFinite(mae) ? fmtPct(mae) : '—'}</td>
      <td>${isFinite(hitRate) ? fmtPct(hitRate, 0) : '—'}</td>
      <td class="${rCls}">${isFinite(r) ? r.toFixed(3) : '—'}</td>
      <td>${isFinite(volRatio) ? volRatio.toFixed(2) + '×' : '—'}</td>
    </tr>`;
  }).join('');
}

let icChartInstance  = null;
let muScatterInstance = null;

/** IC over time bar chart */
function renderIcChart(icSeries, meanIC) {
  const canvas = document.getElementById('ic-chart');
  if (!canvas) return;
  if (icChartInstance) { icChartInstance.destroy(); icChartInstance = null; }

  icChartInstance = new Chart(canvas.getContext('2d'), {
    data: {
      labels: icSeries.map(x => x.date.slice(0, 7)),
      datasets: [
        {
          type: 'line',
          label: `Mean IC ${isFinite(meanIC) ? meanIC.toFixed(3) : ''}`,
          data: icSeries.map(() => meanIC),
          borderColor: '#f59e0b',
          borderDash: [5, 4],
          borderWidth: 1.5,
          pointRadius: 0,
          order: 0,
        },
        {
          type: 'bar',
          label: 'IC (Spearman)',
          data: icSeries.map(x => x.ic),
          backgroundColor: icSeries.map(x => x.ic >= 0 ? 'rgba(37,99,235,0.75)' : 'rgba(220,38,38,0.75)'),
          borderWidth: 0,
          order: 1,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'top', labels: { font: { size: 9 }, boxWidth: 12, padding: 8 } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(3)}` } },
      },
      scales: {
        x: { ticks: { font: { size: 9 }, maxRotation: 45, autoSkip: true, maxTicksLimit: 8 }, grid: { color: '#f1f5f9' } },
        y: { ticks: { font: { size: 9 } }, grid: { color: '#f1f5f9' },
             title: { display: true, text: 'Spearman IC', font: { size: 9 } } },
      },
    },
  });
}

/** Predicted vs realized μ scatter */
function renderMuScatter(scatterPoints) {
  const canvas = document.getElementById('mu-scatter');
  if (!canvas) return;
  if (muScatterInstance) { muScatterInstance.destroy(); muScatterInstance = null; }

  // x = realized μ (wide spread), y = predicted μ (narrow due to shrinkage)
  const xVals = scatterPoints.map(p => p.y);
  const yVals = scatterPoints.map(p => p.x);
  const xPad  = (Math.max(...xVals) - Math.min(...xVals)) * 0.1 || 0.5;
  const yPad  = (Math.max(...yVals) - Math.min(...yVals)) * 0.1 || 0.5;
  const xMin  = Math.min(...xVals) - xPad, xMax = Math.max(...xVals) + xPad;
  const yMin  = Math.min(...yVals) - yPad, yMax = Math.max(...yVals) + yPad;

  muScatterInstance = new Chart(canvas.getContext('2d'), {
    data: {
      datasets: [
        {
          type: 'line',
          label: 'Perfect forecast',
          data: [{ x: xMin, y: xMin }, { x: xMax, y: xMax }],
          borderColor: '#f59e0b',
          borderDash: [4, 3],
          borderWidth: 1.2,
          pointRadius: 0,
          order: 0,
        },
        {
          type: 'scatter',
          label: `${scatterPoints.length} obs (hover for symbol)`,
          data: scatterPoints.map(p => ({ x: p.y, y: p.x })),
          backgroundColor: 'rgba(37,99,235,0.18)',
          borderColor: 'rgba(37,99,235,0.45)',
          borderWidth: 0.5,
          pointRadius: 3,
          pointHoverRadius: 5,
          order: 1,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'top', labels: { font: { size: 9 }, boxWidth: 12, padding: 8 } },
        tooltip: {
          callbacks: {
            label: ctx => {
              if (ctx.datasetIndex !== 1) return null;
              const pt = scatterPoints[ctx.dataIndex];
              if (!pt) return null;
              return ` ${pt.sym}: real ${ctx.parsed.x.toFixed(2)}, pred ${ctx.parsed.y.toFixed(2)}`;
            },
          },
        },
      },
      scales: {
        x: {
          min: xMin, max: xMax,
          ticks: { font: { size: 9 }, callback: v => v.toFixed(1) },
          title: { display: true, text: 'Realized μ (annualised)', font: { size: 9 } },
          grid: { color: '#f1f5f9' },
        },
        y: {
          min: yMin, max: yMax,
          ticks: { font: { size: 9 }, callback: v => v.toFixed(2) },
          title: { display: true, text: 'Predicted μ (annualised)', font: { size: 9 } },
          grid: { color: '#f1f5f9' },
        },
      },
    },
  });
}

/* ================================================================
   10. UI wiring
   ================================================================ */

let stocksData = []; // loaded from stocks.json

/** Set status text with optional class */
function setStatus(msg, cls = '') {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = cls;
}

/** Update the selection count badge */
function updateSelectionCount() {
  const list  = document.getElementById('asset-list');
  const count = list ? Array.from(list.options).filter(o => o.selected).length : 0;
  const el    = document.getElementById('selection-count');
  if (el) el.textContent = `${count} asset${count !== 1 ? 's' : ''} selected`;
}

/** Populate #asset-list from stocksData, filtered by query */
function populateAssetList(query = '') {
  const list = document.getElementById('asset-list');
  if (!list) return;

  // Remember currently selected tickers
  const selected = new Set(
    Array.from(list.options).filter(o => o.selected).map(o => o.value)
  );

  // Clear
  list.innerHTML = '';

  const q = query.toLowerCase().trim();
  const filtered = q
    ? stocksData.filter(s =>
        s.ticker.toLowerCase().includes(q) ||
        s.name.toLowerCase().includes(q) ||
        s.sector.toLowerCase().includes(q)
      )
    : stocksData;

  // Group by sector for readability
  const bySector = {};
  for (const s of filtered) {
    (bySector[s.sector] = bySector[s.sector] || []).push(s);
  }

  for (const sector of Object.keys(bySector).sort()) {
    const group = document.createElement('optgroup');
    group.label = sector;
    for (const s of bySector[sector]) {
      const opt = document.createElement('option');
      opt.value = s.ticker;
      opt.textContent = `${s.ticker} — ${s.name}`;
      opt.selected = selected.has(s.ticker);
      group.appendChild(opt);
    }
    list.appendChild(group);
  }

  updateSelectionCount();
}

/** Load stocks.json, populate list, wire all controls */
async function initUI() {
  // Set default end date to today
  const endInput = document.getElementById('end-date');
  if (endInput && !endInput.value) endInput.value = todayISO();

  // Load stocks.json
  try {
    const resp = await fetch('data/stocks.json');
    stocksData = await resp.json();
  } catch (e) {
    setStatus('Warning: could not load stocks.json. Enter tickers manually.', 'error');
    stocksData = [];
  }

  populateAssetList();

  // ── Asset search ─────────────────────────────────────────────
  document.getElementById('asset-search')
    ?.addEventListener('input', e => populateAssetList(e.target.value));

  // ── Select All / Clear ───────────────────────────────────────
  document.getElementById('select-all-btn')?.addEventListener('click', () => {
    const list = document.getElementById('asset-list');
    Array.from(list.options).forEach(o => { o.selected = true; });
    updateSelectionCount();
  });

  document.getElementById('clear-btn')?.addEventListener('click', () => {
    const list = document.getElementById('asset-list');
    Array.from(list.options).forEach(o => { o.selected = false; });
    updateSelectionCount();
  });

  document.getElementById('asset-list')
    ?.addEventListener('change', updateSelectionCount);

  // ── Slider live values ───────────────────────────────────────
  const gammaSlider    = document.getElementById('gamma-slider');
  const gammaVal       = document.getElementById('gamma-val');
  const maxWtSlider    = document.getElementById('max-wt-slider');
  const maxWtVal       = document.getElementById('max-wt-val');
  const shrinkSlider   = document.getElementById('shrinkage-slider');
  const shrinkVal      = document.getElementById('shrinkage-val');

  gammaSlider?.addEventListener('input', () => {
    gammaVal.textContent = gammaSlider.value;
  });
  maxWtSlider?.addEventListener('input', () => {
    maxWtVal.textContent = maxWtSlider.value + '%';
  });
  shrinkSlider?.addEventListener('input', () => {
    shrinkVal.textContent = shrinkSlider.value + '%';
  });

  // ── Run button ───────────────────────────────────────────────
  document.getElementById('run-btn')?.addEventListener('click', runBacktestUI);
}

/** Collect selected symbols from the multi-select */
function getSelectedSymbols() {
  const list = document.getElementById('asset-list');
  return Array.from(list.options)
    .filter(o => o.selected)
    .map(o => o.value);
}

/** Update a stat card */
function setStat(id, value, applyColor = false) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = value;
  if (applyColor) {
    const num = parseFloat(value);
    el.classList.remove('positive', 'negative');
    if (!isNaN(num)) {
      el.classList.add(num >= 0 ? 'positive' : 'negative');
    }
  }
}

/** Main handler for the Run button */
async function runBacktestUI() {
  const runBtn = document.getElementById('run-btn');
  const symbols = getSelectedSymbols();

  if (symbols.length < 2) {
    setStatus('Please select at least 2 assets.', 'error');
    return;
  }

  // Collect parameters
  const gamma     = parseFloat(document.getElementById('gamma-slider').value);
  const maxWtPct  = parseFloat(document.getElementById('max-wt-slider').value);
  const shrinkPct = parseFloat(document.getElementById('shrinkage-slider').value);
  const rebalFreq = document.getElementById('rebal-freq').value;
  const startDate = document.getElementById('start-date').value;
  const endDate   = document.getElementById('end-date').value;

  if (!startDate || !endDate || startDate >= endDate) {
    setStatus('Invalid date range. Start must be before End.', 'error');
    return;
  }

  const maxWt     = maxWtPct  / 100;
  const shrinkage = shrinkPct / 100;

  // Disable button while running
  runBtn.disabled = true;
  setStatus('Starting backtest…', '');

  // Hide previous results
  const resultsSection = document.getElementById('results-section');
  resultsSection.classList.remove('visible');

  try {
    const result = await runBacktest({
      symbols,
      startDate,
      endDate,
      gamma,
      maxWt,
      shrinkage,
      rebalFreq,
      onProgress: msg => setStatus(msg, ''),
    });

    const { pvPortfolio, pvSpy, pvEW, wtsHistory, dates, symbols: validSymbols,
            predMuLog, realMuLog, droppedSymbols } = result;

    // Show warning banner if any assets were auto-dropped due to short history
    const warnEl = document.getElementById('dropped-assets-warning');
    if (warnEl) {
      if (droppedSymbols && droppedSymbols.length > 0) {
        const list = droppedSymbols.map(d => `${d.sym} (data starts ${d.first})`).join(', ');
        warnEl.textContent = `⚠️ Auto-removed ${droppedSymbols.length} asset(s) with insufficient history for this date range: ${list}`;
        warnEl.style.display = 'block';
      } else {
        warnEl.style.display = 'none';
      }
    }

    // ── Compute stats ──────────────────────────────────────────
    const statsPort = computeStats(pvPortfolio);
    const statsSpy  = computeStats(pvSpy);
    const statsEW   = computeStats(pvEW);

    // Update stat cards (portfolio stats)
    setStat('stat-ret',    fmtPct(statsPort.annReturn), true);
    setStat('stat-sharpe', fmtNum(statsPort.sharpe),    true);
    setStat('stat-dd',     fmtPct(statsPort.maxDD),     true);
    setStat('stat-vol',    fmtPct(statsPort.annVol),    false);

    // ── Live weights panel ─────────────────────────────────────
    renderLiveWeights(wtsHistory, validSymbols, maxWt);

    // ── Render pv chart ────────────────────────────────────────
    renderPvChart({
      labels:    dates,
      portfolio: pvPortfolio,
      spy:       pvSpy,
      ew:        pvEW,
    });

    // ── Render heatmap ─────────────────────────────────────────
    const heatCanvas = document.getElementById('wts-canvas');
    renderWeightsHeatmap(heatCanvas, wtsHistory, validSymbols, maxWt);

    // ── Forecast accuracy ──────────────────────────────────────
    const accSection = document.getElementById('acc-section');
    const accuracy = computeForecastAccuracy(predMuLog, realMuLog, validSymbols);
    if (accuracy && accSection) {
      renderAccuracyTable(accuracy.perAsset);
      renderIcChart(accuracy.icSeries, accuracy.meanIC);
      renderMuScatter(accuracy.scatterPoints);
      accSection.style.display = 'flex';
    }

    // ── Show results ───────────────────────────────────────────
    resultsSection.classList.add('visible');

    const spySharpe = fmtNum(statsSpy.sharpe);
    const ewSharpe  = fmtNum(statsEW.sharpe);
    setStatus(
      `Done. Sharpe: Portfolio ${fmtNum(statsPort.sharpe)} | SPY ${spySharpe} | 1/N ${ewSharpe}`,
      'success'
    );

  } catch (err) {
    setStatus('Error: ' + err.message, 'error');
  } finally {
    runBtn.disabled = false;
  }
}

/* ── Boot ──────────────────────────────────────────────────────── */
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initUI);
} else {
  initUI();
}
