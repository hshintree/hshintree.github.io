/**
 * Cloudflare Worker — Markowitz Engine data proxy
 *
 * Responsibilities:
 *  1. Hold Alpaca API keys server-side (never exposed to browser)
 *  2. Serve as CORS-enabled data endpoint for the GitHub Pages frontend
 *  3. Cache responses in Cloudflare KV so repeated symbol/date combos
 *     are served instantly to ALL users without hitting Alpaca again
 *
 * Secrets (set via `wrangler secret put`):
 *   ALPACA_KEY    — APCA-API-KEY-ID
 *   ALPACA_SECRET — APCA-API-SECRET-KEY
 *
 * KV namespace binding: PRICE_CACHE
 *
 * Endpoint:
 *   GET /bars?symbols=AAPL,MSFT,SPY&start=2022-01-03&end=2025-05-19
 *   → { "AAPL": {"2022-01-03": 182.01, ...}, "MSFT": {...}, ... }
 */

const ALPACA_BARS = 'https://data.alpaca.markets/v2/stocks/bars';

// 30 days for purely historical data; 6 hours for data touching today
const TTL_HISTORICAL_S = 60 * 60 * 24 * 30;
const TTL_RECENT_S     = 60 * 60 * 6;

const CORS = {
  'Access-Control-Allow-Origin':  '*',
  'Access-Control-Allow-Methods': 'GET, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

function jsonResp(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...CORS, 'Content-Type': 'application/json' },
  });
}

async function fetchSymbolFromAlpaca(symbol, start, end, apiKey, apiSecret) {
  const map = {};
  let pageToken = null;

  do {
    const p = new URLSearchParams({
      symbols:    symbol,
      start,
      end,
      timeframe:  '1Day',
      adjustment: 'all',
      feed:       'iex',
      limit:      '10000',
    });
    if (pageToken) p.set('page_token', pageToken);

    const resp = await fetch(`${ALPACA_BARS}?${p}`, {
      headers: {
        'APCA-API-KEY-ID':     apiKey,
        'APCA-API-SECRET-KEY': apiSecret,
      },
    });

    if (!resp.ok) {
      const body = await resp.text().catch(() => '');
      throw new Error(`Alpaca HTTP ${resp.status} for ${symbol}: ${body.slice(0, 200)}`);
    }

    const data = await resp.json();
    const bars = data.bars?.[symbol] || [];
    for (const bar of bars) {
      map[bar.t.slice(0, 10)] = bar.c; // date → adjusted close
    }
    pageToken = data.next_page_token || null;
  } while (pageToken);

  return map;
}

export default {
  async fetch(request, env) {
    // Preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: CORS });
    }

    const url = new URL(request.url);
    if (url.pathname !== '/bars') {
      return jsonResp({ error: 'Not found. Use /bars' }, 404);
    }

    const symbolsParam = url.searchParams.get('symbols') || '';
    const start        = url.searchParams.get('start');
    const end          = url.searchParams.get('end');

    const symbols = symbolsParam.split(',').map(s => s.trim().toUpperCase()).filter(Boolean);

    if (!symbols.length || !start || !end) {
      return jsonResp({ error: 'Required params: symbols, start, end' }, 400);
    }

    const today       = new Date().toISOString().slice(0, 10);
    const isHistorical = end < today;
    const ttl          = isHistorical ? TTL_HISTORICAL_S : TTL_RECENT_S;

    // Process all symbols in parallel
    const results = {};
    await Promise.all(symbols.map(async (sym) => {
      const cacheKey = `bars:${sym}:${start}:${end}`;

      // ── Layer 1: KV cache ────────────────────────────────────
      try {
        const cached = await env.PRICE_CACHE.get(cacheKey, 'json');
        if (cached) {
          results[sym] = cached;
          return;
        }
      } catch (_) {
        // KV miss or error — fall through to Alpaca
      }

      // ── Layer 2: Alpaca API ──────────────────────────────────
      try {
        const priceMap = await fetchSymbolFromAlpaca(
          sym, start, end, env.ALPACA_KEY, env.ALPACA_SECRET
        );
        results[sym] = priceMap;

        // Write to KV (fire-and-forget, don't block response)
        env.PRICE_CACHE.put(cacheKey, JSON.stringify(priceMap), {
          expirationTtl: ttl,
        }).catch(() => {});
      } catch (err) {
        // Return empty map for failed symbols; caller skips them gracefully
        results[sym] = {};
      }
    }));

    return jsonResp(results);
  },
};
