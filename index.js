require('dotenv').config();

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const cheerio = require('cheerio');
const { URL } = require('url');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// ==================== CONFIGURACIÓN ====================
const CONFIG = {
    SELF_PING_URL: process.env.SELF_PING_URL || '',
    VAR_URL: process.env.VAR_URL || '',
    SELF_PING_INTERVAL: 14 * 60 * 1000,
    WORKERS_COUNT: 12,
    REQUEST_TIMEOUT: 12000,
    USER_AGENTS: [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
    ]
};

function randomAgent() {
    return CONFIG.USER_AGENTS[Math.floor(Math.random() * CONFIG.USER_AGENTS.length)];
}

// ==================== CORS ====================
const corsOptions = {
    origin: function (origin, callback) {
        const allowedOrigin = CONFIG.VAR_URL ? CONFIG.VAR_URL.replace(/\/$/, '') : null;

        if (!origin) return callback(null, true);

        if (!allowedOrigin) {
            console.warn('⚠️  CORS bloqueado: VAR_URL no configurado');
            return callback(new Error('CORS: VAR_URL no configurado'), false);
        }

        const selfOrigin = CONFIG.SELF_PING_URL ? CONFIG.SELF_PING_URL.replace(/\/$/, '') : null;
        if (origin === allowedOrigin || origin === selfOrigin) {
            return callback(null, true);
        }

        console.warn(`⚠️  CORS bloqueado para origin: ${origin}`);
        return callback(new Error(`CORS: Origen no permitido: ${origin}`), false);
    },
    methods: ['GET', 'POST', 'PUT', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true
};

app.use(cors(corsOptions));
app.options('*', cors(corsOptions));
app.use(express.json());
app.use(express.static('public'));

app.use((req, res, next) => {
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
    next();
});

// ==================== FILTROS DE SITIO ====================
const SITE_FILTERS = {
    youtube:       ['youtube.com', 'youtu.be'],
    mediafire:     ['mediafire.com'],
    google:        ['drive.google.com', 'docs.google.com'],
    github:        ['github.com', 'gist.github.com'],
    reddit:        ['reddit.com'],
    stackoverflow: ['stackoverflow.com', 'stackexchange.com'],
    medium:        ['medium.com', 'towardsdatascience.com'],
    all:           null
};

const SPAM_KEYWORDS = [
    'spam', 'scam', 'fake', 'virus', 'malware', 'phishing',
    'click here now', 'free money', 'get rich', 'weight loss',
    'viagra', 'casino', 'porn', 'xxx'
];

// ==================== WORKER 1: DuckDuckGo HTML ====================
async function searchDuckDuckGo(keyword, page = 1) {
    const results = [];
    try {
        const offset = (page - 1) * 10;
        const response = await axios.get('https://html.duckduckgo.com/html/', {
            params: { q: keyword, s: offset, dc: offset + 1 },
            headers: {
                'User-Agent': randomAgent(),
                'Accept': 'text/html',
                'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
                'Referer': 'https://duckduckgo.com/'
            },
            timeout: CONFIG.REQUEST_TIMEOUT
        });

        const $ = cheerio.load(response.data);

        $('.result__body').each((i, el) => {
            const title = $(el).find('.result__title').text().trim();
            let link    = $(el).find('.result__url').text().trim();
            const desc  = $(el).find('.result__snippet').text().trim();

            if (link && !link.startsWith('http')) link = 'https://' + link;

            if (title && link) {
                try {
                    new URL(link);
                    results.push({ title, url: link, description: desc || 'Sin descripción', source: extractSourceName(link), engine: 'DuckDuckGo' });
                } catch (e) { /* URL inválida */ }
            }
        });
    } catch (error) {
        console.error('[DuckDuckGo] Error:', error.message);
    }
    return results;
}

// ==================== WORKER 2: Bing ====================
async function searchBing(keyword, offset = 0) {
    const results = [];
    try {
        const response = await axios.get('https://www.bing.com/search', {
            params: { q: keyword, first: offset + 1, count: 20 },
            headers: {
                'User-Agent': randomAgent(),
                'Accept': 'text/html,application/xhtml+xml',
                'Accept-Language': 'es-ES,es;q=0.9',
                'Cache-Control': 'no-cache'
            },
            timeout: CONFIG.REQUEST_TIMEOUT
        });

        const $ = cheerio.load(response.data);

        $('.b_algo').each((i, el) => {
            const title = $(el).find('h2').text().trim();
            const link  = $(el).find('h2 a').attr('href');
            const desc  = $(el).find('.b_caption p, .b_algoSlug').first().text().trim();

            if (title && link && link.startsWith('http')) {
                try {
                    new URL(link);
                    results.push({ title, url: link, description: desc || 'Sin descripción', source: extractSourceName(link), engine: 'Bing' });
                } catch (e) { /* URL inválida */ }
            }
        });
    } catch (error) {
        console.error('[Bing] Error:', error.message);
    }
    return results;
}

// ==================== WORKER 3: Wikipedia API ====================
async function searchWikipedia(keyword) {
    const results = [];
    try {
        for (const lang of ['es', 'en']) {
            const response = await axios.get(`https://${lang}.wikipedia.org/w/api.php`, {
                params: { action: 'query', list: 'search', srsearch: keyword, srlimit: 8, format: 'json', srprop: 'snippet' },
                headers: { 'User-Agent': randomAgent() },
                timeout: CONFIG.REQUEST_TIMEOUT
            });

            (response.data?.query?.search || []).forEach(item => {
                const clean = item.snippet.replace(/<[^>]+>/g, '').replace(/&quot;/g, '"').replace(/&#039;/g, "'");
                results.push({
                    title: item.title,
                    url: `https://${lang}.wikipedia.org/wiki/${encodeURIComponent(item.title.replace(/ /g, '_'))}`,
                    description: clean || 'Artículo de Wikipedia',
                    source: `Wikipedia (${lang.toUpperCase()})`,
                    engine: 'Wikipedia'
                });
            });
        }
    } catch (error) {
        console.error('[Wikipedia] Error:', error.message);
    }
    return results;
}

// ==================== WORKER 4: YouTube ====================
async function searchYouTube(keyword) {
    const results = [];
    try {
        const response = await axios.get(`https://www.youtube.com/results?search_query=${encodeURIComponent(keyword)}&hl=es`, {
            headers: { 'User-Agent': randomAgent(), 'Accept-Language': 'es-ES,es;q=0.9' },
            timeout: CONFIG.REQUEST_TIMEOUT
        });

        const $ = cheerio.load(response.data);
        for (const script of $('script').toArray()) {
            const content = $(script).html() || '';
            if (content.includes('ytInitialData')) {
                try {
                    const match = content.match(/ytInitialData\s*=\s*(\{.+?\});\s*(?:var |window\.|<\/script)/s);
                    if (match) {
                        const data = JSON.parse(match[1]);
                        const sections = data?.contents?.twoColumnSearchResultsRenderer?.primaryContents?.sectionListRenderer?.contents || [];
                        sections.forEach(section => {
                            (section?.itemSectionRenderer?.contents || []).forEach(item => {
                                const video = item?.videoRenderer;
                                if (video?.videoId) {
                                    const ch = video.ownerText?.runs?.[0]?.text || '';
                                    const views = video.viewCountText?.simpleText || '';
                                    results.push({
                                        title: video.title?.runs?.[0]?.text || 'Sin título',
                                        url: `https://www.youtube.com/watch?v=${video.videoId}`,
                                        description: video.descriptionSnippet?.runs?.map(r => r.text).join('') || `Canal: ${ch} | ${views}`,
                                        source: 'YouTube',
                                        engine: 'YouTube'
                                    });
                                }
                            });
                        });
                    }
                } catch (e) { console.error('[YouTube] Parse error:', e.message); }
                break;
            }
        }
    } catch (error) {
        console.error('[YouTube] Error:', error.message);
    }
    return results;
}

// ==================== WORKER 5: GitHub API ====================
async function searchGitHub(keyword) {
    const results = [];
    try {
        const headers = { 'Accept': 'application/vnd.github.v3+json', 'User-Agent': randomAgent() };
        if (process.env.GITHUB_TOKEN) headers['Authorization'] = `token ${process.env.GITHUB_TOKEN}`;

        const [reposRes, issuesRes] = await Promise.allSettled([
            axios.get('https://api.github.com/search/repositories', { params: { q: keyword, sort: 'stars', order: 'desc', per_page: 15 }, headers, timeout: CONFIG.REQUEST_TIMEOUT }),
            axios.get('https://api.github.com/search/issues', { params: { q: `${keyword} type:issue`, sort: 'reactions', per_page: 8 }, headers, timeout: CONFIG.REQUEST_TIMEOUT })
        ]);

        if (reposRes.status === 'fulfilled') {
            reposRes.value.data.items?.forEach(repo => {
                results.push({
                    title: `${repo.full_name}`,
                    url: repo.html_url,
                    description: `⭐ ${repo.stargazers_count} stars | ${repo.language || 'Sin lenguaje'} | ${repo.description || 'Sin descripción'}`,
                    source: 'GitHub', engine: 'GitHub'
                });
            });
        }
        if (issuesRes.status === 'fulfilled') {
            issuesRes.value.data.items?.forEach(issue => {
                results.push({
                    title: `[Issue] ${issue.title}`,
                    url: issue.html_url,
                    description: issue.body ? issue.body.substring(0, 200) : 'Ver issue completo',
                    source: 'GitHub', engine: 'GitHub'
                });
            });
        }
    } catch (error) {
        console.error('[GitHub] Error:', error.message);
    }
    return results;
}

// ==================== WORKER 6: Reddit API ====================
async function searchReddit(keyword) {
    const results = [];
    try {
        const response = await axios.get('https://www.reddit.com/search.json', {
            params: { q: keyword, limit: 25, sort: 'relevance', type: 'link' },
            headers: { 'User-Agent': 'SearchBot/2.0 (compatible)' },
            timeout: CONFIG.REQUEST_TIMEOUT
        });

        response.data?.data?.children?.forEach(post => {
            const d = post.data;
            results.push({
                title: d.title,
                url: `https://www.reddit.com${d.permalink}`,
                description: `r/${d.subreddit} | 👍 ${d.score} | 💬 ${d.num_comments} comentarios${d.selftext ? ' | ' + d.selftext.substring(0, 150) : ''}`,
                source: 'Reddit', engine: 'Reddit'
            });
        });
    } catch (error) {
        console.error('[Reddit] Error:', error.message);
    }
    return results;
}

// ==================== WORKER 7: Stack Overflow API ====================
async function searchStackOverflow(keyword) {
    const results = [];
    try {
        const response = await axios.get('https://api.stackexchange.com/2.3/search/advanced', {
            params: { q: keyword, site: 'stackoverflow', sort: 'relevance', order: 'desc', pagesize: 15 },
            headers: { 'User-Agent': randomAgent() },
            timeout: CONFIG.REQUEST_TIMEOUT
        });

        response.data?.items?.forEach(item => {
            results.push({
                title: item.title,
                url: item.link,
                description: `✅ ${item.answer_count} respuestas | 👁 ${item.view_count} vistas | Score: ${item.score}`,
                source: 'Stack Overflow', engine: 'StackOverflow'
            });
        });
    } catch (error) {
        console.error('[StackOverflow] Error:', error.message);
    }
    return results;
}

// ==================== WORKER 8: DuckDuckGo variante ====================
async function searchDuckDuckGoVariant(keyword) {
    const variants = [`${keyword} tutorial`, `${keyword} guia`, `"${keyword}"`, `${keyword} ejemplo`];
    return searchDuckDuckGo(variants[Math.floor(Math.random() * variants.length)], 2);
}

// ==================== WORKER 9: Bing página 2 ====================
async function searchBingVariant(keyword) {
    return searchBing(keyword, 10);
}

// ==================== WORKER 10: Open Library ====================
async function searchOpenLibrary(keyword) {
    const results = [];
    try {
        const response = await axios.get('https://openlibrary.org/search.json', {
            params: { q: keyword, limit: 10, fields: 'title,author_name,first_publish_year,key' },
            headers: { 'User-Agent': randomAgent() },
            timeout: CONFIG.REQUEST_TIMEOUT
        });

        response.data?.docs?.forEach(book => {
            if (!book.title) return;
            results.push({
                title: book.title,
                url: `https://openlibrary.org${book.key}`,
                description: `📚 Libro | Autor: ${(book.author_name || ['Desconocido']).join(', ')} | Publicado: ${book.first_publish_year || '?'}`,
                source: 'Open Library', engine: 'OpenLibrary'
            });
        });
    } catch (error) {
        console.error('[OpenLibrary] Error:', error.message);
    }
    return results;
}

// ==================== WORKER 11: Hacker News (Algolia) ====================
async function searchHackerNews(keyword) {
    const results = [];
    try {
        const response = await axios.get('https://hn.algolia.com/api/v1/search', {
            params: { query: keyword, hitsPerPage: 15, tags: 'story' },
            headers: { 'User-Agent': randomAgent() },
            timeout: CONFIG.REQUEST_TIMEOUT
        });

        response.data?.hits?.forEach(item => {
            if (!item.title) return;
            results.push({
                title: item.title,
                url: item.url || `https://news.ycombinator.com/item?id=${item.objectID}`,
                description: `🔥 HN | Puntos: ${item.points || 0} | 💬 ${item.num_comments || 0} comentarios | Autor: ${item.author}`,
                source: 'Hacker News', engine: 'HackerNews'
            });
        });
    } catch (error) {
        console.error('[HackerNews] Error:', error.message);
    }
    return results;
}

// ==================== WORKER 12: Archive.org ====================
async function searchArchive(keyword) {
    const results = [];
    try {
        const response = await axios.get('https://archive.org/advancedsearch.php', {
            params: { q: keyword, fl: 'identifier,title,description,mediatype', rows: 12, output: 'json', sort: 'downloads desc' },
            headers: { 'User-Agent': randomAgent() },
            timeout: CONFIG.REQUEST_TIMEOUT
        });

        response.data?.response?.docs?.forEach(item => {
            if (!item.title) return;
            const title = Array.isArray(item.title) ? item.title[0] : item.title;
            const desc  = Array.isArray(item.description) ? item.description[0] : item.description;
            results.push({
                title,
                url: `https://archive.org/details/${item.identifier}`,
                description: `🗄️ Archive.org | Tipo: ${item.mediatype || 'Archivo'} | ${desc ? String(desc).substring(0, 150) : 'Ver en Archive.org'}`,
                source: 'Archive.org', engine: 'Archive'
            });
        });
    } catch (error) {
        console.error('[Archive.org] Error:', error.message);
    }
    return results;
}

// ==================== MAPA DE WORKERS ====================

const WORKER_MAP = {
    1:  (kw) => searchDuckDuckGo(kw, 1),
    2:  (kw) => searchBing(kw, 0),
    3:  (kw) => searchWikipedia(kw),
    4:  (kw) => searchYouTube(kw),
    5:  (kw) => searchGitHub(kw),
    6:  (kw) => searchReddit(kw),
    7:  (kw) => searchStackOverflow(kw),
    8:  (kw) => searchDuckDuckGoVariant(kw),
    9:  (kw) => searchBingVariant(kw),
    10: (kw) => searchOpenLibrary(kw),
    11: (kw) => searchHackerNews(kw),
    12: (kw) => searchArchive(kw)
};

const FILTER_WORKERS = {
    youtube:       { 4: (kw) => searchYouTube(kw), 1: (kw) => searchDuckDuckGo(`site:youtube.com ${kw}`), 2: (kw) => searchBing(`site:youtube.com ${kw}`) },
    github:        { 5: (kw) => searchGitHub(kw), 1: (kw) => searchDuckDuckGo(`site:github.com ${kw}`), 2: (kw) => searchBing(`site:github.com ${kw}`) },
    reddit:        { 6: (kw) => searchReddit(kw), 1: (kw) => searchDuckDuckGo(`site:reddit.com ${kw}`), 2: (kw) => searchBing(`site:reddit.com ${kw}`) },
    stackoverflow: { 7: (kw) => searchStackOverflow(kw), 1: (kw) => searchDuckDuckGo(`site:stackoverflow.com ${kw}`), 2: (kw) => searchBing(`site:stackoverflow.com ${kw}`) },
    mediafire:     { 1: (kw) => searchDuckDuckGo(`site:mediafire.com ${kw}`), 2: (kw) => searchBing(`site:mediafire.com ${kw}`), 8: (kw) => searchDuckDuckGo(`mediafire ${kw} download`) },
    google:        { 1: (kw) => searchDuckDuckGo(`site:drive.google.com ${kw}`), 2: (kw) => searchBing(`site:drive.google.com OR site:docs.google.com ${kw}`) },
    medium:        { 1: (kw) => searchDuckDuckGo(`site:medium.com ${kw}`), 2: (kw) => searchBing(`site:medium.com ${kw}`), 8: (kw) => searchDuckDuckGo(`medium.com ${kw} article`) }
};

async function workerSearch(workerId, keyword, filter) {
    const fn = (filter !== 'all' && FILTER_WORKERS[filter]?.[workerId])
        ? FILTER_WORKERS[filter][workerId]
        : WORKER_MAP[workerId];

    if (!fn) return [];

    console.log(`[Worker ${workerId}] Iniciando`);
    try {
        const results = await fn(keyword);
        console.log(`[Worker ${workerId}] ✅ ${results.length} resultados`);
        return results.map(r => ({ ...r, workerId, timestamp: Date.now() }));
    } catch (error) {
        console.error(`[Worker ${workerId}] ❌ ${error.message}`);
        return [];
    }
}

// ==================== FILTRADO Y RELEVANCIA ====================

function isSpamContent(text) {
    const lower = text.toLowerCase();
    return SPAM_KEYWORDS.some(s => lower.includes(s));
}

function isValidResult(result) {
    if (!result.title || result.title.length < 3 || result.title.length > 300) return false;
    if (!result.url || !result.url.startsWith('http')) return false;
    if (isSpamContent(result.title + ' ' + (result.description || ''))) return false;
    return true;
}

function calculateRelevance(result, keyword) {
    const terms = keyword.toLowerCase().split(/\s+/).filter(t => t.length > 1);
    const title = result.title.toLowerCase();
    const desc  = (result.description || '').toLowerCase();
    const url   = result.url.toLowerCase();
    let score   = 0;

    terms.forEach(term => {
        if (title.includes(term))      score += 35;
        if (title.startsWith(term))    score += 15;
        if (desc.includes(term))       score += 12;
        if (url.includes(term))        score += 8;
    });

    if (title.includes(keyword.toLowerCase())) score += 25;

    const qualitySources = { 'wikipedia': 20, 'github': 15, 'stackoverflow': 18, 'hacker news': 12, 'reddit': 8, 'youtube': 10, 'medium': 10, 'archive.org': 8, 'open library': 10 };
    const src = (result.source || '').toLowerCase();
    Object.entries(qualitySources).forEach(([s, b]) => { if (src.includes(s)) score += b; });

    const engineBonus = { 'Wikipedia': 10, 'GitHub': 8, 'StackOverflow': 8, 'Reddit': 5, 'HackerNews': 5 };
    if (engineBonus[result.engine]) score += engineBonus[result.engine];

    return Math.min(100, score);
}

function extractSourceName(url) {
    try {
        const hostname = new URL(url).hostname.replace('www.', '');
        const map = {
            'youtube.com': 'YouTube', 'youtu.be': 'YouTube',
            'mediafire.com': 'MediaFire',
            'drive.google.com': 'Google Drive', 'docs.google.com': 'Google Docs',
            'github.com': 'GitHub', 'gist.github.com': 'GitHub Gist',
            'reddit.com': 'Reddit',
            'stackoverflow.com': 'Stack Overflow', 'stackexchange.com': 'Stack Exchange',
            'medium.com': 'Medium', 'towardsdatascience.com': 'Towards Data Science',
            'wikipedia.org': 'Wikipedia', 'en.wikipedia.org': 'Wikipedia', 'es.wikipedia.org': 'Wikipedia',
            'news.ycombinator.com': 'Hacker News',
            'archive.org': 'Archive.org',
            'openlibrary.org': 'Open Library'
        };
        return map[hostname] || hostname;
    } catch (e) { return 'Web'; }
}

function removeDuplicates(results) {
    const seen = new Set();
    return results.filter(r => {
        try {
            const u = new URL(r.url);
            const key = (u.hostname.replace('www.', '') + u.pathname).toLowerCase().replace(/\/$/, '');
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        } catch { return false; }
    });
}

// ==================== BÚSQUEDA MULTI-WORKER ====================

async function multiWorkerSearch(keyword, filter) {
    const startTime = Date.now();

    const activeWorkers = (filter !== 'all' && FILTER_WORKERS[filter])
        ? Object.keys(FILTER_WORKERS[filter]).map(Number)
        : Array.from({ length: CONFIG.WORKERS_COUNT }, (_, i) => i + 1);

    console.log(`\n=== Búsqueda: "${keyword}" | Filtro: ${filter} | Workers: [${activeWorkers.join(',')}] ===`);

    const workerPromises = activeWorkers.map(id =>
        workerSearch(id, keyword, filter).catch(err => { console.error(`Worker ${id} falló:`, err.message); return []; })
    );

    const allResults = await Promise.all(workerPromises);
    let combined = allResults.flat();
    combined = combined.filter(isValidResult);
    combined = removeDuplicates(combined);
    combined = combined.map(r => ({ ...r, relevance: calculateRelevance(r, keyword) }));
    combined.sort((a, b) => b.relevance - a.relevance);

    const searchTime = ((Date.now() - startTime) / 1000).toFixed(2);
    console.log(`✅ Completado en ${searchTime}s | ${combined.length} resultados únicos\n`);

    return { results: combined, stats: { totalResults: combined.length, searchTime, workersUsed: activeWorkers.length, timestamp: new Date().toISOString() } };
}

// ==================== RUTAS ====================

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', uptime: process.uptime(), timestamp: new Date().toISOString(), workers: CONFIG.WORKERS_COUNT });
});

app.post('/api/search', async (req, res) => {
    try {
        const { keyword, filter = 'all' } = req.body;
        if (!keyword || keyword.trim().length === 0) return res.status(400).json({ error: 'Keyword es requerido' });
        if (!SITE_FILTERS.hasOwnProperty(filter)) return res.status(400).json({ error: 'Filtro inválido' });

        const searchResults = await multiWorkerSearch(keyword.trim(), filter);
        res.json({ success: true, ...searchResults });
    } catch (error) {
        console.error('Error en búsqueda:', error);
        res.status(500).json({ success: false, error: 'Error interno', message: error.message });
    }
});

app.get('/api/filters', (req, res) => {
    res.json({
        filters: Object.keys(SITE_FILTERS),
        description: {
            all: 'Todos los sitios (12 workers)', youtube: 'Buscar en YouTube',
            mediafire: 'Buscar en MediaFire', google: 'Buscar en Google Drive/Docs',
            github: 'Buscar en GitHub', reddit: 'Buscar en Reddit',
            stackoverflow: 'Buscar en Stack Overflow', medium: 'Buscar en Medium'
        }
    });
});

// ==================== AUTO-PING ====================

let pingInterval = null;

function startSelfPing() {
    if (!CONFIG.SELF_PING_URL) { console.log('⚠️  Auto-ping deshabilitado: SELF_PING_URL no configurado'); return; }
    console.log(`🔄 Auto-ping cada ${CONFIG.SELF_PING_INTERVAL / 60000} min`);
    performSelfPing();
    pingInterval = setInterval(performSelfPing, CONFIG.SELF_PING_INTERVAL);
}

async function performSelfPing() {
    try {
        const pingUrl = `${CONFIG.SELF_PING_URL.replace(/\/$/, '')}/api/health`;
        const response = await axios.get(pingUrl, { timeout: 5000, headers: { 'User-Agent': 'SelfPingBot/1.0' } });
        console.log(`✅ Auto-ping OK | Uptime: ${Math.floor(response.data.uptime)}s`);
    } catch (error) {
        console.error(`❌ Auto-ping falló:`, error.message);
    }
}

function stopSelfPing() {
    if (pingInterval) { clearInterval(pingInterval); pingInterval = null; }
}

app.post('/api/ping/start', (req, res) => {
    if (pingInterval) return res.json({ message: 'Auto-ping ya activo' });
    startSelfPing();
    res.json({ message: 'Auto-ping iniciado', interval: CONFIG.SELF_PING_INTERVAL });
});

app.post('/api/ping/stop', (req, res) => { stopSelfPing(); res.json({ message: 'Auto-ping detenido' }); });

app.get('/api/ping/status', (req, res) => {
    res.json({ active: pingInterval !== null, interval: CONFIG.SELF_PING_INTERVAL, intervalMinutes: CONFIG.SELF_PING_INTERVAL / 60000 });
});

// ==================== MANEJO DE ERRORES ====================

app.use((req, res) => { res.status(404).json({ error: 'Endpoint no encontrado', path: req.path }); });

app.use((error, req, res, next) => {
    if (error.message?.startsWith('CORS:')) return res.status(403).json({ error: error.message });
    console.error('Error no manejado:', error);
    res.status(500).json({ error: 'Error interno', message: error.message });
});

// ==================== INICIO ====================

const server = app.listen(PORT, () => {
    console.log('\n╔═══════════════════════════════════════════════════════╗');
    console.log('║     🚀 BUSCADOR AVANZADO v2.0 - 12 WORKERS           ║');
    console.log('╚═══════════════════════════════════════════════════════╝\n');
    console.log(`📡 Puerto:    ${PORT}`);
    console.log(`🌐 Self URL:  ${CONFIG.SELF_PING_URL || '⚠️  no configurado'}`);
    console.log(`🔗 Frontend:  ${CONFIG.VAR_URL || '⚠️  no configurado'}`);
    console.log(`⚙️  Workers:   ${CONFIG.WORKERS_COUNT}`);
    console.log('\n🔍 Fuentes: DuckDuckGo · Bing · Wikipedia · YouTube · GitHub · Reddit · StackOverflow · HackerNews · Archive.org · OpenLibrary\n');
    startSelfPing();
});

process.on('SIGTERM', () => { stopSelfPing(); server.close(() => process.exit(0)); });
process.on('SIGINT',  () => { stopSelfPing(); server.close(() => process.exit(0)); });

module.exports = app;
