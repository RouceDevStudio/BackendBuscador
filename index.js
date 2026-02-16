require('dotenv').config();

const express = require('express');
const cors    = require('cors');
const axios   = require('axios');
const cheerio = require('cheerio');
const { URL } = require('url');
const path    = require('path');

const app  = express();
const PORT = process.env.PORT || 3000;

// ==================== CONFIGURACIÓN ====================
const CONFIG = {
    SELF_PING_URL:     process.env.SELF_PING_URL || '',
    VAR_URL:           process.env.VAR_URL || '',
    SELF_PING_INTERVAL: 14 * 60 * 1000,
    REQUEST_TIMEOUT:   13000,
    USER_AGENTS: [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0'
    ]
};

const rnd = () => CONFIG.USER_AGENTS[Math.floor(Math.random() * CONFIG.USER_AGENTS.length)];

// ==================== CORS ====================
const corsOptions = {
    origin(origin, cb) {
        const allowed = CONFIG.VAR_URL?.replace(/\/$/, '');
        const self    = CONFIG.SELF_PING_URL?.replace(/\/$/, '');
        if (!origin) return cb(null, true);
        if (!allowed) return cb(new Error('CORS: VAR_URL no configurado'), false);
        if (origin === allowed || origin === self) return cb(null, true);
        cb(new Error(`CORS: Origen no permitido: ${origin}`), false);
    },
    methods: ['GET', 'POST', 'PUT', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true
};
app.use(cors(corsOptions));
app.options('*', cors(corsOptions));
app.use(express.json());
app.use(express.static('public'));
app.use((req, _, next) => { console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`); next(); });

// ==================== EXPANSIÓN INTELIGENTE DE QUERIES ====================
// Genera variaciones del keyword para multiplicar resultados
function expandQuery(keyword) {
    const kw = keyword.trim();
    return [
        kw,
        `"${kw}"`,
        `${kw} download`,
        `${kw} descargar`,
        `${kw} gratis`,
        `${kw} free`,
        `${kw} 2024`,
        `${kw} tutorial`,
        `${kw} como`,
        `${kw} guide`
    ];
}

// ==================== FUENTES BASE ====================

// DDG HTML — no bloquea bots, múltiples páginas
async function ddg(query, page = 1) {
    const results = [];
    try {
        const off = (page - 1) * 10;
        const res = await axios.get('https://html.duckduckgo.com/html/', {
            params: { q: query, s: off, dc: off + 1 },
            headers: { 'User-Agent': rnd(), 'Accept': 'text/html', 'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8', 'Referer': 'https://duckduckgo.com/' },
            timeout: CONFIG.REQUEST_TIMEOUT
        });
        const $ = cheerio.load(res.data);
        $('.result__body').each((_, el) => {
            const title = $(el).find('.result__title').text().trim();
            let   link  = $(el).find('.result__url').text().trim();
            const desc  = $(el).find('.result__snippet').text().trim();
            if (link && !link.startsWith('http')) link = 'https://' + link;
            if (title && link) {
                try { new URL(link); results.push({ title, url: link, description: desc || 'Sin descripción', source: srcName(link), engine: 'DuckDuckGo' }); }
                catch (_) {}
            }
        });
    } catch (e) { console.error('[DDG]', e.message); }
    return results;
}

// Bing — headers rotativos, offset para paginación
async function bing(query, offset = 0) {
    const results = [];
    try {
        const res = await axios.get('https://www.bing.com/search', {
            params: { q: query, first: offset + 1, count: 20 },
            headers: { 'User-Agent': rnd(), 'Accept': 'text/html,application/xhtml+xml', 'Accept-Language': 'es-ES,es;q=0.9', 'Cache-Control': 'no-cache' },
            timeout: CONFIG.REQUEST_TIMEOUT
        });
        const $ = cheerio.load(res.data);
        $('.b_algo').each((_, el) => {
            const title = $(el).find('h2').text().trim();
            const link  = $(el).find('h2 a').attr('href');
            const desc  = $(el).find('.b_caption p, .b_algoSlug').first().text().trim();
            if (title && link?.startsWith('http')) {
                try { new URL(link); results.push({ title, url: link, description: desc || 'Sin descripción', source: srcName(link), engine: 'Bing' }); }
                catch (_) {}
            }
        });
    } catch (e) { console.error('[Bing]', e.message); }
    return results;
}

// Wikipedia API — ES + EN
async function wikipedia(keyword) {
    const results = [];
    try {
        for (const lang of ['es', 'en']) {
            const res = await axios.get(`https://${lang}.wikipedia.org/w/api.php`, {
                params: { action: 'query', list: 'search', srsearch: keyword, srlimit: 10, format: 'json', srprop: 'snippet' },
                headers: { 'User-Agent': rnd() },
                timeout: CONFIG.REQUEST_TIMEOUT
            });
            (res.data?.query?.search || []).forEach(item => {
                const clean = item.snippet.replace(/<[^>]+>/g, '').replace(/&quot;/g, '"').replace(/&#039;/g, "'");
                results.push({ title: item.title, url: `https://${lang}.wikipedia.org/wiki/${encodeURIComponent(item.title.replace(/ /g, '_'))}`, description: clean || 'Artículo de Wikipedia', source: `Wikipedia (${lang.toUpperCase()})`, engine: 'Wikipedia' });
            });
        }
    } catch (e) { console.error('[Wikipedia]', e.message); }
    return results;
}

// YouTube — parser de ytInitialData
async function youtube(keyword) {
    const results = [];
    try {
        const res = await axios.get(`https://www.youtube.com/results?search_query=${encodeURIComponent(keyword)}&hl=es`, {
            headers: { 'User-Agent': rnd(), 'Accept-Language': 'es-ES,es;q=0.9' },
            timeout: CONFIG.REQUEST_TIMEOUT
        });
        const $ = cheerio.load(res.data);
        for (const script of $('script').toArray()) {
            const c = $(script).html() || '';
            if (c.includes('ytInitialData')) {
                try {
                    const m = c.match(/ytInitialData\s*=\s*(\{.+?\});\s*(?:var |window\.|<\/script)/s);
                    if (m) {
                        const data = JSON.parse(m[1]);
                        const sections = data?.contents?.twoColumnSearchResultsRenderer?.primaryContents?.sectionListRenderer?.contents || [];
                        sections.forEach(s => {
                            (s?.itemSectionRenderer?.contents || []).forEach(item => {
                                const v = item?.videoRenderer;
                                if (v?.videoId) {
                                    results.push({
                                        title: v.title?.runs?.[0]?.text || 'Sin título',
                                        url: `https://www.youtube.com/watch?v=${v.videoId}`,
                                        description: v.descriptionSnippet?.runs?.map(r => r.text).join('') || `Canal: ${v.ownerText?.runs?.[0]?.text || ''} | ${v.viewCountText?.simpleText || ''}`,
                                        source: 'YouTube', engine: 'YouTube'
                                    });
                                }
                            });
                        });
                    }
                } catch (e) { console.error('[YouTube parse]', e.message); }
                break;
            }
        }
    } catch (e) { console.error('[YouTube]', e.message); }
    return results;
}

// GitHub API — repos + issues + code search
async function github(keyword) {
    const results = [];
    try {
        const headers = { 'Accept': 'application/vnd.github.v3+json', 'User-Agent': rnd() };
        if (process.env.GITHUB_TOKEN) headers['Authorization'] = `token ${process.env.GITHUB_TOKEN}`;

        const [repos, issues, topics] = await Promise.allSettled([
            axios.get('https://api.github.com/search/repositories', { params: { q: keyword, sort: 'stars', order: 'desc', per_page: 20 }, headers, timeout: CONFIG.REQUEST_TIMEOUT }),
            axios.get('https://api.github.com/search/issues',       { params: { q: `${keyword} type:issue`, sort: 'reactions', per_page: 10 }, headers, timeout: CONFIG.REQUEST_TIMEOUT }),
            axios.get('https://api.github.com/search/topics',       { params: { q: keyword, per_page: 5 }, headers: { ...headers, 'Accept': 'application/vnd.github.mercy-preview+json' }, timeout: CONFIG.REQUEST_TIMEOUT })
        ]);

        if (repos.status === 'fulfilled') repos.value.data.items?.forEach(r => {
            results.push({ title: r.full_name, url: r.html_url, description: `⭐ ${r.stargazers_count} | ${r.language || '?'} | ${r.description || 'Sin descripción'} | 🍴 ${r.forks_count}`, source: 'GitHub', engine: 'GitHub' });
        });
        if (issues.status === 'fulfilled') issues.value.data.items?.forEach(i => {
            results.push({ title: `[Issue] ${i.title}`, url: i.html_url, description: i.body?.substring(0, 200) || 'Ver issue', source: 'GitHub', engine: 'GitHub' });
        });
        if (topics.status === 'fulfilled') topics.value.data.items?.forEach(t => {
            results.push({ title: `[Topic] ${t.name}`, url: `https://github.com/topics/${t.name}`, description: t.short_description || t.description || 'Topic de GitHub', source: 'GitHub', engine: 'GitHub' });
        });
    } catch (e) { console.error('[GitHub]', e.message); }
    return results;
}

// Reddit API — posts + subreddits
async function reddit(keyword) {
    const results = [];
    try {
        const [posts, subs] = await Promise.allSettled([
            axios.get('https://www.reddit.com/search.json', { params: { q: keyword, limit: 25, sort: 'relevance', type: 'link' }, headers: { 'User-Agent': 'SearchBot/3.0' }, timeout: CONFIG.REQUEST_TIMEOUT }),
            axios.get('https://www.reddit.com/search.json', { params: { q: keyword, limit: 10, sort: 'relevance', type: 'sr' },   headers: { 'User-Agent': 'SearchBot/3.0' }, timeout: CONFIG.REQUEST_TIMEOUT })
        ]);

        if (posts.status === 'fulfilled') posts.value.data?.data?.children?.forEach(p => {
            const d = p.data;
            results.push({ title: d.title, url: `https://www.reddit.com${d.permalink}`, description: `r/${d.subreddit} | 👍 ${d.score} | 💬 ${d.num_comments}${d.selftext ? ' | ' + d.selftext.substring(0, 150) : ''}`, source: 'Reddit', engine: 'Reddit' });
        });
        if (subs.status === 'fulfilled') subs.value.data?.data?.children?.forEach(s => {
            const d = s.data;
            results.push({ title: `r/${d.display_name}`, url: `https://www.reddit.com${d.url}`, description: `Subreddit | 👥 ${d.subscribers?.toLocaleString()} miembros | ${d.public_description || d.title || ''}`, source: 'Reddit', engine: 'Reddit' });
        });
    } catch (e) { console.error('[Reddit]', e.message); }
    return results;
}

// Stack Overflow API + Stack Exchange
async function stackoverflow(keyword) {
    const results = [];
    try {
        const [so, se] = await Promise.allSettled([
            axios.get('https://api.stackexchange.com/2.3/search/advanced', { params: { q: keyword, site: 'stackoverflow', sort: 'relevance', order: 'desc', pagesize: 20 }, headers: { 'User-Agent': rnd() }, timeout: CONFIG.REQUEST_TIMEOUT }),
            axios.get('https://api.stackexchange.com/2.3/search/advanced', { params: { q: keyword, site: 'superuser',     sort: 'relevance', order: 'desc', pagesize: 10 }, headers: { 'User-Agent': rnd() }, timeout: CONFIG.REQUEST_TIMEOUT })
        ]);

        if (so.status === 'fulfilled') so.value.data?.items?.forEach(i => {
            results.push({ title: i.title, url: i.link, description: `✅ ${i.answer_count} resp | 👁 ${i.view_count} vistas | Score: ${i.score} | Tags: ${(i.tags || []).slice(0, 4).join(', ')}`, source: 'Stack Overflow', engine: 'StackOverflow' });
        });
        if (se.status === 'fulfilled') se.value.data?.items?.forEach(i => {
            results.push({ title: i.title, url: i.link, description: `✅ ${i.answer_count} resp | 👁 ${i.view_count} vistas | Score: ${i.score}`, source: 'Super User', engine: 'StackOverflow' });
        });
    } catch (e) { console.error('[StackOverflow]', e.message); }
    return results;
}

// Hacker News (Algolia) — stories + comments
async function hackernews(keyword) {
    const results = [];
    try {
        const [stories, comments] = await Promise.allSettled([
            axios.get('https://hn.algolia.com/api/v1/search', { params: { query: keyword, hitsPerPage: 20, tags: 'story' }, headers: { 'User-Agent': rnd() }, timeout: CONFIG.REQUEST_TIMEOUT }),
            axios.get('https://hn.algolia.com/api/v1/search', { params: { query: keyword, hitsPerPage: 10, tags: 'comment' }, headers: { 'User-Agent': rnd() }, timeout: CONFIG.REQUEST_TIMEOUT })
        ]);

        if (stories.status === 'fulfilled') stories.value.data?.hits?.forEach(h => {
            if (!h.title) return;
            results.push({ title: h.title, url: h.url || `https://news.ycombinator.com/item?id=${h.objectID}`, description: `🔥 HN | ⬆️ ${h.points || 0} | 💬 ${h.num_comments || 0} | ${h.author}`, source: 'Hacker News', engine: 'HackerNews' });
        });
        if (comments.status === 'fulfilled') comments.value.data?.hits?.forEach(h => {
            if (!h.comment_text) return;
            results.push({ title: `[HN Comment] ${keyword}`, url: `https://news.ycombinator.com/item?id=${h.objectID}`, description: h.comment_text.substring(0, 200).replace(/<[^>]+>/g, ''), source: 'Hacker News', engine: 'HackerNews' });
        });
    } catch (e) { console.error('[HackerNews]', e.message); }
    return results;
}

// Open Library + Google Books API gratuita
async function books(keyword) {
    const results = [];
    try {
        const [ol, gb] = await Promise.allSettled([
            axios.get('https://openlibrary.org/search.json', { params: { q: keyword, limit: 15, fields: 'title,author_name,first_publish_year,key,subject' }, headers: { 'User-Agent': rnd() }, timeout: CONFIG.REQUEST_TIMEOUT }),
            axios.get('https://www.googleapis.com/books/v1/volumes',       { params: { q: keyword, maxResults: 10, printType: 'all', orderBy: 'relevance' }, timeout: CONFIG.REQUEST_TIMEOUT })
        ]);

        if (ol.status === 'fulfilled') ol.value.data?.docs?.forEach(b => {
            if (!b.title) return;
            results.push({ title: b.title, url: `https://openlibrary.org${b.key}`, description: `📚 OpenLibrary | ${(b.author_name || ['?']).join(', ')} | ${b.first_publish_year || '?'}${b.subject ? ' | ' + b.subject.slice(0, 3).join(', ') : ''}`, source: 'Open Library', engine: 'Books' });
        });
        if (gb.status === 'fulfilled') gb.value.data?.items?.forEach(b => {
            const info = b.volumeInfo;
            if (!info?.title) return;
            results.push({ title: info.title, url: info.infoLink || `https://books.google.com/books?id=${b.id}`, description: `📖 Google Books | ${(info.authors || ['?']).join(', ')} | ${info.publishedDate || '?'} | ${info.description?.substring(0, 120) || 'Ver libro'}`, source: 'Google Books', engine: 'Books' });
        });
    } catch (e) { console.error('[Books]', e.message); }
    return results;
}

// Archive.org — múltiples tipos de media
async function archive(keyword) {
    const results = [];
    try {
        const mediatypes = ['texts', 'movies', 'audio', 'software'];
        const promises = mediatypes.map(mt =>
            axios.get('https://archive.org/advancedsearch.php', {
                params: { q: `${keyword} AND mediatype:${mt}`, fl: 'identifier,title,description,mediatype,downloads', rows: 8, output: 'json', sort: 'downloads desc' },
                headers: { 'User-Agent': rnd() },
                timeout: CONFIG.REQUEST_TIMEOUT
            }).catch(() => null)
        );
        const responses = await Promise.all(promises);
        responses.forEach(res => {
            res?.data?.response?.docs?.forEach(item => {
                if (!item.title) return;
                const title = Array.isArray(item.title) ? item.title[0] : item.title;
                const desc  = Array.isArray(item.description) ? item.description[0] : item.description;
                results.push({ title, url: `https://archive.org/details/${item.identifier}`, description: `🗄️ Archive.org | ${item.mediatype} | ⬇️ ${item.downloads || 0} descargas | ${desc ? String(desc).substring(0, 120) : 'Ver en Archive.org'}`, source: 'Archive.org', engine: 'Archive' });
            });
        });
    } catch (e) { console.error('[Archive]', e.message); }
    return results;
}

// MediaFire — búsqueda especializada con múltiples estrategias
async function mediafire(keyword) {
    const results = [];
    const queries = [
        `site:mediafire.com ${keyword}`,
        `mediafire.com/file ${keyword}`,
        `"mediafire" "${keyword}" download`,
        `mediafire ${keyword} -site:mediafire.com/account -site:mediafire.com/help`
    ];

    try {
        const searches = await Promise.allSettled([
            ddg(queries[0]),
            bing(queries[0]),
            ddg(queries[1]),
            ddg(queries[2]),
            bing(queries[3])
        ]);

        searches.forEach(s => {
            if (s.status === 'fulfilled') {
                s.value.forEach(r => {
                    if (r.url.includes('mediafire.com')) {
                        results.push({ ...r, source: 'MediaFire', engine: 'MediaFire' });
                    }
                });
            }
        });
    } catch (e) { console.error('[MediaFire]', e.message); }
    return results;
}

// NPM registry — útil para búsquedas de código
async function npm(keyword) {
    const results = [];
    try {
        const res = await axios.get('https://registry.npmjs.org/-/v1/search', {
            params: { text: keyword, size: 15 },
            headers: { 'User-Agent': rnd() },
            timeout: CONFIG.REQUEST_TIMEOUT
        });
        res.data?.objects?.forEach(p => {
            const pkg = p.package;
            results.push({ title: pkg.name, url: `https://www.npmjs.com/package/${pkg.name}`, description: `📦 npm | v${pkg.version} | ⬇️ Semanal: ${p.score?.detail?.popularity ? Math.round(p.score.detail.popularity * 100) + '%' : '?'} | ${pkg.description || 'Sin descripción'}`, source: 'npm', engine: 'NPM' });
        });
    } catch (e) { console.error('[NPM]', e.message); }
    return results;
}

// DEV.to API pública
async function devto(keyword) {
    const results = [];
    try {
        const res = await axios.get('https://dev.to/api/articles', {
            params: { per_page: 20, tag: keyword.split(' ')[0], top: 1 },
            headers: { 'User-Agent': rnd() },
            timeout: CONFIG.REQUEST_TIMEOUT
        });
        // También buscar por título
        const res2 = await axios.get('https://dev.to/search/feed_content', {
            params: { per_page: 15, search_fields: keyword, class_name: 'Article' },
            headers: { 'User-Agent': rnd() },
            timeout: CONFIG.REQUEST_TIMEOUT
        }).catch(() => null);

        res.data?.forEach(a => {
            results.push({ title: a.title, url: a.url || `https://dev.to${a.path}`, description: `📝 DEV.to | ❤️ ${a.positive_reactions_count} | 💬 ${a.comments_count} | ${a.description || a.tag_list?.join(', ') || ''}`, source: 'DEV.to', engine: 'DEVto' });
        });
        res2?.data?.result?.forEach(a => {
            if (!a.title) return;
            results.push({ title: a.title, url: `https://dev.to${a.path}`, description: `📝 DEV.to | ❤️ ${a.positive_reactions_count || 0} | ${a.tag_list?.join(', ') || ''}`, source: 'DEV.to', engine: 'DEVto' });
        });
    } catch (e) { console.error('[DEV.to]', e.message); }
    return results;
}

// ==================== SISTEMA DE WORKERS v3 ====================
// Para búsquedas generales: 16 workers, cada uno diferente
// Para filtros: todos los workers relevantes a esa fuente

async function runWorker(name, fn, keyword) {
    console.log(`  [${name}] iniciando...`);
    try {
        const results = await fn(keyword);
        console.log(`  [${name}] ✅ ${results.length} resultados`);
        return results.map(r => ({ ...r, workerName: name, timestamp: Date.now() }));
    } catch (e) {
        console.error(`  [${name}] ❌ ${e.message}`);
        return [];
    }
}

// Workers para búsqueda general "all"
function buildAllWorkers(keyword) {
    const exp = expandQuery(keyword);
    return [
        // Búsqueda directa en múltiples fuentes
        () => runWorker('DDG-p1',       kw => ddg(kw, 1),                   keyword),
        () => runWorker('DDG-p2',       kw => ddg(kw, 2),                   keyword),
        () => runWorker('DDG-var1',     kw => ddg(exp[1]),                   keyword),  // "keyword" exacto
        () => runWorker('DDG-var2',     kw => ddg(exp[6]),                   keyword),  // keyword 2024
        () => runWorker('Bing-p1',      kw => bing(kw, 0),                  keyword),
        () => runWorker('Bing-p2',      kw => bing(kw, 10),                 keyword),
        () => runWorker('Bing-p3',      kw => bing(kw, 20),                 keyword),
        () => runWorker('Wikipedia',    kw => wikipedia(kw),                keyword),
        () => runWorker('YouTube',      kw => youtube(kw),                  keyword),
        () => runWorker('GitHub',       kw => github(kw),                   keyword),
        () => runWorker('Reddit',       kw => reddit(kw),                   keyword),
        () => runWorker('StackOF',      kw => stackoverflow(kw),            keyword),
        () => runWorker('HackerNews',   kw => hackernews(kw),               keyword),
        () => runWorker('Books',        kw => books(kw),                    keyword),
        () => runWorker('Archive',      kw => archive(kw),                  keyword),
        () => runWorker('DEVto',        kw => devto(kw),                    keyword),
        () => runWorker('NPM',          kw => npm(kw),                      keyword),
        () => runWorker('DDG-download', kw => ddg(exp[2]),                   keyword),  // keyword download
    ];
}

// Workers especializados por filtro — máxima cobertura de esa fuente
function buildFilterWorkers(keyword, filter) {
    const exp = expandQuery(keyword);

    const map = {
        youtube: [
            () => runWorker('YT-direct',    () => youtube(keyword),                              keyword),
            () => runWorker('YT-DDG1',      () => ddg(`site:youtube.com ${keyword}`),            keyword),
            () => runWorker('YT-DDG2',      () => ddg(`site:youtu.be ${keyword}`),               keyword),
            () => runWorker('YT-Bing1',     () => bing(`site:youtube.com ${keyword}`),           keyword),
            () => runWorker('YT-Bing2',     () => bing(`youtube.com/watch ${keyword}`),          keyword),
            () => runWorker('YT-tutorial',  () => ddg(`site:youtube.com ${keyword} tutorial`),   keyword),
            () => runWorker('YT-2024',      () => ddg(`site:youtube.com ${keyword} 2024`),       keyword),
            () => runWorker('YT-full',      () => bing(`site:youtube.com "${keyword}"`),         keyword),
        ],
        github: [
            () => runWorker('GH-api',       () => github(keyword),                               keyword),
            () => runWorker('GH-DDG1',      () => ddg(`site:github.com ${keyword}`),             keyword),
            () => runWorker('GH-DDG2',      () => ddg(`github.com ${keyword} repository`),       keyword),
            () => runWorker('GH-Bing1',     () => bing(`site:github.com ${keyword}`),            keyword),
            () => runWorker('GH-Bing2',     () => bing(`github ${keyword} stars`),               keyword),
            () => runWorker('GH-npm',       () => npm(keyword),                                  keyword),
            () => runWorker('GH-topics',    () => ddg(`site:github.com/topics ${keyword}`),      keyword),
            () => runWorker('GH-devto',     () => devto(keyword),                                keyword),
        ],
        reddit: [
            () => runWorker('RD-api',       () => reddit(keyword),                               keyword),
            () => runWorker('RD-DDG1',      () => ddg(`site:reddit.com ${keyword}`),             keyword),
            () => runWorker('RD-Bing1',     () => bing(`site:reddit.com ${keyword}`),            keyword),
            () => runWorker('RD-DDG2',      () => ddg(`reddit ${keyword} discussion`),           keyword),
            () => runWorker('RD-Bing2',     () => bing(`reddit.com/r ${keyword}`),               keyword),
            () => runWorker('RD-DDG3',      () => ddg(`site:old.reddit.com ${keyword}`),         keyword),
            () => runWorker('RD-2024',      () => ddg(`site:reddit.com ${keyword} 2024`),        keyword),
            () => runWorker('RD-ask',       () => ddg(`site:reddit.com "r/AskReddit" ${keyword}`), keyword),
        ],
        stackoverflow: [
            () => runWorker('SO-api',       () => stackoverflow(keyword),                        keyword),
            () => runWorker('SO-DDG1',      () => ddg(`site:stackoverflow.com ${keyword}`),      keyword),
            () => runWorker('SO-Bing1',     () => bing(`site:stackoverflow.com ${keyword}`),     keyword),
            () => runWorker('SO-DDG2',      () => ddg(`stackoverflow ${keyword} solution`),      keyword),
            () => runWorker('SO-error',     () => ddg(`site:stackoverflow.com "${keyword}" error`), keyword),
            () => runWorker('SO-how',       () => ddg(`stackoverflow how to ${keyword}`),        keyword),
            () => runWorker('SE-meta',      () => ddg(`site:stackexchange.com ${keyword}`),      keyword),
            () => runWorker('HN-code',      () => hackernews(keyword),                           keyword),
        ],
        mediafire: [
            () => runWorker('MF-DDG1',      () => ddg(`site:mediafire.com/file ${keyword}`),     keyword),
            () => runWorker('MF-DDG2',      () => ddg(`site:mediafire.com ${keyword}`),          keyword),
            () => runWorker('MF-Bing1',     () => bing(`site:mediafire.com ${keyword}`),         keyword),
            () => runWorker('MF-Bing2',     () => bing(`mediafire.com ${keyword} download`),     keyword),
            () => runWorker('MF-DDG3',      () => ddg(`"mediafire.com" "${keyword}"`),           keyword),
            () => runWorker('MF-DDG4',      () => ddg(`mediafire descargar ${keyword}`),         keyword),
            () => runWorker('MF-DDG5',      () => ddg(`mediafire ${keyword} gratis`),            keyword),
            () => runWorker('MF-Bing3',     () => bing(`"mediafire" "${keyword}" link`),         keyword),
        ],
        google: [
            () => runWorker('GD-DDG1',      () => ddg(`site:drive.google.com ${keyword}`),       keyword),
            () => runWorker('GD-Bing1',     () => bing(`site:drive.google.com ${keyword}`),      keyword),
            () => runWorker('GD-DDG2',      () => ddg(`site:docs.google.com ${keyword}`),        keyword),
            () => runWorker('GD-Bing2',     () => bing(`site:docs.google.com ${keyword}`),       keyword),
            () => runWorker('GD-DDG3',      () => ddg(`google drive "${keyword}" compartido`),   keyword),
            () => runWorker('GD-DDG4',      () => ddg(`"drive.google.com/file" ${keyword}`),     keyword),
            () => runWorker('GD-DDG5',      () => ddg(`"drive.google.com/drive/folders" ${keyword}`), keyword),
            () => runWorker('GD-Bing3',     () => bing(`"drive.google.com" "${keyword}" public`), keyword),
        ],
        medium: [
            () => runWorker('MD-DDG1',      () => ddg(`site:medium.com ${keyword}`),             keyword),
            () => runWorker('MD-Bing1',     () => bing(`site:medium.com ${keyword}`),            keyword),
            () => runWorker('MD-DDG2',      () => ddg(`medium.com ${keyword} article`),          keyword),
            () => runWorker('MD-Bing2',     () => bing(`medium ${keyword} story`),               keyword),
            () => runWorker('MD-DDG3',      () => ddg(`site:towardsdatascience.com ${keyword}`), keyword),
            () => runWorker('MD-devto',     () => devto(keyword),                                keyword),
            () => runWorker('MD-DDG4',      () => ddg(`medium "${keyword}" tutorial`),           keyword),
            () => runWorker('MD-Bing3',     () => bing(`site:medium.com "${keyword}"`),          keyword),
        ]
    };

    return map[filter] || buildAllWorkers(keyword);
}

// ==================== FILTRADO Y RELEVANCIA ====================

const SPAM = ['spam','scam','fake','virus','malware','phishing','click here now','free money','get rich','weight loss','viagra','casino','porn','xxx'];

function isValid(r) {
    if (!r.title || r.title.length < 3 || r.title.length > 300) return false;
    if (!r.url || !r.url.startsWith('http')) return false;
    const txt = (r.title + ' ' + (r.description || '')).toLowerCase();
    if (SPAM.some(s => txt.includes(s))) return false;
    return true;
}

function relevance(result, keyword) {
    const terms = keyword.toLowerCase().split(/\s+/).filter(t => t.length > 1);
    const title = result.title.toLowerCase();
    const desc  = (result.description || '').toLowerCase();
    const url   = result.url.toLowerCase();
    let score   = 0;

    terms.forEach(t => {
        if (title.includes(t))   score += 35;
        if (title.startsWith(t)) score += 15;
        if (desc.includes(t))    score += 12;
        if (url.includes(t))     score += 8;
    });
    if (title.includes(keyword.toLowerCase())) score += 25;

    const srcBonus = { 'wikipedia': 20, 'github': 15, 'stackoverflow': 18, 'super user': 14, 'hacker news': 12, 'npm': 12, 'dev.to': 10, 'reddit': 8, 'youtube': 10, 'medium': 10, 'archive.org': 8, 'open library': 10, 'google books': 10, 'mediafire': 15 };
    const src = (result.source || '').toLowerCase();
    Object.entries(srcBonus).forEach(([s, b]) => { if (src.includes(s)) score += b; });

    const engBonus = { 'Wikipedia': 10, 'GitHub': 8, 'StackOverflow': 8, 'Reddit': 5, 'HackerNews': 5, 'Books': 6, 'NPM': 8, 'DEVto': 6, 'MediaFire': 5 };
    if (engBonus[result.engine]) score += engBonus[result.engine];

    return Math.min(100, score);
}

function srcName(url) {
    try {
        const h = new URL(url).hostname.replace('www.', '');
        const m = { 'youtube.com':'YouTube','youtu.be':'YouTube','mediafire.com':'MediaFire','drive.google.com':'Google Drive','docs.google.com':'Google Docs','github.com':'GitHub','gist.github.com':'GitHub Gist','reddit.com':'Reddit','old.reddit.com':'Reddit','stackoverflow.com':'Stack Overflow','stackexchange.com':'Stack Exchange','superuser.com':'Super User','medium.com':'Medium','towardsdatascience.com':'Towards Data Science','en.wikipedia.org':'Wikipedia','es.wikipedia.org':'Wikipedia','wikipedia.org':'Wikipedia','news.ycombinator.com':'Hacker News','archive.org':'Archive.org','openlibrary.org':'Open Library','books.google.com':'Google Books','npmjs.com':'npm','dev.to':'DEV.to','devto.com':'DEV.to' };
        return m[h] || h;
    } catch { return 'Web'; }
}

function dedupe(results) {
    const seen = new Set();
    return results.filter(r => {
        try {
            const u = new URL(r.url);
            const k = (u.hostname.replace('www.', '') + u.pathname).toLowerCase().replace(/\/$/, '').replace(/[?#].*/, '');
            if (seen.has(k)) return false;
            seen.add(k);
            return true;
        } catch { return false; }
    });
}

// ==================== BÚSQUEDA PRINCIPAL ====================

async function search(keyword, filter) {
    const t0 = Date.now();

    const workerFns = filter === 'all'
        ? buildAllWorkers(keyword)
        : buildFilterWorkers(keyword, filter);

    console.log(`\n=== "${keyword}" | filtro:${filter} | ${workerFns.length} workers ===`);

    // Lanzar todos en paralelo
    const groups = await Promise.all(workerFns.map(fn => fn().catch(() => [])));

    let combined = groups.flat();
    combined = combined.filter(isValid);
    combined = dedupe(combined);
    combined = combined.map(r => ({ ...r, relevance: relevance(r, keyword) }));
    combined.sort((a, b) => b.relevance - a.relevance);

    const elapsed = ((Date.now() - t0) / 1000).toFixed(2);
    console.log(`✅ ${elapsed}s | ${combined.length} resultados únicos\n`);

    return { results: combined, stats: { totalResults: combined.length, searchTime: elapsed, workersUsed: workerFns.length, timestamp: new Date().toISOString() } };
}

// ==================== RUTAS ====================

app.get('/', (_, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));

app.get('/api/health', (_, res) => res.json({ status: 'ok', uptime: process.uptime(), timestamp: new Date().toISOString(), workers: 18 }));

app.post('/api/search', async (req, res) => {
    try {
        const { keyword, filter = 'all' } = req.body;
        if (!keyword?.trim()) return res.status(400).json({ error: 'Keyword requerido' });
        if (!['all','youtube','mediafire','google','github','reddit','stackoverflow','medium'].includes(filter))
            return res.status(400).json({ error: 'Filtro inválido' });

        const data = await search(keyword.trim(), filter);
        res.json({ success: true, ...data });
    } catch (e) {
        console.error('Error búsqueda:', e);
        res.status(500).json({ success: false, error: 'Error interno', message: e.message });
    }
});

app.get('/api/filters', (_, res) => res.json({
    filters: ['all','youtube','mediafire','google','github','reddit','stackoverflow','medium'],
    description: { all:'Todos (18 workers)', youtube:'YouTube (8 workers)', mediafire:'MediaFire (8 workers)', google:'Google Drive (8 workers)', github:'GitHub + npm + DEV.to (8 workers)', reddit:'Reddit (8 workers)', stackoverflow:'Stack Overflow (8 workers)', medium:'Medium + DEV.to (8 workers)' }
}));

// ==================== AUTO-PING ====================
let pingInterval = null;

function startSelfPing() {
    if (!CONFIG.SELF_PING_URL) { console.log('⚠️  Auto-ping deshabilitado'); return; }
    console.log(`🔄 Auto-ping cada ${CONFIG.SELF_PING_INTERVAL / 60000} min`);
    performSelfPing();
    pingInterval = setInterval(performSelfPing, CONFIG.SELF_PING_INTERVAL);
}

async function performSelfPing() {
    try {
        const url = `${CONFIG.SELF_PING_URL.replace(/\/$/, '')}/api/health`;
        const r   = await axios.get(url, { timeout: 5000, headers: { 'User-Agent': 'SelfPingBot/1.0' } });
        console.log(`✅ Ping OK | uptime: ${Math.floor(r.data.uptime)}s`);
    } catch (e) { console.error('❌ Ping falló:', e.message); }
}

function stopSelfPing() { if (pingInterval) { clearInterval(pingInterval); pingInterval = null; } }

app.post('/api/ping/start',  (_, res) => { if (pingInterval) return res.json({ message: 'Ya activo' }); startSelfPing(); res.json({ message: 'Iniciado' }); });
app.post('/api/ping/stop',   (_, res) => { stopSelfPing(); res.json({ message: 'Detenido' }); });
app.get ('/api/ping/status', (_, res) => res.json({ active: !!pingInterval, interval: CONFIG.SELF_PING_INTERVAL, intervalMinutes: CONFIG.SELF_PING_INTERVAL / 60000 }));

// ==================== ERRORES ====================
app.use((req, res) => res.status(404).json({ error: 'No encontrado', path: req.path }));
app.use((err, req, res, _) => {
    if (err.message?.startsWith('CORS:')) return res.status(403).json({ error: err.message });
    res.status(500).json({ error: 'Error interno', message: err.message });
});

// ==================== INICIO ====================
const server = app.listen(PORT, () => {
    console.log('\n╔══════════════════════════════════════════════════════════╗');
    console.log('║   🚀 BUSCADOR v3.0 — 18 workers generales / 8 por filtro ║');
    console.log('╚══════════════════════════════════════════════════════════╝');
    console.log(`📡 Puerto: ${PORT}`);
    console.log(`🌐 Self:   ${CONFIG.SELF_PING_URL || '⚠️ no configurado'}`);
    console.log(`🔗 Front:  ${CONFIG.VAR_URL || '⚠️ no configurado'}`);
    console.log('🔍 Fuentes: DDG×4 · Bing×3 · Wikipedia · YouTube · GitHub · Reddit · StackOverflow · HackerNews · Archive · Books · DEV.to · npm\n');
    startSelfPing();
});

process.on('SIGTERM', () => { stopSelfPing(); server.close(() => process.exit(0)); });
process.on('SIGINT',  () => { stopSelfPing(); server.close(() => process.exit(0)); });

module.exports = app;
