require('dotenv').config();
const express = require('express');
const cors    = require('cors');
const axios   = require('axios');
const cheerio = require('cheerio');
const { URL } = require('url');
const path    = require('path');

const app  = express();
const PORT = process.env.PORT || 3000;

// ── CONFIG ────────────────────────────────────────────────────────────────
const CONFIG = {
    SELF_PING_URL:      process.env.SELF_PING_URL || '',
    VAR_URL:            process.env.VAR_URL || '',
    SELF_PING_INTERVAL: 14 * 60 * 1000,
    TIMEOUT:            13000,
    UAS: [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0'
    ]
};
const ua = () => CONFIG.UAS[Math.floor(Math.random() * CONFIG.UAS.length)];
const ax = (cfg) => axios({ timeout: CONFIG.TIMEOUT, ...cfg });

// ── CORS ──────────────────────────────────────────────────────────────────
app.use(cors({
    origin(origin, cb) {
        const ok  = CONFIG.VAR_URL?.replace(/\/$/, '');
        const self = CONFIG.SELF_PING_URL?.replace(/\/$/, '');
        if (!origin || origin === ok || origin === self) return cb(null, true);
        cb(new Error(`CORS: origen no permitido: ${origin}`), false);
    },
    methods: ['GET','POST','PUT','OPTIONS'],
    allowedHeaders: ['Content-Type','Authorization'],
    credentials: true
}));
app.options('*', cors());
app.use(express.json());
app.use(express.static('public'));
app.use((req, _, next) => { console.log(`${req.method} ${req.path}`); next(); });

// ── UTILIDADES ────────────────────────────────────────────────────────────
const SOURCE_MAP = {
    'youtube.com':'YouTube','youtu.be':'YouTube',
    'mediafire.com':'MediaFire',
    'drive.google.com':'Google Drive','docs.google.com':'Google Docs',
    'github.com':'GitHub','gist.github.com':'GitHub',
    'reddit.com':'Reddit','old.reddit.com':'Reddit',
    'stackoverflow.com':'Stack Overflow','stackexchange.com':'Stack Exchange','superuser.com':'Super User','askubuntu.com':'Ask Ubuntu',
    'medium.com':'Medium','towardsdatascience.com':'Medium','betterhumans.pub':'Medium',
    'en.wikipedia.org':'Wikipedia','es.wikipedia.org':'Wikipedia','wikipedia.org':'Wikipedia',
    'news.ycombinator.com':'Hacker News',
    'archive.org':'Archive.org',
    'openlibrary.org':'Open Library',
    'books.google.com':'Google Books',
    'npmjs.com':'npm',
    'dev.to':'DEV.to',
    'gitlab.com':'GitLab',
    'bitbucket.org':'Bitbucket',
    'codepen.io':'CodePen',
    'replit.com':'Replit',
    'pastebin.com':'Pastebin',
    'slideshare.net':'SlideShare',
    'scribd.com':'Scribd',
    'issuu.com':'Issuu',
    'academia.edu':'Academia',
    'researchgate.net':'ResearchGate',
    'arxiv.org':'arXiv',
    'twitter.com':'Twitter','x.com':'Twitter',
    'linkedin.com':'LinkedIn',
    'pinterest.com':'Pinterest',
    'twitch.tv':'Twitch',
    'vimeo.com':'Vimeo',
    'dailymotion.com':'Dailymotion',
    'soundcloud.com':'SoundCloud',
    'bandcamp.com':'Bandcamp',
    'spotify.com':'Spotify',
    'quora.com':'Quora',
    'producthunt.com':'Product Hunt',
    'alternativeto.net':'AlternativeTo'
};

function srcName(rawUrl) {
    try {
        const h = new URL(rawUrl).hostname.replace(/^www\./, '');
        return SOURCE_MAP[h] || h.split('.').slice(-2).join('.');
    } catch { return 'Web'; }
}

function mkResult(title, url, desc, extra = {}) {
    if (!title || !url) return null;
    try { new URL(url); } catch { return null; }
    return { title: title.trim().substring(0, 250), url, description: (desc || '').trim().substring(0, 400), source: srcName(url), ...extra };
}

// ── FUENTES ───────────────────────────────────────────────────────────────

// DDG — selector actualizado (usa .result en vez de .result__body en versiones recientes)
async function ddg(query, page = 1) {
    const off = (page - 1) * 10;
    try {
        const res = await ax({ method: 'GET', url: 'https://html.duckduckgo.com/html/',
            params: { q: query, s: off, dc: off + 1, v: 'l', o: 'json', api: 'd.js' },
            headers: { 'User-Agent': ua(), 'Accept': 'text/html,application/xhtml+xml', 'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8', 'Referer': 'https://duckduckgo.com/', 'DNT': '1' }
        });
        const $ = cheerio.load(res.data);
        const results = [];

        // Intentar múltiples selectores — DDG cambia el HTML con frecuencia
        const selectors = ['.result__body', '.result', '.web-result', '[data-result]'];
        let found = false;

        for (const sel of selectors) {
            const els = $(sel);
            if (els.length === 0) continue;
            found = true;

            els.each((_, el) => {
                // Intentar múltiples formas de extraer título
                const title = $(el).find('h2, .result__title, .result__a, a[href]').first().text().trim()
                           || $(el).find('a').first().text().trim();

                // Extraer URL — DDG puede tenerla en href directo o en data-href
                let link = $(el).find('a.result__a, h2 a, .result__title a').first().attr('href')
                        || $(el).find('a[href^="http"]').first().attr('href')
                        || $(el).find('a').first().attr('href') || '';

                // DDG a veces usa /l/?uddg= como redirect
                if (link.includes('/l/?')) {
                    try { link = new URL('https://duckduckgo.com' + link).searchParams.get('uddg') || link; } catch {}
                }
                if (link && !link.startsWith('http')) link = '';

                const desc = $(el).find('.result__snippet, .result__body p, p').first().text().trim();

                const r = mkResult(title, link, desc, { engine: 'DuckDuckGo' });
                if (r) results.push(r);
            });
            if (results.length > 0) break;
        }

        // Si ningún selector funcionó, intentar extraer links directamente
        if (!found || results.length === 0) {
            $('a[href^="http"]').each((_, el) => {
                const href  = $(el).attr('href') || '';
                const title = $(el).text().trim();
                if (!href.includes('duckduckgo.com') && title.length > 10) {
                    const r = mkResult(title, href, '', { engine: 'DuckDuckGo' });
                    if (r) results.push(r);
                }
            });
        }

        return results;
    } catch (e) { console.error('[DDG]', e.message); return []; }
}

// Bing — múltiples selectores por si cambia el HTML
async function bing(query, offset = 0) {
    try {
        const res = await ax({ method: 'GET', url: 'https://www.bing.com/search',
            params: { q: query, first: offset + 1, count: 20, setlang: 'es' },
            headers: { 'User-Agent': ua(), 'Accept': 'text/html,application/xhtml+xml', 'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8', 'Cache-Control': 'no-cache', 'Upgrade-Insecure-Requests': '1' }
        });
        const $ = cheerio.load(res.data);
        const results = [];

        $('.b_algo, li.b_algo').each((_, el) => {
            const title = $(el).find('h2').text().trim() || $(el).find('h3').text().trim();
            const link  = $(el).find('h2 a, h3 a').attr('href') || $(el).find('a[href^="http"]').first().attr('href');
            const desc  = $(el).find('.b_caption p').text().trim()
                       || $(el).find('.b_algoSlug').text().trim()
                       || $(el).find('p').first().text().trim();
            const r = mkResult(title, link, desc, { engine: 'Bing' });
            if (r) results.push(r);
        });
        return results;
    } catch (e) { console.error('[Bing]', e.message); return []; }
}

// Wikipedia ES + EN en paralelo
async function wikipedia(kw) {
    try {
        const calls = ['es','en'].map(lang =>
            ax({ method: 'GET', url: `https://${lang}.wikipedia.org/w/api.php`,
                params: { action:'query', list:'search', srsearch: kw, srlimit: 10, format:'json', srprop:'snippet|size|wordcount' },
                headers: { 'User-Agent': ua() }
            }).then(r => (r.data?.query?.search || []).map(i => mkResult(
                i.title,
                `https://${lang}.wikipedia.org/wiki/${encodeURIComponent(i.title.replace(/ /g,'_'))}`,
                i.snippet.replace(/<[^>]+>/g,'').replace(/&[a-z]+;/g,''),
                { engine: 'Wikipedia', source: `Wikipedia ${lang.toUpperCase()}` }
            ))).catch(() => [])
        );
        return (await Promise.all(calls)).flat().filter(Boolean);
    } catch (e) { console.error('[Wikipedia]', e.message); return []; }
}

// YouTube scraping
async function youtube(kw) {
    try {
        const res = await ax({ method: 'GET', url: `https://www.youtube.com/results`,
            params: { search_query: kw, hl: 'es' },
            headers: { 'User-Agent': ua(), 'Accept-Language': 'es-ES,es;q=0.9' }
        });
        const $ = cheerio.load(res.data);
        const results = [];
        for (const script of $('script').toArray()) {
            const c = $(script).html() || '';
            if (!c.includes('ytInitialData')) continue;
            try {
                const m = c.match(/ytInitialData\s*=\s*(\{.+?\});\s*(?:<\/script|var |window\.)/s);
                if (!m) continue;
                const data = JSON.parse(m[1]);
                const items = data?.contents?.twoColumnSearchResultsRenderer?.primaryContents
                    ?.sectionListRenderer?.contents?.flatMap(s => s?.itemSectionRenderer?.contents || []) || [];
                items.forEach(item => {
                    const v = item?.videoRenderer;
                    if (!v?.videoId) return;
                    results.push(mkResult(
                        v.title?.runs?.[0]?.text,
                        `https://www.youtube.com/watch?v=${v.videoId}`,
                        v.descriptionSnippet?.runs?.map(r=>r.text).join('') || `${v.ownerText?.runs?.[0]?.text||''} · ${v.viewCountText?.simpleText||''}`,
                        { engine: 'YouTube', source: 'YouTube', thumbnail: v.thumbnail?.thumbnails?.pop()?.url }
                    ));
                });
            } catch (e) { console.error('[YouTube parse]', e.message); }
            break;
        }
        return results.filter(Boolean);
    } catch (e) { console.error('[YouTube]', e.message); return []; }
}

// GitHub: repos + issues + topics
async function github(kw) {
    const h = { 'Accept':'application/vnd.github.v3+json', 'User-Agent': ua() };
    if (process.env.GITHUB_TOKEN) h['Authorization'] = `token ${process.env.GITHUB_TOKEN}`;
    const [repos, issues, topics] = await Promise.allSettled([
        ax({ method:'GET', url:'https://api.github.com/search/repositories', params:{ q:kw, sort:'stars', order:'desc', per_page:20 }, headers: h }),
        ax({ method:'GET', url:'https://api.github.com/search/issues',       params:{ q:`${kw} type:issue`, sort:'reactions', per_page:10 }, headers: h }),
        ax({ method:'GET', url:'https://api.github.com/search/topics',       params:{ q:kw, per_page:5 }, headers:{ ...h, Accept:'application/vnd.github.mercy-preview+json' } })
    ]);
    const results = [];
    if (repos.status==='fulfilled') repos.value.data.items?.forEach(r => results.push(mkResult(
        r.full_name, r.html_url,
        `⭐${r.stargazers_count} · ${r.language||'?'} · 🍴${r.forks_count} · ${r.description||''}`,
        { engine:'GitHub' }
    )));
    if (issues.status==='fulfilled') issues.value.data.items?.forEach(i => results.push(mkResult(
        `[Issue] ${i.title}`, i.html_url, i.body?.substring(0,200)||'', { engine:'GitHub' }
    )));
    if (topics.status==='fulfilled') topics.value.data.items?.forEach(t => results.push(mkResult(
        `[Topic] ${t.name}`, `https://github.com/topics/${t.name}`,
        t.short_description||t.description||'', { engine:'GitHub', source:'GitHub' }
    )));
    return results.filter(Boolean);
}

// GitLab public API
async function gitlab(kw) {
    try {
        const res = await ax({ method:'GET', url:'https://gitlab.com/api/v4/projects',
            params: { search: kw, order_by:'stars', sort:'desc', per_page:10, visibility:'public' },
            headers: { 'User-Agent': ua() }
        });
        return (res.data||[]).map(p => mkResult(
            p.name_with_namespace, p.web_url,
            `⭐${p.star_count} · 🍴${p.forks_count} · ${p.description||'Sin descripción'}`,
            { engine:'GitLab', source:'GitLab' }
        )).filter(Boolean);
    } catch (e) { console.error('[GitLab]', e.message); return []; }
}

// Reddit: posts + subreddits
async function reddit(kw) {
    const [posts, subs] = await Promise.allSettled([
        ax({ method:'GET', url:'https://www.reddit.com/search.json',
            params:{ q:kw, limit:25, sort:'relevance', type:'link' },
            headers:{ 'User-Agent':'SearchAggregator/3.0' } }),
        ax({ method:'GET', url:'https://www.reddit.com/search.json',
            params:{ q:kw, limit:10, sort:'relevance', type:'sr' },
            headers:{ 'User-Agent':'SearchAggregator/3.0' } })
    ]);
    const results = [];
    if (posts.status==='fulfilled') posts.value.data?.data?.children?.forEach(p => {
        const d = p.data;
        results.push(mkResult(d.title, `https://reddit.com${d.permalink}`,
            `r/${d.subreddit} · 👍${d.score} · 💬${d.num_comments}${d.selftext?' · '+d.selftext.substring(0,120):''}`,
            { engine:'Reddit' }));
    });
    if (subs.status==='fulfilled') subs.value.data?.data?.children?.forEach(s => {
        const d = s.data;
        results.push(mkResult(`r/${d.display_name}`, `https://reddit.com${d.url}`,
            `👥${(d.subscribers||0).toLocaleString()} · ${d.public_description?.substring(0,150)||d.title||''}`,
            { engine:'Reddit' }));
    });
    return results.filter(Boolean);
}

// StackExchange: SO + SuperUser + AskUbuntu
async function stackoverflow(kw) {
    const sites = ['stackoverflow','superuser','askubuntu'];
    const calls = sites.map(site =>
        ax({ method:'GET', url:'https://api.stackexchange.com/2.3/search/advanced',
            params:{ q:kw, site, sort:'relevance', order:'desc', pagesize:12 },
            headers:{ 'User-Agent': ua() }
        }).then(r => (r.data?.items||[]).map(i => mkResult(
            i.title, i.link,
            `✅${i.answer_count} resp · 👁${i.view_count} vistas · Score:${i.score} · [${(i.tags||[]).slice(0,3).join(', ')}]`,
            { engine:'StackOverflow', source: site==='stackoverflow'?'Stack Overflow':site==='superuser'?'Super User':'Ask Ubuntu' }
        ))).catch(() => [])
    );
    return (await Promise.all(calls)).flat().filter(Boolean);
}

// HackerNews stories + best comments
async function hackernews(kw) {
    const [stories, comments] = await Promise.allSettled([
        ax({ method:'GET', url:'https://hn.algolia.com/api/v1/search',
            params:{ query:kw, hitsPerPage:20, tags:'story' }, headers:{ 'User-Agent': ua() } }),
        ax({ method:'GET', url:'https://hn.algolia.com/api/v1/search',
            params:{ query:kw, hitsPerPage:10, tags:'comment', numericFilters:'points>5' }, headers:{ 'User-Agent': ua() } })
    ]);
    const results = [];
    if (stories.status==='fulfilled') stories.value.data?.hits?.forEach(h => {
        if (!h.title) return;
        results.push(mkResult(h.title, h.url||`https://news.ycombinator.com/item?id=${h.objectID}`,
            `⬆️${h.points||0} · 💬${h.num_comments||0} · ${h.author}`, { engine:'HackerNews' }));
    });
    if (comments.status==='fulfilled') comments.value.data?.hits?.forEach(h => {
        if (!h.comment_text) return;
        results.push(mkResult(`[HN] ${kw}`, `https://news.ycombinator.com/item?id=${h.objectID}`,
            h.comment_text.replace(/<[^>]+>/g,'').substring(0,200), { engine:'HackerNews' }));
    });
    return results.filter(Boolean);
}

// OpenLibrary + Google Books
async function books(kw) {
    const [ol, gb] = await Promise.allSettled([
        ax({ method:'GET', url:'https://openlibrary.org/search.json',
            params:{ q:kw, limit:12, fields:'title,author_name,first_publish_year,key,subject' },
            headers:{ 'User-Agent': ua() } }),
        ax({ method:'GET', url:'https://www.googleapis.com/books/v1/volumes',
            params:{ q:kw, maxResults:12, printType:'all', orderBy:'relevance' } })
    ]);
    const results = [];
    if (ol.status==='fulfilled') ol.value.data?.docs?.forEach(b => {
        if (!b.title) return;
        results.push(mkResult(b.title, `https://openlibrary.org${b.key}`,
            `📚 ${(b.author_name||['?']).join(', ')} · ${b.first_publish_year||'?'}${b.subject?' · '+b.subject.slice(0,3).join(', '):''}`,
            { engine:'Books', source:'Open Library' }));
    });
    if (gb.status==='fulfilled') gb.value.data?.items?.forEach(b => {
        const i = b.volumeInfo;
        if (!i?.title) return;
        results.push(mkResult(i.title, i.infoLink||`https://books.google.com/books?id=${b.id}`,
            `📖 ${(i.authors||['?']).join(', ')} · ${i.publishedDate||'?'} · ${i.description?.substring(0,100)||''}`,
            { engine:'Books', source:'Google Books' }));
    });
    return results.filter(Boolean);
}

// Archive.org por tipo de media
async function archive(kw) {
    const types = [
        { mt:'texts',   icon:'📄' },
        { mt:'movies',  icon:'🎬' },
        { mt:'audio',   icon:'🎵' },
        { mt:'software',icon:'💾' }
    ];
    const calls = types.map(({ mt, icon }) =>
        ax({ method:'GET', url:'https://archive.org/advancedsearch.php',
            params:{ q:`${kw} AND mediatype:${mt}`, fl:'identifier,title,description,mediatype,downloads', rows:8, output:'json', sort:'downloads desc' },
            headers:{ 'User-Agent': ua() }
        }).then(r => (r.data?.response?.docs||[]).map(item => {
            const t = Array.isArray(item.title) ? item.title[0] : item.title;
            const d = Array.isArray(item.description) ? item.description[0] : item.description;
            return mkResult(t, `https://archive.org/details/${item.identifier}`,
                `${icon} Archive.org · ${mt} · ⬇️${item.downloads||0} · ${d?String(d).substring(0,120):''}`,
                { engine:'Archive', source:'Archive.org' });
        })).catch(() => [])
    );
    return (await Promise.all(calls)).flat().filter(Boolean);
}

// npm registry
async function npm(kw) {
    try {
        const res = await ax({ method:'GET', url:'https://registry.npmjs.org/-/v1/search',
            params:{ text:kw, size:15 }, headers:{ 'User-Agent': ua() } });
        return (res.data?.objects||[]).map(p => {
            const pkg = p.package;
            return mkResult(pkg.name, `https://npmjs.com/package/${pkg.name}`,
                `📦 v${pkg.version} · ${Math.round((p.score?.detail?.popularity||0)*100)}% popular · ${pkg.description||''}`,
                { engine:'NPM', source:'npm' });
        }).filter(Boolean);
    } catch (e) { console.error('[npm]', e.message); return []; }
}

// DEV.to API pública
async function devto(kw) {
    try {
        const [tag, search] = await Promise.allSettled([
            ax({ method:'GET', url:'https://dev.to/api/articles', params:{ per_page:15, tag:kw.split(' ')[0], top:1 }, headers:{ 'User-Agent': ua() } }),
            ax({ method:'GET', url:'https://dev.to/api/articles', params:{ per_page:10, username:undefined }, headers:{ 'User-Agent': ua() } })
        ]);
        const results = [];
        if (tag.status==='fulfilled') tag.value.data?.forEach(a => {
            results.push(mkResult(a.title, a.url||`https://dev.to${a.path||''}`,
                `📝 DEV.to · ❤️${a.positive_reactions_count} · 💬${a.comments_count} · ${a.description||a.tag_list?.join(',')||''}`,
                { engine:'DEVto', source:'DEV.to' }));
        });
        return results.filter(Boolean);
    } catch (e) { console.error('[DEV.to]', e.message); return []; }
}

// MediaFire — múltiples estrategias combinadas
async function mediafire(kw) {
    const queries = [
        `site:mediafire.com/file ${kw}`,
        `site:mediafire.com ${kw}`,
        `"mediafire.com/file" "${kw}"`,
        `mediafire "${kw}" download`,
        `mediafire descargar "${kw}"`,
        `"mediafire" "${kw}" gratis`,
    ];
    const calls = [
        ddg(queries[0]), ddg(queries[1]), ddg(queries[2]),
        bing(queries[0]), bing(queries[3]), bing(queries[5]),
        ddg(queries[3]), ddg(queries[4])
    ];
    const all = (await Promise.allSettled(calls))
        .flatMap(r => r.status==='fulfilled' ? r.value : [])
        .filter(r => r?.url?.includes('mediafire.com'))
        .map(r => ({ ...r, source:'MediaFire', engine:'MediaFire' }));
    return all;
}

// Quora scraping básico via DDG
async function quora(kw) {
    const results = await ddg(`site:quora.com ${kw}`);
    return results.filter(r => r.url.includes('quora.com'))
        .map(r => ({ ...r, source:'Quora', engine:'Quora' }));
}

// arXiv — artículos académicos
async function arxiv(kw) {
    try {
        const res = await ax({ method:'GET', url:'https://export.arxiv.org/api/query',
            params:{ search_query:`all:${kw}`, start:0, max_results:10, sortBy:'relevance', sortOrder:'descending' },
            headers:{ 'User-Agent': ua() }
        });
        const $ = cheerio.load(res.data, { xmlMode: true });
        const results = [];
        $('entry').each((_, el) => {
            const title   = $('title', el).text().trim();
            const url     = $('id', el).text().trim();
            const summary = $('summary', el).text().trim().substring(0, 200);
            const authors = $('author name', el).map((_, a) => $(a).text()).get().slice(0,3).join(', ');
            results.push(mkResult(title, url, `📊 arXiv · ${authors} · ${summary}`, { engine:'arXiv', source:'arXiv' }));
        });
        return results.filter(Boolean);
    } catch (e) { console.error('[arXiv]', e.message); return []; }
}

// Vimeo — videos alternativos a YouTube
async function vimeo(kw) {
    const results = await ddg(`site:vimeo.com ${kw}`);
    return results.filter(r => r.url.includes('vimeo.com'))
        .map(r => ({ ...r, source:'Vimeo', engine:'Vimeo' }));
}

// ── EXPANSIÓN DE QUERIES ──────────────────────────────────────────────────
function expand(kw) {
    return {
        base:      kw,
        exact:     `"${kw}"`,
        download:  `${kw} download`,
        descargar: `${kw} descargar`,
        gratis:    `${kw} gratis`,
        tutorial:  `${kw} tutorial`,
        y2024:     `${kw} 2024`,
        y2025:     `${kw} 2025`,
        how:       `how to ${kw}`,
        como:      `como ${kw}`,
    };
}

// ── WORKERS POR FILTRO ────────────────────────────────────────────────────
function getWorkers(kw, filter) {
    const q = expand(kw);

    if (filter === 'youtube') return [
        () => youtube(kw),
        () => ddg(`site:youtube.com ${kw}`),
        () => ddg(`site:youtube.com ${q.tutorial}`),
        () => ddg(`site:youtube.com ${q.exact}`),
        () => ddg(`site:youtube.com ${q.y2024}`),
        () => bing(`site:youtube.com ${kw}`),
        () => bing(`site:youtube.com ${q.tutorial}`),
        () => bing(`youtube.com/watch "${kw}"`),
        () => ddg(`youtube ${kw} playlist`),
        () => bing(`site:youtube.com ${q.y2025}`),
    ];

    if (filter === 'github') return [
        () => github(kw),
        () => gitlab(kw),
        () => npm(kw),
        () => devto(kw),
        () => ddg(`site:github.com ${kw}`),
        () => ddg(`site:github.com ${q.exact}`),
        () => bing(`site:github.com ${kw}`),
        () => bing(`github ${kw} stars`),
        () => hackernews(kw),
        () => ddg(`site:gitlab.com ${kw}`),
    ];

    if (filter === 'reddit') return [
        () => reddit(kw),
        () => ddg(`site:reddit.com ${kw}`),
        () => ddg(`site:reddit.com ${q.exact}`),
        () => ddg(`site:old.reddit.com ${kw}`),
        () => bing(`site:reddit.com ${kw}`),
        () => bing(`reddit ${kw} discussion`),
        () => ddg(`site:reddit.com ${q.y2024}`),
        () => ddg(`reddit r/AskReddit ${kw}`),
        () => quora(kw),
        () => ddg(`site:reddit.com ${q.how}`),
    ];

    if (filter === 'stackoverflow') return [
        () => stackoverflow(kw),
        () => ddg(`site:stackoverflow.com ${kw}`),
        () => ddg(`site:stackoverflow.com ${q.exact}`),
        () => bing(`site:stackoverflow.com ${kw}`),
        () => ddg(`site:stackexchange.com ${kw}`),
        () => ddg(`stackoverflow ${kw} solution`),
        () => bing(`stackoverflow how to ${kw}`),
        () => hackernews(kw),
        () => devto(kw),
        () => ddg(`site:stackoverflow.com ${kw} error`),
    ];

    if (filter === 'mediafire') return [
        () => mediafire(kw),
        () => ddg(`site:mediafire.com/file ${kw}`),
        () => ddg(`site:mediafire.com ${kw}`),
        () => bing(`site:mediafire.com ${kw}`),
        () => ddg(`"mediafire.com/file" "${kw}"`),
        () => ddg(`mediafire ${q.descargar}`),
        () => bing(`mediafire "${kw}" link`),
        () => ddg(`mediafire ${q.gratis}`),
        () => bing(`filetype:zip OR filetype:rar mediafire ${kw}`),
        () => ddg(`"download" "mediafire" ${kw}`),
    ];

    if (filter === 'google') return [
        () => ddg(`site:drive.google.com ${kw}`),
        () => ddg(`site:docs.google.com ${kw}`),
        () => bing(`site:drive.google.com ${kw}`),
        () => bing(`site:docs.google.com ${kw}`),
        () => ddg(`"drive.google.com/file" ${kw}`),
        () => ddg(`"drive.google.com/drive/folders" ${kw}`),
        () => ddg(`google drive "${kw}" compartido`),
        () => bing(`"drive.google.com" "${kw}" public`),
        () => ddg(`google docs "${kw}" plantilla`),
        () => bing(`"docs.google.com" "${kw}" template`),
    ];

    if (filter === 'medium') return [
        () => ddg(`site:medium.com ${kw}`),
        () => bing(`site:medium.com ${kw}`),
        () => ddg(`site:medium.com ${q.exact}`),
        () => bing(`medium.com "${kw}" article`),
        () => ddg(`site:towardsdatascience.com ${kw}`),
        () => devto(kw),
        () => ddg(`medium ${q.tutorial}`),
        () => bing(`site:medium.com ${q.y2024}`),
        () => arxiv(kw),
        () => bing(`site:medium.com ${q.y2025}`),
    ];

    // filter === 'all' — 20 workers distintos
    return [
        () => ddg(kw, 1),
        () => ddg(kw, 2),
        () => ddg(q.exact),
        () => ddg(q.download),
        () => bing(kw, 0),
        () => bing(kw, 10),
        () => bing(kw, 20),
        () => wikipedia(kw),
        () => youtube(kw),
        () => github(kw),
        () => reddit(kw),
        () => stackoverflow(kw),
        () => hackernews(kw),
        () => books(kw),
        () => archive(kw),
        () => devto(kw),
        () => npm(kw),
        () => arxiv(kw),
        () => quora(kw),
        () => vimeo(kw),
    ];
}

// ── RELEVANCIA TF-IDF-LIKE ────────────────────────────────────────────────
// En vez de cap duro en 100, usa escala logarítmica y penaliza irrelevantes
function scoreResult(result, keyword) {
    const terms  = keyword.toLowerCase().split(/\s+/).filter(t => t.length > 1);
    const kw_lc  = keyword.toLowerCase();
    const title  = result.title.toLowerCase();
    const desc   = (result.description || '').toLowerCase();
    const url    = result.url.toLowerCase();

    let score = 0;

    // Frecuencia en título (peso mayor)
    terms.forEach(t => {
        const titleFreq = (title.match(new RegExp(t.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'), 'g')) || []).length;
        score += titleFreq * 30;
        if (title.startsWith(t)) score += 20;
        if (title === kw_lc)     score += 50;
    });

    // Frase completa
    if (title.includes(kw_lc)) score += 40;

    // Frecuencia en descripción
    terms.forEach(t => {
        const dFreq = (desc.match(new RegExp(t.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'), 'g')) || []).length;
        score += Math.min(dFreq, 3) * 8;
    });

    // En URL
    if (url.includes(kw_lc.replace(/ /g,'-')) || url.includes(kw_lc.replace(/ /g,'_'))) score += 15;
    terms.forEach(t => { if (url.includes(t)) score += 5; });

    // Bonus por fuente confiable
    const src = (result.source || '').toLowerCase();
    const srcBonuses = {
        'wikipedia': 25, 'stack overflow': 22, 'super user': 18, 'ask ubuntu': 18,
        'github': 18, 'gitlab': 14, 'arxiv': 20, 'hacker news': 15,
        'npm': 15, 'dev.to': 12, 'google books': 12, 'open library': 12,
        'youtube': 12, 'reddit': 10, 'medium': 12, 'quora': 8,
        'archive.org': 8, 'mediafire': 10, 'vimeo': 8
    };
    Object.entries(srcBonuses).forEach(([s, b]) => { if (src.includes(s)) score += b; });

    // Penalizar resultados sin descripción útil
    if (!result.description || result.description.length < 20) score -= 10;

    // Normalizar a 0-100 con logaritmo para mejor dispersión
    return Math.max(0, Math.min(100, Math.round(Math.log1p(score) / Math.log1p(300) * 100)));
}

// ── FILTRADO ──────────────────────────────────────────────────────────────
const SPAM = ['spam','scam','fake','virus','malware','phishing','free money','get rich','viagra','casino','porn','xxx'];

function isValid(r) {
    if (!r?.title || r.title.length < 3) return false;
    if (!r?.url || !r.url.startsWith('http')) return false;
    const txt = (r.title + ' ' + (r.description||'')).toLowerCase();
    return !SPAM.some(s => txt.includes(s));
}

function dedupe(arr) {
    const seen = new Set();
    return arr.filter(r => {
        try {
            const u = new URL(r.url);
            const k = `${u.hostname.replace(/^www\./,'')  }${u.pathname}`.toLowerCase().replace(/\/$/,'').replace(/[?#].*/,'');
            if (seen.has(k)) return false;
            seen.add(k);
            return true;
        } catch { return false; }
    });
}

// ── MOTOR PRINCIPAL ───────────────────────────────────────────────────────
async function search(keyword, filter) {
    const t0 = Date.now();
    const workers = getWorkers(keyword, filter);
    console.log(`\n▶ "${keyword}" [${filter}] — ${workers.length} workers`);

    const groups = await Promise.all(
        workers.map((fn, i) =>
            fn().then(r => { console.log(`  w${i+1} ✓ ${r.length}`); return r; })
               .catch(e  => { console.error(`  w${i+1} ✗ ${e.message}`); return []; })
        )
    );

    let results = groups.flat().filter(isValid);
    results = dedupe(results);
    results = results.map(r => ({ ...r, relevance: scoreResult(r, keyword) }));
    results.sort((a, b) => b.relevance - a.relevance);

    const elapsed = ((Date.now() - t0) / 1000).toFixed(2);
    console.log(`✅ ${elapsed}s | ${results.length} resultados\n`);

    return { results, stats: { totalResults: results.length, searchTime: elapsed, workersUsed: workers.length, timestamp: new Date().toISOString() } };
}

// ── RUTAS ─────────────────────────────────────────────────────────────────
app.get('/', (_, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));

app.get('/api/health', (_, res) => res.json({ status:'ok', uptime: process.uptime(), timestamp: new Date().toISOString(), version:'4.0' }));

const VALID_FILTERS = ['all','youtube','mediafire','google','github','reddit','stackoverflow','medium'];
app.post('/api/search', async (req, res) => {
    try {
        const { keyword, filter = 'all' } = req.body;
        if (!keyword?.trim()) return res.status(400).json({ error:'Keyword requerido' });
        if (!VALID_FILTERS.includes(filter)) return res.status(400).json({ error:'Filtro inválido' });
        const data = await search(keyword.trim(), filter);
        res.json({ success: true, ...data });
    } catch (e) {
        console.error('Error:', e);
        res.status(500).json({ success:false, error:'Error interno', message: e.message });
    }
});

app.get('/api/filters', (_, res) => res.json({
    filters: VALID_FILTERS,
    description: { all:'Todos (20 workers)', youtube:'YouTube (10 workers)', github:'GitHub+GitLab+npm (10)', reddit:'Reddit+Quora (10)', stackoverflow:'StackOverflow×3 (10)', mediafire:'MediaFire (10)', google:'Google Drive/Docs (10)', medium:'Medium+DEV.to+arXiv (10)' }
}));

// ── AUTO-PING ─────────────────────────────────────────────────────────────
let pingInterval = null;
const startPing = () => {
    if (!CONFIG.SELF_PING_URL) return;
    const ping = async () => {
        try { const r = await ax({ method:'GET', url:`${CONFIG.SELF_PING_URL.replace(/\/$/,'')}/api/health`, timeout:5000, headers:{'User-Agent':'SelfPingBot/1.0'} }); console.log(`✅ Ping OK uptime:${Math.floor(r.data.uptime)}s`); }
        catch (e) { console.error('❌ Ping falló:', e.message); }
    };
    ping(); pingInterval = setInterval(ping, CONFIG.SELF_PING_INTERVAL);
    console.log(`🔄 Auto-ping cada ${CONFIG.SELF_PING_INTERVAL/60000}min`);
};
const stopPing = () => { if (pingInterval) { clearInterval(pingInterval); pingInterval = null; } };

app.post('/api/ping/start',  (_, res) => { if (!pingInterval) startPing(); res.json({ message: pingInterval?'ya activo':'iniciado' }); });
app.post('/api/ping/stop',   (_, res) => { stopPing(); res.json({ message:'detenido' }); });
app.get ('/api/ping/status', (_, res) => res.json({ active:!!pingInterval, intervalMinutes: CONFIG.SELF_PING_INTERVAL/60000 }));

app.use((req, res) => res.status(404).json({ error:'No encontrado', path: req.path }));
app.use((err, req, res, _) => {
    if (err.message?.startsWith('CORS:')) return res.status(403).json({ error: err.message });
    res.status(500).json({ error:'Error interno', message: err.message });
});

// ── INICIO ────────────────────────────────────────────────────────────────
const server = app.listen(PORT, () => {
    console.log('\n╔══════════════════════════════════════════════════════════╗');
    console.log('║  🚀 BUSCADOR v4.0  ·  20 workers  ·  15+ fuentes        ║');
    console.log('╚══════════════════════════════════════════════════════════╝');
    console.log(`📡 Puerto: ${PORT} | Self: ${CONFIG.SELF_PING_URL||'—'} | Front: ${CONFIG.VAR_URL||'—'}`);
    console.log('🔍 DDG·Bing·Wikipedia·YouTube·GitHub·GitLab·Reddit·SO·HN·Books·Archive·npm·DEV.to·arXiv·Quora·Vimeo·MediaFire\n');
    startPing();
});
process.on('SIGTERM', () => { stopPing(); server.close(() => process.exit(0)); });
process.on('SIGINT',  () => { stopPing(); server.close(() => process.exit(0)); });
module.exports = app;
