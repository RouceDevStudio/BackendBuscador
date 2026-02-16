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
    TIMEOUT:            14000,
    UAS: [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0'
    ]
};
const ua  = () => CONFIG.UAS[Math.floor(Math.random() * CONFIG.UAS.length)];
const ax  = (cfg) => axios({ timeout: CONFIG.TIMEOUT, ...cfg });
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// ── CORS ──────────────────────────────────────────────────────────────────
app.use(cors({
    origin(origin, cb) {
        const ok   = CONFIG.VAR_URL?.replace(/\/$/, '');
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

// ── MAPA DE FUENTES ───────────────────────────────────────────────────────
const SRC = {
    'youtube.com':'YouTube','youtu.be':'YouTube',
    'mediafire.com':'MediaFire',
    'drive.google.com':'Google Drive','docs.google.com':'Google Docs',
    'mega.nz':'MEGA','mega.co.nz':'MEGA',
    'dropbox.com':'Dropbox',
    'onedrive.live.com':'OneDrive','1drv.ms':'OneDrive',
    'wetransfer.com':'WeTransfer',
    'zippyshare.com':'ZippyShare',
    'sendspace.com':'SendSpace',
    'box.com':'Box',
    'github.com':'GitHub','gist.github.com':'GitHub',
    'gitlab.com':'GitLab',
    'reddit.com':'Reddit','old.reddit.com':'Reddit',
    'stackoverflow.com':'Stack Overflow','stackexchange.com':'Stack Exchange',
    'medium.com':'Medium','towardsdatascience.com':'Medium',
    'en.wikipedia.org':'Wikipedia','es.wikipedia.org':'Wikipedia','wikipedia.org':'Wikipedia',
    'archive.org':'Archive.org',
    'openlibrary.org':'Open Library',
    'books.google.com':'Google Books',
    'scribd.com':'Scribd',
    'issuu.com':'Issuu',
    'academia.edu':'Academia.edu',
    'researchgate.net':'ResearchGate',
    'arxiv.org':'arXiv',
    'z-lib.org':'Z-Library','zlibrary.to':'Z-Library','1lib.net':'Z-Library',
    'libgen.is':'LibGen','libgen.rs':'LibGen','libgen.st':'LibGen',
    'pdfdrive.com':'PDF Drive',
    'pdfroom.com':'PDF Room',
    'ebook3000.com':'eBook3000',
    'gutenberg.org':'Project Gutenberg',
    'manybooks.net':'ManyBooks',
    'freebookspot.es':'FreeBookSpot',
    'slideshare.net':'SlideShare',
    'dailymotion.com':'Dailymotion',
    'vimeo.com':'Vimeo',
    'twitch.tv':'Twitch',
    'soundcloud.com':'SoundCloud',
    'bandcamp.com':'Bandcamp',
    'news.ycombinator.com':'Hacker News',
    'npmjs.com':'npm',
    'dev.to':'DEV.to',
    'quora.com':'Quora',
    'pinterest.com':'Pinterest',
    'imgur.com':'Imgur'
};

function srcName(rawUrl) {
    try {
        const h = new URL(rawUrl).hostname.replace(/^www\./, '');
        return SRC[h] || h.split('.').slice(-2).join('.');
    } catch { return 'Web'; }
}

// Detectar si un resultado contiene un link de descarga real
function isDownloadLink(result) {
    const url = result.url.toLowerCase();
    const title = (result.title + ' ' + result.description).toLowerCase();
    const downloadHosts = ['mediafire.com','drive.google.com','mega.nz','dropbox.com',
        'onedrive.live.com','1drv.ms','archive.org','github.com','gitlab.com',
        'wetransfer.com','zippyshare.com','sendspace.com','box.com','pdfdrive.com',
        'libgen','z-lib','gutenberg.org','scribd.com','academia.edu','researchgate.net',
        'openlibrary.org','issuu.com','slideshare.net','pdfroom.com','manybooks.net',
        'freebookspot.es','ebook3000.com','1lib.net','zlibrary'];
    const dlKeywords = ['download','descargar','pdf','ebook','gratis','free','descarga','link',
        'mega','mediafire','drive','zip','rar','torrent','direct'];

    const isHost = downloadHosts.some(h => url.includes(h));
    const isKw   = dlKeywords.some(k => title.includes(k) || url.includes(k));
    return { isHost, isKw, isDownload: isHost || isKw };
}

function mkResult(title, url, desc, extra = {}) {
    if (!title || !url) return null;
    try { new URL(url); } catch { return null; }
    return {
        title: title.trim().substring(0, 250),
        url,
        description: (desc || '').trim().substring(0, 400),
        source: srcName(url),
        ...extra
    };
}

// ── FUENTES ───────────────────────────────────────────────────────────────

// DuckDuckGo HTML — más robusto, múltiples selectores
async function ddg(query, page = 1) {
    const off = (page - 1) * 10;
    try {
        const res = await ax({
            method: 'GET', url: 'https://html.duckduckgo.com/html/',
            params: { q: query, s: off, dc: off + 1 },
            headers: {
                'User-Agent': ua(), 'Accept': 'text/html,application/xhtml+xml',
                'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
                'Referer': 'https://duckduckgo.com/'
            }
        });
        const $       = cheerio.load(res.data);
        const results = [];

        // Intentar selectores en orden de prioridad
        const parsed = tryParseDDG($);
        return parsed.length > 0 ? parsed : fallbackLinks($, 'duckduckgo.com');
    } catch (e) { console.error('[DDG]', e.message); return []; }
}

function tryParseDDG($) {
    const results = [];
    // DDG usa diferentes layouts, probar todos
    ['.result__body', '.result', '.web-result', '[data-nir]'].forEach(sel => {
        if (results.length > 0) return;
        $(sel).each((_, el) => {
            const titleEl = $(el).find('a.result__a, h2 a, .result__title a, a[href^="http"]').first();
            const title   = titleEl.text().trim() || $(el).find('h2, h3').first().text().trim();
            let   link    = titleEl.attr('href') || '';
            if (link.includes('/l/?')) {
                try { link = new URL('https://duckduckgo.com' + link).searchParams.get('uddg') || link; } catch {}
            }
            if (!link.startsWith('http')) link = '';
            const desc = $(el).find('.result__snippet, p').first().text().trim();
            const r = mkResult(title, link, desc, { engine: 'DuckDuckGo' });
            if (r) results.push(r);
        });
    });
    return results;
}

function fallbackLinks($, excludeDomain) {
    const results = [];
    $('a[href^="http"]').each((_, el) => {
        const href  = $(el).attr('href') || '';
        const title = $(el).text().trim();
        if (!href.includes(excludeDomain) && !href.includes('duckduckgo') && title.length > 8) {
            const r = mkResult(title, href, '', { engine: 'DuckDuckGo' });
            if (r) results.push(r);
        }
    });
    return results.slice(0, 15);
}

// Bing
async function bing(query, offset = 0) {
    try {
        const res = await ax({
            method: 'GET', url: 'https://www.bing.com/search',
            params: { q: query, first: offset + 1, count: 20 },
            headers: {
                'User-Agent': ua(), 'Accept': 'text/html,application/xhtml+xml',
                'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8', 'Cache-Control': 'no-cache'
            }
        });
        const $       = cheerio.load(res.data);
        const results = [];
        $('.b_algo, li.b_algo').each((_, el) => {
            const title = $(el).find('h2').text().trim() || $(el).find('h3').text().trim();
            const link  = $(el).find('h2 a, h3 a').attr('href') || $(el).find('a[href^="http"]').first().attr('href');
            const desc  = $(el).find('.b_caption p').text().trim() || $(el).find('p').first().text().trim();
            const r = mkResult(title, link, desc, { engine: 'Bing' });
            if (r) results.push(r);
        });
        if (results.length === 0) return fallbackLinks($, 'bing.com');
        return results;
    } catch (e) { console.error('[Bing]', e.message); return []; }
}

// Wikipedia
async function wikipedia(kw) {
    try {
        const calls = ['es','en'].map(lang =>
            ax({ method:'GET', url:`https://${lang}.wikipedia.org/w/api.php`,
                params:{ action:'query', list:'search', srsearch:kw, srlimit:8, format:'json', srprop:'snippet' },
                headers:{ 'User-Agent': ua() }
            }).then(r => (r.data?.query?.search||[]).map(i => mkResult(
                i.title,
                `https://${lang}.wikipedia.org/wiki/${encodeURIComponent(i.title.replace(/ /g,'_'))}`,
                i.snippet.replace(/<[^>]+>/g,'').replace(/&[a-z]+;/g,''),
                { engine:'Wikipedia', source:`Wikipedia ${lang.toUpperCase()}` }
            ))).catch(() => [])
        );
        return (await Promise.all(calls)).flat().filter(Boolean);
    } catch (e) { console.error('[Wikipedia]', e.message); return []; }
}

// YouTube
async function youtube(kw) {
    try {
        const res = await ax({
            method:'GET', url:'https://www.youtube.com/results',
            params:{ search_query: kw, hl:'es' },
            headers:{ 'User-Agent': ua(), 'Accept-Language':'es-ES,es;q=0.9' }
        });
        const $ = cheerio.load(res.data);
        const results = [];
        for (const script of $('script').toArray()) {
            const c = $(script).html() || '';
            if (!c.includes('ytInitialData')) continue;
            try {
                const m = c.match(/ytInitialData\s*=\s*(\{.+?\});\s*(?:<\/script|var |window\.)/s);
                if (!m) continue;
                const data  = JSON.parse(m[1]);
                const items = data?.contents?.twoColumnSearchResultsRenderer?.primaryContents
                    ?.sectionListRenderer?.contents
                    ?.flatMap(s => s?.itemSectionRenderer?.contents || []) || [];
                items.forEach(item => {
                    const v = item?.videoRenderer;
                    if (!v?.videoId) return;
                    results.push(mkResult(
                        v.title?.runs?.[0]?.text,
                        `https://www.youtube.com/watch?v=${v.videoId}`,
                        v.descriptionSnippet?.runs?.map(r=>r.text).join('') ||
                            `${v.ownerText?.runs?.[0]?.text||''} · ${v.viewCountText?.simpleText||''}`,
                        { engine:'YouTube', source:'YouTube', thumbnail: v.thumbnail?.thumbnails?.pop()?.url }
                    ));
                });
            } catch(e) { console.error('[YT parse]', e.message); }
            break;
        }
        return results.filter(Boolean);
    } catch (e) { console.error('[YouTube]', e.message); return []; }
}

// GitHub
async function github(kw) {
    const h = { 'Accept':'application/vnd.github.v3+json', 'User-Agent': ua() };
    if (process.env.GITHUB_TOKEN) h['Authorization'] = `token ${process.env.GITHUB_TOKEN}`;
    const [repos, issues] = await Promise.allSettled([
        ax({ method:'GET', url:'https://api.github.com/search/repositories', params:{ q:kw, sort:'stars', order:'desc', per_page:15 }, headers:h }),
        ax({ method:'GET', url:'https://api.github.com/search/issues', params:{ q:`${kw} type:issue`, sort:'reactions', per_page:8 }, headers:h })
    ]);
    const results = [];
    if (repos.status==='fulfilled') repos.value.data.items?.forEach(r => results.push(
        mkResult(r.full_name, r.html_url,
            `⭐${r.stargazers_count} · ${r.language||'?'} · 🍴${r.forks_count} · ${r.description||''}`,
            { engine:'GitHub' })
    ));
    if (issues.status==='fulfilled') issues.value.data.items?.forEach(i => results.push(
        mkResult(`[Issue] ${i.title}`, i.html_url, i.body?.substring(0,200)||'', { engine:'GitHub' })
    ));
    return results.filter(Boolean);
}

// Reddit
async function reddit(kw) {
    const [posts, subs] = await Promise.allSettled([
        ax({ method:'GET', url:'https://www.reddit.com/search.json', params:{ q:kw, limit:25, sort:'relevance', type:'link' }, headers:{ 'User-Agent':'SearchAggregator/3.0' } }),
        ax({ method:'GET', url:'https://www.reddit.com/search.json', params:{ q:kw, limit:8,  sort:'relevance', type:'sr'   }, headers:{ 'User-Agent':'SearchAggregator/3.0' } })
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
            `👥${(d.subscribers||0).toLocaleString()} · ${d.public_description?.substring(0,120)||''}`,
            { engine:'Reddit' }));
    });
    return results.filter(Boolean);
}

// Stack Overflow
async function stackoverflow(kw) {
    const calls = ['stackoverflow','superuser'].map(site =>
        ax({ method:'GET', url:'https://api.stackexchange.com/2.3/search/advanced',
            params:{ q:kw, site, sort:'relevance', order:'desc', pagesize:12 }, headers:{ 'User-Agent': ua() }
        }).then(r => (r.data?.items||[]).map(i => mkResult(
            i.title, i.link,
            `✅${i.answer_count} resp · 👁${i.view_count} · [${(i.tags||[]).slice(0,3).join(', ')}]`,
            { engine:'StackOverflow', source: site==='stackoverflow'?'Stack Overflow':'Super User' }
        ))).catch(() => [])
    );
    return (await Promise.all(calls)).flat().filter(Boolean);
}

// HackerNews
async function hackernews(kw) {
    try {
        const res = await ax({ method:'GET', url:'https://hn.algolia.com/api/v1/search',
            params:{ query:kw, hitsPerPage:20, tags:'story' }, headers:{ 'User-Agent': ua() } });
        return (res.data?.hits||[]).filter(h=>h.title).map(h =>
            mkResult(h.title, h.url||`https://news.ycombinator.com/item?id=${h.objectID}`,
                `⬆️${h.points||0} · 💬${h.num_comments||0} · ${h.author}`, { engine:'HackerNews' })
        ).filter(Boolean);
    } catch (e) { console.error('[HN]', e.message); return []; }
}

// Archive.org — excelente para descargas legales de libros, música, software
async function archive(kw) {
    const types = ['texts','movies','audio','software','image'];
    const calls = types.map(mt =>
        ax({ method:'GET', url:'https://archive.org/advancedsearch.php',
            params:{ q:`${kw} AND mediatype:${mt}`, fl:'identifier,title,description,mediatype,downloads', rows:10, output:'json', sort:'downloads desc' },
            headers:{ 'User-Agent': ua() }
        }).then(r => (r.data?.response?.docs||[]).map(item => {
            const t = Array.isArray(item.title) ? item.title[0] : item.title;
            const d = Array.isArray(item.description) ? item.description[0] : item.description;
            const icons = { texts:'📄', movies:'🎬', audio:'🎵', software:'💾', image:'🖼️' };
            return mkResult(t, `https://archive.org/details/${item.identifier}`,
                `${icons[mt]||'📁'} Archive.org · ${mt} · ⬇️${item.downloads||0} descargas · ${d?String(d).substring(0,100):''}`,
                { engine:'Archive', source:'Archive.org', isDownload: true });
        })).catch(() => [])
    );
    return (await Promise.all(calls)).flat().filter(Boolean);
}

// OpenLibrary + Google Books
async function books(kw) {
    const [ol, gb] = await Promise.allSettled([
        ax({ method:'GET', url:'https://openlibrary.org/search.json', params:{ q:kw, limit:12, fields:'title,author_name,first_publish_year,key,subject,has_fulltext' }, headers:{ 'User-Agent': ua() } }),
        ax({ method:'GET', url:'https://www.googleapis.com/books/v1/volumes', params:{ q:kw, maxResults:10, printType:'all', orderBy:'relevance', filter:'free-ebooks' } })
    ]);
    const results = [];
    if (ol.status==='fulfilled') ol.value.data?.docs?.forEach(b => {
        if (!b.title) return;
        results.push(mkResult(b.title, `https://openlibrary.org${b.key}`,
            `📚 OpenLibrary · ${(b.author_name||['?']).join(', ')} · ${b.first_publish_year||'?'}${b.has_fulltext?' · ✅ Texto completo disponible':''}`,
            { engine:'Books', source:'Open Library' }));
    });
    if (gb.status==='fulfilled') gb.value.data?.items?.forEach(b => {
        const i = b.volumeInfo;
        if (!i?.title) return;
        const pdfLink = b.accessInfo?.pdf?.downloadLink || b.accessInfo?.epub?.downloadLink;
        results.push(mkResult(i.title, pdfLink || i.infoLink || `https://books.google.com/books?id=${b.id}`,
            `📖 Google Books · ${(i.authors||['?']).join(', ')} · ${b.accessInfo?.viewability||'?'}${pdfLink?' · ✅ PDF disponible':''}`,
            { engine:'Books', source:'Google Books', isDownload: !!pdfLink }));
    });
    return results.filter(Boolean);
}

// Project Gutenberg — libros gratis en dominio público
async function gutenberg(kw) {
    try {
        const res = await ax({ method:'GET', url:'https://gutendex.com/books/',
            params:{ search: kw, languages:'es,en' }, headers:{ 'User-Agent': ua() } });
        return (res.data?.results||[]).map(b => mkResult(
            b.title,
            b.formats?.['text/html'] || b.formats?.['application/epub+zip'] || `https://www.gutenberg.org/ebooks/${b.id}`,
            `📗 Gutenberg · ${b.authors?.map(a=>a.name).join(', ')||'?'} · Dominio público · ✅ Descarga gratuita`,
            { engine:'Gutenberg', source:'Project Gutenberg', isDownload: true }
        )).filter(Boolean);
    } catch (e) { console.error('[Gutenberg]', e.message); return []; }
}

// arXiv — papers académicos con PDF directo
async function arxiv(kw) {
    try {
        const res = await ax({ method:'GET', url:'https://export.arxiv.org/api/query',
            params:{ search_query:`all:${kw}`, start:0, max_results:10, sortBy:'relevance', sortOrder:'descending' },
            headers:{ 'User-Agent': ua() } });
        const $ = cheerio.load(res.data, { xmlMode: true });
        const results = [];
        $('entry').each((_, el) => {
            const arxivId = $('id', el).text().trim().replace('http://arxiv.org/abs/','').replace('https://arxiv.org/abs/','');
            const title   = $('title', el).text().trim();
            const summary = $('summary', el).text().trim().substring(0, 200);
            const authors = $('author name', el).map((_, a) => $(a).text()).get().slice(0,3).join(', ');
            results.push(mkResult(title,
                `https://arxiv.org/pdf/${arxivId}.pdf`,
                `📊 arXiv · ${authors} · ${summary} · ✅ PDF libre`,
                { engine:'arXiv', source:'arXiv', isDownload: true }));
        });
        return results.filter(Boolean);
    } catch (e) { console.error('[arXiv]', e.message); return []; }
}

// npm
async function npm(kw) {
    try {
        const res = await ax({ method:'GET', url:'https://registry.npmjs.org/-/v1/search',
            params:{ text:kw, size:12 }, headers:{ 'User-Agent': ua() } });
        return (res.data?.objects||[]).map(p => {
            const pkg = p.package;
            return mkResult(pkg.name, `https://npmjs.com/package/${pkg.name}`,
                `📦 v${pkg.version} · ${Math.round((p.score?.detail?.popularity||0)*100)}% · ${pkg.description||''}`,
                { engine:'NPM', source:'npm' });
        }).filter(Boolean);
    } catch (e) { console.error('[npm]', e.message); return []; }
}

// PDF Drive — buscador de PDFs
async function pdfdrive(kw) {
    try {
        const res = await ax({ method:'GET', url:`https://www.pdfdrive.com/search`,
            params:{ q: kw }, headers:{ 'User-Agent': ua(), 'Accept': 'text/html' } });
        const $       = cheerio.load(res.data);
        const results = [];
        $('.file-left, .book-title, h2 a, .bookTitle').each((_, el) => {
            const title = $(el).text().trim();
            const href  = $(el).closest('a').attr('href') || $(el).find('a').attr('href') || '';
            const link  = href.startsWith('http') ? href : `https://www.pdfdrive.com${href}`;
            const r = mkResult(title, link, `📄 PDF Drive · Descarga PDF directa`, { engine:'PDFDrive', source:'PDF Drive', isDownload: true });
            if (r) results.push(r);
        });
        return results.slice(0, 15);
    } catch (e) { console.error('[PDFDrive]', e.message); return []; }
}

// Scribd via DDG
async function scribd(kw) {
    const results = await ddg(`site:scribd.com ${kw}`);
    return results.filter(r => r.url.includes('scribd.com'))
        .map(r => ({ ...r, source:'Scribd', engine:'Scribd' }));
}

// SlideShare via DDG
async function slideshare(kw) {
    const results = await ddg(`site:slideshare.net ${kw}`);
    return results.filter(r => r.url.includes('slideshare.net'))
        .map(r => ({ ...r, source:'SlideShare', engine:'SlideShare' }));
}

// Academia.edu via DDG
async function academia(kw) {
    const results = await ddg(`site:academia.edu ${kw} filetype:pdf OR download`);
    return results.filter(r => r.url.includes('academia.edu'))
        .map(r => ({ ...r, source:'Academia.edu', engine:'Academia', isDownload: true }));
}

// MediaFire — 8 queries distintas en paralelo
async function mediafire(kw) {
    const queries = [
        `site:mediafire.com/file ${kw}`,
        `site:mediafire.com ${kw}`,
        `"mediafire.com/file" "${kw}"`,
        `"mediafire.com/file" ${kw}`,
        `mediafire ${kw} download link`,
        `mediafire descargar ${kw}`,
        `mediafire ${kw} gratis`,
        `"mediafire" ${kw} -site:reddit.com -site:youtube.com`
    ];
    const calls = queries.map(q => Math.random() > 0.5 ? ddg(q) : bing(q));
    const all = (await Promise.allSettled(calls))
        .flatMap(r => r.status==='fulfilled' ? r.value : [])
        .filter(r => r?.url?.includes('mediafire.com'))
        .map(r => ({ ...r, source:'MediaFire', engine:'MediaFire', isDownload: true }));
    return all;
}

// Google Drive via DDG + Bing
async function googledrive(kw) {
    const queries = [
        `site:drive.google.com ${kw}`,
        `site:docs.google.com ${kw}`,
        `"drive.google.com/file" ${kw}`,
        `"drive.google.com/drive/folders" ${kw}`,
        `google drive "${kw}" compartido descarga`,
        `inurl:drive.google.com ${kw}`,
        `"docs.google.com" "${kw}"`,
        `"drive.google.com" "${kw}" view`
    ];
    const calls = [
        ddg(queries[0]), ddg(queries[1]), ddg(queries[2]), ddg(queries[3]),
        bing(queries[0]), bing(queries[1]), bing(queries[4]), bing(queries[7])
    ];
    const all = (await Promise.allSettled(calls))
        .flatMap(r => r.status==='fulfilled' ? r.value : [])
        .filter(r => r?.url && (r.url.includes('drive.google.com') || r.url.includes('docs.google.com')))
        .map(r => ({ ...r, source: r.url.includes('docs.google.com') ? 'Google Docs' : 'Google Drive', isDownload: true }));
    return all;
}

// MEGA links via DDG
async function mega(kw) {
    const results = await Promise.allSettled([
        ddg(`site:mega.nz ${kw}`),
        bing(`mega.nz ${kw} link`),
        ddg(`"mega.nz" "${kw}"`),
        ddg(`mega descargar ${kw}`)
    ]);
    return results.flatMap(r => r.status==='fulfilled' ? r.value : [])
        .filter(r => r?.url?.includes('mega.nz') || r?.url?.includes('mega.co.nz'))
        .map(r => ({ ...r, source:'MEGA', engine:'MEGA', isDownload: true }));
}

// ── INTELIGENCIA DE QUERIES ───────────────────────────────────────────────
// Detecta si el usuario busca descargas y amplifica el query
function analyzeIntent(kw) {
    const lc = kw.toLowerCase();
    const isDownload = /descargar?|download|gratis|free|pdf|epub|ebook|link|mega|mediafire|drive/i.test(lc);
    const isVideo    = /video|youtube|tutorial|como|how to|clase|curso/i.test(lc);
    const isCode     = /código|code|script|npm|library|librería|github|api/i.test(lc);
    const isBook     = /libro|book|manual|guía|guide|pdf|epub/i.test(lc);
    // Limpiar keywords de download para la búsqueda base
    const cleanKw = kw.replace(/\b(descargar?|download|gratis|free|pdf|epub|ebook|link)\b/gi,'').trim() || kw;
    return { isDownload, isVideo, isCode, isBook, cleanKw };
}

function expand(kw) {
    const { cleanKw } = analyzeIntent(kw);
    return {
        base:      kw,
        clean:     cleanKw,
        exact:     `"${cleanKw}"`,
        pdf:       `${cleanKw} filetype:pdf`,
        download:  `${cleanKw} download`,
        descargar: `${cleanKw} descargar`,
        gratis:    `${cleanKw} gratis free`,
        link:      `${cleanKw} link descarga`,
        mega:      `${cleanKw} mega.nz`,
        mfire:     `${cleanKw} mediafire`,
        drive:     `${cleanKw} drive.google.com`,
        archive:   `${cleanKw} archive.org`,
        tutorial:  `${cleanKw} tutorial`,
        y2024:     `${cleanKw} 2024`,
        y2025:     `${cleanKw} 2025`,
    };
}

// ── WORKERS POR FILTRO ────────────────────────────────────────────────────
function getWorkers(kw, filter) {
    const q    = expand(kw);
    const intent = analyzeIntent(kw);

    // ── ALL: 20 workers mixtos orientados a descarga ──────────────────────
    if (filter === 'all') return [
        () => ddg(kw, 1),
        () => ddg(kw, 2),
        () => ddg(q.pdf),
        () => ddg(q.download),
        () => bing(kw, 0),
        () => bing(kw, 10),
        () => bing(q.descargar),
        () => wikipedia(intent.clean),
        () => youtube(intent.isVideo ? kw : `${intent.clean} tutorial`),
        () => github(intent.clean),
        () => reddit(kw),
        () => stackoverflow(intent.clean),
        () => hackernews(intent.clean),
        () => archive(intent.clean),
        () => books(intent.clean),
        () => gutenberg(intent.clean),
        () => arxiv(intent.clean),
        () => pdfdrive(intent.clean),
        () => slideshare(intent.clean),
        () => scribd(intent.clean),
    ];

    // ── YOUTUBE ───────────────────────────────────────────────────────────
    if (filter === 'youtube') return [
        () => youtube(kw),
        () => youtube(`${intent.clean} tutorial`),
        () => ddg(`site:youtube.com ${kw}`),
        () => ddg(`site:youtube.com ${q.exact}`),
        () => ddg(`site:youtube.com ${q.tutorial}`),
        () => ddg(`site:youtube.com ${q.y2024}`),
        () => ddg(`site:youtube.com ${q.y2025}`),
        () => bing(`site:youtube.com ${kw}`),
        () => bing(`site:youtube.com ${q.tutorial}`),
        () => bing(`youtube.com/watch "${intent.clean}"`),
        () => ddg(`youtube playlist ${kw}`),
        () => ddg(`site:youtu.be ${intent.clean}`),
    ];

    // ── MEDIAFIRE ─────────────────────────────────────────────────────────
    if (filter === 'mediafire') return [
        () => mediafire(intent.clean),
        () => ddg(`site:mediafire.com/file ${intent.clean}`),
        () => ddg(`site:mediafire.com ${kw}`),
        () => ddg(`"mediafire.com/file" "${intent.clean}"`),
        () => bing(`site:mediafire.com ${intent.clean}`),
        () => bing(`mediafire "${intent.clean}" download`),
        () => ddg(`mediafire ${q.descargar}`),
        () => ddg(`mediafire ${q.gratis}`),
        () => bing(`"mediafire.com/file" ${intent.clean}`),
        () => ddg(`"download" "mediafire" ${intent.clean}`),
        () => bing(`filetype:zip OR filetype:rar OR filetype:pdf mediafire ${intent.clean}`),
        () => ddg(`mediafire link ${intent.clean} 2024`),
    ];

    // ── GOOGLE DRIVE ──────────────────────────────────────────────────────
    if (filter === 'google') return [
        () => googledrive(intent.clean),
        () => ddg(`site:drive.google.com ${kw}`),
        () => ddg(`site:docs.google.com ${kw}`),
        () => ddg(`"drive.google.com/file" ${intent.clean}`),
        () => ddg(`"drive.google.com/drive/folders" ${intent.clean}`),
        () => bing(`site:drive.google.com ${kw}`),
        () => bing(`"drive.google.com" "${intent.clean}" public`),
        () => ddg(`google drive "${intent.clean}" compartido`),
        () => bing(`site:docs.google.com "${intent.clean}"`),
        () => ddg(`inurl:drive.google.com "${intent.clean}"`),
        () => ddg(`${intent.clean} google drive link`),
        () => bing(`${intent.clean} "drive.google.com"`),
    ];

    // ── GITHUB ────────────────────────────────────────────────────────────
    if (filter === 'github') return [
        () => github(intent.clean),
        () => npm(intent.clean),
        () => ddg(`site:github.com ${kw}`),
        () => ddg(`site:github.com ${q.exact}`),
        () => bing(`site:github.com ${intent.clean}`),
        () => bing(`github ${intent.clean} stars`),
        () => hackernews(intent.clean),
        () => ddg(`site:gitlab.com ${kw}`),
        () => ddg(`site:github.com/releases ${intent.clean}`),
        () => bing(`github.com/releases/download ${intent.clean}`),
        () => ddg(`github "${intent.clean}" latest release`),
        () => ddg(`site:npmjs.com ${intent.clean}`),
    ];

    // ── REDDIT ────────────────────────────────────────────────────────────
    if (filter === 'reddit') return [
        () => reddit(kw),
        () => reddit(`${intent.clean} download`),
        () => ddg(`site:reddit.com ${kw}`),
        () => ddg(`site:reddit.com ${q.exact}`),
        () => ddg(`site:old.reddit.com ${kw}`),
        () => bing(`site:reddit.com ${kw}`),
        () => bing(`reddit ${kw} link`),
        () => ddg(`reddit "${intent.clean}" mega OR mediafire OR drive`),
        () => ddg(`site:reddit.com ${q.y2024}`),
        () => ddg(`site:reddit.com ${q.y2025}`),
        () => bing(`reddit.com ${intent.clean} download`),
        () => ddg(`site:reddit.com/r ${kw} "link"`),
    ];

    // ── STACK OVERFLOW ────────────────────────────────────────────────────
    if (filter === 'stackoverflow') return [
        () => stackoverflow(kw),
        () => ddg(`site:stackoverflow.com ${kw}`),
        () => ddg(`site:stackoverflow.com ${q.exact}`),
        () => bing(`site:stackoverflow.com ${kw}`),
        () => ddg(`site:stackexchange.com ${kw}`),
        () => hackernews(intent.clean),
        () => ddg(`stackoverflow ${kw} solution`),
        () => bing(`stackoverflow how to ${intent.clean}`),
        () => ddg(`site:superuser.com ${kw}`),
        () => ddg(`site:askubuntu.com ${kw}`),
        () => bing(`site:stackoverflow.com ${q.y2024}`),
        () => ddg(`${kw} solved answer`),
    ];

    // ── MEDIUM ────────────────────────────────────────────────────────────
    if (filter === 'medium') return [
        () => ddg(`site:medium.com ${kw}`),
        () => bing(`site:medium.com ${kw}`),
        () => ddg(`site:medium.com ${q.exact}`),
        () => bing(`medium.com "${intent.clean}" article`),
        () => ddg(`site:towardsdatascience.com ${kw}`),
        () => ddg(`medium ${q.tutorial}`),
        () => bing(`site:medium.com ${q.y2024}`),
        () => scribd(intent.clean),
        () => slideshare(intent.clean),
        () => academia(intent.clean),
        () => arxiv(intent.clean),
        () => pdfdrive(intent.clean),
    ];

    return getWorkers(kw, 'all');
}

// ── RELEVANCIA ────────────────────────────────────────────────────────────
function score(result, keyword) {
    const intent  = analyzeIntent(keyword);
    const terms   = intent.clean.toLowerCase().split(/\s+/).filter(t => t.length > 1);
    const title   = result.title.toLowerCase();
    const desc    = (result.description||'').toLowerCase();
    const url     = result.url.toLowerCase();
    let   pts     = 0;

    // Coincidencia de términos
    terms.forEach(t => {
        const tf = (title.match(new RegExp(t.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'),'g'))||[]).length;
        pts += tf * 30;
        if (title.startsWith(t)) pts += 20;
        if (title === intent.clean.toLowerCase()) pts += 50;
    });
    if (title.includes(intent.clean.toLowerCase())) pts += 40;

    // Descripción
    terms.forEach(t => {
        const df = (desc.match(new RegExp(t.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'),'g'))||[]).length;
        pts += Math.min(df,3) * 8;
    });

    // URL
    if (url.includes(intent.clean.toLowerCase().replace(/ /g,'-'))) pts += 15;
    terms.forEach(t => { if (url.includes(t)) pts += 5; });

    // BONUS por ser link de descarga real
    if (result.isDownload) pts += 30;
    const { isHost } = isDownloadLink(result);
    if (isHost) pts += 25;

    // Si el usuario busca descarga, bonus extra por hosts de descarga
    if (intent.isDownload) {
        if (isHost) pts += 20;
        if (url.includes('.pdf') || url.includes('filetype=pdf')) pts += 15;
    }

    // Bonus por fuente
    const src = (result.source||'').toLowerCase();
    const srcB = {
        'mediafire':20,'archive.org':18,'google drive':18,'google docs':15,
        'mega':20,'dropbox':15,'github':15,'wikipedia':22,'project gutenberg':20,
        'arxiv':18,'open library':16,'google books':14,'pdf drive':18,
        'stack overflow':18,'scribd':14,'slideshare':12,'academia.edu':14,
        'hacker news':12,'reddit':8,'youtube':10,'medium':10,'z-library':15,'libgen':15
    };
    Object.entries(srcB).forEach(([s,b]) => { if (src.includes(s)) pts += b; });

    // Penalizar sin descripción
    if (!result.description || result.description.length < 20) pts -= 10;

    return Math.max(0, Math.min(100, Math.round(Math.log1p(pts) / Math.log1p(350) * 100)));
}

// ── FILTRADO ──────────────────────────────────────────────────────────────
const SPAM = ['spam','scam','fake','virus','malware','phishing','free money','get rich','viagra','casino','xxx'];

function isValid(r) {
    if (!r?.title || r.title.length < 3) return false;
    if (!r?.url  || !r.url.startsWith('http')) return false;
    const txt = (r.title+' '+(r.description||'')).toLowerCase();
    return !SPAM.some(s => txt.includes(s));
}

function dedupe(arr) {
    const seen = new Set();
    return arr.filter(r => {
        try {
            const u = new URL(r.url);
            const k = `${u.hostname.replace(/^www\./,'')}${u.pathname}`.toLowerCase().replace(/\/$/,'').replace(/[?#].*/,'');
            if (seen.has(k)) return false;
            seen.add(k);
            return true;
        } catch { return false; }
    });
}

// ── MOTOR ─────────────────────────────────────────────────────────────────
async function search(keyword, filter) {
    const t0      = Date.now();
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
    results = results.map(r => ({ ...r, relevance: score(r, keyword) }));
    // Ordenar: primero links de descarga reales si hay intent de descarga
    const intent = analyzeIntent(keyword);
    results.sort((a, b) => {
        if (intent.isDownload) {
            const aIsHost = isDownloadLink(a).isHost ? 1 : 0;
            const bIsHost = isDownloadLink(b).isHost ? 1 : 0;
            if (aIsHost !== bIsHost) return bIsHost - aIsHost;
        }
        return b.relevance - a.relevance;
    });

    const elapsed = ((Date.now() - t0) / 1000).toFixed(2);
    console.log(`✅ ${elapsed}s | ${results.length} resultados\n`);
    return { results, stats: { totalResults: results.length, searchTime: elapsed, workersUsed: workers.length, timestamp: new Date().toISOString() } };
}

// ── RUTAS ─────────────────────────────────────────────────────────────────
app.get('/', (_, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));

app.get('/api/health', (_, res) => res.json({ status:'ok', uptime: process.uptime(), timestamp: new Date().toISOString(), version:'5.0' }));

const VALID_FILTERS = ['all','youtube','mediafire','google','github','reddit','stackoverflow','medium'];

app.post('/api/search', async (req, res) => {
    try {
        const { keyword, filter = 'all' } = req.body;
        if (!keyword?.trim()) return res.status(400).json({ error:'Keyword requerido' });
        if (!VALID_FILTERS.includes(filter)) return res.status(400).json({ error:'Filtro inválido' });
        const data = await search(keyword.trim(), filter);
        res.json({ success:true, ...data });
    } catch (e) {
        console.error('Error:', e);
        res.status(500).json({ success:false, error:'Error interno', message: e.message });
    }
});

app.get('/api/filters', (_, res) => res.json({
    filters: VALID_FILTERS,
    description: {
        all:           'General (20 workers · PDFs, videos, código, docs)',
        youtube:       'YouTube (12 workers)',
        mediafire:     'MediaFire (12 workers · links directos)',
        google:        'Google Drive/Docs (12 workers · archivos compartidos)',
        github:        'GitHub + npm (12 workers)',
        reddit:        'Reddit (12 workers · hilos y links)',
        stackoverflow: 'Stack Overflow (12 workers)',
        medium:        'Medium + PDFs + Scribd (12 workers)'
    }
}));

// ── AUTO-PING ─────────────────────────────────────────────────────────────
let pingInterval = null;
const startPing = () => {
    if (!CONFIG.SELF_PING_URL) return;
    const ping = async () => {
        try { const r = await ax({ method:'GET', url:`${CONFIG.SELF_PING_URL.replace(/\/$/,'')}/api/health`, timeout:5000, headers:{'User-Agent':'PingBot/1.0'} }); console.log(`✅ ping uptime:${Math.floor(r.data.uptime)}s`); }
        catch (e) { console.error('❌ ping:', e.message); }
    };
    ping(); pingInterval = setInterval(ping, CONFIG.SELF_PING_INTERVAL);
};
const stopPing = () => { if (pingInterval) { clearInterval(pingInterval); pingInterval = null; } };

app.post('/api/ping/start',  (_, res) => { if (!pingInterval) startPing(); res.json({ message:'ok' }); });
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
    console.log('║  🚀 BUSCADOR v5.0 — Motor de descarga inteligente       ║');
    console.log('╚══════════════════════════════════════════════════════════╝');
    console.log(`📡 Puerto: ${PORT} | Front: ${CONFIG.VAR_URL||'—'}`);
    console.log('🔍 DDG·Bing·YT·GH·Reddit·SO·HN·Archive·Books·Gutenberg·arXiv·PDFDrive·Scribd·Slideshare·Academia·MediaFire·GDrive·MEGA\n');
    startPing();
});
process.on('SIGTERM', () => { stopPing(); server.close(() => process.exit(0)); });
process.on('SIGINT',  () => { stopPing(); server.close(() => process.exit(0)); });
module.exports = app;
