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
    SELF_PING_URL: process.env.SELF_PING_URL || 'https://tu-app.com', // ⚠️ CAMBIAR ESTE URL
    SELF_PING_INTERVAL: 14 * 60 * 1000, // 14 minutos (para servicios gratuitos)
    MAX_RESULTS_PER_WORKER: 20,
    WORKERS_COUNT: 8,
    REQUEST_TIMEOUT: 10000,
    USER_AGENT: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
};

// ==================== MIDDLEWARE ====================
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Logging middleware
app.use((req, res, next) => {
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
    next();
});

// ==================== DATOS DE BÚSQUEDA ====================
const SEARCH_ENGINES = {
    google: {
        url: 'https://www.google.com/search',
        params: (keyword, site) => ({
            q: site ? `site:${site} ${keyword}` : keyword,
            num: 20
        }),
        selector: '.g',
        titleSelector: 'h3',
        linkSelector: 'a',
        descriptionSelector: '.VwiC3b'
    },
    bing: {
        url: 'https://www.bing.com/search',
        params: (keyword, site) => ({
            q: site ? `site:${site} ${keyword}` : keyword,
            count: 20
        }),
        selector: '.b_algo',
        titleSelector: 'h2',
        linkSelector: 'a',
        descriptionSelector: '.b_caption p'
    }
};

const SITE_FILTERS = {
    youtube: ['youtube.com', 'youtu.be'],
    mediafire: ['mediafire.com', 'www.mediafire.com'],
    google: ['drive.google.com', 'docs.google.com'],
    github: ['github.com', 'gist.github.com'],
    reddit: ['reddit.com', 'old.reddit.com'],
    stackoverflow: ['stackoverflow.com', 'stackexchange.com'],
    medium: ['medium.com', 'towardsdatascience.com'],
    all: null
};

const SPAM_KEYWORDS = [
    'spam', 'scam', 'fake', 'virus', 'malware', 'phishing',
    'click here now', 'download now', 'free money', 'get rich',
    'weight loss', 'viagra', 'casino', 'porn', 'xxx'
];

// ==================== FUNCIONES DE SCRAPING ====================

async function scrapeSearchEngine(keyword, site, engine = 'google') {
    const searchEngine = SEARCH_ENGINES[engine];
    const results = [];

    try {
        const params = searchEngine.params(keyword, site);
        const queryString = new URLSearchParams(params).toString();
        const url = `${searchEngine.url}?${queryString}`;

        const response = await axios.get(url, {
            headers: {
                'User-Agent': CONFIG.USER_AGENT,
                'Accept': 'text/html,application/xhtml+xml',
                'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
            },
            timeout: CONFIG.REQUEST_TIMEOUT
        });

        const $ = cheerio.load(response.data);

        $(searchEngine.selector).each((index, element) => {
            try {
                const title = $(element).find(searchEngine.titleSelector).text().trim();
                const link = $(element).find(searchEngine.linkSelector).attr('href');
                const description = $(element).find(searchEngine.descriptionSelector).text().trim();

                if (title && link) {
                    // Limpiar y validar URL
                    let cleanUrl = link;
                    if (link.startsWith('/url?q=')) {
                        const urlObj = new URL(link, searchEngine.url);
                        cleanUrl = urlObj.searchParams.get('q') || link;
                    }

                    // Validar que sea una URL real
                    try {
                        new URL(cleanUrl);
                        
                        results.push({
                            title: title,
                            url: cleanUrl,
                            description: description || 'Sin descripción disponible',
                            source: extractSourceName(cleanUrl),
                            engine: engine
                        });
                    } catch (e) {
                        // URL inválida, ignorar
                    }
                }
            } catch (error) {
                console.error('Error procesando elemento:', error.message);
            }
        });

    } catch (error) {
        console.error(`Error en scraping (${engine}):`, error.message);
    }

    return results;
}

async function searchYouTube(keyword) {
    const results = [];
    try {
        const url = `https://www.youtube.com/results?search_query=${encodeURIComponent(keyword)}`;
        const response = await axios.get(url, {
            headers: { 'User-Agent': CONFIG.USER_AGENT },
            timeout: CONFIG.REQUEST_TIMEOUT
        });

        const $ = cheerio.load(response.data);
        
        // Buscar datos en scripts de YouTube
        const scripts = $('script').toArray();
        for (const script of scripts) {
            const content = $(script).html();
            if (content && content.includes('var ytInitialData')) {
                try {
                    const jsonMatch = content.match(/var ytInitialData = ({.+?});/);
                    if (jsonMatch) {
                        const data = JSON.parse(jsonMatch[1]);
                        const contents = data?.contents?.twoColumnSearchResultsRenderer?.primaryContents?.sectionListRenderer?.contents;
                        
                        if (contents) {
                            contents.forEach(section => {
                                const items = section?.itemSectionRenderer?.contents || [];
                                items.forEach(item => {
                                    const video = item?.videoRenderer;
                                    if (video) {
                                        results.push({
                                            title: video.title?.runs?.[0]?.text || 'Sin título',
                                            url: `https://www.youtube.com/watch?v=${video.videoId}`,
                                            description: video.descriptionSnippet?.runs?.map(r => r.text).join('') || 'Sin descripción',
                                            source: 'YouTube',
                                            thumbnail: video.thumbnail?.thumbnails?.[0]?.url
                                        });
                                    }
                                });
                            });
                        }
                    }
                } catch (e) {
                    console.error('Error parseando datos de YouTube:', e.message);
                }
                break;
            }
        }
    } catch (error) {
        console.error('Error buscando en YouTube:', error.message);
    }
    return results;
}

async function searchGitHub(keyword) {
    const results = [];
    try {
        const response = await axios.get(`https://api.github.com/search/repositories`, {
            params: {
                q: keyword,
                sort: 'stars',
                per_page: 20
            },
            headers: {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': CONFIG.USER_AGENT
            },
            timeout: CONFIG.REQUEST_TIMEOUT
        });

        response.data.items.forEach(repo => {
            results.push({
                title: repo.full_name,
                url: repo.html_url,
                description: repo.description || 'Sin descripción',
                source: 'GitHub',
                stars: repo.stargazers_count,
                language: repo.language
            });
        });
    } catch (error) {
        console.error('Error buscando en GitHub:', error.message);
    }
    return results;
}

async function searchReddit(keyword) {
    const results = [];
    try {
        const response = await axios.get(`https://www.reddit.com/search.json`, {
            params: {
                q: keyword,
                limit: 20,
                sort: 'relevance'
            },
            headers: {
                'User-Agent': CONFIG.USER_AGENT
            },
            timeout: CONFIG.REQUEST_TIMEOUT
        });

        response.data.data.children.forEach(post => {
            const data = post.data;
            results.push({
                title: data.title,
                url: `https://www.reddit.com${data.permalink}`,
                description: data.selftext ? data.selftext.substring(0, 200) : 'Ver post completo',
                source: 'Reddit',
                subreddit: data.subreddit,
                score: data.score
            });
        });
    } catch (error) {
        console.error('Error buscando en Reddit:', error.message);
    }
    return results;
}

// ==================== FILTRADO Y VALIDACIÓN ====================

function isSpamContent(text) {
    const lowerText = text.toLowerCase();
    return SPAM_KEYWORDS.some(spam => lowerText.includes(spam));
}

function isValidResult(result) {
    if (!result.title || result.title.length < 5 || result.title.length > 200) return false;
    if (!result.url || !result.url.startsWith('http')) return false;
    if (isSpamContent(result.title + ' ' + result.description)) return false;
    return true;
}

function calculateRelevance(result, keyword) {
    const searchTerms = keyword.toLowerCase().split(' ');
    const title = result.title.toLowerCase();
    const description = (result.description || '').toLowerCase();
    
    let score = 0;
    
    // Puntos por coincidencias en el título
    searchTerms.forEach(term => {
        if (title.includes(term)) score += 30;
    });
    
    // Puntos por coincidencias en la descripción
    searchTerms.forEach(term => {
        if (description.includes(term)) score += 10;
    });
    
    // Bonus por fuentes confiables
    const trustedSources = ['github', 'stackoverflow', 'medium', 'youtube'];
    if (trustedSources.some(source => result.source.toLowerCase().includes(source))) {
        score += 20;
    }
    
    // Limitar a 100
    return Math.min(100, score);
}

function extractSourceName(url) {
    try {
        const urlObj = new URL(url);
        const hostname = urlObj.hostname.replace('www.', '');
        
        const sourceMap = {
            'youtube.com': 'YouTube',
            'youtu.be': 'YouTube',
            'mediafire.com': 'MediaFire',
            'drive.google.com': 'Google Drive',
            'docs.google.com': 'Google Docs',
            'github.com': 'GitHub',
            'gist.github.com': 'GitHub Gist',
            'reddit.com': 'Reddit',
            'stackoverflow.com': 'Stack Overflow',
            'stackexchange.com': 'Stack Exchange',
            'medium.com': 'Medium',
            'towardsdatascience.com': 'Towards Data Science'
        };
        
        return sourceMap[hostname] || hostname;
    } catch (e) {
        return 'Desconocido';
    }
}

function removeDuplicates(results) {
    const seen = new Set();
    return results.filter(result => {
        const key = result.url.toLowerCase();
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });
}

// ==================== WORKER SYSTEM ====================

async function workerSearch(workerId, keyword, filter) {
    const results = [];
    const sites = SITE_FILTERS[filter];
    
    console.log(`[Worker ${workerId}] Iniciando búsqueda: ${keyword} | Filtro: ${filter}`);
    
    try {
        // Estrategia de búsqueda según el filtro
        if (filter === 'youtube') {
            const youtubeResults = await searchYouTube(keyword);
            results.push(...youtubeResults);
        } else if (filter === 'github') {
            const githubResults = await searchGitHub(keyword);
            results.push(...githubResults);
        } else if (filter === 'reddit') {
            const redditResults = await searchReddit(keyword);
            results.push(...redditResults);
        } else {
            // Búsqueda general o con filtro de sitio
            const site = sites && sites.length > 0 ? sites[Math.floor(Math.random() * sites.length)] : null;
            
            // Alternar entre Google y Bing
            const engine = workerId % 2 === 0 ? 'google' : 'bing';
            const searchResults = await scrapeSearchEngine(keyword, site, engine);
            results.push(...searchResults);
        }
        
        console.log(`[Worker ${workerId}] Encontrados: ${results.length} resultados`);
    } catch (error) {
        console.error(`[Worker ${workerId}] Error:`, error.message);
    }
    
    return results.map(result => ({
        ...result,
        workerId: workerId,
        timestamp: Date.now()
    }));
}

async function multiWorkerSearch(keyword, filter) {
    const startTime = Date.now();
    
    // Crear promesas para todos los workers
    const workerPromises = [];
    for (let i = 1; i <= CONFIG.WORKERS_COUNT; i++) {
        workerPromises.push(
            workerSearch(i, keyword, filter)
                .catch(error => {
                    console.error(`Worker ${i} falló:`, error.message);
                    return [];
                })
        );
    }
    
    // Ejecutar todos los workers en paralelo
    const allResults = await Promise.all(workerPromises);
    
    // Combinar todos los resultados
    let combinedResults = allResults.flat();
    
    // Filtrar resultados inválidos
    combinedResults = combinedResults.filter(isValidResult);
    
    // Eliminar duplicados
    combinedResults = removeDuplicates(combinedResults);
    
    // Calcular relevancia
    combinedResults = combinedResults.map(result => ({
        ...result,
        relevance: calculateRelevance(result, keyword)
    }));
    
    // Ordenar por relevancia
    combinedResults.sort((a, b) => b.relevance - a.relevance);
    
    const endTime = Date.now();
    const searchTime = ((endTime - startTime) / 1000).toFixed(2);
    
    console.log(`Búsqueda completada en ${searchTime}s | Total: ${combinedResults.length} resultados`);
    
    return {
        results: combinedResults,
        stats: {
            totalResults: combinedResults.length,
            searchTime: searchTime,
            workersUsed: CONFIG.WORKERS_COUNT,
            timestamp: new Date().toISOString()
        }
    };
}

// ==================== RUTAS DE LA API ====================

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/api/health', (req, res) => {
    res.json({
        status: 'ok',
        uptime: process.uptime(),
        timestamp: new Date().toISOString(),
        workers: CONFIG.WORKERS_COUNT
    });
});

app.post('/api/search', async (req, res) => {
    try {
        const { keyword, filter = 'all' } = req.body;
        
        if (!keyword || keyword.trim().length === 0) {
            return res.status(400).json({
                error: 'Keyword es requerido'
            });
        }
        
        if (!SITE_FILTERS.hasOwnProperty(filter)) {
            return res.status(400).json({
                error: 'Filtro inválido'
            });
        }
        
        console.log(`\n=== Nueva búsqueda ===`);
        console.log(`Keyword: ${keyword}`);
        console.log(`Filter: ${filter}`);
        
        const searchResults = await multiWorkerSearch(keyword, filter);
        
        res.json({
            success: true,
            ...searchResults
        });
        
    } catch (error) {
        console.error('Error en búsqueda:', error);
        res.status(500).json({
            success: false,
            error: 'Error interno del servidor',
            message: error.message
        });
    }
});

app.get('/api/filters', (req, res) => {
    res.json({
        filters: Object.keys(SITE_FILTERS),
        description: {
            all: 'Buscar en todos los sitios',
            youtube: 'Buscar solo en YouTube',
            mediafire: 'Buscar solo en MediaFire',
            google: 'Buscar en Google Drive/Docs',
            github: 'Buscar en GitHub',
            reddit: 'Buscar en Reddit',
            stackoverflow: 'Buscar en Stack Overflow',
            medium: 'Buscar en Medium'
        }
    });
});

// ==================== SISTEMA DE AUTO-PING ====================

let pingInterval = null;

function startSelfPing() {
    if (!CONFIG.SELF_PING_URL || CONFIG.SELF_PING_URL === 'https://tu-app.com') {
        console.log('⚠️  Auto-ping deshabilitado: SELF_PING_URL no configurado');
        return;
    }
    
    console.log(`🔄 Auto-ping habilitado: ${CONFIG.SELF_PING_URL}`);
    console.log(`⏱️  Intervalo: ${CONFIG.SELF_PING_INTERVAL / 60000} minutos`);
    
    // Hacer ping inmediatamente al iniciar
    performSelfPing();
    
    // Configurar ping periódico
    pingInterval = setInterval(() => {
        performSelfPing();
    }, CONFIG.SELF_PING_INTERVAL);
}

async function performSelfPing() {
    try {
        const pingUrl = `${CONFIG.SELF_PING_URL}/api/health`;
        console.log(`[${new Date().toISOString()}] 🏓 Realizando auto-ping...`);
        
        const response = await axios.get(pingUrl, {
            timeout: 5000,
            headers: {
                'User-Agent': 'SelfPingBot/1.0'
            }
        });
        
        console.log(`✅ Auto-ping exitoso | Status: ${response.data.status} | Uptime: ${Math.floor(response.data.uptime)}s`);
    } catch (error) {
        console.error(`❌ Auto-ping falló:`, error.message);
    }
}

function stopSelfPing() {
    if (pingInterval) {
        clearInterval(pingInterval);
        pingInterval = null;
        console.log('🛑 Auto-ping detenido');
    }
}

// Endpoint para controlar el auto-ping manualmente
app.post('/api/ping/start', (req, res) => {
    if (pingInterval) {
        return res.json({ message: 'Auto-ping ya está activo' });
    }
    startSelfPing();
    res.json({ message: 'Auto-ping iniciado', interval: CONFIG.SELF_PING_INTERVAL });
});

app.post('/api/ping/stop', (req, res) => {
    stopSelfPing();
    res.json({ message: 'Auto-ping detenido' });
});

app.get('/api/ping/status', (req, res) => {
    res.json({
        active: pingInterval !== null,
        url: CONFIG.SELF_PING_URL,
        interval: CONFIG.SELF_PING_INTERVAL,
        intervalMinutes: CONFIG.SELF_PING_INTERVAL / 60000
    });
});

// ==================== MANEJO DE ERRORES ====================

app.use((req, res) => {
    res.status(404).json({
        error: 'Endpoint no encontrado',
        path: req.path
    });
});

app.use((error, req, res, next) => {
    console.error('Error no manejado:', error);
    res.status(500).json({
        error: 'Error interno del servidor',
        message: error.message
    });
});

// ==================== INICIO DEL SERVIDOR ====================

const server = app.listen(PORT, () => {
    console.log('\n╔═══════════════════════════════════════════════════════╗');
    console.log('║     🚀 SERVIDOR DE BÚSQUEDA AVANZADA INICIADO       ║');
    console.log('╚═══════════════════════════════════════════════════════╝\n');
    console.log(`📡 Puerto: ${PORT}`);
    console.log(`🌐 URL: http://localhost:${PORT}`);
    console.log(`⚙️  Workers: ${CONFIG.WORKERS_COUNT}`);
    console.log(`🔍 Motores: Google, Bing, YouTube, GitHub, Reddit`);
    console.log('\n📚 Endpoints disponibles:');
    console.log('   GET  /api/health         - Estado del servidor');
    console.log('   POST /api/search         - Realizar búsqueda');
    console.log('   GET  /api/filters        - Lista de filtros');
    console.log('   GET  /api/ping/status    - Estado del auto-ping');
    console.log('   POST /api/ping/start     - Iniciar auto-ping');
    console.log('   POST /api/ping/stop      - Detener auto-ping');
    console.log('\n' + '─'.repeat(55) + '\n');
    
    // Iniciar auto-ping si está configurado
    startSelfPing();
});

// Manejo de cierre graceful
process.on('SIGTERM', () => {
    console.log('\n🛑 Señal SIGTERM recibida, cerrando servidor...');
    stopSelfPing();
    server.close(() => {
        console.log('✅ Servidor cerrado correctamente');
        process.exit(0);
    });
});

process.on('SIGINT', () => {
    console.log('\n🛑 Señal SIGINT recibida, cerrando servidor...');
    stopSelfPing();
    server.close(() => {
        console.log('✅ Servidor cerrado correctamente');
        process.exit(0);
    });
});

module.exports = app;
