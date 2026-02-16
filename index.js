require('dotenv').config();
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const cheerio = require('cheerio');
const { URL } = require('url');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs').promises;
const MongoClient = require('mongodb').MongoClient;
const Redis = require('redis');

const app = express();
const PORT = process.env.PORT || 3000;

// ══════════════════════════════════════════════════════════════════════════
// 🗄️ DATABASE CONNECTION
// ══════════════════════════════════════════════════════════════════════════

let db = null;
let redisClient = null;

async function connectDatabases() {
    try {
        // MongoDB
        if (process.env.MONGODB_URI) {
            const client = await MongoClient.connect(process.env.MONGODB_URI, {
                useNewUrlParser: true,
                useUnifiedTopology: true
            });
            db = client.db(process.env.MONGODB_DB_NAME || 'nexus_search');
            console.log('✅ MongoDB conectado');
            
            // Crear índices
            await db.collection('searches').createIndex({ query: 1 });
            await db.collection('searches').createIndex({ timestamp: -1 });
            await db.collection('clicks').createIndex({ query: 1, url: 1 });
        }

        // Redis
        if (process.env.REDIS_HOST) {
            redisClient = Redis.createClient({
                host: process.env.REDIS_HOST,
                port: process.env.REDIS_PORT || 6379,
                password: process.env.REDIS_PASSWORD
            });
            
            redisClient.on('error', (err) => console.error('Redis Error:', err));
            await redisClient.connect();
            console.log('✅ Redis conectado');
        }
    } catch (error) {
        console.warn('⚠️  Base de datos no disponible:', error.message);
    }
}

// ══════════════════════════════════════════════════════════════════════════
// 🧠 NEXUS AI BRAIN - SISTEMA NEURONAL MEJORADO
// ══════════════════════════════════════════════════════════════════════════

class NexusAIBrain {
    constructor() {
        this.ready = false;
        this.stats = { 
            queries: 0, 
            learned: 0, 
            avgResponseTime: 0,
            totalImprovement: 0 
        };
        this.learningData = new Map();
        this.initBrain();
    }

    async initBrain() {
        try {
            // Verificar módulos de IA
            const deepLearningPath = path.join(__dirname, 'neural', 'deep_learning.py');
            const rankingPath = path.join(__dirname, 'neural', 'ranking_engine.py');
            
            await fs.access(deepLearningPath);
            await fs.access(rankingPath);
            
            // Inicializar modelo si no existe
            const modelPath = path.join(__dirname, 'models', 'nexus_brain.pkl');
            try {
                await fs.access(modelPath);
            } catch {
                console.log('🔨 Inicializando modelo de IA...');
                await this.trainInitialModel();
            }
            
            this.ready = true;
            console.log('🧠 NEXUS AI Brain: ONLINE (50 neuronas activas)');
        } catch (e) {
            console.warn('⚠️  AI Brain no disponible:', e.message);
        }
    }

    async trainInitialModel() {
        return new Promise((resolve, reject) => {
            const python = spawn('python3', [
                path.join(__dirname, 'neural', 'deep_learning.py')
            ]);

            python.stdout.on('data', (data) => {
                console.log(`🤖 ${data.toString().trim()}`);
            });

            python.on('close', (code) => {
                if (code === 0) {
                    console.log('✅ Modelo de IA inicializado');
                    resolve();
                } else {
                    reject(new Error(`Proceso terminó con código ${code}`));
                }
            });
        });
    }

    async rankResults(query, results, userId = null) {
        if (!this.ready || !results.length) return results;

        const startTime = Date.now();

        return new Promise((resolve) => {
            try {
                const python = spawn('python3', ['-c', `
import sys
import json
sys.path.insert(0, '${__dirname}/neural')
from ranking_engine import RankingEngine
from deep_learning import NexusAI

# Cargar engines
ranking = RankingEngine()
try:
    ranking.load('${__dirname}/data/ranking_state.json')
except:
    pass

nexus_ai = NexusAI('${__dirname}/models/nexus_brain.pkl')

# Parsear datos
query = sys.stdin.readline().strip()
results = json.loads(sys.stdin.readline())
user_id = sys.stdin.readline().strip() or None

# Ranking con IA
ranked_by_engine = ranking.rank_results(query, results)
final_results = nexus_ai.get_personalized_results(query, ranked_by_engine, user_id)

# Guardar estado
ranking.save('${__dirname}/data/ranking_state.json')

print(json.dumps(final_results))
                `]);

                let output = '';
                let errorOutput = '';

                // Enviar datos al proceso Python
                python.stdin.write(query + '\n');
                python.stdin.write(JSON.stringify(results) + '\n');
                python.stdin.write((userId || '') + '\n');
                python.stdin.end();

                python.stdout.on('data', (data) => {
                    output += data.toString();
                });

                python.stderr.on('data', (data) => {
                    errorOutput += data.toString();
                });

                python.on('close', (code) => {
                    const responseTime = Date.now() - startTime;
                    this.stats.avgResponseTime = 
                        (this.stats.avgResponseTime * this.stats.queries + responseTime) / 
                        (this.stats.queries + 1);
                    this.stats.queries++;

                    if (code === 0 && output.trim()) {
                        try {
                            const rankedResults = JSON.parse(output.trim());
                            resolve(rankedResults);
                        } catch (e) {
                            console.error('Error parsing AI output:', e);
                            resolve(results);
                        }
                    } else {
                        if (errorOutput) console.error('AI Error:', errorOutput);
                        resolve(results);
                    }
                });

                // Timeout de 5 segundos
                setTimeout(() => {
                    python.kill();
                    resolve(results);
                }, 5000);

            } catch (error) {
                console.error('Error en ranking AI:', error);
                resolve(results);
            }
        });
    }

    async learnFromClick(query, url, position, dwellTime = null, bounced = false) {
        if (!this.ready) return;

        try {
            const python = spawn('python3', ['-c', `
import sys
import json
sys.path.insert(0, '${__dirname}/neural')
from ranking_engine import RankingEngine
from deep_learning import NexusAI

ranking = RankingEngine()
try:
    ranking.load('${__dirname}/data/ranking_state.json')
except:
    pass

nexus_ai = NexusAI('${__dirname}/models/nexus_brain.pkl')

data = json.loads(sys.stdin.readline())
ranking.record_click(
    data['query'], 
    data['url'], 
    data['position'],
    data['dwellTime'],
    data['bounced']
)

nexus_ai.learn_from_click(
    data['query'],
    {'url': data['url'], 'title': ''},
    data['position']
)

ranking.save('${__dirname}/data/ranking_state.json')
nexus_ai.save_model()

print("OK")
            `]);

            python.stdin.write(JSON.stringify({
                query, url, position, dwellTime, bounced
            }) + '\n');
            python.stdin.end();

            python.on('close', () => {
                this.stats.learned++;
            });
        } catch (error) {
            console.error('Error en aprendizaje:', error);
        }
    }

    getStats() {
        return {
            ...this.stats,
            ready: this.ready,
            uptime: process.uptime()
        };
    }
}

const nexusAI = new NexusAIBrain();

// ══════════════════════════════════════════════════════════════════════════
// 📡 MIDDLEWARE
// ══════════════════════════════════════════════════════════════════════════

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Logging middleware
app.use((req, res, next) => {
    const start = Date.now();
    res.on('finish', () => {
        const duration = Date.now() - start;
        console.log(`${req.method} ${req.path} - ${res.statusCode} (${duration}ms)`);
    });
    next();
});

// ══════════════════════════════════════════════════════════════════════════
// 🔍 SEARCH FUNCTIONS
// ══════════════════════════════════════════════════════════════════════════

const SOURCE_MAP = {
    'youtube.com': 'YouTube', 'youtu.be': 'YouTube',
    'github.com': 'GitHub', 'stackoverflow.com': 'Stack Overflow',
    'reddit.com': 'Reddit', 'medium.com': 'Medium',
    'wikipedia.org': 'Wikipedia', 'twitter.com': 'Twitter',
    'linkedin.com': 'LinkedIn', 'dev.to': 'DEV.to'
};

function getSourceName(url) {
    try {
        const hostname = new URL(url).hostname.replace(/^www\./, '');
        return SOURCE_MAP[hostname] || hostname;
    } catch {
        return 'Web';
    }
}

async function searchGoogle(query) {
    if (!process.env.GOOGLE_API_KEY || !process.env.GOOGLE_CX) {
        return [];
    }

    try {
        const response = await axios.get('https://www.googleapis.com/customsearch/v1', {
            params: {
                key: process.env.GOOGLE_API_KEY,
                cx: process.env.GOOGLE_CX,
                q: query,
                num: 10
            },
            timeout: 5000
        });

        return (response.data.items || []).map(item => ({
            title: item.title,
            url: item.link,
            description: item.snippet,
            source: getSourceName(item.link),
            score: 0.8,
            provider: 'google'
        }));
    } catch (error) {
        console.error('Error en Google Search:', error.message);
        return [];
    }
}

async function searchDuckDuckGo(query) {
    try {
        const response = await axios.get('https://api.duckduckgo.com/', {
            params: {
                q: query,
                format: 'json',
                no_html: 1
            },
            timeout: 5000
        });

        const results = [];
        
        // Abstract
        if (response.data.Abstract) {
            results.push({
                title: response.data.Heading || query,
                url: response.data.AbstractURL,
                description: response.data.Abstract,
                source: 'DuckDuckGo',
                score: 0.9,
                provider: 'duckduckgo'
            });
        }

        // Related Topics
        (response.data.RelatedTopics || []).forEach(topic => {
            if (topic.FirstURL) {
                results.push({
                    title: topic.Text?.split(' - ')[0] || '',
                    url: topic.FirstURL,
                    description: topic.Text || '',
                    source: getSourceName(topic.FirstURL),
                    score: 0.7,
                    provider: 'duckduckgo'
                });
            }
        });

        return results;
    } catch (error) {
        console.error('Error en DuckDuckGo:', error.message);
        return [];
    }
}

async function scrapeWebResults(query) {
    // Búsqueda básica scraping HTML (Bing, etc.)
    try {
        const searchUrl = `https://www.bing.com/search?q=${encodeURIComponent(query)}`;
        const response = await axios.get(searchUrl, {
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            },
            timeout: 5000
        });

        const $ = cheerio.load(response.data);
        const results = [];

        $('.b_algo').each((i, elem) => {
            const titleElem = $(elem).find('h2 a');
            const descElem = $(elem).find('.b_caption p');
            
            const title = titleElem.text();
            const url = titleElem.attr('href');
            const description = descElem.text();

            if (title && url) {
                results.push({
                    title,
                    url,
                    description,
                    source: getSourceName(url),
                    score: 0.6,
                    provider: 'bing'
                });
            }
        });

        return results.slice(0, 10);
    } catch (error) {
        console.error('Error en scraping:', error.message);
        return [];
    }
}

// ══════════════════════════════════════════════════════════════════════════
// 🛣️ ROUTES
// ══════════════════════════════════════════════════════════════════════════

// Search endpoint
app.get('/api/search', async (req, res) => {
    try {
        const { q: query, uid: userId } = req.query;

        if (!query || query.trim().length === 0) {
            return res.status(400).json({ error: 'Query requerido' });
        }

        console.log(`🔍 Buscando: "${query}"`);

        // Verificar caché
        let cachedResults = null;
        if (redisClient) {
            const cacheKey = `search:${query.toLowerCase()}`;
            cachedResults = await redisClient.get(cacheKey);
            
            if (cachedResults) {
                console.log('📦 Resultados desde caché');
                return res.json(JSON.parse(cachedResults));
            }
        }

        // Búsqueda en paralelo desde múltiples fuentes
        const [googleResults, duckResults, scrapedResults] = await Promise.all([
            searchGoogle(query),
            searchDuckDuckGo(query),
            scrapeWebResults(query)
        ]);

        // Combinar y deduplicar resultados
        const allResults = [...googleResults, ...duckResults, ...scrapedResults];
        const uniqueResults = Array.from(
            new Map(allResults.map(r => [r.url, r])).values()
        );

        // Ranking con IA
        const rankedResults = await nexusAI.rankResults(query, uniqueResults, userId);

        const response = {
            query,
            total: rankedResults.length,
            results: rankedResults.slice(0, 50),
            timestamp: new Date().toISOString(),
            ai_powered: nexusAI.ready
        };

        // Guardar en caché
        if (redisClient) {
            const cacheKey = `search:${query.toLowerCase()}`;
            await redisClient.setEx(cacheKey, 3600, JSON.stringify(response)); // 1 hora
        }

        // Guardar búsqueda en DB
        if (db) {
            await db.collection('searches').insertOne({
                query,
                userId,
                timestamp: new Date(),
                resultsCount: rankedResults.length
            });
        }

        res.json(response);

    } catch (error) {
        console.error('Error en búsqueda:', error);
        res.status(500).json({ 
            error: 'Error en búsqueda',
            message: error.message 
        });
    }
});

// Click tracking endpoint
app.post('/api/click', async (req, res) => {
    try {
        const { query, url, position, dwellTime, bounced } = req.body;

        if (!query || !url || position === undefined) {
            return res.status(400).json({ error: 'Datos incompletos' });
        }

        // Aprender del click
        await nexusAI.learnFromClick(query, url, position, dwellTime, bounced);

        // Guardar en DB
        if (db) {
            await db.collection('clicks').insertOne({
                query,
                url,
                position,
                dwellTime,
                bounced,
                timestamp: new Date()
            });
        }

        res.json({ success: true });
    } catch (error) {
        console.error('Error registrando click:', error);
        res.status(500).json({ error: 'Error registrando click' });
    }
});

// Stats endpoint
app.get('/api/stats', async (req, res) => {
    try {
        const aiStats = nexusAI.getStats();
        
        let dbStats = {};
        if (db) {
            const [searchCount, clickCount] = await Promise.all([
                db.collection('searches').countDocuments(),
                db.collection('clicks').countDocuments()
            ]);
            dbStats = { totalSearches: searchCount, totalClicks: clickCount };
        }

        res.json({
            ai: aiStats,
            database: dbStats,
            server: {
                uptime: process.uptime(),
                memory: process.memoryUsage(),
                platform: process.platform
            }
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Health check
app.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        ai: nexusAI.ready,
        database: db !== null,
        cache: redisClient !== null,
        timestamp: new Date().toISOString()
    });
});

// ══════════════════════════════════════════════════════════════════════════
// 🚀 START SERVER
// ══════════════════════════════════════════════════════════════════════════

async function startServer() {
    await connectDatabases();
    
    // Asegurar que existen las carpetas necesarias
    await fs.mkdir(path.join(__dirname, 'models'), { recursive: true });
    await fs.mkdir(path.join(__dirname, 'data'), { recursive: true });
    await fs.mkdir(path.join(__dirname, 'logs'), { recursive: true });

    app.listen(PORT, () => {
        console.log(`
╔═══════════════════════════════════════════════════════════════╗
║                                                                 ║
║   🚀  NEXUS SEARCH ENGINE - POWERED BY AI                      ║
║                                                                 ║
║   🌐  Servidor: http://localhost:${PORT}                        ║
║   🧠  IA: ${nexusAI.ready ? 'ACTIVA (50 neuronas)' : 'DESACTIVADA'}              ║
║   💾  MongoDB: ${db ? 'CONECTADO' : 'DESCONECTADO'}                           ║
║   ⚡  Redis: ${redisClient ? 'CONECTADO' : 'DESCONECTADO'}                      ║
║                                                                 ║
║   📊  API Endpoints:                                            ║
║      GET  /api/search?q=query                                  ║
║      POST /api/click                                           ║
║      GET  /api/stats                                           ║
║      GET  /health                                              ║
║                                                                 ║
╚═══════════════════════════════════════════════════════════════╝
        `);
    });
}

// Manejo de errores no capturados
process.on('uncaughtException', (error) => {
    console.error('Error no capturado:', error);
});

process.on('unhandledRejection', (error) => {
    console.error('Promesa rechazada:', error);
});

// Inicio
startServer();

module.exports = app;
