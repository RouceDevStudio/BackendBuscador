require('dotenv').config();
const express     = require('express');
const cors        = require('cors');
const axios       = require('axios');
const cheerio     = require('cheerio');
const path        = require('path');
const { spawn }   = require('child_process');
const fs          = require('fs').promises;
const MongoClient = require('mongodb').MongoClient;

const app  = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '2mb' }));
app.use(express.static('public'));
// Dar tiempo suficiente para que Ollama genere — calidad > velocidad
app.use((req, res, next) => { res.setTimeout(130000); next(); });

// ══════════════════════════════════════════════════════════════════
//  BASE DE DATOS
// ══════════════════════════════════════════════════════════════════

let db = null;

async function connectDB() {
    if (!process.env.MONGODB_URI) return;
    try {
        const client = await MongoClient.connect(process.env.MONGODB_URI, { serverSelectionTimeoutMS: 10000 });
        db = client.db(process.env.MONGODB_DB_NAME || 'nexus');
        await Promise.all([
            db.collection('messages').createIndex({ conversationId: 1, ts: 1 }),
            db.collection('clicks').createIndex({ query: 1, ts: -1 }),
            db.collection('searches').createIndex({ ts: -1 }),
        ]);
        console.log('✅ MongoDB conectado');
    } catch (e) {
        console.warn('⚠️  MongoDB no disponible:', e.message);
    }
}

// ══════════════════════════════════════════════════════════════════
//  PROCESO PYTHON PERSISTENTE (el cerebro real)
// ══════════════════════════════════════════════════════════════════

class BrainProcess {
    constructor() {
        this.proc          = null;
        this.queue         = [];
        this.ready         = false;
        this.restarts      = 0;
        this.stats         = {};
        this.requestCounter = 0;
        this.lastOllamaError = 0;
        this.ollamaErrorCount = 0;
        // ── Cache de stats para no bloquear el brain mientras Ollama piensa ──
        this._cachedStats    = null;   // última respuesta de stats conocida
        this._statsUpdating  = false;  // evitar peticiones simultáneas de stats
        this._start();
    }

    _start() {
        const brainPath = path.join(__dirname, 'neural', 'brain.py');
        const env = { ...process.env, PYTHONUNBUFFERED: '1' };

        console.log('🧠 Iniciando cerebro NEXUS...');
        this.proc = spawn('python3', ['-u', brainPath], { env });

        let buffer = '';

        this.proc.stdout.on('data', (chunk) => {
            buffer += chunk.toString();
            const parts = buffer.split('\n');
            buffer = parts.pop(); // último fragmento incompleto

            for (const part of parts) {
                const line = part.trim();
                if (!line) continue;

                // Mensajes de log del brain (no JSON)
                if (line.startsWith('✓') || line.startsWith('⚠') || line.startsWith('[')) {
                    console.log('🐍', line);
                    if (line.includes('listo') || line.includes('ready')) {
                        this.ready = true;
                    }
                    continue;
                }

                // Respuesta JSON para la cola
                try {
                    const response = JSON.parse(line);
                    
                    // ✅ FIX #4: Correlación por requestId
                    const requestId = response._requestId;
                    if (requestId) {
                        const idx = this.queue.findIndex(p => p.requestId === requestId);
                        if (idx !== -1) {
                            const pending = this.queue.splice(idx, 1)[0];
                            clearTimeout(pending.timeoutId);
                            delete response._requestId; // Limpiar metadata
                            pending.resolve(response);
                        }
                    } else {
                        // Fallback: FIFO
                        const pending = this.queue.shift();
                        if (pending) {
                            clearTimeout(pending.timeoutId);
                            pending.resolve(response);
                        }
                    }
                } catch (e) {
                    console.error('❌ Parse error:', line.slice(0, 120), e.message);
                    const pending = this.queue.shift();
                    if (pending) {
                        clearTimeout(pending.timeoutId);
                        pending.reject(new Error('JSON parse error'));
                    }
                }
            }
        });

        // ✅ v5.0: Control de errores repetitivos
        this.lastOllamaError = 0;
        this.ollamaErrorCount = 0;
        
        this.proc.stderr.on('data', (d) => {
            const msg = d.toString().trim();
            
            // ✅ FILTRAR ERRORES REPETITIVOS DE OLLAMA
            if (msg.includes('Ollama') && msg.includes('HTTP Error 500')) {
                const now = Date.now();
                // Solo mostrar cada 10 segundos
                if (now - this.lastOllamaError > 10000) {
                    console.error('⚠️  Ollama error (usando fallback Smart Mode)');
                    this.lastOllamaError = now;
                    this.ollamaErrorCount++;
                    
                    // Sugerencia después de varios errores
                    if (this.ollamaErrorCount === 3) {
                        console.log('\n💡 Ollama tiene problemas. El sistema funciona en Smart Mode.');
                        console.log('   Para resolver: verifica que "ollama serve" esté corriendo\n');
                    }
                }
                return; // No mostrar más
            }
            
            // ✅ Filtrar fallback messages repetitivos
            if (msg.includes('ResponseGen') && msg.includes('fallback')) {
                return; // Silenciar
            }
            
            // Mostrar otros errores normalmente
            if (msg && !msg.includes('UserWarning')) {
                console.error('🐍 ERR:', msg);
            }
        });

        this.proc.on('close', (code) => {
            console.warn(`⚠️  Brain cerró (code=${code}). Reiniciando...`);
            this.ready = false;
            for (const p of this.queue) {
                clearTimeout(p.timeoutId);
                p.reject(new Error('Brain process died'));
            }
            this.queue = [];
            this.restarts++;
            if (this.restarts < 15) {
                setTimeout(() => this._start(), 2500);
            } else {
                console.error('❌ Brain no puede reiniciarse. Revisa Python.');
            }
        });

        // ✅ FIX #11: Esperar señal real en lugar de timeout arbitrario
        // El ready ahora se activa cuando detectamos "ready" en stdout
    }

    _send(data, timeoutMs = 120000) {  // 120s por defecto
        return new Promise((resolve, reject) => {
            // ✅ FIX #4: Agregar requestId para correlación
            const requestId = `${Date.now()}_${this.requestCounter++}`;
            data._requestId = requestId;

            const timeoutId = setTimeout(() => {
                const idx = this.queue.findIndex(p => p.requestId === requestId);
                if (idx !== -1) {
                    this.queue.splice(idx, 1);
                }
                reject(new Error('Brain timeout'));
            }, timeoutMs);

            this.queue.push({ resolve, reject, timeoutId, requestId });
            
            try {
                this.proc.stdin.write(JSON.stringify(data) + '\n');
            } catch (e) {
                const idx = this.queue.findIndex(p => p.requestId === requestId);
                if (idx !== -1) {
                    this.queue.splice(idx, 1);
                }
                clearTimeout(timeoutId);
                reject(new Error('Failed to write to brain process'));
            }
        });
    }

    async process(message, history = [], searchResults = null) {
        return this._send({ 
            action: 'process', 
            message, 
            history, 
            search_results: searchResults 
        }, 120000);  // 120s — Ollama puede tardar, calidad > velocidad
    }

    async learn(message, response, wasHelpful = true, searchResults = []) {
        return this._send({ 
            action: 'learn', 
            message, 
            response, 
            was_helpful: wasHelpful, 
            search_results: searchResults 
        }, 10000);
    }

    async click(query, url, position, dwellTime, bounced) {
        return this._send({ 
            action: 'click', 
            query, 
            url, 
            position, 
            dwell_time: dwellTime, 
            bounced: !!bounced 
        }, 8000);
    }

    async getStats() {
        // Si hay stats cacheadas, devolverlas inmediatamente sin bloquear el brain.
        // Actualizar en background solo si no hay ya una actualización en curso.
        if (this._cachedStats) {
            if (!this._statsUpdating) {
                this._statsUpdating = true;
                this._send({ action: 'stats' }, 120000)
                    .then(s => { this._cachedStats = s; this.stats = s; })
                    .catch(() => {})
                    .finally(() => { this._statsUpdating = false; });
            }
            return this._cachedStats;
        }
        // Primera vez: esperar la respuesta real
        const s = await this._send({ action: 'stats' }, 120000);
        this._cachedStats = s;
        this.stats = s;
        return s;
    }

    shutdown() {
        if (this.proc) {
            this.proc.kill('SIGTERM');
        }
    }
}

const brain = new BrainProcess();

// ✅ FIX #10: Manejo apropiado de señales de terminación
process.on('SIGTERM', () => {
    console.log('📛 Recibida señal SIGTERM, cerrando...');
    brain.shutdown();
    if (db) {
        // Cerrar conexión MongoDB si existe
        db.client?.close();
    }
    process.exit(0);
});

process.on('SIGINT', () => {
    console.log('📛 Recibida señal SIGINT (Ctrl+C), cerrando...');
    brain.shutdown();
    if (db) {
        db.client?.close();
    }
    process.exit(0);
});

// ══════════════════════════════════════════════════════════════════
//  BÚSQUEDA WEB
// ══════════════════════════════════════════════════════════════════

const UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36';
const HEADERS = { 'User-Agent': UA, 'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8' };

async function searchDDG(query) {
    try {
        // ✅ FIX #14: Sanitizar query
        const sanitizedQuery = encodeURIComponent(query.trim());
        const r = await axios.get('https://api.duckduckgo.com/', {
            params: { q: sanitizedQuery, format: 'json', no_html: 1, skip_disambig: 1 },
            headers: HEADERS, 
            timeout: 6000
        });
        const results = [];
        const d = r.data;
        if (d.AbstractText) {
            results.push({
                title: d.Heading || query,
                url: d.AbstractURL || '',
                description: d.AbstractText,
                snippet: d.AbstractText,
                source: d.AbstractSource || 'Wikipedia'
            });
        }
        for (const t of (d.RelatedTopics || [])) {
            if (t.FirstURL && t.Text) {
                results.push({
                    title: t.Text.split(' - ')[0].slice(0, 90),
                    url: t.FirstURL,
                    description: t.Text,
                    snippet: t.Text,
                    source: 'DuckDuckGo'
                });
                if (results.length >= 6) break;
            }
        }
        return results;
    } catch (error) {
        // ✅ FIX #7: Loggear errores de búsqueda
        console.error('[DDG Search] Error:', error.message);
        return [];
    }
}

async function searchBing(query) {
    try {
        // ✅ FIX #14: Sanitizar query
        const sanitizedQuery = encodeURIComponent(query.trim());
        const url = `https://www.bing.com/search?q=${sanitizedQuery}&setlang=es`;
        const r = await axios.get(url, { headers: HEADERS, timeout: 8000 });
        const $ = cheerio.load(r.data);
        const results = [];
        $('.b_algo').each((i, el) => {
            if (results.length >= 7) return false;
            const titleEl = $(el).find('h2 a');
            const descEl  = $(el).find('.b_caption p, .b_algoSlug');
            const title   = titleEl.text().trim();
            const href    = titleEl.attr('href');
            const desc    = descEl.first().text().trim();
            if (title && href && href.startsWith('http')) {
                results.push({ title, url: href, description: desc, snippet: desc, source: 'Bing' });
            }
        });
        return results;
    } catch (error) {
        // ✅ FIX #7: Loggear errores de búsqueda
        console.error('[Bing Search] Error:', error.message);
        return [];
    }
}

async function searchAll(query) {
    const [ddg, bing] = await Promise.allSettled([searchDDG(query), searchBing(query)]);
    const all = [
        ...(ddg.status === 'fulfilled'  ? ddg.value  : []),
        ...(bing.status === 'fulfilled' ? bing.value : [])
    ];
    
    // ✅ FIX #7: Informar si ambas búsquedas fallaron
    if (all.length === 0) {
        console.warn('[Search] Ambas búsquedas (DDG + Bing) fallaron o no retornaron resultados');
    }
    
    const seen = new Set();
    return all.filter(r => r.url && !seen.has(r.url) && seen.add(r.url));
}

// ══════════════════════════════════════════════════════════════════
//  RUTAS API
// ══════════════════════════════════════════════════════════════════

// POST /api/chat ──────────────────────────────────────────────────
app.post('/api/chat', async (req, res) => {
    const { message, conversationId, userId, history } = req.body;
    if (!message?.trim()) return res.status(400).json({ error: 'Mensaje vacío' });

    const convId = conversationId || `conv_${Date.now()}`;
    console.log(`💬 [${convId.slice(-6)}] "${message.slice(0, 70)}"`);

    try {
        // Búsqueda web previa si el mensaje claramente lo requiere
        // (el brain también puede decidir buscar internamente)
        let searchResults = null;
        const searchKeywords = ['busca', 'buscar', 'encuentra', 'información sobre', 'noticias de'];
        const needsSearch = searchKeywords.some(kw => message.toLowerCase().includes(kw));
        
        if (needsSearch) {
            const searchQuery = message.replace(/^(busca|buscar|encuentra|información sobre|info sobre|noticias de)\s+/i, '').trim();
            console.log(`🔍 Pre-buscando: "${searchQuery}"`);
            searchResults = await searchAll(searchQuery);
            console.log(`   → ${searchResults.length} resultados`);
            if (searchResults.length > 0) {
                searchResults.forEach((r, i) => { r._position = i + 1; });
            }
        }
        
        // Historial de conversación para contexto (máximo 8 turnos)
        const conversationHistory = Array.isArray(history) ? history.slice(-8) : [];
        
        // Procesar con el cerebro — puede tomarse el tiempo que necesite
        const thought = await brain.process(message, conversationHistory, searchResults);

        const responseText = thought.response || thought.message || 'Lo siento, no pude generar una respuesta en este momento.';

        // Actualizar cache de stats con la actividad neural de esta respuesta
        if (thought.neural_activity) {
            brain._cachedStats = thought.neural_activity;
        }

        // Aprendizaje asíncrono en background
        setTimeout(() => {
            brain.learn(message, responseText, true, searchResults || []).catch(err => {
                console.error('[Learn] Error:', err.message);
            });
        }, 100);

        // Persistir en MongoDB
        if (db) {
            db.collection('messages').insertMany([
                { conversationId: convId, role: 'user', content: message, ts: new Date() },
                { conversationId: convId, role: 'assistant', content: responseText,
                  neuralActivity: thought.neural_activity, llmUsed: thought.llm_used, ts: new Date() }
            ]).catch(err => {
                console.error('[MongoDB] Error guardando mensajes:', err.message);
            });
        }

        res.json({
            message:         responseText,
            conversationId:  convId,
            neuralActivity:  thought.neural_activity || {},
            confidence:      thought.confidence || 0.8,
            searchPerformed: !!searchResults && searchResults.length > 0,
            resultsCount:    searchResults ? searchResults.length : 0,
            intent:          thought.intent,
            llmUsed:         thought.llm_used || false,
            llmModel:        thought.llm_model || null,
            processingTime:  thought.processing_time || null,
            ts:              new Date().toISOString()
        });

    } catch (error) {
        console.error('[/api/chat] Error:', error.message, error.stack);
        res.status(500).json({
            error: 'Error procesando mensaje',
            message: 'Hubo un problema procesando tu mensaje. Por favor intenta de nuevo.',
            conversationId: convId,
            ts: new Date().toISOString()
        });
    }
});

// GET /api/search ─────────────────────────────────────────────────
app.get('/api/search', async (req, res) => {
    const { q: query } = req.query;
    if (!query) return res.status(400).json({ error: 'Query requerido' });
    
    console.log(`🔍 Search directo: "${query}"`);
    
    try {
        const raw = await searchAll(query);
        raw.forEach((r, i) => { r._position = i + 1; });
        
        const thought = await brain.process(query, [], raw.length ? raw : null);
        
        if (db) {
            db.collection('searches').insertOne({ 
                query, 
                count: raw.length, 
                ts: new Date() 
            }).catch(err => {
                console.error('[MongoDB] Error guardando búsqueda:', err.message);
            });
        }
        
        res.json({
            query,
            total:          thought.ranked_results?.length || raw.length,
            results:        thought.ranked_results || raw,
            neuralActivity: thought.neural_activity || {},
            ts:             new Date().toISOString()
        });
    } catch (error) {
        console.error('[/api/search] Error:', error.message);
        res.status(500).json({ error: error.message });
    }
});

// POST /api/click ─────────────────────────────────────────────────
app.post('/api/click', async (req, res) => {
    const { query, url, position, dwellTime, bounced } = req.body;
    if (!query || !url) return res.status(400).json({ error: 'Datos incompletos' });
    
    try {
        await brain.click(query, url, position || 1, dwellTime || 0, bounced);
        
        if (db) {
            db.collection('clicks').insertOne({ 
                query, 
                url, 
                position, 
                dwellTime, 
                bounced, 
                ts: new Date() 
            }).catch(err => {
                console.error('[MongoDB] Error guardando click:', err.message);
            });
        }
        
        res.json({ ok: true });
    } catch (error) {
        console.error('[/api/click] Error:', error.message);
        res.status(500).json({ error: error.message });
    }
});

// POST /api/feedback ──────────────────────────────────────────────
app.post('/api/feedback', async (req, res) => {
    const { message, response, helpful } = req.body;
    if (!message) return res.status(400).json({ error: 'Datos incompletos' });
    
    try {
        await brain.learn(message, response || '', helpful !== false);
        res.json({ ok: true });
    } catch (error) {
        console.error('[/api/feedback] Error:', error.message);
        res.status(500).json({ error: error.message });
    }
});

// GET /api/stats ──────────────────────────────────────────────────
app.get('/api/stats', async (req, res) => {
    try {
        const neural = await brain.getStats();
        let dbStats  = {};
        
        if (db) {
            try {
                const [msgs, clicks, searches] = await Promise.all([
                    db.collection('messages').countDocuments(),
                    db.collection('clicks').countDocuments(),
                    db.collection('searches').countDocuments()
                ]);
                dbStats = { messages: msgs, clicks, searches };
            } catch (dbError) {
                // No crítico
            }
        }
        
        res.json({
            neural,
            db:     dbStats,
            server: { 
                uptime:     Math.round(process.uptime()), 
                restarts:   brain.restarts, 
                port:       PORT,
                brainReady: brain.ready
            }
        });
    } catch (error) {
        // Si falla, devolver lo que tenemos en cache sin error 500
        if (brain._cachedStats) {
            return res.json({
                neural:  brain._cachedStats,
                db:      {},
                server:  { uptime: Math.round(process.uptime()), restarts: brain.restarts, port: PORT, brainReady: brain.ready },
                cached:  true
            });
        }
        res.status(500).json({ error: error.message });
    }
});

// GET /health ─────────────────────────────────────────────────────
app.get('/health', (req, res) => {
    res.json({
        status:     brain.ready ? 'ok' : 'initializing',
        brainReady: brain.ready,
        db:         db !== null,
        restarts:   brain.restarts,
        uptime:     process.uptime(),
        ts:         new Date().toISOString()
    });
});

// ══════════════════════════════════════════════════════════════════
//  INICIO
// ══════════════════════════════════════════════════════════════════

async function start() {
    await connectDB();
    
    // Crear directorios necesarios
    for (const d of ['models','data','logs','cache']) {
        await fs.mkdir(path.join(__dirname, d), { recursive: true });
    }
    
    app.listen(PORT, '0.0.0.0', () => {
        console.log(`
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🧠  NEXUS v5.0 ENHANCED — IA con Caché Inteligente         ║
║                                                               ║
║   🌐  http://localhost:${PORT.toString().padEnd(46)}║
║   ⚡  Caché multicapa + Analytics en tiempo real              ║
║   🤖  6 Redes Neuronales (~248k parámetros)                   ║
║   💾  MongoDB + Memoria episódica/semántica/working           ║
║   📈  Backpropagation REAL + Aprendizaje continuo             ║
║   🔥  LLM: Ollama/Groq con fallback inteligente               ║
║                                                               ║
║   ✅ Mejoras v5.0:                                            ║
║     • Caché inteligente (response/embedding/search)           ║
║     • Performance <3s (con caché <0.2s)                       ║
║     • Analytics detallado y métricas                          ║
║     • Filtro de errores repetitivos                           ║
║     • Quality Network (6ta red neuronal)                      ║
║     • Timeout optimizado (35s)                                ║
║     • Sin bucles infinitos de errores                         ║
║                                                               ║
║   Creado por: Jhonatan David Castro Galvis                    ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
        `);
    });
}

start().catch(err => {
    console.error('❌ Error al iniciar servidor:', err);
    process.exit(1);
});

module.exports = app;
