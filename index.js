require('dotenv').config();
const express     = require('express');
const cors        = require('cors');
const axios       = require('axios');
const cheerio     = require('cheerio');
const path        = require('path');
const { spawn }   = require('child_process');
const fs          = require('fs').promises;
const fsSync      = require('fs');
const MongoClient = require('mongodb').MongoClient;
const bcrypt      = require('bcryptjs');
const jwt         = require('jsonwebtoken');
const crypto      = require('crypto');

// ── Multipart / File Upload ──────────────────────────────────────────
let multer, sharp, mammoth, pdfParse;
try { multer   = require('multer');   } catch(e) { console.warn('⚠️  multer no disponible'); }
try { sharp    = require('sharp');    } catch(e) { console.warn('⚠️  sharp no disponible'); }
try { mammoth  = require('mammoth');  } catch(e) { console.warn('⚠️  mammoth no disponible'); }
try { pdfParse = require('pdf-parse'); } catch(e) { console.warn('⚠️  pdf-parse no disponible'); }

// ── Upload dirs ─────────────────────────────────────────────────────
const UPLOAD_DIR   = path.join(__dirname, 'uploads_tmp');
const GENERATED_DIR= path.join(__dirname, 'generated');
[UPLOAD_DIR, GENERATED_DIR].forEach(d => { try { fsSync.mkdirSync(d, { recursive: true }); } catch(e){} });

// Multer storage — guardar en disco temporal
const _multerStorage = multer ? multer.diskStorage({
    destination: (req, file, cb) => cb(null, UPLOAD_DIR),
    filename:    (req, file, cb) => cb(null, `${Date.now()}_${crypto.randomBytes(6).toString('hex')}${path.extname(file.originalname)}`)
}) : null;
const upload = multer ? multer({
    storage: _multerStorage,
    limits:  { fileSize: 50 * 1024 * 1024 }, // 50 MB
    fileFilter: (req, file, cb) => {
        const allowed = [
            'image/jpeg','image/png','image/gif','image/webp','image/svg+xml',
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'text/plain','text/html','text/css','application/javascript',
            'application/json','text/csv','application/xml','text/xml',
            'application/x-python','text/x-python',
            'application/zip','application/x-zip-compressed'
        ];
        const ext = path.extname(file.originalname).toLowerCase();
        const extraExts = ['.py','.js','.ts','.jsx','.tsx','.cpp','.c','.h','.cs','.java',
                           '.go','.rs','.php','.rb','.swift','.kt','.sh','.bash','.sql',
                           '.yaml','.yml','.toml','.env','.md','.mdx','.txt','.csv',
                           '.html','.css','.json','.xml','.svg','.zip','.pdf','.docx','.xlsx'];
        if (allowed.includes(file.mimetype) || extraExts.includes(ext)) { cb(null, true); }
        else { cb(new Error(`Tipo de archivo no soportado: ${file.mimetype}`)); }
    }
}) : null;

const app  = express();
const PORT = process.env.PORT || 3000;
const JWT_SECRET = process.env.JWT_SECRET || 'nexus_fallback_secret_change_in_prod';

// ── Configuración de pagos ─────────────────────────────────────────
const PAYPAL_EMAIL     = 'jhonatandavidcastrogalviz@gmail.com';
const PLAN_PRICE       = 10.00;
const PLAN_CURRENCY    = 'USD';
const FREE_MSG_PER_DAY = 10;
const PLAN_DURATION_MS = 30 * 24 * 60 * 60 * 1000;

// ── Cuentas VIP permanentes ────────────────────────────────────────
const VIP_ACCOUNTS = [
  'jhonatandavidcastrogalviz@gmail.com',
  'theimonsterl141@gmail.com'
];

// ── Stores en memoria (anti-fraude) ───────────────────────────────
const rateLimitStore = new Map();
const loginAttempts  = new Map();

app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ extended: true, limit: '100mb' }));
app.use(express.static('public'));
app.use('/generated', express.static(GENERATED_DIR));
app.use((req, res, next) => { res.setTimeout(600000); next(); }); // 10 min timeout para generación de archivos grandes

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
            db.collection('messages').createIndex({ userId: 1, ts: -1 }),
            db.collection('clicks').createIndex({ query: 1, ts: -1 }),
            db.collection('searches').createIndex({ ts: -1 }),
            db.collection('users').createIndex({ email: 1 }, { unique: true }),
            db.collection('users').createIndex({ username: 1 }, { unique: true }),
            db.collection('payments').createIndex({ transactionId: 1 }, { unique: true }),
            db.collection('payments').createIndex({ userId: 1, ts: -1 }),
            db.collection('fraud_log').createIndex({ ts: -1 }),
            db.collection('fraud_log').createIndex({ ip: 1, ts: -1 }),
            db.collection('used_transactions').createIndex({ transactionId: 1 }, { unique: true }),
            db.collection('fraud_blacklist').createIndex({ email: 1 }),
        ]);
        console.log('✅ MongoDB conectado');
        setInterval(runMonthlyCheck, 60 * 60 * 1000);
        runMonthlyCheck();
    } catch (e) {
        console.warn('⚠️  MongoDB no disponible:', e.message);
    }
}

// ══════════════════════════════════════════════════════════════════
//  VERIFICACIÓN MENSUAL — degradar planes vencidos
// ══════════════════════════════════════════════════════════════════
async function runMonthlyCheck() {
    if (!db) return;
    try {
        const now = new Date();
        const expired = await db.collection('users').find({
            plan: 'premium',
            isVip: { $ne: true },
            planExpiresAt: { $lt: now }
        }).toArray();
        for (const user of expired) {
            await db.collection('users').updateOne(
                { _id: user._id },
                { $set: { plan: 'free', planExpired: true, planDegradedAt: now } }
            );
            console.log(`📉 Plan degradado a FREE: ${user.email}`);
        }
        if (expired.length > 0) console.log(`✅ Verificación mensual: ${expired.length} plan(es) degradado(s)`);
    } catch (e) { console.error('[monthlyCheck]', e.message); }
}

// ══════════════════════════════════════════════════════════════════
//  ANTI-FRAUDE
// ══════════════════════════════════════════════════════════════════
function getClientIP(req) {
    return (
        req.headers['x-forwarded-for']?.split(',')[0]?.trim() ||
        req.headers['x-real-ip'] ||
        req.connection?.remoteAddress ||
        req.socket?.remoteAddress ||
        '0.0.0.0'
    );
}

async function logFraud(type, details, req) {
    console.warn(`🚨 FRAUDE [${type}]`, details);
    if (!db) return;
    try {
        await db.collection('fraud_log').insertOne({
            type, ip: getClientIP(req), ua: req.headers['user-agent'] || '', details, ts: new Date()
        });
    } catch (e) {}
}

function checkRateLimit(req, res, maxReq = 100, windowMs = 60000) {
    const ip  = getClientIP(req);
    const now = Date.now();
    const entry = rateLimitStore.get(ip);
    if (!entry || now > entry.resetAt) {
        rateLimitStore.set(ip, { count: 1, resetAt: now + windowMs });
        return true;
    }
    entry.count++;
    if (entry.count > maxReq) {
        res.status(429).json({ error: 'Demasiadas solicitudes. Espera un momento.' });
        return false;
    }
    return true;
}

function checkLoginAttempts(req, res, identifier) {
    const ip  = getClientIP(req);
    const key = `${ip}:${identifier}`;
    const now = Date.now();
    const entry = loginAttempts.get(key);
    if (entry && now < entry.lockedUntil) {
        const minLeft = Math.ceil((entry.lockedUntil - now) / 60000);
        res.status(429).json({ error: `Cuenta bloqueada por intentos fallidos. Intenta en ${minLeft} min.` });
        return false;
    }
    return true;
}

function recordFailedLogin(req, identifier) {
    const ip  = getClientIP(req);
    const key = `${ip}:${identifier}`;
    const now = Date.now();
    const entry = loginAttempts.get(key) || { count: 0, lockedUntil: 0 };
    entry.count++;
    if (entry.count >= 5) {
        entry.lockedUntil = now + 15 * 60 * 1000;
        logFraud('brute_force_login', { identifier, attempts: entry.count }, { headers: {}, connection: { remoteAddress: ip } });
    }
    loginAttempts.set(key, entry);
}

function clearFailedLogins(req, identifier) {
    loginAttempts.delete(`${getClientIP(req)}:${identifier}`);
}

function isValidPaypalTxId(txId) {
    return /^[A-Z0-9]{13,25}$/.test(txId.trim().toUpperCase());
}

async function detectFraudPatterns(userId, transactionId, payerEmail, req) {
    if (!db) return { ok: true };
    const ip      = getClientIP(req);
    const now     = new Date();
    const hourAgo = new Date(now - 60 * 60 * 1000);
    const dayAgo  = new Date(now - 24 * 60 * 60 * 1000);

    // 1. Transacción ya usada en otra cuenta
    const txUsed = await db.collection('used_transactions').findOne({ transactionId });
    if (txUsed) {
        await logFraud('duplicate_transaction', { transactionId, originalUserId: txUsed.userId, attemptUserId: userId }, req);
        return { ok: false, reason: 'Esta transacción ya fue utilizada en otra cuenta.' };
    }

    // 2. Múltiples verificaciones desde la misma IP en la última hora (max 3)
    const ipVerifications = await db.collection('payments').countDocuments({ ip, ts: { $gte: hourAgo } });
    if (ipVerifications >= 3) {
        await logFraud('ip_payment_flood', { ip, count: ipVerifications }, req);
        return { ok: false, reason: 'Demasiados intentos de pago desde esta IP. Intenta más tarde.' };
    }

    // 3. Usuario con más de 5 intentos de verificación en 24h (spam de txIds falsos)
    const userFraudAttempts = await db.collection('fraud_log').countDocuments({ 'details.userId': userId, ts: { $gte: dayAgo } });
    if (userFraudAttempts >= 5) {
        await logFraud('user_payment_spam', { userId }, req);
        return { ok: false, reason: 'Demasiados intentos fallidos. Contacta soporte.' };
    }

    // 4. Email del pagador en lista negra
    if (payerEmail) {
        const blacklisted = await db.collection('fraud_blacklist').findOne({ email: payerEmail.toLowerCase() });
        if (blacklisted) {
            await logFraud('blacklisted_payer', { payerEmail }, req);
            return { ok: false, reason: 'El email del pagador está reportado como fraudulento.' };
        }
    }

    // 5. Mismo email pagador usado en más de 3 cuentas distintas
    if (payerEmail) {
        const samePayerAccounts = await db.collection('payments').distinct('userId', {
            payerEmail: payerEmail.toLowerCase(), verified: true
        });
        if (samePayerAccounts.length >= 3) {
            await logFraud('payer_multi_account', { payerEmail, accounts: samePayerAccounts.length }, req);
            return { ok: false, reason: 'Este email de PayPal ya fue usado en demasiadas cuentas.' };
        }
    }

    // 6. Formato de txId inválido (no es un ID real de PayPal)
    if (!isValidPaypalTxId(transactionId)) {
        await logFraud('invalid_tx_format', { transactionId, userId }, req);
        return { ok: false, reason: 'ID de transacción con formato inválido. Verifica que lo copiaste correctamente de PayPal.' };
    }

    // 7. Misma IP con más de 2 cuentas creadas en 24h (cuentas falsas masivas)
    const ipAccounts = await db.collection('users').countDocuments({ registrationIp: ip, createdAt: { $gte: dayAgo } });
    if (ipAccounts >= 3) {
        await logFraud('ip_account_farm', { ip, count: ipAccounts }, req);
        return { ok: false, reason: 'Demasiadas cuentas creadas desde esta red. Contacta soporte.' };
    }

    // 8. TxId con caracteres repetidos (ej: AAAAAAAAAAAAAAAA — obviamente falso)
    if (/^(.)\1{8,}$/.test(transactionId.trim())) {
        await logFraud('fake_tx_pattern', { transactionId, userId }, req);
        return { ok: false, reason: 'ID de transacción inválido.' };
    }

    return { ok: true };
}

// ══════════════════════════════════════════════════════════════════
//  AUTH MIDDLEWARE
// ══════════════════════════════════════════════════════════════════
function authMiddleware(req, res, next) {
    const auth = req.headers['authorization'];
    if (!auth || !auth.startsWith('Bearer ')) { req.user = null; return next(); }
    try { req.user = jwt.verify(auth.slice(7), JWT_SECRET); }
    catch (e) { req.user = null; }
    next();
}

function requireAuth(req, res, next) {
    if (!req.user) return res.status(401).json({ error: 'No autenticado' });
    next();
}

app.use(authMiddleware);

// ══════════════════════════════════════════════════════════════════
//  PROCESO PYTHON (cerebro neural)
// ══════════════════════════════════════════════════════════════════
class BrainProcess {
    /**
     * @param {string} scriptName  nombre del archivo en /neural/ (ej. 'brain.py' | 'brain_vip.py')
     * @param {string} label       etiqueta para logs
     */
    constructor(scriptName = 'brain.py', label = 'BASE') {
        this.scriptName = scriptName;
        this.label = label;
        this.proc = null; this.queue = []; this.ready = false;
        this.restarts = 0; this.stats = {}; this.requestCounter = 0;
        this.lastOllamaError = 0; this.ollamaErrorCount = 0;
        this._cachedStats = null; this._statsUpdating = false;
        this._start();
    }

    _start() {
        const brainPath = path.join(__dirname, 'neural', this.scriptName);
        const env = { ...process.env, PYTHONUNBUFFERED: '1' };
        console.log(`🧠 Iniciando cerebro NEXUS [${this.label}] → ${this.scriptName}`);
        this.proc = spawn('python3', ['-u', brainPath], { env });

        let buffer = '';
        this.proc.stdout.on('data', (chunk) => {
            buffer += chunk.toString();
            const parts = buffer.split('\n');
            buffer = parts.pop();
            for (const part of parts) {
                const line = part.trim();
                if (!line) continue;
                if (line.startsWith('✓') || line.startsWith('⚠') || line.startsWith('[')) {
                    console.log('🐍', line);
                    if (line.includes('listo') || line.includes('ready')) this.ready = true;
                    continue;
                }
                try {
                    const response = JSON.parse(line);
                    const requestId = response._requestId;
                    if (requestId) {
                        const idx = this.queue.findIndex(p => p.requestId === requestId);
                        if (idx !== -1) { const p = this.queue.splice(idx,1)[0]; clearTimeout(p.timeoutId); delete response._requestId; p.resolve(response); }
                    } else {
                        const p = this.queue.shift();
                        if (p) { clearTimeout(p.timeoutId); p.resolve(response); }
                    }
                } catch (e) {
                    const p = this.queue.shift();
                    if (p) { clearTimeout(p.timeoutId); p.reject(new Error('JSON parse error')); }
                }
            }
        });

        this.proc.stderr.on('data', (d) => {
            const msg = d.toString().trim();
            if (msg.includes('Ollama') && msg.includes('HTTP Error 500')) {
                if (Date.now() - this.lastOllamaError > 10000) {
                    console.error('⚠️  Ollama error (Smart Mode)');
                    this.lastOllamaError = Date.now(); this.ollamaErrorCount++;
                }
                return;
            }
            if (msg.includes('ResponseGen') && msg.includes('fallback')) return;
            if (msg && !msg.includes('UserWarning')) console.error('🐍 ERR:', msg);
        });

        this.proc.on('close', (code) => {
            console.warn(`⚠️  Brain [${this.label}] cerró (code=${code}). Reiniciando...`);
            this.ready = false;
            for (const p of this.queue) { clearTimeout(p.timeoutId); p.reject(new Error('Brain died')); }
            this.queue = []; this.restarts++;
            if (this.restarts < 15) setTimeout(() => this._start(), 2500);
        });
    }

    _send(data, timeoutMs = 120000) {
        return new Promise((resolve, reject) => {
            const requestId = `${Date.now()}_${this.requestCounter++}`;
            data._requestId = requestId;
            const timeoutId = setTimeout(() => {
                const idx = this.queue.findIndex(p => p.requestId === requestId);
                if (idx !== -1) this.queue.splice(idx, 1);
                reject(new Error('Brain timeout'));
            }, timeoutMs);
            this.queue.push({ resolve, reject, timeoutId, requestId });
            try { this.proc.stdin.write(JSON.stringify(data) + '\n'); }
            catch (e) {
                const idx = this.queue.findIndex(p => p.requestId === requestId);
                if (idx !== -1) this.queue.splice(idx, 1);
                clearTimeout(timeoutId); reject(e);
            }
        });
    }

    async process(msg, hist=[], sr=null, userCtx=null) { return this._send({ action:'process', message:msg, history:hist, search_results:sr, user_context:userCtx }, 120000); }
    async learn(msg, res, helpful=true, sr=[]) { return this._send({ action:'learn', message:msg, response:res, was_helpful:helpful, search_results:sr }, 10000); }
    async click(q, url, pos, dwell, bounced) { return this._send({ action:'click', query:q, url, position:pos, dwell_time:dwell, bounced:!!bounced }, 8000); }

    async getStats() {
        if (this._cachedStats) {
            if (!this._statsUpdating) {
                this._statsUpdating = true;
                this._send({ action:'stats' }, 120000)
                    .then(s=>{ this._cachedStats=s; this.stats=s; })
                    .catch(()=>{})
                    .finally(()=>{ this._statsUpdating=false; });
            }
            return this._cachedStats;
        }
        const s = await this._send({ action:'stats' }, 120000);
        this._cachedStats = s; this.stats = s; return s;
    }

    shutdown() { if (this.proc) this.proc.kill('SIGTERM'); }
}

// ── Dos instancias del cerebro ─────────────────────────────────────
// brainBase  → brain.py     (usuarios free)
// brainVip   → brain_vip.py (premium / VIP / creador)
const brainBase = new BrainProcess('brain.py',     'BASE');
const brainVip  = new BrainProcess('brain_vip.py', 'ULTRA');

// Alias de compatibilidad para rutas que no dependen del plan
const brain = brainBase;

process.on('SIGTERM', () => { brainBase.shutdown(); brainVip.shutdown(); if (db) db.client?.close(); process.exit(0); });
process.on('SIGINT',  () => { brainBase.shutdown(); brainVip.shutdown(); if (db) db.client?.close(); process.exit(0); });

// ══════════════════════════════════════════════════════════════════
//  BÚSQUEDA WEB
// ══════════════════════════════════════════════════════════════════
const UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36';
const HEADERS = { 'User-Agent': UA, 'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8' };

async function searchDDG(query) {
    try {
        const q = encodeURIComponent(query.trim());
        const r = await axios.get('https://api.duckduckgo.com/', { params:{q,format:'json',no_html:1,skip_disambig:1}, headers:HEADERS, timeout:6000 });
        const results=[]; const d=r.data;
        if (d.AbstractText) results.push({ title:d.Heading||query, url:d.AbstractURL||'', description:d.AbstractText, snippet:d.AbstractText, source:d.AbstractSource||'Wikipedia' });
        for (const t of (d.RelatedTopics||[])) { if (t.FirstURL&&t.Text) { results.push({ title:t.Text.split(' - ')[0].slice(0,90), url:t.FirstURL, description:t.Text, snippet:t.Text, source:'DuckDuckGo' }); if (results.length>=6) break; } }
        return results;
    } catch { return []; }
}

async function searchBing(query) {
    try {
        const q = encodeURIComponent(query.trim());
        const r = await axios.get(`https://www.bing.com/search?q=${q}&setlang=es`, { headers:HEADERS, timeout:8000 });
        const $ = cheerio.load(r.data); const results=[];
        $('.b_algo').each((i,el)=>{
            if (results.length>=7) return false;
            const te=$(el).find('h2 a'), de=$(el).find('.b_caption p,.b_algoSlug');
            const title=te.text().trim(), href=te.attr('href'), desc=de.first().text().trim();
            if (title&&href&&href.startsWith('http')) results.push({ title, url:href, description:desc, snippet:desc, source:'Bing' });
        });
        return results;
    } catch { return []; }
}

async function searchAll(query) {
    const [ddg,bing] = await Promise.allSettled([searchDDG(query), searchBing(query)]);
    const all=[...(ddg.status==='fulfilled'?ddg.value:[]), ...(bing.status==='fulfilled'?bing.value:[])];
    const seen=new Set();
    return all.filter(r=>r.url&&!seen.has(r.url)&&seen.add(r.url));
}

// ══════════════════════════════════════════════════════════════════
//  HELPERS PLAN
// ══════════════════════════════════════════════════════════════════
// ── Emails del creador (acceso total + trato especial) ────────────
const CREATOR_EMAILS = [
    'jhonatandavidcastrogalviz@gmail.com',
    'theimonsterl141@gmail.com'
];

function isVipAccount(email) {
    return VIP_ACCOUNTS.includes(email?.toLowerCase()?.trim());
}

function isCreatorAccount(email) {
    return CREATOR_EMAILS.includes(email?.toLowerCase()?.trim());
}

async function getPlanStatus(user) {
    if (!user) return { plan:'free', active:false };
    if (user.isVip || isVipAccount(user.email)) return { plan:'premium', active:true, isVip:true, expiresAt:null };
    if (user.plan==='premium' && user.planExpiresAt) {
        if (new Date(user.planExpiresAt) > new Date()) return { plan:'premium', active:true, isVip:false, expiresAt:user.planExpiresAt };
        if (db) await db.collection('users').updateOne({ _id:user._id }, { $set:{ plan:'free', planExpired:true } });
        return { plan:'free', active:false, expired:true };
    }
    return { plan:'free', active:false };
}

async function getMessagesToday(userId) {
    if (!db) return 0;
    const today = new Date(); today.setHours(0,0,0,0);
    return db.collection('messages').countDocuments({ userId, role:'user', ts:{ $gte:today } });
}

function generateToken(user) {
    return jwt.sign(
        {
            id:        user._id.toString(),
            email:     user.email,
            username:  user.username,
            plan:      user.plan||'free',
            isVip:     user.isVip || isVipAccount(user.email),
            isCreator: isCreatorAccount(user.email)
        },
        JWT_SECRET,
        { expiresIn:'30d' }
    );
}

function sanitizeUser(user) {
    return {
        id:          user._id.toString(),
        email:       user.email,
        username:    user.username,
        displayName: user.displayName || user.username,
        createdAt:   user.createdAt,
        isVip:       user.isVip || isVipAccount(user.email),
        isCreator:   isCreatorAccount(user.email)
    };
}

// ══════════════════════════════════════════════════════════════════
//  RUTAS AUTH
// ══════════════════════════════════════════════════════════════════

// POST /api/auth/register
app.post('/api/auth/register', async (req, res) => {
    if (!checkRateLimit(req, res, 10, 60000)) return;
    const { email, username, password } = req.body;
    if (!email||!username||!password) return res.status(400).json({ error:'Email, usuario y contraseña requeridos' });
    if (password.length<6) return res.status(400).json({ error:'Contraseña mínimo 6 caracteres' });
    if (username.length<3) return res.status(400).json({ error:'Usuario mínimo 3 caracteres' });
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return res.status(400).json({ error:'Email inválido' });
    if (!/^[a-zA-Z0-9_.-]{3,30}$/.test(username)) return res.status(400).json({ error:'Usuario: solo letras, números, _, -, . (3-30 chars)' });
    if (!db) return res.status(503).json({ error:'Base de datos no disponible' });
    try {
        const exists = await db.collection('users').findOne({ $or:[{ email:email.toLowerCase() },{ username:username.toLowerCase() }] });
        if (exists) {
            if (exists.email===email.toLowerCase()) return res.status(409).json({ error:'El email ya está registrado' });
            return res.status(409).json({ error:'El nombre de usuario ya está en uso' });
        }
        const isVip = isVipAccount(email);
        const hash  = await bcrypt.hash(password, 12);
        const result = await db.collection('users').insertOne({
            email:email.toLowerCase(), username:username.toLowerCase(), displayName:username,
            password:hash, plan:isVip?'premium':'free', isVip,
            planExpiresAt:null, createdAt:new Date(), updatedAt:new Date(), registrationIp:getClientIP(req)
        });
        const user  = await db.collection('users').findOne({ _id:result.insertedId });
        const token = generateToken(user);
        const planStatus = await getPlanStatus(user);
        const msgsToday  = planStatus.plan==='free' ? 0 : null;
        res.json({ token, user:sanitizeUser(user), plan:planStatus, messagesUsed:msgsToday, messagesLimit:FREE_MSG_PER_DAY });
    } catch (e) { console.error('[register]',e.message); res.status(500).json({ error:'Error al registrar' }); }
});

// POST /api/auth/login
app.post('/api/auth/login', async (req, res) => {
    if (!checkRateLimit(req, res, 20, 60000)) return;
    const { identifier, password } = req.body;
    if (!identifier||!password) return res.status(400).json({ error:'Credenciales requeridas' });
    if (!db) return res.status(503).json({ error:'Base de datos no disponible' });
    if (!checkLoginAttempts(req, res, identifier)) { await logFraud('login_blocked',{ identifier },req); return; }
    try {
        const id   = identifier.toLowerCase().trim();
        const user = await db.collection('users').findOne({ $or:[{ email:id },{ username:id }] });
        if (!user) { recordFailedLogin(req,identifier); return res.status(401).json({ error:'Credenciales incorrectas' }); }
        const valid = await bcrypt.compare(password, user.password);
        if (!valid) { recordFailedLogin(req,identifier); await logFraud('failed_login',{ identifier, userId:user._id.toString() },req); return res.status(401).json({ error:'Credenciales incorrectas' }); }
        clearFailedLogins(req, identifier);
        if (isVipAccount(user.email) && !user.isVip) {
            await db.collection('users').updateOne({ _id:user._id },{ $set:{ isVip:true, plan:'premium' } });
            user.isVip=true; user.plan='premium';
        }
        await db.collection('users').updateOne({ _id:user._id },{ $set:{ lastLoginAt:new Date(), lastLoginIp:getClientIP(req) } });
        const token     = generateToken(user);
        const planStatus= await getPlanStatus(user);
        const msgsToday = planStatus.plan==='free' ? await getMessagesToday(user._id.toString()) : null;
        res.json({ token, user:sanitizeUser(user), plan:planStatus, messagesUsed:msgsToday, messagesLimit:FREE_MSG_PER_DAY });
    } catch (e) { console.error('[login]',e.message); res.status(500).json({ error:'Error al iniciar sesión' }); }
});

// GET /api/auth/me
app.get('/api/auth/me', requireAuth, async (req, res) => {
    if (!db) return res.status(503).json({ error:'BD no disponible' });
    try {
        const { ObjectId } = require('mongodb');
        const user = await db.collection('users').findOne({ _id:new ObjectId(req.user.id) });
        if (!user) return res.status(404).json({ error:'Usuario no encontrado' });
        if (isVipAccount(user.email) && (!user.isVip||user.plan!=='premium')) {
            await db.collection('users').updateOne({ _id:user._id },{ $set:{ isVip:true, plan:'premium' } });
            user.isVip=true; user.plan='premium';
        }
        const planStatus = await getPlanStatus(user);
        const msgsToday  = planStatus.plan==='free' ? await getMessagesToday(user._id.toString()) : null;
        const resetsAt   = planStatus.plan==='free' ? (() => { const d=new Date(); d.setHours(24,0,0,0); return d; })() : null;
        res.json({ user:sanitizeUser(user), plan:planStatus, messagesUsed:msgsToday, messagesLimit:FREE_MSG_PER_DAY, resetsAt });
    } catch (e) { res.status(500).json({ error:'Error al obtener usuario' }); }
});

// PATCH /api/auth/profile
app.patch('/api/auth/profile', requireAuth, async (req, res) => {
    const { displayName, username } = req.body;
    if (!db) return res.status(503).json({ error:'BD no disponible' });
    try {
        const { ObjectId } = require('mongodb');
        const updates = { updatedAt:new Date() };
        if (displayName!==undefined) {
            if (!displayName.trim()) return res.status(400).json({ error:'Nombre no puede estar vacío' });
            updates.displayName = displayName.trim().slice(0,50);
        }
        if (username!==undefined) {
            if (!/^[a-zA-Z0-9_.-]{3,30}$/.test(username)) return res.status(400).json({ error:'Usuario inválido' });
            const taken = await db.collection('users').findOne({ username:username.toLowerCase(), _id:{ $ne:new ObjectId(req.user.id) } });
            if (taken) return res.status(409).json({ error:'Nombre de usuario en uso' });
            updates.username = username.toLowerCase();
        }
        await db.collection('users').updateOne({ _id:new ObjectId(req.user.id) },{ $set:updates });
        const user  = await db.collection('users').findOne({ _id:new ObjectId(req.user.id) });
        const token = generateToken(user);
        res.json({ token, user:sanitizeUser(user) });
    } catch (e) { res.status(500).json({ error:'Error al actualizar perfil' }); }
});

// ══════════════════════════════════════════════════════════════════
//  RUTAS PAGO
// ══════════════════════════════════════════════════════════════════

// GET /api/payment/info
app.get('/api/payment/info', requireAuth, async (req, res) => {
    const { ObjectId } = require('mongodb');
    let user = null;
    if (db) user = await db.collection('users').findOne({ _id:new ObjectId(req.user.id) });
    const planStatus = user ? await getPlanStatus(user) : { plan:'free' };
    res.json({
        plan: planStatus.plan, isVip: planStatus.isVip||false, expiresAt: planStatus.expiresAt||null,
        method:'PayPal', paypalTarget:Buffer.from(PAYPAL_EMAIL).toString('base64'),
        amount:PLAN_PRICE.toFixed(2), currency:PLAN_CURRENCY, period:'mensual',
        description:'NEXUS AI — Plan Premium Mensual ($10 USD/mes)'
    });
});

// POST /api/payment/verify
app.post('/api/payment/verify', requireAuth, async (req, res) => {
    if (!checkRateLimit(req, res, 5, 60000)) return;
    const { transactionId, payerEmail } = req.body;
    if (!transactionId) return res.status(400).json({ error:'ID de transacción requerido' });
    if (!db) return res.status(503).json({ error:'BD no disponible' });

    const userId = req.user.id;
    const txId   = transactionId.trim().toUpperCase();

    try {
        const { ObjectId } = require('mongodb');
        const user = await db.collection('users').findOne({ _id:new ObjectId(userId) });

        // VIP no necesita pago
        if (user && (user.isVip||isVipAccount(user.email))) {
            return res.json({ ok:true, plan:'premium', isVip:true, message:'Cuenta VIP — acceso permanente sin pago.' });
        }

        // Anti-fraude
        const fraudCheck = await detectFraudPatterns(userId, txId, payerEmail?.toLowerCase(), req);
        if (!fraudCheck.ok) {
            await logFraud('payment_rejected',{ userId, transactionId:txId, reason:fraudCheck.reason },req);
            return res.status(403).json({ error:fraudCheck.reason });
        }

        // Registrar tx usada
        await db.collection('used_transactions').insertOne({ transactionId:txId, userId, ts:new Date() });

        const planExpiresAt = new Date(Date.now() + PLAN_DURATION_MS);

        await db.collection('payments').insertOne({
            userId, transactionId:txId, amount:PLAN_PRICE, currency:PLAN_CURRENCY,
            payerEmail:payerEmail?.toLowerCase()||'unknown', verified:true,
            ip:getClientIP(req), planExpiresAt, ts:new Date()
        });

        await db.collection('users').updateOne(
            { _id:new ObjectId(userId) },
            { $set:{ plan:'premium', planExpiresAt, planExpired:false, lastPaymentAt:new Date() } }
        );

        console.log(`✅ Premium activado: ${user?.email} → hasta ${planExpiresAt.toISOString()}`);
        res.json({ ok:true, plan:'premium', expiresAt:planExpiresAt, message:`¡Plan Premium activado! Válido hasta el ${planExpiresAt.toLocaleDateString('es')}.` });
    } catch (e) {
        if (e.code===11000) {
            await logFraud('duplicate_tx_attempt',{ userId, transactionId:txId },req);
            return res.status(409).json({ error:'Esta transacción ya fue registrada anteriormente.' });
        }
        console.error('[payment/verify]',e.message);
        res.status(500).json({ error:'Error al verificar pago' });
    }
});

// GET /api/payment/status
app.get('/api/payment/status', requireAuth, async (req, res) => {
    if (!db) return res.status(503).json({ error:'BD no disponible' });
    try {
        const { ObjectId } = require('mongodb');
        const user = await db.collection('users').findOne({ _id:new ObjectId(req.user.id) });
        if (!user) return res.status(404).json({ error:'Usuario no encontrado' });
        const planStatus = await getPlanStatus(user);
        const msgsToday  = planStatus.plan==='free' ? await getMessagesToday(req.user.id) : null;
        const resetsAt   = planStatus.plan==='free' ? (() => { const d=new Date(); d.setHours(24,0,0,0); return d; })() : null;
        res.json({ plan:planStatus, messagesUsed:msgsToday, messagesLimit:FREE_MSG_PER_DAY, resetsAt });
    } catch (e) { res.status(500).json({ error:'Error' }); }
});

// ══════════════════════════════════════════════════════════════════
//  CHAT
// ══════════════════════════════════════════════════════════════════
app.post('/api/chat', requireAuth, async (req, res) => {
    if (!checkRateLimit(req, res, 60, 60000)) return;
    const { message, conversationId, history } = req.body;
    if (!message?.trim()) return res.status(400).json({ error:'Mensaje vacío' });

    const userId = req.user.id;
    const { ObjectId } = require('mongodb');
    const user = db ? await db.collection('users').findOne({ _id:new ObjectId(userId) }) : null;
    const planStatus = user ? await getPlanStatus(user) : { plan:'free' };

    if (planStatus.plan !== 'premium') {
        const msgsToday = await getMessagesToday(userId);
        if (msgsToday >= FREE_MSG_PER_DAY) {
            const resetsAt = new Date(); resetsAt.setHours(24,0,0,0);
            return res.status(402).json({
                error:'limit_reached',
                message:`Límite de ${FREE_MSG_PER_DAY} mensajes diarios gratuitos alcanzado. Se renueva a medianoche o actualiza a Premium ($10/mes).`,
                messagesUsed:msgsToday, messagesLimit:FREE_MSG_PER_DAY, resetsAt
            });
        }
    }

    const convId = conversationId || `conv_${Date.now()}`;
    const userEmail = user?.email || req.user.email || '';
    const isCreator = isCreatorAccount(userEmail);
    const isVip     = user?.isVip || isVipAccount(userEmail);

    // ── Contexto de usuario para el brain ─────────────────────────
    const userContext = {
        userId,
        email:       userEmail,
        username:    user?.username || req.user.username || '',
        displayName: user?.displayName || user?.username || req.user.username || '',
        plan:        planStatus.plan,
        isVip,
        isCreator
    };

    console.log(`💬 [${convId.slice(-6)}] [${planStatus.plan}]${isCreator?' 👑 CREATOR':''} "${message.slice(0,70)}"`);

    // ── Selección de cerebro según plan ──────────────────────────
    const useVipBrain = isCreator || isVip || planStatus.plan === 'premium';
    const activeBrain = useVipBrain ? brainVip : brainBase;
    const brainVersion = useVipBrain ? 'ultra' : 'base';
    console.log(`🔀 Cerebro activo: ${brainVersion.toUpperCase()} [${useVipBrain ? 'brain_vip.py' : 'brain.py'}]`);

    try {
        let searchResults = null;
        const skws = ['busca','buscar','encuentra','información sobre','noticias de'];
        if (skws.some(kw=>message.toLowerCase().includes(kw))) {
            const q=message.replace(/^(busca|buscar|encuentra|información sobre|info sobre|noticias de)\s+/i,'').trim();
            searchResults = await searchAll(q);
            searchResults.forEach((r,i)=>{ r._position=i+1; });
        }
        const conversationHistory = Array.isArray(history) ? history.slice(-8) : [];
        const thought = await activeBrain.process(message, conversationHistory, searchResults, userContext);
        const responseText = thought.response||thought.message||'Lo siento, no pude generar una respuesta.';

        if (thought.neural_activity) activeBrain._cachedStats = thought.neural_activity;
        setTimeout(()=>{ activeBrain.learn(message,responseText,true,searchResults||[]).catch(()=>{}); },100);

        if (db) {
            db.collection('messages').insertMany([
                { conversationId:convId, userId, role:'user', content:message, ts:new Date() },
                { conversationId:convId, userId, role:'assistant', content:responseText, neuralActivity:thought.neural_activity, llmUsed:thought.llm_used, ts:new Date() }
            ]).catch(()=>{});
        }

        const msgsToday = planStatus.plan==='free' ? await getMessagesToday(userId) : null;
        res.json({
            message:responseText, conversationId:convId, neuralActivity:thought.neural_activity||{},
            confidence:thought.confidence||0.8, searchPerformed:!!searchResults?.length,
            resultsCount:searchResults?.length||0, intent:thought.intent,
            llmUsed:thought.llm_used||false, llmModel:thought.llm_model||null,
            processingTime:thought.processing_time||null,
            plan:planStatus.plan, messagesUsed:msgsToday, messagesLimit:FREE_MSG_PER_DAY,
            isCreator, brainVersion,
            ts:new Date().toISOString()
        });
    } catch (error) {
        console.error('[/api/chat]',error.message);
        res.status(500).json({ error:'Error procesando mensaje', message:'Hubo un problema. Intenta de nuevo.', conversationId:convId, ts:new Date().toISOString() });
    }
});

app.get('/api/search', async (req, res) => {
    const { q:query } = req.query;
    if (!query) return res.status(400).json({ error:'Query requerido' });
    try {
        const raw=await searchAll(query); raw.forEach((r,i)=>{r._position=i+1;});
        const thought=await brain.process(query,[],raw.length?raw:null);
        if (db) db.collection('searches').insertOne({ query,count:raw.length,ts:new Date() }).catch(()=>{});
        res.json({ query,total:thought.ranked_results?.length||raw.length,results:thought.ranked_results||raw,neuralActivity:thought.neural_activity||{},ts:new Date().toISOString() });
    } catch (error) { res.status(500).json({ error:error.message }); }
});

app.post('/api/click', async (req, res) => {
    const { query,url,position,dwellTime,bounced } = req.body;
    if (!query||!url) return res.status(400).json({ error:'Datos incompletos' });
    try { await brain.click(query,url,position||1,dwellTime||0,bounced); if (db) db.collection('clicks').insertOne({ query,url,position,dwellTime,bounced,ts:new Date() }).catch(()=>{}); res.json({ ok:true }); }
    catch (error) { res.status(500).json({ error:error.message }); }
});

app.post('/api/feedback', async (req, res) => {
    const { message,response,helpful } = req.body;
    if (!message) return res.status(400).json({ error:'Datos incompletos' });
    try { await brain.learn(message,response||'',helpful!==false); res.json({ ok:true }); }
    catch (error) { res.status(500).json({ error:error.message }); }
});

app.get('/api/stats', async (req, res) => {
    try {
        const neural = await brainBase.getStats();
        let dbStats={};
        if (db) {
            try {
                const [msgs,clicks,searches,users,premiumUsers] = await Promise.all([
                    db.collection('messages').countDocuments(),
                    db.collection('clicks').countDocuments(),
                    db.collection('searches').countDocuments(),
                    db.collection('users').countDocuments(),
                    db.collection('users').countDocuments({ plan:'premium' })
                ]);
                dbStats={ messages:msgs,clicks,searches,users,premiumUsers };
            } catch (_) {}
        }
        res.json({
            neural, db:dbStats,
            server:{
                uptime:Math.round(process.uptime()),
                restarts:brainBase.restarts,
                restartsVip:brainVip.restarts,
                port:PORT,
                brainReady:brainBase.ready,
                brainVipReady:brainVip.ready
            }
        });
    } catch (error) {
        if (brainBase._cachedStats) return res.json({ neural:brainBase._cachedStats,db:{},server:{ uptime:Math.round(process.uptime()),restarts:brainBase.restarts,port:PORT,brainReady:brainBase.ready },cached:true });
        res.status(500).json({ error:error.message });
    }
});

app.get('/health', (req, res) => {
    res.json({ status:brainBase.ready?'ok':'initializing',brainReady:brainBase.ready,brainVipReady:brainVip.ready,db:db!==null,restarts:brainBase.restarts,restartsVip:brainVip.restarts,uptime:process.uptime(),ts:new Date().toISOString() });
});

// ══════════════════════════════════════════════════════════════════
//  UTILIDADES DE ARCHIVO — extracción de contenido
// ══════════════════════════════════════════════════════════════════

async function extractFileContent(filePath, mimeType, originalName) {
    const ext = path.extname(originalName).toLowerCase();
    try {
        // Imágenes → base64 para enviar al LLM
        if (mimeType?.startsWith('image/') || ['.jpg','.jpeg','.png','.gif','.webp','.svg'].includes(ext)) {
            const buf = await fs.readFile(filePath);
            const b64 = buf.toString('base64');
            let meta = { width: 0, height: 0, format: ext.slice(1) };
            if (sharp) { try { meta = await sharp(buf).metadata(); } catch(e){} }
            return {
                type: 'image',
                base64: b64,
                mimeType: mimeType || `image/${ext.slice(1)}`,
                meta,
                textSummary: `[Imagen adjunta: ${originalName}, ${meta.width}x${meta.height} ${meta.format}]`
            };
        }
        // PDF → texto
        if (mimeType === 'application/pdf' || ext === '.pdf') {
            if (pdfParse) {
                const buf = await fs.readFile(filePath);
                const result = await pdfParse(buf);
                return { type: 'pdf', text: result.text, pages: result.numpages, textSummary: `[PDF: ${originalName}, ${result.numpages} páginas]\n\n${result.text.slice(0, 50000)}` };
            }
            return { type: 'pdf', text: '[PDF — instala pdf-parse para extracción de texto]', textSummary: `[PDF adjunto: ${originalName}]` };
        }
        // DOCX → texto
        if (['.docx','.doc'].includes(ext) || mimeType?.includes('wordprocessingml')) {
            if (mammoth) {
                const buf = await fs.readFile(filePath);
                const result = await mammoth.extractRawText({ buffer: buf });
                return { type: 'docx', text: result.value, textSummary: `[Documento Word: ${originalName}]\n\n${result.value.slice(0, 50000)}` };
            }
            return { type: 'docx', text: '[DOCX — instala mammoth para extracción]', textSummary: `[DOCX adjunto: ${originalName}]` };
        }
        // Texto plano, código, etc.
        const textExts = ['.txt','.md','.js','.ts','.jsx','.tsx','.py','.cpp','.c','.h','.cs',
                          '.java','.go','.rs','.php','.rb','.swift','.kt','.sh','.bash','.sql',
                          '.yaml','.yml','.toml','.env','.html','.css','.json','.xml','.csv','.log'];
        if (textExts.includes(ext) || mimeType?.startsWith('text/') || mimeType === 'application/json') {
            const text = await fs.readFile(filePath, 'utf-8');
            return { type: 'code', ext: ext.slice(1), text, textSummary: `[Archivo ${originalName} — ${text.split('\n').length} líneas]\n\n${text}` };
        }
        return { type: 'binary', textSummary: `[Archivo binario adjunto: ${originalName}]` };
    } catch (e) {
        return { type: 'error', textSummary: `[Error leyendo ${originalName}: ${e.message}]` };
    }
}

// ── POST /api/upload — subir archivo y procesar ────────────────────
app.post('/api/upload', requireAuth, (req, res) => {
    if (!upload) return res.status(501).json({ error: 'Módulo multer no instalado. Ejecuta: npm install multer' });
    const uploader = upload.single('file');
    uploader(req, res, async (err) => {
        if (err) return res.status(400).json({ error: err.message });
        if (!req.file) return res.status(400).json({ error: 'No se recibió ningún archivo' });

        try {
            const { path: filePath, mimetype, originalname, size } = req.file;
            const content = await extractFileContent(filePath, mimetype, originalname);

            // Si es imagen, generar thumbnail si sharp disponible
            let thumbBase64 = null;
            if (content.type === 'image' && sharp) {
                try {
                    const thumbBuf = await sharp(filePath).resize(400, 400, { fit: 'inside' }).jpeg({ quality: 80 }).toBuffer();
                    thumbBase64 = thumbBuf.toString('base64');
                } catch(e) {}
            }

            // Limpiar archivo temporal
            fs.unlink(filePath).catch(() => {});

            console.log(`📎 [upload] ${originalname} (${(size/1024).toFixed(1)}KB) → tipo: ${content.type}`);
            res.json({
                ok: true,
                type: content.type,
                name: originalname,
                size,
                mimeType: mimetype,
                base64: content.base64 || null,
                thumbBase64,
                text: content.text || null,
                textSummary: content.textSummary,
                meta: content.meta || null,
                pages: content.pages || null,
                ext: content.ext || null
            });
        } catch (e) {
            console.error('[upload]', e.message);
            res.status(500).json({ error: `Error procesando archivo: ${e.message}` });
        }
    });
});

// ── POST /api/chat-with-file — chat enviando archivos ──────────────
app.post('/api/chat-with-file', requireAuth, async (req, res) => {
    if (!checkRateLimit(req, res, 60, 60000)) return;
    const { message, conversationId, history, fileData } = req.body;

    const userId = req.user.id;
    const { ObjectId } = require('mongodb');
    const user = db ? await db.collection('users').findOne({ _id: new ObjectId(userId) }) : null;
    const planStatus = user ? await getPlanStatus(user) : { plan: 'free' };

    if (planStatus.plan !== 'premium') {
        const msgsToday = await getMessagesToday(userId);
        if (msgsToday >= FREE_MSG_PER_DAY) {
            return res.status(402).json({
                error: 'limit_reached',
                message: `Límite de ${FREE_MSG_PER_DAY} mensajes diarios alcanzado. Actualiza a Premium.`
            });
        }
    }

    const userEmail = user?.email || req.user.email || '';
    const isCreator = isCreatorAccount(userEmail);
    const isVip     = user?.isVip || isVipAccount(userEmail);
    const useVipBrain = isCreator || isVip || planStatus.plan === 'premium';
    const activeBrain = useVipBrain ? brainVip : brainBase;

    // Construir el mensaje enriquecido con el archivo
    let enrichedMessage = message || '';
    if (fileData) {
        enrichedMessage = `${fileData.textSummary}\n\n${message || 'Analiza este archivo y responde.'}`;
    }

    const userContext = {
        userId, email: userEmail,
        username: user?.username || req.user.username || '',
        displayName: user?.displayName || user?.username || '',
        plan: planStatus.plan, isVip, isCreator,
        hasFile: !!fileData,
        fileType: fileData?.type || null,
        fileName: fileData?.name || null
    };

    try {
        const conversationHistory = Array.isArray(history) ? history.slice(-8) : [];
        const convId = conversationId || `conv_${Date.now()}`;
        const thought = await activeBrain.process(enrichedMessage, conversationHistory, null, userContext);
        const responseText = thought.response || thought.message || 'Lo siento, no pude procesar el archivo.';

        setTimeout(() => { activeBrain.learn(enrichedMessage, responseText, true, []).catch(() => {}); }, 100);

        if (db) {
            db.collection('messages').insertMany([
                { conversationId: convId, userId, role: 'user', content: message || '[Archivo adjunto]', hasFile: !!fileData, fileName: fileData?.name, ts: new Date() },
                { conversationId: convId, userId, role: 'assistant', content: responseText, ts: new Date() }
            ]).catch(() => {});
        }

        res.json({ message: responseText, conversationId: convId, plan: planStatus.plan, ts: new Date().toISOString() });
    } catch (e) {
        console.error('[chat-with-file]', e.message);
        res.status(500).json({ error: 'Error procesando archivo con el cerebro' });
    }
});

// ══════════════════════════════════════════════════════════════════
//  GENERACIÓN DE ARCHIVOS (código, docs, etc.) — SIN LÍMITE
// ══════════════════════════════════════════════════════════════════

// POST /api/generate-file — genera archivo de cualquier tipo y tamaño
app.post('/api/generate-file', requireAuth, async (req, res) => {
    if (!checkRateLimit(req, res, 20, 60000)) return;
    const { prompt, fileType, fileName, currentContent, operation } = req.body;
    if (!prompt && !currentContent) return res.status(400).json({ error: 'Se requiere prompt o contenido' });

    const userId = req.user.id;
    const { ObjectId } = require('mongodb');
    const user = db ? await db.collection('users').findOne({ _id: new ObjectId(userId) }) : null;
    const planStatus = user ? await getPlanStatus(user) : { plan: 'free' };
    const userEmail = user?.email || req.user.email || '';
    const isCreator = isCreatorAccount(userEmail);
    const isVip     = user?.isVip || isVipAccount(userEmail);
    const useVipBrain = isCreator || isVip || planStatus.plan === 'premium';
    const activeBrain = useVipBrain ? brainVip : brainBase;

    // Operaciones: 'create' | 'edit' | 'analyze' | 'fix' | 'extend'
    const op = operation || 'create';

    let enrichedPrompt;
    if (op === 'edit' && currentContent) {
        enrichedPrompt = `OPERACIÓN: EDITAR ARCHIVO EXISTENTE
ARCHIVO: ${fileName || 'archivo'}
TIPO: ${fileType || 'texto'}
INSTRUCCIÓN: ${prompt}

CONTENIDO ACTUAL DEL ARCHIVO (${currentContent.split('\n').length} líneas):
\`\`\`
${currentContent}
\`\`\`

IMPORTANTE: Devuelve SOLO el archivo completo con las modificaciones aplicadas. No omitas ninguna línea. El archivo debe estar íntegro y funcional.`;
    } else if (op === 'analyze' && currentContent) {
        enrichedPrompt = `ANALIZA este archivo (${fileName || 'archivo'}, ${currentContent.split('\n').length} líneas) y responde: ${prompt}\n\nCONTENIDO:\n\`\`\`\n${currentContent}\n\`\`\``;
    } else if (op === 'fix' && currentContent) {
        enrichedPrompt = `CORRIGE ERRORES en este archivo (${fileName || 'archivo'}):
PROBLEMA REPORTADO: ${prompt}

CONTENIDO ACTUAL (${currentContent.split('\n').length} líneas):
\`\`\`
${currentContent}
\`\`\`

Devuelve el archivo completo corregido sin omitir nada.`;
    } else {
        enrichedPrompt = `GENERA ARCHIVO: ${fileName || `archivo.${fileType || 'txt'}`}
TIPO: ${fileType || 'texto'}
REQUISITOS: ${prompt}

Genera el contenido completo del archivo. No pongas explicaciones, solo el contenido.`;
    }

    try {
        console.log(`📝 [generate-file] op:${op} tipo:${fileType} prompt:${prompt?.slice(0,60)}`);
        const thought = await activeBrain.process(enrichedPrompt, [], null, {
            userId, email: userEmail, isVip, isCreator,
            plan: planStatus.plan,
            fileGenerationMode: true,
            unlimitedOutput: true
        });

        const content = thought.response || thought.message || '';

        // Guardar en disco si es un archivo real (no solo análisis)
        let savedFileName = null;
        let downloadUrl   = null;
        if (op !== 'analyze') {
            const safeName = (fileName || `nexus_${Date.now()}.${fileType || 'txt'}`).replace(/[^a-zA-Z0-9._-]/g, '_');
            const outPath  = path.join(GENERATED_DIR, safeName);
            // Extraer solo el código si viene envuelto en bloques markdown
            let cleanContent = content;
            const codeMatch = content.match(/```[\w]*\n?([\s\S]*?)```/);
            if (codeMatch) cleanContent = codeMatch[1];
            await fs.writeFile(outPath, cleanContent, 'utf-8');
            savedFileName = safeName;
            downloadUrl   = `/generated/${safeName}`;

            // Auto-limpiar archivos viejos (> 1 hora)
            setTimeout(async () => {
                try {
                    const files = await fs.readdir(GENERATED_DIR);
                    const now   = Date.now();
                    for (const f of files) {
                        const fp = path.join(GENERATED_DIR, f);
                        const st = await fs.stat(fp);
                        if (now - st.mtimeMs > 60 * 60 * 1000) await fs.unlink(fp).catch(() => {});
                    }
                } catch(e) {}
            }, 5000);
        }

        res.json({
            ok: true, operation: op, fileType, fileName,
            content, savedFileName, downloadUrl,
            lines: content.split('\n').length,
            chars: content.length,
            ts: new Date().toISOString()
        });
    } catch (e) {
        console.error('[generate-file]', e.message);
        res.status(500).json({ error: `Error generando archivo: ${e.message}` });
    }
});

// GET /api/generated/:filename — descargar archivo generado
app.get('/api/generated/:filename', requireAuth, async (req, res) => {
    const safeName = req.params.filename.replace(/[^a-zA-Z0-9._-]/g, '_');
    const filePath = path.join(GENERATED_DIR, safeName);
    try {
        await fs.access(filePath);
        res.download(filePath, safeName);
    } catch (e) {
        res.status(404).json({ error: 'Archivo no encontrado o expirado' });
    }
});

// ══════════════════════════════════════════════════════════════════
//  INICIO
// ══════════════════════════════════════════════════════════════════
async function start() {
    await connectDB();
    for (const d of ['models','data','logs','cache','uploads_tmp','generated']) await fs.mkdir(path.join(__dirname, d), { recursive:true });

    app.listen(PORT, '0.0.0.0', () => {
        console.log(`
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🧠  NEXUS v8.0 — Multimodal + CodeGen + Anti-Fraude + VIP     ║
║                                                                  ║
║   🌐  http://localhost:${PORT.toString().padEnd(47)}║
║   💳  PayPal $${PLAN_PRICE}/mes · Reset automático a medianoche        ║
║   🆓  Free: ${FREE_MSG_PER_DAY} msgs/día · 👑 VIP: ${VIP_ACCOUNTS.length} cuentas permanentes    ║
║   📎  Upload: imágenes, PDF, DOCX, código (50MB max)            ║
║   📝  CodeGen: genera archivos sin límite de tamaño             ║
║   🛡️   Anti-fraude: brute-force, IP flood, tx duplicada,         ║
║       payer blacklist, multi-account, fake tx pattern           ║
║                                                                  ║
║   Creado por: Jhonatan David Castro Galvis                       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝`);

        const SELF_PING_URL = process.env.SELF_PING_URL;
        if (SELF_PING_URL) {
            const ping = async () => { try { await axios.get(`${SELF_PING_URL.replace(/\/$/,'')}/health`,{ timeout:10000 }); } catch {} };
            setTimeout(ping, 30000);
            setInterval(ping, 14 * 60 * 1000);
        }
    });
}

start().catch(err => { console.error('❌ Error al iniciar:', err); process.exit(1); });
module.exports = app;
