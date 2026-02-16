// NEXUS AI Search - Configuration
// Configuración centralizada del sistema

module.exports = {
    // Server Configuration
    server: {
        port: process.env.PORT || 3000,
        env: process.env.NODE_ENV || 'development',
        cors: {
            enabled: true,
            origins: process.env.CORS_ORIGINS?.split(',') || ['*']
        }
    },

    // AI Configuration
    ai: {
        enabled: process.env.AI_ENABLED === 'true',
        neurons: parseInt(process.env.NEURONS || '50'),
        modelPath: process.env.MODEL_PATH || 'models/brain.pkl',
        learning: {
            enabled: true,
            autoSave: true,
            saveInterval: 10 // clicks
        }
    },

    // Cache Configuration
    cache: {
        enabled: process.env.CACHE_ENABLED !== 'false',
        maxSize: parseInt(process.env.CACHE_MAX_SIZE || '1000'),
        ttl: parseInt(process.env.CACHE_TTL || '3600'), // 1 hour
        persistent: true
    },

    // Search Configuration
    search: {
        timeout: 14000, // 14 seconds
        maxWorkers: 4,
        resultsPerPage: 20,
        engines: {
            duckduckgo: { enabled: true, weight: 1.0 },
            bing: { enabled: true, weight: 0.9 },
            youtube: { enabled: false, weight: 0.8 },
            github: { enabled: false, weight: 0.95 }
        }
    },

    // User Agents
    userAgents: [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
    ],

    // Logging
    logging: {
        level: process.env.LOG_LEVEL || 'info',
        file: process.env.LOG_FILE || 'logs/nexus.log',
        console: true,
        colorize: true
    },

    // Rate Limiting
    rateLimit: {
        enabled: true,
        windowMs: 15 * 60 * 1000, // 15 minutes
        max: 100 // requests per window
    },

    // Analytics
    analytics: {
        enabled: process.env.ANALYTICS_ENABLED === 'true',
        trackQueries: true,
        trackClicks: true,
        anonymize: true
    },

    // Performance
    performance: {
        compression: true,
        etag: true,
        clustering: process.env.CLUSTERING === 'true'
    }
};
