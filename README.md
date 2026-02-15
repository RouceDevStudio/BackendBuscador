# 🚀 Backend - Buscador Avanzado Multi-Worker

## 📋 Descripción

Backend completo con Node.js y Express que proporciona una API REST para búsqueda web avanzada con múltiples workers paralelos y sistema de auto-ping.

## ✨ Características del Backend

### 🔍 **Sistema de Búsqueda**
- **8 workers paralelos** que buscan simultáneamente
- **Múltiples motores de búsqueda**: Google, Bing, YouTube, GitHub, Reddit
- **Scraping inteligente** con Cheerio
- **Filtrado automático** de spam y contenido basura
- **Cálculo de relevancia** basado en múltiples factores
- **Eliminación de duplicados**

### 🤖 **Auto-Ping System**
- Sistema automático para mantener el servidor activo
- Configurable vía variable de entorno `SELF_PING_URL`
- Intervalo personalizable (por defecto 14 minutos)
- Endpoints para controlar el auto-ping manualmente
- Ideal para servicios gratuitos que se duermen (Render, Heroku, Railway)

### 🛡️ **Seguridad y Validación**
- Validación de inputs
- Filtrado de contenido malicioso
- CORS configurado
- Manejo de errores robusto
- Timeouts configurables

## 📦 Instalación

### 1. Instalar Node.js
Asegúrate de tener Node.js 14 o superior instalado:
```bash
node --version
```

### 2. Clonar o descargar el proyecto
```bash
# Si usas git
git clone <tu-repositorio>
cd advanced-search-backend

# O simplemente descarga los archivos
```

### 3. Instalar dependencias
```bash
npm install
```

### 4. Configurar variables de entorno

Copia el archivo `.env.example` a `.env`:
```bash
cp .env.example .env
```

Edita el archivo `.env` y configura tus variables:
```env
PORT=3000
SELF_PING_URL=https://tu-app.onrender.com  # ⚠️ IMPORTANTE: Cambia esto
SELF_PING_INTERVAL=840000
```

## 🚀 Uso

### Modo Desarrollo
```bash
npm run dev
```

### Modo Producción
```bash
npm start
```

El servidor se iniciará en `http://localhost:3000` (o el puerto que hayas configurado).

## 📡 API Endpoints

### 1. Health Check
```http
GET /api/health
```

**Respuesta:**
```json
{
  "status": "ok",
  "uptime": 3600,
  "timestamp": "2024-02-15T10:30:00.000Z",
  "workers": 8
}
```

### 2. Realizar Búsqueda
```http
POST /api/search
Content-Type: application/json

{
  "keyword": "python tutorial",
  "filter": "youtube"
}
```

**Parámetros:**
- `keyword` (string, requerido): Palabra clave a buscar
- `filter` (string, opcional): Filtro de sitio
  - Opciones: `all`, `youtube`, `mediafire`, `google`, `github`, `reddit`, `stackoverflow`, `medium`

**Respuesta exitosa:**
```json
{
  "success": true,
  "results": [
    {
      "id": "1-1708000000000-0",
      "title": "Python Tutorial - Full Course for Beginners",
      "description": "Learn Python programming from scratch...",
      "url": "https://youtube.com/watch?v=...",
      "source": "YouTube",
      "relevance": 95,
      "workerId": 1,
      "timestamp": 1708000000000
    }
  ],
  "stats": {
    "totalResults": 156,
    "searchTime": "2.45",
    "workersUsed": 8,
    "timestamp": "2024-02-15T10:30:00.000Z"
  }
}
```

**Respuesta con error:**
```json
{
  "success": false,
  "error": "Keyword es requerido"
}
```

### 3. Obtener Filtros Disponibles
```http
GET /api/filters
```

**Respuesta:**
```json
{
  "filters": ["all", "youtube", "mediafire", "google", "github", "reddit", "stackoverflow", "medium"],
  "description": {
    "all": "Buscar en todos los sitios",
    "youtube": "Buscar solo en YouTube",
    "mediafire": "Buscar solo en MediaFire",
    "google": "Buscar en Google Drive/Docs",
    "github": "Buscar en GitHub",
    "reddit": "Buscar en Reddit",
    "stackoverflow": "Buscar en Stack Overflow",
    "medium": "Buscar en Medium"
  }
}
```

### 4. Estado del Auto-Ping
```http
GET /api/ping/status
```

**Respuesta:**
```json
{
  "active": true,
  "url": "https://tu-app.onrender.com",
  "interval": 840000,
  "intervalMinutes": 14
}
```

### 5. Iniciar Auto-Ping
```http
POST /api/ping/start
```

**Respuesta:**
```json
{
  "message": "Auto-ping iniciado",
  "interval": 840000
}
```

### 6. Detener Auto-Ping
```http
POST /api/ping/stop
```

**Respuesta:**
```json
{
  "message": "Auto-ping detenido"
}
```

## 🔧 Configuración Avanzada

### Variables de Entorno

| Variable | Descripción | Valor por Defecto |
|----------|-------------|-------------------|
| `PORT` | Puerto del servidor | `3000` |
| `SELF_PING_URL` | URL para auto-ping | `https://tu-app.com` |
| `SELF_PING_INTERVAL` | Intervalo de ping (ms) | `840000` (14 min) |
| `USER_AGENT` | User agent para requests | Mozilla/5.0... |
| `REQUEST_TIMEOUT` | Timeout de peticiones (ms) | `10000` |
| `WORKERS_COUNT` | Número de workers | `8` |
| `MAX_RESULTS_PER_WORKER` | Resultados por worker | `20` |

### Configuración del Auto-Ping

El sistema de auto-ping es crucial para mantener tu aplicación activa en servicios gratuitos que la "duermen" después de inactividad.

**Pasos para configurar:**

1. **Obtén la URL de tu aplicación desplegada**
   - Render: `https://tu-app.onrender.com`
   - Heroku: `https://tu-app.herokuapp.com`
   - Railway: `https://tu-app.railway.app`

2. **Configura la variable de entorno**
   ```env
   SELF_PING_URL=https://tu-app.onrender.com
   ```

3. **Ajusta el intervalo si es necesario**
   - Servicios gratuitos suelen dormir después de 15-30 minutos
   - Recomendado: 14 minutos (840000 ms)
   - Mínimo recomendado: 10 minutos
   - Máximo antes de dormir: Depende del servicio

## 🌐 Deployment

### Render (Recomendado)

1. Crea una cuenta en [Render](https://render.com)
2. Crea un nuevo "Web Service"
3. Conecta tu repositorio de GitHub
4. Configuración:
   - **Build Command:** `npm install`
   - **Start Command:** `npm start`
   - **Environment Variables:**
     ```
     SELF_PING_URL=https://tu-app.onrender.com
     SELF_PING_INTERVAL=840000
     ```
5. Deploy!

### Heroku

1. Instala Heroku CLI
2. Comandos:
```bash
heroku login
heroku create tu-app-nombre
git push heroku main

# Configurar variables
heroku config:set SELF_PING_URL=https://tu-app-nombre.herokuapp.com
heroku config:set SELF_PING_INTERVAL=840000
```

### Railway

1. Instala Railway CLI o usa la web
2. Comandos:
```bash
railway login
railway init
railway up

# Configurar variables en el dashboard
```

### Variables de Entorno en Producción

**IMPORTANTE:** Después del deploy, configura `SELF_PING_URL` con la URL real:

```bash
# Render
# Ir a Dashboard → Environment → Add Environment Variable

# Heroku
heroku config:set SELF_PING_URL=https://tu-app.herokuapp.com

# Railway
# Ir a tu proyecto → Variables → Add Variable
```

## 🧪 Testing

### Test Manual con cURL

```bash
# Health check
curl http://localhost:3000/api/health

# Búsqueda
curl -X POST http://localhost:3000/api/search \
  -H "Content-Type: application/json" \
  -d '{"keyword":"python","filter":"github"}'

# Estado del ping
curl http://localhost:3000/api/ping/status

# Iniciar ping
curl -X POST http://localhost:3000/api/ping/start

# Detener ping
curl -X POST http://localhost:3000/api/ping/stop
```

### Test con Postman

1. Importa la colección (puedes crearla con los endpoints de arriba)
2. Configura la variable de entorno `{{base_url}}` = `http://localhost:3000`
3. Ejecuta las peticiones

## 📊 Monitoreo

### Logs en Producción

```bash
# Render
# Ver logs en Dashboard → Logs

# Heroku
heroku logs --tail

# Railway
railway logs
```

### Verificar Auto-Ping

Los logs mostrarán:
```
[2024-02-15T10:30:00.000Z] 🏓 Realizando auto-ping...
✅ Auto-ping exitoso | Status: ok | Uptime: 3600s
```

## 🔒 Seguridad

### Recomendaciones

1. **Rate Limiting:** Agrega rate limiting en producción
   ```bash
   npm install express-rate-limit
   ```

2. **Helmet:** Para headers de seguridad
   ```bash
   npm install helmet
   ```

3. **Variables Sensibles:** Nunca commitees el archivo `.env`

4. **CORS:** Configura CORS solo para dominios específicos en producción

## 🐛 Troubleshooting

### El servidor no inicia
- Verifica que el puerto esté disponible
- Revisa las dependencias: `npm install`
- Verifica la versión de Node.js: `node --version`

### Auto-ping no funciona
- Verifica que `SELF_PING_URL` esté configurado correctamente
- Revisa los logs para ver errores de ping
- Asegúrate de que la URL sea accesible públicamente

### No se obtienen resultados
- Algunas páginas tienen anti-scraping
- Verifica tu conexión a internet
- Algunos sitios pueden bloquear tu IP temporalmente

### Error de CORS
- Verifica la configuración de CORS en `server.js`
- En producción, configura los dominios permitidos

## 📝 Notas Importantes

### Sobre Web Scraping
- El scraping puede estar limitado por los términos de servicio de los sitios
- Usa APIs oficiales cuando estén disponibles
- Implementa delays entre peticiones para ser respetuoso
- Este proyecto es para fines educativos

### Sobre Auto-Ping
- No todos los servicios permiten auto-ping
- Algunos pueden limitar la frecuencia
- Verifica los términos de servicio de tu proveedor
- El auto-ping consume recursos mínimos

## 🤝 Contribuciones

Para contribuir:

1. Fork el proyecto
2. Crea una rama: `git checkout -b feature/mejora`
3. Commit: `git commit -m 'Agrega mejora'`
4. Push: `git push origin feature/mejora`
5. Pull Request

## 📄 Licencia

MIT License - libre para uso personal y comercial

## 👨‍💻 Soporte

Si tienes problemas:
1. Revisa esta documentación
2. Revisa los logs del servidor
3. Verifica las variables de entorno
4. Abre un issue en GitHub

---

**Desarrollado con ❤️ usando Node.js, Express y Claude AI**
