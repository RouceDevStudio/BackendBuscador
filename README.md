# 🧠 NEXUS AI Search v6.0

> Motor de búsqueda de próxima generación con inteligencia artificial neuronal

![NEXUS AI](https://img.shields.io/badge/AI-Neural%20Powered-blue?style=for-the-badge)
![Version](https://img.shields.io/badge/version-6.0-green?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-orange?style=for-the-badge)

## 🚀 Características Revolucionarias

### 🧠 Inteligencia Artificial Neuronal
- **50 Neuronas de Aprendizaje**: Sistema neural primitivo que mejora con cada búsqueda
- **Ranking Inteligente**: Algoritmo de relevancia que aprende de tus preferencias
- **Análisis Semántico**: Entiende el contexto, no solo palabras clave
- **Aprendizaje Continuo**: Se vuelve más inteligente con el tiempo

### 🎨 Diseño Ultra-Moderno
- **Glassmorphism**: Interfaz moderna con efectos de cristal
- **Animaciones Fluidas**: Transiciones suaves y profesionales
- **Modo Oscuro/Claro**: Adaptación automática a tus preferencias
- **Responsive Design**: Perfecto en cualquier dispositivo
- **Bento Layout**: Diseño organizado tipo Google pero mejor

### ⚡ Super Rendimiento
- **Búsqueda Paralela**: Múltiples workers trabajando simultáneamente
- **Cache Inteligente**: Resultados instantáneos para búsquedas repetidas
- **Scraping Optimizado**: DuckDuckGo, Bing, y más motores
- **API RESTful**: Fácil de integrar en tus proyectos

### 🔍 Fuentes de Búsqueda
- DuckDuckGo (privacidad primero)
- Bing (resultados globales)
- YouTube (videos)
- GitHub (código)
- Stack Overflow (programación)
- Wikipedia (conocimiento)
- Reddit (comunidad)
- Archive.org (contenido histórico)
- Y muchas más...

## 📦 Instalación

### Requisitos Previos
- Node.js 16+ 
- Python 3.8+
- npm o yarn

### Paso 1: Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/nexus-ai-search.git
cd nexus-ai-search
```

### Paso 2: Instalar Dependencias Node.js
```bash
npm install
```

### Paso 3: Configurar Python
```bash
# Instalar dependencias Python (si las hubiera)
# No hay dependencias externas por ahora, todo es Python puro
```

### Paso 4: Configurar Variables de Entorno
```bash
cp .env.example .env
# Editar .env según tus necesidades
```

### Paso 5: Iniciar el Servidor
```bash
npm start
```

El servidor estará disponible en `http://localhost:3000`

## 🎯 Uso

### Interfaz Web
1. Abre tu navegador en `http://localhost:3000`
2. Escribe tu búsqueda en la barra principal
3. ¡Disfruta de resultados potenciados por IA!

### API REST

#### Buscar
```bash
POST /api/search
Content-Type: application/json

{
  "keyword": "python tutorial"
}
```

**Respuesta:**
```json
{
  "success": true,
  "results": [
    {
      "title": "Python Tutorial",
      "url": "https://example.com",
      "description": "Learn Python...",
      "source": "Python.org",
      "neuralScore": 95.5
    }
  ],
  "stats": {
    "totalResults": 42,
    "searchTime": "1.23",
    "aiRanked": true
  }
}
```

#### Aprendizaje (cuando el usuario hace click)
```bash
POST /api/learn
Content-Type: application/json

{
  "query": "python tutorial",
  "url": "https://example.com"
}
```

#### Sugerencias Automáticas
```bash
GET /api/suggest?q=pytho

Respuesta:
{
  "suggestions": [
    { "text": "python tutorial", "popularity": 15 },
    { "text": "python projects", "popularity": 8 }
  ]
}
```

#### Estado del Sistema
```bash
GET /api/health

Respuesta:
{
  "status": "online",
  "uptime": 3600,
  "ai": {
    "ready": true,
    "queries": 150,
    "learned": 45
  },
  "version": "6.0-NEXUS"
}
```

## 🧠 Cómo Funciona la IA

### Sistema Neural de 50 Neuronas

NEXUS AI implementa un sistema neural primitivo con 50 neuronas que procesan:

1. **Coincidencia de Título** (35% peso)
   - Exacta: Bonus total
   - Parcial: Proporcional a tokens coincidentes

2. **Coincidencia de Descripción** (20% peso)
   - Análisis semántico del contenido

3. **Relevancia de URL** (15% peso)
   - Presencia de términos en el dominio

4. **Autoridad de Fuente** (10% peso)
   - Wikipedia: 100%
   - GitHub/Stack Overflow: 95%
   - Archive.org: 85%

5. **Frescura del Contenido** (8% peso)
   - Resultados más recientes tienen prioridad

6. **Historial de Usuario** (5% peso)
   - Aprende de tus clicks anteriores

7. **Similitud Semántica** (4% peso)
   - TF-IDF y similitud coseno

8. **Análisis de Intención** (3% peso)
   - Detecta si buscas descargas, código, videos, etc.

### Aprendizaje Continuo

Cada vez que haces click en un resultado:
1. Se registra el patrón query → url
2. Se incrementa el contador de clicks
3. Cada 10 clicks, los pesos neuronales se ajustan
4. El modelo se guarda automáticamente

### Persistencia

El modelo se guarda en `models/brain.pkl` usando pickle de Python, permitiendo:
- Continuidad entre reinicios
- Acumulación de conocimiento
- Mejora progresiva de resultados

## 🎨 Personalización

### Cambiar Colores
Edita las variables CSS en `public/index.html`:
```css
:root {
    --primary: #6366f1;     /* Color principal */
    --secondary: #8b5cf6;   /* Color secundario */
    --accent: #ec4899;      /* Color de acento */
}
```

### Añadir Fuentes de Búsqueda
En `server.js`, añade nuevas funciones de scraping:
```javascript
async function myCustomSource(query) {
    // Tu lógica de scraping aquí
    return results;
}
```

### Ajustar Pesos Neuronales
En `neural/brain.py`, modifica el diccionario `weights`:
```python
self.weights = {
    'title_exact': 0.40,        # Aumentar peso de título
    'user_history': 0.10,       # Aumentar peso de historial
    # ...
}
```

## 📊 Arquitectura

```
nexus-search/
├── server.js              # Servidor Express + integración IA
├── neural/
│   ├── brain.py          # Cerebro neural principal
│   └── __init__.py       # Wrapper para comunicación
├── models/
│   └── brain.pkl         # Modelo entrenado (se genera)
├── public/
│   └── index.html        # Interfaz ultra-moderna
├── package.json          # Dependencias Node.js
├── .env.example          # Configuración de ejemplo
└── README.md             # Esta documentación
```

## 🔧 Desarrollo

### Modo Desarrollo con Auto-Reload
```bash
npm run dev
```

### Testing
```bash
npm test
```

### Estructura de Logs
```
🔍 Searching: "python tutorial"
  Worker 1 ✓ 10
  Worker 2 ✓ 8
  Worker 3 ✓ 12
  Worker 4 ✓ 9
🧠 Neural Ranking: 39 results
✅ 1.23s | 39 resultados
```

## 🚀 Despliegue

### Render / Railway / Heroku
1. Conecta tu repositorio
2. Variables de entorno:
   ```
   PORT=3000
   AI_ENABLED=true
   ```
3. Build Command: `npm install`
4. Start Command: `npm start`

### Docker
```dockerfile
FROM node:18
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
```

## 🆚 NEXUS vs Google

| Característica | NEXUS AI | Google |
|----------------|----------|--------|
| IA Neuronal | ✅ 50 neuronas | ❌ Caja negra |
| Privacidad | ✅ Sin tracking | ❌ Tracking total |
| Open Source | ✅ Código abierto | ❌ Propietario |
| Aprendizaje Local | ✅ En tu máquina | ❌ En sus servers |
| Personalizable | ✅ 100% customizable | ❌ Cerrado |
| Gratuito | ✅ Sin límites | ⚠️ Con anuncios |

## 🛣️ Roadmap

### v6.1 (Próximamente)
- [ ] Búsqueda de imágenes con IA
- [ ] Reconocimiento de voz
- [ ] Traducción automática
- [ ] Modo offline con cache

### v7.0 (Futuro)
- [ ] Red neuronal profunda (200+ neuronas)
- [ ] Embeddings vectoriales
- [ ] Búsqueda multimodal (texto + imagen)
- [ ] Plugin system

### v8.0 (Visión)
- [ ] Transformer-based ranking
- [ ] Generación de respuestas (como ChatGPT)
- [ ] Búsqueda federada P2P
- [ ] Blockchain para privacidad

## 🤝 Contribuir

¡Las contribuciones son bienvenidas!

1. Fork el proyecto
2. Crea tu rama (`git checkout -b feature/amazing`)
3. Commit tus cambios (`git commit -m 'Add amazing feature'`)
4. Push a la rama (`git push origin feature/amazing`)
5. Abre un Pull Request

## 📄 Licencia

MIT License - puedes usar NEXUS para lo que quieras, incluso comercialmente.

## 💬 Soporte

- 📧 Email: support@nexus-ai.dev
- 💬 Discord: [discord.gg/nexus](https://discord.gg/nexus)
- 🐛 Issues: GitHub Issues
- 📚 Docs: [docs.nexus-ai.dev](https://docs.nexus-ai.dev)

## 🌟 Créditos

Desarrollado con ❤️ por el equipo NEXUS

Inspirado por:
- Google Search (para competir contra ellos)
- DuckDuckGo (por la privacidad)
- Neural Networks (por la inteligencia)

## 🎯 Filosofía

> "La mejor manera de predecir el futuro es inventarlo. NEXUS es el futuro de la búsqueda."

NEXUS no es solo un motor de búsqueda, es una declaración de principios:
- **Privacidad primero**: Tus búsquedas son tuyas
- **Open Source**: La tecnología debe ser libre
- **IA Transparente**: Sabes cómo funciona
- **Aprendizaje Local**: La IA está en tu máquina, no en la nube

---

Hecho con 🧠 y ⚡ por desarrolladores que creen en un internet mejor.

**¿Te gusta NEXUS? ¡Dale una ⭐ en GitHub!**
