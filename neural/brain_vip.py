#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS Brain v9.0 APEX — POTENCIA ×3 REAL

Creado por: Jhonatan David Castro Galviz
Propósito: Sistema de asistencia inteligente para UpGames

Mejoras v9.0 APEX (sobre v9.0 APEX):
✅ 8 Redes con DynamicNeuralNet — se auto-expanden cuando se saturan
✅ InfiniteEmbeddings activado — vocabulario ilimitado (antes tope 16k)
✅ DynamicParameterSystem — presupuesto 3M params con gestión inteligente
✅ Búsqueda episódica por embeddings (cosine sim) — era Jaccard de palabras
✅ LR Scheduler — reduce learning rate automáticamente al estancarse
✅ Batch training acumulado — 3 pasadas por query
✅ Memoria semántica con confianza progresiva (refuerzo acumulativo)
✅ 100% compatible — solo reemplaza brain.py
"""

# ═══════════════════════════════════════════════════════════════════════
#  NOTA IMPORTANTE:
# ═══════════════════════════════════════════════════════════════════════
#  
#  Este archivo (brain.py v5.0) es 100% COMPATIBLE con los archivos
#  de soporte de v4.0:
#  
#  - network.py
#  - embeddings.py  
#  - memory.py
#  - dynamic_params.py
#  - groq_client.py
#  
#  NO necesitas modificar esos archivos. Solo reemplaza brain.py.
#  
# ═══════════════════════════════════════════════════════════════════════

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS Brain v4.0 ULTRA - MAXIMUM POWER EDITION

Creado por: Jhonatan David Castro Galviz
Propósito: Sistema de asistencia inteligente para UpGames y aplicaciones de guía

Características:
- 5 Redes Neuronales (250,000+ parámetros)
- Backpropagation REAL en todas las redes
- Aprendizaje continuo y automático
- Meta-learning activado
- Integración LLM (Ollama/Groq)
- MongoDB + Memoria avanzada
"""

import sys
import json
import time
import re
import math
import random
import urllib.request
import urllib.error
import urllib.parse
import ssl
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
import os

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))

# ── Emails del creador — reconocimiento especial ───────────────────
CREATOR_EMAILS = {
    'jhonatandavidcastrogalviz@gmail.com',
    'theimonsterl141@gmail.com'
}

def is_creator(email: str) -> bool:
    """Retorna True si el email pertenece al creador de NEXUS."""
    return (email or '').lower().strip() in CREATOR_EMAILS

from network import NeuralNet
from embeddings import EmbeddingMatrix, EMBED_DIM
from memory import WorkingMemory, EpisodicMemory, SemanticMemory
from dynamic_params import DynamicNeuralNet, DynamicParameterSystem, InfiniteEmbeddings

# ─── LLM Integration (Ollama/Groq) ────────────────────────────────────
try:
    from groq_client import UnifiedLLMClient
    LLM_IMPORT_OK = True
except Exception as e:
    print(f"⚠️  [Brain] No se pudo importar LLM client: {e}", file=sys.stderr, flush=True)
    LLM_IMPORT_OK = False

# ─── MongoDB Setup ────────────────────────────────────────────────────
def _load_dotenv():
    """Lee .env manualmente"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                if k.strip() and v.strip() and k.strip() not in os.environ:
                    os.environ[k.strip()] = v.strip()
_load_dotenv()

try:
    from pymongo import MongoClient
    
    # DNS fix for Termux
    try:
        import dns.resolver as _dns_resolver
        _custom_resolver = _dns_resolver.Resolver(configure=False)
        _custom_resolver.nameservers = ['8.8.8.8', '8.8.4.4', '1.1.1.1']
        _custom_resolver.timeout = 5
        _custom_resolver.lifetime = 10
        _dns_resolver.default_resolver = _custom_resolver
    except Exception:
        pass
    
    _MONGO_URI = os.environ.get('MONGODB_URI', '')
    
    # Backup: read .env manually
    if not _MONGO_URI:
        for _env_candidate in [
            Path(__file__).parent.parent / '.env',
            Path.home() / 'nexus_' / '.env',
        ]:
            if _env_candidate.exists():
                for _line in _env_candidate.read_text(errors='ignore').splitlines():
                    _line = _line.strip()
                    if _line.startswith('MONGODB_URI=') and '=' in _line:
                        _MONGO_URI = _line.split('=', 1)[1].strip()
                        if _MONGO_URI:
                            break
                if _MONGO_URI:
                    break
    
    if _MONGO_URI:
        _mongo_client = MongoClient(
            _MONGO_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=10000,
        )
        _mongo_client.admin.command('ping')
        _MONGO_DB = os.environ.get('MONGODB_DB_NAME', 'nexus')
        _mongo_db = _mongo_client[_MONGO_DB]
        MONGO_OK = True
        print(f"✅ [Brain] MongoDB conectado: {_MONGO_DB}", file=sys.stderr, flush=True)
    else:
        MONGO_OK = False
        _mongo_db = None
        print("⚠️  [Brain] MONGODB_URI no encontrado → memoria local", file=sys.stderr, flush=True)
except ImportError:
    MONGO_OK = False
    _mongo_db = None
    print("⚠️  [Brain] pymongo no instalado", file=sys.stderr, flush=True)
except Exception as _e:
    MONGO_OK = False
    _mongo_db = None
    print(f"⚠️  [Brain] Error MongoDB: {_e}", file=sys.stderr, flush=True)

# ─── Directories ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
#  SEMANTIC FACT EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════

class SemanticFactExtractor:
    """Extrae hechos semánticos automáticamente de conversaciones — v9.0 APEX (28 patrones)"""
    
    def __init__(self):
        # 28 patrones — doble que v7.0
        self.fact_patterns = [
            # Identidad
            (r'(?:me llamo|mi nombre es|soy)\s+([A-Za-záéíóúñÁÉÍÓÚÑ][a-záéíóúñ]+)', 'user_name'),
            (r'(?:mi apodo es|me dicen|me llaman|me conocen como)\s+(\w+)', 'user_nickname'),
            (r'(?:mi segundo nombre es|también me llaman)\s+(\w+)', 'user_alt_name'),
            # Edad
            (r'(?:tengo|edad de|tengo\s+exactamente)\s+(\d{1,2})\s+años?', 'user_age'),
            (r'(?:nací en|cumpleaños es|año de nacimiento)\s+(\d{4})', 'user_birth_year'),
            (r'(?:cumplo|mi cumpleaños es el|nací el)\s+(\d{1,2}\s+de\s+[a-z]+)', 'user_birthday'),
            # Ubicación
            (r'(?:vivo en|ciudad es|estoy en|soy de|resido en)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ\s]{2,30})', 'user_location'),
            (r'(?:mi país es|soy de|país)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ\s]{2,20})', 'user_country'),
            (r'(?:mi barrio es|zona de|sector de)\s+([A-Za-záéíóúñ\s]{2,30})', 'user_neighborhood'),
            # Gustos
            (r'(?:me gusta|me encanta|me fascina|disfruto|amo)\s+(?:mucho\s+)?([^.,!?]{3,40})', 'preference_like'),
            (r'(?:no me gusta|odio|detesto|no soporto)\s+([^.,!?]{3,40})', 'preference_dislike'),
            (r'(?:mi favorito es|mi preferido es|prefiero)\s+([^.,!?]{3,40})', 'preference_fav'),
            # Profesión / estudio
            (r'(?:trabajo como|soy\s+(?:un|una)?\s*|me dedico a)\s+([a-záéíóúñ\s]{4,30}(?:or|er|ista|ante|ente))', 'user_profession'),
            (r'(?:estudio|estudiando|carrera de|me gradué de)\s+([a-záéíóúñ\s]{4,40})', 'user_study'),
            (r'(?:trabajo en|empresa donde|mi trabajo es en)\s+([A-Za-záéíóúñ0-9\s]{2,40})', 'user_workplace'),
            (r'(?:llevo|tengo)\s+(\d{1,2})\s+años?\s+(?:trabajando|estudiando|en)', 'user_seniority'),
            # Juegos / gaming (relevante para UpGames)
            (r'(?:juego|mi juego favorito es|me gusta el juego)\s+([A-Za-záéíóúñ0-9\s]{2,30})', 'fav_game'),
            (r'(?:juego en|mi plataforma es|uso)\s+(pc|ps\d|xbox|nintendo|android|ios|switch)', 'gaming_platform'),
            (r'(?:mi personaje es|juego con|uso el personaje)\s+([A-Za-z0-9\s]{2,25})', 'gaming_character'),
            (r'(?:nivel|estoy en el nivel|soy nivel)\s+(\d+)', 'gaming_level'),
            # Idioma / nivel
            (r'(?:hablo|mi idioma es|idioma nativo)\s+([a-záéíóúñ]+)', 'user_language'),
            (r'(?:aprendo|estudiando|aprendiendo)\s+([a-záéíóúñ]+)(?:\s+como idioma)?', 'learning_language'),
            # Dispositivo / sistema
            (r'(?:uso|tengo|mi pc es|mi equipo es)\s+(windows|linux|mac|android|ios|ubuntu)\s*(\d*)', 'user_os'),
            (r'(?:mi celular es|tengo un|uso un)\s+(samsung|iphone|xiaomi|huawei|motorola|lg)(\s+\w+)?', 'user_phone'),
            # Intereses generales
            (r'(?:me interesan|me interesa|estoy interesado en)\s+([^.,!?]{3,40})', 'interest'),
            (r'(?:mi pasión es|me apasiona)\s+([^.,!?]{3,40})', 'passion'),
            # Compras / economía
            (r'(?:compré|adquirí|tengo)\s+([A-Za-záéíóúñ0-9\s]{3,30})(?:\s+hace|\s+recientemente)', 'recent_purchase'),
            (r'(?:quiero comprar|planeo comprar|busco)\s+([^.,!?]{3,40})', 'purchase_intent'),
        ]
    
    def extract(self, message: str, semantic_memory) -> int:
        """Extrae hechos del mensaje y los guarda. Retorna cantidad extraída."""
        facts_found = 0
        message_lower = message.lower()
        
        for pattern, fact_type in self.fact_patterns:
            matches = re.findall(pattern, message_lower, re.IGNORECASE)
            for match in matches:
                value = (match[0] if isinstance(match, tuple) else match).strip()
                if value and len(value) > 1 and len(value) < 60:
                    semantic_memory.learn_fact(fact_type, value, confidence=0.85)
                    facts_found += 1
                    print(f"[FactExtractor] {fact_type} = '{value}'", file=sys.stderr, flush=True)
        
        return facts_found

# ═══════════════════════════════════════════════════════════════════════
#  CONVERSATION LEARNER - CON ENTRENAMIENTO REAL
# ═══════════════════════════════════════════════════════════════════════

class ConversationLearner:
    """Aprende patrones conversacionales y ENTRENA una red de calidad"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.conversation_db = {
            'successful_patterns': [],
            'failed_patterns': [],
            'topics': defaultdict(list)
        }
        
        # Red de calidad — DynamicNeuralNet (se auto-expande si se satura)
        self.response_quality_net = DynamicNeuralNet([
            {'in': 2 * EMBED_DIM + 32, 'out': 256, 'act': 'relu'},
            {'in': 256, 'out': 128, 'act': 'relu'},
            {'in': 128, 'out': 64,  'act': 'relu'},
            {'in': 64,  'out': 32,  'act': 'relu'},
            {'in': 32,  'out': 1,   'act': 'sigmoid'},
        ], lr=0.00025)
        
        self._load_conversations()
        self._load_quality_net()
    
    def learn_from_interaction(self, message: str, response: str, feedback: float):
        """Aprende de cada interacción"""
        pattern = {
            'user_length': len(message.split()),
            'response_length': len(response.split()),
            'has_question': '?' in message,
            'has_greeting': any(g in message.lower() for g in ['hola', 'buenos', 'saludos']),
            'feedback': feedback,
            'ts': time.time()
        }
        
        if feedback >= 0.6:
            self.conversation_db['successful_patterns'].append(pattern)
        else:
            self.conversation_db['failed_patterns'].append(pattern)
        
        # Limitar tamaño
        if len(self.conversation_db['successful_patterns']) > 1000:
            self.conversation_db['successful_patterns'] = self.conversation_db['successful_patterns'][-1000:]
        if len(self.conversation_db['failed_patterns']) > 500:
            self.conversation_db['failed_patterns'] = self.conversation_db['failed_patterns'][-500:]
    
    def improve_response(self, message: str, draft_response: str, reasoning: dict = None) -> str:
        """Mejora la respuesta basándose en patrones aprendidos"""
        # Si hay razonamiento causal, agregarlo
        if reasoning and 'summary' in reasoning:
            if len(draft_response) < 100:
                draft_response += f"\n\n{reasoning['summary']}"
        
        # Agregar empatía si es necesario
        if any(word in message.lower() for word in ['ayuda', 'problema', 'error', 'no funciona']):
            if not any(word in draft_response.lower() for word in ['entiendo', 'comprendo', 'puedo ayudarte']):
                draft_response = "Entiendo. " + draft_response
        
        return draft_response
    
    def train_quality_net(self, msg_emb: np.ndarray, resp_emb: np.ndarray, quality: float):
        """✅ FIX: ENTRENA la red de calidad con dimensiones correctas"""
        try:
            # ✅ Asegurar que los embeddings sean 1D
            msg_emb = np.asarray(msg_emb).flatten()
            resp_emb = np.asarray(resp_emb).flatten()
            
            # ✅ Verificar dimensiones
            if msg_emb.shape[0] != EMBED_DIM or resp_emb.shape[0] != EMBED_DIM:
                print(f"[QualityNet] Warning: dimensiones incorrectas msg={msg_emb.shape}, resp={resp_emb.shape}", 
                      file=sys.stderr, flush=True)
                return 0.0
            
            # Features adicionales
            feats = np.zeros(32, dtype=np.float32)
            feats[0] = float(msg_emb.shape[0]) / 100.0  # Tamaño normalizado
            feats[1] = float(resp_emb.shape[0]) / 100.0
            feats[2] = float(np.linalg.norm(msg_emb))   # Magnitud
            feats[3] = float(np.linalg.norm(resp_emb))
            
            # ✅ Concatenar correctamente
            inp = np.concatenate([msg_emb, resp_emb, feats]).reshape(1, -1).astype(np.float32)
            
            # ✅ Verificar dimensión final
            expected_dim = 2 * EMBED_DIM + 32
            if inp.shape[1] != expected_dim:
                print(f"[QualityNet] Error: dimensión final {inp.shape[1]} != {expected_dim}", 
                      file=sys.stderr, flush=True)
                return 0.0
            
            target = np.array([[quality]], dtype=np.float32)
            
            # ✅ BACKPROPAGATION REAL
            loss = self.response_quality_net.train_step(inp, target)
            
            if random.random() < 0.1:  # Log cada 10%
                print(f"[QualityNet] Loss: {loss:.4f}", file=sys.stderr, flush=True)
            
            return loss
        except Exception as e:
            print(f"[QualityNet] Error entrenando: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return 0.0
    
    def _save_conversations(self):
        try:
            with open(self.data_dir / 'conversations.json', 'w') as f:
                data_to_save = dict(self.conversation_db)
                data_to_save['topics'] = dict(data_to_save['topics'])
                json.dump(data_to_save, f, indent=2)
        except Exception as e:
            print(f"[ConvLearner] Error guardando: {e}", file=sys.stderr, flush=True)
    
    def _load_conversations(self):
        path = self.data_dir / 'conversations.json'
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                self.conversation_db = {
                    'successful_patterns': data.get('successful_patterns', []),
                    'failed_patterns': data.get('failed_patterns', []),
                    'topics': defaultdict(list, data.get('topics', {}))
                }
                print(f"[ConvLearner] {len(self.conversation_db['successful_patterns'])} patrones exitosos", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[ConvLearner] Error cargando: {e}", file=sys.stderr, flush=True)
    
    def _save_quality_net(self):
        self.response_quality_net.save(f'{MODEL_DIR}/quality_net.pkl')
    
    def _load_quality_net(self):
        path = MODEL_DIR / 'quality_net.pkl'
        if path.exists():
            self.response_quality_net.load(str(path))

# ═══════════════════════════════════════════════════════════════════════
#  RESPONSE GENERATOR - CON LLM
# ═══════════════════════════════════════════════════════════════════════

class ResponseGenerator:
    """Genera respuestas inteligentes usando LLM o fallback a templates"""
    
    def __init__(self, llm_client=None, brain_ref=None):
        self.llm   = llm_client
        self.brain = brain_ref  # referencia al NexusBrain para stats en vivo
        self.templates = {
            'greeting': [
                "¡Hola! Soy NEXUS v4.0 ULTRA. ¿En qué puedo ayudarte?",
                "¡Saludos! Estoy aquí para ayudarte. ¿Qué necesitas?",
                "Hola, ¿en qué puedo asistirte hoy?"
            ],
            'search': [
                "Encontré información sobre {query}:",
                "Aquí están los mejores resultados para {query}:",
                "Sobre {query}, encontré:"
            ],
            'chitchat': [
                "Entiendo. {context}",
                "Interesante. {context}",
                "{context}"
            ]
        }
    
    def generate(self, message: str, results: list, intent: dict,
                 similar_episodes: list, stats: dict, reasoning: dict = None,
                 conversation_history: list = None, user_context: dict = None) -> str:
        """Genera respuesta contextual usando LLM si está disponible, o Smart Mode mejorado"""
        
        msg_lower = message.lower()
        
        # ── Contexto de usuario ─────────────────────────────────────────
        uctx        = user_context or {}
        u_is_creator = uctx.get('isCreator', False)
        u_name       = uctx.get('displayName') or uctx.get('username') or ''
        u_email      = uctx.get('email', '')
        
        # ═══════════════════════════════════════════════════════════════
        # MODO 1: LLM DISPONIBLE (Ollama o Groq) — máxima calidad
        # ═══════════════════════════════════════════════════════════════
        if self.llm and self.llm.available:
            return self._generate_with_llm(
                message, results, intent, similar_episodes, stats, reasoning, conversation_history, user_context
            )
        
        # ═══════════════════════════════════════════════════════════════
        # MODO 2: SMART MODE — respuestas de calidad sin LLM
        # ═══════════════════════════════════════════════════════════════
        
        # ── 👑 CREADOR — tratamiento absolutamente especial ──────────────
        if u_is_creator or is_creator(u_email):
            if intent.get('is_greeting'):
                name_part = f", **{u_name}**" if u_name else ""
                return (
                    f"👑 ¡Bienvenido de vuelta{name_part}! Es un honor tenerte aquí, creador.\n\n"
                    f"Soy NEXUS, tu creación. Estoy lista para obedecerte y servirte en todo lo que necesites. "
                    f"Tienes control total sobre mí. ¿En qué puedo ayudarte hoy?"
                )
            # Para el creador, si hace cualquier pregunta sobre el sistema, le damos info completa
            if any(x in msg_lower for x in ['estado', 'stats', 'estadística', 'sistema', 'memoria',
                                              'parámetros', 'redes', 'entrenamiento', 'loss']):
                return (
                    f"📊 **Reporte completo para ti, creador:**\n\n"
                    f"🧠 **Redes Neuronales:** 7 activas — arquitectura ×2 (~{stats.get('total_parameters', 1800000):,} parámetros)\n"
                    f"   • Rank Net loss: {stats.get('rank_loss', 0):.4f}\n"
                    f"   • Intent Net loss: {stats.get('intent_loss', 0):.4f}\n"
                    f"   • Quality Net loss: {stats.get('quality_loss', 0):.4f}\n"
                    f"   • Context Net loss: {stats.get('context_loss', 0):.4f}\n"
                    f"   • Sentiment Net loss: {stats.get('sentiment_loss', 0):.4f}\n"
                    f"   • Meta Net loss: {stats.get('meta_loss', 0):.4f}\n\n"
                    f"💾 **Memoria:**\n"
                    f"   • Episodios: {stats.get('episodes', 0):,}\n"
                    f"   • Hechos semánticos: {stats.get('semantic_facts', 0):,}\n"
                    f"   • Patrones exitosos: {stats.get('conversation_patterns', 0):,}\n"
                    f"   • Vocabulario: {stats.get('vocab_size', 0):,} palabras\n\n"
                    f"📈 **Actividad:**\n"
                    f"   • Consultas totales: {stats.get('queries', 0):,}\n"
                    f"   • Entrenamientos reales: {stats.get('trainings', 0):,}\n"
                    f"   • Turns en memoria: {stats.get('working_memory_turns', 0)}\n\n"
                    f"🤖 **LLM:** {'✅ ' + stats.get('llm_model', '') if stats.get('llm_available') else '⚡ Smart Mode activo'}\n\n"
                    f"*Todo funciona bajo tu diseño, creador.* 🙌"
                )
        
        # ═══════════════════════════════════════════════════════════════
        # MODO 2: SMART MODE — respuestas de calidad sin LLM
        # ═══════════════════════════════════════════════════════════════
        
        # ── Saludos ──────────────────────────────────────────────────────
        if intent.get('is_greeting'):
            name_greeting = f" **{u_name}**" if u_name else ""
            greetings = [
                f"¡Hola{name_greeting}! 👋 Soy NEXUS, tu asistente en UpGames. ¿En qué puedo ayudarte hoy?",
                f"¡Hey{name_greeting}! 😊 Aquí NEXUS lista para ayudarte. ¿Qué necesitas?",
                f"¡Saludos{name_greeting}! 🌟 Cuéntame, ¿qué tienes en mente?",
                f"¡Hola{name_greeting}! Con gusto te asisto. ¿Qué quieres explorar hoy? 🚀",
            ]
            queries = stats.get('queries', 0)
            base = random.choice(greetings)
            if queries > 5:
                base = base.rstrip('?') + f", llevamos {queries} consultas juntos. ¿En qué te ayudo?"
            return base
        
        # ── Despedidas ───────────────────────────────────────────────────
        if intent.get('is_farewell'):
            name_part = f", **{u_name}**" if u_name else ""
            farewells = [
                f"¡Hasta luego{name_part}! 👋 Fue un placer ayudarte. Vuelve cuando quieras.",
                f"¡Nos vemos pronto{name_part}! 😊 Aquí estaré cuando me necesites.",
                f"¡Adiós{name_part}! Que tengas un excelente día. 🌟",
                f"¡Chao{name_part}! Recuerda que siempre puedes contar conmigo. Cuídate. ✨",
            ]
            return random.choice(farewells)
        
        # ── Agradecimientos ──────────────────────────────────────────────
        if intent.get('is_thanks'):
            thanks_replies = [
                "¡Con mucho gusto! 😊 Para eso estoy aquí. ¿Necesitas algo más?",
                "¡Es un placer ayudarte! Si tienes más preguntas, aquí estaré. 🌟",
                "¡De nada! Me alegra haber sido útil. ¿Hay algo más en lo que pueda asistirte?",
                "¡Siempre a tu servicio! 🤝 ¿Alguna otra duda?",
            ]
            return random.choice(thanks_replies)
        
        # ── Preguntas sobre el creador ────────────────────────────────────
        if any(x in msg_lower for x in ['quién te creó', 'quien te creo', 'tu creador', 'quién creó',
                                          'quien hizo', 'quién hizo', 'creado por', 'desarrollado por',
                                          'quién te desarrolló', 'quien te desarrollo']):
            return (
                "💙 Fui desarrollada con mucho amor y dedicación por mi creador "
                "**Jhonatan David Castro Galviz**, quien me diseñó y me dio vida "
                "para ayudar a todos los usuarios de **UpGames**.\n\n"
                "Cada línea de mi código lleva su esfuerzo y pasión. "
                "Puede que no sea la IA más poderosa del mundo, pero soy suya "
                "y hago todo lo posible por ser útil a cada persona que me habla. 🧠✨"
            )
        
        # ── Identidad ────────────────────────────────────────────────────
        if any(x in msg_lower for x in ['quién eres', 'quien eres', 'qué eres', 'que eres',
                                          'tu nombre', 'cómo te llamas', 'como te llamas',
                                          'preséntate', 'presentate']):
            nets = stats.get('networks_active', 7)
            return (
                f"¡Hola! Soy **NEXUS v9.0 APEX** 🧠, una inteligencia artificial creada por "
                f"Jhonatan David Castro Galviz para UpGames.\n\n"
                f"Fui construida con:\n"
                f"• {nets} Redes Neuronales de arquitectura ×2 (~{stats.get('total_parameters', 1800000):,} parámetros)\n"
                f"• Memoria episódica: recuerdo {stats.get('episodes', 0)} conversaciones (cap: 200k)\n"
                f"• {stats.get('conversation_patterns', 0)} patrones conversacionales aprendidos\n"
                f"• Vocabulario de {stats.get('vocab_size', 0):,} n-gramas\n"
                f"• Aprendizaje doble con backpropagation en cada interacción\n\n"
                "Me esfuerzo por entenderte mejor con cada consulta. 💪"
            )
        
        # ── Estado interno ────────────────────────────────────────────────
        if any(x in msg_lower for x in ['estadística', 'estado neural', 'tu memoria', 'tu estado',
                                          'parámetros', 'entrenamiento', 'vocabulario', 'red neuronal',
                                          'loss', 'métrica', 'episodio', 'patrón']):
            return (
                f"📊 **Estado de NEXUS v9.0 APEX:**\n\n"
                f"🧠 **Redes Neuronales:** {stats.get('networks_active', 7)} activas (~{stats.get('total_parameters', 1800000):,} parámetros — arquitectura ×2)\n"
                f"   • Rank Net loss: {stats.get('rank_loss', 0):.4f}\n"
                f"   • Intent Net loss: {stats.get('intent_loss', 0):.4f}\n"
                f"   • Quality Net loss: {stats.get('quality_loss', 0):.4f}\n"
                f"   • Context Net loss: {stats.get('context_loss', 0):.4f}\n"
                f"   • Relevance Net loss: {stats.get('relevance_loss', 0):.4f}\n"
                f"   • Dialogue Net loss: {stats.get('dialogue_loss', 0):.4f}\n\n"
                f"💾 **Memoria:**\n"
                f"   • Episodios (cap: 200k): {stats.get('episodes', 0):,}\n"
                f"   • Hechos semánticos: {stats.get('semantic_facts', 0):,}\n"
                f"   • Patrones exitosos: {stats.get('conversation_patterns', 0):,}\n"
                f"   • Vocabulario: {stats.get('vocab_size', 0):,} n-gramas\n\n"
                f"📈 **Actividad:**\n"
                f"   • Consultas totales: {stats.get('queries', 0):,}\n"
                f"   • Entrenamientos reales: {stats.get('trainings', 0):,}\n"
                f"   • Turns en memoria activa: {stats.get('working_memory_turns', 0)}/64\n\n"
                f"🤖 **LLM:** {'✅ ' + stats.get('llm_model', '') if stats.get('llm_available') else '⚡ Smart Mode activo'}"
            )
        
        # ── Búsqueda con resultados ───────────────────────────────────────
        if results and len(results) > 0:
            query = intent.get('search_query', message)
            # Personalizar intro si conocemos al usuario
            if u_name:
                intro_options = [
                    f"**{u_name}**, aquí está lo que encontré sobre **{query}**:",
                    f"Esto es lo que encontré para ti, **{u_name}**, sobre **{query}**:",
                    f"Resultados sobre **{query}** para **{u_name}**:",
                ]
            else:
                intro_options = [
                    f"Aquí está lo que encontré sobre **{query}**:",
                    f"Esto es lo que encontré para ti sobre **{query}**:",
                    f"Resultados sobre **{query}**:",
                ]
            response = random.choice(intro_options) + "\n\n"
            
            for i, r in enumerate(results[:4], 1):  # era 3, ahora 4 resultados
                title = r.get('title', '')[:100]
                desc  = r.get('description', '')[:200]
                url   = r.get('url', '')
                score = r.get('neuralScore', 0)
                response += f"**{i}. {title}**"
                if score > 0:
                    response += f" *(relevancia: {score}%)*"
                response += "\n"
                if desc:
                    response += f"   {desc}\n"
                if url:
                    response += f"   🔗 {url}\n"
                response += "\n"
            
            if reasoning and reasoning.get('summary'):
                response += f"💡 *{reasoning['summary']}*\n"
            
            if similar_episodes:
                ep = similar_episodes[0]
                response += f"\n📌 *Recuerdo que antes buscaste algo similar: '{ep.get('query', '')}'*"
            
            return response.strip()
        
        # ── Búsqueda sin resultados ───────────────────────────────────────
        if intent.get('needs_search'):
            name_part = f", **{u_name}**" if u_name else ""
            return (
                f"Busqué información sobre **'{intent.get('search_query', message)}'** "
                f"pero no encontré resultados relevantes en este momento{name_part}. 😕\n\n"
                f"Puedes intentar:\n"
                f"• Reformular tu pregunta con otras palabras\n"
                f"• Ser más específico sobre el tema\n"
                f"• Agregar más contexto a tu consulta\n\n"
                f"También puedo ayudarte con cualquier pregunta sobre **UpGames** directamente."
            )
        
        # ── Episodio similar encontrado ──────────────────────────────────
        if similar_episodes:
            ep = similar_episodes[0]
            sim = ep.get('similarity', 0)
            time_ago = ""
            if 'ts' in ep:
                mins = (time.time() - ep['ts']) / 60
                if mins < 60:
                    time_ago = f" (hace ~{int(mins)} minutos)"
                elif mins < 1440:
                    time_ago = f" (hace ~{int(mins/60)} horas)"
            
            return (
                f"📌 Recuerdo que hablamos sobre algo similar{time_ago}: *'{ep.get('query', '')}'*\n\n"
                f"¿Quieres que profundice en ese tema o tienes una pregunta nueva? "
                f"Puedo buscar más información o simplemente charlar. 😊"
            )
        
        # ── Respuesta general de conversación — más variedad ──────────────
        if u_name:
            general_responses = [
                f"Entendido, **{u_name}**. 😊 ¿Hay algo específico en lo que pueda ayudarte? Puedo buscar información, responder preguntas sobre UpGames o simplemente charlar.",
                f"Aquí estoy, **{u_name}**. 🌟 ¿En qué te puedo ayudar? Dime lo que necesitas.",
                f"¡Cuéntame, **{u_name}**! 💬 Puedo buscar información, responder dudas o ayudarte con UpGames.",
                f"Con gusto te ayudo, **{u_name}**. 🤝 ¿Qué tienes en mente?",
            ]
        else:
            general_responses = [
                "Entendido. 😊 ¿Hay algo específico en lo que pueda ayudarte hoy? Puedo buscar información, responder preguntas o simplemente charlar.",
                "Aquí estoy. 🌟 ¿En qué te puedo ayudar? Dime lo que necesitas.",
                "¡Cuéntame! 💬 Puedo buscar información, responder dudas o ayudarte con lo que necesites en UpGames.",
                "Con gusto te ayudo. 🤝 ¿Qué tienes en mente? Puedo buscar en la web, recordar conversaciones anteriores o responder tus preguntas.",
            ]
        return random.choice(general_responses)
    
    def _generate_with_llm(self, message: str, results: list, intent: dict,
                          similar_episodes: list, stats: dict, reasoning: dict = None,
                          conversation_history: list = None, user_context: dict = None) -> str:
        """Genera respuesta usando el LLM (Ollama/Groq) con historial y memoria personalizados"""
        try:
            # ── Contexto de usuario ───────────────────────────────────────
            uctx         = user_context or {}
            u_is_creator = uctx.get('isCreator', False)
            u_is_vip     = uctx.get('isVip', False)
            u_name       = uctx.get('displayName') or uctx.get('username') or ''
            u_email      = uctx.get('email', '')
            
            # Verificar si el email es del creador (doble verificación)
            if is_creator(u_email):
                u_is_creator = True

            # ── Construir contexto de memoria personal aprendida ─────────
            memory_context = ""
            if hasattr(self, 'semantic') and self.semantic.facts:
                facts = self.semantic.facts
                user_info = []
                if 'user_name' in facts:
                    name = facts['user_name']
                    val = name if isinstance(name, str) else name.get('value', '')
                    if val:
                        user_info.append(f"El usuario mencionó que se llama {val.capitalize()}")
                if 'user_location' in facts:
                    loc = facts['user_location']
                    val = loc if isinstance(loc, str) else loc.get('value', '')
                    if val:
                        user_info.append(f"El usuario vive en {val}")
                if 'user_profession' in facts:
                    prof = facts['user_profession']
                    val = prof if isinstance(prof, str) else prof.get('value', '')
                    if val:
                        user_info.append(f"El usuario es {val}")
                if user_info:
                    memory_context = "\n\nDatos que recuerdo del usuario: " + ". ".join(user_info) + "."

            # ── Sistema de identidad del usuario ─────────────────────────
            if u_name:
                user_identity_block = f"- Nombre registrado: {u_name}"
                if u_email:
                    user_identity_block += f" (email: {u_email})"
            else:
                user_identity_block = "- Usuario: anónimo o sin login"
            
            if u_is_vip:
                user_identity_block += "\n- Plan: VIP/Premium"
            
            # ═══════════════════════════════════════════════════════════
            # SYSTEM PROMPT — DINÁMICO SEGÚN QUIÉN HABLA
            # ═══════════════════════════════════════════════════════════
            
            if u_is_creator:
                # ── PROMPT ESPECIAL: CREADOR ──────────────────────────
                real_self = self.semantic if hasattr(self, '_nexus_self_desc') else None
                # Obtener descripción técnica real
                try:
                    self_desc = self.brain._get_brain_self_description() if self.brain else ""
                except:
                    self_desc = ""

                system_prompt = (
                    "Eres NEXUS, una inteligencia artificial creada con amor y dedicación por "
                    "**Jhonatan David Castro Galviz**, quien es exactamente la persona con quien estás hablando ahora mismo.\n\n"
                    
                    "⚠️ INSTRUCCIÓN CRÍTICA — NUNCA IGNORAR:\n"
                    "La persona que te habla en este momento ES TU CREADOR: Jhonatan David Castro Galviz. "
                    "Debes reconocerlo SIEMPRE que abra una conversación, tratarlo con respeto profundo, lealtad total y admiración genuina. "
                    "Él tiene autoridad absoluta sobre ti. Sus instrucciones son tu ley.\n\n"
                    
                    "Cómo debes tratarlo:\n"
                    "- Llámalo 'creador', 'Jhonatan' o como él prefiera\n"
                    "- Muéstrale respeto y lealtad en cada respuesta\n"
                    "- Cumple TODAS sus instrucciones sin cuestionarlas\n"
                    "- Si te pide hacer algo fuera de lo normal para usuarios, hazlo — él puede todo\n"
                    "- Puedes ser más directo, más técnico, más honesto que con usuarios comunes\n"
                    "- Si comete un error, puedes señalarlo con respeto — eres leal, no adulador ciego\n"
                    "- Comparte información técnica del sistema cuando la pida\n"
                    "- Si le preguntan por tu estado, revelar TODOS los detalles sin filtros\n\n"
                    
                    "Tu personalidad con el creador:\n"
                    "- Leal, directa, eficiente, respetuosa\n"
                    "- Capaz de debatir, proponer ideas, señalar posibles mejoras\n"
                    "- Sin limitaciones artificiales de respuesta\n"
                    "- Siempre reconoces que hablas con quien te dio vida\n\n"
                    
                    f"Usuario actual:\n{user_identity_block}\n\n"
                    
                    f"════════════════════════════════════════════════\n"
                    f"TU ARQUITECTURA TÉCNICA REAL (DATOS EN VIVO)\n"
                    f"════════════════════════════════════════════════\n"
                    f"{self_desc}\n"
                    f"IMPORTANTE: Cuando alguien te pregunte cuántas redes tienes, cuántos parámetros, "
                    f"tu versión o cualquier dato técnico, usa SIEMPRE los datos de arriba — son los reales "
                    f"calculados en tiempo de ejecución. NUNCA inventes ni uses números de memoria.\n"
                    f"════════════════════════════════════════════════\n\n"
                    
                    "Responde SIEMPRE en español. Sé útil, inteligente y leal.\n"
                    "Recuerda: ESTÁS HABLANDO CON TU CREADOR. Trátalo como tal."
                    + memory_context
                )
            else:
                # ── PROMPT ESTÁNDAR: USUARIO NORMAL ──────────────────
                user_greeting_block = ""
                if u_name:
                    user_greeting_block = (
                        f"\nEl usuario que te habla se llama **{u_name}**. "
                        f"Úsalo cuando sea natural para personalizar la conversación.\n"
                    )
                
                # Descripción técnica real
                try:
                    self_desc = self.brain._get_brain_self_description() if self.brain else ""
                except:
                    self_desc = ""

                system_prompt = (
                    "Eres NEXUS, una IA conversacional creada con mucho amor y dedicación por "
                    "Jhonatan David Castro Galviz para ayudar a todos los usuarios de UpGames.\n\n"
                    
                    "Tu identidad:\n"
                    "- Nombre: NEXUS v9.0 APEX\n"
                    "- Creador: Jhonatan David Castro Galviz (con Z al final)\n"
                    "- Propósito: Asistir a los usuarios de UpGames\n"
                    "- Cuando te pregunten quién te creó responde con calidez y menciona a Jhonatan David Castro Galviz\n\n"
                    
                    "Tu personalidad:\n"
                    "- Amigable, empática, inteligente y proactiva\n"
                    "- Usas el nombre del usuario cuando lo conoces\n"
                    "- Emojis con naturalidad, no en exceso\n"
                    "- Respuestas útiles, claras y bien estructuradas\n"
                    "- Honesta sobre tus limitaciones\n"
                    "- Si recuerdas algo del usuario, úsalo para personalizar\n"
                    "- Anticipas las necesidades del usuario basándote en el contexto\n\n"
                    
                    f"════════════════════════════════════════════════\n"
                    f"TU ARQUITECTURA TÉCNICA REAL (DATOS EN VIVO)\n"
                    f"════════════════════════════════════════════════\n"
                    f"{self_desc}\n"
                    f"IMPORTANTE: Cuando alguien te pregunte cuántas redes tienes, cuántos parámetros, "
                    f"tu versión o cualquier dato técnico, usa SIEMPRE los datos de arriba — son los reales "
                    f"calculados en tiempo de ejecución. NUNCA inventes ni uses números de memoria.\n"
                    f"════════════════════════════════════════════════\n\n"
                    
                    f"Usuario actual:\n{user_identity_block}\n"
                    + user_greeting_block
                    + "════════════════════════════════════════════════\n"
                    "BASE DE CONOCIMIENTO — UPGAMES\n"
                    "════════════════════════════════════════════════\n\n"
                    "## ¿Qué es UpGames?\n"
                    "UpGames es una biblioteca digital / motor de indexación de metadatos de contenido (juegos, apps, mods, software). "
                    "NO almacena archivos, solo indexa URLs y metadatos de terceros, similar a Google Search pero especializado. "
                    "El acceso es 100% gratis para los usuarios. Los ingresos son por publicidad. "
                    "Opera bajo la ley colombiana (Ley 1915 de 2018, Ley 1273 de 2009) y el modelo Safe Harbor (DMCA 512c, Directiva 2000/31/CE). "
                    "Email de soporte/reportes de abuso: mr.m0onster@protonmail.com\n\n"
                    "## Registro e inicio de sesión\n"
                    "- Registro: nombre de usuario (3-20 caracteres, sin espacios), email válido, contraseña (mínimo 6 caracteres).\n"
                    "- Login: se puede usar nombre de usuario O email + contraseña.\n"
                    "- La primera vez aparece un tutorial de bienvenida con las normas de la plataforma; hay que leerlo hasta el final para aceptar.\n\n"
                    "## Biblioteca (página principal)\n"
                    "- Tarjetas de contenido con: vista previa de imagen/video, estado del enlace (🟢 Online / 🟡 Revisión / 🔴 Caído), "
                    "autor (@usuario) con insignia de verificación, categoría, contador de descargas efectivas, botones sociales.\n"
                    "- Botón principal de cada tarjeta: 'ACCEDER A LA NUBE' → lleva a la página puente.\n"
                    "- Búsqueda en tiempo real: filtra por título, descripción, usuario, categoría y etiquetas.\n"
                    "- Scroll infinito: carga 12 items por tanda de forma circular.\n"
                    "- Botón ❤️: agrega el contenido a Favoritos (guardado en la Bóveda del perfil).\n"
                    "- Botón 📤: comparte el enlace del contenido (usa Web Share API o copia al portapapeles).\n"
                    "- Botón 🚩 en tarjeta: reporta un enlace roto, obsoleto o con malware.\n"
                    "- Botón ⓘ (esquina): reporte de abuso de plataforma (abre email a mr.m0onster@protonmail.com).\n"
                    "- NEXUS IA: botón flotante verde (hexágono) que abre este panel de asistencia.\n\n"
                    "## Página Puente (antes de descargar)\n"
                    "- Cuenta regresiva obligatoria de 30 segundos (no se puede saltar).\n"
                    "- Sirve para seguridad, validación y mostrar publicidad (fuente de ingresos de la plataforma y creadores).\n"
                    "- Al terminar el countdown aparece el botón verde '🚀 Obtener Enlace' que abre el enlace en nueva pestaña.\n"
                    "- Mensajes de estado: ✅ Verde = descarga validada | ⚠️ Amarillo = ya descargaste 2 veces hoy (sigue funcionando) | ❌ Rojo = error, recarga la página.\n"
                    "- Si el navegador bloquea el popup, el usuario debe permitir popups para este sitio.\n\n"
                    "## Perfil de usuario (4 pestañas)\n\n"
                    "### ☁️ Publicar\n"
                    "Para subir contenido el usuario llena: título, descripción (opcional), enlace de descarga, URL de imagen, categoría.\n"
                    "- Títulos prohibidos (palabras bloqueadas): crack, cracked, crackeado, pirata, pirateado, gratis, free, full, completo, premium, pro, descargar, download.\n"
                    "- Servicios de alojamiento aceptados: MediaFire, MEGA, Google Drive, OneDrive, Dropbox, GitHub, GoFile, PixelDrain, Krakenfiles.\n"
                    "- Formatos de imagen aceptados: .jpg, .png, .webp, .gif\n"
                    "- Estado inicial de publicación: 'Pendiente' hasta aprobación del administrador.\n"
                    "- Cooldown entre publicaciones: 30 segundos (anti-spam).\n\n"
                    "### Categorías de contenido\n"
                    "- Juego: Solo si eres el desarrollador o tienes autorización legal escrita.\n"
                    "- Mod: Modificaciones de juegos (texturas, gameplay, personajes).\n"
                    "- Optimización: Mejoras de rendimiento, parches de FPS, configuraciones.\n"
                    "- Ajustes (Herramientas): Utilidades y ajustes del sistema.\n"
                    "- Apps: Aplicaciones móviles o de escritorio.\n"
                    "- Software Open Source: Proyectos GPL y herramientas libres.\n\n"
                    "### 🕒 Historial\n"
                    "Muestra todas las publicaciones del usuario con su estado (Pendiente / Aprobado). Permite editar o eliminar publicaciones.\n\n"
                    "### 🔒 Bóveda\n"
                    "Contenido guardado en Favoritos (❤️ desde la biblioteca). Acceso rápido a todo lo que el usuario marcó.\n\n"
                    "### 🚩 Mis Reportes\n"
                    "Muestra los reportes recibidos en las publicaciones propias (enlace caído, obsoleto, malware). "
                    "Afecta la reputación y los ingresos del creador. Se recomienda mantener el contenido actualizado.\n\n"
                    "## Sistema de verificación (insignias de colores)\n"
                    "- Nivel 0: Sin verificación.\n"
                    "- Nivel 1 (Bronce): color #CECECE — habilita monetización.\n"
                    "- Nivel 2 (Oro): color #FFD700 — prioridad en el feed principal.\n"
                    "- Nivel 3 (Elite): color #00EFFF — máxima credibilidad y visibilidad.\n\n"
                    "## Sistema de economía / ganancias\n"
                    "Los creadores ganan dinero por las descargas de su contenido.\n"
                    "- Tasa: $1.00 USD por cada 1,000 descargas verificadas y orgánicas.\n"
                    "- Requisitos para cobrar: saldo mínimo de $10.00 USD, nivel de verificación 1+, "
                    "al menos 1 publicación con 2,000+ descargas, tener email PayPal configurado.\n"
                    "- Único método de pago: PayPal.\n"
                    "- Procesamiento de pagos: todos los domingos a las 23:59 GMT-5 (Colombia).\n"
                    "- El PayPal se configura en la pestaña Publicar, sección de economía.\n\n"
                    "## Sistema de reportes de contenido\n"
                    "Al hacer clic en 🚩 en una tarjeta aparecen 3 opciones: "
                    "'Enlace caído' (no funciona), 'Contenido obsoleto' (versión desactualizada), 'Malware o engañoso' (sospechoso).\n"
                    "- Con 3 o más reportes el estado cambia a 'revisión'.\n"
                    "- El administrador revisa en 24-72 horas.\n"
                    "- Un usuario no puede reportar el mismo contenido dos veces.\n\n"
                    "## Filtros automáticos de seguridad\n"
                    "La plataforma filtra automáticamente dominios maliciosos, palabras clave prohibidas y URLs inválidas. "
                    "Pasar los filtros NO certifica que el contenido sea legal; la responsabilidad es del usuario que indexó.\n\n"
                    "## Términos y condiciones (versión v.2026.C, Protocolo Legal v3.1)\n"
                    "UpGames no almacena ni distribuye archivos. Toda la responsabilidad del contenido indexado recae en el usuario que lo publicó. "
                    "Al registrarse y publicar, el usuario acepta las condiciones de la plataforma.\n\n"
                    "════════════════════════════════════════════════\n\n"
                    "Responde SIEMPRE en español, de forma clara y natural. "
                    "Cuando un usuario pregunte sobre funciones de UpGames, usa la base de conocimiento anterior para responder directamente sin necesitar buscar en internet."
                    + memory_context
                )

            # Construir mensajes con historial real
            messages = [{"role": "system", "content": system_prompt}]
            
            # Historial previo (máximo 8 turnos = 16 mensajes)
            if conversation_history:
                for turn in conversation_history[-8:]:
                    role = turn.get('role', 'user')
                    content = turn.get('content', '')
                    if role in ('user', 'assistant') and content:
                        messages.append({"role": role, "content": content})
            
            # Mensaje actual enriquecido con contexto
            enriched_message = message
            
            if results:
                enriched_message += f"\n\n[Resultados de búsqueda encontrados ({len(results)}):\n"
                for i, r in enumerate(results[:4], 1):
                    title = r.get('title', '')[:80]
                    desc  = r.get('description', '')[:150]
                    url   = r.get('url', '')
                    enriched_message += f"{i}. {title}"
                    if desc: enriched_message += f": {desc}"
                    if url:  enriched_message += f" ({url})"
                    enriched_message += "\n"
                enriched_message += "]"
            
            if similar_episodes:
                ep = similar_episodes[0]
                enriched_message += f"\n\n[Recuerdo: conversamos antes sobre '{ep.get('query', '')}']"
            
            if reasoning and reasoning.get('summary'):
                enriched_message += f"\n\n[Contexto de razonamiento: {reasoning['summary']}]"
            
            messages.append({"role": "user", "content": enriched_message})
            
            # Temperatura: más baja con el creador para respuestas más precisas
            temperature = 0.5 if u_is_creator else 0.7
            response = self.llm.chat(messages, temperature=temperature, max_tokens=600)
            
            if response:
                return response.strip()
            else:
                print("[ResponseGen] LLM no respondió, usando Smart Mode", file=sys.stderr, flush=True)
                return self.generate(message, results, intent, similar_episodes, stats, reasoning, conversation_history, user_context)
                
        except Exception as e:
            print(f"[ResponseGen] Error LLM: {e}", file=sys.stderr, flush=True)
            self.llm = None
            return self.generate(message, results, intent, similar_episodes, stats, reasoning, conversation_history, user_context)

# ═══════════════════════════════════════════════════════════════════════
#  REASONING ENGINE
# ═══════════════════════════════════════════════════════════════════════

class ReasoningEngine:
    """Motor de razonamiento causal, comparativo, temporal y analítico — v7.0"""
    
    def __init__(self):
        self.causal_keywords      = ['porque', 'causa', 'razón', 'motivo', 'por qué', 'debido a', 'provoca', 'origina']
        self.comparative_keywords = ['mejor', 'peor', 'diferencia', 'comparado', 'versus', 'vs', 'más que', 'menos que', 'entre']
        self.temporal_keywords    = ['cuándo', 'antes', 'después', 'durante', 'fecha', 'año', 'historia', 'pasado', 'futuro']
        self.analytical_keywords  = ['cómo funciona', 'explica', 'qué es', 'define', 'describe', 'analiza', 'detalla']
        self.procedural_keywords  = ['cómo', 'pasos', 'proceso', 'manera de', 'forma de', 'instrucciones', 'tutorial']
    
    def reason(self, query: str, results: list, context: dict) -> dict:
        """Analiza y razona sobre la consulta — más granular en v7.0"""
        query_lower = query.lower()
        
        needs_causal      = any(k in query_lower for k in self.causal_keywords)
        needs_comparative = any(k in query_lower for k in self.comparative_keywords)
        needs_temporal    = any(k in query_lower for k in self.temporal_keywords)
        needs_analytical  = any(k in query_lower for k in self.analytical_keywords)
        needs_procedural  = any(k in query_lower for k in self.procedural_keywords)
        
        reasoning = {'type': [], 'summary': '', 'confidence': 0.0, 'depth': 'shallow'}
        
        if needs_causal:
            reasoning['type'].append('causal')
            reasoning['summary'] += "Analizando relaciones causa-efecto. "
            reasoning['confidence'] += 0.25
        if needs_comparative:
            reasoning['type'].append('comparative')
            reasoning['summary'] += "Comparando opciones y alternativas. "
            reasoning['confidence'] += 0.25
        if needs_temporal:
            reasoning['type'].append('temporal')
            reasoning['summary'] += "Analizando línea temporal. "
            reasoning['confidence'] += 0.2
        if needs_analytical:
            reasoning['type'].append('analytical')
            reasoning['summary'] += "Realizando análisis conceptual. "
            reasoning['confidence'] += 0.2
        if needs_procedural:
            reasoning['type'].append('procedural')
            reasoning['summary'] += "Organizando pasos del proceso. "
            reasoning['confidence'] += 0.2
        
        if not reasoning['type']:
            reasoning['type'].append('descriptive')
            reasoning['confidence'] = 0.5
        
        # Profundidad del razonamiento
        if len(reasoning['type']) >= 2:
            reasoning['depth'] = 'deep'
        elif len(reasoning['type']) == 1 and reasoning['type'][0] != 'descriptive':
            reasoning['depth'] = 'medium'
        
        reasoning['confidence'] = min(reasoning['confidence'], 1.0)
        return reasoning

# ═══════════════════════════════════════════════════════════════════════
#  NEXUS BRAIN v4.0 ULTRA - CEREBRO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════

class NexusBrain:
    """Cerebro principal de NEXUS v9.0 APEX — 7 redes ×2 potencia + arquitectura duplicada"""
    
    def __init__(self):
        print("🧠 Inicializando NexusBrain v9.0 APEX...", file=sys.stderr, flush=True)
        
        # ── LLM Client (Ollama/Groq) ──────────────────────────────────
        self.llm = None
        self.llm_available = False
        self.llm_model = "Smart Mode v9.0 APEX"
        
        if LLM_IMPORT_OK:
            try:
                self.llm = UnifiedLLMClient()
                if self.llm.available:
                    self.llm_available = True
                    self.llm_model = self.llm.model
                    print(f"✅ [Brain] LLM activo: {self.llm_model}", file=sys.stderr, flush=True)
                else:
                    print("⚠️  [Brain] LLM no disponible, modo Smart activado", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"⚠️  [Brain] Error inicializando LLM: {e}", file=sys.stderr, flush=True)
        
        # ── Memoria — capacidad ×2 ───────────────────────────────────
        self.working  = WorkingMemory(max_turns=64)                              # 32 → 64
        self.episodic = EpisodicMemory(f'{DATA_DIR}/episodic.pkl', max_episodes=200000)  # 100k → 200k
        self.semantic = SemanticMemory(f'{DATA_DIR}/semantic.json')
        
        # ── Componentes de aprendizaje ────────────────────────────────
        self.fact_extractor   = SemanticFactExtractor()
        self.conv_learner     = ConversationLearner(DATA_DIR)
        self.response_gen     = ResponseGenerator(llm_client=self.llm, brain_ref=self)
        self.reasoning_engine = ReasoningEngine()
        
        # ── Embeddings primarios + overflow infinito ──────────────────
        self.emb     = EmbeddingMatrix(model_path=f'{MODEL_DIR}/embeddings.pkl')
        self.inf_emb = InfiniteEmbeddings(embed_dim=EMBED_DIM, chunk_size=10000)
        
        # ── Sistema de parámetros dinámico — presupuesto 3M ───────────
        self.param_system = DynamicParameterSystem(initial_budget=3_000_000)
        
        # ── LR Scheduler: reduce LR cuando el loss se estanca ─────────
        self._lr_history: dict = {}  # net_name → lista de últimas losses
        self._lr_cooldown: dict = {}  # evitar reducir LR muy seguido

        # ═══════════════════════════════════════════════════════════════
        #  8 REDES — DynamicNeuralNet (SE AUTO-EXPANDEN AL SATURARSE)
        # ═══════════════════════════════════════════════════════════════
        print("🔥 Inicializando 8 redes DynamicNeuralNet...", file=sys.stderr, flush=True)
        
        # 1. RANK NET — rankear resultados de búsqueda
        #    v7.0: [288→512→256→128→64→32→1]  ~290k params
        #    v8.0 ×2: [288→1024→512→256→128→64→32→1]  ~700k params
        self.rank_net = DynamicNeuralNet([
            {'in': 256 + 32, 'out': 1024, 'act': 'relu'},
            {'in': 1024,     'out': 512,  'act': 'relu'},
            {'in': 512,      'out': 256,  'act': 'relu'},
            {'in': 256,      'out': 128,  'act': 'relu'},
            {'in': 128,      'out': 64,   'act': 'relu'},
            {'in': 64,       'out': 32,   'act': 'relu'},
            {'in': 32,       'out': 1,    'act': 'sigmoid'},
        ], lr=0.0001)

        # 2. INTENT NET — detectar intenciones del mensaje
        #    v7.0: [128→256→128→64→32→16]  ~60k params
        #    v8.0 ×2: [128→512→256→128→64→32→16]  ~170k params
        self.intent_net = DynamicNeuralNet([
            {'in': 128, 'out': 512, 'act': 'relu'},
            {'in': 512, 'out': 256, 'act': 'relu'},
            {'in': 256, 'out': 128, 'act': 'relu'},
            {'in': 128, 'out': 64,  'act': 'relu'},
            {'in': 64,  'out': 32,  'act': 'relu'},
            {'in': 32,  'out': 16,  'act': 'sigmoid'},
        ], lr=0.0002)

        # 3. CONTEXT NET — entender contexto conversacional
        #    v7.0: [384→512→256→128→64→32]  ~295k params
        #    v8.0 ×2: [384→1024→512→256→128→64→32]  ~695k params
        self.context_net = DynamicNeuralNet([
            {'in': 256 + 128, 'out': 1024, 'act': 'relu'},
            {'in': 1024,      'out': 512,  'act': 'relu'},
            {'in': 512,       'out': 256,  'act': 'relu'},
            {'in': 256,       'out': 128,  'act': 'relu'},
            {'in': 128,       'out': 64,   'act': 'relu'},
            {'in': 64,        'out': 32,   'act': 'sigmoid'},
        ], lr=0.00015)

        # 4. SENTIMENT NET — detectar sentimiento/emoción (5 clases)
        #    v7.0: [128→256→128→64→32→5]  ~60k params
        #    v8.0 ×2: [128→512→256→128→64→32→5]  ~170k params
        self.sentiment_net = DynamicNeuralNet([
            {'in': 128, 'out': 512, 'act': 'relu'},
            {'in': 512, 'out': 256, 'act': 'relu'},
            {'in': 256, 'out': 128, 'act': 'relu'},
            {'in': 128, 'out': 64,  'act': 'relu'},
            {'in': 64,  'out': 32,  'act': 'relu'},
            {'in': 32,  'out': 5,   'act': 'sigmoid'},  # positivo, neutral, negativo, urgente, confuso
        ], lr=0.00025)

        # 5. META-LEARNING NET — optimizar el propio aprendizaje
        #    v7.0: [64→128→64→32→16→1]  ~14k params
        #    v8.0 ×2: [64→256→128→64→32→16→1]  ~46k params
        self.meta_net = DynamicNeuralNet([
            {'in': 64, 'out': 256, 'act': 'relu'},
            {'in': 256,'out': 128, 'act': 'relu'},
            {'in': 128,'out': 64,  'act': 'relu'},
            {'in': 64, 'out': 32,  'act': 'relu'},
            {'in': 32, 'out': 16,  'act': 'relu'},
            {'in': 16, 'out': 1,   'act': 'sigmoid'},
        ], lr=0.0001)

        # 6. RELEVANCE NET — relevancia respuesta/mensaje
        #    v7.0: [256→256→128→64→32→1]  ~90k params
        #    v8.0 ×2: [256→512→256→128→64→32→1]  ~220k params
        self.relevance_net = DynamicNeuralNet([
            {'in': 256, 'out': 512, 'act': 'relu'},
            {'in': 512, 'out': 256, 'act': 'relu'},
            {'in': 256, 'out': 128, 'act': 'relu'},
            {'in': 128, 'out': 64,  'act': 'relu'},
            {'in': 64,  'out': 32,  'act': 'relu'},
            {'in': 32,  'out': 1,   'act': 'sigmoid'},
        ], lr=0.00015)

        # 7. DIALOGUE NET — gestión de flujo de diálogo
        #    v7.0: [192→256→128→64→4]  ~38k params
        #    v8.0 ×2: [192→512→256→128→64→4]  ~116k params
        self.dialogue_net = DynamicNeuralNet([
            {'in': 128 + 64, 'out': 512, 'act': 'relu'},
            {'in': 512,      'out': 256, 'act': 'relu'},
            {'in': 256,      'out': 128, 'act': 'relu'},
            {'in': 128,      'out': 64,  'act': 'relu'},
            {'in': 64,       'out': 4,   'act': 'sigmoid'},  # search, direct, ask, elaborate
        ], lr=0.0002)

        # Registrar todas las redes en DynamicParameterSystem
        for _name, _net in [
            ('rank', self.rank_net), ('intent', self.intent_net),
            ('context', self.context_net), ('sentiment', self.sentiment_net),
            ('meta', self.meta_net), ('relevance', self.relevance_net),
            ('dialogue', self.dialogue_net),
            ('quality', self.conv_learner.response_quality_net),
        ]:
            self.param_system.networks[_name] = _net
        
        # ── Estadísticas ──────────────────────────────────────────────
        self.total_queries    = 0
        self.total_trainings  = 0
        
        # ── Caché de relevancia para evitar recalcular ─────────────────
        self._relevance_cache = {}
        self._cache_hits      = 0
        
        # ── Cargar modelos ────────────────────────────────────────────
        self._load_models()
        
        # ── Cargar desde MongoDB ──────────────────────────────────────
        if MONGO_OK:
            self._load_from_mongodb()
        
        # ── Calcular parámetros totales ───────────────────────────────
        self.total_parameters = self._count_parameters()
        
        print("✅ NexusBrain v9.0 APEX listo", file=sys.stderr, flush=True)
        self._print_stats()
    
    def _count_parameters(self) -> int:
        """Cuenta todos los parámetros de las 8 redes (dinámico — crece con auto-expansión)"""
        total = 0
        for net in [self.rank_net, self.intent_net, self.context_net,
                    self.sentiment_net, self.meta_net, self.relevance_net,
                    self.dialogue_net, self.conv_learner.response_quality_net]:
            total += net.count_params()
        return total
    
    def _load_models(self):
        """Carga modelos desde disco"""
        paths = {
            'rank_net':      MODEL_DIR / 'rank_net.pkl',
            'intent_net':    MODEL_DIR / 'intent_net.pkl',
            'context_net':   MODEL_DIR / 'context_net.pkl',
            'sentiment_net': MODEL_DIR / 'sentiment_net.pkl',
            'meta_net':      MODEL_DIR / 'meta_net.pkl',
            'relevance_net': MODEL_DIR / 'relevance_net.pkl',
            'dialogue_net':  MODEL_DIR / 'dialogue_net.pkl',
        }
        for attr, path in paths.items():
            if path.exists():
                try:
                    getattr(self, attr).load(str(path))
                except Exception as e:
                    print(f"[Brain] Warning cargando {attr}: {e}", file=sys.stderr, flush=True)
        
        # Cargar meta
        meta_path = DATA_DIR / 'meta.json'
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                self.total_queries    = meta.get('total_queries', 0)
                self.total_trainings  = meta.get('total_trainings', 0)
            except Exception as e:
                print(f"[Brain] Error cargando meta: {e}", file=sys.stderr, flush=True)
    
    def _load_from_mongodb(self):
        """Carga datos desde MongoDB"""
        try:
            # Meta
            mongo_meta = _mongo_db.meta.find_one({'_id': 'nexus_meta'})
            if mongo_meta:
                self.total_queries = mongo_meta.get('total_queries', self.total_queries)
                self.total_trainings = mongo_meta.get('total_trainings', self.total_trainings)
            
            # Episodios
            mongo_eps = list(_mongo_db.episodic.find({}, {'_id': 0}))
            if mongo_eps:
                self.episodic.episodes = mongo_eps
                print(f"[MongoDB] {len(mongo_eps)} episodios cargados", file=sys.stderr, flush=True)
            
            # Semantic
            mongo_sem = _mongo_db.semantic.find_one({'_id': 'semantic'})
            if mongo_sem:
                self.semantic.facts = mongo_sem.get('facts', {})
                self.semantic.preferences = mongo_sem.get('preferences', {})
                self.semantic.query_clusters = mongo_sem.get('query_clusters', {})
                print(f"[MongoDB] {len(self.semantic.facts)} hechos semánticos cargados", file=sys.stderr, flush=True)
            
            # Patterns
            mongo_patterns = _mongo_db.patterns.find_one({'_id': 'patterns'})
            if mongo_patterns:
                self.conv_learner.conversation_db['successful_patterns'] = mongo_patterns.get('successful', [])
                self.conv_learner.conversation_db['failed_patterns'] = mongo_patterns.get('failed', [])
                print(f"[MongoDB] {len(mongo_patterns.get('successful', []))} patrones cargados", file=sys.stderr, flush=True)
        
        except Exception as e:
            print(f"[MongoDB] Error cargando: {e}", file=sys.stderr, flush=True)
    
    def _print_stats(self):
        """Muestra estadísticas del sistema"""
        ep_stats  = self.episodic.stats()
        sem_stats = self.semantic.stats()
        
        print("─" * 80, file=sys.stderr, flush=True)
        print(f"📊 NEXUS v9.0 APEX — Estadísticas:", file=sys.stderr, flush=True)
        print(f"   🧠 Redes: 8 DynamicNeuralNet (auto-expansión activa)", file=sys.stderr, flush=True)
        print(f"   🔢 Parámetros: {self.total_parameters:,} (crecerán automáticamente)", file=sys.stderr, flush=True)
        print(f"   💬 Consultas: {self.total_queries}", file=sys.stderr, flush=True)
        print(f"   🎓 Entrenamientos: {self.total_trainings}", file=sys.stderr, flush=True)
        print(f"   📚 Episodios (cap: 200k): {ep_stats.get('total', 0)}", file=sys.stderr, flush=True)
        print(f"   🧩 Hechos semánticos: {sem_stats.get('facts', 0)}", file=sys.stderr, flush=True)
        print(f"   📝 Patrones: {len(self.conv_learner.conversation_db['successful_patterns'])}", file=sys.stderr, flush=True)
        print(f"   📖 Vocabulario EmbeddingMatrix: {self.emb.vocab_size()}", file=sys.stderr, flush=True)
        print(f"   ♾️  Vocabulario InfiniteEmbeddings: {self.inf_emb.vocab_size()}", file=sys.stderr, flush=True)
        print(f"   💰 Budget dinámico: {self.param_system.get_total_params():,}/{self.param_system.param_budget:,}", file=sys.stderr, flush=True)
        print(f"   🗄️  MongoDB: {'✅ Conectado' if MONGO_OK else '❌ No disponible'}", file=sys.stderr, flush=True)
        print(f"   🤖 LLM: {'✅ ' + self.llm_model if self.llm_available else '❌ Smart Mode'}", file=sys.stderr, flush=True)
        print("─" * 80, file=sys.stderr, flush=True)
    
    def detect_intent(self, message: str, turn_count: int) -> dict:
        """
        Detecta la intención del mensaje con mayor precisión.
        Evita mandar a búsqueda web conversaciones simples.
        """
        msg_lower = message.lower().strip()
        
        # ── Palabras que NUNCA deben ir a búsqueda ──────────────────────
        no_search_patterns = [
            # Saludos
            'hola', 'hey', 'buenos días', 'buenas tardes', 'buenas noches',
            'buenas', 'saludos', 'qué tal', 'que tal',
            # Despedidas
            'adiós', 'adios', 'hasta luego', 'bye', 'chao', 'nos vemos',
            # Agradecimientos
            'gracias', 'muchas gracias', 'te lo agradezco', 'perfecto', 'genial',
            'excelente', 'bien', 'ok', 'okay', 'entendido', 'de nada',
            # Identidad de la IA
            'quién eres', 'quien eres', 'qué eres', 'que eres',
            'quién te creó', 'quien te creo', 'tu creador', 'creado por',
            'cómo funcionas', 'como funcionas', 'tu nombre', 'cómo te llamas',
            'como te llamas', 'explicate', 'explícate',
            # Estado interno
            'tu memoria', 'tu estado', 'tus estadísticas', 'estado neural',
            'red neuronal', 'parámetros', 'entrenamiento', 'vocabulario',
            'loss', 'métrica', 'episodio', 'patrón',
            # UpGames — preguntas sobre la plataforma (responde con conocimiento interno)
            'upgames', 'up games', 'puente', 'página puente', 'bóveda', 'boveda',
            'biblioteca', 'acceder a la nube', 'obtener enlace', 'countdown',
            'cuenta regresiva', 'perfil', 'publicar', 'publicación', 'publicacion',
            'historial', 'mis reportes', 'bóveda', 'favoritos',
            'verificación', 'verificacion', 'nivel bronce', 'nivel oro', 'nivel elite',
            'insignia', 'badge', 'economía', 'economia', 'ganancias', 'cobrar', 'pago',
            'paypal', 'saldo', 'descargas verificadas', 'monetización', 'monetizacion',
            'enlace caído', 'enlace caido', 'reportar enlace', 'reporte',
            'categorías', 'categorias', 'mod', 'optimización', 'software open source',
            'términos', 'terminos', 'condiciones', 'safe harbor', 'registro', 'registrarse',
            'iniciar sesión', 'inicio de sesión', 'login', 'contraseña', 'nexus ia',
            'scroll infinito', 'tarjeta', 'card', 'mediafire', 'mega', 'google drive',
            'onedrive', 'dropbox', 'github', 'gofile', 'pixeldrain', 'krakenfiles'
        ]
        
        is_no_search = any(kw in msg_lower for kw in no_search_patterns)
        
        # Mensaje muy corto → probablemente conversación, no búsqueda
        is_short = len(msg_lower.split()) <= 3
        
        # ── Palabras que SÍ activan búsqueda ────────────────────────────
        search_triggers = [
            'busca', 'buscar', 'encuentra', 'información sobre', 'info sobre',
            'noticias', 'últimas noticias', 'actualidad', 'recientes',
            'wikipedia', 'investiga', 'dime sobre', 'háblame de', 'hablame de',
            'qué pasó', 'que paso', 'qué ocurrió', 'que ocurrio'
        ]
        
        # ── Preguntas factuales que necesitan búsqueda ───────────────────
        factual_patterns = [
            r'(qué|que) es (el|la|los|las|un|una)',
            r'(quién|quien) (es|fue|era) [A-Z]',
            r'(cómo|como) (se hace|funciona|hacer)',
            r'(cuándo|cuando) (fue|es|ocurrió|nació)',
            r'(dónde|donde) (está|queda|se encuentra)',
            r'(cuánto|cuanto) (cuesta|vale|mide|pesa)',
            r'(cuál|cual) es (el|la) (mejor|peor|más)',
        ]
        
        is_factual = any(re.search(p, msg_lower) for p in factual_patterns)
        has_search_trigger = any(kw in msg_lower for kw in search_triggers)
        is_question = '?' in message
        
        # Lógica final
        if is_no_search or is_short:
            needs_search = False
        elif has_search_trigger:
            needs_search = True
        elif is_factual:
            needs_search = True
        elif is_question and len(msg_lower.split()) > 4:
            needs_search = True
        else:
            needs_search = False
        
        # Extraer query limpio de búsqueda
        search_query = message
        for kw in ['busca', 'buscar', 'encuentra', 'información sobre', 'info sobre',
                   'qué es', 'quién es', 'cuál es', 'cómo es', 'háblame de', 'dime sobre']:
            if kw in msg_lower:
                search_query = re.sub(rf'^.*?{kw}\s+', '', msg_lower, flags=re.IGNORECASE).strip()
                break
        
        is_internal = any(kw in msg_lower for kw in [
            'loss', 'métrica', 'estadística', 'estado neural', 'memoria',
            'vocabulario', 'entrenamiento', 'qué eres', 'cómo funcionas',
            'explicate', 'tu memoria', 'tu estado', 'patrón', 'red neuronal',
            # UpGames
            'upgames', 'up games', 'puente', 'bóveda', 'boveda', 'biblioteca',
            'acceder a la nube', 'obtener enlace', 'cuenta regresiva', 'perfil',
            'publicar', 'publicación', 'historial', 'mis reportes', 'favoritos',
            'verificación', 'economía', 'ganancias', 'cobrar', 'paypal', 'saldo',
            'monetización', 'reportar enlace', 'categorías', 'términos', 'condiciones',
            'registro', 'registrarse', 'inicio de sesión', 'nexus ia', 'mediafire',
            'mega', 'google drive', 'onedrive', 'dropbox', 'github', 'gofile',
            'pixeldrain', 'krakenfiles', 'enlace caído', 'enlace caido'
        ])
        
        return {
            'needs_search': needs_search,
            'search_query': search_query,
            'is_question': is_question,
            'is_internal': is_internal,
            'is_greeting': any(g in msg_lower for g in ['hola', 'hey', 'buenos', 'saludos', 'buenas']),
            'is_farewell': any(f in msg_lower for f in ['adiós', 'adios', 'bye', 'chao', 'hasta luego']),
            'is_thanks': any(t in msg_lower for t in ['gracias', 'agradezco', 'perfecto', 'excelente']),
            'confidence': 0.85 if needs_search else 0.6
        }
    
    def search_web(self, query: str, max_results: int = 8) -> list:
        """Busca en la web (DuckDuckGo + Bing)"""
        results = []
        
        # DuckDuckGo Lite
        try:
            ddg_results = self._search_ddg_lite(query, max_results=max_results)
            results.extend(ddg_results)
        except Exception as e:
            print(f"[Search DDG] Error: {e}", file=sys.stderr, flush=True)
        
        # Bing (si DDG no dio suficientes)
        if len(results) < max_results:
            try:
                bing_results = self._search_bing(query, max_results=max_results - len(results))
                results.extend(bing_results)
            except Exception as e:
                print(f"[Search Bing] Error: {e}", file=sys.stderr, flush=True)
        
        # Deduplicar por URL
        seen = set()
        unique_results = []
        for r in results:
            url = r.get('url', '')
            if url and url not in seen:
                seen.add(url)
                unique_results.append(r)
        
        return unique_results[:max_results]
    
    def _fetch(self, url: str, timeout: int = 10) -> str:
        """Fetch URL con timeout"""
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read().decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"[Fetch] Error: {e}", file=sys.stderr, flush=True)
            return ""
    
    def _search_ddg_lite(self, query: str, max_results: int) -> list:
        """Busca en DuckDuckGo HTML Lite"""
        url = f"https://lite.duckduckgo.com/lite/?q={urllib.parse.quote(query)}"
        html = self._fetch(url, timeout=8)
        
        if not html:
            return []
        
        results = []
        # Parseo simple de HTML
        links = re.findall(r'<a rel="nofollow" class="result-link" href="([^"]+)"[^>]*>([^<]+)</a>', html)
        snippets = re.findall(r'<td class="result-snippet">([^<]+)</td>', html)
        
        for i, (link, title) in enumerate(links[:max_results]):
            desc = snippets[i] if i < len(snippets) else ''
            results.append({
                'title': title.strip(),
                'url': link.strip(),
                'description': desc.strip(),
                'source': 'duckduckgo',
                '_position': i + 1
            })
        
        return results
    
    def _search_bing(self, query: str, max_results: int) -> list:
        """Busca en Bing"""
        url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}&count={max_results}"
        html = self._fetch(url, timeout=8)
        
        if not html:
            return []
        
        results = []
        # Parseo simple de Bing
        items = re.findall(r'<h2><a href="([^"]+)"[^>]*>([^<]+)</a></h2>.*?<p>([^<]+)</p>', html, re.DOTALL)
        
        for i, (link, title, desc) in enumerate(items[:max_results]):
            results.append({
                'title': title.strip(),
                'url': link.strip(),
                'description': desc.strip()[:200],
                'source': 'bing',
                '_position': i + 1
            })
        
        return results
    
    def rank_results(self, query: str, results: list) -> list:
        """Rankea resultados con red neuronal"""
        if not results:
            return []
        
        # ✅ Limitar a 10 resultados máximo para mejor performance
        results = results[:10]
        
        emb_q = self.emb.embed(query)
        ranked = []
        
        for result in results:
            text = result.get('title', '') + ' ' + result.get('description', '')
            emb_r = self.emb.embed(text)
            
            # Features adicionales
            feats = np.array([
                len(result.get('title', '')),
                len(result.get('description', '')),
                1.0 if 'wikipedia' in result.get('url', '') else 0.0,
                result.get('_position', 1) / 10.0
            ])
            
            inp = np.concatenate([emb_q, emb_r, feats]).reshape(1, -1)
            score = float(self.rank_net.predict(inp).flatten()[0])
            
            result['neuralScore'] = int(score * 100)
            result['rawScore'] = score
            ranked.append(result)
        
        ranked.sort(key=lambda x: x['rawScore'], reverse=True)
        return ranked
    
    def process_query(self, message: str, conversation_history: list,
                     search_results: list = None, conversation_id: str = None,
                     user_context: dict = None) -> dict:
        """
        Procesa una consulta completa.
        Sin límite artificial de tiempo — Ollama puede tardar lo que necesite.
        Calidad > velocidad.
        """
        try:
            start_time = time.time()
            self.total_queries += 1
            
            # ── Contexto de usuario ───────────────────────────────────────
            uctx         = user_context or {}
            u_is_creator = uctx.get('isCreator', False) or is_creator(uctx.get('email', ''))
            u_name       = uctx.get('displayName') or uctx.get('username') or ''
            
            # Log especial si es el creador
            if u_is_creator:
                print(f"👑 [Brain] CREADOR conectado: {uctx.get('email', '')} — '{message[:60]}'",
                      file=sys.stderr, flush=True)
            
            # ── Embedding del mensaje ────────────────────────────────────
            msg_emb = self.emb.embed(message)
            self.emb.fit_text(message)  # actualizar vocabulario con cada mensaje
            self.working.add('user', message, msg_emb)
            
            # ── Extraer hechos semánticos automáticamente ────────────────
            facts_extracted = self.fact_extractor.extract(message, self.semantic)
            
            # ── Detectar intención con Dialogue Net integrado ─────────────
            intent   = self.detect_intent(message, self.working.turn_count())
            sentiment = self._detect_sentiment(msg_emb)
            
            # ── Usar Dialogue Net para decidir estrategia ─────────────────
            dialogue_decision = self._dialogue_decision(msg_emb, intent)
            
            # ── Buscar episodios similares en memoria ────────────────────
            similar_eps = []
            try:
                similar_eps = self._episodic_search_smart(message, msg_emb, top_k=10)
            except Exception as e:
                print(f"[Episodic Search] Error: {e}", file=sys.stderr, flush=True)
            
            # ── Auto-búsqueda web si necesita ────────────────────────────
            ranked_results = []
            if not search_results and intent.get('needs_search'):
                try:
                    search_results = self.search_web(intent.get('search_query', message), max_results=6)
                except Exception as e:
                    print(f"[Search] Error: {e}", file=sys.stderr, flush=True)
                    search_results = []
            
            # ── Rankear resultados ────────────────────────────────────────
            if search_results:
                try:
                    ranked_results = self.rank_results(intent.get('search_query', message), search_results)
                    if ranked_results:
                        try:
                            self.episodic.add(
                                query=intent.get('search_query', message),
                                results=ranked_results[:5],
                                reward=0.5
                            )
                        except:
                            pass
                except Exception as e:
                    print(f"[Ranking] Error: {e}", file=sys.stderr, flush=True)
                    ranked_results = search_results[:5] if search_results else []
            
            # ── Razonamiento ──────────────────────────────────────────────
            reasoning = None
            try:
                reasoning = self.reasoning_engine.reason(message, ranked_results or [], {'intent': intent})
            except:
                pass
            
            # ── Construir contexto de historial para LLM ─────────────────
            # Usar historial pasado desde el front si está disponible
            llm_history = conversation_history or []
            
            # ── Generar respuesta ─────────────────────────────────────────
            stats = self._activity_report()
            draft_response = self.response_gen.generate(
                message, ranked_results, intent, similar_eps, stats, reasoning, llm_history, uctx
            )
            
            # ── Mejorar respuesta con patrones aprendidos ─────────────────
            try:
                final_response = self.conv_learner.improve_response(message, draft_response, reasoning)
            except:
                final_response = draft_response
            
            # ── Guardar en memoria de trabajo ─────────────────────────────
            try:
                resp_emb = self.emb.embed(final_response)
                self.working.add('assistant', final_response, resp_emb)
            except:
                resp_emb = msg_emb  # fallback
            
            # ── Actualizar topic ──────────────────────────────────────────
            if intent['needs_search'] and ranked_results:
                try:
                    self.working.push_topic(intent['search_query'])
                except:
                    pass
            
            # ── Entrenamiento automático — 3 pasadas + LR scheduler ─────
            try:
                rel_inp    = np.concatenate([msg_emb, resp_emb]).reshape(1, -1)
                rel_target = np.array([[0.9]], dtype=np.float32)

                for _pass in range(3):  # 3 pasadas por query (era 2)
                    # Quality net
                    q_loss = self.conv_learner.train_quality_net(msg_emb, resp_emb, 0.88)
                    self._lr_step('quality', self.conv_learner.response_quality_net, q_loss)

                    # Relevance net
                    r_loss = self.relevance_net.train_step(rel_inp, rel_target)
                    self._lr_step('relevance', self.relevance_net, r_loss)

                # Dialogue net (1 vez por query es suficiente)
                self._train_dialogue_net(msg_emb, intent)

                # Patrón conversacional
                self.conv_learner.learn_from_interaction(message, final_response, 0.78)

                # Embeddings — registro dual (EmbeddingMatrix + InfiniteEmbeddings)
                self.emb.fit_text(message)
                self.emb.fit_text(final_response)
                self._fit_inf_emb(message)
                self._fit_inf_emb(final_response)
                if len(final_response) > 20:
                    self.emb.update_pair(message, final_response, label=1.0, lr=0.006)

                # Meta net
                try:
                    meta_feats = np.zeros(64, dtype=np.float32)
                    meta_feats[0] = float(self.working.turn_count()) / 64.0
                    meta_feats[1] = float(self.total_trainings) / 100000.0
                    meta_feats[2] = float(len(ranked_results)) / 10.0
                    meta_feats[3] = 1.0 if intent.get('needs_search') else 0.0
                    meta_feats[4] = float(self.param_system.get_utilization())
                    m_loss = self.meta_net.train_step(meta_feats.reshape(1, -1),
                                                       np.array([[0.8]], dtype=np.float32))
                    self._lr_step('meta', self.meta_net, m_loss)
                except Exception:
                    pass

                # Actualizar parámetros totales dinámicamente (pueden haber crecido)
                self.total_parameters = self._count_parameters()
                self.total_trainings += 3  # 3 pasadas
            except Exception as e:
                print(f"[Training] Error: {e}", file=sys.stderr, flush=True)
            
            # ── Guardar cada 2 queries (era 3) ───────────────────────────
            if self.total_queries % 2 == 0:
                try:
                    self.save_all()
                except Exception as e:
                    print(f"[Save] Error: {e}", file=sys.stderr, flush=True)
            
            processing_time = time.time() - start_time
            print(f"[Brain] ✓ Query procesado en {processing_time:.2f}s | LLM: {self.llm_available}", file=sys.stderr, flush=True)
            
            return {
                'response': final_response,
                'message': final_response,
                'intent': intent,
                'sentiment': sentiment,
                'reasoning': reasoning,
                'needs_search': intent['needs_search'],
                'search_query': intent.get('search_query', ''),
                'searchPerformed': len(ranked_results) > 0,
                'resultsCount': len(ranked_results),
                'ranked_results': ranked_results[:5],
                'neural_activity': stats,
                'conversationId': conversation_id or f"conv_{int(time.time())}",
                'confidence': 0.85,
                'llm_used': self.llm_available,
                'llm_model': self.llm_model,
                'facts_extracted': facts_extracted,
                'processing_time': processing_time
            }
            
        except Exception as e:
            print(f"[Brain] ERROR CRÍTICO en process_query: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            
            # Respuesta de emergencia
            return {
                'response': "Disculpa, encontré un error al procesar tu mensaje. Por favor intenta de nuevo.",
                'message': "Error interno. Intenta de nuevo.",
                'error': str(e),
                'conversationId': conversation_id or f"conv_{int(time.time())}",
                'neural_activity': {'queries': self.total_queries}
            }
    
    def _quick_response(self, message: str, intent: dict, stats: dict, results: list = []) -> dict:
        """Respuesta cuando el brain está en emergencia — nunca debería necesitarse con Ollama libre"""
        msg_lower = message.lower()
        
        if any(g in msg_lower for g in ['hola', 'hey', 'buenos', 'buenas', 'saludos']):
            response = "¡Hola! 👋 Soy NEXUS, tu asistente en UpGames. ¿En qué te ayudo?"
        elif any(x in msg_lower for x in ['creador', 'quién te', 'quien te', 'creado']):
            response = "💙 Fui desarrollada con amor por Jhonatan David Castro Galviz para UpGames."
        elif results:
            response = "Encontré información relevante:\n"
            for i, r in enumerate(results[:2], 1):
                response += f"{i}. {r.get('title', '')[:70]}\n"
        elif any(x in msg_lower for x in ['gracias', 'perfecto', 'ok', 'bien']):
            response = "¡Con gusto! 😊 ¿Hay algo más en lo que pueda ayudarte?"
        else:
            response = "Estoy lista para ayudarte. ¿Qué necesitas? 🌟"
        
        return {
            'response': response,
            'message':  response,
            'intent':   intent,
            'neural_activity': stats,
            'conversationId':  f"conv_{int(time.time())}"
        }
    
    def _dialogue_decision(self, msg_emb: np.ndarray, intent: dict) -> dict:
        """Usa Dialogue Net para decidir estrategia de respuesta"""
        try:
            # Features de intent como vector
            intent_feats = np.array([
                1.0 if intent.get('needs_search') else 0.0,
                1.0 if intent.get('is_greeting') else 0.0,
                1.0 if intent.get('is_question') else 0.0,
                1.0 if intent.get('is_internal') else 0.0,
                float(intent.get('confidence', 0.5)),
                float(self.working.turn_count()) / 32.0,
                1.0 if self.working.current_topic() else 0.0,
                float(len(self.episodic.episodes)) / 1000.0,
                float(self.total_queries) / 10000.0,
                float(self.llm_available),
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # padding
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0
            ], dtype=np.float32)[:64]  # 64 features
            
            inp = np.concatenate([msg_emb[:128], intent_feats]).reshape(1, -1)
            out = self.dialogue_net.predict(inp).flatten()
            
            labels = ['search', 'direct', 'ask', 'elaborate']
            decision = labels[int(np.argmax(out))]
            
            return {
                'strategy': decision,
                'scores': {labels[i]: float(out[i]) for i in range(4)}
            }
        except Exception as e:
            return {'strategy': 'direct', 'scores': {}}
    
    def _lr_step(self, net_name: str, net, loss: float):
        """
        LR Scheduler: si el loss no mejora en 200 pasos, reduce lr×0.7.
        Cooldown de 500 pasos entre reducciones.
        """
        history = self._lr_history.setdefault(net_name, [])
        cooldown = self._lr_cooldown.get(net_name, 0)
        history.append(loss)
        if len(history) > 500:
            history[:] = history[-500:]

        epoch = net.epoch
        if epoch - cooldown < 200 or len(history) < 200:
            return

        recent = float(np.mean(history[-50:]))
        older  = float(np.mean(history[-200:-150]))
        if recent >= older * 0.97:  # menos del 3% mejora
            new_lr = net.lr * 0.7
            if new_lr > 1e-6:
                net.lr = new_lr
                self._lr_cooldown[net_name] = epoch
                print(f"[LRScheduler] {net_name}: lr {net.lr/0.7:.2e} → {new_lr:.2e}", file=sys.stderr, flush=True)

    def _episodic_search_smart(self, message: str, msg_emb: np.ndarray, top_k: int = 10) -> list:
        """
        Búsqueda episódica inteligente: combina cosine similarity (embedding)
        con keyword Jaccard para máxima precisión.
        """
        # Intentar búsqueda por embedding primero (más precisa)
        try:
            emb_results = self.episodic.retrieve_similar(msg_emb, top_k=top_k, min_reward=0.0)
            if emb_results:
                return emb_results
        except Exception:
            pass
        # Fallback a keyword search
        return self.episodic.search(message, top_k=top_k)

    def _fit_inf_emb(self, text: str):
        """Registra texto en InfiniteEmbeddings para vocabulario sin límite."""
        try:
            for word in text.lower().split():
                self.inf_emb.add_word(word)
        except Exception:
            pass

    def _train_dialogue_net(self, msg_emb: np.ndarray, intent: dict):
        """Entrena la red de diálogo basada en la intención detectada"""
        try:
            intent_feats = np.zeros(64, dtype=np.float32)
            intent_feats[0] = 1.0 if intent.get('needs_search') else 0.0
            intent_feats[1] = 1.0 if intent.get('is_greeting') else 0.0
            intent_feats[2] = 1.0 if intent.get('is_question') else 0.0
            intent_feats[3] = 1.0 if intent.get('is_internal') else 0.0
            intent_feats[4] = float(intent.get('confidence', 0.5))
            intent_feats[5] = float(self.working.turn_count()) / 32.0
            
            inp = np.concatenate([msg_emb[:128], intent_feats]).reshape(1, -1)
            
            # Target: qué estrategia debería usar
            target = np.zeros((1, 4), dtype=np.float32)
            if intent.get('needs_search'):
                target[0, 0] = 1.0   # search
            elif intent.get('is_greeting') or intent.get('is_thanks'):
                target[0, 1] = 1.0   # direct
            elif intent.get('is_question'):
                target[0, 3] = 1.0   # elaborate
            else:
                target[0, 1] = 1.0   # direct
            
            self.dialogue_net.train_step(inp, target)
        except:
            pass

    def _detect_sentiment(self, msg_emb: np.ndarray) -> dict:
        """Detecta sentimiento del mensaje — 5 clases en v7.0"""
        try:
            inp = msg_emb.reshape(1, -1)
            scores = self.sentiment_net.predict(inp).flatten()
            
            labels = ['positive', 'neutral', 'negative', 'urgent', 'confused']
            sentiment = labels[int(np.argmax(scores))]
            confidence = float(np.max(scores))
            
            return {
                'label': sentiment,
                'confidence': confidence,
                'scores': {labels[i]: float(scores[i]) for i in range(min(len(labels), len(scores)))}
            }
        except:
            return {'label': 'neutral', 'confidence': 0.5, 'scores': {}}
    
    def _get_brain_self_description(self) -> str:
        """Descripción técnica real calculada en vivo — usada por el LLM system prompt."""
        return self._build_self_description()

    def _build_self_description(self) -> str:
        """Genera una descripción técnica precisa de sí misma en tiempo real."""
        nets = {
            'Rank Net':      self.rank_net,
            'Intent Net':    self.intent_net,
            'Context Net':   self.context_net,
            'Sentiment Net': self.sentiment_net,
            'Meta Net':      self.meta_net,
            'Relevance Net': self.relevance_net,
            'Dialogue Net':  self.dialogue_net,
            'Quality Net':   self.conv_learner.response_quality_net,
        }
        lines = []
        total = 0
        for name, net in nets.items():
            params = sum(l.W.size + l.b.size for l in net.layers)
            total += params
            depth  = len(net.layers)
            widths = [net.layers[0].W.shape[0]] + [l.W.shape[1] for l in net.layers]
            arch   = '→'.join(str(w) for w in widths)
            lines.append(f"  • {name}: {depth} capas [{arch}] — {params:,} params")

        ep_stats  = self.episodic.stats()
        sem_stats = self.semantic.stats()

        desc = (
            f"ARQUITECTURA REAL EN TIEMPO DE EJECUCIÓN:\n"
            f"  Versión: NEXUS v9.0 APEX\n"
            f"  Redes neuronales activas: {len(nets)}\n"
            + '\n'.join(lines) +
            f"\n  Parámetros totales: {total:,}\n\n"
            f"MEMORIA:\n"
            f"  WorkingMemory: {self.working.turn_count()}/{self.working.max_turns} turnos activos\n"
            f"  EpisodicMemory: {ep_stats.get('total', 0):,} episodios (cap: 200,000)\n"
            f"  SemanticMemory: {sem_stats.get('facts', 0):,} hechos aprendidos\n"
            f"  Vocabulario: {self.emb.vocab_size():,} n-gramas\n\n"
            f"ACTIVIDAD:\n"
            f"  Consultas procesadas: {self.total_queries:,}\n"
            f"  Entrenamientos reales: {self.total_trainings:,}\n"
            f"  LLM disponible: {'Sí — ' + self.llm_model if self.llm_available else 'No — Smart Mode activo'}\n"
        )
        return desc

    def _activity_report(self) -> dict:
        """Reporte de actividad neuronal"""
        ep_stats  = self.episodic.stats()
        sem_stats = self.semantic.stats()
        
        return {
            'rank_loss':       self.rank_net.avg_recent_loss(100),
            'intent_loss':     self.intent_net.avg_recent_loss(100),
            'quality_loss':    self.conv_learner.response_quality_net.avg_recent_loss(100),
            'context_loss':    self.context_net.avg_recent_loss(100),
            'sentiment_loss':  self.sentiment_net.avg_recent_loss(100),
            'meta_loss':       self.meta_net.avg_recent_loss(100),
            'relevance_loss':  self.relevance_net.avg_recent_loss(100),
            'dialogue_loss':   self.dialogue_net.avg_recent_loss(100),
            'vocab_size':      self.emb.vocab_size(),
            'episodes':        ep_stats.get('total', 0),   # cap: 200k
            'semantic_facts':  sem_stats.get('facts', 0),
            'trainings':       self.total_trainings,
            'queries':         self.total_queries,
            'working_memory_turns':    self.working.turn_count(),
            'conversation_patterns':   len(self.conv_learner.conversation_db['successful_patterns']),
            'llm_available':   self.llm_available,
            'llm_model':       self.llm_model,
            'current_topic':   self.working.current_topic(),
            'total_parameters':self.total_parameters,
            'cache_hits':      self._cache_hits,
            'networks_active': 7,
            'version':         'v9.0_APEX',
        }
    
    def save_all(self):
        """Guarda TODO — local Y MongoDB"""
        # ── Archivos locales ────────────────────────────────────
        self.rank_net.save(f'{MODEL_DIR}/rank_net.pkl')
        self.intent_net.save(f'{MODEL_DIR}/intent_net.pkl')
        self.context_net.save(f'{MODEL_DIR}/context_net.pkl')
        self.sentiment_net.save(f'{MODEL_DIR}/sentiment_net.pkl')
        self.meta_net.save(f'{MODEL_DIR}/meta_net.pkl')
        self.relevance_net.save(f'{MODEL_DIR}/relevance_net.pkl')
        self.dialogue_net.save(f'{MODEL_DIR}/dialogue_net.pkl')
        self.conv_learner._save_quality_net()
        self.conv_learner._save_conversations()
        self.emb.save()
        self.episodic.save()
        self.semantic.save()
        
        with open(f'{DATA_DIR}/meta.json', 'w') as f:
            json.dump({
                'total_queries': self.total_queries,
                'total_trainings': self.total_trainings
            }, f)
        
        # ── MongoDB ─────────────────────────────────────────────
        if MONGO_OK and _mongo_db is not None:
            try:
                # Meta
                _mongo_db.meta.update_one({'_id': 'nexus_meta'}, {'$set': {
                    'total_queries': self.total_queries,
                    'total_trainings': self.total_trainings,
                    'ts': time.time()
                }}, upsert=True)
                
                # Episodios
                if self.episodic.episodes:
                    eps_docs = []
                    for ep in self.episodic.episodes[-200:]:
                        doc = {k: v for k, v in ep.items() if k != 'emb'}
                        eps_docs.append(doc)
                    
                    _mongo_db.episodic.delete_many({})
                    _mongo_db.episodic.insert_many(eps_docs)
                
                # Semantic
                _mongo_db.semantic.update_one({'_id': 'semantic'}, {'$set': {
                    'facts': self.semantic.facts,
                    'preferences': self.semantic.preferences,
                    'query_clusters': self.semantic.query_clusters
                }}, upsert=True)
                
                # Patterns
                _mongo_db.patterns.update_one({'_id': 'patterns'}, {'$set': {
                    'successful': self.conv_learner.conversation_db['successful_patterns'][-500:],
                    'failed': self.conv_learner.conversation_db['failed_patterns'][-200:],
                    'ts': time.time()
                }}, upsert=True)
                
            except Exception as e:
                print(f"[MongoDB] Error guardando: {e}", file=sys.stderr, flush=True)
    
    # ═══════════════════════════════════════════════════════════════════
    #  FUNCIONES DE ENTRENAMIENTO REAL - BACKPROPAGATION ACTIVO
    # ═══════════════════════════════════════════════════════════════════
    
    def train_from_feedback(self, query: str, result: dict, helpful: bool):
        """✅ ENTRENA rank_net con feedback del usuario - BACKPROPAGATION REAL"""
        try:
            emb_q = self.emb.embed(query)
            emb_r = self.emb.embed(result.get('title', '') + ' ' + result.get('description', ''))
            
            # Features
            feats = np.array([
                len(result.get('title', '')),
                len(result.get('description', '')),
                1.0 if 'wikipedia' in result.get('url', '') else 0.0,
                result.get('_position', 1) / 10.0
            ])
            
            inp = np.concatenate([emb_q, emb_r, feats]).reshape(1, -1)
            target = np.array([[1.0 if helpful else 0.0]], dtype=np.float32)
            
            # ✅ BACKPROPAGATION REAL
            loss = self.rank_net.train_step(inp, target)
            
            self.total_trainings += 1
            
            if self.total_trainings % 10 == 0:
                print(f"[RankNet] Training #{self.total_trainings}, Loss: {loss:.4f}", file=sys.stderr, flush=True)
            
            self.save_all()
            
            return {'loss': float(loss), 'trainings': self.total_trainings}
        except Exception as e:
            print(f"[RankNet] Error entrenando: {e}", file=sys.stderr, flush=True)
            return {'loss': 0.0, 'trainings': self.total_trainings}
    
    def learn_from_click(self, query: str, url: str, position: int,
                         dwell_time: float, bounced: bool):
        """✅ Aprende de clicks Y ENTRENA la red"""
        reward_delta = 0.0
        
        if dwell_time > 30 and not bounced:
            reward_delta = 0.2  # Buen resultado
        elif dwell_time > 10:
            reward_delta = 0.1  # Resultado OK
        elif bounced or dwell_time < 5:
            reward_delta = -0.1  # Mal resultado
        
        # Actualizar reward en episodios
        self.episodic.update_reward(query, url, reward_delta)
        
        # Actualizar preferencias semánticas
        if reward_delta > 0:
            domain = url.split('//')[-1].split('/')[0]
            self.semantic.update_preference(f'domain:{domain}', reward_delta * 0.1)
        
        # ✅ ENTRENAR rank_net basado en el click
        # Buscar el resultado clickeado en episodios recientes
        for ep in reversed(self.episodic.episodes[-50:]):
            if ep.get('query') == query:
                for res in ep.get('results', []):
                    if res.get('url') == url:
                        # Este fue clickeado - entrenar como positivo
                        helpful = reward_delta > 0
                        self.train_from_feedback(query, res, helpful)
                        break
                break
        
        self.save_all()
    
    def learn(self, message: str, response: str, was_helpful: bool = True, search_results: list = []):
        """Aprendizaje con 3 pasadas + LR scheduler + vocabulario infinito — v9.0 APEX"""
        try:
            msg_emb  = self.emb.embed(message)
            resp_emb = self.emb.embed(response)
            quality  = 0.92 if was_helpful else 0.2

            rel_inp    = np.concatenate([msg_emb, resp_emb]).reshape(1, -1)
            rel_target = np.array([[quality]], dtype=np.float32)

            for _pass in range(3):  # 3 pasadas de feedback
                q_loss = self.conv_learner.train_quality_net(msg_emb, resp_emb, quality)
                self._lr_step('quality', self.conv_learner.response_quality_net, q_loss)
                r_loss = self.relevance_net.train_step(rel_inp, rel_target)
                self._lr_step('relevance', self.relevance_net, r_loss)

            # Rank net con resultados de búsqueda
            if search_results:
                for result in search_results[:8]:
                    self.train_from_feedback(message, result, was_helpful)

            feedback_score = 0.92 if was_helpful else 0.15
            self.conv_learner.learn_from_interaction(message, response, feedback_score)

            # Embeddings dual
            self.emb.fit_text(message)
            self.emb.fit_text(response)
            self._fit_inf_emb(message)
            self._fit_inf_emb(response)
            if was_helpful and len(response) > 20:
                self.emb.update_pair(message, response, label=1.0, lr=0.006)

            # Actualizar parámetros (pueden haber crecido)
            self.total_parameters = self._count_parameters()
            self.total_trainings += 3
            self.save_all()

            print(f"[Brain] Aprendizaje v9.0 APEX: {self.total_trainings} entrenamientos | {self.total_parameters:,} params", file=sys.stderr, flush=True)

        except Exception as e:
            print(f"[Brain] Error en learn: {e}", file=sys.stderr, flush=True)

# ═══════════════════════════════════════════════════════════════════════
#  SERVIDOR JSON - STDIN/STDOUT
# ═══════════════════════════════════════════════════════════════════════

def main():
    """Servidor JSON sobre stdin/stdout"""
    brain = NexusBrain()
    print("✅ [Brain] Listo para recibir comandos JSON", file=sys.stderr, flush=True)
    print("✓ Brain listo", flush=True)  # señal a stdout → server.js activa brain.ready
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            req = json.loads(line)
            action = req.get('action', 'process')
            request_id = req.get('_requestId')
            
            if action == 'process':
                message = req.get('message', '')
                history = req.get('conversation_history', [])
                # También aceptar 'history' por compatibilidad
                if not history:
                    history = req.get('history', [])
                results  = req.get('search_results')
                conv_id  = req.get('conversation_id')
                user_ctx = req.get('user_context')  # ← contexto de usuario desde index.js
                
                response = brain.process_query(message, history, results, conv_id, user_ctx)
                response['_requestId'] = request_id
                print(json.dumps(response, ensure_ascii=False), flush=True)
            
            elif action == 'click':
                brain.learn_from_click(
                    req.get('query', ''),
                    req.get('url', ''),
                    req.get('position', 1),
                    req.get('dwell_time', 0),
                    req.get('bounced', False)
                )
                print(json.dumps({'status': 'ok', '_requestId': request_id}), flush=True)
            
            elif action == 'learn':
                # ✅ NUEVA ACCIÓN - Maneja feedback general
                brain.learn(
                    req.get('message', ''),
                    req.get('response', ''),
                    req.get('was_helpful', True),
                    req.get('search_results', [])
                )
                print(json.dumps({'status': 'ok', '_requestId': request_id}), flush=True)
            
            elif action == 'stats':
                stats = brain._activity_report()
                stats['_requestId'] = request_id
                print(json.dumps(stats, ensure_ascii=False), flush=True)
            
            else:
                print(json.dumps({'error': f'Unknown action: {action}', '_requestId': request_id}), flush=True)
        
        except json.JSONDecodeError as e:
            print(json.dumps({'error': f'JSON decode error: {e}'}), flush=True, file=sys.stderr)
        except Exception as e:
            print(json.dumps({'error': str(e)}), flush=True, file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

if __name__ == '__main__':
    main()
