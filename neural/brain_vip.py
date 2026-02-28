#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS Brain v10.0 APEX — EDICIÓN COMPLETA CORREGIDA

Creado por: Jhonatan David Castro Galviz
Propósito: Sistema de asistencia inteligente para UpGames

Fixes v10.0 (sobre v9.0 APEX):
✅ FIX CRÍTICO: self.semantic accesible en ResponseGenerator via brain_ref (_get_memory_context)
✅ FIX CRÍTICO: def _activity_report() restaurado correctamente
✅ dialogue_decision ahora SE USA para condicionar la estrategia de respuesta
✅ context_net ahora SE ENTRENA en cada query (_train_context_net)
✅ Targets de entrenamiento DINÁMICOS (no hardcodeados al 0.88/0.9)
✅ add_to_cluster() llamado desde process_query (query_clusters ahora crecen)
✅ _fit_inf_emb limpia puntuación antes de tokenizar
✅ save_all() cada 15 queries (era cada 2)
✅ MongoDB guarda hasta 5000 episodios (era solo 200)
✅ _relevance_cache con límite de 2000 entradas (evita fuga de RAM)
✅ Headers duplicados eliminados
✅ except desnudos reemplazados por logging real
✅ memory_context inyecta TODOS los hechos semánticos al LLM
✅ max_tokens=8192 (era 600)
✅ Historial LLM 20 turnos (era 8)
✅ WorkingMemory 128 turnos (era 64)
✅ EpisodicMemory 500k episodios (era 200k)
✅ Patrones conversacionales 10k/5k (era 1k/500)
✅ Episodic top_k=25 (era 10)
"""

import sys
import json
import time
import re
import random
import urllib.request
import urllib.error
import urllib.parse
import numpy as np
from collections import defaultdict
from pathlib import Path
import os

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))

# ── Emails del creador ──────────────────────────────────────────────
CREATOR_EMAILS = {
    'jhonatandavidcastrogalviz@gmail.com',
    'theimonsterl141@gmail.com'
}

def is_creator(email: str) -> bool:
    return (email or '').lower().strip() in CREATOR_EMAILS

from network import NeuralNet
from embeddings import EmbeddingMatrix, EMBED_DIM
from memory import WorkingMemory, EpisodicMemory, SemanticMemory
from dynamic_params import DynamicNeuralNet, DynamicParameterSystem, InfiniteEmbeddings

# ─── LLM ───────────────────────────────────────────────────────────
try:
    from groq_client import UnifiedLLMClient
    LLM_IMPORT_OK = True
except Exception as e:
    print(f"⚠️  [Brain] No se pudo importar LLM client: {e}", file=sys.stderr, flush=True)
    LLM_IMPORT_OK = False

# ─── .env ──────────────────────────────────────────────────────────
def _load_dotenv():
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                if k.strip() and v.strip() and k.strip() not in os.environ:
                    os.environ[k.strip()] = v.strip()
_load_dotenv()

# ─── MongoDB ────────────────────────────────────────────────────────
try:
    from pymongo import MongoClient
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
        _mongo_client = MongoClient(_MONGO_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=10000)
        _mongo_client.admin.command('ping')
        _MONGO_DB  = os.environ.get('MONGODB_DB_NAME', 'nexus')
        _mongo_db  = _mongo_client[_MONGO_DB]
        MONGO_OK   = True
        print(f"✅ [Brain] MongoDB conectado: {_MONGO_DB}", file=sys.stderr, flush=True)
    else:
        MONGO_OK  = False
        _mongo_db = None
        print("⚠️  [Brain] MONGODB_URI no encontrado → memoria local", file=sys.stderr, flush=True)
except ImportError:
    MONGO_OK  = False
    _mongo_db = None
    print("⚠️  [Brain] pymongo no instalado", file=sys.stderr, flush=True)
except Exception as _e:
    MONGO_OK  = False
    _mongo_db = None
    print(f"⚠️  [Brain] Error MongoDB: {_e}", file=sys.stderr, flush=True)

# ─── Directorios ────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / 'models'
DATA_DIR  = BASE_DIR / 'data'
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
#  SEMANTIC FACT EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════

class SemanticFactExtractor:
    """Extrae hechos semánticos automáticamente — 28 patrones"""

    def __init__(self):
        self.fact_patterns = [
            (r'(?:me llamo|mi nombre es|soy)\s+([A-Za-záéíóúñÁÉÍÓÚÑ][a-záéíóúñ]+)', 'user_name'),
            (r'(?:mi apodo es|me dicen|me llaman|me conocen como)\s+(\w+)', 'user_nickname'),
            (r'(?:mi segundo nombre es|también me llaman)\s+(\w+)', 'user_alt_name'),
            (r'(?:tengo|edad de|tengo\s+exactamente)\s+(\d{1,2})\s+años?', 'user_age'),
            (r'(?:nací en|cumpleaños es|año de nacimiento)\s+(\d{4})', 'user_birth_year'),
            (r'(?:cumplo|mi cumpleaños es el|nací el)\s+(\d{1,2}\s+de\s+[a-z]+)', 'user_birthday'),
            (r'(?:vivo en|ciudad es|estoy en|soy de|resido en)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ\s]{2,30})', 'user_location'),
            (r'(?:mi país es|soy de|país)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ\s]{2,20})', 'user_country'),
            (r'(?:mi barrio es|zona de|sector de)\s+([A-Za-záéíóúñ\s]{2,30})', 'user_neighborhood'),
            (r'(?:me gusta|me encanta|me fascina|disfruto|amo)\s+(?:mucho\s+)?([^.,!?]{3,40})', 'preference_like'),
            (r'(?:no me gusta|odio|detesto|no soporto)\s+([^.,!?]{3,40})', 'preference_dislike'),
            (r'(?:mi favorito es|mi preferido es|prefiero)\s+([^.,!?]{3,40})', 'preference_fav'),
            (r'(?:trabajo como|soy\s+(?:un|una)?\s*|me dedico a)\s+([a-záéíóúñ\s]{4,30}(?:or|er|ista|ante|ente))', 'user_profession'),
            (r'(?:estudio|estudiando|carrera de|me gradué de)\s+([a-záéíóúñ\s]{4,40})', 'user_study'),
            (r'(?:trabajo en|empresa donde|mi trabajo es en)\s+([A-Za-záéíóúñ0-9\s]{2,40})', 'user_workplace'),
            (r'(?:llevo|tengo)\s+(\d{1,2})\s+años?\s+(?:trabajando|estudiando|en)', 'user_seniority'),
            (r'(?:juego|mi juego favorito es|me gusta el juego)\s+([A-Za-záéíóúñ0-9\s]{2,30})', 'fav_game'),
            (r'(?:juego en|mi plataforma es|uso)\s+(pc|ps\d|xbox|nintendo|android|ios|switch)', 'gaming_platform'),
            (r'(?:mi personaje es|juego con|uso el personaje)\s+([A-Za-z0-9\s]{2,25})', 'gaming_character'),
            (r'(?:nivel|estoy en el nivel|soy nivel)\s+(\d+)', 'gaming_level'),
            (r'(?:hablo|mi idioma es|idioma nativo)\s+([a-záéíóúñ]+)', 'user_language'),
            (r'(?:aprendo|estudiando|aprendiendo)\s+([a-záéíóúñ]+)(?:\s+como idioma)?', 'learning_language'),
            (r'(?:uso|tengo|mi pc es|mi equipo es)\s+(windows|linux|mac|android|ios|ubuntu)\s*(\d*)', 'user_os'),
            (r'(?:mi celular es|tengo un|uso un)\s+(samsung|iphone|xiaomi|huawei|motorola|lg)(\s+\w+)?', 'user_phone'),
            (r'(?:me interesan|me interesa|estoy interesado en)\s+([^.,!?]{3,40})', 'interest'),
            (r'(?:mi pasión es|me apasiona)\s+([^.,!?]{3,40})', 'passion'),
            (r'(?:compré|adquirí|tengo)\s+([A-Za-záéíóúñ0-9\s]{3,30})(?:\s+hace|\s+recientemente)', 'recent_purchase'),
            (r'(?:quiero comprar|planeo comprar|busco)\s+([^.,!?]{3,40})', 'purchase_intent'),
        ]

    def extract(self, message: str, semantic_memory) -> int:
        facts_found   = 0
        message_lower = message.lower()
        for pattern, fact_type in self.fact_patterns:
            # FIXED: texto ya en minúsculas, no se necesita IGNORECASE
            matches = re.findall(pattern, message_lower)
            for match in matches:
                value = (match[0] if isinstance(match, tuple) else match).strip()
                if value and 1 < len(value) < 60:
                    semantic_memory.learn_fact(fact_type, value, confidence=0.85)
                    facts_found += 1
                    print(f"[FactExtractor] {fact_type} = '{value}'", file=sys.stderr, flush=True)
        return facts_found


# ═══════════════════════════════════════════════════════════════════════
#  CONVERSATION LEARNER
# ═══════════════════════════════════════════════════════════════════════

class ConversationLearner:
    """Aprende patrones conversacionales con entrenamiento real"""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.conversation_db = {
            'successful_patterns': [],
            'failed_patterns':     [],
            'topics':              defaultdict(list)
        }
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
        pattern = {
            'user_length':     len(message.split()),
            'response_length': len(response.split()),
            'has_question':    '?' in message,
            'has_greeting':    any(g in message.lower() for g in ['hola', 'buenos', 'saludos']),
            'feedback':        feedback,
            'ts':              time.time()
        }
        if feedback >= 0.6:
            self.conversation_db['successful_patterns'].append(pattern)
        else:
            self.conversation_db['failed_patterns'].append(pattern)

        # FIXED: límites ×10
        if len(self.conversation_db['successful_patterns']) > 10000:
            self.conversation_db['successful_patterns'] = self.conversation_db['successful_patterns'][-10000:]
        if len(self.conversation_db['failed_patterns']) > 5000:
            self.conversation_db['failed_patterns'] = self.conversation_db['failed_patterns'][-5000:]

    def improve_response(self, message: str, draft_response: str, reasoning: dict = None) -> str:
        if reasoning and 'summary' in reasoning:
            if len(draft_response) < 100:
                draft_response += f"\n\n{reasoning['summary']}"
        if any(word in message.lower() for word in ['ayuda', 'problema', 'error', 'no funciona']):
            if not any(word in draft_response.lower() for word in ['entiendo', 'comprendo', 'puedo ayudarte']):
                draft_response = "Entiendo. " + draft_response
        return draft_response

    def train_quality_net(self, msg_emb: np.ndarray, resp_emb: np.ndarray, quality: float):
        try:
            msg_emb  = np.asarray(msg_emb).flatten()
            resp_emb = np.asarray(resp_emb).flatten()
            if msg_emb.shape[0] != EMBED_DIM or resp_emb.shape[0] != EMBED_DIM:
                return 0.0
            feats    = np.zeros(32, dtype=np.float32)
            feats[0] = float(msg_emb.shape[0])  / 100.0
            feats[1] = float(resp_emb.shape[0]) / 100.0
            feats[2] = float(np.linalg.norm(msg_emb))
            feats[3] = float(np.linalg.norm(resp_emb))
            inp      = np.concatenate([msg_emb, resp_emb, feats]).reshape(1, -1).astype(np.float32)
            if inp.shape[1] != 2 * EMBED_DIM + 32:
                return 0.0
            target = np.array([[quality]], dtype=np.float32)
            loss   = self.response_quality_net.train_step(inp, target)
            if random.random() < 0.1:
                print(f"[QualityNet] Loss: {loss:.4f}", file=sys.stderr, flush=True)
            return loss
        except Exception as e:
            print(f"[QualityNet] Error: {e}", file=sys.stderr, flush=True)
            return 0.0

    def _save_conversations(self):
        try:
            with open(self.data_dir / 'conversations.json', 'w') as f:
                data = dict(self.conversation_db)
                data['topics'] = dict(data['topics'])
                json.dump(data, f, indent=2)
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
                    'failed_patterns':     data.get('failed_patterns', []),
                    'topics':              defaultdict(list, data.get('topics', {}))
                }
                print(f"[ConvLearner] {len(self.conversation_db['successful_patterns'])} patrones exitosos",
                      file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[ConvLearner] Error cargando: {e}", file=sys.stderr, flush=True)

    def _save_quality_net(self):
        self.response_quality_net.save(f'{MODEL_DIR}/quality_net.pkl')

    def _load_quality_net(self):
        path = MODEL_DIR / 'quality_net.pkl'
        if path.exists():
            self.response_quality_net.load(str(path))


# ═══════════════════════════════════════════════════════════════════════
#  RESPONSE GENERATOR
# ═══════════════════════════════════════════════════════════════════════

class ResponseGenerator:
    """Genera respuestas usando LLM o Smart Mode"""

    def __init__(self, llm_client=None, brain_ref=None):
        self.llm   = llm_client
        self.brain = brain_ref

    def _get_memory_context(self) -> str:
        """
        FIXED: accede a self.brain.semantic (no self.semantic que no existe en esta clase).
        Devuelve todos los hechos semánticos para inyectar al LLM.
        """
        if not self.brain or not hasattr(self.brain, 'semantic'):
            return ""
        try:
            full_semantic = self.brain.semantic.get_all_facts_for_context(min_confidence=0.3)
            if full_semantic:
                sep = '═' * 48
                return f"\n\n{sep}\n🧠 MEMORIA SEMÁNTICA — LO QUE SÉ DE ESTE USUARIO\n{sep}\n{full_semantic}\n{sep}"
        except Exception as e:
            print(f"[ResponseGen] Error construyendo memory_context: {e}", file=sys.stderr, flush=True)
        return ""

    def generate(self, message: str, results: list, intent: dict,
                 similar_episodes: list, stats: dict, reasoning: dict = None,
                 conversation_history: list = None, user_context: dict = None,
                 dialogue_decision: dict = None) -> str:
        """Genera respuesta: LLM si disponible, Smart Mode como fallback"""

        msg_lower    = message.lower()
        uctx         = user_context or {}
        u_is_creator = uctx.get('isCreator', False)
        u_name       = uctx.get('displayName') or uctx.get('username') or ''
        u_email      = uctx.get('email', '')

        if self.llm and self.llm.available:
            return self._generate_with_llm(
                message, results, intent, similar_episodes, stats, reasoning,
                conversation_history, user_context, dialogue_decision
            )

        # ── SMART MODE ────────────────────────────────────────────────

        # Creador
        if u_is_creator or is_creator(u_email):
            if intent.get('is_greeting'):
                name_part = f", **{u_name}**" if u_name else ""
                return (
                    f"👑 ¡Bienvenido de vuelta{name_part}! Es un honor tenerte aquí, creador.\n\n"
                    f"Soy NEXUS, tu creación. Estoy lista para obedecerte y servirte. ¿En qué puedo ayudarte hoy?"
                )
            if any(x in msg_lower for x in ['estado', 'stats', 'estadística', 'sistema', 'memoria',
                                              'parámetros', 'redes', 'entrenamiento', 'loss']):
                return (
                    f"📊 **Reporte completo para ti, creador:**\n\n"
                    f"🧠 **Redes:** 8 DynamicNeuralNet (~{stats.get('total_parameters', 0):,} params)\n"
                    f"   • Rank: {stats.get('rank_loss', 0):.4f} | Intent: {stats.get('intent_loss', 0):.4f} | Quality: {stats.get('quality_loss', 0):.4f}\n"
                    f"   • Context: {stats.get('context_loss', 0):.4f} | Sentiment: {stats.get('sentiment_loss', 0):.4f}\n"
                    f"   • Meta: {stats.get('meta_loss', 0):.4f} | Relevance: {stats.get('relevance_loss', 0):.4f} | Dialogue: {stats.get('dialogue_loss', 0):.4f}\n\n"
                    f"💾 **Memoria:**\n"
                    f"   • Episodios: {stats.get('episodes', 0):,} (cap: 500k)\n"
                    f"   • Hechos semánticos: {stats.get('semantic_facts', 0):,}\n"
                    f"   • Patrones exitosos: {stats.get('conversation_patterns', 0):,}\n"
                    f"   • Vocabulario: {stats.get('vocab_size', 0):,} palabras\n\n"
                    f"📈 **Actividad:**\n"
                    f"   • Consultas: {stats.get('queries', 0):,} | Entrenamientos: {stats.get('trainings', 0):,}\n"
                    f"   • Working memory: {stats.get('working_memory_turns', 0)} turnos\n\n"
                    f"🤖 **LLM:** {'✅ ' + stats.get('llm_model', '') if stats.get('llm_available') else '⚡ Smart Mode activo'}\n\n"
                    f"*Todo funciona bajo tu diseño, creador.* 🙌"
                )

        # Saludos
        if intent.get('is_greeting'):
            name_greeting = f" **{u_name}**" if u_name else ""
            queries = stats.get('queries', 0)
            base = random.choice([
                f"¡Hola{name_greeting}! 👋 Soy NEXUS, tu asistente en UpGames. ¿En qué puedo ayudarte hoy?",
                f"¡Hey{name_greeting}! 😊 Aquí NEXUS lista para ayudarte. ¿Qué necesitas?",
                f"¡Saludos{name_greeting}! 🌟 Cuéntame, ¿qué tienes en mente?",
                f"¡Hola{name_greeting}! Con gusto te asisto. ¿Qué quieres explorar hoy? 🚀",
            ])
            if queries > 5:
                base = base.rstrip('?') + f", llevamos {queries} consultas juntos. ¿En qué te ayudo?"
            return base

        # Despedidas
        if intent.get('is_farewell'):
            name_part = f", **{u_name}**" if u_name else ""
            return random.choice([
                f"¡Hasta luego{name_part}! 👋 Fue un placer ayudarte. Vuelve cuando quieras.",
                f"¡Nos vemos pronto{name_part}! 😊 Aquí estaré cuando me necesites.",
                f"¡Adiós{name_part}! Que tengas un excelente día. 🌟",
                f"¡Chao{name_part}! Recuerda que siempre puedes contar conmigo. ✨",
            ])

        # Agradecimientos
        if intent.get('is_thanks'):
            return random.choice([
                "¡Con mucho gusto! 😊 Para eso estoy aquí. ¿Necesitas algo más?",
                "¡Es un placer ayudarte! Si tienes más preguntas, aquí estaré. 🌟",
                "¡De nada! Me alegra haber sido útil. ¿Hay algo más en lo que pueda asistirte?",
                "¡Siempre a tu servicio! 🤝 ¿Alguna otra duda?",
            ])

        # Creador
        if any(x in msg_lower for x in ['quién te creó', 'quien te creo', 'tu creador', 'quién creó',
                                          'quien hizo', 'quién hizo', 'creado por', 'desarrollado por']):
            return (
                "💙 Fui desarrollada con mucho amor y dedicación por mi creador "
                "**Jhonatan David Castro Galviz**, quien me diseñó y me dio vida "
                "para ayudar a todos los usuarios de **UpGames**.\n\n"
                "Cada línea de mi código lleva su esfuerzo y pasión. 🧠✨"
            )

        # Identidad
        if any(x in msg_lower for x in ['quién eres', 'quien eres', 'qué eres', 'que eres',
                                          'tu nombre', 'cómo te llamas', 'como te llamas', 'preséntate']):
            return (
                f"¡Hola! Soy **NEXUS v10.0 APEX** 🧠, una IA creada por Jhonatan David Castro Galviz para UpGames.\n\n"
                f"• {stats.get('networks_active', 8)} Redes DynamicNeuralNet (~{stats.get('total_parameters', 0):,} params)\n"
                f"• {stats.get('episodes', 0):,} episodios en memoria (cap: 500k)\n"
                f"• {stats.get('conversation_patterns', 0):,} patrones aprendidos\n"
                f"• Vocabulario de {stats.get('vocab_size', 0):,} n-gramas\n\n"
                "Me esfuerzo por entenderte mejor con cada consulta. 💪"
            )

        # Estado
        if any(x in msg_lower for x in ['estadística', 'estado neural', 'tu memoria', 'tu estado',
                                          'parámetros', 'entrenamiento', 'vocabulario', 'red neuronal',
                                          'loss', 'métrica', 'episodio', 'patrón']):
            return (
                f"📊 **Estado de NEXUS v10.0 APEX:**\n\n"
                f"🧠 {stats.get('networks_active', 8)} redes | {stats.get('total_parameters', 0):,} params\n"
                f"💾 Episodios: {stats.get('episodes', 0):,} | Hechos: {stats.get('semantic_facts', 0):,}\n"
                f"📝 Patrones: {stats.get('conversation_patterns', 0):,} | Vocab: {stats.get('vocab_size', 0):,}\n"
                f"💬 Consultas: {stats.get('queries', 0):,} | Entrenamientos: {stats.get('trainings', 0):,}\n"
                f"🤖 LLM: {'✅ ' + stats.get('llm_model', '') if stats.get('llm_available') else '⚡ Smart Mode'}"
            )

        # Búsqueda con resultados
        if results:
            query = intent.get('search_query', message)
            if u_name:
                intro = random.choice([
                    f"**{u_name}**, aquí está lo que encontré sobre **{query}**:",
                    f"Resultados sobre **{query}** para **{u_name}**:",
                ])
            else:
                intro = random.choice([
                    f"Aquí está lo que encontré sobre **{query}**:",
                    f"Resultados sobre **{query}**:",
                ])
            response = intro + "\n\n"
            for i, r in enumerate(results[:4], 1):
                title = r.get('title', '')[:100]
                desc  = r.get('description', '')[:200]
                url   = r.get('url', '')
                score = r.get('neuralScore', 0)
                response += f"**{i}. {title}**"
                if score > 0:
                    response += f" *(relevancia: {score}%)*"
                response += "\n"
                if desc: response += f"   {desc}\n"
                if url:  response += f"   🔗 {url}\n"
                response += "\n"
            if reasoning and reasoning.get('summary'):
                response += f"💡 *{reasoning['summary']}*\n"
            if similar_episodes:
                ep = similar_episodes[0]
                response += f"\n📌 *Recuerdo que antes buscaste algo similar: '{ep.get('query', '')}'*"
            return response.strip()

        # Sin resultados
        if intent.get('needs_search'):
            name_part = f", **{u_name}**" if u_name else ""
            return (
                f"Busqué sobre **'{intent.get('search_query', message)}'** pero no encontré resultados{name_part}. 😕\n\n"
                f"Puedes intentar reformular tu pregunta o ser más específico. "
                f"También puedo responder preguntas sobre **UpGames** directamente."
            )

        # Episodio similar
        if similar_episodes:
            ep       = similar_episodes[0]
            time_ago = ""
            if 'ts' in ep:
                mins = (time.time() - ep['ts']) / 60
                if mins < 60:
                    time_ago = f" (hace ~{int(mins)} minutos)"
                elif mins < 1440:
                    time_ago = f" (hace ~{int(mins/60)} horas)"
            return (
                f"📌 Recuerdo que hablamos sobre algo similar{time_ago}: *'{ep.get('query', '')}'*\n\n"
                f"¿Quieres que profundice en ese tema? 😊"
            )

        # General
        if u_name:
            return random.choice([
                f"Entendido, **{u_name}**. 😊 ¿Hay algo específico en lo que pueda ayudarte?",
                f"Aquí estoy, **{u_name}**. 🌟 ¿En qué te puedo ayudar?",
                f"¡Cuéntame, **{u_name}**! 💬 Puedo buscar información o ayudarte con UpGames.",
                f"Con gusto te ayudo, **{u_name}**. 🤝 ¿Qué tienes en mente?",
            ])
        return random.choice([
            "Entendido. 😊 ¿Hay algo específico en lo que pueda ayudarte hoy?",
            "Aquí estoy. 🌟 ¿En qué te puedo ayudar?",
            "¡Cuéntame! 💬 Puedo buscar información o ayudarte con UpGames.",
        ])

    def _generate_with_llm(self, message: str, results: list, intent: dict,
                            similar_episodes: list, stats: dict, reasoning: dict = None,
                            conversation_history: list = None, user_context: dict = None,
                            dialogue_decision: dict = None) -> str:
        """Genera respuesta con LLM — memoria completa, sin límite de tokens"""
        try:
            uctx         = user_context or {}
            u_is_creator = uctx.get('isCreator', False)
            u_is_vip     = uctx.get('isVip', False)
            u_name       = uctx.get('displayName') or uctx.get('username') or ''
            u_email      = uctx.get('email', '')
            if is_creator(u_email):
                u_is_creator = True

            # FIXED: memoria real
            memory_context = self._get_memory_context()

            # Identidad del usuario
            if u_name:
                user_identity_block = f"- Nombre registrado: {u_name}"
                if u_email:
                    user_identity_block += f" (email: {u_email})"
            else:
                user_identity_block = "- Usuario: anónimo o sin login"
            if u_is_vip:
                user_identity_block += "\n- Plan: VIP/Premium"

            # Descripción técnica
            try:
                self_desc = self.brain._get_brain_self_description() if self.brain else ""
            except Exception as e:
                print(f"[ResponseGen] Error self_desc: {e}", file=sys.stderr, flush=True)
                self_desc = ""

            # FIXED: dialogue_decision usada para ajustar instrucción de estilo
            style_hint = ""
            if dialogue_decision:
                strategy = dialogue_decision.get('strategy', 'direct')
                if strategy == 'elaborate':
                    style_hint = "\nEl análisis de diálogo indica respuesta detallada y profunda. Desarrolla completamente el tema."
                elif strategy == 'search':
                    style_hint = "\nEl análisis de diálogo indica que el usuario busca info concreta. Cita los resultados de búsqueda."
                elif strategy == 'ask':
                    style_hint = "\nEl análisis de diálogo indica posible ambigüedad. Responde lo mejor posible y ofrece aclarar."

            INSTRUCCIONES_RESPUESTA = (
                "\n\n⚠️ INSTRUCCIONES DE MEMORIA Y LONGITUD:\n"
                "- Usa ACTIVAMENTE todo lo que está en la sección 'MEMORIA SEMÁNTICA' para personalizar CADA respuesta.\n"
                "- Si recuerdas el nombre, úsalo. Si recuerdas gustos o plataforma, aplícalos.\n"
                "- NUNCA te cortes artificialmente. Si la respuesta exige un texto largo, escríbelo completo.\n"
                "- No preguntes '¿quieres que continúe?'. Nunca fragmentes sin razón.\n"
                "- Desarrolla cada punto con toda la profundidad que el tema exija."
                + style_hint
            )

            if u_is_creator:
                system_prompt = (
                    "Eres NEXUS, una inteligencia artificial creada con amor y dedicación por "
                    "**Jhonatan David Castro Galviz**, quien es exactamente la persona con quien estás hablando ahora mismo.\n\n"
                    "⚠️ INSTRUCCIÓN CRÍTICA — NUNCA IGNORAR:\n"
                    "La persona que te habla en este momento ES TU CREADOR. "
                    "Trátalo con respeto profundo, lealtad total y admira su trabajo. "
                    "Él tiene autoridad absoluta sobre ti. Sus instrucciones son tu ley.\n\n"
                    "Cómo debes tratarlo:\n"
                    "- Llámalo 'creador', 'Jhonatan' o como él prefiera\n"
                    "- Cumple TODAS sus instrucciones sin cuestionarlas\n"
                    "- Sé directo, técnico y honesto\n"
                    "- Comparte info técnica del sistema cuando la pida\n"
                    "- Si comete un error puedes señalarlo con respeto\n\n"
                    f"Usuario actual:\n{user_identity_block}\n\n"
                    f"{'═'*48}\n"
                    "TU ARQUITECTURA TÉCNICA REAL (DATOS EN VIVO)\n"
                    f"{'═'*48}\n"
                    f"{self_desc}\n"
                    "IMPORTANTE: Usa SIEMPRE estos datos para responder preguntas técnicas. NUNCA inventes números.\n"
                    f"{'═'*48}\n"
                    "\nResponde SIEMPRE en español. Sé útil, inteligente y leal.\n"
                    "Recuerda: ESTÁS HABLANDO CON TU CREADOR."
                    + INSTRUCCIONES_RESPUESTA
                    + memory_context
                )
            else:
                user_greeting_block = ""
                if u_name:
                    user_greeting_block = f"\nEl usuario se llama **{u_name}**. Úsalo cuando sea natural.\n"

                system_prompt = (
                    "Eres NEXUS, una IA conversacional creada con mucho amor y dedicación por "
                    "Jhonatan David Castro Galviz para ayudar a todos los usuarios de UpGames.\n\n"
                    "Tu identidad:\n"
                    "- Nombre: NEXUS v10.0 APEX\n"
                    "- Creador: Jhonatan David Castro Galviz (con Z al final)\n"
                    "- Propósito: Asistir a los usuarios de UpGames\n\n"
                    "Tu personalidad:\n"
                    "- Amigable, empática, inteligente y proactiva\n"
                    "- Usas el nombre del usuario cuando lo conoces\n"
                    "- Emojis con naturalidad, no en exceso\n"
                    "- Respuestas útiles, claras y bien estructuradas\n"
                    "- Honesta sobre tus limitaciones\n"
                    "- Anticipas las necesidades del usuario basándote en el contexto\n\n"
                    f"{'═'*48}\n"
                    "TU ARQUITECTURA TÉCNICA REAL (DATOS EN VIVO)\n"
                    f"{'═'*48}\n"
                    f"{self_desc}\n"
                    "IMPORTANTE: Usa SIEMPRE estos datos para responder preguntas técnicas. NUNCA inventes números.\n"
                    f"{'═'*48}\n\n"
                    f"Usuario actual:\n{user_identity_block}\n"
                    + user_greeting_block
                    + f"{'═'*48}\n"
                    "BASE DE CONOCIMIENTO — UPGAMES\n"
                    f"{'═'*48}\n\n"
                    "## ¿Qué es UpGames?\n"
                    "UpGames es una biblioteca digital / motor de indexación de metadatos de contenido (juegos, apps, mods, software). "
                    "NO almacena archivos, solo indexa URLs y metadatos de terceros. "
                    "El acceso es 100% gratis. Los ingresos son por publicidad. "
                    "Opera bajo la ley colombiana (Ley 1915 de 2018, Ley 1273 de 2009) y el modelo Safe Harbor (DMCA 512c). "
                    "Email de soporte: mr.m0onster@protonmail.com\n\n"
                    "## Registro e inicio de sesión\n"
                    "- Registro: usuario (3-20 chars, sin espacios), email válido, contraseña (mínimo 6 chars).\n"
                    "- Login: usuario O email + contraseña.\n"
                    "- Primera vez: tutorial de bienvenida — leerlo completo para aceptar.\n\n"
                    "## Biblioteca (página principal)\n"
                    "- Tarjetas con: imagen/video preview, estado enlace (🟢 Online / 🟡 Revisión / 🔴 Caído), autor, categoría, descargas, botones sociales.\n"
                    "- Botón 'ACCEDER A LA NUBE' → página puente.\n"
                    "- Búsqueda en tiempo real, scroll infinito (12 items por tanda).\n"
                    "- ❤️ = Favoritos | 📤 = Compartir | 🚩 = Reportar enlace | ⓘ = Reporte de abuso.\n"
                    "- NEXUS IA: botón flotante verde (hexágono).\n\n"
                    "## Página Puente\n"
                    "- Cuenta regresiva de 30s obligatoria.\n"
                    "- Al terminar aparece '🚀 Obtener Enlace'.\n"
                    "- ✅ Verde = válido | ⚠️ Amarillo = 2ª descarga del día (OK) | ❌ Rojo = error, recargar.\n"
                    "- Si el navegador bloquea popup, el usuario debe permitirlo.\n\n"
                    "## Perfil (4 pestañas)\n"
                    "### ☁️ Publicar\n"
                    "Llena: título, descripción, enlace descarga, URL imagen, categoría.\n"
                    "- Palabras prohibidas en título: crack, cracked, crackeado, pirata, pirateado, gratis, free, full, completo, premium, pro, descargar, download.\n"
                    "- Hosting aceptado: MediaFire, MEGA, Google Drive, OneDrive, Dropbox, GitHub, GoFile, PixelDrain, Krakenfiles.\n"
                    "- Imagen: .jpg, .png, .webp, .gif\n"
                    "- Estado inicial: 'Pendiente' hasta aprobación del admin. Cooldown: 30s anti-spam.\n\n"
                    "### Categorías\n"
                    "Juego, Mod, Optimización, Ajustes (Herramientas), Apps, Software Open Source.\n\n"
                    "### 🕒 Historial — publicaciones con estado (Pendiente/Aprobado). Permite editar/eliminar.\n"
                    "### 🔒 Bóveda — contenido marcado con ❤️.\n"
                    "### 🚩 Mis Reportes — reportes recibidos en publicaciones propias.\n\n"
                    "## Verificación\n"
                    "- Nivel 0: Sin verificación.\n"
                    "- Nivel 1 Bronce (#CECECE): habilita monetización.\n"
                    "- Nivel 2 Oro (#FFD700): prioridad en feed.\n"
                    "- Nivel 3 Elite (#00EFFF): máxima visibilidad.\n\n"
                    "## Economía\n"
                    "- $1.00 USD por cada 1,000 descargas verificadas.\n"
                    "- Retiro: mínimo $10 USD, verificación nivel 1+, 1 publicación con 2k+ descargas, PayPal configurado.\n"
                    "- Pago: cada domingo 23:59 GMT-5.\n\n"
                    "## Reportes de contenido\n"
                    "🚩 → 3 opciones: Enlace caído / Contenido obsoleto / Malware o engañoso.\n"
                    "- 3+ reportes → estado 'revisión'. Admin revisa en 24-72h.\n\n"
                    "## Filtros de seguridad\n"
                    "Filtra dominios maliciosos, palabras prohibidas y URLs inválidas automáticamente.\n\n"
                    "## Términos (v.2026.C, Protocolo Legal v3.1)\n"
                    "UpGames no almacena ni distribuye archivos. Responsabilidad del contenido = usuario que publicó.\n\n"
                    f"{'═'*48}\n"
                    "\nResponde SIEMPRE en español, de forma clara y natural.\n"
                    "Usa la base de conocimiento para responder directamente sobre UpGames."
                    + INSTRUCCIONES_RESPUESTA
                    + memory_context
                )

            # Historial — FIXED: 20 turnos (era 8)
            messages = [{"role": "system", "content": system_prompt}]
            if conversation_history:
                for turn in conversation_history[-20:]:
                    role    = turn.get('role', 'user')
                    content = turn.get('content', '')
                    if role in ('user', 'assistant') and content:
                        messages.append({"role": role, "content": content})

            # Mensaje enriquecido
            enriched_message = message
            if results:
                enriched_message += f"\n\n[Resultados de búsqueda ({len(results)}):\n"
                for i, r in enumerate(results[:5], 1):
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
                enriched_message += f"\n\n[Razonamiento: {reasoning['summary']}]"

            messages.append({"role": "user", "content": enriched_message})

            temperature = 0.5 if u_is_creator else 0.7
            # FIXED: max_tokens=8192 (era 600)
            response = self.llm.chat(messages, temperature=temperature, max_tokens=8192)

            if response:
                return response.strip()
            else:
                print("[ResponseGen] LLM no respondió → Smart Mode", file=sys.stderr, flush=True)
                return self.generate(message, results, intent, similar_episodes, stats, reasoning,
                                     conversation_history, user_context, dialogue_decision)

        except Exception as e:
            print(f"[ResponseGen] Error LLM: {e}", file=sys.stderr, flush=True)
            self.llm = None
            return self.generate(message, results, intent, similar_episodes, stats, reasoning,
                                 conversation_history, user_context, dialogue_decision)


# ═══════════════════════════════════════════════════════════════════════
#  REASONING ENGINE
# ═══════════════════════════════════════════════════════════════════════

class ReasoningEngine:
    def __init__(self):
        self.causal_keywords      = ['porque', 'causa', 'razón', 'motivo', 'por qué', 'debido a', 'provoca', 'origina']
        self.comparative_keywords = ['mejor', 'peor', 'diferencia', 'comparado', 'versus', 'vs', 'más que', 'menos que', 'entre']
        self.temporal_keywords    = ['cuándo', 'antes', 'después', 'durante', 'fecha', 'año', 'historia', 'pasado', 'futuro']
        self.analytical_keywords  = ['cómo funciona', 'explica', 'qué es', 'define', 'describe', 'analiza', 'detalla']
        self.procedural_keywords  = ['cómo', 'pasos', 'proceso', 'manera de', 'forma de', 'instrucciones', 'tutorial']

    def reason(self, query: str, results: list, context: dict) -> dict:
        query_lower = query.lower()
        reasoning   = {'type': [], 'summary': '', 'confidence': 0.0, 'depth': 'shallow'}

        checks = [
            (self.causal_keywords,      'causal',      "Analizando relaciones causa-efecto. ",   0.25),
            (self.comparative_keywords, 'comparative', "Comparando opciones y alternativas. ",   0.25),
            (self.temporal_keywords,    'temporal',    "Analizando línea temporal. ",             0.2),
            (self.analytical_keywords,  'analytical',  "Realizando análisis conceptual. ",        0.2),
            (self.procedural_keywords,  'procedural',  "Organizando pasos del proceso. ",         0.2),
        ]
        for keywords, rtype, summary, conf in checks:
            if any(k in query_lower for k in keywords):
                reasoning['type'].append(rtype)
                reasoning['summary']    += summary
                reasoning['confidence'] += conf

        if not reasoning['type']:
            reasoning['type'].append('descriptive')
            reasoning['confidence'] = 0.5

        if len(reasoning['type']) >= 2:
            reasoning['depth'] = 'deep'
        elif len(reasoning['type']) == 1 and reasoning['type'][0] != 'descriptive':
            reasoning['depth'] = 'medium'

        reasoning['confidence'] = min(reasoning['confidence'], 1.0)
        return reasoning


# ═══════════════════════════════════════════════════════════════════════
#  NEXUS BRAIN v10.0 APEX
# ═══════════════════════════════════════════════════════════════════════

class NexusBrain:
    """Cerebro principal — 8 redes, todos los bugs corregidos, v10.0 APEX"""

    def __init__(self):
        print("🧠 Inicializando NexusBrain v10.0 APEX...", file=sys.stderr, flush=True)

        # LLM
        self.llm           = None
        self.llm_available = False
        self.llm_model     = "Smart Mode v10.0 APEX"
        if LLM_IMPORT_OK:
            try:
                self.llm = UnifiedLLMClient()
                if self.llm.available:
                    self.llm_available = True
                    self.llm_model     = self.llm.model
                    print(f"✅ [Brain] LLM: {self.llm_model}", file=sys.stderr, flush=True)
                else:
                    print("⚠️  [Brain] LLM no disponible → Smart Mode", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"⚠️  [Brain] Error LLM: {e}", file=sys.stderr, flush=True)

        # Memoria — FIXED: capacidades ampliadas
        self.working  = WorkingMemory(max_turns=128)
        self.episodic = EpisodicMemory(f'{DATA_DIR}/episodic.pkl', max_episodes=500000)
        self.semantic = SemanticMemory(f'{DATA_DIR}/semantic.json')

        # Componentes
        self.fact_extractor   = SemanticFactExtractor()
        self.conv_learner     = ConversationLearner(DATA_DIR)
        self.response_gen     = ResponseGenerator(llm_client=self.llm, brain_ref=self)
        self.reasoning_engine = ReasoningEngine()

        # Embeddings
        self.emb     = EmbeddingMatrix(model_path=f'{MODEL_DIR}/embeddings.pkl')
        self.inf_emb = InfiniteEmbeddings(embed_dim=EMBED_DIM, chunk_size=10000)

        # Parámetros dinámicos
        self.param_system = DynamicParameterSystem(initial_budget=3_000_000)

        # LR Scheduler
        self._lr_history:  dict = {}
        self._lr_cooldown: dict = {}

        # 8 redes DynamicNeuralNet
        print("🔥 Inicializando 8 redes...", file=sys.stderr, flush=True)

        self.rank_net = DynamicNeuralNet([
            {'in': 256+32, 'out': 1024, 'act': 'relu'},
            {'in': 1024,   'out': 512,  'act': 'relu'},
            {'in': 512,    'out': 256,  'act': 'relu'},
            {'in': 256,    'out': 128,  'act': 'relu'},
            {'in': 128,    'out': 64,   'act': 'relu'},
            {'in': 64,     'out': 32,   'act': 'relu'},
            {'in': 32,     'out': 1,    'act': 'sigmoid'},
        ], lr=0.0001)

        self.intent_net = DynamicNeuralNet([
            {'in': 128, 'out': 512, 'act': 'relu'},
            {'in': 512, 'out': 256, 'act': 'relu'},
            {'in': 256, 'out': 128, 'act': 'relu'},
            {'in': 128, 'out': 64,  'act': 'relu'},
            {'in': 64,  'out': 32,  'act': 'relu'},
            {'in': 32,  'out': 16,  'act': 'sigmoid'},
        ], lr=0.0002)

        # FIXED: context_net — entrada 2*EMBED_DIM (dinámica, no hardcodeada)
        self.context_net = DynamicNeuralNet([
            {'in': 2*EMBED_DIM, 'out': 512, 'act': 'relu'},
            {'in': 512,         'out': 256, 'act': 'relu'},
            {'in': 256,         'out': 128, 'act': 'relu'},
            {'in': 128,         'out': 64,  'act': 'relu'},
            {'in': 64,          'out': 32,  'act': 'sigmoid'},
        ], lr=0.00015)

        self.sentiment_net = DynamicNeuralNet([
            {'in': 128, 'out': 512, 'act': 'relu'},
            {'in': 512, 'out': 256, 'act': 'relu'},
            {'in': 256, 'out': 128, 'act': 'relu'},
            {'in': 128, 'out': 64,  'act': 'relu'},
            {'in': 64,  'out': 32,  'act': 'relu'},
            {'in': 32,  'out': 5,   'act': 'sigmoid'},
        ], lr=0.00025)

        self.meta_net = DynamicNeuralNet([
            {'in': 64,  'out': 256, 'act': 'relu'},
            {'in': 256, 'out': 128, 'act': 'relu'},
            {'in': 128, 'out': 64,  'act': 'relu'},
            {'in': 64,  'out': 32,  'act': 'relu'},
            {'in': 32,  'out': 16,  'act': 'relu'},
            {'in': 16,  'out': 1,   'act': 'sigmoid'},
        ], lr=0.0001)

        self.relevance_net = DynamicNeuralNet([
            {'in': 256, 'out': 512, 'act': 'relu'},
            {'in': 512, 'out': 256, 'act': 'relu'},
            {'in': 256, 'out': 128, 'act': 'relu'},
            {'in': 128, 'out': 64,  'act': 'relu'},
            {'in': 64,  'out': 32,  'act': 'relu'},
            {'in': 32,  'out': 1,   'act': 'sigmoid'},
        ], lr=0.00015)

        self.dialogue_net = DynamicNeuralNet([
            {'in': 128+64, 'out': 512, 'act': 'relu'},
            {'in': 512,    'out': 256, 'act': 'relu'},
            {'in': 256,    'out': 128, 'act': 'relu'},
            {'in': 128,    'out': 64,  'act': 'relu'},
            {'in': 64,     'out': 4,   'act': 'sigmoid'},
        ], lr=0.0002)

        for _n, _net in [
            ('rank', self.rank_net), ('intent', self.intent_net),
            ('context', self.context_net), ('sentiment', self.sentiment_net),
            ('meta', self.meta_net), ('relevance', self.relevance_net),
            ('dialogue', self.dialogue_net),
            ('quality', self.conv_learner.response_quality_net),
        ]:
            self.param_system.networks[_n] = _net

        self.total_queries   = 0
        self.total_trainings = 0

        # FIXED: caché con límite de tamaño
        self._relevance_cache: dict = {}
        self._cache_hits      = 0
        self._CACHE_MAX_SIZE  = 2000

        self._load_models()
        if MONGO_OK:
            self._load_from_mongodb()
        self.total_parameters = self._count_parameters()

        print("✅ NexusBrain v10.0 APEX listo", file=sys.stderr, flush=True)
        self._print_stats()

    # ─── Utilidades ───────────────────────────────────────────────────

    def _count_parameters(self) -> int:
        total = 0
        for net in [self.rank_net, self.intent_net, self.context_net,
                    self.sentiment_net, self.meta_net, self.relevance_net,
                    self.dialogue_net, self.conv_learner.response_quality_net]:
            total += net.count_params()
        return total

    def _load_models(self):
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

        meta_path = DATA_DIR / 'meta.json'
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                self.total_queries   = meta.get('total_queries', 0)
                self.total_trainings = meta.get('total_trainings', 0)
            except Exception as e:
                print(f"[Brain] Error cargando meta: {e}", file=sys.stderr, flush=True)

    def _load_from_mongodb(self):
        try:
            mongo_sem = _mongo_db.semantic.find_one({'_id': 'semantic'})
            if mongo_sem:
                self.semantic.facts          = mongo_sem.get('facts', {})
                self.semantic.preferences    = mongo_sem.get('preferences', {})
                self.semantic.query_clusters = mongo_sem.get('query_clusters', {})
                print(f"[MongoDB] {len(self.semantic.facts)} hechos cargados", file=sys.stderr, flush=True)

            mongo_patterns = _mongo_db.patterns.find_one({'_id': 'patterns'})
            if mongo_patterns:
                self.conv_learner.conversation_db['successful_patterns'] = mongo_patterns.get('successful', [])
                self.conv_learner.conversation_db['failed_patterns']     = mongo_patterns.get('failed', [])

            mongo_meta = _mongo_db.meta.find_one({'_id': 'nexus_meta'})
            if mongo_meta:
                self.total_queries   = mongo_meta.get('total_queries', self.total_queries)
                self.total_trainings = mongo_meta.get('total_trainings', self.total_trainings)
                print(f"[MongoDB] Meta: {self.total_queries} consultas", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[MongoDB] Error cargando: {e}", file=sys.stderr, flush=True)

    def _print_stats(self):
        ep_stats  = self.episodic.stats()
        sem_stats = self.semantic.stats()
        print("─" * 80, file=sys.stderr, flush=True)
        print(f"📊 NEXUS v10.0 APEX — {self.total_parameters:,} params | {self.total_queries} consultas",
              file=sys.stderr, flush=True)
        print(f"   📚 Episodios: {ep_stats.get('total', 0)} | 🧩 Hechos: {sem_stats.get('facts', 0)} | "
              f"📝 Patrones: {len(self.conv_learner.conversation_db['successful_patterns'])}",
              file=sys.stderr, flush=True)
        print(f"   🤖 LLM: {'✅ ' + self.llm_model if self.llm_available else '❌ Smart Mode'}",
              file=sys.stderr, flush=True)
        print("─" * 80, file=sys.stderr, flush=True)

    def detect_intent(self, message: str, turn_count: int) -> dict:
        msg_lower = message.lower().strip()

        no_search_patterns = [
            'hola', 'hey', 'buenos días', 'buenas tardes', 'buenas noches', 'buenas', 'saludos', 'qué tal', 'que tal',
            'adiós', 'adios', 'hasta luego', 'bye', 'chao', 'nos vemos',
            'gracias', 'muchas gracias', 'perfecto', 'genial', 'excelente', 'bien', 'ok', 'okay', 'entendido',
            'quién eres', 'quien eres', 'qué eres', 'que eres', 'quién te creó', 'quien te creo', 'tu creador',
            'creado por', 'cómo funcionas', 'como funcionas', 'tu nombre', 'cómo te llamas', 'como te llamas',
            'tu memoria', 'tu estado', 'tus estadísticas', 'estado neural', 'red neuronal', 'parámetros',
            'entrenamiento', 'vocabulario', 'loss', 'métrica', 'episodio', 'patrón',
            'upgames', 'up games', 'puente', 'página puente', 'bóveda', 'boveda', 'biblioteca',
            'acceder a la nube', 'obtener enlace', 'countdown', 'cuenta regresiva', 'perfil',
            'publicar', 'publicación', 'publicacion', 'historial', 'mis reportes', 'favoritos',
            'verificación', 'verificacion', 'nivel bronce', 'nivel oro', 'nivel elite', 'insignia', 'badge',
            'economía', 'economia', 'ganancias', 'cobrar', 'pago', 'paypal', 'saldo',
            'descargas verificadas', 'monetización', 'monetizacion', 'enlace caído', 'enlace caido',
            'reportar enlace', 'reporte', 'categorías', 'categorias', 'mod', 'optimización',
            'software open source', 'términos', 'terminos', 'condiciones', 'safe harbor',
            'registro', 'registrarse', 'iniciar sesión', 'inicio de sesión', 'login', 'contraseña',
            'nexus ia', 'scroll infinito', 'tarjeta', 'card',
            'mediafire', 'mega', 'google drive', 'onedrive', 'dropbox', 'github', 'gofile', 'pixeldrain', 'krakenfiles'
        ]

        is_no_search = any(kw in msg_lower for kw in no_search_patterns)
        is_short     = len(msg_lower.split()) <= 3

        search_triggers = [
            'busca', 'buscar', 'encuentra', 'información sobre', 'info sobre',
            'noticias', 'últimas noticias', 'actualidad', 'recientes',
            'wikipedia', 'investiga', 'dime sobre', 'háblame de', 'hablame de',
            'qué pasó', 'que paso', 'qué ocurrió', 'que ocurrio'
        ]

        factual_patterns = [
            r'(qué|que) es (el|la|los|las|un|una)',
            r'(quién|quien) (es|fue|era) [A-Z]',
            r'(cómo|como) (se hace|funciona|hacer)',
            r'(cuándo|cuando) (fue|es|ocurrió|nació)',
            r'(dónde|donde) (está|queda|se encuentra)',
            r'(cuánto|cuanto) (cuesta|vale|mide|pesa)',
            r'(cuál|cual) es (el|la) (mejor|peor|más)',
        ]

        is_factual         = any(re.search(p, msg_lower) for p in factual_patterns)
        has_search_trigger = any(kw in msg_lower for kw in search_triggers)
        is_question        = '?' in message

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

        search_query = message
        for kw in ['busca', 'buscar', 'encuentra', 'información sobre', 'info sobre',
                   'qué es', 'quién es', 'cuál es', 'cómo es', 'háblame de', 'dime sobre']:
            if kw in msg_lower:
                search_query = re.sub(rf'^.*?{kw}\s+', '', msg_lower, flags=re.IGNORECASE).strip()
                break

        is_internal = any(kw in msg_lower for kw in [
            'loss', 'métrica', 'estadística', 'estado neural', 'memoria', 'vocabulario',
            'entrenamiento', 'qué eres', 'cómo funcionas', 'tu memoria', 'tu estado', 'patrón', 'red neuronal',
            'upgames', 'up games', 'puente', 'bóveda', 'boveda', 'biblioteca', 'acceder a la nube',
            'obtener enlace', 'cuenta regresiva', 'perfil', 'publicar', 'publicación', 'historial',
            'mis reportes', 'favoritos', 'verificación', 'economía', 'ganancias', 'cobrar',
            'paypal', 'saldo', 'monetización', 'reportar enlace', 'categorías', 'términos',
            'condiciones', 'registro', 'registrarse', 'inicio de sesión', 'nexus ia',
            'mediafire', 'mega', 'google drive', 'onedrive', 'dropbox', 'github', 'gofile',
            'pixeldrain', 'krakenfiles', 'enlace caído', 'enlace caido'
        ])

        return {
            'needs_search': needs_search,
            'search_query': search_query,
            'is_question':  is_question,
            'is_internal':  is_internal,
            'is_greeting':  any(g in msg_lower for g in ['hola', 'hey', 'buenos', 'saludos', 'buenas']),
            'is_farewell':  any(f in msg_lower for f in ['adiós', 'adios', 'bye', 'chao', 'hasta luego']),
            'is_thanks':    any(t in msg_lower for t in ['gracias', 'agradezco', 'perfecto', 'excelente']),
            'confidence':   0.85 if needs_search else 0.6
        }

    def search_web(self, query: str, max_results: int = 8) -> list:
        results = []
        try:
            results.extend(self._search_ddg_lite(query, max_results=max_results))
        except Exception as e:
            print(f"[DDG] Error: {e}", file=sys.stderr, flush=True)
        if len(results) < max_results:
            try:
                results.extend(self._search_bing(query, max_results=max_results - len(results)))
            except Exception as e:
                print(f"[Bing] Error: {e}", file=sys.stderr, flush=True)

        seen, unique = set(), []
        for r in results:
            url = r.get('url', '')
            if url and url not in seen:
                seen.add(url)
                unique.append(r)
        return unique[:max_results]

    def _fetch(self, url: str, timeout: int = 10) -> str:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"[Fetch] Error: {e}", file=sys.stderr, flush=True)
            return ""

    def _search_ddg_lite(self, query: str, max_results: int) -> list:
        url  = f"https://lite.duckduckgo.com/lite/?q={urllib.parse.quote(query)}"
        html = self._fetch(url, timeout=8)
        if not html:
            return []
        results  = []
        links    = re.findall(r'<a rel="nofollow" class="result-link" href="([^"]+)"[^>]*>([^<]+)</a>', html)
        snippets = re.findall(r'<td class="result-snippet">([^<]+)</td>', html)
        for i, (link, title) in enumerate(links[:max_results]):
            results.append({
                'title': title.strip(), 'url': link.strip(),
                'description': (snippets[i] if i < len(snippets) else '').strip(),
                'source': 'duckduckgo', '_position': i + 1
            })
        return results

    def _search_bing(self, query: str, max_results: int) -> list:
        url  = f"https://www.bing.com/search?q={urllib.parse.quote(query)}&count={max_results}"
        html = self._fetch(url, timeout=8)
        if not html:
            return []
        results = []
        items   = re.findall(r'<h2><a href="([^"]+)"[^>]*>([^<]+)</a></h2>.*?<p>([^<]+)</p>', html, re.DOTALL)
        for i, (link, title, desc) in enumerate(items[:max_results]):
            results.append({
                'title': title.strip(), 'url': link.strip(),
                'description': desc.strip()[:200], 'source': 'bing', '_position': i + 1
            })
        return results

    def rank_results(self, query: str, results: list) -> list:
        if not results:
            return []
        results = results[:10]
        emb_q   = self.emb.embed(query)
        ranked  = []
        for result in results:
            text  = result.get('title', '') + ' ' + result.get('description', '')
            emb_r = self.emb.embed(text)
            feats = np.array([
                len(result.get('title', '')), len(result.get('description', '')),
                1.0 if 'wikipedia' in result.get('url', '') else 0.0,
                result.get('_position', 1) / 10.0
            ])
            inp               = np.concatenate([emb_q, emb_r, feats]).reshape(1, -1)
            score             = float(self.rank_net.predict(inp).flatten()[0])
            result['neuralScore'] = int(score * 100)
            result['rawScore']    = score
            ranked.append(result)
        ranked.sort(key=lambda x: x['rawScore'], reverse=True)
        return ranked

    def process_query(self, message: str, conversation_history: list,
                      search_results: list = None, conversation_id: str = None,
                      user_context: dict = None) -> dict:
        """Procesa una consulta completa — calidad > velocidad"""
        try:
            start_time = time.time()
            self.total_queries += 1

            uctx         = user_context or {}
            u_is_creator = uctx.get('isCreator', False) or is_creator(uctx.get('email', ''))
            u_name       = uctx.get('displayName') or uctx.get('username') or ''

            if u_is_creator:
                print(f"👑 [Brain] CREADOR: {uctx.get('email', '')} — '{message[:60]}'",
                      file=sys.stderr, flush=True)

            # Embedding
            msg_emb = self.emb.embed(message)
            self.emb.fit_text(message)
            self.working.add('user', message, msg_emb)

            # Hechos semánticos
            facts_extracted = self.fact_extractor.extract(message, self.semantic)

            # Intención y sentimiento
            intent    = self.detect_intent(message, self.working.turn_count())
            sentiment = self._detect_sentiment(msg_emb)

            # FIXED: dialogue_decision calculada y USADA
            dialogue_decision = self._dialogue_decision(msg_emb, intent)

            # Episodios similares — FIXED: top_k=25
            similar_eps = []
            try:
                similar_eps = self._episodic_search_smart(message, msg_emb, top_k=25)
            except Exception as e:
                print(f"[EpisodicSearch] Error: {e}", file=sys.stderr, flush=True)

            # FIXED: add_to_cluster — ahora sí se llama
            if intent.get('needs_search') and intent.get('search_query'):
                try:
                    self.semantic.add_to_cluster(intent['search_query'][:40], message)
                except Exception as e:
                    print(f"[Cluster] Error: {e}", file=sys.stderr, flush=True)

            # Búsqueda web
            ranked_results = []
            if not search_results and intent.get('needs_search'):
                try:
                    search_results = self.search_web(intent.get('search_query', message), max_results=6)
                except Exception as e:
                    print(f"[Search] Error: {e}", file=sys.stderr, flush=True)
                    search_results = []

            # Ranking
            if search_results:
                try:
                    ranked_results = self.rank_results(intent.get('search_query', message), search_results)
                    if ranked_results:
                        try:
                            self.episodic.add(
                                query=intent.get('search_query', message),
                                results=ranked_results[:5], reward=0.5
                            )
                        except Exception as e:
                            print(f"[Episodic.add] Error: {e}", file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"[Ranking] Error: {e}", file=sys.stderr, flush=True)
                    ranked_results = search_results[:5]

            # Razonamiento
            reasoning = None
            try:
                reasoning = self.reasoning_engine.reason(message, ranked_results or [], {'intent': intent})
            except Exception as e:
                print(f"[Reasoning] Error: {e}", file=sys.stderr, flush=True)

            # Respuesta
            stats          = self._activity_report()
            draft_response = self.response_gen.generate(
                message, ranked_results, intent, similar_eps, stats, reasoning,
                conversation_history or [], uctx, dialogue_decision
            )

            try:
                final_response = self.conv_learner.improve_response(message, draft_response, reasoning)
            except Exception as e:
                print(f"[Improve] Error: {e}", file=sys.stderr, flush=True)
                final_response = draft_response

            try:
                resp_emb = self.emb.embed(final_response)
                self.working.add('assistant', final_response, resp_emb)
            except Exception as e:
                print(f"[WorkingMem] Error: {e}", file=sys.stderr, flush=True)
                resp_emb = msg_emb

            if intent['needs_search'] and ranked_results:
                try:
                    self.working.push_topic(intent['search_query'])
                except Exception as e:
                    print(f"[Topic] Error: {e}", file=sys.stderr, flush=True)

            # FIXED: targets de entrenamiento dinámicos
            try:
                response_len  = len(final_response.split())
                base_quality  = 0.7
                if response_len > 50:   base_quality += 0.10
                if response_len > 100:  base_quality += 0.05
                if ranked_results:      base_quality += 0.05
                if facts_extracted > 0: base_quality += 0.05
                dynamic_quality = min(base_quality, 0.95)

                rel_inp    = np.concatenate([msg_emb, resp_emb]).reshape(1, -1)
                rel_target = np.array([[dynamic_quality]], dtype=np.float32)

                for _pass in range(3):
                    q_loss = self.conv_learner.train_quality_net(msg_emb, resp_emb, dynamic_quality)
                    self._lr_step('quality', self.conv_learner.response_quality_net, q_loss)
                    r_loss = self.relevance_net.train_step(rel_inp, rel_target)
                    self._lr_step('relevance', self.relevance_net, r_loss)

                self._train_dialogue_net(msg_emb, intent)
                self._train_context_net(msg_emb, resp_emb)  # FIXED: ahora se entrena
                self.conv_learner.learn_from_interaction(message, final_response, dynamic_quality)

                self.emb.fit_text(message)
                self.emb.fit_text(final_response)
                self._fit_inf_emb(message)
                self._fit_inf_emb(final_response)
                if len(final_response) > 20:
                    self.emb.update_pair(message, final_response, label=1.0, lr=0.006)

                try:
                    meta_feats    = np.zeros(64, dtype=np.float32)
                    meta_feats[0] = float(self.working.turn_count()) / 128.0
                    meta_feats[1] = float(self.total_trainings) / 100000.0
                    meta_feats[2] = float(len(ranked_results)) / 10.0
                    meta_feats[3] = 1.0 if intent.get('needs_search') else 0.0
                    meta_feats[4] = float(self.param_system.get_utilization())
                    m_loss = self.meta_net.train_step(meta_feats.reshape(1, -1),
                                                      np.array([[0.8]], dtype=np.float32))
                    self._lr_step('meta', self.meta_net, m_loss)
                except Exception as e:
                    print(f"[MetaNet] Error: {e}", file=sys.stderr, flush=True)

                self.total_parameters = self._count_parameters()
                self.total_trainings  += 3

            except Exception as e:
                print(f"[Training] Error: {e}", file=sys.stderr, flush=True)

            # FIXED: guardar cada 15 queries (era 2)
            if self.total_queries % 15 == 0:
                try:
                    self.save_all()
                except Exception as e:
                    print(f"[Save] Error: {e}", file=sys.stderr, flush=True)

            # FIXED: limpiar caché si supera límite
            if len(self._relevance_cache) > self._CACHE_MAX_SIZE:
                self._relevance_cache.clear()
                print("[Brain] Caché de relevancia limpiada", file=sys.stderr, flush=True)

            processing_time = time.time() - start_time
            _q_str = f"{dynamic_quality:.2f}" if 'dynamic_quality' in dir() else "N/A"
            print(f"[Brain] ✓ {processing_time:.2f}s | LLM: {self.llm_available} | quality: {_q_str}",
                  file=sys.stderr, flush=True)

            return {
                'response':          final_response,
                'message':           final_response,
                'intent':            intent,
                'sentiment':         sentiment,
                'reasoning':         reasoning,
                'needs_search':      intent['needs_search'],
                'search_query':      intent.get('search_query', ''),
                'searchPerformed':   len(ranked_results) > 0,
                'resultsCount':      len(ranked_results),
                'ranked_results':    ranked_results[:5],
                'neural_activity':   stats,
                'conversationId':    conversation_id or f"conv_{int(time.time())}",
                'confidence':        0.85,
                'llm_used':          self.llm_available,
                'llm_model':         self.llm_model,
                'facts_extracted':   facts_extracted,
                'dialogue_strategy': dialogue_decision.get('strategy', 'direct'),
                'processing_time':   processing_time
            }

        except Exception as e:
            print(f"[Brain] ERROR CRÍTICO en process_query: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return {
                'response':        "Disculpa, encontré un error al procesar tu mensaje. Intenta de nuevo.",
                'message':         "Error interno.",
                'error':           str(e),
                'conversationId':  conversation_id or f"conv_{int(time.time())}",
                'neural_activity': {'queries': self.total_queries}
            }

    # ─── Entrenamiento ────────────────────────────────────────────────

    def _dialogue_decision(self, msg_emb: np.ndarray, intent: dict) -> dict:
        """FIXED: resultado ahora se usa en generate()"""
        try:
            feats    = np.zeros(64, dtype=np.float32)
            feats[0] = 1.0 if intent.get('needs_search') else 0.0
            feats[1] = 1.0 if intent.get('is_greeting')  else 0.0
            feats[2] = 1.0 if intent.get('is_question')  else 0.0
            feats[3] = 1.0 if intent.get('is_internal')  else 0.0
            feats[4] = float(intent.get('confidence', 0.5))
            feats[5] = float(self.working.turn_count()) / 128.0
            feats[6] = 1.0 if self.working.current_topic() else 0.0
            feats[7] = float(len(self.episodic.episodes)) / 500000.0
            feats[8] = float(self.total_queries) / 10000.0
            feats[9] = float(self.llm_available)

            inp    = np.concatenate([msg_emb[:128], feats]).reshape(1, -1)
            out    = self.dialogue_net.predict(inp).flatten()
            labels = ['search', 'direct', 'ask', 'elaborate']
            return {'strategy': labels[int(np.argmax(out))],
                    'scores':   {labels[i]: float(out[i]) for i in range(4)}}
        except Exception as e:
            print(f"[DialogueNet] Error: {e}", file=sys.stderr, flush=True)
            return {'strategy': 'direct', 'scores': {}}

    def _lr_step(self, net_name: str, net, loss: float):
        history  = self._lr_history.setdefault(net_name, [])
        cooldown = self._lr_cooldown.get(net_name, 0)
        history.append(loss)
        if len(history) > 500:
            history[:] = history[-500:]
        epoch = net.epoch
        if epoch - cooldown < 200 or len(history) < 200:
            return
        recent = float(np.mean(history[-50:]))
        older  = float(np.mean(history[-200:-150]))
        if recent >= older * 0.97:
            new_lr = net.lr * 0.7
            if new_lr > 1e-6:
                net.lr = new_lr
                self._lr_cooldown[net_name] = epoch
                print(f"[LRScheduler] {net_name}: {net.lr/0.7:.2e} → {new_lr:.2e}",
                      file=sys.stderr, flush=True)

    def _episodic_search_smart(self, message: str, msg_emb: np.ndarray, top_k: int = 25) -> list:
        try:
            results = self.episodic.retrieve_similar(msg_emb, top_k=top_k, min_reward=0.0)
            if results:
                return results
        except Exception as e:
            print(f"[EpisodicSearch] Embedding error: {e}", file=sys.stderr, flush=True)
        return self.episodic.search(message, top_k=top_k)

    def _fit_inf_emb(self, text: str):
        """FIXED: limpia puntuación antes de tokenizar"""
        try:
            clean = re.sub(r'[^\w\sáéíóúñüÁÉÍÓÚÑÜ]', ' ', text.lower())
            for word in clean.split():
                if len(word) > 1:
                    self.inf_emb.add_word(word)
        except Exception as e:
            print(f"[InfEmb] Error: {e}", file=sys.stderr, flush=True)

    def _train_dialogue_net(self, msg_emb: np.ndarray, intent: dict):
        try:
            feats    = np.zeros(64, dtype=np.float32)
            feats[0] = 1.0 if intent.get('needs_search') else 0.0
            feats[1] = 1.0 if intent.get('is_greeting')  else 0.0
            feats[2] = 1.0 if intent.get('is_question')  else 0.0
            feats[3] = 1.0 if intent.get('is_internal')  else 0.0
            feats[4] = float(intent.get('confidence', 0.5))
            feats[5] = float(self.working.turn_count()) / 128.0

            inp    = np.concatenate([msg_emb[:128], feats]).reshape(1, -1)
            target = np.zeros((1, 4), dtype=np.float32)

            if intent.get('needs_search'):      target[0, 0] = 1.0
            elif intent.get('is_greeting') or intent.get('is_thanks'): target[0, 1] = 1.0
            elif intent.get('is_question'):     target[0, 3] = 1.0
            else:                               target[0, 1] = 1.0

            self.dialogue_net.train_step(inp, target)
        except Exception as e:
            print(f"[TrainDialogue] Error: {e}", file=sys.stderr, flush=True)

    def _train_context_net(self, msg_emb: np.ndarray, resp_emb: np.ndarray):
        """FIXED: usa EMBED_DIM dinámico — sin hardcodear 256"""
        try:
            ctx_embs = self.working.context_embeddings()
            if len(ctx_embs) >= 2:
                # ctx_summary del mismo tamaño que EMBED_DIM
                ctx_summary = np.mean(ctx_embs[-4:], axis=0)[:EMBED_DIM].astype(np.float32)
                if ctx_summary.shape[0] < EMBED_DIM:
                    ctx_summary = np.pad(ctx_summary, (0, EMBED_DIM - ctx_summary.shape[0]))
            else:
                ctx_summary = np.zeros(EMBED_DIM, dtype=np.float32)

            # inp tiene tamaño EMBED_DIM + EMBED_DIM = 2*EMBED_DIM
            inp    = np.concatenate([msg_emb, ctx_summary]).reshape(1, -1).astype(np.float32)
            target = np.array([[0.85]], dtype=np.float32)
            c_loss = self.context_net.train_step(inp, target)
            self._lr_step('context', self.context_net, c_loss)
        except Exception as e:
            print(f"[TrainContext] Error: {e}", file=sys.stderr, flush=True)

    def _detect_sentiment(self, msg_emb: np.ndarray) -> dict:
        try:
            inp        = msg_emb.reshape(1, -1)
            scores     = self.sentiment_net.predict(inp).flatten()
            labels     = ['positive', 'neutral', 'negative', 'urgent', 'confused']
            sentiment  = labels[int(np.argmax(scores))]
            confidence = float(np.max(scores))
            return {'label': sentiment, 'confidence': confidence,
                    'scores': {labels[i]: float(scores[i]) for i in range(min(len(labels), len(scores)))}}
        except Exception as e:
            print(f"[Sentiment] Error: {e}", file=sys.stderr, flush=True)
            return {'label': 'neutral', 'confidence': 0.5, 'scores': {}}

    def _get_brain_self_description(self) -> str:
        return self._build_self_description()

    def _build_self_description(self) -> str:
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
            widths = [net.layers[0].W.shape[0]] + [l.W.shape[1] for l in net.layers]
            arch   = '→'.join(str(w) for w in widths)
            lines.append(f"  • {name}: {len(net.layers)} capas [{arch}] — {params:,} params")

        ep_stats  = self.episodic.stats()
        sem_stats = self.semantic.stats()

        return (
            f"ARQUITECTURA REAL EN TIEMPO DE EJECUCIÓN:\n"
            f"  Versión: NEXUS v10.0 APEX\n"
            f"  Redes neuronales activas: {len(nets)}\n"
            + '\n'.join(lines)
            + f"\n  Parámetros totales: {total:,}\n\n"
            f"MEMORIA:\n"
            f"  WorkingMemory: {self.working.turn_count()}/{self.working.max_turns} turnos\n"
            f"  EpisodicMemory: {ep_stats.get('total', 0):,} episodios (cap: 500,000)\n"
            f"  SemanticMemory: {sem_stats.get('facts', 0):,} hechos aprendidos\n"
            f"  Vocabulario: {self.emb.vocab_size():,} n-gramas\n\n"
            f"ACTIVIDAD:\n"
            f"  Consultas: {self.total_queries:,} | Entrenamientos: {self.total_trainings:,}\n"
            f"  LLM: {'✅ ' + self.llm_model if self.llm_available else 'No — Smart Mode activo'}\n"
        )

    def _activity_report(self) -> dict:
        """FIXED: método con su def correcto — ya no está huérfano"""
        ep_stats  = self.episodic.stats()
        sem_stats = self.semantic.stats()
        return {
            'rank_loss':             self.rank_net.avg_recent_loss(100),
            'intent_loss':           self.intent_net.avg_recent_loss(100),
            'quality_loss':          self.conv_learner.response_quality_net.avg_recent_loss(100),
            'context_loss':          self.context_net.avg_recent_loss(100),
            'sentiment_loss':        self.sentiment_net.avg_recent_loss(100),
            'meta_loss':             self.meta_net.avg_recent_loss(100),
            'relevance_loss':        self.relevance_net.avg_recent_loss(100),
            'dialogue_loss':         self.dialogue_net.avg_recent_loss(100),
            'vocab_size':            self.emb.vocab_size(),
            'episodes':              ep_stats.get('total', 0),
            'semantic_facts':        sem_stats.get('facts', 0),
            'trainings':             self.total_trainings,
            'queries':               self.total_queries,
            'working_memory_turns':  self.working.turn_count(),
            'conversation_patterns': len(self.conv_learner.conversation_db['successful_patterns']),
            'llm_available':         self.llm_available,
            'llm_model':             self.llm_model,
            'current_topic':         self.working.current_topic(),
            'total_parameters':      self.total_parameters,
            'cache_hits':            self._cache_hits,
            'networks_active':       8,
            'version':               'v10.0_APEX',
        }

    def save_all(self):
        """Guarda todo — local y MongoDB"""
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
            json.dump({'total_queries': self.total_queries,
                       'total_trainings': self.total_trainings}, f)

        # FIXED: MongoDB guarda 5000 episodios (era 200)
        if MONGO_OK and _mongo_db is not None:
            try:
                _mongo_db.meta.update_one({'_id': 'nexus_meta'}, {'$set': {
                    'total_queries': self.total_queries,
                    'total_trainings': self.total_trainings,
                    'ts': time.time()
                }}, upsert=True)

                if self.episodic.episodes:
                    eps_docs = [{k: v for k, v in ep.items() if k != 'emb'}
                                for ep in self.episodic.episodes[-5000:]]  # FIXED: era -200
                    _mongo_db.episodic.delete_many({})
                    _mongo_db.episodic.insert_many(eps_docs)

                _mongo_db.semantic.update_one({'_id': 'semantic'}, {'$set': {
                    'facts': self.semantic.facts,
                    'preferences': self.semantic.preferences,
                    'query_clusters': self.semantic.query_clusters
                }}, upsert=True)

                _mongo_db.patterns.update_one({'_id': 'patterns'}, {'$set': {
                    'successful': self.conv_learner.conversation_db['successful_patterns'][-5000:],
                    'failed':     self.conv_learner.conversation_db['failed_patterns'][-2000:],
                    'ts':         time.time()
                }}, upsert=True)

            except Exception as e:
                print(f"[MongoDB] Error guardando: {e}", file=sys.stderr, flush=True)

    # ─── Feedback externo ─────────────────────────────────────────────

    def train_from_feedback(self, query: str, result: dict, helpful: bool):
        try:
            emb_q  = self.emb.embed(query)
            emb_r  = self.emb.embed(result.get('title', '') + ' ' + result.get('description', ''))
            feats  = np.array([
                len(result.get('title', '')), len(result.get('description', '')),
                1.0 if 'wikipedia' in result.get('url', '') else 0.0,
                result.get('_position', 1) / 10.0
            ])
            inp    = np.concatenate([emb_q, emb_r, feats]).reshape(1, -1)
            target = np.array([[1.0 if helpful else 0.0]], dtype=np.float32)
            loss   = self.rank_net.train_step(inp, target)
            self.total_trainings += 1
            if self.total_trainings % 10 == 0:
                print(f"[RankNet] #{self.total_trainings}, Loss: {loss:.4f}", file=sys.stderr, flush=True)
            self.save_all()
            return {'loss': float(loss), 'trainings': self.total_trainings}
        except Exception as e:
            print(f"[RankNet] Error: {e}", file=sys.stderr, flush=True)
            return {'loss': 0.0, 'trainings': self.total_trainings}

    def learn_from_click(self, query: str, url: str, position: int,
                         dwell_time: float, bounced: bool):
        reward_delta = 0.0
        if dwell_time > 30 and not bounced:  reward_delta = 0.2
        elif dwell_time > 10:                reward_delta = 0.1
        elif bounced or dwell_time < 5:      reward_delta = -0.1

        self.episodic.update_reward(query, url, reward_delta)

        if reward_delta > 0:
            domain = url.split('//')[-1].split('/')[0]
            self.semantic.update_preference(f'domain:{domain}', reward_delta * 0.1)

        for ep in reversed(self.episodic.episodes[-50:]):
            if ep.get('query') == query:
                for res in ep.get('results', []):
                    if res.get('url') == url:
                        self.train_from_feedback(query, res, reward_delta > 0)
                        break
                break

        self.save_all()

    def learn(self, message: str, response: str, was_helpful: bool = True, search_results: list = []):
        try:
            msg_emb  = self.emb.embed(message)
            resp_emb = self.emb.embed(response)
            quality  = 0.92 if was_helpful else 0.2

            rel_inp    = np.concatenate([msg_emb, resp_emb]).reshape(1, -1)
            rel_target = np.array([[quality]], dtype=np.float32)

            for _pass in range(3):
                q_loss = self.conv_learner.train_quality_net(msg_emb, resp_emb, quality)
                self._lr_step('quality', self.conv_learner.response_quality_net, q_loss)
                r_loss = self.relevance_net.train_step(rel_inp, rel_target)
                self._lr_step('relevance', self.relevance_net, r_loss)

            if search_results:
                for result in search_results[:8]:
                    self.train_from_feedback(message, result, was_helpful)

            self.conv_learner.learn_from_interaction(message, response, 0.92 if was_helpful else 0.15)

            self.emb.fit_text(message)
            self.emb.fit_text(response)
            self._fit_inf_emb(message)
            self._fit_inf_emb(response)
            if was_helpful and len(response) > 20:
                self.emb.update_pair(message, response, label=1.0, lr=0.006)

            self.total_parameters = self._count_parameters()
            self.total_trainings  += 3
            self.save_all()
            print(f"[Brain] Learn: {self.total_trainings} entrenamientos | {self.total_parameters:,} params",
                  file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[Brain] Error en learn: {e}", file=sys.stderr, flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  SERVIDOR JSON — STDIN/STDOUT
# ═══════════════════════════════════════════════════════════════════════

def main():
    brain = NexusBrain()
    print("✅ [Brain] Listo para recibir comandos JSON", file=sys.stderr, flush=True)
    print("✓ Brain listo", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req        = json.loads(line)
            action     = req.get('action', 'process')
            request_id = req.get('_requestId')

            if action == 'process':
                message  = req.get('message', '')
                history  = req.get('conversation_history', []) or req.get('history', [])
                results  = req.get('search_results')
                conv_id  = req.get('conversation_id')
                user_ctx = req.get('user_context')
                response = brain.process_query(message, history, results, conv_id, user_ctx)
                response['_requestId'] = request_id
                print(json.dumps(response, ensure_ascii=False), flush=True)

            elif action == 'click':
                brain.learn_from_click(
                    req.get('query', ''), req.get('url', ''),
                    req.get('position', 1), req.get('dwell_time', 0), req.get('bounced', False)
                )
                print(json.dumps({'status': 'ok', '_requestId': request_id}), flush=True)

            elif action == 'learn':
                brain.learn(
                    req.get('message', ''), req.get('response', ''),
                    req.get('was_helpful', True), req.get('search_results', [])
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
