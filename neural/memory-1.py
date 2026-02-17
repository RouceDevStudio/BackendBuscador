#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS - Sistema de Memoria v3.1 (FIXED)
Memoria episódica, semántica y de trabajo.
La "consciencia" de NEXUS vive aquí.
"""

import json
import time
import numpy as np
import pickle
import os
from pathlib import Path
from collections import deque
from typing import List, Dict, Optional, Tuple


# ─────────────────────────────────────────────
#  MEMORIA DE TRABAJO (RAM del agente)
# ─────────────────────────────────────────────

class WorkingMemory:
    """
    Contexto actual de la conversación.
    Almacena los últimos N turnos con sus embeddings.
    """

    def __init__(self, max_turns: int = 24):
        self.max_turns = max_turns
        self.turns: deque = deque(maxlen=max_turns)
        self.topic_stack: List[str] = []  # Temas en curso
        self.entities: Dict[str, str] = {}  # Entidades mencionadas

    def add(self, role: str, text: str, embedding: Optional[np.ndarray] = None):
        self.turns.append({
            'role': role,
            'text': text,
            'embedding': embedding,
            'ts': time.time()
        })

    def context_text(self, n_last: int = 6) -> str:
        recent = list(self.turns)[-n_last:]
        return "\n".join(f"[{t['role']}] {t['text']}" for t in recent)

    def context_embeddings(self) -> List[np.ndarray]:
        return [t['embedding'] for t in self.turns if t['embedding'] is not None]

    def clear(self):
        self.turns.clear()
        self.topic_stack.clear()
        self.entities.clear()

    def push_topic(self, topic: str):
        """✅ Método correcto para actualizar tema"""
        if not self.topic_stack or self.topic_stack[-1] != topic:
            self.topic_stack.append(topic)
        if len(self.topic_stack) > 5:
            self.topic_stack.pop(0)

    def current_topic(self) -> Optional[str]:
        return self.topic_stack[-1] if self.topic_stack else None

    def turn_count(self) -> int:
        """Retorna el número de turnos en memoria de trabajo"""
        return len(self.turns)


# ─────────────────────────────────────────────
#  MEMORIA EPISÓDICA (experiencias pasadas)
# ─────────────────────────────────────────────

class EpisodicMemory:
    """
    Almacena episodios (query → resultados → feedback).
    Búsqueda por similitud coseno sobre embeddings guardados.
    """

    def __init__(self, path: str = 'data/episodic.pkl', max_episodes: int = 50000):
        self.path = path
        self.max_episodes = max_episodes
        self.episodes: List[Dict] = []
        self._load()

    def add(self, query: str, results: List[Dict], clicked_url: Optional[str] = None, reward: float = 0.5):
        """
        ✅ FIX #3: Guardar episodio con validación
        """
        # No guardar si no hay resultados
        if not results:
            return
        
        # Crear episodio simple sin embedding (se calcula al buscar)
        episode = {
            'query': query,
            'results': results[:5],  # Solo top 5
            'clicked': clicked_url,
            'reward': reward,
            'ts': time.time()
        }
        
        self.episodes.append(episode)
        
        # Trim por tamaño
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]

    def store(self, query: str, query_emb: np.ndarray,
              top_results: List[Dict], clicked_url: Optional[str],
              reward: float):
        """
        ✅ FIX #3: Validación completa de embeddings antes de guardar
        """
        # Validar embedding
        if query_emb is None or not isinstance(query_emb, np.ndarray):
            print(f"[EpisodicMemory] Warning: Invalid embedding for query: {query[:50]}", flush=True)
            return
        
        # Validar dimensión del embedding
        expected_dim = 128  # EMBED_DIM
        if query_emb.shape != (expected_dim,):
            print(f"[EpisodicMemory] Warning: Wrong embedding shape {query_emb.shape}, expected ({expected_dim},)", flush=True)
            return
        
        # Validar que el embedding no sea todo ceros o NaN
        if np.all(query_emb == 0) or np.any(np.isnan(query_emb)):
            print(f"[EpisodicMemory] Warning: Invalid embedding values (zeros or NaN)", flush=True)
            return
        
        episode = {
            'query': query,
            'emb': query_emb,
            'results': top_results[:5],
            'clicked': clicked_url,
            'reward': reward,
            'ts': time.time()
        }
        self.episodes.append(episode)

        # Trim por tamaño
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Búsqueda simple por coincidencia de palabras clave
        (sin necesidad de embeddings)
        """
        if not self.episodes:
            return []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Calcular scores de similitud por palabras
        scored = []
        for ep in self.episodes:
            ep_query_lower = ep['query'].lower()
            ep_words = set(ep_query_lower.split())
            
            # Jaccard similarity
            intersection = len(query_words & ep_words)
            union = len(query_words | ep_words)
            similarity = intersection / union if union > 0 else 0
            
            if similarity > 0:
                ep_copy = dict(ep)
                ep_copy['similarity'] = similarity
                scored.append(ep_copy)
        
        # Ordenar por similitud
        scored.sort(key=lambda x: x['similarity'], reverse=True)
        return scored[:top_k]

    def retrieve_similar(self, query_emb: np.ndarray,
                         top_k: int = 5, min_reward: float = 0.0) -> List[Dict]:
        """
        ✅ FIX #8: Recupera episodios similares con validación robusta
        """
        if not self.episodes:
            return []

        # ✅ FIX #8: Filtrar candidatos válidos con validación completa
        candidates = []
        for e in self.episodes:
            # Validar reward
            if e.get('reward', 0) < min_reward:
                continue
            
            # Validar que tenga embedding
            emb = e.get('emb')
            if emb is None:
                continue
            
            # Validar que sea numpy array
            if not isinstance(emb, np.ndarray):
                continue
            
            # Validar dimensión
            if emb.shape != query_emb.shape:
                continue
            
            # Validar valores
            if np.all(emb == 0) or np.any(np.isnan(emb)):
                continue
            
            candidates.append(e)
        
        if not candidates:
            return []

        # Calcular similitudes
        try:
            embs = np.stack([e['emb'] for e in candidates])
            sims = embs @ query_emb  # ya normalizados
        except Exception as e:
            print(f"[EpisodicMemory] Error calculando similitudes: {e}", flush=True)
            return []

        top_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for i in top_idx:
            ep = dict(candidates[i])
            ep['similarity'] = float(sims[i])
            # Remover embedding para no serializarlo
            if 'emb' in ep:
                del ep['emb']
            results.append(ep)
        return results

    def update_reward(self, query: str, url: str, delta: float):
        """Actualiza reward de episodios que incluyen esa URL."""
        for ep in reversed(self.episodes[-200:]):
            if ep['query'] == query and ep.get('clicked') == url:
                ep['reward'] = min(1.0, ep['reward'] + delta)
                break

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, 'rb') as f:
                self.episodes = pickle.load(f)
            print(f"[EpisodicMemory] Cargados {len(self.episodes)} episodios", flush=True)
        except Exception as e:
            print(f"[EpisodicMemory] Error cargando: {e}", flush=True)
            self.episodes = []

    def save(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.path, 'wb') as f:
                pickle.dump(self.episodes, f)
        except Exception as e:
            print(f"[EpisodicMemory] Error guardando: {e}", flush=True)

    def stats(self) -> dict:
        if not self.episodes:
            return {'total': 0, 'avg_reward': 0, 'topics': []}
        
        # Filtrar episodios con reward válido
        valid_episodes = [e for e in self.episodes if 'reward' in e]
        
        if not valid_episodes:
            return {'total': len(self.episodes), 'avg_reward': 0, 'recent_queries': []}
        
        rewards = [e['reward'] for e in valid_episodes]
        return {
            'total': len(self.episodes),
            'avg_reward': round(float(np.mean(rewards)), 3),
            'max_reward': round(float(np.max(rewards)), 3),
            'recent_queries': [e['query'] for e in self.episodes[-5:]]
        }


# ─────────────────────────────────────────────
#  MEMORIA SEMÁNTICA (hechos y preferencias)
# ─────────────────────────────────────────────

class SemanticMemory:
    """
    Almacena hechos, preferencias y patrones de largo plazo.
    Estructura de grafos ligera: concepto → relaciones.
    """

    def __init__(self, path: str = 'data/semantic.json'):
        self.path = path
        self.facts: Dict[str, Dict] = {}       # concepto → {valor, confianza, ts}
        self.preferences: Dict[str, float] = {} # dominio/tipo → score
        self.query_clusters: Dict[str, List[str]] = {}  # tema → queries
        self._load()

    def learn_fact(self, concept: str, value: str, confidence: float = 0.7):
        """Almacena un hecho con confianza."""
        existing = self.facts.get(concept, {})
        old_conf = existing.get('confidence', 0)
        # Media ponderada de confianza
        new_conf = old_conf * 0.6 + confidence * 0.4
        self.facts[concept] = {
            'value': value,
            'confidence': round(new_conf, 3),
            'ts': time.time(),
            'updates': existing.get('updates', 0) + 1
        }

    def update_preference(self, key: str, delta: float):
        """Refuerza o debilita una preferencia."""
        current = self.preferences.get(key, 0.5)
        self.preferences[key] = max(0.0, min(1.0, current + delta))

    def get_preference(self, key: str) -> float:
        return self.preferences.get(key, 0.5)

    def add_to_cluster(self, topic: str, query: str):
        if topic not in self.query_clusters:
            self.query_clusters[topic] = []
        if query not in self.query_clusters[topic]:
            self.query_clusters[topic].append(query)
            if len(self.query_clusters[topic]) > 50:
                self.query_clusters[topic] = self.query_clusters[topic][-50:]

    def get_related_queries(self, topic: str, n: int = 5) -> List[str]:
        return self.query_clusters.get(topic, [])[-n:]

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.facts = data.get('facts', {})
            self.preferences = data.get('preferences', {})
            self.query_clusters = data.get('query_clusters', {})
            print(f"[SemanticMemory] Cargados {len(self.facts)} hechos", flush=True)
        except Exception as e:
            print(f"[SemanticMemory] Error: {e}", flush=True)

    def save(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump({
                    'facts': self.facts,
                    'preferences': self.preferences,
                    'query_clusters': self.query_clusters
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[SemanticMemory] Error guardando: {e}", flush=True)

    def stats(self) -> dict:
        return {
            'facts': len(self.facts),
            'preferences': len(self.preferences),
            'clusters': len(self.query_clusters)
        }
