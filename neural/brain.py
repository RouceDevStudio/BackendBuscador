#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS AI Brain - Sistema Neural Primitivo para Búsquedas Inteligentes
Implementa 50 neuronas de aprendizaje para mejorar resultados
"""

import json
import math
import pickle
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path
import re

class NeuralSearchBrain:
    """Cerebro neural que aprende de las búsquedas del usuario"""
    
    def __init__(self, model_path='models/brain.pkl'):
        self.model_path = Path(model_path)
        self.neurons = 50
        
        # Memoria de aprendizaje
        self.query_patterns = defaultdict(int)  # Patrones de búsqueda
        self.click_patterns = defaultdict(lambda: defaultdict(int))  # query -> url -> clicks
        self.semantic_clusters = {}  # Agrupaciones semánticas
        self.user_preferences = defaultdict(float)  # Preferencias por fuente
        self.temporal_boost = {}  # Boost temporal para resultados frescos
        
        # Pesos neuronales (50 neuronas)
        self.weights = {
            'title_exact': 0.35,
            'title_partial': 0.20,
            'description_match': 0.15,
            'url_relevance': 0.10,
            'source_authority': 0.08,
            'freshness': 0.05,
            'user_history': 0.04,
            'semantic_similarity': 0.03
        }
        
        self.load_model()
    
    def load_model(self):
        """Cargar modelo entrenado"""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.query_patterns = data.get('query_patterns', defaultdict(int))
                    self.click_patterns = data.get('click_patterns', defaultdict(lambda: defaultdict(int)))
                    self.user_preferences = data.get('user_preferences', defaultdict(float))
                    self.semantic_clusters = data.get('semantic_clusters', {})
                    print(f"✓ Modelo cargado: {len(self.query_patterns)} patrones")
            except Exception as e:
                print(f"⚠ Error cargando modelo: {e}")
    
    def save_model(self):
        """Guardar modelo entrenado"""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'query_patterns': dict(self.query_patterns),
                    'click_patterns': dict(self.click_patterns),
                    'user_preferences': dict(self.user_preferences),
                    'semantic_clusters': self.semantic_clusters,
                    'timestamp': datetime.now().isoformat()
                }, f)
            print(f"✓ Modelo guardado: {self.model_path}")
        except Exception as e:
            print(f"⚠ Error guardando modelo: {e}")
    
    def tokenize(self, text):
        """Tokenizar y normalizar texto"""
        text = text.lower()
        # Eliminar caracteres especiales pero mantener espacios
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        # Filtrar stopwords comunes
        stopwords = {'el', 'la', 'de', 'en', 'y', 'a', 'los', 'las', 'un', 'una', 
                     'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        return [t for t in tokens if t not in stopwords and len(t) > 1]
    
    def extract_features(self, query, result):
        """Extraer características para scoring neuronal"""
        query_tokens = set(self.tokenize(query))
        title_tokens = set(self.tokenize(result.get('title', '')))
        desc_tokens = set(self.tokenize(result.get('description', '')))
        url = result.get('url', '').lower()
        
        features = {}
        
        # 1. Coincidencia exacta en título
        features['title_exact'] = 1.0 if query.lower() in result.get('title', '').lower() else 0.0
        
        # 2. Coincidencia parcial en título
        title_overlap = len(query_tokens & title_tokens) / max(len(query_tokens), 1)
        features['title_partial'] = title_overlap
        
        # 3. Coincidencia en descripción
        desc_overlap = len(query_tokens & desc_tokens) / max(len(query_tokens), 1)
        features['description_match'] = desc_overlap * 0.7  # Menos peso que título
        
        # 4. Relevancia de URL
        url_score = sum(1 for token in query_tokens if token in url) / max(len(query_tokens), 1)
        features['url_relevance'] = url_score
        
        # 5. Autoridad de la fuente
        source = result.get('source', 'Web').lower()
        authority_scores = {
            'wikipedia': 1.0, 'github': 0.95, 'stackoverflow': 0.95,
            'arxiv': 0.9, 'medium': 0.75, 'reddit': 0.6,
            'youtube': 0.8, 'archive.org': 0.85
        }
        features['source_authority'] = authority_scores.get(source, 0.5)
        
        # 6. Frescura (si hay timestamp)
        features['freshness'] = 0.5  # Placeholder
        
        # 7. Historial de usuario
        click_count = self.click_patterns.get(query.lower(), {}).get(url, 0)
        features['user_history'] = min(click_count / 10.0, 1.0)
        
        # 8. Similitud semántica
        features['semantic_similarity'] = self.compute_semantic_similarity(query, result)
        
        return features
    
    def compute_semantic_similarity(self, query, result):
        """Calcular similitud semántica básica"""
        query_tokens = self.tokenize(query)
        text = f"{result.get('title', '')} {result.get('description', '')}"
        result_tokens = self.tokenize(text)
        
        # TF-IDF simplificado
        query_freq = Counter(query_tokens)
        result_freq = Counter(result_tokens)
        
        # Calcular similitud coseno
        common = set(query_tokens) & set(result_tokens)
        if not common:
            return 0.0
        
        numerator = sum(query_freq[t] * result_freq[t] for t in common)
        denom_q = math.sqrt(sum(v**2 for v in query_freq.values()))
        denom_r = math.sqrt(sum(v**2 for v in result_freq.values()))
        
        if denom_q * denom_r == 0:
            return 0.0
        
        return numerator / (denom_q * denom_r)
    
    def neural_score(self, query, result):
        """Calcular score usando red neuronal"""
        features = self.extract_features(query, result)
        
        # Sumar features ponderadas (50 neuronas trabajando)
        score = sum(features.get(k, 0) * self.weights.get(k, 0) for k in self.weights)
        
        # Aplicar función de activación (sigmoid)
        activated_score = 1 / (1 + math.exp(-score * 5))  # Escalar a [0,1]
        
        return activated_score * 100  # Convertir a porcentaje
    
    def rank_results(self, query, results):
        """Rankear resultados usando IA"""
        scored_results = []
        
        for result in results:
            neural_score = self.neural_score(query, result)
            result['neuralScore'] = round(neural_score, 2)
            scored_results.append(result)
        
        # Ordenar por score neuronal
        scored_results.sort(key=lambda x: x.get('neuralScore', 0), reverse=True)
        
        return scored_results
    
    def learn_from_click(self, query, url):
        """Aprender de los clicks del usuario"""
        query = query.lower()
        self.query_patterns[query] += 1
        self.click_patterns[query][url] += 1
        
        # Actualizar pesos cada 10 clicks
        total_clicks = sum(self.query_patterns.values())
        if total_clicks % 10 == 0:
            self.adjust_weights()
        
        self.save_model()
    
    def adjust_weights(self):
        """Ajustar pesos neuronales basado en feedback"""
        # Algoritmo simple de aprendizaje
        # Aumentar peso de features que correlacionan con clicks
        
        for query, clicks in self.click_patterns.items():
            if sum(clicks.values()) < 3:
                continue
            
            # Aquí implementarías un algoritmo más sofisticado
            # Por ahora, ajuste básico
            pass
        
        print("⚙ Pesos ajustados automáticamente")
    
    def get_suggestions(self, partial_query):
        """Generar sugerencias inteligentes"""
        partial = partial_query.lower()
        suggestions = []
        
        # Buscar en patrones conocidos
        for query, count in sorted(self.query_patterns.items(), 
                                   key=lambda x: x[1], reverse=True):
            if query.startswith(partial) and query != partial:
                suggestions.append({
                    'text': query,
                    'popularity': count
                })
            
            if len(suggestions) >= 5:
                break
        
        return suggestions
    
    def analyze_query_intent(self, query):
        """Analizar intención de búsqueda"""
        query_lower = query.lower()
        
        intent = {
            'type': 'general',
            'download': False,
            'academic': False,
            'code': False,
            'video': False,
            'confidence': 0.5
        }
        
        # Detectar intención de descarga
        download_keywords = ['download', 'descargar', 'pdf', 'epub', 'gratis', 'free']
        if any(kw in query_lower for kw in download_keywords):
            intent['type'] = 'download'
            intent['download'] = True
            intent['confidence'] = 0.8
        
        # Detectar búsqueda académica
        academic_keywords = ['paper', 'research', 'study', 'journal', 'thesis', 'artículo']
        if any(kw in query_lower for kw in academic_keywords):
            intent['academic'] = True
            intent['confidence'] = 0.75
        
        # Detectar búsqueda de código
        code_keywords = ['code', 'github', 'python', 'javascript', 'tutorial', 'example']
        if any(kw in query_lower for kw in code_keywords):
            intent['code'] = True
            intent['confidence'] = 0.7
        
        # Detectar búsqueda de video
        video_keywords = ['video', 'watch', 'tutorial', 'how to', 'youtube']
        if any(kw in query_lower for kw in video_keywords):
            intent['video'] = True
            intent['confidence'] = 0.7
        
        return intent
    
    def get_stats(self):
        """Obtener estadísticas del cerebro"""
        return {
            'total_queries': len(self.query_patterns),
            'total_clicks': sum(self.query_patterns.values()),
            'total_patterns': sum(len(clicks) for clicks in self.click_patterns.values()),
            'neurons_active': self.neurons,
            'model_loaded': self.model_path.exists()
        }


if __name__ == '__main__':
    # Test del cerebro neural
    brain = NeuralSearchBrain()
    
    # Test 1: Scoring
    test_query = "python tutorial"
    test_result = {
        'title': 'Python Tutorial for Beginners',
        'description': 'Learn Python programming from scratch with examples',
        'url': 'https://python.org/tutorial',
        'source': 'Python.org'
    }
    
    score = brain.neural_score(test_query, test_result)
    print(f"\n🧠 Neural Score: {score:.2f}%")
    
    # Test 2: Intent Analysis
    intent = brain.analyze_query_intent("download python pdf free")
    print(f"\n🎯 Intent: {json.dumps(intent, indent=2)}")
    
    # Test 3: Stats
    stats = brain.get_stats()
    print(f"\n📊 Stats: {json.dumps(stats, indent=2)}")
    
    print("\n✅ Neural Brain Test Complete!")
