#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS AI - Intent Analyzer Module
Análisis avanzado de intención de búsqueda usando NLP básico
"""

import re
from collections import Counter
from typing import Dict, List, Tuple

class IntentAnalyzer:
    """Analizador de intención de búsqueda"""
    
    def __init__(self):
        # Patrones de intención
        self.intent_patterns = {
            'download': {
                'keywords': ['download', 'descargar', 'bajar', 'gratis', 'free', 'pdf', 'epub', 
                            'torrent', 'mega', 'mediafire', 'direct', 'link'],
                'weight': 1.0
            },
            'learn': {
                'keywords': ['tutorial', 'learn', 'aprender', 'cómo', 'how to', 'guide', 
                            'curso', 'clase', 'lesson', 'enseñar'],
                'weight': 0.9
            },
            'code': {
                'keywords': ['code', 'programming', 'programación', 'github', 'example', 
                            'snippet', 'python', 'javascript', 'java', 'c++', 'algorithm'],
                'weight': 0.9
            },
            'research': {
                'keywords': ['paper', 'research', 'study', 'journal', 'article', 'científico',
                            'thesis', 'dissertation', 'arxiv', 'scholar'],
                'weight': 0.85
            },
            'video': {
                'keywords': ['video', 'watch', 'ver', 'youtube', 'stream', 'película',
                            'movie', 'series', 'documentary'],
                'weight': 0.8
            },
            'news': {
                'keywords': ['news', 'noticias', 'latest', 'breaking', 'today', 'hoy',
                            'current', 'acontecimiento'],
                'weight': 0.75
            },
            'shop': {
                'keywords': ['buy', 'comprar', 'price', 'precio', 'shop', 'store',
                            'tienda', 'amazon', 'ebay'],
                'weight': 0.7
            },
            'local': {
                'keywords': ['near me', 'cerca', 'local', 'restaurant', 'hotel',
                            'direcciones', 'directions', 'maps'],
                'weight': 0.7
            }
        }
        
        # Modificadores de consulta
        self.modifiers = {
            'best': 1.2,
            'mejor': 1.2,
            'top': 1.15,
            'latest': 1.1,
            'nuevo': 1.1,
            'official': 1.25,
            'oficial': 1.25,
            'free': 1.1,
            'gratis': 1.1
        }
        
        # Stopwords en español e inglés
        self.stopwords = {
            'el', 'la', 'de', 'en', 'y', 'a', 'los', 'las', 'un', 'una', 'para',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has'
        }
    
    def analyze(self, query: str) -> Dict:
        """
        Analizar intención de búsqueda
        
        Returns:
            Dict con tipo, confianza y metadata
        """
        query_lower = query.lower()
        words = self.tokenize(query_lower)
        
        # Detectar intenciones
        intent_scores = {}
        for intent_type, config in self.intent_patterns.items():
            score = 0
            matches = []
            
            for keyword in config['keywords']:
                if keyword in query_lower:
                    score += config['weight']
                    matches.append(keyword)
            
            if score > 0:
                intent_scores[intent_type] = {
                    'score': score,
                    'matches': matches
                }
        
        # Aplicar modificadores
        modifier_boost = 1.0
        modifier_matches = []
        for modifier, boost in self.modifiers.items():
            if modifier in query_lower:
                modifier_boost *= boost
                modifier_matches.append(modifier)
        
        # Determinar intención principal
        primary_intent = 'general'
        confidence = 0.5
        
        if intent_scores:
            primary_intent = max(intent_scores.items(), key=lambda x: x[1]['score'])[0]
            max_score = intent_scores[primary_intent]['score']
            confidence = min(max_score / 3.0, 1.0) * modifier_boost
        
        # Extraer entidades nombradas (básico)
        entities = self.extract_entities(query)
        
        # Detectar preguntas
        is_question = self.is_question(query)
        
        return {
            'primary': primary_intent,
            'confidence': round(confidence, 2),
            'all_intents': intent_scores,
            'modifiers': modifier_matches,
            'entities': entities,
            'is_question': is_question,
            'word_count': len(words),
            'complexity': self.calculate_complexity(words)
        }
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenizar texto"""
        # Eliminar puntuación pero mantener palabras
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        # Filtrar stopwords
        return [w for w in words if w not in self.stopwords and len(w) > 1]
    
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extraer entidades nombradas básicas"""
        entities = {
            'technologies': [],
            'file_types': [],
            'brands': [],
            'years': []
        }
        
        # Tecnologías
        tech_keywords = ['python', 'javascript', 'java', 'c++', 'react', 'node',
                        'django', 'flask', 'tensorflow', 'pytorch', 'docker']
        for tech in tech_keywords:
            if tech in query.lower():
                entities['technologies'].append(tech)
        
        # Tipos de archivo
        file_types = re.findall(r'\.(pdf|epub|docx|xlsx|pptx|mp4|mp3|zip|rar)', query.lower())
        entities['file_types'] = list(set(file_types))
        
        # Años (1900-2099)
        years = re.findall(r'\b(19|20)\d{2}\b', query)
        entities['years'] = years
        
        # Marcas comunes
        brands = ['google', 'microsoft', 'apple', 'amazon', 'facebook', 'netflix']
        for brand in brands:
            if brand in query.lower():
                entities['brands'].append(brand)
        
        return entities
    
    def is_question(self, query: str) -> bool:
        """Detectar si es una pregunta"""
        question_words = ['what', 'qué', 'how', 'cómo', 'why', 'por qué', 
                         'when', 'cuándo', 'where', 'dónde', 'who', 'quién']
        query_lower = query.lower()
        
        # Empieza con palabra interrogativa o tiene signo de interrogación
        starts_with_q = any(query_lower.startswith(qw) for qw in question_words)
        has_question_mark = '?' in query
        
        return starts_with_q or has_question_mark
    
    def calculate_complexity(self, words: List[str]) -> str:
        """Calcular complejidad de la consulta"""
        word_count = len(words)
        
        if word_count <= 2:
            return 'simple'
        elif word_count <= 5:
            return 'medium'
        else:
            return 'complex'
    
    def suggest_filters(self, intent_data: Dict) -> List[str]:
        """Sugerir filtros basados en la intención"""
        primary = intent_data['primary']
        
        filter_map = {
            'download': ['mediafire', 'google', 'archive'],
            'code': ['github', 'stackoverflow'],
            'video': ['youtube'],
            'research': ['arxiv', 'scholar', 'medium'],
            'learn': ['youtube', 'medium'],
            'news': ['reddit', 'news']
        }
        
        return filter_map.get(primary, ['all'])
    
    def generate_query_variations(self, query: str, intent_data: Dict) -> List[str]:
        """Generar variaciones de la consulta para mejorar resultados"""
        variations = [query]
        primary = intent_data['primary']
        
        # Añadir términos según intención
        if primary == 'download':
            variations.append(f"{query} pdf")
            variations.append(f"{query} free download")
        
        elif primary == 'learn':
            variations.append(f"{query} tutorial")
            variations.append(f"how to {query}")
        
        elif primary == 'code':
            variations.append(f"{query} example")
            variations.append(f"{query} github")
        
        elif primary == 'video':
            variations.append(f"{query} video")
        
        return variations[:3]  # Limitar a 3 variaciones


# ══════════════════════════════════════════════════════════════════════
# TESTING
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    analyzer = IntentAnalyzer()
    
    # Test queries
    test_queries = [
        "python tutorial",
        "download pdf gratis machine learning",
        "best react examples github",
        "cómo aprender javascript",
        "latest news about AI",
        "research paper deep learning",
        "buy iphone 15 pro"
    ]
    
    print("\n🎯 INTENT ANALYZER TEST\n" + "="*50)
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        result = analyzer.analyze(query)
        
        print(f"   Primary Intent: {result['primary']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Is Question: {result['is_question']}")
        print(f"   Complexity: {result['complexity']}")
        
        if result['entities']['technologies']:
            print(f"   Technologies: {', '.join(result['entities']['technologies'])}")
        
        if result['modifiers']:
            print(f"   Modifiers: {', '.join(result['modifiers'])}")
        
        suggestions = analyzer.suggest_filters(result)
        print(f"   Suggested Filters: {', '.join(suggestions)}")
        
        variations = analyzer.generate_query_variations(query, result)
        if len(variations) > 1:
            print(f"   Query Variations:")
            for var in variations[1:]:
                print(f"      - {var}")
    
    print("\n" + "="*50)
    print("✅ Intent Analysis Complete!\n")
