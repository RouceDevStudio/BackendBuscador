"""
NEXUS RANKING ENGINE
Sistema de ranking inteligente que aprende de las interacciones del usuario
"""

import numpy as np
import json
from datetime import datetime, timedelta
from collections import defaultdict
import math

class RankingEngine:
    """Motor de ranking con aprendizaje automático"""
    
    def __init__(self):
        self.click_data = defaultdict(list)  # query -> [(url, position, timestamp)]
        self.url_quality = defaultdict(float)  # url -> quality score
        self.query_url_relevance = defaultdict(lambda: defaultdict(float))  # query -> url -> relevance
        self.decay_factor = 0.95  # Factor de decaimiento temporal
        
    def calculate_ctr(self, url: str, query: str = None) -> float:
        """Calcular Click-Through Rate"""
        if query:
            clicks = sum(1 for q, u, _, _ in self.click_data.get(query, []) if u == url)
            impressions = len(self.click_data.get(query, []))
        else:
            clicks = sum(len([c for c in clicks if c[1] == url]) 
                        for clicks in self.click_data.values())
            impressions = sum(len(clicks) for clicks in self.click_data.values())
        
        return clicks / max(impressions, 1)
    
    def calculate_dwell_time_score(self, dwell_time: float) -> float:
        """
        Convertir tiempo de permanencia en score de calidad
        < 3s = 0.1 (bounce)
        3-10s = 0.3
        10-30s = 0.6
        30s-2m = 0.8
        > 2m = 1.0
        """
        if dwell_time < 3:
            return 0.1
        elif dwell_time < 10:
            return 0.3
        elif dwell_time < 30:
            return 0.6
        elif dwell_time < 120:
            return 0.8
        else:
            return 1.0
    
    def record_click(self, query: str, url: str, position: int, 
                    dwell_time: float = None, bounced: bool = False):
        """Registrar un click y actualizar scores"""
        timestamp = datetime.now()
        
        # Guardar click
        self.click_data[query].append((url, position, timestamp, dwell_time))
        
        # Calcular score del click basado en posición (clicks en posiciones bajas son más valiosos)
        position_bias = 1.0 / math.log2(position + 2)
        
        # Ajustar por dwell time
        quality_signal = 1.0
        if dwell_time:
            quality_signal = self.calculate_dwell_time_score(dwell_time)
        
        if bounced:
            quality_signal *= 0.3  # Penalizar bounces
        
        # Actualizar relevancia query-url
        current_relevance = self.query_url_relevance[query][url]
        new_signal = position_bias * quality_signal
        
        # Promedio móvil exponencial
        self.query_url_relevance[query][url] = (
            current_relevance * 0.8 + new_signal * 0.2
        )
        
        # Actualizar calidad general del URL
        self.url_quality[url] = (
            self.url_quality[url] * 0.9 + quality_signal * 0.1
        )
    
    def apply_temporal_decay(self):
        """Aplicar decaimiento temporal a los datos antiguos"""
        cutoff_date = datetime.now() - timedelta(days=30)
        
        for query in self.click_data:
            self.click_data[query] = [
                (url, pos, ts, dwell) 
                for url, pos, ts, dwell in self.click_data[query]
                if ts > cutoff_date
            ]
    
    def rank_results(self, query: str, results: list, user_context: dict = None) -> list:
        """
        Rankear resultados usando múltiples señales
        
        Args:
            query: Query de búsqueda
            results: Lista de resultados con scores base
            user_context: Contexto del usuario (ubicación, historial, etc.)
        """
        scored_results = []
        
        for result in results:
            url = result.get('url', '')
            base_score = result.get('score', 0.5)
            
            # 1. Score base del algoritmo de búsqueda (30%)
            final_score = base_score * 0.3
            
            # 2. Relevancia aprendida query-url (40%)
            learned_relevance = self.query_url_relevance[query].get(url, 0.5)
            final_score += learned_relevance * 0.4
            
            # 3. Calidad general del URL (20%)
            url_quality = self.url_quality.get(url, 0.5)
            final_score += url_quality * 0.2
            
            # 4. Frescura del contenido (10%)
            freshness = self.calculate_freshness(result.get('published_date'))
            final_score += freshness * 0.1
            
            # Ajustes por contexto de usuario
            if user_context:
                # Preferencia de idioma
                if user_context.get('language') == result.get('language'):
                    final_score *= 1.1
                
                # Preferencia de dominio
                if result.get('domain') in user_context.get('favorite_domains', []):
                    final_score *= 1.2
            
            result['final_score'] = final_score
            result['score_breakdown'] = {
                'base': base_score,
                'learned': learned_relevance,
                'quality': url_quality,
                'freshness': freshness
            }
            
            scored_results.append(result)
        
        # Ordenar por score final
        scored_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Aplicar diversidad (evitar que todos los resultados sean del mismo dominio)
        diversified = self.apply_diversity(scored_results)
        
        return diversified
    
    def calculate_freshness(self, published_date) -> float:
        """Calcular score de frescura basado en fecha de publicación"""
        if not published_date:
            return 0.5
        
        try:
            if isinstance(published_date, str):
                pub_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
            else:
                pub_date = published_date
            
            days_old = (datetime.now() - pub_date).days
            
            # Muy reciente (< 7 días): 1.0
            # Reciente (< 30 días): 0.8
            # Medio (< 90 días): 0.6
            # Viejo (< 180 días): 0.4
            # Muy viejo (> 180 días): 0.2
            
            if days_old < 7:
                return 1.0
            elif days_old < 30:
                return 0.8
            elif days_old < 90:
                return 0.6
            elif days_old < 180:
                return 0.4
            else:
                return 0.2
        except:
            return 0.5
    
    def apply_diversity(self, results: list, max_per_domain: int = 2) -> list:
        """Aplicar diversidad de dominios en los resultados"""
        domain_count = defaultdict(int)
        diversified = []
        deferred = []
        
        for result in results:
            domain = result.get('domain', '')
            
            if domain_count[domain] < max_per_domain:
                diversified.append(result)
                domain_count[domain] += 1
            else:
                deferred.append(result)
        
        # Agregar resultados diferidos al final
        diversified.extend(deferred)
        
        return diversified
    
    def get_query_suggestions(self, partial_query: str, limit: int = 10) -> list:
        """Obtener sugerencias de queries basadas en historial"""
        # Queries que comienzan con el texto parcial
        matching_queries = [
            q for q in self.click_data.keys()
            if q.lower().startswith(partial_query.lower())
        ]
        
        # Ordenar por popularidad (número de clicks)
        query_popularity = [
            (q, len(self.click_data[q]))
            for q in matching_queries
        ]
        query_popularity.sort(key=lambda x: x[1], reverse=True)
        
        return [q for q, _ in query_popularity[:limit]]
    
    def get_trending_queries(self, time_window_hours: int = 24, limit: int = 10) -> list:
        """Obtener queries trending en una ventana de tiempo"""
        cutoff = datetime.now() - timedelta(hours=time_window_hours)
        
        query_counts = defaultdict(int)
        
        for query, clicks in self.click_data.items():
            recent_clicks = [c for c in clicks if c[2] > cutoff]
            query_counts[query] = len(recent_clicks)
        
        # Ordenar por cantidad
        trending = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [q for q, count in trending[:limit]]
    
    def export_stats(self) -> dict:
        """Exportar estadísticas del ranking engine"""
        return {
            'total_queries': len(self.click_data),
            'total_clicks': sum(len(clicks) for clicks in self.click_data.values()),
            'total_urls': len(self.url_quality),
            'avg_query_length': np.mean([len(q.split()) for q in self.click_data.keys()]),
            'top_queries': self.get_trending_queries(limit=20),
            'top_urls': sorted(
                self.url_quality.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:20]
        }
    
    def save(self, filepath: str):
        """Guardar estado del ranking engine"""
        data = {
            'click_data': {
                query: [(url, pos, ts.isoformat(), dwell) 
                       for url, pos, ts, dwell in clicks]
                for query, clicks in self.click_data.items()
            },
            'url_quality': dict(self.url_quality),
            'query_url_relevance': {
                query: dict(urls)
                for query, urls in self.query_url_relevance.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Cargar estado del ranking engine"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Reconstruir click_data
            self.click_data = defaultdict(list)
            for query, clicks in data.get('click_data', {}).items():
                self.click_data[query] = [
                    (url, pos, datetime.fromisoformat(ts), dwell)
                    for url, pos, ts, dwell in clicks
                ]
            
            # Reconstruir url_quality
            self.url_quality = defaultdict(float, data.get('url_quality', {}))
            
            # Reconstruir query_url_relevance
            self.query_url_relevance = defaultdict(lambda: defaultdict(float))
            for query, urls in data.get('query_url_relevance', {}).items():
                self.query_url_relevance[query] = defaultdict(float, urls)
            
            return True
        except Exception as e:
            print(f"Error loading ranking data: {e}")
            return False


# Ejemplo de uso
if __name__ == "__main__":
    engine = RankingEngine()
    
    # Simular algunos clicks
    engine.record_click("python tutorial", "https://realpython.com", 1, dwell_time=120)
    engine.record_click("python tutorial", "https://w3schools.com", 2, dwell_time=45)
    engine.record_click("python tutorial", "https://example.com", 3, dwell_time=5, bounced=True)
    
    # Simular resultados de búsqueda
    results = [
        {'url': 'https://realpython.com', 'title': 'Real Python', 'score': 0.7},
        {'url': 'https://w3schools.com', 'title': 'W3Schools', 'score': 0.65},
        {'url': 'https://newsite.com', 'title': 'New Tutorial', 'score': 0.8},
    ]
    
    # Rankear
    ranked = engine.rank_results("python tutorial", results)
    
    print("Resultados rankeados:")
    for i, r in enumerate(ranked, 1):
        print(f"{i}. {r['title']} - Score: {r['final_score']:.3f}")
        print(f"   Breakdown: {r['score_breakdown']}")
