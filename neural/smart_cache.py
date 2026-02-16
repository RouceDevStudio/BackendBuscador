#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS AI - Smart Cache Module
Sistema de caché inteligente con TTL y priorización
"""

import json
import time
import hashlib
from pathlib import Path
from collections import OrderedDict
from typing import Any, Optional, Dict

class SmartCache:
    """Cache inteligente con LRU y TTL"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600, cache_dir: str = 'cache'):
        """
        Args:
            max_size: Número máximo de entradas en cache
            ttl: Tiempo de vida en segundos (default 1 hora)
            cache_dir: Directorio para persistencia
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache en memoria (LRU)
        self.memory_cache: OrderedDict[str, Dict] = OrderedDict()
        
        # Estadísticas
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired': 0
        }
        
        # Cargar cache persistente
        self.load_persistent_cache()
    
    def _generate_key(self, data: Any) -> str:
        """Generar clave hash para datos"""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def get(self, key: Any) -> Optional[Any]:
        """Obtener valor del cache"""
        cache_key = self._generate_key(key)
        
        # Buscar en memoria
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            
            # Verificar TTL
            if time.time() - entry['timestamp'] > self.ttl:
                # Expirado
                self.memory_cache.pop(cache_key)
                self.stats['expired'] += 1
                self.stats['misses'] += 1
                return None
            
            # Hit! Mover al final (LRU)
            self.memory_cache.move_to_end(cache_key)
            self.stats['hits'] += 1
            return entry['data']
        
        # Miss
        self.stats['misses'] += 1
        return None
    
    def set(self, key: Any, value: Any) -> None:
        """Guardar valor en cache"""
        cache_key = self._generate_key(key)
        
        # Si cache está lleno, eliminar el más antiguo (LRU)
        if len(self.memory_cache) >= self.max_size:
            oldest_key, _ = self.memory_cache.popitem(last=False)
            self.stats['evictions'] += 1
        
        # Guardar en memoria
        self.memory_cache[cache_key] = {
            'data': value,
            'timestamp': time.time()
        }
        
        # Mover al final
        self.memory_cache.move_to_end(cache_key)
    
    def clear(self) -> None:
        """Limpiar todo el cache"""
        self.memory_cache.clear()
        
        # Limpiar archivos persistentes
        for cache_file in self.cache_dir.glob('*.json'):
            cache_file.unlink()
    
    def cleanup_expired(self) -> int:
        """Limpiar entradas expiradas"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.memory_cache.items():
            if current_time - entry['timestamp'] > self.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.memory_cache.pop(key)
        
        count = len(expired_keys)
        self.stats['expired'] += count
        return count
    
    def get_stats(self) -> Dict:
        """Obtener estadísticas del cache"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'size': len(self.memory_cache),
            'max_size': self.max_size,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': round(hit_rate, 2),
            'evictions': self.stats['evictions'],
            'expired': self.stats['expired'],
            'ttl': self.ttl
        }
    
    def save_persistent_cache(self) -> None:
        """Guardar cache en disco (top 100 más usados)"""
        # Ordenar por timestamp (más reciente primero)
        sorted_items = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        
        # Guardar top 100
        persistent_data = {}
        for key, entry in sorted_items[:100]:
            persistent_data[key] = entry
        
        cache_file = self.cache_dir / 'persistent.json'
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(persistent_data, f)
        except Exception as e:
            print(f"⚠ Error guardando cache: {e}")
    
    def load_persistent_cache(self) -> None:
        """Cargar cache desde disco"""
        cache_file = self.cache_dir / 'persistent.json'
        
        if not cache_file.exists():
            return
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                persistent_data = json.load(f)
            
            # Cargar en memoria
            for key, entry in persistent_data.items():
                # Solo cargar si no ha expirado
                if time.time() - entry['timestamp'] <= self.ttl:
                    self.memory_cache[key] = entry
            
            print(f"✓ Cache cargado: {len(self.memory_cache)} entradas")
        except Exception as e:
            print(f"⚠ Error cargando cache: {e}")


class QueryCache:
    """Cache especializado para consultas de búsqueda"""
    
    def __init__(self):
        self.cache = SmartCache(max_size=500, ttl=1800)  # 30 minutos
    
    def get_cached_results(self, query: str, filters: Optional[str] = None) -> Optional[Dict]:
        """Obtener resultados cacheados"""
        cache_key = {'query': query.lower().strip(), 'filters': filters}
        return self.cache.get(cache_key)
    
    def cache_results(self, query: str, results: Dict, filters: Optional[str] = None) -> None:
        """Cachear resultados de búsqueda"""
        cache_key = {'query': query.lower().strip(), 'filters': filters}
        self.cache.set(cache_key, results)
    
    def invalidate_query(self, query: str) -> None:
        """Invalidar cache de una consulta específica"""
        # En una implementación real, necesitaríamos un índice inverso
        # Por ahora, simplemente limpiar todo
        pass
    
    def get_stats(self) -> Dict:
        """Estadísticas del cache de queries"""
        return self.cache.get_stats()


# ══════════════════════════════════════════════════════════════════════
# TESTING
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import random
    
    print("\n💾 SMART CACHE TEST\n" + "="*50)
    
    # Test 1: Basic operations
    print("\n📝 Test 1: Operaciones básicas")
    cache = SmartCache(max_size=5, ttl=10)
    
    # Set values
    for i in range(5):
        cache.set(f'key{i}', {'data': f'value{i}'})
    
    # Get values
    for i in range(5):
        result = cache.get(f'key{i}')
        print(f"   key{i}: {result}")
    
    stats = cache.get_stats()
    print(f"\n   📊 Stats: {stats['hits']} hits, {stats['misses']} misses")
    
    # Test 2: LRU eviction
    print("\n📝 Test 2: LRU Eviction")
    cache.set('key5', {'data': 'value5'})  # Debería eliminar key0
    
    result = cache.get('key0')
    print(f"   key0 (should be None): {result}")
    
    stats = cache.get_stats()
    print(f"   📊 Evictions: {stats['evictions']}")
    
    # Test 3: TTL expiration
    print("\n📝 Test 3: TTL Expiration")
    cache_ttl = SmartCache(max_size=10, ttl=2)  # 2 segundos TTL
    
    cache_ttl.set('temp', {'data': 'temporary'})
    print(f"   Inmediato: {cache_ttl.get('temp')}")
    
    print("   Esperando 3 segundos...")
    time.sleep(3)
    
    result = cache_ttl.get('temp')
    print(f"   Después de 3s (should be None): {result}")
    
    # Test 4: Query Cache
    print("\n📝 Test 4: Query Cache")
    qcache = QueryCache()
    
    # Cache some queries
    qcache.cache_results('python tutorial', {
        'results': ['result1', 'result2'],
        'count': 2
    })
    
    # Retrieve
    cached = qcache.get_cached_results('python tutorial')
    print(f"   Cached results: {cached}")
    
    # Miss
    missed = qcache.get_cached_results('javascript tutorial')
    print(f"   Missed query: {missed}")
    
    stats = qcache.get_stats()
    print(f"\n   📊 Query Cache Stats:")
    print(f"      Hit Rate: {stats['hit_rate']}%")
    print(f"      Size: {stats['size']}/{stats['max_size']}")
    
    # Test 5: Performance
    print("\n📝 Test 5: Performance Test")
    perf_cache = SmartCache(max_size=1000, ttl=3600)
    
    # Insert 1000 items
    start = time.time()
    for i in range(1000):
        perf_cache.set(f'perf_key_{i}', {'data': f'value_{i}' * 10})
    insert_time = time.time() - start
    
    # Random reads
    start = time.time()
    for _ in range(1000):
        perf_cache.get(f'perf_key_{random.randint(0, 999)}')
    read_time = time.time() - start
    
    print(f"   Insert 1000 items: {insert_time:.3f}s")
    print(f"   Read 1000 random: {read_time:.3f}s")
    
    stats = perf_cache.get_stats()
    print(f"   Hit Rate: {stats['hit_rate']:.2f}%")
    
    print("\n" + "="*50)
    print("✅ Cache Test Complete!\n")
