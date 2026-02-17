#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UnifiedLLMClient - Cliente unificado para Groq y Ollama
Creado por: Jhonatan David Castro Galvis

Lógica de disponibilidad dinámica:
  - Ambos clientes (Groq + Ollama) se inicializan siempre
  - En cada llamada: intenta el preferido primero, luego el otro
  - Si ninguno responde → retorna None → brain usa Smart Mode
  - Re-chequea disponibilidad cada N llamadas para auto-recuperarse
  - Smart Mode SOLO cuando ningún LLM responde en esa llamada
"""

import json
import urllib.request
import urllib.error
import os
import sys


class GroqClient:
    """Cliente para Groq Cloud API"""
    
    def __init__(self):
        self.api_key  = os.environ.get('GROQ_API_KEY', '')
        self.model    = os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')
        self.base_url = 'https://api.groq.com/openai/v1'
        self.available = False
        self._fail_count = 0        # fallos consecutivos en runtime
        self._MAX_FAILS  = 3        # tras 3 fallos seguidos, marca como no disponible
        self.check()
    
    def check(self):
        """
        Verifica si Groq está disponible haciendo un mini chat real.
        NO usa /v1/models porque ese endpoint da 403 en cuentas gratuitas.
        Usa /v1/chat/completions con max_tokens=1 para ser lo más ligero posible.
        """
        if not self.api_key:
            print("⚠️  GROQ_API_KEY no encontrada — obtén una gratis en https://console.groq.com", flush=True)
            self.available = False
            return False
        try:
            payload = json.dumps({
                'model':      self.model,
                'messages':   [{'role': 'user', 'content': 'hi'}],
                'max_tokens': 1
            }).encode('utf-8')
            req = urllib.request.Request(
                f'{self.base_url}/chat/completions',
                data=payload,
                headers={
                    'Authorization':  f'Bearer {self.api_key}',
                    'Content-Type':   'application/json',
                    'User-Agent':     'groq-python/0.11.0',
                    'X-Stainless-OS': 'Linux',
                    'Accept':         'application/json',
                }
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                if r.status == 200:
                    self.available   = True
                    self._fail_count = 0
                    print(f"✓ Groq disponible — modelo: {self.model}", flush=True)
                    return True
                else:
                    print(f"⚠️  Groq error: HTTP {r.status}", flush=True)
                    self.available = False
                    return False
        except urllib.error.HTTPError as e:
            body = ''
            try: body = e.read().decode('utf-8')[:120]
            except: pass
            print(f"⚠️  Groq no disponible: HTTP {e.code} — {body}", flush=True)
            self.available = False
            return False
        except Exception as e:
            print(f"⚠️  Groq no disponible: {e}", flush=True)
            self.available = False
            return False
    
    def chat(self, messages: list, temperature: float = 0.7, max_tokens: int = 600) -> str:
        if not self.available:
            return None
        payload = {
            'model':       self.model,
            'messages':    messages,
            'temperature': temperature,
            'max_tokens':  max_tokens
        }
        try:
            req = urllib.request.Request(
                f'{self.base_url}/chat/completions',
                data=json.dumps(payload).encode('utf-8'),
                headers={
                    'Authorization':  f'Bearer {self.api_key}',
                    'Content-Type':   'application/json',
                    'User-Agent':     'groq-python/0.11.0',
                    'X-Stainless-OS': 'Linux',
                    'Accept':         'application/json',
                }
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode('utf-8'))
                self._fail_count = 0   # reset en éxito
                return data['choices'][0]['message']['content']
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            print(f"[Groq] HTTP Error {e.code}: {error_body[:200]}", flush=True)
            self._fail_count += 1
            if self._fail_count >= self._MAX_FAILS:
                print(f"[Groq] {self._MAX_FAILS} fallos consecutivos — marcando no disponible temporalmente", flush=True)
                self.available = False
            return None
        except Exception as e:
            print(f"[Groq] Error en chat: {e}", flush=True)
            self._fail_count += 1
            if self._fail_count >= self._MAX_FAILS:
                self.available = False
            return None
    
    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature)


class OllamaClient:
    """
    Cliente para Ollama local.
    Sin límite artificial de tiempo — Ollama puede tardar lo que necesite.
    """
    
    def __init__(self):
        self.base_url  = os.environ.get('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')
        self.model     = os.environ.get('OLLAMA_MODEL', 'llama3.2:1b')
        self.available = False
        self._fail_count = 0
        self._MAX_FAILS  = 3
        self.check()
    
    def check(self):
        """Verifica si Ollama está disponible. Se puede llamar en cualquier momento."""
        try:
            req = urllib.request.Request(self.base_url)
            with urllib.request.urlopen(req, timeout=3) as r:
                if r.status == 200:
                    self.available   = True
                    self._fail_count = 0
                    print(f"✓ Ollama disponible — modelo: {self.model}", flush=True)
                    return True
                else:
                    print(f"⚠️  Ollama error: HTTP {r.status}", flush=True)
                    self.available = False
                    return False
        except Exception as e:
            print(f"⚠️  Ollama no disponible: {e}", flush=True)
            self.available = False
            return False
    
    def chat(self, messages: list, temperature: float = 0.7, max_tokens: int = 600) -> str:
        if not self.available:
            return None
        
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        payload = {
            'model':   self.model,
            'prompt':  prompt,
            'stream':  False,
            'options': {
                'temperature': temperature,
                'num_predict': max_tokens
            }
        }
        
        try:
            req = urllib.request.Request(
                f'{self.base_url}/api/generate',
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            # Sin timeout — Ollama puede tardar lo que necesite
            with urllib.request.urlopen(req) as r:
                data = json.loads(r.read().decode('utf-8'))
                self._fail_count = 0
                return data.get('response', '')
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            print(f"[Ollama] HTTP Error {e.code}: {error_body[:200]}", flush=True, file=sys.stderr)
            self._fail_count += 1
            if self._fail_count >= self._MAX_FAILS:
                self.available = False
            return None
        except Exception as e:
            print(f"[Ollama] Error en chat: {e}", flush=True, file=sys.stderr)
            self._fail_count += 1
            if self._fail_count >= self._MAX_FAILS:
                self.available = False
            return None
    
    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        if not self.available:
            return None
        
        payload = {
            'model':   self.model,
            'prompt':  prompt,
            'stream':  False,
            'options': {'temperature': temperature}
        }
        
        try:
            req = urllib.request.Request(
                f'{self.base_url}/api/generate',
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req) as r:
                data = json.loads(r.read().decode('utf-8'))
                self._fail_count = 0
                return data.get('response', '')
        except Exception as e:
            print(f"[Ollama] Error en generate: {e}", flush=True, file=sys.stderr)
            self._fail_count += 1
            if self._fail_count >= self._MAX_FAILS:
                self.available = False
            return None


class UnifiedLLMClient:
    """
    Cliente unificado con disponibilidad DINÁMICA.

    Ambos clientes se inicializan siempre.
    En cada llamada:
      1. Intenta el preferido (por defecto Groq, configurable con LLM_PREFER=ollama)
      2. Si falla o no está disponible → intenta el otro automáticamente
      3. Si ninguno responde → retorna None → brain activa Smart Mode para esa llamada

    Auto-recuperación: cada RECHECK_EVERY llamadas vuelve a chequear disponibilidad,
    por si Groq recuperó créditos o Ollama fue levantado.
    """
    
    RECHECK_EVERY = 20   # re-chequear disponibilidad cada 20 llamadas
    
    def __init__(self):
        self.groq   = None
        self.ollama = None
        self._call_count = 0

        # Inicializar AMBOS siempre
        try:
            self.groq = GroqClient()
        except Exception as e:
            print(f"⚠️  No se pudo instanciar GroqClient: {e}", flush=True)

        try:
            self.ollama = OllamaClient()
        except Exception as e:
            print(f"⚠️  No se pudo instanciar OllamaClient: {e}", flush=True)

        # Orden de preferencia (configurable)
        prefer = os.environ.get('LLM_PREFER', 'groq').lower()
        if prefer == 'ollama':
            self._order = [self.ollama, self.groq]
            self._names = ['Ollama', 'Groq']
        else:
            self._order = [self.groq, self.ollama]
            self._names = ['Groq', 'Ollama']

        self._log_status()
    
    def _log_status(self):
        groq_ok   = self.groq   and self.groq.available
        ollama_ok = self.ollama and self.ollama.available
        
        if groq_ok and ollama_ok:
            print(f"✓ LLM: Groq ✅ + Ollama ✅ (ambos disponibles, usando {self._names[0]} primero)", flush=True)
        elif groq_ok:
            print(f"✓ LLM activo: Groq/{self.groq.model}", flush=True)
        elif ollama_ok:
            print(f"✓ LLM activo: Ollama/{self.ollama.model}", flush=True)
        else:
            print("⚠️  Sin LLM disponible — NEXUS funcionará en Smart Mode", flush=True)
    
    @property
    def available(self) -> bool:
        """True si AL MENOS UNO está disponible."""
        groq_ok   = bool(self.groq   and self.groq.available)
        ollama_ok = bool(self.ollama and self.ollama.available)
        return groq_ok or ollama_ok
    
    @property
    def model(self) -> str:
        """Nombre del modelo activo (el preferido que esté disponible)."""
        for client, name in zip(self._order, self._names):
            if client and client.available:
                m = getattr(client, 'model', '?')
                return f"{name}/{m}"
        return "sin LLM (Smart Mode)"

    def _maybe_recheck(self):
        """Cada RECHECK_EVERY llamadas, vuelve a verificar disponibilidad."""
        self._call_count += 1
        if self._call_count % self.RECHECK_EVERY == 0:
            changed = False
            if self.groq:
                was = self.groq.available
                self.groq.check()
                if self.groq.available != was:
                    changed = True
            if self.ollama:
                was = self.ollama.available
                self.ollama.check()
                if self.ollama.available != was:
                    changed = True
            if changed:
                print("[LLM] Estado de disponibilidad cambió:", flush=True)
                self._log_status()

    def _try_in_order(self, method: str, *args, **kwargs) -> str:
        """
        Llama al método (chat/generate) en el orden de preferencia.
        Si el primero falla o no está disponible, prueba el segundo.
        """
        self._maybe_recheck()

        for client, name in zip(self._order, self._names):
            if not client or not client.available:
                continue
            try:
                result = getattr(client, method)(*args, **kwargs)
                if result is not None:
                    return result
                # result == None significa que falló en runtime
                print(f"[LLM] {name} retornó None → probando siguiente...", flush=True)
            except Exception as e:
                print(f"[LLM] {name} excepción inesperada: {e} → probando siguiente...", flush=True)

        # Ninguno funcionó
        return None

    def chat(self, messages: list, temperature: float = 0.7, max_tokens: int = 600) -> str:
        return self._try_in_order('chat', messages, temperature, max_tokens)
    
    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        return self._try_in_order('generate', prompt, temperature)


if __name__ == '__main__':
    print("=== Test UnifiedLLMClient (disponibilidad dinámica) ===")
    client = UnifiedLLMClient()
    print(f"\n✓ Disponible: {client.available}")
    print(f"✓ Modelo activo: {client.model}\n")

    if client.available:
        response = client.generate("Di 'Hola' en 5 idiomas diferentes")
        print(f"Generate test:\n{response}\n")
        messages = [
            {"role": "system",  "content": "Eres un asistente conciso."},
            {"role": "user",    "content": "¿Qué es una red neuronal en 20 palabras?"}
        ]
        response = client.chat(messages)
        print(f"Chat test:\n{response}\n")
    else:
        print("❌ No hay LLM disponible — Smart Mode activo")
        print("Configura GROQ_API_KEY o levanta Ollama")
