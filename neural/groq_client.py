#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UnifiedLLMClient - Cliente unificado para Groq y Ollama
Creado por: Jhonatan David Castro Galvis

Intenta Groq primero, luego Ollama como fallback.
Smart Mode SOLO se activa si ningún LLM está disponible.
Ollama puede tomarse el tiempo que necesite — calidad > velocidad.
"""

import json
import urllib.request
import urllib.error
import os
import sys


class GroqClient:
    """Cliente para Groq Cloud API"""
    
    def __init__(self):
        self.api_key = os.environ.get('GROQ_API_KEY', '')
        self.model = os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')
        self.base_url = 'https://api.groq.com/openai/v1'
        self.available = False
        self.check()
    
    def check(self):
        if not self.api_key:
            print("⚠️  GROQ_API_KEY no encontrada — obtén una gratis en https://console.groq.com", flush=True)
            return
        try:
            req = urllib.request.Request(
                f'{self.base_url}/models',
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    self.available = True
                    print(f"✓ Groq disponible — modelo: {self.model}", flush=True)
                else:
                    print(f"⚠️  Groq error: HTTP {r.status}", flush=True)
        except Exception as e:
            print(f"⚠️  Groq no disponible: {e}", flush=True)
    
    def chat(self, messages: list, temperature: float = 0.7, max_tokens: int = 600) -> str:
        if not self.available:
            return None
        payload = {
            'model': self.model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        try:
            req = urllib.request.Request(
                f'{self.base_url}/chat/completions',
                data=json.dumps(payload).encode('utf-8'),
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode('utf-8'))
                return data['choices'][0]['message']['content']
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            print(f"[Groq] HTTP Error {e.code}: {error_body}", flush=True)
            return None
        except Exception as e:
            print(f"[Groq] Error en chat: {e}", flush=True)
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
        self.base_url = os.environ.get('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')
        self.model = os.environ.get('OLLAMA_MODEL', 'llama3.2:1b')
        self.available = False
        self.check()
    
    def check(self):
        try:
            req = urllib.request.Request(self.base_url)
            with urllib.request.urlopen(req, timeout=3) as r:
                if r.status == 200:
                    self.available = True
                    print(f"✓ Ollama disponible — modelo: {self.model}", flush=True)
                else:
                    print(f"⚠️  Ollama error: HTTP {r.status}", flush=True)
        except Exception as e:
            print(f"⚠️  Ollama no disponible: {e}", flush=True)
    
    def chat(self, messages: list, temperature: float = 0.7, max_tokens: int = 600) -> str:
        if not self.available:
            return None
        
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        payload = {
            'model': self.model,
            'prompt': prompt,
            'stream': False,
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
                return data.get('response', '')
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            print(f"[Ollama] HTTP Error {e.code}: {error_body}", flush=True, file=sys.stderr)
            return None
        except Exception as e:
            print(f"[Ollama] Error en chat: {e}", flush=True, file=sys.stderr)
            return None
    
    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        if not self.available:
            return None
        
        payload = {
            'model': self.model,
            'prompt': prompt,
            'stream': False,
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
                return data.get('response', '')
        except Exception as e:
            print(f"[Ollama] Error en generate: {e}", flush=True, file=sys.stderr)
            return None


class UnifiedLLMClient:
    """
    Cliente unificado: Groq primero, Ollama como fallback.
    Smart Mode SOLO si ningún LLM está disponible.
    """
    
    def __init__(self):
        self.groq = None
        self.ollama = None
        self.active_client = None
        self.available = False
        self.model = "sin LLM"
        
        try:
            self.groq = GroqClient()
            if self.groq.available:
                self.active_client = self.groq
                self.available = True
                self.model = f"Groq/{self.groq.model}"
                print(f"✓ LLM activo: {self.model}", flush=True)
                return
        except Exception as e:
            print(f"⚠️  Groq falló: {e}", flush=True)
        
        try:
            self.ollama = OllamaClient()
            if self.ollama.available:
                self.active_client = self.ollama
                self.available = True
                self.model = f"Ollama/{self.ollama.model}"
                print(f"✓ LLM activo (fallback): {self.model}", flush=True)
                return
        except Exception as e:
            print(f"⚠️  Ollama también falló: {e}", flush=True)
        
        print("⚠️  Sin LLM disponible — NEXUS funcionará en Smart Mode", flush=True)
    
    def chat(self, messages: list, temperature: float = 0.7, max_tokens: int = 600) -> str:
        if self.active_client:
            return self.active_client.chat(messages, temperature, max_tokens)
        return None
    
    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        if self.active_client:
            return self.active_client.generate(prompt, temperature)
        return None


if __name__ == '__main__':
    print("=== Test de Groq Client ===")
    client = UnifiedLLMClient()
    if client.available:
        print(f"\n✓ Cliente activo: {client.model}\n")
        response = client.generate("Di 'Hola' en 5 idiomas diferentes")
        print(f"Generate test:\n{response}\n")
        messages = [
            {"role": "system", "content": "Eres un asistente conciso."},
            {"role": "user", "content": "¿Qué es una red neuronal en 20 palabras?"}
        ]
        response = client.chat(messages)
        print(f"Chat test:\n{response}\n")
    else:
        print("\n❌ No hay LLM disponible")
        print("Configura GROQ_API_KEY o instala Ollama")
