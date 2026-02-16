#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS AI - Neural Interface Wrapper
Recibe comandos vía stdin/stdout para comunicarse con Node.js
"""

import sys
import json
from brain import NeuralSearchBrain

def main():
    brain = NeuralSearchBrain()
    
    # Leer comando desde stdin
    try:
        line = sys.stdin.readline()
        if not line:
            sys.exit(0)
            
        data = json.loads(line.strip())
        action = data.get('action')
        
        if action == 'rank':
            # Rankear resultados con IA
            query = data.get('query', '')
            results = data.get('results', [])
            ranked = brain.rank_results(query, results)
            print(json.dumps(ranked))
            
        elif action == 'learn':
            # Aprender de un click
            query = data.get('query', '')
            url = data.get('url', '')
            brain.learn_from_click(query, url)
            print(json.dumps({'success': True}))
            
        elif action == 'suggest':
            # Generar sugerencias
            query = data.get('query', '')
            suggestions = brain.get_suggestions(query)
            print(json.dumps(suggestions))
            
        elif action == 'stats':
            # Estadísticas del cerebro
            stats = brain.get_stats()
            print(json.dumps(stats))
            
        else:
            print(json.dumps({'error': 'Unknown action'}))
            
    except json.JSONDecodeError as e:
        print(json.dumps({'error': f'JSON decode error: {str(e)}'}), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({'error': str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
