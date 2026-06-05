import sys, os
sys.path.insert(0, '.')
import requests
import time
import json
from collections import defaultdict

API_URL = os.getenv('API_URL', 'http://localhost:8000')

IRIS_SAMPLES = [
    [5.1, 3.5, 1.4, 0.2],  # setosa
    [4.9, 3.0, 1.4, 0.2],  # setosa
    [6.3, 3.3, 6.0, 2.5],  # virginica
    [5.8, 2.7, 5.1, 1.9],  # virginica
    [5.7, 2.8, 4.5, 1.3],  # versicolor
    [6.3, 2.5, 4.9, 1.5],  # versicolor
    [7.2, 3.6, 6.1, 2.5],  # virginica
    [5.0, 3.4, 1.5, 0.2],  # setosa
]

def run_ab_test(n_requests: int = 100):
    results = defaultdict(list)
    errors  = 0

    print(f'Running {n_requests} requests against {API_URL}...')

    for i in range(n_requests):
        sample  = IRIS_SAMPLES[i % len(IRIS_SAMPLES)]
        try:
            resp = requests.post(
                f'{API_URL}/predict',
                json={'features': sample},
                timeout=5
            )
            if resp.status_code == 200:
                data    = resp.json()
                version = data.get('served_by', 'unknown')
                results[version].append({
                    'prediction': data['prediction'],
                    'confidence': data['confidence'],
                })
            else:
                errors += 1
        except Exception as e:
            errors += 1
        time.sleep(0.05)

    print(f'\n--- A/B Test Results ({n_requests} requests) ---')
    total = sum(len(v) for v in results.values())
    for version, preds in results.items():
        count    = len(preds)
        pct      = count / total * 100 if total > 0 else 0
        avg_conf = sum(p['confidence'] for p in preds) / count if count > 0 else 0
        print(f'  {version:12s}: {count:4d} requests ({pct:.1f}%)  avg_confidence={avg_conf:.4f}')

    print(f'  errors      : {errors}')
    print(f'  total       : {total + errors}')
    return results

if __name__ == '__main__':
    health = requests.get(f'{API_URL}/health').json()
    print(f'API health: {health["status"]}')
    print(f'AB mode:    {health.get("ab_mode", False)}')
    print(f'Shadow mode:{health.get("shadow_mode", False)}')
    print()
    run_ab_test(n_requests=100)
