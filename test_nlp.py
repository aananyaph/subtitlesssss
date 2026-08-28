import sys
sys.path.insert(0, 'backend')
from backend.asr.nlp_processor import NLPPostProcessor

nlp = NLPPostProcessor(punctuation_enabled=True, hallucination_filter_enabled=True, duplicate_filter_enabled=True)

# Test 1: OpenVINO
result = nlp.process_final_transcript(text='OpenVINO optimizes inference on Intel CPUs', language='en', no_speech_prob=0.0)
output = result['text'] if not result['suppressed'] else f'[SUPPRESSED:{result["reason"]}]'
t1 = 'OpenVINO' in output
print(f'Test 1 - OpenVINO: "{output}" - OpenVINO preserved: {t1}')

# Test 2: javascript
result2 = nlp.process_final_transcript(text='i am learning javascript today', language='en', no_speech_prob=0.0)
output2 = result2['text'] if not result2['suppressed'] else f'[SUPPRESSED:{result2["reason"]}]'
t2 = 'JavaScript' in output2
print(f'Test 2 - javascript: "{output2}" - JavaScript preserved: {t2}')

# Test 3: cpu
result3 = nlp.process_final_transcript(text='the model runs on cpu', language='en', no_speech_prob=0.0)
output3 = result3['text'] if not result3['suppressed'] else f'[SUPPRESSED:{result3["reason"]}]'
t3 = 'CPU' in output3
print(f'Test 3 - cpu: "{output3}" - CPU preserved: {t3}')

all_pass = t1 and t2 and t3
print(f'\nAll tests pass: {all_pass}')
