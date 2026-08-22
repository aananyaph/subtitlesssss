import sys
sys.path.insert(0, 'backend')
from backend.asr.nlp_processor import SileroPunctuationEngine

e = SileroPunctuationEngine()
print('restore_punctuation in dir:', 'restore_punctuation' in dir(e))
print('restore_punctuation in __dict__:', 'restore_punctuation' in SileroPunctuationEngine.__dict__)
if 'restore_punctuation' in SileroPunctuationEngine.__dict__:
    result = e.restore_punctuation('test input')
    print(f'Result: {result}')
" && python -X utf8 test_nlp.py 2>&1
