"""Compare rule-based vs Silero punctuation on a proper test set."""
import sys
sys.path.insert(0, 'backend')

from backend.asr.nlp_processor import NLPPostProcessor, SileroPunctuationEngine

# ========== SETUP ==========
# Initialize both engines
nlp = NLPPostProcessor(punctuation_enabled=True, hallucination_filter_enabled=True, duplicate_filter_enabled=True)

# Patch SileroPunctuationEngine to add restore_punctuation method
# (since the direct load has the name issue, we'll use the apply_te approach)
import torch
model, example_texts, languages, punct, apply_te = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_te',
    trust_repo=True
)

silero_engine = SileroPunctuationEngine()
silero_engine.model = model
silero_engine.apply_te = apply_te
silero_engine.punct = punct

# We'll use NLPPostProcessor's internal methods for rule-based
# Rule-based is the default in NLPPostProcessor.restore_punctuation

# Test sentences covering all the important cases
test_sentences = [
    # Proper nouns
    "i am learning javascript and python today",
    # Contractions
    "i dont know if youre coming to the meeting",
    # Questions
    "what time does the store close",
    # Multi-clause
    "i went to the store but they were closed so i came back home",
    # Numbers/technical terms
    "the model runs at zero point two two rtf on cpu",
    # Names
    "sonu said the meeting is at three pm",
    # Exclamation/emphasis
    "wow that is amazing i cant believe it",
    # Long run-on (realistic ASR output)
    "so today we are going to talk about how whisper works and why it matters for real time captioning",
    # Mixed case
    "openvino optimizes inference on intel cpus",
    # With extra spaces (common ASR artifact)
    "hello  world  this  has  extra  spaces",
]

# ========== RUN TESTS ==========
results = []

for raw_input in test_sentences:
    # === RULE-BASED ===
    res_rule = nlp.process_final_transcript(
        text=raw_input, language='en', no_speech_prob=0.0
    )
    rule_output = res_rule['text'] if not res_rule['suppressed'] else f"[SUPPRESSED:{res_rule['reason']}]"
    rule_latency = res_rule['nlp_latency_ms']
    
    # === SILERO ===
    try:
        silero_output = silero_engine.restore_punctuation(raw_input)
        silero_latency = 1.5  # approximate, ONNX inference is fast
    except Exception as e:
        silero_output = f"[ERROR: {e}]"
        silero_latency = 'N/A'
    
    # Determine winner
    # Simple heuristic: more punctuation marks = potentially better, but check capitalization too
    rule_punct = sum(1 for c in rule_output if c in '.!?')
    silero_punct = sum(1 for c in silero_output if c in '.!?') if isinstance(silero_output, str) else 0
    
    # Check proper noun capitalization
    rule_has_javascript = 'JavaScript' in rule_output
    silero_has_javascript = 'JavaScript' in silero_output
    
    # Choose winner
    if silero_punct > rule_punct and silero_has_javascript:
        winner = 'Silero'
    elif rule_punct >= silero_punct and rule_has_javascript:
        winner = 'Rule-based'
    else:
        winner = 'Wash'
    
    results.append({
        'input': raw_input,
        'rule': rule_output,
        'silero': silero_output,
        'rule_punct': rule_punct,
        'silero_punct': silero_punct,
        'rule_javascript': rule_has_javascript,
        'silero_javascript': silero_has_javascript,
        'winner': winner,
    })

# ========== PRINT RESULTS ==========
print("=" * 80)
print("SILERO vs RULE-BASED PUNCTUATION COMPARISON")
print("=" * 80)

silero_wins = 0
rule_wins = 0
washes = 0

for r in results:
    input_trunc = r['input'][:50] + '...' if len(r['input']) > 50 else r['input']
    print(f"\nINPUT:  \"{r['input']}\"")
    print(f"Rule:   \"{r['rule']}\"")
    print(f"Silero: \"{r['silero']}\"")
    print(f"Winner: {r['winner']}")
    
    if r['winner'] == 'Silero':
        silero_wins += 1
    elif r['winner'] == 'Rule-based':
        rule_wins += 1
    else:
        washes += 1

print(f"\n{'=' * 80}")
print(f"RESULT: Silero {silero_wins} / Rule-based {rule_wins} / Wash {washes}")
print(f"Decision: {'using Silero alone' if silero_wins > rule_wins else 'using Silero + capitalization patch' if silero_wins > 0 and rule_wins > 0 else 'keeping rule-based'}")
print("=" * 80)
EOF
python -X utf8 test_punctuation_comparison.py 2>&1