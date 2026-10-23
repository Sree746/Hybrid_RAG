from src.generation.generator import GroqGenerator

generator = GroqGenerator()

result = generator.generate(
    prompt=(
        "Determine whether the evidence supports the claim. "
        "Output exactly SUPPORTED or NOT_SUPPORTED.\n\n"
        "CLAIM: Encoder self-attention uses queries, keys, and "
        "values from the same source.\n\n"
        "EVIDENCE: In a self-attention layer, all keys, values, "
        "and queries come from the output of the previous "
        "encoder layer.\n\n"
        "VERDICT:"
    ),
    max_new_tokens=128
)

print("Final result:", repr(result))