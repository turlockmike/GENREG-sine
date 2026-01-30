"""
Sine Test Configuration
Testing Emergent Hybrid Computation Theory

Architecture (matching original GENREG):
  x ─┬─→ Controller (MLP) ─→ prediction
     │
     └─→ Protein Cascade ─→ trust ─→ fitness modifier

MLP and Proteins work in PARALLEL.
Trust from proteins modifies fitness, creating selection pressure.
"""

# ================================================================
# TASK: Sine Wave Approximation
# ================================================================
# Input: x in [-2π, 2π]
# Output: sin(x)
# Fitness = -MSE * trust_multiplier

INPUT_SIZE = 1
OUTPUT_SIZE = 1
NUM_TEST_POINTS = 100  # Points sampled across [-2π, 2π]

# ================================================================
# MLP CONTROLLER
# ================================================================
# Input expansion with TRUE signals + NOISY competing signals
# Network must learn to filter noise and focus on true signal
# This tests if saturation helps "gate off" irrelevant inputs

INPUT_EXPANSION = True
TRUE_SIGNAL_SIZE = 16     # Fourier-like basis (the REAL useful signal)
NOISE_SIGNAL_SIZE = 240   # Competing noisy/misleading/conflicting signals
EXPANSION_SIZE = TRUE_SIGNAL_SIZE + NOISE_SIGNAL_SIZE  # Total: 256

HIDDEN_SIZE = 8           # Hidden layer neurons (MUCH fewer than inputs!)
# 256 inputs → 8 hidden = 32:1 compression ratio!
# Network MUST learn to filter noise - saturation should help "gate off" irrelevant inputs

# ================================================================
# PROTEIN CASCADE
# ================================================================
PROTEINS_ENABLED = True
PROTEIN_MUTATION_RATE = 0.15
PROTEIN_MUTATION_SCALE = 0.1

# Trust scaling (how much trust affects fitness)
PROTEIN_TRUST_SCALE = 0.1

# ================================================================
# SATURATION TRACKING
# ================================================================
SATURATION_THRESHOLD = 0.95
TRACK_ACTIVATIONS = True

# ================================================================
# POPULATION & EVOLUTION
# ================================================================
POPULATION_SIZE = 300
GENERATIONS = 5000

# Selection tiers (must sum to 1.0)
ELITE_PCT = 0.20
SURVIVE_PCT = 0.30
CLONE_MUTATE_PCT = 0.40
RANDOM_PCT = 0.10

# ================================================================
# MUTATION
# ================================================================
MUTATION_RATE = 0.10
MUTATION_SCALE = 0.1

# ================================================================
# LOGGING
# ================================================================
LOG_EVERY = 10
CHECKPOINT_EVERY = 5  # Save best genome every N generations

# ================================================================
# HARDWARE
# ================================================================
DEVICE = "cuda"
SEED = 42
