# Legacy Code

This folder contains the original GENREG repository code. It is kept for reference but is no longer actively used.

**For new experiments, use the `usen` package instead.**

## Files

| File | Description |
|------|-------------|
| `sine_train.py` | Original training script (GA-based) |
| `sine_population.py` | Population management with tiered evolution |
| `sine_genome.py` | Genome class with MLP + proteins |
| `sine_controller.py` | 2-layer MLP with input expansion |
| `sine_config.py` | Configuration parameters |
| `sine_sweep.py` | Experimental sweep runner |
| `sine_proteins.py` | Protein cascade (unused) |
| `sine_analyze.py` | Analysis/visualization |
| `sine_visualize.py` | Live training visualization |
| `sine_annealing.py` | Simulated annealing variant |
| `sine_hillclimb.py` | Hill climbing variant |
| `sine_cmaes.py` | CMA-ES variant |

## Migration

The key concepts from this code have been extracted into the `usen` package:

| Legacy | New Location |
|--------|--------------|
| `SineController` | `usen.SparseNet` |
| `SinePopulation.evolve()` | `usen.train_gsa()` |
| SA training | `usen.train_sa()` |

## Running Legacy Code

If you need to run the original experiments:

```bash
cd legacy
python sine_train.py
```

Note: The legacy code has dependencies on `sine_config.py` and other files in this folder.
