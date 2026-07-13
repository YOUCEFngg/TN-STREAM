# TNStreamCUSUM — Accelerated Drift-Aware Stream Clustering

**Python implementation of TNStream** (Zeng et al., 2025) enhanced with a custom **AbruptCUSUM** drift detector that doubles reaction speed without sacrificing cluster quality.

---

## Overview

This repository implements the TNStream algorithm from [*TNStream: Temporal Nearest Neighbour Stream Clustering*](https://arxiv.org/abs/2505.00359) (arXiv:2505.00359), merged with a **modified AbruptCUSUM detector** specifically tuned for **abrupt drift detection only**.

The core innovation: when abrupt drift is confirmed, TNStream reacts **2× faster** while maintaining identical cluster quality — achieved through temporary structural relaxation rather than destructive window manipulation.

---

## Key Innovation: Structural-Adaptation v4

Unlike prior approaches that reset state (v1), compress windows uniformly (v2), or selectively compress (v3), this implementation uses **parameter relaxation**:

| Mechanism | Effect |
|-----------|--------|
| **Surprise normalisation** | Distance-to-MC-radius ratio — `0` during warmup, `>1` outside clusters |
| **Macro-stability gate** | CUSUM disabled until first macro cluster forms — eliminates cold-start false alarms |
| **`n_micro` relaxation** | Drops to `1` post-drift so single new-regime micro-clusters anchor macros immediately |
| **Pool sensitivity boost** | Crystallisation threshold drops to `N_fast` — new points become MCs in half the time |
| **Self-healing expiry** | Old-regime MCs age out via accelerated window; new-regime MCs protected automatically |

No window surgery, no forced cleanup — existing structure remains stable while new structure forms faster.

---

## Performance

- **Detection lag**: ~2 steps saved per drift vs. standard CUSUM (`confirm=2` vs. `3`)
- **Reaction speed**: 2× faster label recovery post-drift
- **Quality preservation**: Identical macro-cluster stability during stable phases

---

## Quick Start

```bash
# Setup
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt

# Run benchmarks
python benchmark.py          # Full benchmark suite
python benchmark.py --quick  # Quick evaluation mode
```

---

## Authors

- **Abid Zakaria**
- **Negad Youcef**

Supervised by **Nait Bahloul**

---

## License

MIT License — see the [LICENSE](LICENSE) file for details.
