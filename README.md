# Sparse Cosine Optimized Policy Evolution (SCOPE) for Hexapod Gait Evolution

[Jim O'Connor](https://oconnor.digital.conncoll.edu) | [Jay B. Nash](https://www.linkedin.com/in/jaybnash/) | [Derin Gezgin](https://deringezgin.github.io) | [Gary B. Parker](https://oak.conncoll.edu/parker/)

*Published in IJCCI Conference on Evolutionary Computation and Theory and Applications, 2025*

## Overview

This repository contains the full source code and assets used for the paper *SCOPE: Spectral Compression for Online Policy Evolution of Hexapod Gaits*.

It implements a **steady–state genetic algorithm** (SSGA) that evolves a discrete cosine transform-based controller (SCOPE) directly inside the physics simulator **Webots** to obtain efficient and statically stable walking gaits for a realistic hexapod, *Mantis*.

All code is written in Python and uses the native Webots *Robot* / *Supervisor* APIs.  The workflow runs entirely inside a single Webots simulation – no external training loop is required.

---

## Repository structure

```text
SCOPE-for-Hexapod-Gait-Generation/
├── hexapod_simulation/
│   ├── worlds/
│   │   └── mantis.wbt
│   ├── controllers/
│   │   ├── config.txt
│   │   ├── SCOPE.py
│   │   ├── mantis/
│   │   │   ├── mantis.py
│   │   │   └── RobotControl.py
│   │   └── supervisor/
│   │       ├── SimulationControl.py
│   │       ├── SteadyStateGA.py
│   │       └── supervisor.py
├── plotting/
│   ├── complete_plotter.py
│   └── data_preprocess.py
├── test_final_individual/
│   ├── worlds/
│   │   └── mantis.wbt
│   └── controllers/
│       ├── bestIndv.npy
│       └── ...
└──
```

---

## Quick start

### 1. Clone the repository
```bash
git clone https://github.com/<your-fork>/SCOPE-for-Hexapod-Gait-Generation.git
cd SCOPE-for-Hexapod-Gait-Generation
```

### 2. Train a controller (GA + SCOPE)
1. Launch Webots with the training world:
   ```bash
   webots hexapod_simulation/worlds/mantis.wbt
   ```
2. Press **Play** in the Webots GUI.  
   The supervisor immediately starts the steady–state GA defined in
   `hexapod_simulation/controllers/supervisor/supervisor.py`.
3. Runtime artefacts are written to `DATA/<timestamp>/`, e.g.
   * `*.csv` – fitness & stability metrics per evaluation
   * `BEST_INDV/*.npy` – numpy arrays storing the best genome after each generation

> • **Stopping / resuming** – simply save a desired `*_BestIndv.npy` file and replay it with the test project (see below).  
> • **Auto-restart** – the GA automatically reloads the world every `AUTO_RESTART` generations (see `config.txt`).

### 3. Replay the final individual (no training)
```bash
webots test_final_individual/worlds/mantis.wbt
```
The test world loads `test_final_individual/controllers/supervisor/bestIndv.npy` and lets the robot walk indefinitely for qualitative inspection.

---

## Configuration

All experiment parameters are collected in
`hexapod_simulation/controllers/config.txt` (JSON). Frequently changed keys:

| Key | Meaning | Default |
|-----|---------|---------|
| `MODEL_TYPE` | `"scope"` (DCT-compressed input) or `"ssga"` (raw input) | `"scope"` |
| `POP_SIZE` | Population size of GA | `100` |
| `MAX_GENERATIONS` | Hard cap on generations (set high for continuous) | `1_000_000` |
| `TRIAL_STEPS` | Simulation steps per evaluation | `470` |
| `LIMIT_ROM_DEGREES` | Joint limit per control step | `5` |
| `F` | Sine frequency for leg trajectory helper | `0.5` |

After editing the file, simply re-launch the world, no recompilation is required.

## Citation

Please use the following citation when citing this work:

```bibtex
@article{o2025scope,
  title={SCOPE for Hexapod Gait Generation},
  author={O'Connor, Jim and Nash, Jay B and Gezgin, Derin and Parker, Gary B},
  journal={arXiv preprint arXiv:2507.13539},
  year={2025}
}
```
