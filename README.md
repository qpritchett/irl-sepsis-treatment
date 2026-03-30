# IRL for Sepsis Treatment

Inverse Reinforcement Learning to recover reward functions from clinician treatment decisions for ICU sepsis patients, using MIMIC-III data.


## Install

Requires Python 3.11+.

```bash
uv sync        # or: pip install -e .
```

### Data

Data comes from MIMIC-III via the [CMU AI-Clinician-MIMICIV](https://github.com/cmudig/AI-Clinician-MIMICIV) pipeline. Place the pipeline output in `data/raw/`
