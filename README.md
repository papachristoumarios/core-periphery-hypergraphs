# Supplementary code for "Core-periphery Models for Hypergraphs"

## Setup 

Install required packages with

```bash
pip install -r requirements.txt
```

Download [data](https://doi.org/10.5281/zenodo.5943043) from Zenodo and set the `DATA_ROOT` variable in `base.py` to point at the data.

The options for running the goodness-of-fit experiments can be found with

```bash
python goodness_of_fit.py --help
```

## Examples

```bash
python goodness_of_fit.py --name threads-math-sx-filtered --learnable_ranks --pipeline cigam -H 0.5,1 --order_max 2 --k_core 2
```

## Zenodo Links

 * [Datasets](https://doi.org/10.5281/zenodo.5943043)
 * [Source Code](https://doi.org/10.5281/zenodo.5965856)

## Citation

Please cite the paper, data and source code as 

```bibtex
@inproceedings{cigam_paper,
  title		= {Core-periphery Models for Hypergraphs},
  author	= {Papachristou, Marios and Kleinberg, Jon},
  booktitle	= {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery & Data Mining},
  year		= {2022}
}

@dataset{cigam_datasets,
  author       = {Papachristou, Marios and Kleinberg, Jon},
  title        = {Datasets - Core-periphery Models for Hypergraphs},
  month        = feb,
  year         = 2022,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.5943044},
  url          = {https://doi.org/10.5281/zenodo.5943044}
}

@software{cigam_source_code,
  author       = {Papachristou, Marios and Kleinberg, Jon},
  title        = {Code - Core-periphery Models for Hypergraphs},
  month        = feb,
  year         = 2022,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.5965856},
  url          = {https://doi.org/10.5281/zenodo.5965856}
}
```
