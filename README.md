# Improving Generalizability of Kolmogorov–Arnold Networks via Error-Correcting Output Codes

This is an official implementation of the following paper:
> Youngjoon Lee, Jinu Gong, and Joonhyuk Kang.
**[Improving Generalizability of Kolmogorov–Arnold Networks via Error-Correcting Output Codes](https://arxiv.org/abs/2505.05798)**  
_arXiv:2505.05798_.

## Docker Image
`docker pull pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel`

## Dataset
- Blood cell classification dataset ([A dataset of microscopic peripheral blood cell images for development of automatic recognition systems](https://www.sciencedirect.com/science/article/pii/S2352340920303681))

## Experiments

To run the 'Impact of ECOC' experiment:
`bash shell/exp1.sh`

To run the 'Impact of Hyperparameter Configuration' experiment:
`bash shell/exp2.sh`

To run the 'Ablation Study' experiment:
`bash shell/ablation.sh`

## Citation
If this codebase can help you, please cite our paper: 
```bibtex
@article{lee2025improvinggeneralizabilitykolmogorovarnoldnetworks,
  title={Improving Generalizability of Kolmogorov–Arnold Networks via Error-Correcting Output Codes},
  author={Youngjoon Lee and Jinu Gong and Joonhyuk Kang},
  journal={arXiv preprint arXiv:2505.05798},
  year={2025}
}
```

## References
This repository draws inspiration from:
- https://github.com/woodenchild95/FL-Simulator
- https://github.com/ZiyaoLi/fast-kan
- https://github.com/Blealtan/efficient-kan
- https://github.com/AthanasiosDelis/faster-kan