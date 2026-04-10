<div align="center">
    <h1>
    SVOR (<b>S</b>table <b>V</b>ideo <b>O</b>bject <b>R</b>emoval)
    </h1>
    <p>
    Official PyTorch code for <em>From Ideal to Real: Stable Video Object Removal under Imperfect Conditions</em><br>    </p>
    </p>
    <a href="https://arxiv.org/abs/2603.09283"><img src="https://img.shields.io/badge/arXiv-2603.09283-b31b1b" alt="version"></a>
    <a href="https://xiaomi-research.github.io/svor" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
    </a>
    <a href='https://huggingface.co/HigherHu/SVOR'>
        <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow'>
    </a>
    <a href='https://huggingface.co/datasets/HigherHu/RORD-50'>
        <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-RORD--50-orange'>
    </a>
    <!-- <a href="https://huggingface.co/spaces/xiaomi/SVOR" target='_blank'>
        <img src="https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue">
    </a> -->
    <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="mit"></a>


⭐ If SVOR is helpful to your projects, please help star this repo. Thanks! 🤗

</div>


## News

* **`Apr. 10th, 2026`:** The [Video Removal Skill](https://clawhub.ai/wangfei1204/mi-visionforge-svor) is now live on ClawHub! Powered by SVOR (an internally updated version) and [MiMo-V2-Omni](https://mimo.xiaomi.com/mimo-v2-omni), it removes objects from your videos using just a text prompt — **no mask required**. Pro tip: Pair it with [MiMo-V2-Pro](https://mimo.xiaomi.com/mimo-v2-pro) for the ultimate experience!🎉

* **`Apr. 10th, 2026`:** Github repository and [project page](https://xiaomi-research.github.io/svor) is now available. 🎉
* **`Mar. 10th, 2026`:** We released our paper on [Arxiv](https://arxiv.org/abs/2603.09283).


## Updates

- [ ] Release Inference Code and Pretrained Models
- [x] Release Skill, use this SVOR_API_KEY: `sk-mipixgen-test`
- [x] Release Github repository and Project Page
- [x] Release Paper


## Overview

![overall_structure](asset/framework.png)

Removing objects from videos remains difficult in the presence of real-world imperfections such as shadows, abrupt motion, and defective masks. Existing diffusion-based video inpainting models often struggle to maintain temporal stability and visual consistency under these challenges. We propose **Stable Video Object Removal (SVOR)**, a robust framework that achieves shadow-free, flicker-free, and mask-defect-tolerant removal through three key designs: (1) **Mask Union for Stable Erasure (MUSE)**, a windowed union strategy applied during temporal mask downsampling to preserve all target regions observed within each window, effectively handling abrupt motion and reducing missed removals; (2) **Denoising-Aware Segmentation (DA-Seg)**, a lightweight segmentation head on a decoupled side branch equipped with {Denoising-Aware AdaLN } and trained with mask degradation to provide an internal diffusion-aware localization prior without affecting content generation; and (3) **Curriculum Two-Stage Training**: where Stage I performs self-supervised pretraining on unpaired real-background videos with online random masks to learn realistic background and temporal priors, and Stage II refines on synthetic pairs using mask degradation and side-effect-weighted losses, jointly removing objects and their associated shadows/reflections while improving cross-domain robustness. Extensive experiments show that SVOR attains new state-of-the-art results across multiple datasets and degraded-mask benchmarks, advancing video object removal from ideal settings toward real-world applications.

## Results

For more visual results, go checkout our <a href="https://xiaomi-research.github.io/svor/" target="_blank">project page</a>

<h3>Common Masks</h3>
<table>
  <thead>
    <tr>
      <th>Masked Input</th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <!-- <video width="480" autoplay loop muted playsinline controls> <source src="asset/examples/input/bmx-bumps.mp4" type="video/mp4"> Your browser does not support the video tag. </video> -->
        <img src="asset/examples/input/bmx-bumps.gif" width="100%">
      </td>
	  <td>
        <!-- <video width="480" autoplay loop muted playsinline controls> <source src="asset/examples/result/bmx-bumps.mp4" type="video/mp4"> Your browser does not support the video tag. </video> -->
        <img src="asset/examples/result/bmx-bumps.gif" width="100%">
      </td>
    </tr>
    <tr>
      <td>
        <!-- <video width="480" autoplay loop muted playsinline controls> <source src="asset/examples/input/boat.mp4" type="video/mp4"> Your browser does not support the video tag. </video> -->
        <img src="asset/examples/input/boat.gif" width="100%">
      </td>
      <td>
        <!-- <video width="480" autoplay loop muted playsinline controls> <source src="asset/examples/result/boat.mp4" type="video/mp4"> Your browser does not support the video tag. </video> -->
        <img src="asset/examples/result/boat.gif" width="100%">
      </td>
    </tr>
    <tr>
      <td>
        <!-- <video width="480" autoplay loop muted playsinline controls> <source src="asset/examples/input/bus.mp4" type="video/mp4"> Your browser does not support the video tag. </video> -->
        <img src="asset/examples/input/bus.gif" width="100%">
      </td>
      <td>
        <!-- <video width="480" autoplay loop muted playsinline controls> <source src="asset/examples/result/bus.mp4" type="video/mp4"> Your browser does not support the video tag. </video> -->
        <img src="asset/examples/result/bus.gif" width="100%">
      </td>
    </tr>
    <tr>
      <td>
        <!-- <video width="480" autoplay loop muted playsinline controls> <source src="asset/examples/input/varanus-cage.mp4" type="video/mp4"> Your browser does not support the video tag. </video> -->
        <img src="asset/examples/input/varanus-cage.gif" width="100%">
      </td>
      <td>
        <!-- <video width="480" autoplay loop muted playsinline controls> <source src="asset/examples/result/varanus-cage.mp4" type="video/mp4"> Your browser does not support the video tag. </video> -->
        <img src="asset/examples/result/varanus-cage.gif" width="100%">
      </td>
    </tr>
    <tr>
  </tbody>
</table>

<h3>Defective Masks</h3>
<table>
  <thead>
    <tr>
      <th>Masked Input</th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <!-- <video width="480" autoplay loop muted playsinline controls> <source src="asset/examples/input_maskdrop0.5/camel.mp4" type="video/mp4"> Your browser does not support the video tag. </video> -->
        <img src="asset/examples/input_maskdrop0.5/camel.gif" width="100%">
      </td>
	  <td>
        <!-- <video width="480" autoplay loop muted playsinline controls> <source src="asset/examples/result_maskdrop0.5/camel.mp4" type="video/mp4"> Your browser does not support the video tag. </video> -->
        <img src="asset/examples/result_maskdrop0.5/camel.gif" width="100%">
      </td>
    </tr>
    <tr>
      <td>
        <!-- <video width="480" autoplay loop muted playsinline controls> <source src="asset/examples/input_maskdrop0.5/dog-gooses.mp4" type="video/mp4"> Your browser does not support the video tag. </video> -->
        <img src="asset/examples/input_maskdrop0.5/dog-gooses.gif" width="100%">
      </td>
      <td>
        <!-- <video width="480" autoplay loop muted playsinline controls> <source src="asset/examples/result_maskdrop0.5/dog-gooses.mp4" type="video/mp4"> Your browser does not support the video tag. </video> -->
        <img src="asset/examples/result_maskdrop0.5/dog-gooses.gif" width="100%">
      </td>
    </tr>
    <tr>
      <td>
        <!-- <video width="480" autoplay loop muted playsinline controls> <source src="asset/examples/input_maskdrop0.5/elephant.mp4" type="video/mp4"> Your browser does not support the video tag. </video>  -->
        <img src="asset/examples/input_maskdrop0.5/elephant.gif" width="100%">
      </td>
      <td>
        <!-- <video width="480" autoplay loop muted playsinline controls> <source src="asset/examples/result_maskdrop0.5/elephant.mp4" type="video/mp4"> Your browser does not support the video tag. </video>  -->
        <img src="asset/examples/result_maskdrop0.5/elephant.gif" width="100%">
      </td>
    </tr>
    <tr>
      <td>
        <!-- <video width="480" autoplay loop muted playsinline controls> <source src="asset/examples/input_maskdrop0.5/kite-walk.mp4" type="video/mp4"> Your browser does not support the video tag. </video>  -->
        <img src="asset/examples/input_maskdrop0.5/kite-walk.gif" width="100%">
      </td>
      <td>
        <!-- <video width="480" autoplay loop muted playsinline controls> <source src="asset/examples/result_maskdrop0.5/kite-walk.mp4" type="video/mp4"> Your browser does not support the video tag. </video>  -->
        <img src="asset/examples/result_maskdrop0.5/kite-walk.gif" width="100%">
      </td>
    </tr>
    <tr>
  </tbody>
</table>



## Acknowledgement

Our work benefit from the following open-source projects:

- [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)
- [VACE](https://github.com/ali-vilab/VACE)
- [ROSE](https://github.com/Kunbyte-AI/ROSE)
- [SAM2 - Segment Anything Model 2](https://github.com/facebookresearch/sam2)
- [RORD](https://github.com/Forty-lock/RORD)

## Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
@article{hu2026svor,
   title={From Ideal to Real: Stable Video Object Removal under Imperfect Conditions},
   author={Hu, Jiagao and Chen, Yuxuan and Li, Fuhao and Wang, Zepeng and Wang, Fei and Zhou, Daiguo and Luan, Jian},
   journal={arXiv preprint arXiv:2603.09283},
   year={2026}
}
```
