# HiFi-SAN: Enhancing Speech Synthesis with Slicing Adversarial Networks

HiFi-SAN leverages **Slicing Adversarial Networks ([SAN](https://arxiv.org/pdf/2301.12811))** to improve the efficiency and fidelity of speech synthesis, building on the foundation of [HiFi-GAN](https://arxiv.org/pdf/2010.05646). This approach integrates SAN into the discriminator, drawing inspiration from the paper [BigVSAN](https://arxiv.org/pdf/2309.02836), based on the work of [BigVGAN](https://arxiv.org/pdf/2206.04658). The result is enhanced adversarial training stability and superior performance..

## Key Changes

1. Integration of **SANConv2d** layers into the discriminator.
2. Refactored `DiscriminatorP_SAN` to employ SAN, allowing improved parameter scaling and normalization during training.
3. Updated `MultiPeriodDiscriminator` to include both SAN-based and traditional GAN-based discriminators for robust adversarial training.

## Acknowledgements

This work builds on the foundations of previous projects:

- [WaveGlow](https://github.com/NVIDIA/waveglow)
- [MelGAN](https://github.com/descriptinc/melgan-neurips)
- [Tacotron2](https://github.com/NVIDIA/tacotron2)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)
- [BigVSAN](https://github.com/sony/bigvsan)
