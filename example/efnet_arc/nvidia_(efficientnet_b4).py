EfficientNet(
  (stem): Sequential(
    (conv): Conv2d(3, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(48, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
    (activation): SiLU(inplace=True)
  )
  (layers): Sequential(
    (0): Sequential(
      (block0): MBConvBlock(
        (depsep): Sequential(
          (conv): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
          (bn): BatchNorm2d(48, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=48, out_features=12, bias=True)
          (expand): Linear(in_features=12, out_features=48, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(48, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(24, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_quantizer): Identity()
      )
      (block1): MBConvBlock(
        (depsep): Sequential(
          (conv): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
          (bn): BatchNorm2d(24, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=24, out_features=6, bias=True)
          (expand): Linear(in_features=6, out_features=24, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(24, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
    )
    (1): Sequential(
      (block0): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(144, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(144, 144, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=144, bias=False)
          (bn): BatchNorm2d(144, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=144, out_features=6, bias=True)
          (expand): Linear(in_features=6, out_features=144, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(32, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_quantizer): Identity()
      )
      (block1): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=192, out_features=8, bias=True)
          (expand): Linear(in_features=8, out_features=192, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(32, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
      (block2): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=192, out_features=8, bias=True)
          (expand): Linear(in_features=8, out_features=192, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(32, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
      (block3): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=192, out_features=8, bias=True)
          (expand): Linear(in_features=8, out_features=192, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(32, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
    )
    (2): Sequential(
      (block0): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(192, 192, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=192, bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=192, out_features=8, bias=True)
          (expand): Linear(in_features=8, out_features=192, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(192, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(56, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_quantizer): Identity()
      )
      (block1): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(56, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(336, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(336, 336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=336, bias=False)
          (bn): BatchNorm2d(336, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=336, out_features=14, bias=True)
          (expand): Linear(in_features=14, out_features=336, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(336, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(56, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
      (block2): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(56, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(336, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(336, 336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=336, bias=False)
          (bn): BatchNorm2d(336, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=336, out_features=14, bias=True)
          (expand): Linear(in_features=14, out_features=336, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(336, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(56, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
      (block3): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(56, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(336, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(336, 336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=336, bias=False)
          (bn): BatchNorm2d(336, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=336, out_features=14, bias=True)
          (expand): Linear(in_features=14, out_features=336, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(336, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(56, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
    )
    (3): Sequential(
      (block0): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(56, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(336, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(336, 336, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=336, bias=False)
          (bn): BatchNorm2d(336, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=336, out_features=14, bias=True)
          (expand): Linear(in_features=14, out_features=336, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(336, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(112, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_quantizer): Identity()
      )
      (block1): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=672, bias=False)
          (bn): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=672, out_features=28, bias=True)
          (expand): Linear(in_features=28, out_features=672, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(112, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
      (block2): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=672, bias=False)
          (bn): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=672, out_features=28, bias=True)
          (expand): Linear(in_features=28, out_features=672, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(112, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
      (block3): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=672, bias=False)
          (bn): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=672, out_features=28, bias=True)
          (expand): Linear(in_features=28, out_features=672, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(112, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
      (block4): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=672, bias=False)
          (bn): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=672, out_features=28, bias=True)
          (expand): Linear(in_features=28, out_features=672, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(112, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
      (block5): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=672, bias=False)
          (bn): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=672, out_features=28, bias=True)
          (expand): Linear(in_features=28, out_features=672, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(112, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
    )
    (4): Sequential(
      (block0): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
          (bn): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=672, out_features=28, bias=True)
          (expand): Linear(in_features=28, out_features=672, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(672, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_quantizer): Identity()
      )
      (block1): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
          (bn): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=960, out_features=40, bias=True)
          (expand): Linear(in_features=40, out_features=960, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
      (block2): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
          (bn): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=960, out_features=40, bias=True)
          (expand): Linear(in_features=40, out_features=960, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
      (block3): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
          (bn): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=960, out_features=40, bias=True)
          (expand): Linear(in_features=40, out_features=960, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
      (block4): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
          (bn): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=960, out_features=40, bias=True)
          (expand): Linear(in_features=40, out_features=960, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
      (block5): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
          (bn): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=960, out_features=40, bias=True)
          (expand): Linear(in_features=40, out_features=960, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
    )
    (5): Sequential(
      (block0): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(960, 960, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=960, bias=False)
          (bn): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=960, out_features=40, bias=True)
          (expand): Linear(in_features=40, out_features=960, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(960, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(272, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_quantizer): Identity()
      )
      (block1): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
          (bn): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=1632, out_features=68, bias=True)
          (expand): Linear(in_features=68, out_features=1632, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(272, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
      (block2): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
          (bn): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=1632, out_features=68, bias=True)
          (expand): Linear(in_features=68, out_features=1632, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(272, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
      (block3): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
          (bn): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=1632, out_features=68, bias=True)
          (expand): Linear(in_features=68, out_features=1632, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(272, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
      (block4): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
          (bn): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=1632, out_features=68, bias=True)
          (expand): Linear(in_features=68, out_features=1632, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(272, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
      (block5): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
          (bn): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=1632, out_features=68, bias=True)
          (expand): Linear(in_features=68, out_features=1632, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(272, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
      (block6): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
          (bn): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=1632, out_features=68, bias=True)
          (expand): Linear(in_features=68, out_features=1632, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(272, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
      (block7): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
          (bn): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=1632, out_features=68, bias=True)
          (expand): Linear(in_features=68, out_features=1632, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(272, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
    )
    (6): Sequential(
      (block0): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(1632, 1632, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1632, bias=False)
          (bn): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=1632, out_features=68, bias=True)
          (expand): Linear(in_features=68, out_features=1632, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(1632, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(448, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_quantizer): Identity()
      )
      (block1): MBConvBlock(
        (expand): Sequential(
          (conv): Conv2d(448, 2688, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(2688, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (depsep): Sequential(
          (conv): Conv2d(2688, 2688, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2688, bias=False)
          (bn): BatchNorm2d(2688, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (se): SequentialSqueezeAndExcitation(
          (squeeze): Linear(in_features=2688, out_features=112, bias=True)
          (expand): Linear(in_features=112, out_features=2688, bias=True)
          (activation): SiLU(inplace=True)
          (sigmoid): Sigmoid()
          (mul_a_quantizer): Identity()
          (mul_b_quantizer): Identity()
        )
        (proj): Sequential(
          (conv): Conv2d(2688, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(448, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        (residual_add): StochasticDepthResidual()
        (residual_quantizer): Identity()
      )
    )
  )
  (features): Sequential(
    (conv): Conv2d(448, 1792, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(1792, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
    (activation): SiLU(inplace=True)
  )
  (classifier): Sequential(
    (pooling): AdaptiveAvgPool2d(output_size=1)
    (squeeze): Flatten()
    (dropout): Dropout(p=0.4, inplace=False)
    (fc): Linear(in_features=1792, out_features=1000, bias=True)
  )
)
