EfficientNet(
  (_conv_): Conv2dStaticSamePadding(
    3, 48, kernel_size=(3, 3), stride=(2, 2), bias=False
    (static_padding): ZeroPad2d((0, 1, 0, 1))
  )
  (_bn0): BatchNorm2d(48, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
  (_blocks): ModuleList(
    (0): MBConvBlock(
      (_depthwise_conv): Conv2dStaticSamePadding(
        48, 48, kernel_size=(3, 3), stride=[1, 1], groups=48, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(48, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        48, 12, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        12, 48, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        48, 24, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(24, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (1): MBConvBlock(
      (_depthwise_conv): Conv2dStaticSamePadding(
        24, 24, kernel_size=(3, 3), stride=(1, 1), groups=24, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(24, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        24, 6, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        6, 24, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(24, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (2): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(144, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        144, 144, kernel_size=(3, 3), stride=[2, 2], groups=144, bias=False
        (static_padding): ZeroPad2d((0, 1, 0, 1))
      )
      (_bn1): BatchNorm2d(144, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        144, 6, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        6, 144, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(32, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (3): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(192, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        192, 192, kernel_size=(3, 3), stride=(1, 1), groups=192, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(192, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        192, 8, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        8, 192, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(32, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (4): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(192, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        192, 192, kernel_size=(3, 3), stride=(1, 1), groups=192, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(192, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        192, 8, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        8, 192, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(32, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (5): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(192, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        192, 192, kernel_size=(3, 3), stride=(1, 1), groups=192, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(192, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        192, 8, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        8, 192, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(32, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (6): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(192, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        192, 192, kernel_size=(5, 5), stride=[2, 2], groups=192, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(192, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        192, 8, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        8, 192, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        192, 56, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(56, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (7): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        56, 336, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(336, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        336, 336, kernel_size=(5, 5), stride=(1, 1), groups=336, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(336, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        336, 14, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        14, 336, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        336, 56, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(56, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (8): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        56, 336, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(336, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        336, 336, kernel_size=(5, 5), stride=(1, 1), groups=336, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(336, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        336, 14, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        14, 336, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        336, 56, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(56, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (9): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        56, 336, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(336, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        336, 336, kernel_size=(5, 5), stride=(1, 1), groups=336, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(336, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        336, 14, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        14, 336, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        336, 56, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(56, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (10): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        56, 336, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(336, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        336, 336, kernel_size=(3, 3), stride=[2, 2], groups=336, bias=False
        (static_padding): ZeroPad2d((0, 1, 0, 1))
      )
      (_bn1): BatchNorm2d(336, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        336, 14, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        14, 336, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        336, 112, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(112, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (11): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        672, 672, kernel_size=(3, 3), stride=(1, 1), groups=672, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        672, 28, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        28, 672, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(112, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (12): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        672, 672, kernel_size=(3, 3), stride=(1, 1), groups=672, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        672, 28, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        28, 672, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(112, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (13): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        672, 672, kernel_size=(3, 3), stride=(1, 1), groups=672, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        672, 28, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        28, 672, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(112, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (14): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        672, 672, kernel_size=(3, 3), stride=(1, 1), groups=672, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        672, 28, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        28, 672, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(112, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (15): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        672, 672, kernel_size=(3, 3), stride=(1, 1), groups=672, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        672, 28, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        28, 672, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(112, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (16): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        672, 672, kernel_size=(5, 5), stride=[1, 1], groups=672, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(672, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        672, 28, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        28, 672, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        672, 160, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(160, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (17): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        960, 960, kernel_size=(5, 5), stride=(1, 1), groups=960, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        960, 40, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        40, 960, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(160, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (18): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        960, 960, kernel_size=(5, 5), stride=(1, 1), groups=960, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        960, 40, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        40, 960, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(160, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (19): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        960, 960, kernel_size=(5, 5), stride=(1, 1), groups=960, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        960, 40, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        40, 960, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(160, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (20): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        960, 960, kernel_size=(5, 5), stride=(1, 1), groups=960, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        960, 40, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        40, 960, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(160, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (21): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        960, 960, kernel_size=(5, 5), stride=(1, 1), groups=960, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        960, 40, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        40, 960, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(160, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (22): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        960, 960, kernel_size=(5, 5), stride=[2, 2], groups=960, bias=False
        (static_padding): ZeroPad2d((1, 2, 1, 2))
      )
      (_bn1): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        960, 40, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        40, 960, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        960, 272, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(272, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (23): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        1632, 1632, kernel_size=(5, 5), stride=(1, 1), groups=1632, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        1632, 68, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        68, 1632, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(272, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (24): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        1632, 1632, kernel_size=(5, 5), stride=(1, 1), groups=1632, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        1632, 68, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        68, 1632, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(272, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (25): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        1632, 1632, kernel_size=(5, 5), stride=(1, 1), groups=1632, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        1632, 68, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        68, 1632, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(272, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (26): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        1632, 1632, kernel_size=(5, 5), stride=(1, 1), groups=1632, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        1632, 68, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        68, 1632, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(272, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (27): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        1632, 1632, kernel_size=(5, 5), stride=(1, 1), groups=1632, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        1632, 68, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        68, 1632, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(272, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (28): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        1632, 1632, kernel_size=(5, 5), stride=(1, 1), groups=1632, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        1632, 68, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        68, 1632, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(272, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (29): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        1632, 1632, kernel_size=(5, 5), stride=(1, 1), groups=1632, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        1632, 68, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        68, 1632, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(272, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (30): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        1632, 1632, kernel_size=(3, 3), stride=[1, 1], groups=1632, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(1632, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        1632, 68, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        68, 1632, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        1632, 448, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(448, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (31): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        448, 2688, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(2688, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        2688, 2688, kernel_size=(3, 3), stride=(1, 1), groups=2688, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(2688, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        2688, 112, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        112, 2688, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        2688, 448, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(448, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
  )
  (_conv_head): Conv2dStaticSamePadding(
    448, 1792, kernel_size=(1, 1), stride=(1, 1), bias=False
    (static_padding): Identity()
  )
  (_bn1): BatchNorm2d(1792, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
  (_avg_pooling): AdaptiveAvgPool2d(output_size=1)
  (_dropout): Dropout(p=0.4, inplace=False)
  (_fc): Linear(in_features=1792, out_features=1000, bias=True)
  (_swish): MemoryEfficientSwish()
)
