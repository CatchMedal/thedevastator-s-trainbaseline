Loaded pretrained weights for efficientnet-b7
EfficientNet(
  (_conv_stem): Conv2dStaticSamePadding(
    3, 64, kernel_size=(3, 3), stride=(2, 2), bias=False
    (static_padding): ZeroPad2d((0, 1, 0, 1))
  )
  (_bn0): BatchNorm2d(64, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
  (_blocks): ModuleList(
    (0): MBConvBlock(
      (_depthwise_conv): Conv2dStaticSamePadding(
        64, 64, kernel_size=(3, 3), stride=[1, 1], groups=64, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(64, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        64, 16, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        16, 64, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(32, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (1): MBConvBlock(
      (_depthwise_conv): Conv2dStaticSamePadding(
        32, 32, kernel_size=(3, 3), stride=(1, 1), groups=32, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(32, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        32, 8, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        8, 32, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(32, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (2): MBConvBlock(
      (_depthwise_conv): Conv2dStaticSamePadding(
        32, 32, kernel_size=(3, 3), stride=(1, 1), groups=32, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(32, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        32, 8, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        8, 32, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(32, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (3): MBConvBlock(
      (_depthwise_conv): Conv2dStaticSamePadding(
        32, 32, kernel_size=(3, 3), stride=(1, 1), groups=32, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(32, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        32, 8, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        8, 32, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False
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
        192, 192, kernel_size=(3, 3), stride=[2, 2], groups=192, bias=False
        (static_padding): ZeroPad2d((0, 1, 0, 1))
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
        192, 48, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(48, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (5): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(288, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        288, 288, kernel_size=(3, 3), stride=(1, 1), groups=288, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(288, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        288, 12, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        12, 288, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(48, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (6): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(288, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        288, 288, kernel_size=(3, 3), stride=(1, 1), groups=288, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(288, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        288, 12, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        12, 288, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(48, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (7): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(288, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        288, 288, kernel_size=(3, 3), stride=(1, 1), groups=288, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(288, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        288, 12, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        12, 288, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(48, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (8): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(288, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        288, 288, kernel_size=(3, 3), stride=(1, 1), groups=288, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(288, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        288, 12, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        12, 288, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(48, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (9): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(288, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        288, 288, kernel_size=(3, 3), stride=(1, 1), groups=288, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(288, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        288, 12, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        12, 288, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(48, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (10): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(288, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        288, 288, kernel_size=(3, 3), stride=(1, 1), groups=288, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(288, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        288, 12, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        12, 288, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(48, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (11): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(288, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        288, 288, kernel_size=(5, 5), stride=[2, 2], groups=288, bias=False
        (static_padding): ZeroPad2d((1, 2, 1, 2))
      )
      (_bn1): BatchNorm2d(288, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        288, 12, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        12, 288, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        288, 80, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(80, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (12): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(480, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        480, 480, kernel_size=(5, 5), stride=(1, 1), groups=480, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(480, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        480, 20, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        20, 480, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(80, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (13): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(480, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        480, 480, kernel_size=(5, 5), stride=(1, 1), groups=480, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(480, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        480, 20, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        20, 480, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(80, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (14): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(480, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        480, 480, kernel_size=(5, 5), stride=(1, 1), groups=480, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(480, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        480, 20, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        20, 480, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(80, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (15): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(480, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        480, 480, kernel_size=(5, 5), stride=(1, 1), groups=480, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(480, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        480, 20, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        20, 480, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(80, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (16): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(480, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        480, 480, kernel_size=(5, 5), stride=(1, 1), groups=480, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(480, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        480, 20, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        20, 480, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(80, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (17): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(480, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        480, 480, kernel_size=(5, 5), stride=(1, 1), groups=480, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(480, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        480, 20, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        20, 480, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(80, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (18): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(480, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        480, 480, kernel_size=(3, 3), stride=[2, 2], groups=480, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(480, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        480, 20, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        20, 480, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        480, 160, kernel_size=(1, 1), stride=(1, 1), bias=False
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
        960, 960, kernel_size=(3, 3), stride=(1, 1), groups=960, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
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
        960, 960, kernel_size=(3, 3), stride=(1, 1), groups=960, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
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
        960, 960, kernel_size=(3, 3), stride=(1, 1), groups=960, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
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
        960, 960, kernel_size=(3, 3), stride=(1, 1), groups=960, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
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
    (23): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        960, 960, kernel_size=(3, 3), stride=(1, 1), groups=960, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
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
    (24): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        960, 960, kernel_size=(3, 3), stride=(1, 1), groups=960, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
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
    (25): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        960, 960, kernel_size=(3, 3), stride=(1, 1), groups=960, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
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
    (26): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        960, 960, kernel_size=(3, 3), stride=(1, 1), groups=960, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
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
    (27): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        960, 960, kernel_size=(3, 3), stride=(1, 1), groups=960, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
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
    (28): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        960, 960, kernel_size=(5, 5), stride=[1, 1], groups=960, bias=False
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
        960, 224, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(224, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (29): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        1344, 1344, kernel_size=(5, 5), stride=(1, 1), groups=1344, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        1344, 56, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        56, 1344, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(224, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (30): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        1344, 1344, kernel_size=(5, 5), stride=(1, 1), groups=1344, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        1344, 56, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        56, 1344, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(224, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (31): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        1344, 1344, kernel_size=(5, 5), stride=(1, 1), groups=1344, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        1344, 56, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        56, 1344, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(224, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (32): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        1344, 1344, kernel_size=(5, 5), stride=(1, 1), groups=1344, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        1344, 56, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        56, 1344, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(224, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (33): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        1344, 1344, kernel_size=(5, 5), stride=(1, 1), groups=1344, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        1344, 56, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        56, 1344, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(224, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (34): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        1344, 1344, kernel_size=(5, 5), stride=(1, 1), groups=1344, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        1344, 56, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        56, 1344, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(224, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (35): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        1344, 1344, kernel_size=(5, 5), stride=(1, 1), groups=1344, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        1344, 56, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        56, 1344, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(224, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (36): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        1344, 1344, kernel_size=(5, 5), stride=(1, 1), groups=1344, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        1344, 56, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        56, 1344, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(224, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (37): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        1344, 1344, kernel_size=(5, 5), stride=(1, 1), groups=1344, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        1344, 56, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        56, 1344, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(224, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (38): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        1344, 1344, kernel_size=(5, 5), stride=[2, 2], groups=1344, bias=False
        (static_padding): ZeroPad2d((1, 2, 1, 2))
      )
      (_bn1): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        1344, 56, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        56, 1344, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        1344, 384, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(384, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (39): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        2304, 2304, kernel_size=(5, 5), stride=(1, 1), groups=2304, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        2304, 96, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        96, 2304, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(384, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (40): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        2304, 2304, kernel_size=(5, 5), stride=(1, 1), groups=2304, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        2304, 96, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        96, 2304, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(384, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (41): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        2304, 2304, kernel_size=(5, 5), stride=(1, 1), groups=2304, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        2304, 96, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        96, 2304, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(384, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (42): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        2304, 2304, kernel_size=(5, 5), stride=(1, 1), groups=2304, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        2304, 96, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        96, 2304, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(384, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (43): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        2304, 2304, kernel_size=(5, 5), stride=(1, 1), groups=2304, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        2304, 96, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        96, 2304, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(384, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (44): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        2304, 2304, kernel_size=(5, 5), stride=(1, 1), groups=2304, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        2304, 96, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        96, 2304, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(384, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (45): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        2304, 2304, kernel_size=(5, 5), stride=(1, 1), groups=2304, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        2304, 96, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        96, 2304, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(384, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (46): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        2304, 2304, kernel_size=(5, 5), stride=(1, 1), groups=2304, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        2304, 96, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        96, 2304, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(384, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (47): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        2304, 2304, kernel_size=(5, 5), stride=(1, 1), groups=2304, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        2304, 96, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        96, 2304, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(384, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (48): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        2304, 2304, kernel_size=(5, 5), stride=(1, 1), groups=2304, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        2304, 96, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        96, 2304, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(384, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (49): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        2304, 2304, kernel_size=(5, 5), stride=(1, 1), groups=2304, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        2304, 96, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        96, 2304, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(384, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (50): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        2304, 2304, kernel_size=(5, 5), stride=(1, 1), groups=2304, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        2304, 96, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        96, 2304, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(384, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (51): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        2304, 2304, kernel_size=(3, 3), stride=[1, 1], groups=2304, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        2304, 96, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        96, 2304, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        2304, 640, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(640, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (52): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        640, 3840, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(3840, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        3840, 3840, kernel_size=(3, 3), stride=(1, 1), groups=3840, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(3840, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        3840, 160, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        160, 3840, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(640, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (53): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        640, 3840, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(3840, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        3840, 3840, kernel_size=(3, 3), stride=(1, 1), groups=3840, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(3840, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        3840, 160, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        160, 3840, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(640, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (54): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        640, 3840, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(3840, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        3840, 3840, kernel_size=(3, 3), stride=(1, 1), groups=3840, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(3840, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        3840, 160, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        160, 3840, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(640, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
  )
  (_conv_head): Conv2dStaticSamePadding(
    640, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False
    (static_padding): Identity()
  )
  (_bn1): BatchNorm2d(2560, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
  (_avg_pooling): AdaptiveAvgPool2d(output_size=1)
  (_dropout): Dropout(p=0.5, inplace=False)
  (_fc): Linear(in_features=2560, out_features=1000, bias=True)
  (_swish): MemoryEfficientSwish()
)
