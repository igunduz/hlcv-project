import torchvision.transforms as transforms


presets = dict(
    CIFAR10=dict(
        train=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        ),
        eval=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        )
    ),
    
    CIFAR10_VGG=dict(
        train=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),
        eval=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    ),

    #### TODO ####
    # Define different presets here and try them by specifying in the config file
    # E.g. CIFAR10_WithFlip=dict()

    ##############

    # Keep 'eval' the same for all transforms
    CIFAR10_WithFlip=dict(
        train=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5)),
        ]),
        eval=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))
        ])
    ),

    CIFAR10_WithFlipAndCrop=dict(
        train=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),  # Add padding and then crop or else there is no change
            # Also add RandomRotation by a small angle
            # Avoid transforms like GrayScale because it reduces variance. Instead use RandomGrayscale
            # Or RandomApply + GrayScale can be used to apply a transform with a certain probability
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5)),
        ]),
        eval=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))
        ])
    ),
)