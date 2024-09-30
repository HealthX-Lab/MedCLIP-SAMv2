from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet, PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_batchnorm
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0, InitWeights_He
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from torch import nn , randn
from torchsummary import summary
from torchviz import make_dot



class nnUNetTrainerBN(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        num_stages = len(configuration_manager.conv_kernel_sizes)

        dim = len(configuration_manager.conv_kernel_sizes[0])
        conv_op = convert_dim_to_conv_op(dim)

        label_manager = plans_manager.get_label_manager(dataset_json)

        segmentation_network_class_name = configuration_manager.UNet_class_name
        mapping = {
            'PlainConvUNet': PlainConvUNet,
            'ResidualEncoderUNet': ResidualEncoderUNet
        }
        kwargs = {
            'PlainConvUNet': {
                'conv_bias': True,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            },
            'ResidualEncoderUNet': {
                'conv_bias': True,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            }
        }
        assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                                  'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                                  'into either this ' \
                                                                  'function (get_network_from_plans) or ' \
                                                                  'the init of your nnUNetModule to accomodate that.'
        network_class = mapping[segmentation_network_class_name]

        conv_or_blocks_per_stage = {
            'n_conv_per_stage'
            if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': configuration_manager.n_conv_per_stage_encoder,
            'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
        }
        # network class name!!
        model = network_class(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                    configuration_manager.unet_max_num_features) for i in range(num_stages)],
            conv_op=conv_op,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            num_classes=label_manager.num_segmentation_heads,
            deep_supervision=enable_deep_supervision,
            **conv_or_blocks_per_stage,
            **kwargs[segmentation_network_class_name]
        )
        model.apply(InitWeights_He(1e-2))
        if network_class == ResidualEncoderUNet:
            model.apply(init_last_bn_before_add_to_0)

        #$ Display the model summary
        try:
            input_shape = (160,160)
            summary(model, input_size=(num_input_channels, input_shape))

            # Create a dummy input tensor
            dummy_input = randn(1, num_input_channels, input_shape)

            # Generate a graph of the model architecture
            graph = make_dot(model(dummy_input), params=dict(model.named_parameters()))

            # Save the graph as an image
            nnUNet_results.joinpath("model_architecture").mkdir(exist_ok=True)
            path = nnUNet_results.joinpath("model_architecture", "model_architecture.png")
            graph.render(path, format="png")

            print("model summary saved to ", path)
        except:
            print("Could not save the model summary")


        return model
