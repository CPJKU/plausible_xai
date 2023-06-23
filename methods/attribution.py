def attribute_image_features(net, algorithm, input, target, **kwargs):
    net.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=target,
                                              **kwargs
                                              )

    return tensor_attributions
