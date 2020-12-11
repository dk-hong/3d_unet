def dice_loss(input, target, weight=None):
    # input and target shapes must match
    assert input.size() == target.size(), "'input' ({}) and 'target' ({}) must have the same shape".format(
        input.size(), target.size())
    
    smooth = 1.

    input = input.reshape(-1)
    target = target.reshape(-1)
    
    intersection = (input * target).sum()

    if weight is not None:
        intersection = weight * intersection
    
    return 1 - ((2. * intersection + smooth) / (input.sum() + target.sum() + smooth))
