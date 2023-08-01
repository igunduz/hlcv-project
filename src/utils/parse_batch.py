import torch

def parse_encoded_input(encoded_inputs):
    """
    Parse the batch returned by the data loader.

    :param batch: A batch of data returned by the data loader.
    :return: A tuple containing the parsed data.
    """
    images = []
    masks = []
    for encoded_input in encoded_inputs:
        image, mask = torch.tensor(encoded_input['pixel_values']), torch.tensor(encoded_input['labels'])
        images.append(image)
        masks.append(mask)

    images = torch.stack(images)
    masks = torch.stack(masks)
    return images, masks

def parse_object_labels(batch_labels):
    """
    Parse the object labels returned by the data loader.

    :param object_labels: A batch of object labels returned by the data loader.
    :return: A tuple containing the parsed data.
    """
    batch_id = []
    batch_dims = []
    for object_labels in batch_labels:
        image_ids, image_labels = [], []
        for object_label in object_labels:
            if object_label[0] != 0:
                object_id, object_dims = object_label[0], object_label[1:]
                image_ids.append(torch.tensor(object_id))
                image_labels.append(torch.tensor(object_dims))
        batch_id.append(torch.stack(image_ids))
        batch_dims.append(torch.stack(image_labels))

    return batch_id, batch_dims 