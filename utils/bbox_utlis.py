def center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)


def width_of_bbox(bbox):
    return bbox[2]-bbox[0]
