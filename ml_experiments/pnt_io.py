"""Read-only helpers for parsing DotDotGoose .pnt files for ML experiments."""
import json


def load_pnt(pnt_path):
    with open(pnt_path, 'r') as f:
        return json.load(f)


def points_for_image(pnt_data, image_name, class_name):
    classes = pnt_data['points'].get(image_name, {})
    points = classes.get(class_name, [])
    return [(p['x'], p['y']) for p in points]


def all_points_for_image(pnt_data, image_name):
    """All points regardless of class, for negative-patch exclusion."""
    return [(x, y) for x, y, _ in all_points_with_class_for_image(pnt_data, image_name)]


def all_points_with_class_for_image(pnt_data, image_name):
    """All points regardless of class, each tagged with its own species -- used when
    pooling classes as generic "bird" positives but still wanting to know afterward which
    species actually contributed (see train.py's pooled_classes())."""
    classes = pnt_data['points'].get(image_name, {})
    points = []
    for class_name, class_points in classes.items():
        points.extend((p['x'], p['y'], class_name) for p in class_points)
    return points
