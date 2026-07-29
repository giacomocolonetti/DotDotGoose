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
    classes = pnt_data['points'].get(image_name, {})
    points = []
    for class_points in classes.values():
        points.extend((p['x'], p['y']) for p in class_points)
    return points
