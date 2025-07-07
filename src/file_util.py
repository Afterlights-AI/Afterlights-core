import os
def resolve_model_path(path):
    # Always resolve relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_path = os.path.join(project_root, path) if not os.path.isabs(path) else path
    return abs_path
