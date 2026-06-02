from lib.models.monodf import build_monodf


def build_model(cfg, model_name=None):
    """Build the MonoDF model and criterion.

    The optional ``model_name`` argument is kept for backward compatibility
    with the old MonoDETR entry point but is ignored: this project always
    builds MonoDF.
    """
    return build_monodf(cfg)
