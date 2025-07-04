FOCAL_LENGTH = 1.0
REND_SIZE = 256  # Size of target masks for silhouette loss.
BBOX_EXPANSION_FACTOR = 0.3  # Amount to pad the target masks for silhouette loss.
RENDER_H, RENDER_W = 384, 384

BBOX_EXPANSION = {
    "default": 0.3,
}
BBOX_EXPANSION_PARTS = {
    "default": 0.3,
}