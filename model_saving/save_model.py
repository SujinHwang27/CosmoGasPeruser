from typing import Dict, Tuple, List

# TODO: get data and model information to save models
def get_physics_values(redshift: float) -> List[str]:
    """
    Get physics values from config.
    
    Args:
        redshift: The redshift value to process (kept for compatibility)
        
    Returns:
        List of physics values from config
    """
    if len(PHYSICS_VALUES) < 2:
        raise ValueError(f"Dataset must contain at least 2 physics values. Found: {len(PHYSICS_VALUES)}")
    
    logger.info(f"Using physics values: {PHYSICS_VALUES}")
    return PHYSICS_VALUES