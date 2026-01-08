"""Useful utils for model checkpointing"""

from flax import serialization
from flax.training import checkpoints

def save_state_parameters(state, path):
    """
    Save Flax model parameters as a single file.
    Ideal for saving final trained models.
    
    Args:
        state: Flax model state or parameters
        path: File path where parameters will be saved
    """
    state_bytes = serialization.to_bytes(state)
    with open(path, 'wb') as f:
        f.write(state_bytes)
    print(f"Parameters saved to {path}")

def load_state_parameters(state_template, path):
    """
    Load Flax model parameters from a file.
    
    Args:
        state_template: Template state with expected structure
        path: Path to the parameter file
    
    Returns:
        Deserialized Flax state with same structure as state_template
    """
    with open(path, 'rb') as f:
        state_bytes = f.read()
    print(f"Load parameters from {path}")
    return serialization.from_bytes(state_template, state_bytes)

def save_training_checkpoint(checkpoint_dir, state, step, keep):
    """
    Save training checkpoint with version management.
    Ideal for saving intermediate states during training.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        state: Flax model state to save
        step: Training step or epoch number for versioning
        keep: Number of recent checkpoints to keep
    """
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=state,
        step=step,
        overwrite=True,
        keep=keep
    )
    print(f'Training checkpoint saved to {checkpoint_dir} (step {step})')

def load_training_checkpoint(checkpoint_dir, state_template):
    """
    Load the latest training checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        state_template: Template state with expected structure
    
    Returns:
        Restored Flax state from the latest checkpoint
    """
    restored_state = checkpoints.restore_checkpoint(checkpoint_dir, state_template)
    print(f'Loaded checkpoint from {checkpoint_dir}')
    return restored_state