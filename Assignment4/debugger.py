def checkpoint(debug, msg="Checkpoint reached.", stop=False):
    """
    Prints a checkpoint message and optionally raises an error if debugging is enabled.

    Parameters
    ----------
    debug : bool
        If True, the checkpoint message will be printed.
    msg : str, optional
        The message to display at the checkpoint. Default is "Checkpoint reached."
    stop : bool, optional
        If True, raises CheckpointReachedError to stop execution.
    """
    if debug:
        print(f"Checkpoint: {msg}")
        if stop:
            raise ValueError(f"Execution stopped at checkpoint: {msg}")