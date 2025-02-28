def _get_opt(run):
    if (
        run.config["opt"] == "askotchv2"
        and run.config["precond_params"] is not None
        and run.config["precond_params"]["type"] == "newton"
    ):
        if run.config["accelerated"]:
            return "nsap"
        return "sap"
    if run.config["opt"] == "askotchv2" and not run.config["accelerated"]:
        return "skotchv2"
    if run.config["opt"] == "pcg" and run.config["precond_params"]["type"] == "falkon":
        return "falkon"
    return run.config["opt"]
